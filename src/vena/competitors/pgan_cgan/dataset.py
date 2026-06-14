"""UCSF-PDGM image-H5 → pGAN 2D-slice dataset.

The dataset is deterministic (no augmentation) by contract: VENA owns the
augmentation regime, the competitor's loader does not. Each ``__getitem__``
call returns the same tensor for the same index.

Pipeline per slice:

1. Resolve ``(patient_idx, axial_z)`` from a pre-built index.
2. Read the four modality slices and the brain mask slice from H5.
3. Apply ``percentile_normalise(0, 99.5, foreground_only=True)`` *over the
   patient-wide foreground statistics* (cached at dataset init time so we
   don't re-read the volume on every slice access).
4. Pad H/W from native (240, 240) to ``image_size`` (default 256×256,
   divisible by 4 as required by pGAN's two stride-2 downsampling layers).
5. Rescale ``[0, 1] → [-1, 1]`` to match pGAN's tanh output range.

Returned dict matches pGAN's `CreateDataset` contract:
``{'A': source_tensor, 'B': target_tensor, 'A_paths': str, 'B_paths': str}``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from vena.common import percentile_normalise

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class DatasetError(Exception):
    """Raised on malformed H5 or invalid split selection."""


def _decode_ids(arr: np.ndarray) -> list[str]:
    """Decode an H5 vlen-str dataset into a list of Python ``str``."""
    out: list[str] = []
    for s in arr:
        if isinstance(s, bytes):
            out.append(s.decode("utf-8"))
        else:
            out.append(str(s))
    return out


def _percentile_thresholds_per_patient(
    h5_path: Path,
    patient_idx: int,
    modalities: Sequence[str],
    upper: float,
    foreground_threshold: float,
) -> dict[str, tuple[float, float]]:
    """Precompute (low, high) percentile thresholds per modality over the full volume.

    Mirrors the foreground-only branch of ``vena.common.percentile_normalise`` so the
    per-slice normalisation at ``__getitem__`` time can be applied with just (low,
    high) without reading the whole volume again.
    """
    out: dict[str, tuple[float, float]] = {}
    with h5py.File(h5_path, "r") as f:
        for mod in modalities:
            vol = f[f"images/{mod}"][patient_idx]  # (H, W, D)
            arr = np.asarray(vol, dtype=np.float32)
            fg = arr[arr > foreground_threshold]
            if fg.size == 0:
                out[mod] = (0.0, 1.0)
                continue
            low = float(np.percentile(fg, 0.0))
            high = float(np.percentile(fg, upper))
            if high <= low:
                high = low + 1e-8
            out[mod] = (low, high)
    return out


class UCSFPDGMSliceDataset(Dataset[dict[str, torch.Tensor | str]]):
    """UCSF-PDGM image-H5 → 2D axial slices for pGAN-style training.

    Parameters
    ----------
    image_h5:
        Path to ``UCSFPDGM_image.h5`` (schema 2.0.0 per
        ``.claude/rules/h5-design-principles.md``).
    fold:
        CV fold (0..4) for non-test phases.
    phase:
        One of ``"train"``, ``"val"``, ``"test"``.
    input_modalities:
        Channels of the source image. Default ``("t1pre", "t2", "flair")``.
        SWAN is absent from the UCSF-PDGM H5; use the BraTS-GLI cohort if SWAN
        becomes required downstream.
    target_modality:
        The target. Default ``"t1c"``.
    image_size:
        Pad H/W to this size (must be divisible by 4). Default 256.
    min_brain_voxels:
        Drop slices whose brain-mask voxel count is below this threshold. This
        is data validity filtering, not augmentation. Default 1000.
    max_patients:
        If set, only the first N patients of the chosen split are used (smoke
        runs).
    """

    def __init__(
        self,
        image_h5: Path | str,
        fold: int,
        phase: str,
        input_modalities: Sequence[str] = ("t1pre", "t2", "flair"),
        target_modality: str = "t1c",
        image_size: int = 256,
        min_brain_voxels: int = 1000,
        max_patients: int | None = None,
    ) -> None:
        self.image_h5 = Path(image_h5)
        if not self.image_h5.is_file():
            raise DatasetError(f"image H5 not found at {self.image_h5}")
        if phase not in {"train", "val", "test"}:
            raise DatasetError(f"phase must be one of train/val/test, got {phase!r}")
        if image_size % 4 != 0:
            raise DatasetError(
                f"image_size must be divisible by 4 (pGAN downsamples ×4); got {image_size}"
            )

        self.fold = fold
        self.phase = phase
        self.input_modalities = tuple(input_modalities)
        self.target_modality = target_modality
        self.image_size = image_size
        self.min_brain_voxels = min_brain_voxels

        with h5py.File(self.image_h5, "r") as f:
            all_ids = _decode_ids(np.asarray(f["ids"]))
            if phase == "test":
                key = "splits/test"
            else:
                key = f"splits/cv/fold_{fold}/{phase}"
            if key not in f:
                raise DatasetError(f"{key} missing from {self.image_h5}")
            split_ids = _decode_ids(np.asarray(f[key]))

        id_to_idx = {pid: i for i, pid in enumerate(all_ids)}
        missing = [pid for pid in split_ids if pid not in id_to_idx]
        if missing:
            raise DatasetError(
                f"split {key!r} references {len(missing)} ids absent from /ids"
            )
        if max_patients is not None:
            split_ids = split_ids[:max_patients]

        self.patient_ids: list[str] = split_ids
        self.patient_indices: list[int] = [id_to_idx[pid] for pid in split_ids]

        logger.info(
            "UCSFPDGMSliceDataset[%s/fold%d]: %d patients, modalities=%s → %s",
            phase, fold, len(self.patient_ids), self.input_modalities, target_modality,
        )

        # Build (patient, slice) index — and cache per-patient percentile thresholds.
        self._slice_index: list[tuple[int, int]] = []
        self._thresholds: dict[int, dict[str, tuple[float, float]]] = {}
        all_mods = tuple(self.input_modalities) + (self.target_modality,)
        with h5py.File(self.image_h5, "r") as f:
            brain = f["masks/brain"]
            for pidx in self.patient_indices:
                bmask = np.asarray(brain[pidx])  # (H, W, D)
                # Count brain voxels per axial slice once.
                per_z = bmask.reshape(-1, bmask.shape[-1]).sum(axis=0)
                for z in np.flatnonzero(per_z >= self.min_brain_voxels):
                    self._slice_index.append((int(pidx), int(z)))
        # Defer threshold computation to first __getitem__ on each patient
        # (cheaper than scanning every volume up-front, and threadsafe in workers
        # because dict insertion is GIL-protected for a single key).
        self._all_modalities = all_mods
        self._h5: h5py.File | None = None
        logger.info(
            "UCSFPDGMSliceDataset: %d total slices (min_brain_voxels=%d)",
            len(self._slice_index), self.min_brain_voxels,
        )

    def _open(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.image_h5, "r", swmr=True)
        return self._h5

    def _get_thresholds(self, pidx: int) -> dict[str, tuple[float, float]]:
        if pidx not in self._thresholds:
            self._thresholds[pidx] = _percentile_thresholds_per_patient(
                self.image_h5,
                pidx,
                self._all_modalities,
                upper=99.5,
                foreground_threshold=0.0,
            )
        return self._thresholds[pidx]

    def __len__(self) -> int:
        return len(self._slice_index)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor | str]:
        pidx, z = self._slice_index[i]
        thresholds = self._get_thresholds(pidx)
        f = self._open()

        input_slices: list[torch.Tensor] = []
        for mod in self.input_modalities:
            raw = np.asarray(f[f"images/{mod}"][pidx, :, :, z], dtype=np.float32)
            low, high = thresholds[mod]
            x = np.clip((raw - low) / (high - low), 0.0, 1.0)
            input_slices.append(torch.from_numpy(x))

        raw_tgt = np.asarray(
            f[f"images/{self.target_modality}"][pidx, :, :, z], dtype=np.float32
        )
        low_t, high_t = thresholds[self.target_modality]
        y = np.clip((raw_tgt - low_t) / (high_t - low_t), 0.0, 1.0)
        target = torch.from_numpy(y).unsqueeze(0)  # (1, H, W)

        source = torch.stack(input_slices, dim=0)  # (C_in, H, W)

        # Pad H/W from native (240, 240) → (image_size, image_size). Centred.
        source = _pad_to(source, self.image_size)
        target = _pad_to(target, self.image_size)

        # tanh-range rescale (range was [0, 1] after percentile normalisation).
        source = source.mul_(2.0).sub_(1.0)
        target = target.mul_(2.0).sub_(1.0)

        path = f"{self.image_h5}#patient{pidx}_z{z}"
        return {"A": source, "B": target, "A_paths": path, "B_paths": path}

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_h5"] = None  # h5py handles are not picklable across workers.
        return state


def _pad_to(x: torch.Tensor, size: int) -> torch.Tensor:
    """Centred zero-pad (or centred crop) so the last two dims are ``(size, size)``."""
    h, w = x.shape[-2], x.shape[-1]
    pad_h = size - h
    pad_w = size - w
    if pad_h < 0 or pad_w < 0:
        # Centred crop.
        top = (-pad_h) // 2
        left = (-pad_w) // 2
        x = x[..., top : top + size, left : left + size]
        return x
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return F.pad(x, (left, right, top, bottom), mode="constant", value=0.0)


# Mirror pGAN's CreateDataLoader API so external callers can pretend nothing changed.
class CustomDatasetDataLoader:
    name = lambda self: "VENAUCSFPDGMSliceLoader"  # noqa: E731

    def __init__(self, dataset: UCSFPDGMSliceDataset, batch_size: int, num_workers: int) -> None:
        from torch.utils.data import DataLoader

        self.dataset = dataset
        # NOTE: shuffle is intentionally False at this layer. The runner shuffles
        # by passing a shuffled list of indices into a SubsetRandomSampler when needed
        # — keeping the underlying dataset deterministic preserves the no-augmentation
        # contract under DataLoader inspection.
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
        )

    def load_data(self):  # noqa: D401 — pGAN's exact method name.
        return self

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        for batch in self.dataloader:
            yield batch
