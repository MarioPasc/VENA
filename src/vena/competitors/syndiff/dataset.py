"""Cohort image-H5 → SynDiff 2D-slice dataset (single- and multi-cohort).

SynDiff's upstream ``CreateDatasetSynthesis`` produces a
``torch.utils.data.TensorDataset`` over two pre-sliced ``.mat`` v7.3 caches
with values in ``[-1, 1]`` (``data = (x - 0.5) / 0.5`` after rescaling raw
intensities into ``[0, 1]``). Each item is a 2-tuple
``(x1_slice, x2_slice)`` where ``x1`` is *contrast1* (target) and ``x2`` is
*contrast2* (source).

We keep that contract — the runner's training loop unpacks
``(x1, x2) = batch`` directly. The differences from upstream are:

1. Source of data: VENA cohort H5 (per ``h5-design-principles.md``), not
   ``.mat`` files.
2. Per-patient percentile-norm with ``foreground_only=True`` over the
   skull-stripped brain mask, matching VENA's encoding convention.
3. Centred zero-pad / centred crop to ``image_size`` (256, divisible by
   ``2**(len(ch_mult)-1) = 32``).
4. Deterministic — no augmentation; VENA owns the augmentation regime. Pinned
   by ``test_dataset_is_deterministic``.

The multi-cohort variant assembles per-cohort datasets via
``torch.utils.data.ConcatDataset`` against a VENA corpus-registry JSON, with
the same longitudinal-prefix-match, flat-splits fallback, and
missing-cohort-warn semantics as the pGAN / T1C-RFlow integrations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Dataset

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
    """Compute (low, high) percentile thresholds per modality over the full volume.

    Foreground-only branch — mirrors ``vena.common.percentile_normalise``.
    """
    out: dict[str, tuple[float, float]] = {}
    with h5py.File(h5_path, "r") as f:
        for mod in modalities:
            vol = f[f"images/{mod}"][patient_idx]
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


def _pad_to(x: torch.Tensor, size: int) -> torch.Tensor:
    """Centred zero-pad (or centred crop) so the last two dims are ``(size, size)``."""
    h, w = x.shape[-2], x.shape[-1]
    pad_h = size - h
    pad_w = size - w
    if pad_h < 0 or pad_w < 0:
        top = (-pad_h) // 2
        left = (-pad_w) // 2
        return x[..., top : top + size, left : left + size]
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return F.pad(x, (left, right, top, bottom), mode="constant", value=0.0)


class SynDiffSliceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Per-cohort 2D-slice loader matching SynDiff's ``(x1, x2)`` contract.

    Parameters
    ----------
    image_h5:
        Path to a VENA cohort image H5 (e.g. ``UCSFPDGM_image.h5``).
    fold:
        CV fold (0..4) for non-test phases.
    phase:
        One of ``"train"``, ``"val"``, ``"test"``.
    target_modality:
        SynDiff's contrast1. Returned as the first element of each item.
    source_modality:
        SynDiff's contrast2. Returned as the second element.
    image_size:
        Pad H/W to this size. Must be divisible by ``2**(num_resolutions-1)``;
        with ``ch_mult=[1,1,2,2,4,4]`` that means ``size % 32 == 0``. Default 256.
    min_brain_voxels:
        Drop slices whose brain-mask voxel count is below this threshold.
        Data-validity filter, not augmentation. Default 1000.
    max_patients:
        If set, only the first N resolved patients are used (smoke runs).
    """

    def __init__(
        self,
        image_h5: Path | str,
        fold: int,
        phase: str,
        target_modality: str = "t1c",
        source_modality: str = "t1pre",
        image_size: int = 256,
        min_brain_voxels: int = 1000,
        max_patients: int | None = None,
    ) -> None:
        self.image_h5 = Path(image_h5)
        if not self.image_h5.is_file():
            raise DatasetError(f"image H5 not found at {self.image_h5}")
        if phase not in {"train", "val", "test"}:
            raise DatasetError(f"phase must be one of train/val/test, got {phase!r}")
        if image_size % 32 != 0:
            raise DatasetError(
                "image_size must be divisible by 32 (SynDiff's 6-level NCSN++); "
                f"got {image_size}"
            )
        if source_modality == target_modality:
            raise DatasetError(
                f"source_modality and target_modality must differ; both were {source_modality!r}"
            )

        self.fold = fold
        self.phase = phase
        self.target_modality = target_modality
        self.source_modality = source_modality
        self.image_size = image_size
        self.min_brain_voxels = min_brain_voxels

        with h5py.File(self.image_h5, "r") as f:
            all_ids = _decode_ids(np.asarray(f["ids"]))
            if phase == "test":
                candidates = ["splits/test"]
            else:
                candidates = [f"splits/cv/fold_{fold}/{phase}", f"splits/{phase}"]
            key = next((c for c in candidates if c in f), None)
            if key is None:
                raise DatasetError(
                    f"none of {candidates} present in {self.image_h5}"
                )
            split_ids = _decode_ids(np.asarray(f[key]))

        id_to_idx = {pid: i for i, pid in enumerate(all_ids)}
        resolved_indices: list[int] = []
        resolved_ids: list[str] = []
        missing: list[str] = []
        for pid in split_ids:
            if pid in id_to_idx:
                resolved_indices.append(id_to_idx[pid])
                resolved_ids.append(pid)
                continue
            matched = False
            prefix_dash = f"{pid}-"
            prefix_uscr = f"{pid}_"
            for full_id, idx in id_to_idx.items():
                if full_id.startswith(prefix_dash) or full_id.startswith(prefix_uscr):
                    resolved_indices.append(idx)
                    resolved_ids.append(full_id)
                    matched = True
            if not matched:
                missing.append(pid)
        if missing:
            raise DatasetError(
                f"split {key!r} references {len(missing)} ids absent from /ids "
                f"(both exact and prefix match failed; e.g. {missing[:3]})"
            )
        if max_patients is not None:
            resolved_ids = resolved_ids[:max_patients]
            resolved_indices = resolved_indices[:max_patients]

        self.patient_ids: list[str] = resolved_ids
        self.patient_indices: list[int] = resolved_indices

        logger.info(
            "SynDiffSliceDataset[%s/fold%d]: %d patients, source=%s target=%s",
            phase, fold, len(self.patient_ids), source_modality, target_modality,
        )

        # Build the (patient_idx, axial_z) index using the brain-mask voxel count.
        self._slice_index: list[tuple[int, int]] = []
        self._thresholds: dict[int, dict[str, tuple[float, float]]] = {}
        all_mods = (source_modality, target_modality)
        with h5py.File(self.image_h5, "r") as f:
            brain = f["masks/brain"]
            for pidx in self.patient_indices:
                bmask = np.asarray(brain[pidx])
                per_z = bmask.reshape(-1, bmask.shape[-1]).sum(axis=0)
                for z in np.flatnonzero(per_z >= self.min_brain_voxels):
                    self._slice_index.append((int(pidx), int(z)))
        self._all_modalities = all_mods
        self._h5: h5py.File | None = None
        logger.info(
            "SynDiffSliceDataset: %d total slices (min_brain_voxels=%d)",
            len(self._slice_index), self.min_brain_voxels,
        )

    def _open(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.image_h5, "r")
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

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        pidx, z = self._slice_index[i]
        thresholds = self._get_thresholds(pidx)
        f = self._open()

        raw_src = np.asarray(
            f[f"images/{self.source_modality}"][pidx, :, :, z], dtype=np.float32
        )
        low_s, high_s = thresholds[self.source_modality]
        src = np.clip((raw_src - low_s) / (high_s - low_s), 0.0, 1.0)

        raw_tgt = np.asarray(
            f[f"images/{self.target_modality}"][pidx, :, :, z], dtype=np.float32
        )
        low_t, high_t = thresholds[self.target_modality]
        tgt = np.clip((raw_tgt - low_t) / (high_t - low_t), 0.0, 1.0)

        source = torch.from_numpy(src).unsqueeze(0)
        target = torch.from_numpy(tgt).unsqueeze(0)

        source = _pad_to(source, self.image_size)
        target = _pad_to(target, self.image_size)

        # SynDiff range: [-1, 1].
        source = source.mul_(2.0).sub_(1.0)
        target = target.mul_(2.0).sub_(1.0)

        # (x1=contrast1=target, x2=contrast2=source) — matches upstream tuple order.
        return target, source

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_h5"] = None
        return state


def _load_corpus_registry(path: Path | str) -> list[dict]:
    """Read a VENA corpus_registry JSON and return the cohort list."""
    with Path(path).open("r") as f:
        registry = json.load(f)
    if "cohorts" not in registry:
        raise DatasetError(f"corpus registry {path} missing 'cohorts'")
    return registry["cohorts"]


class MultiCohortSynDiffSliceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """ConcatDataset over per-cohort ``SynDiffSliceDataset`` instances.

    Same registry contract as ``MultiCohortImageSliceDataset`` in the pGAN
    integration: filter ``role == role_filter`` (default ``"cv"``), skip cohorts
    whose ``image_h5`` is missing with a WARNING, skip cohorts whose split is
    empty for the requested ``(fold, phase)``.
    """

    def __init__(
        self,
        corpus_registry: Path | str,
        fold: int,
        phase: str,
        target_modality: str = "t1c",
        source_modality: str = "t1pre",
        image_size: int = 256,
        min_brain_voxels: int = 1000,
        max_patients_per_cohort: int | None = None,
        role_filter: str = "cv",
        path_overrides: dict[str, Path | str] | None = None,
    ) -> None:
        self.corpus_registry = Path(corpus_registry)
        self.fold = fold
        self.phase = phase
        self.target_modality = target_modality
        self.source_modality = source_modality
        self.image_size = image_size
        self.min_brain_voxels = min_brain_voxels
        self.max_patients_per_cohort = max_patients_per_cohort
        self.role_filter = role_filter
        self.path_overrides = {k: Path(v) for k, v in (path_overrides or {}).items()}

        cohorts = _load_corpus_registry(self.corpus_registry)
        datasets: list[SynDiffSliceDataset] = []
        self.cohort_names: list[str] = []
        self.cohort_sizes: list[int] = []
        for entry in cohorts:
            name = entry["name"]
            if entry.get("role") != role_filter:
                continue
            h5 = self.path_overrides.get(name, Path(entry["image_h5"]))
            if not h5.is_file():
                logger.warning(
                    "MultiCohortSynDiffSliceDataset: skipping cohort %s — H5 missing at %s",
                    name, h5,
                )
                continue
            try:
                ds = SynDiffSliceDataset(
                    image_h5=h5,
                    fold=fold,
                    phase=phase,
                    target_modality=target_modality,
                    source_modality=source_modality,
                    image_size=image_size,
                    min_brain_voxels=min_brain_voxels,
                    max_patients=max_patients_per_cohort,
                )
            except DatasetError as exc:
                logger.warning(
                    "MultiCohortSynDiffSliceDataset: skipping cohort %s — %s",
                    name, exc,
                )
                continue
            if len(ds) == 0:
                logger.warning(
                    "MultiCohortSynDiffSliceDataset: cohort %s has 0 slices for "
                    "fold=%d phase=%s — skipped", name, fold, phase,
                )
                continue
            datasets.append(ds)
            self.cohort_names.append(name)
            self.cohort_sizes.append(len(ds))

        if not datasets:
            raise DatasetError(
                f"no usable cohorts in {self.corpus_registry} (fold={fold}, "
                f"phase={phase}, role={role_filter})."
            )

        self._concat = ConcatDataset(datasets)
        logger.info(
            "MultiCohortSynDiffSliceDataset[%s/fold%d]: %d cohorts, %d total "
            "slices (per-cohort: %s)",
            phase, fold, len(datasets), len(self._concat),
            ", ".join(f"{n}={s}" for n, s in zip(self.cohort_names, self.cohort_sizes)),
        )

    def __len__(self) -> int:
        return len(self._concat)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._concat[i]
