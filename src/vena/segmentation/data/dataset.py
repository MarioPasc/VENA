"""Image-domain dataset for segmentation training and prediction.

Serves ``{t1pre, t2, flair}`` volumes z-scored on the brain mask plus GT soft
targets ``(2, H, W, D) ∈ [0, 1]`` for ``[TC, NETC]``.

H5 layout expected (schema 2.0.0, per :mod:`vena.rules.h5-design-principles`):

.. code-block:: text

    <cohort>_image.h5
    ├── ids                  vlen-str  (N,)   scan-level IDs (schema ≥2.0.0)
    ├── patients/
    │   ├── keys             vlen-str  (P,)   patient-level keys
    │   └── offsets          int32     (P+1,) CSR: patient k → ids[offsets[k]:offsets[k+1]]
    ├── images/
    │   ├── t1pre            float32   (N, H, W, D)
    │   ├── t2               float32   (N, H, W, D)
    │   └── flair            float32   (N, H, W, D)
    └── masks/
        ├── tumor            int8      (N, H, W, D)  BraTS integer labels
        └── brain            float32   (N, H, W, D)  binary skull-strip mask

Legacy H5s (schema <2.0.0) may use ``patient_ids`` instead of ``ids``;
:func:`_build_id_index` falls back gracefully.

H5 path resolution: the registry entry's absolute ``image_h5`` path is tried
first; ``image_h5_root / filename`` is used as a fallback.  This lets tests
redirect to a temporary directory while the registry JSON keeps absolute
production paths.

Normalisation convention: **z-score on brain** (nonzero voxels, per channel),
independent of the VAE 99.95 percentile.  The ``downstream_seg`` convention.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset

from vena.segmentation.exceptions import SegDataError
from vena.segmentation.targets.soft_targets import make_soft_targets

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from vena.segmentation.config import DataConfig, TargetConfig

logger = logging.getLogger(__name__)

__all__ = ["SegImageDataset"]

# Epsilon for z-score denominator stability
_EPS = 1e-8


# ---------------------------------------------------------------------------
# H5 index building
# ---------------------------------------------------------------------------


def _build_id_index(
    corpus_registry_path: Path,
    image_h5_root: Path,
) -> dict[str, tuple[Path, int]]:
    """Build a scan_id → (h5_path, row_index) mapping.

    Reads the corpus registry JSON, locates per-cohort H5 files, and scans the
    ``ids`` dataset (or legacy ``patient_ids``) in each H5 to build the reverse
    index used by :meth:`SegImageDataset.__getitem__`.

    H5 path resolution (Bug 2 fix): the registry's absolute ``image_h5`` path
    is tried first.  If absent, falls back to ``image_h5_root / basename``.
    This handles per-cohort nested layouts (e.g.
    ``BRATS_GLI/PRE_OPERATIVE/h5/BraTS_GLI_image.h5``) without requiring all
    H5s to live in the same flat directory.

    ID key resolution (Bug 1 fix): prefers ``ids`` (schema 2.0.0) and falls
    back to ``patient_ids`` (schema <2.0.0) for forward-compatibility.

    Parameters
    ----------
    corpus_registry_path:
        Path to a corpus registry JSON following the
        ``routines/fm/train/configs/corpus/corpus_*.json`` schema.
    image_h5_root:
        Fallback directory when the registry's absolute ``image_h5`` path does
        not exist on the current host.

    Returns
    -------
    dict[str, tuple[Path, int]]
        Mapping from scan ID string to ``(h5_path, row_index)`` where
        ``row_index`` is the 0-based position in the H5 ``ids`` array.

    Raises
    ------
    SegDataError
        If the corpus registry cannot be parsed or contains no usable cohorts.
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required for SegImageDataset. Install it via: pip install h5py"
        ) from exc

    if not corpus_registry_path.exists():
        raise SegDataError(f"Corpus registry not found: {corpus_registry_path}")

    with corpus_registry_path.open("r") as fh:
        registry = json.load(fh)

    cohorts = registry.get("cohorts", [])
    if not cohorts:
        raise SegDataError(f"Corpus registry has no 'cohorts' list: {corpus_registry_path}")

    id_index: dict[str, tuple[Path, int]] = {}

    for cohort in cohorts:
        name = cohort.get("name", "<unnamed>")
        raw_h5 = cohort.get("image_h5")
        if not raw_h5:
            logger.debug("Cohort '%s' has no image_h5 — skipping.", name)
            continue

        # Bug 2 fix: try the registry's absolute path first; fall back to
        # image_h5_root / filename only when the absolute path does not exist.
        # The old behaviour (always use image_h5_root / name) silently failed
        # for cohorts whose H5 files live in nested per-cohort subdirectories
        # (e.g. BRATS_GLI/PRE_OPERATIVE/h5/BraTS_GLI_image.h5).
        abs_h5 = Path(raw_h5)
        if abs_h5.exists():
            h5_path = abs_h5
            logger.debug("Cohort '%s': H5 resolved via absolute path: %s", name, h5_path)
        else:
            h5_path = image_h5_root / abs_h5.name
            if h5_path.exists():
                logger.debug(
                    "Cohort '%s': H5 resolved via image_h5_root fallback: %s", name, h5_path
                )

        if not h5_path.exists():
            logger.warning("H5 not found for cohort '%s': %s — skipping.", name, h5_path)
            continue

        try:
            with h5py.File(h5_path, "r") as hf:
                # Bug 1 fix: real H5 schema 2.0.0 uses 'ids' (scan-level),
                # not 'patient_ids'.  Fall back to 'patient_ids' for
                # forward-compatibility with any legacy single-session H5s
                # that predate the 2026-05-19 schema.
                if "ids" in hf:
                    id_key = "ids"
                elif "patient_ids" in hf:
                    id_key = "patient_ids"
                    logger.debug(
                        "H5 '%s': using legacy 'patient_ids' key (schema <2.0.0).", h5_path
                    )
                else:
                    logger.warning(
                        "H5 '%s' has neither 'ids' nor 'patient_ids' dataset — "
                        "skipping cohort '%s'.",
                        h5_path,
                        name,
                    )
                    continue
                # Read as decoded strings (h5py returns bytes for fixed-str, str for vlen)
                raw_ids = hf[id_key][:]
                patient_ids = [
                    (pid.decode() if isinstance(pid, bytes) else str(pid)) for pid in raw_ids
                ]
        except Exception as exc:
            raise SegDataError(f"Failed to read ids from '{h5_path}': {exc}") from exc

        for row_idx, pid in enumerate(patient_ids):
            if pid in id_index:
                logger.debug(
                    "Duplicate patient_id '%s' (already from %s); keeping first.",
                    pid,
                    id_index[pid][0].name,
                )
            else:
                id_index[pid] = (h5_path, row_idx)

    logger.info(
        "Built H5 index: %d patient IDs from %d cohorts.",
        len(id_index),
        len(cohorts),
    )
    return id_index


# ---------------------------------------------------------------------------
# Z-score on brain
# ---------------------------------------------------------------------------


def _zscore_brain(
    volume: NDArray,
    brain_mask: NDArray,
) -> NDArray:
    """Z-score a volume over its brain (nonzero-mask) voxels.

    Parameters
    ----------
    volume:
        Float32 array of shape ``(H, W, D)``.
    brain_mask:
        Boolean or float32 array of shape ``(H, W, D)``.
        Non-zero elements define the brain foreground.

    Returns
    -------
    NDArray
        Float32 array of the same shape.  Brain voxels have mean ≈ 0 and
        std ≈ 1.  Background voxels (brain_mask == 0) are set to 0.0.
    """
    fg = brain_mask.astype(bool)
    if not fg.any():
        # No foreground — return zeros (degenerate scan; should not happen)
        return np.zeros_like(volume, dtype=np.float32)

    brain_vals = volume[fg]
    mean = float(brain_vals.mean())
    std = float(brain_vals.std())
    out = np.zeros_like(volume, dtype=np.float32)
    out[fg] = (brain_vals - mean) / (std + _EPS)
    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SegImageDataset(Dataset):
    """Image-domain dataset serving z-scored MRI + soft tumour targets.

    Returns a dict per sample:

    .. code-block:: text

        {
            "image":      torch.Tensor  (3, H, W, D)  float32  z-scored on brain
            "target":     torch.Tensor  (2, H, W, D)  float32  soft ∈ [0, 1]
            "brain":      torch.Tensor  (1, H, W, D)  float32  binary brain mask
            "patient_id": str
        }

    The channel order of ``"image"`` follows ``cfg.modalities`` (default
    ``("t1pre", "t2", "flair")``).  t1c is never included — its presence
    would constitute direct label leakage.

    Parameters
    ----------
    ids:
        Patient IDs to serve.  Must be a subset of IDs indexed in the corpus
        H5 files.
    cfg:
        Frozen :class:`~vena.segmentation.config.DataConfig`.
    augment:
        If ``True``, apply the full augmentation pipeline from
        :func:`~vena.segmentation.data.augment.build_augmentation`.
        If ``False``, return raw z-scored volumes (deterministic).
    target_fn:
        Callable with signature
        ``(label: (H,W,D) int ndarray, cfg: TargetConfig, image=None)
        → (2,H,W,D) float32``.
        Defaults to :func:`~vena.segmentation.targets.soft_targets.make_soft_targets`.
        Tests may inject a stub that returns fixed synthetic targets without
        running the full SDT pipeline.
    target_cfg:
        :class:`~vena.segmentation.config.TargetConfig` passed to ``target_fn``.
        If ``None``, a default ``TargetConfig()`` is used.
    augmentation_pipeline:
        Pre-built augmentation callable.  If ``None`` and ``augment=True``,
        :func:`~vena.segmentation.data.augment.build_augmentation` is called
        with ``cfg`` at construction time.
    id_index:
        Optional pre-built ``{patient_id: (h5_path, row_index)}`` mapping.
        When provided, the corpus registry is not read.  Useful for injecting
        synthetic H5 fixtures in unit tests.
    """

    def __init__(
        self,
        ids: Sequence[str],
        cfg: DataConfig,
        *,
        augment: bool,
        target_fn: Callable = make_soft_targets,
        target_cfg: TargetConfig | None = None,
        augmentation_pipeline: Callable | None = None,
        id_index: dict[str, tuple[Path, int]] | None = None,
    ) -> None:
        self._ids: tuple[str, ...] = tuple(ids)
        self._cfg = cfg
        self._augment = augment
        self._target_fn = target_fn

        # Resolve target config — import here to avoid circular at module level
        if target_cfg is None:
            from vena.segmentation.config import TargetConfig

            target_cfg = TargetConfig()
        self._target_cfg = target_cfg

        # Build the H5 index (patient_id → (h5_path, row_index))
        if id_index is not None:
            self._index = id_index
        else:
            self._index = _build_id_index(
                corpus_registry_path=cfg.corpus_registry,
                image_h5_root=cfg.image_h5_root,
            )

        # Validate that all requested IDs are in the index
        missing = [pid for pid in self._ids if pid not in self._index]
        if missing:
            raise SegDataError(
                f"{len(missing)} patient IDs not found in H5 index: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )

        # Build augmentation pipeline lazily or accept pre-built
        if augment:
            if augmentation_pipeline is not None:
                self._pipeline = augmentation_pipeline
            else:
                from vena.segmentation.data.augment import build_augmentation

                self._pipeline = build_augmentation(cfg)
        else:
            self._pipeline = None

        logger.info(
            "SegImageDataset: %d patients | augment=%s | modalities=%s",
            len(self._ids),
            augment,
            cfg.modalities,
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, idx: int) -> dict:
        """Load and return a single sample.

        Parameters
        ----------
        idx:
            Integer index into ``self._ids``.

        Returns
        -------
        dict
            Keys: ``"image"`` ``(3,H,W,D)`` float32 tensor,
            ``"target"`` ``(2,H,W,D)`` float32 tensor soft ∈ [0,1],
            ``"brain"`` ``(1,H,W,D)`` float32 tensor,
            ``"patient_id"`` str.
        """
        patient_id = self._ids[idx]
        h5_path, row = self._index[patient_id]

        # Load raw data from H5
        image_vols, label, brain = self._load_h5(h5_path, row)
        # image_vols: dict[str, NDArray] mapping modality -> (H,W,D)
        # label: NDArray (H,W,D) int
        # brain: NDArray (H,W,D) float32

        # Z-score each modality on brain mask
        zscored: dict[str, NDArray] = {}
        for mod, vol in image_vols.items():
            zscored[mod] = _zscore_brain(vol, brain)

        # Compute soft target
        target = self._target_fn(label, self._target_cfg)
        # target: (2, H, W, D) float32

        # Build the augmentation dict
        sample: dict = {
            **{mod: zscored[mod] for mod in self._cfg.modalities},
            "target": target,
            "brain": brain,
        }

        # Apply augmentation pipeline (intensity + spatial + dropout)
        if self._pipeline is not None:
            sample = self._pipeline(sample)

        # Stack modality channels into (C, H, W, D) image tensor
        stacked = np.stack(
            [np.asarray(sample[mod]) for mod in self._cfg.modalities],
            axis=0,
        ).astype(np.float32)

        target_arr = np.asarray(sample["target"]).astype(np.float32)
        brain_arr = np.asarray(sample["brain"])
        if brain_arr.ndim == 3:
            brain_arr = brain_arr[np.newaxis]  # (1,H,W,D)
        brain_arr = brain_arr.astype(np.float32)

        return {
            "image": torch.from_numpy(stacked),
            "target": torch.from_numpy(target_arr),
            "brain": torch.from_numpy(brain_arr),
            "patient_id": patient_id,
        }

    # ------------------------------------------------------------------
    # Internal H5 reader
    # ------------------------------------------------------------------

    def _load_h5(
        self,
        h5_path: Path,
        row: int,
    ) -> tuple[dict[str, NDArray], NDArray, NDArray]:
        """Load modality volumes, tumour label, and brain mask for one patient.

        Parameters
        ----------
        h5_path:
            Path to the cohort-level image H5 file.
        row:
            Row index (0-based) for this patient within the H5 arrays.

        Returns
        -------
        tuple
            ``(image_vols, label, brain)`` where:

            * ``image_vols`` — ``dict[modality, float32 (H,W,D)]``
            * ``label``      — ``int8/int16 (H,W,D)`` BraTS tumour label
            * ``brain``      — ``float32 (H,W,D)`` binary skull-strip mask
        """
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("h5py is required; install via: pip install h5py") from exc

        try:
            with h5py.File(h5_path, "r") as hf:
                image_vols: dict[str, NDArray] = {}
                for mod in self._cfg.modalities:
                    key = f"images/{mod}"
                    if key not in hf:
                        raise SegDataError(
                            f"H5 '{h5_path}' missing dataset '{key}' "
                            f"(required by cfg.modalities={self._cfg.modalities})"
                        )
                    image_vols[mod] = hf[key][row].astype(np.float32)

                if "masks/tumor" not in hf:
                    raise SegDataError(f"H5 '{h5_path}' missing dataset 'masks/tumor'")
                label: NDArray = hf["masks/tumor"][row]

                if "masks/brain" in hf:
                    brain: NDArray = hf["masks/brain"][row].astype(np.float32)
                else:
                    # Fallback: brain = nonzero union of all modalities
                    logger.debug(
                        "H5 '%s' has no 'masks/brain' — deriving from nonzero union.",
                        h5_path,
                    )
                    brain = np.zeros_like(next(iter(image_vols.values())), dtype=np.float32)
                    for vol in image_vols.values():
                        brain = np.maximum(brain, (vol != 0).astype(np.float32))

        except SegDataError:
            raise
        except Exception as exc:
            raise SegDataError(f"Failed to read row {row} from '{h5_path}': {exc}") from exc

        return image_vols, label, brain
