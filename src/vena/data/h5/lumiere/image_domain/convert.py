"""NIfTI → H5 converter for the LUMIERE image-domain cache.

Streaming layout:

1. Index source sessions via :class:`vena.data.niigz.lumiere.LUMIEREDataset`.
2. Compute patient-level splits (nested CV, role="cv"; no stratification
   today — the LUMIERE demographics CSV is not consumed yet).
3. Pre-allocate stacked H5 datasets ``(N, 182, 218, 182)`` with
   ``chunks=(1, 182, 218, 182)`` and gzip-4 compression.
4. Dispatch one task per session to a joblib worker pool; each task loads
   the four DeepBraTumIA atlas/skull-strip modalities, the seg, and the
   atlas-space brain mask. Reorients to LPS (no-op for MNI152 LPS atlases),
   casts dtypes, asserts the shape contract, computes the brain-centred
   crop origin from the provided brain mask.
5. The main process consumes the worker generator and writes each payload
   into its row; h5py is not safe for parallel writes from workers.
6. Write CSR patient grouping (patients/offsets, patients/keys).
7. Write splits (splits/test, splits/cv/fold_k/{train,val}).
8. Validate the file against the manifest before returning the path.

Intensity policy: no normalisation at write time. Raw skull-stripped floats
from the DeepBraTumIA atlas are kept verbatim; the MAISI percentile norm at
encode time handles the cohort-specific intensity scale.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from vena.data.h5.shared import (
    H5ConvertError,
    H5Writer,
    NestedCVSplits,
    assert_h5_valid,
    assign_row,
    compute_crop_origin,
    make_cohort_splits,
    now_iso_utc,
    resolve_git_sha,
)
from vena.data.niigz.lumiere import LUMIEREDataset, LUMIERESession
from vena.data.niigz.shared.geometry import reorient_to_axcodes
from vena.data.niigz.shared.io import load_nii

from .manifest import (
    LUMIERE_IMAGE_EXPECTED_SHAPE,
    LUMIERE_IMAGE_MANIFEST,
    LUMIERE_LABEL_SYSTEM,
)

logger = logging.getLogger(__name__)

_PRODUCER_VERSION = "0.1.0"
_PRODUCER = f"vena.data.h5.lumiere.image_domain.convert:{_PRODUCER_VERSION}"
_LPS: tuple[str, str, str] = ("L", "P", "S")

# Relative paths under each session directory (mirrors the niigz reader).
_SKULL_STRIP_RELPATH = Path("DeepBraTumIA-segmentation") / "atlas" / "skull_strip"
_SEG_RELPATH = Path("DeepBraTumIA-segmentation") / "atlas" / "segmentation" / "seg_mask.nii.gz"
_BRAIN_MASK_RELPATH = _SKULL_STRIP_RELPATH / "brain_mask.nii.gz"

_LUMIERE_MODALITY_FILENAME: dict[str, str] = {
    "t1pre": "t1_skull_strip.nii.gz",
    "t1c": "ct1_skull_strip.nii.gz",
    "t2": "t2_skull_strip.nii.gz",
    "flair": "flair_skull_strip.nii.gz",
}


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------


class LUMIEREImageH5Config(BaseModel):
    """Resolved configuration for one execution of the LUMIERE converter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_root: Path
    output_path: Path

    n_jobs: int = 8
    shard_size: int = 32
    crop_box: tuple[int, int, int] = (192, 224, 192)
    test_fraction: float = 0.11  # ~10 of 91 patients.
    n_test_min: int = 10
    n_folds: int = 5
    seed: int = 42

    overwrite: bool = False
    limit: int | None = Field(
        default=None,
        description=(
            "Convert only the first ``limit`` patients (smoke runs). All "
            "sessions of each kept patient are included, preserving CSR contiguity."
        ),
    )
    log_level: str = "INFO"

    def to_json(self) -> str:
        return self.model_dump_json()


# ----------------------------------------------------------------------------
# Worker-side payload
# ----------------------------------------------------------------------------


def _load_lps(path: Path) -> NDArray[Any]:
    """Load a NIfTI and reorient voxel axes to LPS."""
    vol = load_nii(path)
    return reorient_to_axcodes(np.asarray(vol.array), vol.affine, _LPS)


def _load_session_payload(
    session_root: Path,
    session_id: str,
    expected_shape: tuple[int, int, int],
    crop_box: tuple[int, int, int],
) -> dict[str, NDArray[Any]]:
    """Load one session's modalities + tumour seg + brain mask, reoriented to LPS.

    Runs inside a joblib worker; must be picklable.

    Raises
    ------
    H5ConvertError
        On missing files, shape mismatches, or a brain extent the common crop
        box cannot contain.
    """
    out: dict[str, NDArray[Any]] = {}
    skull_strip_dir = session_root / _SKULL_STRIP_RELPATH

    for slug, fname in _LUMIERE_MODALITY_FILENAME.items():
        path = skull_strip_dir / fname
        if not path.exists():
            raise H5ConvertError(f"{session_id}: missing {fname}")
        arr = np.ascontiguousarray(_load_lps(path), dtype=np.float32)
        if arr.shape != expected_shape:
            raise H5ConvertError(
                f"{session_id}: {slug} shape {arr.shape} != expected {expected_shape}"
            )
        out[f"images/{slug}"] = arr

    seg_path = session_root / _SEG_RELPATH
    if not seg_path.exists():
        raise H5ConvertError(f"{session_id}: missing tumour seg {seg_path.name}")
    seg = _load_lps(seg_path)
    if seg.shape != expected_shape:
        raise H5ConvertError(f"{session_id}: seg shape {seg.shape} != expected {expected_shape}")
    out["masks/tumor"] = seg.astype(np.int8, copy=False)

    brain_path = session_root / _BRAIN_MASK_RELPATH
    if not brain_path.exists():
        raise H5ConvertError(f"{session_id}: missing brain mask {brain_path.name}")
    brain = _load_lps(brain_path)
    if brain.shape != expected_shape:
        raise H5ConvertError(
            f"{session_id}: brain shape {brain.shape} != expected {expected_shape}"
        )
    brain_bin = (brain > 0.5).astype(np.int8)
    out["masks/brain"] = brain_bin

    try:
        origin = compute_crop_origin(brain_bin, crop_box)
    except ValueError as exc:
        raise H5ConvertError(f"{session_id}: crop geometry failed: {exc}") from exc
    out["crop/origin"] = np.asarray(origin, dtype=np.int32)
    return out


def _worker(
    row_index: int,
    session_id: str,
    session_root: Path,
    expected_shape: tuple[int, int, int],
    crop_box: tuple[int, int, int],
) -> tuple[int, str, dict[str, NDArray[Any]] | None, str | None]:
    """Adapter returning a uniform tuple per task for the main process."""
    try:
        payload = _load_session_payload(session_root, session_id, expected_shape, crop_box)
        return (row_index, session_id, payload, None)
    except H5ConvertError as exc:
        return (row_index, session_id, None, str(exc))
    except Exception as exc:
        return (row_index, session_id, None, f"unexpected: {exc!r}")


# ----------------------------------------------------------------------------
# Converter
# ----------------------------------------------------------------------------


class LUMIEREImageH5Converter:
    """Run one end-to-end conversion of the LUMIERE source tree to H5."""

    def __init__(self, cfg: LUMIEREImageH5Config) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------ public

    def run(self) -> Path:
        cfg = self.cfg
        dataset = LUMIEREDataset(cfg.source_root)
        patient_groups = dataset.patient_groups()

        if cfg.limit is not None:
            patient_groups = patient_groups[: cfg.limit]
        if not patient_groups:
            raise H5ConvertError(f"No sessions discovered under {cfg.source_root}")

        patient_ids = [pid for pid, _ in patient_groups]
        all_sessions = dataset.sessions()
        sessions = [all_sessions[row_idx] for _, indices in patient_groups for row_idx in indices]
        n = len(sessions)
        n_patients = len(patient_ids)
        logger.info(
            "LUMIERE image-domain conversion: n_sessions=%d n_patients=%d",
            n,
            n_patients,
        )

        session_ids = [s.session_id for s in sessions]
        splits = self._build_splits(patient_ids)

        timestamp = now_iso_utc()
        git_sha = resolve_git_sha()
        manifest = LUMIERE_IMAGE_MANIFEST

        with H5Writer(
            cfg.output_path,
            manifest=manifest,
            config_json=cfg.to_json(),
            producer=_PRODUCER,
            created_at=timestamp,
            git_sha=git_sha,
            extra_root_attrs={
                "split_role": "cv",
                "longitudinal": True,
                "label_system": LUMIERE_LABEL_SYSTEM,
                "crop_box": json.dumps(list(cfg.crop_box)),
                "orientation": "LPS",
            },
            overwrite=cfg.overwrite,
        ) as w:
            ids_dset = w.create_1d(manifest.get("ids"), n=n)
            ids_dset[:] = np.asarray(session_ids, dtype=object)

            image_dsets = {
                slug: w.create_stacked(
                    manifest.get(f"images/{slug}"),
                    n=n,
                    spatial_shape=LUMIERE_IMAGE_EXPECTED_SHAPE,
                )
                for slug in _LUMIERE_MODALITY_FILENAME
            }
            tumor_dset = w.create_stacked(
                manifest.get("masks/tumor"),
                n=n,
                spatial_shape=LUMIERE_IMAGE_EXPECTED_SHAPE,
            )
            brain_dset = w.create_stacked(
                manifest.get("masks/brain"),
                n=n,
                spatial_shape=LUMIERE_IMAGE_EXPECTED_SHAPE,
            )
            crop_origin_dset = w.create_stacked(
                manifest.get("crop/origin"),
                n=n,
                spatial_shape=(3,),
            )

            skipped: list[dict[str, str]] = []
            log_every = max(1, n // 50)
            t0 = time.monotonic()
            done = 0
            for shard_start in range(0, n, cfg.shard_size):
                shard = sessions[shard_start : shard_start + cfg.shard_size]
                shard_tasks = [
                    delayed(_worker)(
                        shard_start + j,
                        s.session_id,
                        s.root,
                        LUMIERE_IMAGE_EXPECTED_SHAPE,
                        cfg.crop_box,
                    )
                    for j, s in enumerate(shard)
                ]
                results = Parallel(n_jobs=cfg.n_jobs, backend="loky")(shard_tasks)
                for row_index, session_id, payload, error in results:
                    done += 1
                    if error is not None:
                        skipped.append({"session_id": session_id, "reason": error})
                        logger.warning("skip %s: %s", session_id, error)
                    else:
                        for slug in _LUMIERE_MODALITY_FILENAME:
                            assign_row(image_dsets[slug], row_index, payload[f"images/{slug}"])
                        assign_row(tumor_dset, row_index, payload["masks/tumor"])
                        assign_row(brain_dset, row_index, payload["masks/brain"])
                        assign_row(crop_origin_dset, row_index, payload["crop/origin"])
                    if done % log_every == 0 or done == n:
                        elapsed = time.monotonic() - t0
                        rate = done / elapsed if elapsed > 0 else 0.0
                        eta = (n - done) / rate if rate > 0 else float("inf")
                        logger.info(
                            "progress %d/%d (%.1f%%) rate=%.2f sessions/s eta=%.0fs skipped=%d",
                            done,
                            n,
                            100.0 * done / n,
                            rate,
                            eta,
                            len(skipped),
                        )
                        sys.stdout.flush()
                del results, shard_tasks
                w.file.flush()

            if skipped:
                logger.warning("Skipped %d session(s). See attrs/skipped_json.", len(skipped))
                w.file.attrs["skipped_json"] = json.dumps(skipped)

            # ---- metadata --------------------------------------------------
            self._write_metadata(w, sessions, n)

            # ---- CSR patient grouping --------------------------------------
            group_sizes = [len(indices) for _, indices in patient_groups]
            offsets = np.zeros(n_patients + 1, dtype=np.int32)
            offsets[1:] = np.cumsum(group_sizes, dtype=np.int32)
            assert int(offsets[-1]) == n, (
                f"CSR invariant violated: offsets[-1]={offsets[-1]} != n_sessions={n}"
            )
            w.write_int_1d(
                "patients/offsets",
                offsets,
                dtype="int32",
                description=(
                    "CSR offsets; sessions of patient k are rows [offsets[k]:offsets[k+1]]."
                ),
            )
            w.write_vlen_str_1d(
                "patients/keys",
                patient_ids,
                description="Unique patient keys (Patient-NNN) in offset order.",
            )

            # ---- splits ----------------------------------------------------
            self._write_splits(w, splits)

            w.file.attrs["n_sessions_written"] = n - len(skipped)
            w.file.attrs["n_patients"] = n_patients

        try:
            assert_h5_valid(cfg.output_path, manifest)
        except Exception:
            cfg.output_path.unlink(missing_ok=True)
            raise
        logger.info("Wrote LUMIERE H5 cache: %s", cfg.output_path)
        return cfg.output_path

    # ------------------------------------------------------------------ helpers

    def _build_splits(self, patient_ids: list[str]) -> NestedCVSplits:
        cfg = self.cfg
        return make_cohort_splits(
            patient_ids,
            n_folds=cfg.n_folds,
            test_fraction=cfg.test_fraction,
            n_test_min=cfg.n_test_min,
            seed=cfg.seed,
            stratify_by=None,
            role="cv",
        )

    def _write_splits(self, w: H5Writer, splits: NestedCVSplits) -> None:
        w.write_vlen_str_1d("splits/test", splits["test"])
        for fold_idx, fold in splits["folds"].items():
            w.write_vlen_str_1d(f"splits/cv/fold_{fold_idx}/train", fold["train"])
            w.write_vlen_str_1d(f"splits/cv/fold_{fold_idx}/val", fold["val"])
        grp = w.file["splits"]
        grp.attrs["description"] = (
            "Patient-ID-based nested CV splits. splits/test is the held-out "
            "set shared across folds; splits/cv/fold_K/{train,val} are the "
            "per-fold CV partitions of the remaining patients."
        )
        grp.attrs["n_folds"] = len(splits["folds"])

    def _write_metadata(self, w: H5Writer, sessions: list[LUMIERESession], n: int) -> None:
        manifest = LUMIERE_IMAGE_MANIFEST
        pid_dset = w.create_1d(manifest.get("metadata/patient_id"), n=n)
        pid_dset[:] = np.asarray([s.patient_id for s in sessions], dtype=object)
        week_dset = w.create_1d(manifest.get("metadata/week"), n=n)
        week_dset[:] = np.asarray(
            [int(s.metadata.get("week", -1)) for s in sessions], dtype=np.int32
        )
        repeat_dset = w.create_1d(manifest.get("metadata/week_repeat"), n=n)
        repeat_dset[:] = np.asarray(
            [int(s.metadata.get("week_repeat", -1)) for s in sessions], dtype=np.int32
        )
