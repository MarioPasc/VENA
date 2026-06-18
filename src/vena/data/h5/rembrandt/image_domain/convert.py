"""NIfTI → H5 converter for the REMBRANDT image-domain cache.

Streaming layout
----------------

1. Index source patients via :class:`REMBRANDTDataset`.
2. Build a deterministic single 53/5/5 train/val/test split (N=63 too small
   for stable nested K-fold CV; mirrors IvyGAP design).
3. Pre-allocate stacked H5 datasets ``(N, 240, 240, 155)`` with
   ``chunks=(1, 240, 240, 155)`` and gzip-4 compression.
4. Dispatch one task per patient to a joblib worker pool; each task loads the
   four modalities (already HD-BET skull-stripped) + the GlistrBoost tumour
   seg, reorients to LPS (no-op for SRI24-preprocessed REMBRANDT), derives
   the brain mask from the union of non-zero voxels across the four
   modalities (post-strip background is exactly 0), and computes the
   brain-centred crop origin.
5. The main process writes each payload into its row; ``h5py`` is not safe
   for parallel writes from worker processes.
6. Validate the file against the manifest before returning.

Intensity policy: no normalisation at write time — native scanner intensities
inside the brain are preserved. Per-modality percentile normalisation is
applied at encode time by the MAISI front-end with ``foreground_only=True``.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from vena.data.h5.shared import (
    H5ConvertError,
    H5Writer,
    assert_h5_valid,
    assign_row,
    compute_crop_origin,
    now_iso_utc,
    resolve_git_sha,
)
from vena.data.niigz.rembrandt import REMBRANDTDataset
from vena.data.niigz.shared.geometry import reorient_to_axcodes
from vena.data.niigz.shared.io import load_nii

from .manifest import (
    REMBRANDT_IMAGE_EXPECTED_SHAPE,
    REMBRANDT_IMAGE_MANIFEST,
    REMBRANDT_IMAGE_SEQUENCE_MAP,
    REMBRANDT_LABEL_SYSTEM,
)

logger = logging.getLogger(__name__)

_PRODUCER_VERSION = "0.1.0"
_PRODUCER = f"vena.data.h5.rembrandt.image_domain.convert:{_PRODUCER_VERSION}"
_LPS: tuple[str, str, str] = ("L", "P", "S")
_COHORT_TAG = "REMBRANDT"
_SEG_FILENAME_SUFFIX = "GlistrBoost_out"


class _Splits(TypedDict):
    train: list[str]
    val: list[str]
    test: list[str]


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------


class REMBRANDTImageH5Config(BaseModel):
    """Resolved configuration for one execution of the REMBRANDT converter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_root: Path
    output_path: Path

    n_jobs: int = 8
    shard_size: int = 16
    crop_box: tuple[int, int, int] = (192, 224, 192)
    n_val: int = 5
    n_test: int = 5
    seed: int = 42

    overwrite: bool = False
    limit: int | None = Field(
        default=None,
        description="Convert only the first ``limit`` patients (smoke runs).",
    )
    log_level: str = "INFO"

    def to_json(self) -> str:
        return self.model_dump_json()


# ----------------------------------------------------------------------------
# Worker-side payload
# ----------------------------------------------------------------------------


def _load_lps(path: Path) -> NDArray[Any]:
    """Load a NIfTI and reorient voxel axes to LPS (no-op for SRI24)."""
    vol = load_nii(path)
    return reorient_to_axcodes(np.asarray(vol.array), vol.affine, _LPS)


def _load_patient_payload(
    patient_id: str,
    patient_root: Path,
    expected_shape: tuple[int, int, int],
    crop_box: tuple[int, int, int],
) -> dict[str, NDArray[Any]]:
    """Load one patient's modalities + tumour seg + brain mask.

    Runs inside a joblib worker; must be picklable. Brain mask is derived from
    the union of nonzero voxels across the four modalities (background is
    exactly zero post HD-BET skull-strip).

    Raises
    ------
    H5ConvertError
        On missing files, shape mismatches, or a brain extent the common crop
        box cannot contain.
    """
    out: dict[str, NDArray[Any]] = {}
    nonzero_union: NDArray[Any] | None = None

    for slug, suffix in REMBRANDT_IMAGE_SEQUENCE_MAP.items():
        path = patient_root / f"{patient_id}_{suffix}_LPS_rSRI.nii.gz"
        if not path.exists():
            raise H5ConvertError(f"{patient_id}: missing {path.name}")
        arr = np.ascontiguousarray(_load_lps(path), dtype=np.float32)
        if arr.shape != expected_shape:
            raise H5ConvertError(
                f"{patient_id}: {slug} shape {arr.shape} != expected {expected_shape}"
            )
        out[f"images/{slug}"] = arr
        nonzero = arr != 0.0
        nonzero_union = nonzero if nonzero_union is None else (nonzero_union | nonzero)

    seg_path = patient_root / f"{patient_id}_{_SEG_FILENAME_SUFFIX}.nii.gz"
    if not seg_path.exists():
        raise H5ConvertError(f"{patient_id}: missing tumour segmentation {seg_path.name}")
    seg = _load_lps(seg_path)
    if seg.shape != expected_shape:
        raise H5ConvertError(f"{patient_id}: seg shape {seg.shape} != expected {expected_shape}")
    out["masks/tumor"] = seg.astype(np.int8, copy=False)

    # Brain mask: union of nonzero across modalities + CC-clean.
    # See `.claude/notes/data/2026-06-18_data_audit.md`.
    assert nonzero_union is not None  # at least one modality loaded above.
    brain_bin = clean_brain_mask(nonzero_union.astype(np.int8))
    out["masks/brain"] = brain_bin

    try:
        origin = compute_crop_origin(brain_bin, crop_box)
    except ValueError as exc:
        raise H5ConvertError(f"{patient_id}: crop geometry failed: {exc}") from exc
    out["crop/origin"] = np.asarray(origin, dtype=np.int32)
    return out


def _worker(
    row_index: int,
    patient_id: str,
    patient_root: Path,
    expected_shape: tuple[int, int, int],
    crop_box: tuple[int, int, int],
) -> tuple[int, str, dict[str, NDArray[Any]] | None, str | None]:
    """Adapter so the main loop receives a uniform tuple per task."""
    try:
        payload = _load_patient_payload(patient_id, patient_root, expected_shape, crop_box)
        return (row_index, patient_id, payload, None)
    except H5ConvertError as exc:
        return (row_index, patient_id, None, str(exc))
    except Exception as exc:
        return (row_index, patient_id, None, f"unexpected: {exc!r}")


# ----------------------------------------------------------------------------
# Converter
# ----------------------------------------------------------------------------


class REMBRANDTImageH5Converter:
    """Run one end-to-end conversion of REMBRANDT to image-domain H5."""

    def __init__(self, cfg: REMBRANDTImageH5Config) -> None:
        self.cfg = cfg

    def run(self) -> Path:
        cfg = self.cfg
        dataset = REMBRANDTDataset(cfg.source_root)
        patients = list(dataset)
        if cfg.limit is not None:
            patients = patients[: cfg.limit]
        if not patients:
            raise H5ConvertError(f"No patients discovered under {cfg.source_root}")
        patient_ids = [p.patient_id for p in patients]
        n = len(patient_ids)
        logger.info("REMBRANDT conversion: n_patients=%d", n)

        splits = self._build_splits(patient_ids)

        manifest = REMBRANDT_IMAGE_MANIFEST
        timestamp = now_iso_utc()
        git_sha = resolve_git_sha()

        with H5Writer(
            cfg.output_path,
            manifest=manifest,
            config_json=cfg.to_json(),
            producer=_PRODUCER,
            created_at=timestamp,
            git_sha=git_sha,
            extra_root_attrs={
                "split_role": "internal",
                "longitudinal": False,
                "label_system": REMBRANDT_LABEL_SYSTEM,
                "crop_box": json.dumps(list(cfg.crop_box)),
                "orientation": "LPS",
                "subset_label": "REMBRANDT-CBICA-preprocessed",
            },
            overwrite=cfg.overwrite,
        ) as w:
            ids_dset = w.create_1d(manifest.get("ids"), n=n)
            ids_dset[:] = np.asarray(patient_ids, dtype=object)

            image_dsets = {
                slug: w.create_stacked(
                    manifest.get(f"images/{slug}"),
                    n=n,
                    spatial_shape=REMBRANDT_IMAGE_EXPECTED_SHAPE,
                )
                for slug in REMBRANDT_IMAGE_SEQUENCE_MAP
            }
            tumor_dset = w.create_stacked(
                manifest.get("masks/tumor"),
                n=n,
                spatial_shape=REMBRANDT_IMAGE_EXPECTED_SHAPE,
            )
            brain_dset = w.create_stacked(
                manifest.get("masks/brain"),
                n=n,
                spatial_shape=REMBRANDT_IMAGE_EXPECTED_SHAPE,
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
                shard = patients[shard_start : shard_start + cfg.shard_size]
                shard_tasks = [
                    delayed(_worker)(
                        shard_start + j,
                        p.patient_id,
                        p.root,
                        REMBRANDT_IMAGE_EXPECTED_SHAPE,
                        cfg.crop_box,
                    )
                    for j, p in enumerate(shard)
                ]
                results = Parallel(n_jobs=cfg.n_jobs, backend="loky")(shard_tasks)
                for row_index, patient_id, payload, error in results:
                    done += 1
                    if error is not None:
                        skipped.append({"patient_id": patient_id, "reason": error})
                        logger.warning("skip %s: %s", patient_id, error)
                    else:
                        for slug in REMBRANDT_IMAGE_SEQUENCE_MAP:
                            assign_row(image_dsets[slug], row_index, payload[f"images/{slug}"])
                        assign_row(tumor_dset, row_index, payload["masks/tumor"])
                        assign_row(brain_dset, row_index, payload["masks/brain"])
                        assign_row(crop_origin_dset, row_index, payload["crop/origin"])
                    if done % log_every == 0 or done == n:
                        elapsed = time.monotonic() - t0
                        rate = done / elapsed if elapsed > 0 else 0.0
                        eta = (n - done) / rate if rate > 0 else float("inf")
                        logger.info(
                            "progress %d/%d (%.1f%%) rate=%.2f patients/s eta=%.0fs skipped=%d",
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
                logger.warning("Skipped %d patient(s). See attrs/skipped_json.", len(skipped))
                w.file.attrs["skipped_json"] = json.dumps(skipped)

            # ---- CSR patient grouping (trivial 1:1) ------------------------
            w.write_int_1d(
                "patients/offsets",
                np.arange(n + 1, dtype=np.int32),
                dtype="int32",
                description=(
                    "CSR offsets; scans of patient k are rows [offsets[k]:offsets[k+1]] "
                    "(1:1 for REMBRANDT cross-sectional)."
                ),
            )
            w.write_vlen_str_1d(
                "patients/keys",
                list(patient_ids),
                description="Unique patient keys (<pid>_<date>) in offset order.",
            )

            # ---- splits ----------------------------------------------------
            self._write_splits(w, splits)

            w.file.attrs["n_patients_written"] = n - len(skipped)

        try:
            assert_h5_valid(cfg.output_path, manifest)
        except Exception:
            cfg.output_path.unlink(missing_ok=True)
            raise
        logger.info("Wrote REMBRANDT H5 cache: %s", cfg.output_path)
        return cfg.output_path

    # ---- splits builder (single random partition; mirrors IvyGAP) ----------

    def _build_splits(self, patient_ids: list[str]) -> _Splits:
        cfg = self.cfg
        n = len(patient_ids)
        # Floors so small smoke runs still produce a valid partition.
        n_test = min(cfg.n_test, max(0, n - 2))
        n_val = min(cfg.n_val, max(0, n - n_test - 1))
        if n_test + n_val >= n:
            raise H5ConvertError(f"split sizes too large: n_test={n_test} + n_val={n_val} >= n={n}")
        rng = np.random.default_rng(cfg.seed)
        perm = rng.permutation(n)
        test_idx = np.sort(perm[:n_test])
        val_idx = np.sort(perm[n_test : n_test + n_val])
        train_idx = np.sort(perm[n_test + n_val :])
        ids = list(patient_ids)
        splits: _Splits = {
            "train": [ids[i] for i in train_idx],
            "val": [ids[i] for i in val_idx],
            "test": [ids[i] for i in test_idx],
        }
        logger.info(
            "REMBRANDT splits: train=%d val=%d test=%d (seed=%d)",
            len(splits["train"]),
            len(splits["val"]),
            len(splits["test"]),
            cfg.seed,
        )
        return splits

    def _write_splits(self, w: H5Writer, splits: _Splits) -> None:
        # Canonical layout (mirrors BraTS-GLI / UCSF-PDGM / LUMIERE):
        #   splits/test                       — held-out test patient IDs
        #   splits/cv/fold_0/{train,val}      — single-fold partition of the rest
        # The trainer (vena.model.fm.lightning.data) reads
        # ``splits/cv/fold_<fold>/{train,val}`` directly — without these keys
        # the LatentH5DataModule raises a KeyError at setup. N=63 is too small
        # for stable nested K-fold CV, hence a single fold (mirrors IvyGAP's
        # 24/5/5 intent but in the canonical CV layout).
        w.write_vlen_str_1d("splits/test", splits["test"])
        w.write_vlen_str_1d("splits/cv/fold_0/train", splits["train"])
        w.write_vlen_str_1d("splits/cv/fold_0/val", splits["val"])
        grp = w.file["splits"]
        grp.attrs["description"] = (
            "Patient-ID-based single-fold split. splits/test is the held-out "
            "set; splits/cv/fold_0/{train,val} is the single-fold partition "
            "(N=63 too small for stable nested K-fold CV)."
        )
        grp.attrs["n_folds"] = 1
