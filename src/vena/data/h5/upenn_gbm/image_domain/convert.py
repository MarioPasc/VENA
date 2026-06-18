"""NIfTI → H5 converter for the UPENN-GBM image-domain cache.

Mirrors the UCSF-PDGM converter pattern:

1. Index source patients via :class:`vena.data.niigz.UPENNGBMDataset`. The
   reader filters out the ~60 structural patients with no segmentation
   (manual or automated) so they never enter ``ids``/``patients/keys``.
2. Compute splits in patient-ID space (no dependence on stack order).
3. Pre-allocate stacked H5 datasets ``(N, 240, 240, 155)`` with
   ``chunks=(1, 240, 240, 155)`` and gzip-4 compression.
4. Dispatch one task per patient to a joblib worker pool; each task loads
   the four MR sequences and the resolved tumour segmentation (manual or
   automated, decided by the reader and threaded through as a path), casts
   them to the target dtypes, builds the brain mask as the union of
   nonzero voxels across modalities, asserts the shape contract, and
   returns a small payload.
5. Per-patient metadata (``brats21_id``, ``brats21_data_collection``,
   ``seg_source``) is written from the in-memory metadata dict the reader
   already populated.
6. Validate the file against the manifest before returning the output path.

Intensity policy: no normalisation at write time (principle 6 of the H5
design rules). Encode-time percentile normalisation lives in
``routines/encode/maisi/configs/upenn_gbm_server3.yaml``.
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
from vena.data.niigz import UPENNGBMDataset
from vena.data.niigz.shared.geometry import reorient_to_axcodes
from vena.data.niigz.shared.io import load_nii

from .manifest import (
    UPENN_GBM_IMAGE_EXPECTED_SHAPE,
    UPENN_GBM_IMAGE_MANIFEST,
    UPENN_GBM_IMAGE_SEQUENCE_MAP,
    UPENN_GBM_LABEL_SYSTEM,
    UPENN_GBM_METADATA_FIELDS,
    MetadataFieldSpec,
)

logger = logging.getLogger(__name__)

_PRODUCER_VERSION = "0.1.0"
_PRODUCER = f"vena.data.h5.upenn_gbm.image_domain.convert:{_PRODUCER_VERSION}"
_LPS: tuple[str, str, str] = ("L", "P", "S")


# ----------------------------------------------------------------------------
# Worker-side payload
# ----------------------------------------------------------------------------


def _load_lps(path: Path) -> NDArray[Any]:
    """Load a NIfTI and reorient its voxel axes to LPS (identity for UPenn)."""
    vol = load_nii(path)
    return reorient_to_axcodes(np.asarray(vol.array), vol.affine, _LPS)


def _load_patient_payload(
    patient_root: Path,
    patient_id: str,
    seg_path: Path,
    expected_shape: tuple[int, int, int],
    crop_box: tuple[int, int, int],
) -> dict[str, NDArray[Any]]:
    """Load one patient's four modalities + tumour seg + derive brain mask.

    Runs inside a joblib worker; must be picklable. Returns a plain dict so
    the main process can iterate without re-instantiating any project class.
    Derives the brain mask from the union of nonzero voxels across the four
    skull-stripped modalities (BraTS-PED pattern).
    """
    out: dict[str, NDArray[Any]] = {}
    nonzero_union: NDArray[np.bool_] | None = None
    for slug, suffix in UPENN_GBM_IMAGE_SEQUENCE_MAP.items():
        f = patient_root / f"{patient_id}_{suffix}.nii.gz"
        if not f.exists():
            raise H5ConvertError(f"{patient_id}: missing {f.name}")
        arr = np.ascontiguousarray(_load_lps(f), dtype=np.float32)
        if arr.shape != expected_shape:
            raise H5ConvertError(
                f"{patient_id}: {slug} shape {arr.shape} != expected {expected_shape}"
            )
        out[f"images/{slug}"] = arr
        nonzero = arr != 0
        nonzero_union = nonzero if nonzero_union is None else (nonzero_union | nonzero)

    if not seg_path.exists():
        raise H5ConvertError(f"{patient_id}: missing tumor seg {seg_path.name}")
    seg = _load_lps(seg_path)
    if seg.shape != expected_shape:
        raise H5ConvertError(
            f"{patient_id}: tumor seg shape {seg.shape} != expected {expected_shape}"
        )
    # BraTS-2021 labels {0, 1, 2, 4} — fits comfortably in int8.
    out["masks/tumor"] = seg.astype(np.int8, copy=False)

    # Brain mask: union of nonzero across modalities, then CC-clean to
    # drop boundary jitter. See `.claude/notes/data/2026-06-18_data_audit.md`.
    assert nonzero_union is not None
    brain_bin = clean_brain_mask(nonzero_union.astype(np.int8, copy=False))
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
    seg_path: Path,
    expected_shape: tuple[int, int, int],
    crop_box: tuple[int, int, int],
) -> tuple[int, str, dict[str, NDArray[Any]] | None, str | None]:
    """Adapter so the main loop receives a uniform tuple per task."""
    try:
        payload = _load_patient_payload(
            patient_root, patient_id, seg_path, expected_shape, crop_box
        )
        return (row_index, patient_id, payload, None)
    except H5ConvertError as exc:
        return (row_index, patient_id, None, str(exc))
    except Exception as exc:  # pragma: no cover - last-ditch annotation
        return (row_index, patient_id, None, f"unexpected: {exc!r}")


# ----------------------------------------------------------------------------
# Metadata helpers
# ----------------------------------------------------------------------------


def _cast_metadata(value: Any, cast: str) -> Any:
    """Cast an in-memory metadata cell to the type declared in the manifest.

    NaN/None policy:
      * ``str``  → ``""``
      * ``int``  → ``-1``
      * ``float`` → ``NaN`` (preserved)
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        if cast == "str":
            return ""
        if cast == "int":
            return -1
        return float("nan")
    if cast == "str":
        return str(value)
    if cast == "int":
        try:
            return int(value)
        except (TypeError, ValueError):
            return -1
    if cast == "float":
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")
    raise ValueError(f"unknown cast: {cast!r}")


def _extract_metadata_column(
    patients_metadata: list[dict[str, Any]],
    field: MetadataFieldSpec,
) -> NDArray[Any]:
    col = field["csv_column"]
    cast = field["cast"]
    values = [_cast_metadata(meta.get(col, None), cast) for meta in patients_metadata]
    if cast == "str":
        return np.asarray(values, dtype=object)
    if cast == "int":
        return np.asarray(values, dtype=np.int8)
    if cast == "float":
        return np.asarray(values, dtype=np.float32)
    raise ValueError(f"unknown cast: {cast!r}")


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------


class UPENNGBMImageH5Config(BaseModel):
    """Resolved configuration for one execution of the converter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_root: Path
    metadata_csv: Path
    output_path: Path

    n_jobs: int = 8
    shard_size: int = 32
    crop_box: tuple[int, int, int] = (192, 224, 192)
    test_fraction: float = 0.10
    n_test_min: int = 25
    n_folds: int = 5
    seed: int = 42

    overwrite: bool = False
    limit: int | None = Field(
        default=None,
        description="Optional: convert only the first ``limit`` patients (smoke runs).",
    )
    log_level: str = "INFO"

    def to_json(self) -> str:
        return self.model_dump_json()


# ----------------------------------------------------------------------------
# Converter
# ----------------------------------------------------------------------------


class UPENNGBMImageH5Converter:
    """Run one end-to-end conversion of the UPENN-GBM source tree to H5."""

    def __init__(self, cfg: UPENNGBMImageH5Config) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------ public

    def run(self) -> Path:
        cfg = self.cfg
        dataset = UPENNGBMDataset(cfg.source_root, cfg.metadata_csv)
        patients = list(dataset)
        if cfg.limit is not None:
            patients = patients[: cfg.limit]
        if not patients:
            raise H5ConvertError(f"No patients discovered under {cfg.source_root}")
        patient_ids = [p.patient_id for p in patients]
        patient_metadata = [dict(p.metadata) for p in patients]
        # Resolve seg paths up-front (main process); workers receive the
        # absolute path so they do not need to know the manual/auto layout.
        seg_paths: list[Path] = []
        for p in patients:
            seg_path, seg_source = dataset._resolve_seg(p.patient_id)
            if seg_path is None:
                # Defensive: the reader already filtered these out.
                raise H5ConvertError(
                    f"{p.patient_id}: no segmentation found "
                    "(reader filter should have dropped this)"
                )
            seg_paths.append(seg_path)
            # Mirror seg_source into the metadata dict so the manifest field is populated.
            patient_metadata[len(seg_paths) - 1]["seg_source"] = seg_source

        n = len(patient_ids)
        logger.info("UPENN-GBM image-domain conversion: n_patients=%d", n)

        # ---- splits ---------------------------------------------------------
        splits = self._build_splits(patient_ids)

        # ---- write header & allocate datasets ------------------------------
        timestamp = now_iso_utc()
        git_sha = resolve_git_sha()
        manifest = UPENN_GBM_IMAGE_MANIFEST

        with H5Writer(
            cfg.output_path,
            manifest=manifest,
            config_json=cfg.to_json(),
            producer=_PRODUCER,
            created_at=timestamp,
            git_sha=git_sha,
            extra_root_attrs={
                "split_role": "cv",
                "longitudinal": False,
                "label_system": UPENN_GBM_LABEL_SYSTEM,
                "crop_box": json.dumps(list(cfg.crop_box)),
                "orientation": "LPS",
            },
            overwrite=cfg.overwrite,
        ) as w:
            ids_spec = manifest.get("ids")
            ids_dset = w.create_1d(ids_spec, n=n)
            ids_dset[:] = np.asarray(patient_ids, dtype=object)

            image_dsets = {
                slug: w.create_stacked(
                    manifest.get(f"images/{slug}"),
                    n=n,
                    spatial_shape=UPENN_GBM_IMAGE_EXPECTED_SHAPE,
                )
                for slug in UPENN_GBM_IMAGE_SEQUENCE_MAP
            }
            tumor_dset = w.create_stacked(
                manifest.get("masks/tumor"),
                n=n,
                spatial_shape=UPENN_GBM_IMAGE_EXPECTED_SHAPE,
            )
            brain_dset = w.create_stacked(
                manifest.get("masks/brain"),
                n=n,
                spatial_shape=UPENN_GBM_IMAGE_EXPECTED_SHAPE,
            )
            crop_origin_dset = w.create_stacked(
                manifest.get("crop/origin"),
                n=n,
                spatial_shape=(3,),
            )

            # ---- per-patient parallel fill (sharded to bound peak RAM) -----
            skipped: list[dict[str, str]] = []
            log_every = max(1, n // 50)
            t0 = time.monotonic()
            done = 0
            for shard_start in range(0, n, cfg.shard_size):
                shard = patients[shard_start : shard_start + cfg.shard_size]
                shard_seg = seg_paths[shard_start : shard_start + cfg.shard_size]
                shard_tasks = [
                    delayed(_worker)(
                        shard_start + j,
                        p.patient_id,
                        p.root,
                        s,
                        UPENN_GBM_IMAGE_EXPECTED_SHAPE,
                        cfg.crop_box,
                    )
                    for j, (p, s) in enumerate(zip(shard, shard_seg, strict=True))
                ]
                results = Parallel(n_jobs=cfg.n_jobs, backend="loky")(shard_tasks)
                for row_index, patient_id, payload, error in results:
                    done += 1
                    if error is not None:
                        skipped.append({"patient_id": patient_id, "reason": error})
                        logger.warning("skip %s: %s", patient_id, error)
                    else:
                        for slug in UPENN_GBM_IMAGE_SEQUENCE_MAP:
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

            # ---- metadata --------------------------------------------------
            for field in UPENN_GBM_METADATA_FIELDS:
                values = _extract_metadata_column(patient_metadata, field)
                dset = w.create_1d(manifest.get(field["path"]), n=n)
                dset[:] = values

            # ---- CSR patient grouping (trivial 1:1 for cross-sectional) ----
            w.write_int_1d(
                "patients/offsets",
                np.arange(n + 1, dtype=np.int32),
                dtype="int32",
                description=(
                    "CSR offsets; scans of patient k are rows [offsets[k]:offsets[k+1]] "
                    "(1:1 for UPENN-GBM cross-sectional)."
                ),
            )
            w.write_vlen_str_1d(
                "patients/keys",
                list(patient_ids),
                description="Unique patient keys (UPENN-GBM-NNNNN_NN) in offset order.",
            )

            # ---- splits ----------------------------------------------------
            self._write_splits(w, splits)

            w.file.attrs["n_scans_written"] = n - len(skipped)
            w.file.attrs["n_patients"] = n
            # Record VAE checkpoint placeholder so consumers can patch on encode.
            # (Image H5 does not depend on the VAE; the latent H5 stamps the actual sha.)

        # Validate before returning; unlink on failure.
        try:
            assert_h5_valid(cfg.output_path, manifest)
        except Exception:
            cfg.output_path.unlink(missing_ok=True)
            raise
        logger.info("Wrote UPENN-GBM H5 cache: %s", cfg.output_path)
        return cfg.output_path

    # ------------------------------------------------------------------ splits

    def _build_splits(self, patient_ids: list[str]) -> NestedCVSplits:
        cfg = self.cfg
        return make_cohort_splits(
            patient_ids,
            n_folds=cfg.n_folds,
            test_fraction=cfg.test_fraction,
            n_test_min=cfg.n_test_min,
            seed=cfg.seed,
            stratify_by=None,  # No grade metadata in the BraTS-21 lookup CSV.
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
