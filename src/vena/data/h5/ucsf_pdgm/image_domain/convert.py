"""NIfTI → H5 converter for the UCSF-PDGM image-domain cache.

Streaming layout:

1. Index source patients via :class:`vena.data.niigz.UCSFPDGMDataset`.
2. Compute splits in patient-ID space (no dependence on stack order).
3. Pre-allocate stacked H5 datasets ``(N, 240, 240, 155)`` with
   ``chunks=(1, 240, 240, 155)`` and gzip-4 compression.
4. Dispatch one task per patient to a joblib worker pool; each task loads
   the five NIfTI volumes (4 sequences + tumour seg), casts them to the
   target dtypes, asserts the shape contract, and returns a small payload.
5. The main process consumes the worker generator in completion order and
   writes each payload into its row in the H5. h5py is not safe for
   parallel writes from worker processes, so all writes happen here.
6. Validate the file against the manifest before returning the output path.

Intensity policy: no normalisation at write time (principle 6 of the H5
design rules). Bias-corrected NIfTIs are cast to ``float32`` as-is.
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
    sha256_file,
)
from vena.data.niigz import UCSFPDGMDataset
from vena.data.niigz.shared.io import load_nii
from vena.data.niigz.shared.geometry import reorient_to_axcodes

from .manifest import (
    UCSF_PDGM_IMAGE_EXPECTED_SHAPE,
    UCSF_PDGM_IMAGE_MANIFEST,
    UCSF_PDGM_IMAGE_SEQUENCE_MAP,
    UCSF_PDGM_LABEL_SYSTEM,
    UCSF_PDGM_METADATA_FIELDS,
    MetadataFieldSpec,
)

logger = logging.getLogger(__name__)

_PRODUCER_VERSION = "0.2.0"
_PRODUCER = f"vena.data.h5.ucsf_pdgm.image_domain.convert:{_PRODUCER_VERSION}"
_LPS: tuple[str, str, str] = ("L", "P", "S")


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------


class UCSFPDGMImageH5Config(BaseModel):
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
    stratify_by: str | None = "WHO CNS Grade"

    overwrite: bool = False
    limit: int | None = Field(
        default=None,
        description="Optional: convert only the first ``limit`` patients (smoke runs).",
    )
    log_level: str = "INFO"

    def to_json(self) -> str:
        return self.model_dump_json()


# ----------------------------------------------------------------------------
# Worker-side payload
# ----------------------------------------------------------------------------


def _load_lps(path: Path) -> NDArray[Any]:
    """Load a NIfTI and reorient its voxel axes to LPS (identity for UCSF)."""
    vol = load_nii(path)
    return reorient_to_axcodes(np.asarray(vol.array), vol.affine, _LPS)


def _load_patient_payload(
    patient_root: Path,
    patient_id: str,
    expected_shape: tuple[int, int, int],
    crop_box: tuple[int, int, int],
) -> dict[str, NDArray[Any]]:
    """Load one patient's modalities + tumour/brain masks, reoriented to LPS.

    Runs inside a joblib worker; must be picklable. Returns a plain dict so the
    main process can iterate without re-instantiating any project class. Also
    computes the brain-centred crop origin from the brain mask.

    Raises
    ------
    H5ConvertError
        On missing files, shape mismatches, or a brain extent that the common
        crop box cannot contain. Captured by the main process and recorded in
        the skip list rather than aborting the whole conversion.
    """
    out: dict[str, NDArray[Any]] = {}
    for slug, suffix in UCSF_PDGM_IMAGE_SEQUENCE_MAP.items():
        f = patient_root / f"{patient_id}_{suffix}.nii.gz"
        if not f.exists():
            raise H5ConvertError(f"{patient_id}: missing {f.name}")
        arr = np.ascontiguousarray(_load_lps(f), dtype=np.float32)
        if arr.shape != expected_shape:
            raise H5ConvertError(
                f"{patient_id}: {slug} shape {arr.shape} != expected {expected_shape}"
            )
        out[f"images/{slug}"] = arr

    seg_path = patient_root / f"{patient_id}_tumor_segmentation.nii.gz"
    if not seg_path.exists():
        raise H5ConvertError(f"{patient_id}: missing tumor segmentation {seg_path.name}")
    seg = _load_lps(seg_path)
    if seg.shape != expected_shape:
        raise H5ConvertError(
            f"{patient_id}: tumor seg shape {seg.shape} != expected {expected_shape}"
        )
    # BraTS labels live in {0, 1, 2, 4} — fits comfortably in int8.
    out["masks/tumor"] = seg.astype(np.int8, copy=False)

    brain_path = patient_root / f"{patient_id}_brain_segmentation.nii.gz"
    if not brain_path.exists():
        raise H5ConvertError(f"{patient_id}: missing brain segmentation {brain_path.name}")
    brain = _load_lps(brain_path)
    if brain.shape != expected_shape:
        raise H5ConvertError(
            f"{patient_id}: brain seg shape {brain.shape} != expected {expected_shape}"
        )
    brain_bin = (brain > 0.5).astype(np.int8)
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
        payload = _load_patient_payload(patient_root, patient_id, expected_shape, crop_box)
        return (row_index, patient_id, payload, None)
    except H5ConvertError as exc:
        return (row_index, patient_id, None, str(exc))
    except Exception as exc:
        return (row_index, patient_id, None, f"unexpected: {exc!r}")


# ----------------------------------------------------------------------------
# Metadata helpers
# ----------------------------------------------------------------------------


def _cast_metadata(value: Any, cast: str) -> Any:
    """Cast a CSV cell to the type declared in the manifest.

    NaN policy:
      * ``str``  → ``""``
      * ``int``  → ``-1``
      * ``float`` → ``NaN`` (preserved)
    """
    if isinstance(value, float) and np.isnan(value):
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
    metadata: dict[str, dict[str, Any]],
    patient_ids: list[str],
    field: MetadataFieldSpec,
) -> NDArray[Any]:
    col = field["csv_column"]
    cast = field["cast"]
    values = [
        _cast_metadata(metadata.get(pid, {}).get(col, float("nan")), cast) for pid in patient_ids
    ]
    if cast == "str":
        return np.asarray(values, dtype=object)
    if cast == "int":
        return np.asarray(values, dtype=np.int8)
    if cast == "float":
        return np.asarray(values, dtype=np.float32)
    raise ValueError(f"unknown cast: {cast!r}")


# ----------------------------------------------------------------------------
# Converter
# ----------------------------------------------------------------------------


class UCSFPDGMImageH5Converter:
    """Run one end-to-end conversion of the UCSF-PDGM source tree to H5."""

    def __init__(self, cfg: UCSFPDGMImageH5Config) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------ public

    def run(self) -> Path:
        cfg = self.cfg
        dataset = UCSFPDGMDataset(cfg.source_root, cfg.metadata_csv)
        patients = list(dataset)
        if cfg.limit is not None:
            patients = patients[: cfg.limit]
        if not patients:
            raise H5ConvertError(f"No patients discovered under {cfg.source_root}")
        patient_ids = [p.patient_id for p in patients]
        n = len(patient_ids)
        logger.info("UCSF-PDGM image-domain conversion: n_patients=%d", n)

        # ---- splits ---------------------------------------------------------
        splits = self._build_splits(dataset, patient_ids)

        # ---- write header & allocate datasets ------------------------------
        timestamp = now_iso_utc()
        git_sha = resolve_git_sha()
        manifest = UCSF_PDGM_IMAGE_MANIFEST

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
                "label_system": UCSF_PDGM_LABEL_SYSTEM,
                "crop_box": json.dumps(list(cfg.crop_box)),
                "orientation": "LPS",
            },
            overwrite=cfg.overwrite,
        ) as w:
            # ids first so consumers can read it without scanning images.
            ids_spec = manifest.get("ids")
            ids_dset = w.create_1d(ids_spec, n=n)
            ids_dset[:] = np.asarray(patient_ids, dtype=object)

            image_dsets = {
                slug: w.create_stacked(
                    manifest.get(f"images/{slug}"),
                    n=n,
                    spatial_shape=UCSF_PDGM_IMAGE_EXPECTED_SHAPE,
                )
                for slug in UCSF_PDGM_IMAGE_SEQUENCE_MAP
            }
            tumor_dset = w.create_stacked(
                manifest.get("masks/tumor"),
                n=n,
                spatial_shape=UCSF_PDGM_IMAGE_EXPECTED_SHAPE,
            )
            brain_dset = w.create_stacked(
                manifest.get("masks/brain"),
                n=n,
                spatial_shape=UCSF_PDGM_IMAGE_EXPECTED_SHAPE,
            )
            crop_origin_dset = w.create_stacked(
                manifest.get("crop/origin"),
                n=n,
                spatial_shape=(3,),
            )

            # ---- per-patient parallel fill (sharded to bound peak RAM) -----
            # Each worker returns a ~160 MB payload (four float32 volumes + two
            # masks). Streaming every patient through one Parallel buffers the
            # results faster than the gzip writes drain them and OOMs RAM. We
            # process the cohort in shards of ``cfg.shard_size`` patients: each
            # shard runs a blocking Parallel, its results are written and freed
            # before the next shard starts, so peak RAM is bounded by
            # ~shard_size payloads regardless of write throughput.
            skipped: list[dict[str, str]] = []
            log_every = max(1, n // 50)  # ~50 log lines over the full run.
            t0 = time.monotonic()
            done = 0
            for shard_start in range(0, n, cfg.shard_size):
                shard = patients[shard_start : shard_start + cfg.shard_size]
                shard_tasks = [
                    delayed(_worker)(
                        shard_start + j,
                        p.patient_id,
                        p.root,
                        UCSF_PDGM_IMAGE_EXPECTED_SHAPE,
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
                        for slug in UCSF_PDGM_IMAGE_SEQUENCE_MAP:
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
                # Drop shard payloads and persist before the next shard.
                del results, shard_tasks
                w.file.flush()

            if skipped:
                logger.warning("Skipped %d patient(s). See attrs/skipped_json.", len(skipped))
                w.file.attrs["skipped_json"] = json.dumps(skipped)
                # If anyone was skipped we leave the all-zero rows in place rather
                # than rewriting the whole file. Consumers can mask via
                # ``np.setdiff1d(ids, [s["patient_id"] for s in skipped])``.

            # ---- metadata --------------------------------------------------
            metadata = dataset._metadata
            for field in UCSF_PDGM_METADATA_FIELDS:
                values = _extract_metadata_column(metadata, patient_ids, field)
                dset = w.create_1d(manifest.get(field["path"]), n=n)
                dset[:] = values

            # ---- CSR patient grouping (trivial 1:1 for cross-sectional UCSF) -
            w.write_int_1d(
                "patients/offsets",
                np.arange(n + 1, dtype=np.int32),
                dtype="int32",
                description="CSR offsets; scans of patient k are rows [offsets[k]:offsets[k+1]].",
            )
            w.write_vlen_str_1d(
                "patients/keys",
                list(patient_ids),
                description="Unique patient keys in offset order (1:1 with scans for UCSF).",
            )

            # ---- splits ----------------------------------------------------
            self._write_splits(w, splits)

            # Provenance: SHA-256 of the metadata CSV, recorded so a future
            # consumer can detect a regenerated source CSV.
            w.file.attrs["metadata_csv_sha256"] = sha256_file(cfg.metadata_csv)
            w.file.attrs["n_patients_written"] = n - len(skipped)

        # Validate before returning. Unlink on failure so a non-conformant
        # artifact never lingers on disk (principle 7).
        try:
            assert_h5_valid(cfg.output_path, manifest)
        except Exception:
            cfg.output_path.unlink(missing_ok=True)
            raise
        logger.info("Wrote H5 cache: %s", cfg.output_path)
        return cfg.output_path

    # ------------------------------------------------------------------ splits

    def _build_splits(
        self,
        dataset: UCSFPDGMDataset,
        patient_ids: list[str],
    ) -> NestedCVSplits:
        cfg = self.cfg
        stratify: list[int] | None = None
        if cfg.stratify_by is not None:
            col = cfg.stratify_by
            try:
                stratify = [int(dataset._metadata.get(pid, {}).get(col, -1)) for pid in patient_ids]
            except (TypeError, ValueError):
                logger.warning(
                    "stratify column %r is not coercible to int; falling back to random splits",
                    col,
                )
                stratify = None
        return make_cohort_splits(
            patient_ids,
            n_folds=cfg.n_folds,
            test_fraction=cfg.test_fraction,
            n_test_min=cfg.n_test_min,
            seed=cfg.seed,
            stratify_by=stratify,
            role="cv",
        )

    def _write_splits(self, w: H5Writer, splits: NestedCVSplits) -> None:
        w.write_vlen_str_1d("splits/test", splits["test"])
        for fold_idx, fold in splits["folds"].items():
            w.write_vlen_str_1d(f"splits/cv/fold_{fold_idx}/train", fold["train"])
            w.write_vlen_str_1d(f"splits/cv/fold_{fold_idx}/val", fold["val"])
        # Document the splits layout once at the splits group level.
        grp = w.file["splits"]
        grp.attrs["description"] = (
            "Patient-ID-based nested CV splits. splits/test is the held-out "
            "set shared across folds; splits/cv/fold_K/{train,val} are the "
            "per-fold CV partitions of the remaining patients."
        )
        grp.attrs["n_folds"] = len(splits["folds"])
