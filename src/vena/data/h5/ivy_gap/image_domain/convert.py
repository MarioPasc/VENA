"""NIfTI → H5 converter for the IvyGAP image-domain cache.

Streaming layout:

1. Index source patients via :class:`vena.data.niigz.IvyGAPDataset` — the
   reader handles registration-variant precedence and UPenn segmentation
   resolution.
2. Build a deterministic single 24/5/5 train/val/test split (N=34 is too
   small for stable nested K-fold CV; see ``configs/default.yaml``).
3. Pre-allocate stacked H5 datasets ``(N, 240, 240, 155)`` with
   ``chunks=(1, 240, 240, 155)`` and gzip-4 compression.
4. Dispatch one task per patient to a joblib worker pool; each task loads
   four NIfTI volumes (resolved by the reader) plus the UPenn tumour seg,
   reorients to LPS (identity for SRI24), derives the brain mask as the
   nonzero foreground of t1pre, and computes the brain-centred crop origin.
5. The main process writes each payload into its row; ``h5py`` is not safe
   for parallel writes from worker processes.
6. Validate the file against the manifest before returning.

Intensity policy: no normalisation at write time. Raw float32 volumes.
"""

from __future__ import annotations

import json
import logging
import math
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
from vena.data.niigz.ivy_gap import IvyGAPDataset, IvyGAPPatient
from vena.data.niigz.shared.geometry import reorient_to_axcodes
from vena.data.niigz.shared.io import load_nii

from .manifest import (
    IVY_GAP_IMAGE_EXPECTED_SHAPE,
    IVY_GAP_IMAGE_MANIFEST,
    IVY_GAP_IMAGE_SEQUENCE_MAP,
    IVY_GAP_LABEL_SYSTEM,
)

logger = logging.getLogger(__name__)

_PRODUCER_VERSION = "0.1.0"
_PRODUCER = f"vena.data.h5.ivy_gap.image_domain.convert:{_PRODUCER_VERSION}"
_LPS: tuple[str, str, str] = ("L", "P", "S")


class _Splits(TypedDict):
    train: list[str]
    val: list[str]
    test: list[str]


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------


class IvyGAPImageH5Config(BaseModel):
    """Resolved configuration for one execution of the IvyGAP converter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_root: Path
    output_path: Path

    n_jobs: int = 8
    shard_size: int = 16
    crop_box: tuple[int, int, int] = (192, 224, 192)
    n_val: int = 5
    n_test: int = 5
    seed: int = 42
    tumor_seg_source: str = "upenn"

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
    modality_paths: dict[str, str],
    upenn_seg_path: str,
    expected_shape: tuple[int, int, int],
    crop_box: tuple[int, int, int],
) -> dict[str, NDArray[Any]]:
    """Load one patient's modalities + tumour seg + brain mask.

    Runs inside a joblib worker; must be picklable. Paths come pre-resolved
    from the reader so the worker does no directory scanning.

    Raises
    ------
    H5ConvertError
        On missing files, shape mismatches, or a brain extent that the common
        crop box cannot contain.
    """
    out: dict[str, NDArray[Any]] = {}
    t1pre_lps: NDArray[Any] | None = None

    for slug in IVY_GAP_IMAGE_SEQUENCE_MAP:
        path = Path(modality_paths[slug])
        if not path.exists():
            raise H5ConvertError(f"{patient_id}: missing {path.name}")
        arr = np.ascontiguousarray(_load_lps(path), dtype=np.float32)
        if arr.shape != expected_shape:
            raise H5ConvertError(
                f"{patient_id}: {slug} shape {arr.shape} != expected {expected_shape}"
            )
        out[f"images/{slug}"] = arr
        if slug == "t1pre":
            t1pre_lps = arr

    seg_path = Path(upenn_seg_path)
    if not seg_path.exists():
        raise H5ConvertError(f"{patient_id}: missing UPenn seg {seg_path.name}")
    seg = _load_lps(seg_path)
    if seg.shape != expected_shape:
        raise H5ConvertError(
            f"{patient_id}: UPenn seg shape {seg.shape} != expected {expected_shape}"
        )
    out["masks/tumor"] = seg.astype(np.int8, copy=False)

    # Brain mask: t1pre nonzero foreground, then drop sub-threshold CCs
    # (cloud-like blurs at axial slice extremes). See
    # `.claude/notes/data/2026-06-18_data_audit.md` — IvyGAP samples carry
    # 35-148 spurious CCs before cleaning.
    assert t1pre_lps is not None  # t1pre is loaded above.
    brain_bin = clean_brain_mask((t1pre_lps > 0).astype(np.int8))
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
    modality_paths: dict[str, str],
    upenn_seg_path: str,
    expected_shape: tuple[int, int, int],
    crop_box: tuple[int, int, int],
) -> tuple[int, str, dict[str, NDArray[Any]] | None, str | None]:
    """Adapter so the main loop receives a uniform tuple per task."""
    try:
        payload = _load_patient_payload(
            patient_id, modality_paths, upenn_seg_path, expected_shape, crop_box
        )
        return (row_index, patient_id, payload, None)
    except H5ConvertError as exc:
        return (row_index, patient_id, None, str(exc))
    except Exception as exc:
        return (row_index, patient_id, None, f"unexpected: {exc!r}")


# ----------------------------------------------------------------------------
# Converter
# ----------------------------------------------------------------------------


class IvyGAPImageH5Converter:
    """Run one end-to-end conversion of the IvyGAP source tree to H5."""

    def __init__(self, cfg: IvyGAPImageH5Config) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------ public

    def run(self) -> Path:
        cfg = self.cfg
        dataset = IvyGAPDataset(cfg.source_root, tumor_seg_source="upenn")
        patients = list(dataset)
        if cfg.limit is not None:
            patients = patients[: cfg.limit]
        if not patients:
            raise H5ConvertError(f"No patients discovered under {cfg.source_root}")
        patient_ids = [p.patient_id for p in patients]
        n = len(patient_ids)
        logger.info("IvyGAP image-domain conversion: n_patients=%d", n)

        splits = self._build_splits(patient_ids)

        timestamp = now_iso_utc()
        git_sha = resolve_git_sha()
        manifest = IVY_GAP_IMAGE_MANIFEST

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
                "label_system": IVY_GAP_LABEL_SYSTEM,
                "crop_box": json.dumps(list(cfg.crop_box)),
                "orientation": "LPS",
            },
            overwrite=cfg.overwrite,
        ) as w:
            ids_dset = w.create_1d(manifest.get("ids"), n=n)
            ids_dset[:] = np.asarray(patient_ids, dtype=object)

            image_dsets = {
                slug: w.create_stacked(
                    manifest.get(f"images/{slug}"),
                    n=n,
                    spatial_shape=IVY_GAP_IMAGE_EXPECTED_SHAPE,
                )
                for slug in IVY_GAP_IMAGE_SEQUENCE_MAP
            }
            tumor_dset = w.create_stacked(
                manifest.get("masks/tumor"),
                n=n,
                spatial_shape=IVY_GAP_IMAGE_EXPECTED_SHAPE,
            )
            brain_dset = w.create_stacked(
                manifest.get("masks/brain"),
                n=n,
                spatial_shape=IVY_GAP_IMAGE_EXPECTED_SHAPE,
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
                        {
                            slug: str(p.root / p.metadata[f"source_basename_{slug}"])
                            for slug in IVY_GAP_IMAGE_SEQUENCE_MAP
                        },
                        str(p.metadata["upenn_seg_path"]),
                        IVY_GAP_IMAGE_EXPECTED_SHAPE,
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
                        for slug in IVY_GAP_IMAGE_SEQUENCE_MAP:
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
            self._write_metadata(w, patients, n)

            # ---- CSR patient grouping (trivial 1:1) ------------------------
            w.write_int_1d(
                "patients/offsets",
                np.arange(n + 1, dtype=np.int32),
                dtype="int32",
                description=(
                    "CSR offsets; scans of patient k are rows [offsets[k]:offsets[k+1]] "
                    "(1:1 for IvyGAP cross-sectional)."
                ),
            )
            w.write_vlen_str_1d(
                "patients/keys",
                list(patient_ids),
                description="Unique patient keys (W<N>) in offset order.",
            )

            # ---- splits ----------------------------------------------------
            self._write_splits(w, splits)

            w.file.attrs["n_patients_written"] = n - len(skipped)

        try:
            assert_h5_valid(cfg.output_path, manifest)
        except Exception:
            cfg.output_path.unlink(missing_ok=True)
            raise
        logger.info("Wrote IvyGAP H5 cache: %s", cfg.output_path)
        return cfg.output_path

    # ------------------------------------------------------------------ helpers

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
            "IvyGAP splits: train=%d val=%d test=%d (seed=%d)",
            len(splits["train"]),
            len(splits["val"]),
            len(splits["test"]),
            cfg.seed,
        )
        return splits

    def _write_splits(self, w: H5Writer, splits: _Splits) -> None:
        w.write_vlen_str_1d("splits/train", splits["train"])
        w.write_vlen_str_1d("splits/val", splits["val"])
        w.write_vlen_str_1d("splits/test", splits["test"])
        grp = w.file["splits"]
        grp.attrs["description"] = (
            "Single random patient-ID split into train/val/test. The cohort "
            "(N=34) is too small for stable nested K-fold CV."
        )
        grp.attrs["n_folds"] = 1

    def _write_metadata(self, w: H5Writer, patients: list[IvyGAPPatient], n: int) -> None:
        manifest = IVY_GAP_IMAGE_MANIFEST

        def _str_col(field: str) -> NDArray[Any]:
            return np.asarray([str(p.metadata.get(field, "")) for p in patients], dtype=object)

        for field in (
            "scan_date",
            "tumor_seg_source",
            "source_basename_t1pre",
            "source_basename_t1c",
            "source_basename_t2",
            "source_basename_flair",
            "cwru_seg_path",
        ):
            dset = w.create_1d(manifest.get(f"metadata/{field}"), n=n)
            dset[:] = _str_col(field)

        # Sanity: not enforced in the manifest, but the canonical floor that
        # surfaces a misconfigured split before launching a long encode job.
        if n < 3 and math.inf > 0:  # pragma: no cover — diagnostic guard
            logger.warning("IvyGAP: only %d patient(s) — splits will be degenerate.", n)
