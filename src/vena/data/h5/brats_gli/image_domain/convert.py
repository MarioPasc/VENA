"""NIfTI → H5 converter for the BraTS-GLI image-domain cache.

Streaming layout:

1. Index source sessions via :class:`vena.data.niigz.brats_gli.BraTSGLIDataset`.
2. Optionally limit to the first ``limit`` patients (keeping all their sessions).
3. Compute patient-level splits (nested CV, role="cv"; no stratification —
   no metadata CSV exists for this cohort).
4. Pre-allocate stacked H5 datasets ``(N, 182, 218, 182)`` with
   ``chunks=(1, 182, 218, 182)`` and gzip-4 compression.
5. Dispatch one task per session to a joblib worker pool; each task loads
   the four NIfTI volumes + tumour seg, reorients LAS→LPS, casts dtypes,
   asserts the shape contract, derives the brain mask from t1n foreground,
   and returns a small payload dict.
6. The main process consumes the worker generator and writes each payload
   into its row in H5; h5py is not safe for parallel writes.
7. Write CSR patient grouping (patients/offsets, patients/keys).
8. Write splits (splits/test, splits/cv/fold_k/{train,val}).
9. Validate before returning; unlink on failure.

Intensity policy: no normalisation at write time. Raw NIfTIs cast to float32.
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
from vena.data.h5.shared.brain_mask import clean_brain_mask
from vena.data.niigz.brats_gli import BraTSGLIDataset
from vena.data.niigz.shared.geometry import reorient_to_axcodes
from vena.data.niigz.shared.io import load_nii

from .manifest import (
    BRATS_GLI_IMAGE_EXPECTED_SHAPE,
    BRATS_GLI_IMAGE_MANIFEST,
    BRATS_GLI_IMAGE_SEQUENCE_MAP,
    BRATS_GLI_LABEL_SYSTEM,
)

logger = logging.getLogger(__name__)

_PRODUCER_VERSION = "0.1.0"
_PRODUCER = f"vena.data.h5.brats_gli.image_domain.convert:{_PRODUCER_VERSION}"
_LPS: tuple[str, str, str] = ("L", "P", "S")


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------


class BraTSGLIImageH5Config(BaseModel):
    """Resolved configuration for one execution of the BraTS-GLI converter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_root: Path
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
        description=(
            "Convert only the first ``limit`` patients (smoke runs). All sessions "
            "of each kept patient are included, preserving CSR contiguity."
        ),
    )
    log_level: str = "INFO"

    def to_json(self) -> str:
        return self.model_dump_json()


# ----------------------------------------------------------------------------
# Worker-side helpers
# ----------------------------------------------------------------------------


def _load_lps(path: Path) -> NDArray[Any]:
    """Load a NIfTI file and reorient voxel axes to LPS.

    Parameters
    ----------
    path
        Absolute path to a ``.nii.gz`` file.

    Returns
    -------
    NDArray[Any]
        Array reoriented to LPS axis codes.
    """
    vol = load_nii(path)
    return reorient_to_axcodes(np.asarray(vol.array), vol.affine, _LPS)


def _load_session_payload(
    session_root: Path,
    session_id: str,
    expected_shape: tuple[int, int, int],
    crop_box: tuple[int, int, int],
) -> dict[str, NDArray[Any]]:
    """Load one session's modalities + tumour seg, reoriented to LPS.

    Runs inside a joblib worker; must be picklable. Returns a plain dict so the
    main process can iterate without re-instantiating any project class. Derives
    the brain mask from the nonzero foreground of t1n. Also computes the
    brain-centred crop origin.

    Parameters
    ----------
    session_root
        Directory containing ``{session_id}-{suffix}.nii.gz`` files.
    session_id
        Full session name, e.g. ``BraTS-GLI-00001-000``.
    expected_shape
        Expected voxel shape after reorientation (H, W, D).
    crop_box
        Target crop box (H, W, D) for ``compute_crop_origin``.

    Raises
    ------
    H5ConvertError
        On missing files, shape mismatches, or a brain extent that the common
        crop box cannot contain.
    """
    out: dict[str, NDArray[Any]] = {}
    t1n_lps: NDArray[Any] | None = None

    for slug, suffix in BRATS_GLI_IMAGE_SEQUENCE_MAP.items():
        f = session_root / f"{session_id}-{suffix}.nii.gz"
        if not f.exists():
            raise H5ConvertError(f"{session_id}: missing {f.name}")
        arr = np.ascontiguousarray(_load_lps(f), dtype=np.float32)
        if arr.shape != expected_shape:
            raise H5ConvertError(
                f"{session_id}: {slug} shape {arr.shape} != expected {expected_shape}"
            )
        out[f"images/{slug}"] = arr
        # Keep t1n for brain mask derivation.
        if slug == "t1pre":
            t1n_lps = arr

    # Tumour segmentation — BraTS2023 labels {0, 1, 2, 3}.
    seg_path = session_root / f"{session_id}-seg.nii.gz"
    if not seg_path.exists():
        raise H5ConvertError(f"{session_id}: missing tumour segmentation {seg_path.name}")
    seg = _load_lps(seg_path)
    if seg.shape != expected_shape:
        raise H5ConvertError(f"{session_id}: seg shape {seg.shape} != expected {expected_shape}")
    out["masks/tumor"] = seg.astype(np.int8, copy=False)

    # Brain mask: nonzero foreground of t1n after LPS reorientation, then
    # drop sub-threshold connected components (skull-strip jitter on the
    # first/last z-slices). See `.claude/notes/data/2026-06-18_data_audit.md`.
    assert t1n_lps is not None  # t1pre is always loaded above.
    brain_bin = clean_brain_mask((t1n_lps > 0).astype(np.int8))
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
    """Adapter returning a uniform tuple per task for the main process.

    Parameters
    ----------
    row_index
        Pre-assigned row in the H5 stacked datasets.
    session_id
        Full session identifier.
    session_root
        Directory containing the session's NIfTI files.
    expected_shape
        Expected voxel shape (H, W, D).
    crop_box
        Crop box dimensions (H, W, D).

    Returns
    -------
    tuple
        ``(row_index, session_id, payload_or_None, error_or_None)``.
    """
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


class BraTSGLIImageH5Converter:
    """Run one end-to-end conversion of the BraTS-GLI source tree to H5."""

    def __init__(self, cfg: BraTSGLIImageH5Config) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------ public

    def run(self) -> Path:
        """Execute the conversion and return the output H5 path.

        Returns
        -------
        Path
            Absolute path to the validated H5 file.

        Raises
        ------
        H5ConvertError
            If no sessions are discovered or the produced file fails validation.
        """
        cfg = self.cfg
        dataset = BraTSGLIDataset(cfg.source_root)
        patient_groups = dataset.patient_groups()

        # Apply patient-level limit to preserve CSR contiguity.
        if cfg.limit is not None:
            patient_groups = patient_groups[: cfg.limit]

        if not patient_groups:
            raise H5ConvertError(f"No sessions discovered under {cfg.source_root}")

        patient_ids = [pid for pid, _ in patient_groups]
        # Rebuild session list from the (possibly truncated) patient groups.
        all_sessions = dataset.sessions()
        # Collect sessions that belong to the kept patients in CSR order.
        sessions = [all_sessions[row_idx] for _, indices in patient_groups for row_idx in indices]
        n = len(sessions)
        n_patients = len(patient_ids)
        logger.info(
            "BraTS-GLI image-domain conversion: n_sessions=%d n_patients=%d",
            n,
            n_patients,
        )

        session_ids = [s.session_id for s in sessions]
        splits = self._build_splits(patient_ids)

        timestamp = now_iso_utc()
        git_sha = resolve_git_sha()
        manifest = BRATS_GLI_IMAGE_MANIFEST

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
                "label_system": BRATS_GLI_LABEL_SYSTEM,
                "crop_box": json.dumps(list(cfg.crop_box)),
                "orientation": "LPS",
            },
            overwrite=cfg.overwrite,
        ) as w:
            # ids first so consumers can read IDs without scanning images.
            ids_spec = manifest.get("ids")
            ids_dset = w.create_1d(ids_spec, n=n)
            ids_dset[:] = np.asarray(session_ids, dtype=object)

            image_dsets = {
                slug: w.create_stacked(
                    manifest.get(f"images/{slug}"),
                    n=n,
                    spatial_shape=BRATS_GLI_IMAGE_EXPECTED_SHAPE,
                )
                for slug in BRATS_GLI_IMAGE_SEQUENCE_MAP
            }
            tumor_dset = w.create_stacked(
                manifest.get("masks/tumor"),
                n=n,
                spatial_shape=BRATS_GLI_IMAGE_EXPECTED_SHAPE,
            )
            brain_dset = w.create_stacked(
                manifest.get("masks/brain"),
                n=n,
                spatial_shape=BRATS_GLI_IMAGE_EXPECTED_SHAPE,
            )
            crop_origin_dset = w.create_stacked(
                manifest.get("crop/origin"),
                n=n,
                spatial_shape=(3,),
            )

            # ---- per-session parallel fill (sharded to bound peak RAM) -----
            # Each worker returns a ~130 MB payload; streaming all 1251 sessions
            # through one Parallel buffers them faster than the gzip writes drain
            # and OOMs RAM. Process in shards of ``cfg.shard_size`` sessions: each
            # shard runs a blocking Parallel, results are written and freed before
            # the next shard, bounding peak RAM to ~shard_size payloads.
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
                        BRATS_GLI_IMAGE_EXPECTED_SHAPE,
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
                        for slug in BRATS_GLI_IMAGE_SEQUENCE_MAP:
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

            # ---- CSR patient grouping --------------------------------------
            # Build offsets from group sizes derived from patient_groups.
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
                description="Unique patient keys (BraTS-GLI-PPPPP) in offset order.",
            )

            # ---- splits ----------------------------------------------------
            self._write_splits(w, splits)

            w.file.attrs["n_sessions_written"] = n - len(skipped)
            w.file.attrs["n_patients"] = n_patients

        # Validate before returning; unlink on failure.
        try:
            assert_h5_valid(cfg.output_path, manifest)
        except Exception:
            cfg.output_path.unlink(missing_ok=True)
            raise
        logger.info("Wrote BraTS-GLI H5 cache: %s", cfg.output_path)
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
            stratify_by=None,  # No metadata CSV available for this cohort.
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
