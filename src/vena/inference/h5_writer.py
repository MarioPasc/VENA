"""Predictions-H5 writer + validator (per validation §5.3 schema).

The validation protocol §5.3 prescribes one HDF5 per (method × ring)
holding the harmonised + raw predicted T1c, the harmonised reference
modalities, brain/WT masks, per-volume residuals, and a metadata block.
We persist at finer granularity — per (method × cohort × NFE) — and
pool to ring-level downstream; the *contents* of each file match §5.3
verbatim and the validator enforces the cross-field invariants.

Storage policy (per ``.claude/rules/h5-design-principles.md``):

* compression ``gzip`` level 4 on every bulky dataset
* chunking ``(1, H, W, D)`` so reading one scan is one read
* float32 intensities, int8 masks, vlen-str IDs
* every dataset carries ``units`` and ``description`` attrs
* root carries ``schema_version``, ``created_at``, ``producer``,
  ``config_json``, ``git_sha`` so the file is self-describing.

The validator returns the list of violations (empty when valid); the
producer must call :func:`assert_predictions_valid` before returning a
path from ``Engine.run()``.
"""

from __future__ import annotations

import datetime as _dt  # Py3.10 compat — datetime.UTC is 3.11+; use _dt.datetime.utcnow()
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from vena.inference.harmonisation import HARMONISATION_RECIPE

# Anchor the _dt usage at module scope so ruff autoflake never strips the import.
_DATETIME_MOD = _dt

SCHEMA_VERSION = "1.0"
PRODUCER = "routines.fm.inference:v0.1.0"

_RTOL_RESIDUAL = 1e-5
_ATOL_RESIDUAL = 1e-4


class PredictionsH5Error(Exception):
    """Raised on a malformed predictions H5 (write- or read-side)."""


@dataclass(frozen=True)
class PerPatientRecord:
    """One row of the predictions H5 (one patient × one NFE)."""

    patient_id: str
    cohort: str
    t1c_synthetic_harmonised: np.ndarray  # (H, W, D) float32 in [0, 1]
    t1c_synthetic_raw: np.ndarray  # (H, W, D) float32
    t1c_real_harmonised: np.ndarray  # (H, W, D) float32 in [0, 1]
    t1pre_harmonised: np.ndarray  # (H, W, D) float32 in [0, 1]
    t2_harmonised: np.ndarray  # (H, W, D) float32 in [0, 1]
    flair_harmonised: np.ndarray  # (H, W, D) float32 in [0, 1]
    brain_mask: np.ndarray  # (H, W, D) int8 {0, 1}
    wt_mask: np.ndarray  # (H, W, D) int8 {0, 1}
    inference_seconds: float
    peak_vram_mb: float

    def shape(self) -> tuple[int, int, int]:
        return tuple(int(d) for d in self.t1c_synthetic_harmonised.shape)  # type: ignore[return-value]


def _as_np(t: torch.Tensor | np.ndarray, dtype: type[np.generic]) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        arr = t.detach().cpu().numpy()
    else:
        arr = np.asarray(t)
    return np.ascontiguousarray(arr).astype(dtype, copy=False)


def _vlen_str_dataset(grp: h5py.Group, name: str, values: list[str]) -> None:
    dt = h5py.string_dtype(encoding="utf-8")
    arr = np.asarray(values, dtype=object)
    grp.create_dataset(name, data=arr, dtype=dt)


def write_predictions_h5(
    path: Path | str,
    records: list[PerPatientRecord],
    *,
    method: str,
    cohort: str,
    nfe: int,
    ring: str,
    git_sha: str | None = None,
    checkpoint_path: Path | str | None = None,
    checkpoint_sha256: str | None = None,
    vae_checkpoint_sha256: str | None = None,
    run_id_tag: str | None = None,
    extra_config: dict[str, object] | None = None,
) -> Path:
    """Write one (method × cohort × NFE) predictions H5 and return its path.

    Parameters
    ----------
    path
        Destination file. Parent directories are created.
    records
        One :class:`PerPatientRecord` per scan. The first record's shape
        sets ``(H, W, D)``; every other record must match.
    method, cohort, nfe, ring
        Provenance written to root attrs.
    git_sha, checkpoint_path, checkpoint_sha256, vae_checkpoint_sha256, run_id_tag
        Optional provenance.
    extra_config
        Free-form JSON-serialisable dict, persisted as ``config_json``
        per ``h5-design-principles.md`` rule 3.

    Raises
    ------
    PredictionsH5Error
        If the records are empty, shape-inconsistent, or contain NaN/Inf.
    """
    if not records:
        raise PredictionsH5Error(
            f"write_predictions_h5: no records provided for "
            f"method={method!r} cohort={cohort!r} nfe={nfe}"
        )

    shape_ref = records[0].shape()
    for r in records[1:]:
        if r.shape() != shape_ref:
            raise PredictionsH5Error(
                f"shape mismatch across records: {records[0].patient_id} "
                f"has {shape_ref}, {r.patient_id} has {r.shape()}"
            )

    n = len(records)
    h, w, d = shape_ref
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _vol_dset(
        grp: h5py.Group, key: str, np_dtype: type[np.generic], description: str, units: str
    ) -> h5py.Dataset:
        ds = grp.create_dataset(
            key,
            shape=(n, h, w, d),
            dtype=np_dtype,
            chunks=(1, h, w, d),
            compression="gzip",
            compression_opts=4,
        )
        ds.attrs["description"] = description
        ds.attrs["units"] = units
        ds.attrs["dtype"] = np.dtype(np_dtype).name
        ds.attrs["leading_dim"] = "n_scans"
        return ds

    with h5py.File(path, "w") as f:
        # ---- root attrs (h5-design-principles rules 1-3) ----
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["created_at"] = _dt.datetime.utcnow().isoformat() + "Z"
        f.attrs["producer"] = PRODUCER
        f.attrs["method"] = method
        f.attrs["cohort"] = cohort
        f.attrs["nfe"] = int(nfe)
        f.attrs["ring"] = ring
        f.attrs["harmonisation_recipe"] = HARMONISATION_RECIPE
        if git_sha:
            f.attrs["git_sha"] = git_sha
        if checkpoint_path is not None:
            f.attrs["checkpoint_path"] = str(checkpoint_path)
        if checkpoint_sha256:
            f.attrs["checkpoint_sha256"] = checkpoint_sha256
        if vae_checkpoint_sha256:
            f.attrs["vae_checkpoint_sha256"] = vae_checkpoint_sha256
        if run_id_tag:
            f.attrs["run_id_tag"] = run_id_tag
        f.attrs["config_json"] = json.dumps(extra_config or {}, sort_keys=True)

        # ---- volumetric datasets ----
        g_pred = f.create_group("predictions")
        g_pred.attrs["description"] = "Predicted T1c volumes (harmonised + raw)."
        ds_synth = _vol_dset(
            g_pred,
            "t1c_synthetic_harmonised",
            np.float32,
            "Predicted T1c after §4.1 percentile harmonisation, range [0, 1] inside brain mask.",
            "dimensionless",
        )
        ds_raw = _vol_dset(
            g_pred,
            "t1c_synthetic_raw",
            np.float32,
            "Method-native predicted T1c before §4.1 harmonisation (audit only).",
            "dimensionless",
        )

        g_ref = f.create_group("reference")
        g_ref.attrs["description"] = "Per-scan reference modalities (harmonised)."
        ds_t1c = _vol_dset(
            g_ref,
            "t1c_real_harmonised",
            np.float32,
            "Real T1c after §4.1 harmonisation.",
            "dimensionless",
        )
        ds_t1pre = _vol_dset(
            g_ref,
            "t1pre_harmonised",
            np.float32,
            "Real T1pre after §4.1 harmonisation.",
            "dimensionless",
        )
        ds_t2 = _vol_dset(
            g_ref, "t2_harmonised", np.float32, "Real T2 after §4.1 harmonisation.", "dimensionless"
        )
        ds_flair = _vol_dset(
            g_ref,
            "flair_harmonised",
            np.float32,
            "Real FLAIR after §4.1 harmonisation.",
            "dimensionless",
        )

        g_msk = f.create_group("masks")
        g_msk.attrs["description"] = "Binary brain and whole-tumour masks."
        ds_brain = _vol_dset(
            g_msk, "brain", np.int8, "Brain mask (HD-BET / CBICA), binary {0, 1}.", "binary"
        )
        ds_wt = _vol_dset(
            g_msk, "wt", np.int8, "Whole-tumour mask, derived as (masks/tumor > 0).", "binary"
        )

        g_res = f.create_group("residuals")
        g_res.attrs["description"] = "Per-scan residual maps (real - synth, harmonised)."
        ds_resid = _vol_dset(
            g_res,
            "raw",
            np.float32,
            "t1c_real_harmonised - t1c_synthetic_harmonised.",
            "dimensionless",
        )

        # ---- metadata datasets ----
        g_meta = f.create_group("metadata")
        g_meta.attrs["description"] = "Per-scan provenance and per-volume timing."
        _vlen_str_dataset(g_meta, "patient_id", [r.patient_id for r in records])
        _vlen_str_dataset(g_meta, "cohort", [r.cohort for r in records])
        g_meta.create_dataset(
            "inference_seconds",
            data=np.asarray([r.inference_seconds for r in records], dtype=np.float32),
        )
        g_meta["inference_seconds"].attrs["units"] = "s"
        g_meta["inference_seconds"].attrs["description"] = (
            "CUDA-synced wall-clock for the full predict() body per validation §5.2."
        )
        g_meta.create_dataset(
            "peak_vram_mb",
            data=np.asarray([r.peak_vram_mb for r in records], dtype=np.float32),
        )
        g_meta["peak_vram_mb"].attrs["units"] = "MB"
        g_meta["peak_vram_mb"].attrs["description"] = (
            "torch.cuda.max_memory_allocated read at the end of each predict() call."
        )
        g_meta.create_dataset(
            "nfe",
            data=np.full((n,), int(nfe), dtype=np.int32),
        )
        g_meta["nfe"].attrs["units"] = "count"
        g_meta["nfe"].attrs["description"] = (
            "Number of function evaluations; constant per file, kept per-row "
            "for validation §5.3 compatibility when pooled across NFEs."
        )
        scan_shape = np.tile(np.asarray(shape_ref, dtype=np.int32), (n, 1))
        g_meta.create_dataset("scan_shape", data=scan_shape)
        g_meta["scan_shape"].attrs["units"] = "voxels"
        g_meta["scan_shape"].attrs["description"] = "Volume shape (H, W, D) per scan."

        # ---- fill volumetric datasets, computing residuals in-line ----
        for i, r in enumerate(records):
            synth = _as_np(r.t1c_synthetic_harmonised, np.float32)
            raw = _as_np(r.t1c_synthetic_raw, np.float32)
            t1c_real = _as_np(r.t1c_real_harmonised, np.float32)
            t1pre = _as_np(r.t1pre_harmonised, np.float32)
            t2 = _as_np(r.t2_harmonised, np.float32)
            flair = _as_np(r.flair_harmonised, np.float32)
            brain = _as_np(r.brain_mask, np.int8)
            wt = _as_np(r.wt_mask, np.int8)

            ds_synth[i] = synth
            ds_raw[i] = raw
            ds_t1c[i] = t1c_real
            ds_t1pre[i] = t1pre
            ds_t2[i] = t2
            ds_flair[i] = flair
            ds_brain[i] = brain
            ds_wt[i] = wt
            ds_resid[i] = t1c_real - synth

    return path


# ---------------------------------------------------------------------------- validator


def validate_predictions(path: Path | str) -> list[str]:
    """Return a list of cross-field validation violations.

    The list is empty when the H5 conforms to validation §5.3 +
    ``h5-design-principles.md``.
    """
    violations: list[str] = []
    path = Path(path)
    if not path.is_file():
        return [f"file not found: {path}"]

    with h5py.File(path, "r") as f:
        # 1. schema version present
        sv = f.attrs.get("schema_version")
        if sv != SCHEMA_VERSION:
            violations.append(f"schema_version={sv!r} (expected {SCHEMA_VERSION!r})")

        # 2. mandatory datasets
        required = [
            "predictions/t1c_synthetic_harmonised",
            "predictions/t1c_synthetic_raw",
            "reference/t1c_real_harmonised",
            "reference/t1pre_harmonised",
            "reference/t2_harmonised",
            "reference/flair_harmonised",
            "masks/brain",
            "masks/wt",
            "residuals/raw",
            "metadata/patient_id",
            "metadata/cohort",
            "metadata/inference_seconds",
            "metadata/peak_vram_mb",
            "metadata/nfe",
            "metadata/scan_shape",
        ]
        for key in required:
            if key not in f:
                violations.append(f"missing dataset: {key}")
        if violations:
            return violations

        synth = f["predictions/t1c_synthetic_harmonised"]
        real = f["reference/t1c_real_harmonised"]
        brain = f["masks/brain"]
        resid = f["residuals/raw"]

        # 3. shape match across volumetric datasets
        if synth.shape != real.shape:
            violations.append(
                f"predictions/t1c_synthetic_harmonised shape {synth.shape} "
                f"!= reference/t1c_real_harmonised shape {real.shape}"
            )
        if synth.shape != brain.shape:
            violations.append(
                f"predictions/t1c_synthetic_harmonised shape {synth.shape} "
                f"!= masks/brain shape {brain.shape}"
            )

        # 4. patient_id uniqueness + length consistency
        n = synth.shape[0]
        pid_raw = f["metadata/patient_id"][:]
        pids = [b.decode() if isinstance(b, bytes) else str(b) for b in pid_raw]
        if len(pids) != n:
            violations.append(f"metadata/patient_id has {len(pids)} entries, expected {n}")
        if len(set(pids)) != len(pids):
            violations.append("metadata/patient_id has duplicates")

        # 5. per-scan numeric checks — cheap; iterate at most n scans
        for i in range(n):
            s = synth[i]
            r_i = real[i]
            b = brain[i].astype(bool)
            if not np.all(np.isfinite(s)):
                violations.append(f"row {i} (pid={pids[i]}): NaN/Inf in t1c_synthetic_harmonised")
                continue
            if not np.all(np.isfinite(r_i)):
                violations.append(f"row {i} (pid={pids[i]}): NaN/Inf in t1c_real_harmonised")
                continue
            # range [0, 1] inside brain mask, exterior == 0
            if b.any():
                inside = s[b]
                if (inside < -1e-6).any() or (inside > 1.0 + 1e-6).any():
                    violations.append(
                        f"row {i} (pid={pids[i]}): synth out of [0, 1] inside brain mask "
                        f"(min={float(inside.min()):.4f} max={float(inside.max()):.4f})"
                    )
                outside = s[~b]
                if outside.size and float(np.max(np.abs(outside))) > 1e-6:
                    violations.append(
                        f"row {i} (pid={pids[i]}): synth nonzero outside brain mask "
                        f"(max|outside|={float(np.max(np.abs(outside))):.4f})"
                    )
            # residual identity
            expected = r_i - s
            got = resid[i]
            if not np.allclose(expected, got, rtol=_RTOL_RESIDUAL, atol=_ATOL_RESIDUAL):
                violations.append(
                    f"row {i} (pid={pids[i]}): residuals/raw != real - synth "
                    f"(max|diff|={float(np.max(np.abs(expected - got))):.4f})"
                )

    return violations


def assert_predictions_valid(path: Path | str) -> None:
    """Raise :class:`PredictionsH5Error` if any §5.3 invariant fails."""
    violations = validate_predictions(path)
    if violations:
        joined = "\n  - ".join(violations)
        raise PredictionsH5Error(f"predictions H5 {path} failed §5.3 validation:\n  - {joined}")


__all__ = [
    "PRODUCER",
    "SCHEMA_VERSION",
    "PerPatientRecord",
    "PredictionsH5Error",
    "assert_predictions_valid",
    "validate_predictions",
    "write_predictions_h5",
]
