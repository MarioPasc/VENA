"""Shared pytest fixtures for tests/validation/.

All three Phase-2 task agents (paired fidelity §4.2, spatial residual §4.3,
downstream seg §4.4) reuse these fixtures.  Building them once here prevents
divergence and keeps individual test files short.

Fixture overview
----------------
``synth_shard`` (session-scoped, ``tmp_path_factory``)
    A minimal schema-2.0 shard on disk with:

    - One prediction file per method × cohort pair.
    - One reference file per cohort.
    - A longitudinal cohort (``"LUMIERE-like"``) with 3 patients / 5 scans
      to exercise :func:`~vena.validation.stats.collapse_to_patient`.
    - Reference rows in **reversed** scan order so tests cannot pass an
      index-join by accident (SHARED_CONTRACTS §11 join-trap).
    - ``references_h5`` attr on every prediction file pointing at the correct
      reference (relative to shard root).

``pred_path`` / ``ref_path`` (function-scoped)
    Convenience paths derived from ``synth_shard`` for single-file tests.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Volume shape — small for speed; large enough for dilation tests.
# ---------------------------------------------------------------------------
_H, _W, _D = 16, 16, 16


# ---------------------------------------------------------------------------
# Low-level writers
# ---------------------------------------------------------------------------


def _vlen_str(grp: h5py.Group, name: str, values: list[str]) -> None:
    dt = h5py.string_dtype(encoding="utf-8")
    grp.create_dataset(name, data=np.asarray(values, dtype=object), dtype=dt)


def _write_pred_h5(
    path: Path,
    scan_ids: list[str],
    patient_ids: list[str],
    *,
    method: str,
    cohort: str,
    nfe: int,
    ring: str,
    references_h5: str,
    rng_seed: int = 0,
) -> None:
    """Write a minimal schema-2.0 prediction H5 at *path*."""
    n = len(scan_ids)
    rng = np.random.default_rng(rng_seed)
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = "2.0"
        f.attrs["method"] = method
        f.attrs["cohort"] = cohort
        f.attrs["nfe"] = nfe
        f.attrs["ring"] = ring
        f.attrs["references_h5"] = references_h5

        g_pred = f.create_group("predictions")
        data = rng.random((n, _H, _W, _D)).astype(np.float32)
        g_pred.create_dataset("t1c_synthetic_harmonised", data=data, chunks=(1, _H, _W, _D))
        g_pred.create_dataset("t1c_synthetic_raw", data=data.copy(), chunks=(1, _H, _W, _D))

        g_msk = f.create_group("masks")
        masks = np.ones((n, _H, _W, _D), dtype=np.int8)
        g_msk.create_dataset("brain", data=masks, chunks=(1, _H, _W, _D))
        g_msk.create_dataset("wt", data=masks, chunks=(1, _H, _W, _D))

        g_meta = f.create_group("metadata")
        _vlen_str(g_meta, "scan_id", scan_ids)
        _vlen_str(g_meta, "patient_id", patient_ids)
        _vlen_str(g_meta, "cohort", [cohort] * n)
        g_meta.create_dataset("inference_seconds", data=np.ones(n, dtype=np.float32))
        g_meta.create_dataset("peak_vram_mb", data=np.full(n, 1000.0, dtype=np.float32))
        g_meta.create_dataset("nfe", data=np.full(n, nfe, dtype=np.int32))


def _write_ref_h5(
    path: Path,
    scan_ids: list[str],
    patient_ids: list[str],
    *,
    cohort: str,
    rng_seed: int = 42,
) -> None:
    """Write a minimal schema-2.0 reference H5 at *path*.

    Note: scan_ids are written in the order given.  Pass a *reversed* list to
    exercise the scan_id join (not row-index join) in iter_scans.
    """
    n = len(scan_ids)
    rng = np.random.default_rng(rng_seed)
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = "2.0"
        f.attrs["cohort"] = cohort

        g_ref = f.create_group("reference")
        vol = rng.random((n, _H, _W, _D)).astype(np.float32)
        for name in (
            "t1c_real_harmonised",
            "t1pre_harmonised",
            "t2_harmonised",
            "flair_harmonised",
        ):
            g_ref.create_dataset(name, data=vol.copy(), chunks=(1, _H, _W, _D))

        g_msk = f.create_group("masks")
        masks = np.ones((n, _H, _W, _D), dtype=np.int8)
        g_msk.create_dataset("brain", data=masks, chunks=(1, _H, _W, _D))
        g_msk.create_dataset("wt", data=masks, chunks=(1, _H, _W, _D))

        g_meta = f.create_group("metadata")
        _vlen_str(g_meta, "scan_id", scan_ids)
        _vlen_str(g_meta, "patient_id", patient_ids)
        _vlen_str(g_meta, "cohort", [cohort] * n)


# ---------------------------------------------------------------------------
# Shard layout
# ---------------------------------------------------------------------------

#: Methods written into the synthetic shard.
_METHODS = ["VENA-S1-v3b-rw", "C0-Identity"]

#: (cohort, ring, scan_ids, patient_ids) tuples.
#: LUMIERE-like has 3 patients × repeated scans (longitudinal); deliberately
#: more scans than patients to exercise collapse_to_patient.
_COHORTS: list[tuple[str, str, list[str], list[str]]] = [
    (
        "TestCohortA",
        "A",
        ["scanA1", "scanA2", "scanA3"],
        ["ptA1", "ptA2", "ptA3"],
    ),
    (
        "LUMIERE-like",
        "A",
        # 3 patients, 5 scans: pt1 has 2 scans, pt2 has 2, pt3 has 1.
        ["lum_s1", "lum_s2", "lum_s3", "lum_s4", "lum_s5"],
        ["lum_pt1", "lum_pt1", "lum_pt2", "lum_pt2", "lum_pt3"],
    ),
    (
        "TestCohortB",
        "B",
        ["scanB1", "scanB2"],
        ["ptB1", "ptB2"],
    ),
]


@pytest.fixture(scope="session")
def synth_shard(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped synthetic shard on disk.

    Returns the shard root path.  Every test that reads predictions or
    references should derive paths from here rather than building their own
    fixtures.

    Layout::

        <shard>/
        ├── predictions/<METHOD>/<COHORT>/nfe_5.h5
        └── references/<COHORT>.h5

    Reference rows are written in **reversed** scan order to catch index-join
    bugs (SHARED_CONTRACTS §11).
    """
    root = tmp_path_factory.mktemp("shard")

    for cohort, ring, scan_ids, patient_ids in _COHORTS:
        ref_dir = root / "references"
        ref_dir.mkdir(parents=True, exist_ok=True)
        # Deliberately reversed reference order.
        _write_ref_h5(
            ref_dir / f"{cohort}.h5",
            list(reversed(scan_ids)),
            list(reversed(patient_ids)),
            cohort=cohort,
        )

        for method in _METHODS:
            pred_dir = root / "predictions" / method / cohort
            pred_dir.mkdir(parents=True, exist_ok=True)
            _write_pred_h5(
                pred_dir / "nfe_005.h5",
                scan_ids,
                patient_ids,
                method=method,
                cohort=cohort,
                nfe=5,
                ring=ring,
                references_h5=f"references/{cohort}.h5",
            )

    return root


@pytest.fixture()
def pred_path(synth_shard: Path) -> Path:
    """Path to a single prediction file (VENA-S1-v3b-rw / TestCohortA / nfe_005)."""
    return synth_shard / "predictions" / "VENA-S1-v3b-rw" / "TestCohortA" / "nfe_005.h5"


@pytest.fixture()
def ref_path(synth_shard: Path) -> Path:
    """Path to the TestCohortA reference H5."""
    return synth_shard / "references" / "TestCohortA.h5"
