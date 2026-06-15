"""Multi-cohort latent-H5 tests for the T1C-RFlow wrapper.

Mirrors ``tests/competitors/pgan_cgan/test_multicohort.py`` behaviour-by-
behaviour: synthetic two-cohort registry + missing-cohort skip + role filter
+ max-patients-per-cohort. Pinned to the latent contract instead of the
image contract.

Citation: Eidex et al. 2025, arXiv:2509.24194.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from vena.competitors.t1c_rflow.dataset import (
    DatasetError,
    MultiCohortT1CRFlowLatentDataset,
)

pytestmark = pytest.mark.unit


_C = 4
_LH, _LW, _LD = 8, 8, 6


def _write_synth_latent(out: Path, n_patients: int, cohort_prefix: str) -> Path:
    """Tiny per-cohort synthetic latent H5."""
    rng = np.random.default_rng(hash(cohort_prefix) % (1 << 31))
    with h5py.File(out, "w") as f:
        ids = np.asarray(
            [f"{cohort_prefix}-{i:04d}".encode() for i in range(n_patients)]
        )
        f.create_dataset("ids", data=ids)
        for mod in ("t1pre", "flair", "t1c"):
            arr = rng.standard_normal(
                size=(n_patients, _C, _LH, _LW, _LD)
            ).astype(np.float32)
            f.create_dataset(f"latents/{mod}", data=arr)
        # Splits: first half train, next quarter val.
        n_train = max(1, n_patients // 2)
        n_val = max(1, n_patients // 4)
        f.create_dataset(
            "splits/cv/fold_0/train", data=ids[:n_train]
        )
        f.create_dataset(
            "splits/cv/fold_0/val", data=ids[n_train : n_train + n_val]
        )
        f.create_dataset(
            "splits/test",
            data=ids[n_train + n_val : n_train + n_val + 1],
        )
        f.attrs["schema_version"] = "2.0.0"
    return out


@pytest.fixture
def two_cohort_corpus(tmp_path: Path) -> Path:
    """Build two latent H5s + a registry JSON pointing at them.

    Cohort A: 6 patients, role=cv.
    Cohort B: 4 patients, role=cv.
    Cohort C-MISSING: a non-existent latent_h5 path, role=cv (skip with WARNING).
    Cohort T-IGNORED: 2 patients, role=test_only (excluded from role=cv).
    """
    a = _write_synth_latent(tmp_path / "cohortA.h5", 6, "COHORT-A")
    b = _write_synth_latent(tmp_path / "cohortB.h5", 4, "COHORT-B")
    t = _write_synth_latent(tmp_path / "cohortT.h5", 2, "COHORT-T")
    registry = {
        "schema_version": "2.0.0",
        "name": "synth",
        "cohorts": [
            {
                "name": "COHORT-A",
                "pathology": "preoperative_glioma",
                "role": "cv",
                "longitudinal": False,
                "latent_h5": str(a),
                "image_h5": str(a),
            },
            {
                "name": "COHORT-B",
                "pathology": "preoperative_glioma",
                "role": "cv",
                "longitudinal": False,
                "latent_h5": str(b),
                "image_h5": str(b),
            },
            {
                "name": "COHORT-C-MISSING",
                "pathology": "preoperative_glioma",
                "role": "cv",
                "longitudinal": False,
                "latent_h5": str(tmp_path / "does_not_exist.h5"),
                "image_h5": str(tmp_path / "does_not_exist.h5"),
            },
            {
                "name": "COHORT-T-IGNORED",
                "pathology": "preoperative_glioma",
                "role": "test_only",
                "longitudinal": False,
                "latent_h5": str(t),
                "image_h5": str(t),
            },
        ],
    }
    out = tmp_path / "registry.json"
    out.write_text(json.dumps(registry, indent=2))
    return out


def test_multicohort_concats_two_cohorts(two_cohort_corpus: Path) -> None:
    ds = MultiCohortT1CRFlowLatentDataset(
        corpus_registry=two_cohort_corpus, fold=0, phase="train",
    )
    # 3 train patients in cohort A + 2 train in cohort B = 5; missing-cohort
    # and test_only entries are skipped.
    assert ds.cohort_names == ["COHORT-A", "COHORT-B"]
    assert len(ds) == sum(ds.cohort_sizes) == 5


def test_multicohort_sample_shape_consistent(two_cohort_corpus: Path) -> None:
    ds = MultiCohortT1CRFlowLatentDataset(
        corpus_registry=two_cohort_corpus, fold=0, phase="train",
    )
    for i in range(len(ds)):
        s = ds[i]
        assert s["z_t1pre"].shape == (_C, _LH, _LW, _LD)
        assert s["z_flair"].shape == (_C, _LH, _LW, _LD)
        assert s["z_t1c"].shape == (_C, _LH, _LW, _LD)


def test_multicohort_role_filter(two_cohort_corpus: Path) -> None:
    ds = MultiCohortT1CRFlowLatentDataset(
        corpus_registry=two_cohort_corpus,
        fold=0,
        phase="train",
        role_filter="test_only",
    )
    assert ds.cohort_names == ["COHORT-T-IGNORED"]


def test_multicohort_max_patients_per_cohort(two_cohort_corpus: Path) -> None:
    ds = MultiCohortT1CRFlowLatentDataset(
        corpus_registry=two_cohort_corpus,
        fold=0,
        phase="train",
        max_patients_per_cohort=1,
    )
    assert len(ds) == 2  # 1 per cohort × 2 cohorts


def test_multicohort_no_usable_cohorts_raises(tmp_path: Path) -> None:
    registry = {
        "schema_version": "2.0.0",
        "name": "empty",
        "cohorts": [
            {
                "name": "C1",
                "pathology": "preoperative_glioma",
                "role": "cv",
                "longitudinal": False,
                "latent_h5": str(tmp_path / "nope_a.h5"),
                "image_h5": str(tmp_path / "nope_a.h5"),
            },
        ],
    }
    out = tmp_path / "empty.json"
    out.write_text(json.dumps(registry))
    with pytest.raises(DatasetError, match="no usable cohorts"):
        MultiCohortT1CRFlowLatentDataset(
            corpus_registry=out, fold=0, phase="train",
        )


def test_multicohort_path_overrides(two_cohort_corpus: Path, tmp_path: Path) -> None:
    """Path overrides redirect a cohort's latent_h5 lookup."""
    # Move cohort-A's H5 to a side path and override the lookup.
    new_a = tmp_path / "moved_cohortA.h5"
    _write_synth_latent(new_a, 4, "COHORT-A")
    ds = MultiCohortT1CRFlowLatentDataset(
        corpus_registry=two_cohort_corpus,
        fold=0,
        phase="train",
        path_overrides={"COHORT-A": new_a},
    )
    # cohort-A now has 4 patients (2 train), cohort-B has 4 patients (2 train).
    a_idx = ds.cohort_names.index("COHORT-A")
    assert ds.cohort_sizes[a_idx] == 2
