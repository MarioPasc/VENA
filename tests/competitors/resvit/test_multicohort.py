"""Tests for ``MultiCohortImageSliceDataset`` (ResViT)."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from vena.competitors.resvit.dataset import (
    DatasetError,
    MultiCohortImageSliceDataset,
)

pytestmark = pytest.mark.unit


def _build_synth_h5(path: Path, N: int, shape: tuple[int, int, int]) -> None:
    H, W, D = shape
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        ids = np.array([f"{path.stem.upper()}-{i:04d}" for i in range(N)], dtype="S20")
        f.create_dataset("ids", data=ids)
        for mod in ("t1pre", "t1c", "t2", "flair"):
            f.create_dataset(
                f"images/{mod}",
                data=rng.uniform(0.0, 100.0, size=(N, H, W, D)).astype(np.float32),
            )
        brain = np.zeros((N, H, W, D), dtype=np.int8)
        zc, hc, wc = D // 2, H // 2, W // 2
        for z in range(D):
            if abs(z - zc) / max(zc, 1) < 0.7:
                brain[:, hc - 30 : hc + 30, wc - 30 : wc + 30, z] = 1
        f.create_dataset("masks/brain", data=brain)
        ids_bytes = list(ids)
        f.create_dataset("splits/cv/fold_0/train", data=np.array(ids_bytes[::2]))
        f.create_dataset("splits/cv/fold_0/val", data=np.array(ids_bytes[1::2]))
        f.create_dataset("splits/test", data=np.array(ids_bytes[:1]))


@pytest.fixture
def two_cohort_corpus(tmp_path: Path) -> Path:
    """Two synthetic cohorts + a corpus_registry.json pointing at them."""
    h5_a = tmp_path / "cohort_A.h5"
    h5_b = tmp_path / "cohort_B.h5"
    _build_synth_h5(h5_a, N=6, shape=(240, 240, 155))
    _build_synth_h5(h5_b, N=4, shape=(182, 218, 182))
    registry = {
        "schema_version": "1.0.0",
        "name": "test_corpus",
        "cohorts": [
            {"name": "COHORT-A", "role": "cv", "image_h5": str(h5_a)},
            {"name": "COHORT-B", "role": "cv", "image_h5": str(h5_b)},
            {"name": "COHORT-C-MISSING", "role": "cv",
             "image_h5": str(tmp_path / "does_not_exist.h5")},
            {"name": "COHORT-T-IGNORED", "role": "test_only",
             "image_h5": str(h5_a)},
        ],
    }
    cr = tmp_path / "corpus.json"
    cr.write_text(json.dumps(registry))
    return cr


def test_multicohort_concats_two_cohorts(two_cohort_corpus: Path) -> None:
    ds = MultiCohortImageSliceDataset(
        corpus_registry=two_cohort_corpus, fold=0, phase="train",
    )
    assert len(ds) > 200
    assert sorted(ds.cohort_names) == ["COHORT-A", "COHORT-B"]
    # Missing-cohort + test_only entries are skipped silently with WARNING.
    assert "COHORT-C-MISSING" not in ds.cohort_names
    assert "COHORT-T-IGNORED" not in ds.cohort_names


def test_multicohort_sample_shape_consistent(two_cohort_corpus: Path) -> None:
    ds = MultiCohortImageSliceDataset(
        corpus_registry=two_cohort_corpus, fold=0, phase="train", image_size=256,
    )
    source_shapes = set()
    target_shapes = set()
    for i in range(min(20, len(ds))):
        sample = ds[i]
        source_shapes.add(tuple(sample["A"].shape))
        target_shapes.add(tuple(sample["B"].shape))
    # Both cohorts have different native (H, W) but must produce (3, 256, 256)
    # for A and (1, 256, 256) for B.
    assert source_shapes == {(3, 256, 256)}
    assert target_shapes == {(1, 256, 256)}


def test_multicohort_role_filter(two_cohort_corpus: Path) -> None:
    ds = MultiCohortImageSliceDataset(
        corpus_registry=two_cohort_corpus, fold=0, phase="train",
        role_filter="test_only",
    )
    assert ds.cohort_names == ["COHORT-T-IGNORED"]


def test_multicohort_max_patients_per_cohort(two_cohort_corpus: Path) -> None:
    ds = MultiCohortImageSliceDataset(
        corpus_registry=two_cohort_corpus, fold=0, phase="train",
        max_patients_per_cohort=1,
    )
    assert len(ds) < 300


def test_multicohort_no_usable_cohorts_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({
        "schema_version": "1.0.0",
        "cohorts": [
            {"name": "X", "role": "test_only",
             "image_h5": str(tmp_path / "nope.h5")},
        ],
    }))
    with pytest.raises(DatasetError):
        MultiCohortImageSliceDataset(corpus_registry=bad, fold=0, phase="train")


def test_multicohort_path_overrides(two_cohort_corpus: Path, tmp_path: Path) -> None:
    h5_a = tmp_path / "cohort_A.h5"
    bad_registry = json.loads(two_cohort_corpus.read_text())
    bad_registry["cohorts"][0]["image_h5"] = "/nonexistent/picasso/path.h5"
    overridden = tmp_path / "overridden.json"
    overridden.write_text(json.dumps(bad_registry))
    ds = MultiCohortImageSliceDataset(
        corpus_registry=overridden, fold=0, phase="train",
        path_overrides={"COHORT-A": h5_a},
    )
    assert "COHORT-A" in ds.cohort_names
