"""Patient-level nested CV splits: disjointness, coverage, stratification."""

from __future__ import annotations

import numpy as np
import pytest

from vena.data.h5.shared import make_nested_cv_splits


@pytest.mark.unit
def test_splits_are_disjoint_and_cover() -> None:
    ids = [f"P{i:04d}" for i in range(100)]
    out = make_nested_cv_splits(ids, n_folds=5, n_test=10, seed=0)
    test = set(out["test"])
    assert len(test) == 10
    cv_ids = set(ids) - test
    for k, fold in out["folds"].items():
        train = set(fold["train"])
        val = set(fold["val"])
        # each fold partitions the CV pool
        assert train | val == cv_ids
        assert train.isdisjoint(val)
        assert train.isdisjoint(test)
        assert val.isdisjoint(test)
        assert len(val) > 0
        assert len(train) > 0


@pytest.mark.unit
def test_splits_are_deterministic() -> None:
    ids = [f"P{i:04d}" for i in range(50)]
    a = make_nested_cv_splits(ids, n_folds=5, n_test=10, seed=42)
    b = make_nested_cv_splits(ids, n_folds=5, n_test=10, seed=42)
    assert a == b


@pytest.mark.unit
def test_stratification_preserves_label_distribution() -> None:
    n = 200
    ids = [f"P{i:04d}" for i in range(n)]
    labels = [i % 4 for i in range(n)]  # 4 equally-sized classes

    out = make_nested_cv_splits(ids, n_folds=5, n_test=40, seed=0, stratify_by=labels)

    # Held-out test must keep roughly the same per-class ratio as the cohort.
    test_idx = [ids.index(p) for p in out["test"]]
    test_labels = [labels[i] for i in test_idx]
    counts = np.bincount(test_labels, minlength=4)
    expected = 40 / 4
    np.testing.assert_allclose(counts, expected, atol=1)


@pytest.mark.unit
def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        make_nested_cv_splits(["a", "b", "c"], n_folds=2, n_test=5, seed=0)
    with pytest.raises(ValueError):
        make_nested_cv_splits(["a", "b", "c", "d"], n_folds=1, n_test=1, seed=0)
    with pytest.raises(ValueError):
        make_nested_cv_splits(["a", "b", "c", "d"], n_folds=2, n_test=1, seed=0, stratify_by=[0, 1])
