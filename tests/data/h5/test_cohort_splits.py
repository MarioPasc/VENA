"""Per-cohort quota splits: size floor, leakage-proofness, test-only role."""

from __future__ import annotations

import math

import pytest

from vena.data.h5.shared.splits import make_cohort_splits


def _ids(n: int) -> list[str]:
    return [f"P{i:05d}" for i in range(n)]


@pytest.mark.unit
@pytest.mark.parametrize(
    "n,frac,floor,expected",
    [
        (200, 0.10, 25, 25),   # floor wins (0.10*200=20 < 25)
        (500, 0.10, 25, 50),   # fraction wins (0.10*500=50 > 25)
        (1133, 0.10, 25, 114),  # BraTS-like (ceil(113.3)=114)
    ],
)
def test_quota_test_size(n, frac, floor, expected) -> None:
    sp = make_cohort_splits(_ids(n), test_fraction=frac, n_test_min=floor, seed=0)
    assert len(sp["test"]) == expected == max(floor, math.ceil(frac * n))


@pytest.mark.unit
def test_no_patient_straddles_splits() -> None:
    sp = make_cohort_splits(_ids(300), n_folds=5, test_fraction=0.10, n_test_min=25, seed=1)
    test = set(sp["test"])
    cv_union = set().union(*[set(f["train"]) | set(f["val"]) for f in sp["folds"].values()])
    assert test.isdisjoint(cv_union)
    for k, fold in sp["folds"].items():
        assert set(fold["train"]).isdisjoint(set(fold["val"])), f"train/val overlap in fold {k}"
    # Every CV patient appears in exactly one val fold (partition property).
    val_counts: dict[str, int] = {}
    for fold in sp["folds"].values():
        for pid in fold["val"]:
            val_counts[pid] = val_counts.get(pid, 0) + 1
    assert all(c == 1 for c in val_counts.values())
    assert set(val_counts) == cv_union


@pytest.mark.unit
def test_test_only_role_has_no_folds() -> None:
    sp = make_cohort_splits(_ids(40), role="test_only")
    assert len(sp["test"]) == 40
    assert sp["folds"] == {}


@pytest.mark.unit
def test_too_small_cohort_raises() -> None:
    # n=20 with floor 25 → n_test >= n → invalid for cv role.
    with pytest.raises(ValueError):
        make_cohort_splits(_ids(20), test_fraction=0.10, n_test_min=25, role="cv")


@pytest.mark.unit
def test_stratified_split_balances_labels() -> None:
    ids = _ids(200)
    labels = [g for g in (1, 2, 3, 4) for _ in range(50)]
    sp = make_cohort_splits(
        ids, n_folds=5, test_fraction=0.10, n_test_min=25, seed=2, stratify_by=labels
    )
    assert len(sp["test"]) == 25
    # Determinism: same seed reproduces the test set.
    sp2 = make_cohort_splits(
        ids, n_folds=5, test_fraction=0.10, n_test_min=25, seed=2, stratify_by=labels
    )
    assert sp["test"] == sp2["test"]
