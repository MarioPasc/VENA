"""Tests for vena.segmentation.data.kfold — fold plan determinism and leakage guards.

All tests use synthetic patient ID lists; no real cohort data is read.

Synthetic ID convention: ``COHA_001``, ``COHA_002`` … for cohort A;
``COHB_001`` … for cohort B.  The ``_extract_cohort`` heuristic strips
trailing digits → ``COHA``, ``COHB`` which satisfies the stratification path.
"""

from __future__ import annotations

import json
from typing import Literal
from unittest.mock import MagicMock

import pytest

from vena.segmentation.data.kfold import FoldPlan, _extract_cohort, build_fold_plan, oof_assignment

pytestmark = pytest.mark.segmentation

# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------


def _make_ids(cohort: str, n: int, offset: int = 0) -> list[str]:
    """Make n synthetic patient IDs for one cohort."""
    return [f"{cohort}_{i + offset:03d}" for i in range(n)]


def _make_splits(
    train: list[str],
    val: list[str],
    test: list[str],
) -> dict[str, list[str]]:
    return {"train": train, "val": val, "test": test}


def _make_cfg(k_folds: int = 5, fold_seed: int = 42) -> object:
    """Return a minimal DataConfig-like object (avoids real filesystem I/O)."""
    cfg = MagicMock()
    cfg.k_folds = k_folds
    cfg.fold_seed = fold_seed
    return cfg


# ---------------------------------------------------------------------------
# _extract_cohort heuristic
# ---------------------------------------------------------------------------


class TestExtractCohort:
    def test_brats_gli(self) -> None:
        assert _extract_cohort("BraTS-GLI-00001") == "BraTS-GLI"

    def test_ucsf_pdgm(self) -> None:
        assert _extract_cohort("UCSF-PDGM-001") == "UCSF-PDGM"

    def test_synthetic_underscore(self) -> None:
        assert _extract_cohort("COHA_001") == "COHA"

    def test_no_digits(self) -> None:
        # Degenerate: no trailing digits → cohort == patient_id
        assert _extract_cohort("ABC") == "ABC"

    def test_only_digits(self) -> None:
        # Degenerate: all digits strip → empty → fall back to patient_id
        assert _extract_cohort("001") == "001"


# ---------------------------------------------------------------------------
# FoldPlan serialisation
# ---------------------------------------------------------------------------


class TestFoldPlanSerialization:
    def test_round_trip(self) -> None:
        plan = FoldPlan(
            k=3,
            fm_train_ids=("a", "b", "c"),
            folds=(("a",), ("b",), ("c",)),
            fm_val_ids=("v1",),
            fm_test_ids=("t1",),
        )
        d = plan.to_dict()
        plan2 = FoldPlan.from_dict(d)
        assert plan2 == plan

    def test_json_round_trip(self) -> None:
        plan = FoldPlan(
            k=2,
            fm_train_ids=("x", "y"),
            folds=(("x",), ("y",)),
            fm_val_ids=(),
            fm_test_ids=(),
        )
        j = plan.to_json()
        plan2 = FoldPlan.from_dict(json.loads(j))
        assert plan2 == plan


# ---------------------------------------------------------------------------
# build_fold_plan — determinism
# ---------------------------------------------------------------------------


class TestBuildFoldPlanDeterminism:
    def _build(self, n_train: int = 50, k: int = 5, seed: int = 42) -> FoldPlan:
        train = _make_ids("COHA", n_train // 2) + _make_ids("COHB", n_train // 2)
        val = _make_ids("COHVAL", 5)
        test = _make_ids("COHTEST", 5)
        splits = _make_splits(train, val, test)
        cfg = _make_cfg(k_folds=k, fold_seed=seed)
        return build_fold_plan(cfg, splits)

    def test_two_calls_equal(self) -> None:
        plan1 = self._build()
        plan2 = self._build()
        assert plan1 == plan2, "Two calls with same inputs must produce identical FoldPlan"

    def test_different_seed_differs(self) -> None:
        plan1 = self._build(seed=42)
        plan2 = self._build(seed=99)
        # With 50 patients across two cohorts, different seeds almost certainly differ
        assert plan1.folds != plan2.folds

    def test_different_k_differs(self) -> None:
        plan3 = self._build(k=3)
        plan5 = self._build(k=5)
        assert plan3.k == 3
        assert plan5.k == 5
        assert len(plan3.folds) == 3
        assert len(plan5.folds) == 5


# ---------------------------------------------------------------------------
# build_fold_plan — structural invariants
# ---------------------------------------------------------------------------


class TestBuildFoldPlanStructure:
    def _make_plan(self, n_per_cohort: int = 20, k: int = 5) -> FoldPlan:
        train = _make_ids("COHA", n_per_cohort) + _make_ids("COHB", n_per_cohort)
        val = _make_ids("VAL", 4)
        test = _make_ids("TST", 4)
        splits = _make_splits(train, val, test)
        cfg = _make_cfg(k_folds=k)
        return build_fold_plan(cfg, splits)

    def test_k_matches_config(self) -> None:
        plan = self._make_plan()
        assert plan.k == 5
        assert len(plan.folds) == 5

    def test_folds_disjoint(self) -> None:
        plan = self._make_plan()
        all_ids: list[str] = [pid for fold in plan.folds for pid in fold]
        assert len(all_ids) == len(set(all_ids)), "Folds must be pairwise disjoint"

    def test_union_equals_train(self) -> None:
        plan = self._make_plan()
        all_in_folds = set(pid for fold in plan.folds for pid in fold)
        assert all_in_folds == set(plan.fm_train_ids), "⋃ folds must equal fm_train_ids"

    def test_val_ids_not_in_folds(self) -> None:
        plan = self._make_plan()
        all_in_folds = set(pid for fold in plan.folds for pid in fold)
        overlap = all_in_folds & set(plan.fm_val_ids)
        assert not overlap, f"FM-val IDs leaked into folds: {overlap}"

    def test_test_ids_not_in_folds(self) -> None:
        plan = self._make_plan()
        all_in_folds = set(pid for fold in plan.folds for pid in fold)
        overlap = all_in_folds & set(plan.fm_test_ids)
        assert not overlap, f"FM-test IDs leaked into folds: {overlap}"

    def test_fm_train_ids_sorted(self) -> None:
        plan = self._make_plan()
        assert list(plan.fm_train_ids) == sorted(plan.fm_train_ids)


# ---------------------------------------------------------------------------
# Cohort coverage (stratification)
# ---------------------------------------------------------------------------


class TestCohortCoverage:
    def test_each_cohort_in_each_fold(self) -> None:
        """With ≥k patients per cohort, every fold should contain both cohorts."""
        k = 5
        n_per_cohort = 30  # 6 per fold on average — coverage is very likely
        train = _make_ids("COHA", n_per_cohort) + _make_ids("COHB", n_per_cohort)
        val = _make_ids("VAL", 2)
        test = _make_ids("TST", 2)
        splits = _make_splits(train, val, test)
        cfg = _make_cfg(k_folds=k)
        plan = build_fold_plan(cfg, splits)

        for fold_idx, fold in enumerate(plan.folds):
            cohorts_in_fold = {_extract_cohort(pid) for pid in fold}
            assert "COHA" in cohorts_in_fold, f"Fold {fold_idx} missing cohort COHA"
            assert "COHB" in cohorts_in_fold, f"Fold {fold_idx} missing cohort COHB"

    def test_single_cohort_fallback(self) -> None:
        """Single-cohort input must still produce valid k-fold split."""
        train = _make_ids("COHA", 20)
        splits = _make_splits(train, [], [])
        cfg = _make_cfg(k_folds=4)
        plan = build_fold_plan(cfg, splits)
        assert set(pid for fold in plan.folds for pid in fold) == set(train)

    def test_fold_sizes_approx_equal(self) -> None:
        """Fold sizes should differ by at most 1 (StratifiedKFold property)."""
        k = 5
        n = 25
        train = _make_ids("COHA", n // 2) + _make_ids("COHB", n // 2)
        splits = _make_splits(train, [], [])
        cfg = _make_cfg(k_folds=k)
        plan = build_fold_plan(cfg, splits)

        sizes = [len(f) for f in plan.folds]
        assert max(sizes) - min(sizes) <= 1, f"Fold sizes too unequal: {sizes}"


# ---------------------------------------------------------------------------
# Transitive dedup leakage guard (ITER-9 load-bearing)
# ---------------------------------------------------------------------------


class TestTransitiveDedupLeakage:
    """A dedup-duplicate of an FM-val/test ID must not appear in any fold."""

    def _make_plan_with_dedup(
        self,
        dedup_val_alias: str,
        dedup_map: dict[str, list[str]],
    ) -> FoldPlan:
        """Build a plan where one train ID is an alias of a val ID."""
        # train set includes the alias; val set includes the original
        train = _make_ids("COHA", 15) + _make_ids("COHB", 15)
        # Inject the dedup alias into the train set
        train = [*train, dedup_val_alias]
        val = ["VAL_001"]  # the original val ID
        test = ["TST_001"]
        splits = _make_splits(train, val, test)
        cfg = _make_cfg(k_folds=3)
        return build_fold_plan(cfg, splits, dedup_duplicates=dedup_map)

    def test_dedup_alias_excluded_from_folds(self) -> None:
        """The duplicate of a val ID must not appear in any fold."""
        alias = "COHB_XTRAIN_999"  # appears in train but is dedup of VAL_001
        dedup_map = {
            "VAL_001": [alias],
            alias: ["VAL_001"],  # symmetric
        }
        plan = self._make_plan_with_dedup(alias, dedup_map)
        all_in_folds = {pid for fold in plan.folds for pid in fold}
        assert alias not in all_in_folds, f"Dedup alias '{alias}' of a val ID leaked into folds"

    def test_direct_leakage_also_guarded(self) -> None:
        """Direct check: val/test IDs themselves must not appear in folds."""
        # Val ID accidentally also in train
        val_id = "VAL_001"
        train = _make_ids("COHA", 10) + _make_ids("COHB", 10) + [val_id]
        val = [val_id]
        test = ["TST_001"]
        splits = _make_splits(train, val, test)
        cfg = _make_cfg(k_folds=3)
        # Should still build successfully (the implementation removes overlaps)
        plan = build_fold_plan(cfg, splits)
        all_in_folds = {pid for fold in plan.folds for pid in fold}
        assert val_id not in all_in_folds, (
            f"Val ID '{val_id}' leaked into folds despite being in fm_splits['val']"
        )

    def test_no_dedup_map_skips_transitive(self) -> None:
        """Without dedup_duplicates, plan still builds; only direct check runs."""
        train = _make_ids("COHA", 10)
        val = _make_ids("VAL", 2)
        test = _make_ids("TST", 2)
        plan = build_fold_plan(
            _make_cfg(),
            _make_splits(train, val, test),
            dedup_duplicates=None,  # explicit None
        )
        # No crash — plan is structurally valid
        assert set(pid for fold in plan.folds for pid in fold) == set(plan.fm_train_ids)


# ---------------------------------------------------------------------------
# oof_assignment
# ---------------------------------------------------------------------------


class TestOofAssignment:
    def _make_plan(self) -> FoldPlan:
        train = _make_ids("COHA", 15) + _make_ids("COHB", 15)
        val = _make_ids("VAL", 3)
        test = _make_ids("TST", 3)
        splits = _make_splits(train, val, test)
        cfg = _make_cfg(k_folds=5)
        return build_fold_plan(cfg, splits)

    def test_val_returns_all_train(self) -> None:
        plan = self._make_plan()
        for pid in plan.fm_val_ids:
            result = oof_assignment(plan, pid)
            assert result == "all_train", (
                f"FM-val id '{pid}' should map to 'all_train', got {result!r}"
            )

    def test_test_returns_all_train(self) -> None:
        plan = self._make_plan()
        for pid in plan.fm_test_ids:
            result = oof_assignment(plan, pid)
            assert result == "all_train"

    def test_train_returns_fold_index(self) -> None:
        plan = self._make_plan()
        for fold_idx, fold in enumerate(plan.folds):
            for pid in fold:
                result = oof_assignment(plan, pid)
                assert result == fold_idx, (
                    f"ID '{pid}' is in fold {fold_idx} but oof_assignment returned {result}"
                )

    def test_all_train_ids_covered(self) -> None:
        plan = self._make_plan()
        assignments = {pid: oof_assignment(plan, pid) for pid in plan.fm_train_ids}
        # Every train ID maps to an int in [0, k)
        for pid, assignment in assignments.items():
            assert isinstance(assignment, int), (
                f"Expected int for train id '{pid}', got {assignment!r}"
            )
            assert 0 <= assignment < plan.k

    def test_unknown_id_raises(self) -> None:
        from vena.segmentation.exceptions import SegDataError

        plan = self._make_plan()
        with pytest.raises(SegDataError):
            oof_assignment(plan, "nonexistent_patient_xyz")

    def test_assignment_type_annotation(self) -> None:
        """Verify literal typing — all_train is the string, not an int."""
        plan = self._make_plan()
        result: int | Literal["all_train"] = oof_assignment(plan, plan.fm_val_ids[0])
        assert result == "all_train"
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestBuildFoldPlanErrors:
    def test_missing_train_key(self) -> None:
        from vena.segmentation.exceptions import SegDataError

        cfg = _make_cfg()
        with pytest.raises(SegDataError, match="missing required key"):
            build_fold_plan(cfg, {"val": [], "test": []})

    def test_empty_train(self) -> None:
        from vena.segmentation.exceptions import SegDataError

        cfg = _make_cfg()
        with pytest.raises(SegDataError, match="empty"):
            build_fold_plan(cfg, {"train": [], "val": [], "test": []})

    def test_too_few_patients_for_k(self) -> None:
        from vena.segmentation.exceptions import SegDataError

        cfg = _make_cfg(k_folds=10)
        train = _make_ids("COHA", 5)  # only 5, need 10
        with pytest.raises(SegDataError, match="Too few"):
            build_fold_plan(cfg, _make_splits(train, [], []))
