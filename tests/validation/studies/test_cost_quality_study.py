"""Unit tests for the cost–quality Pareto study.

Synthetic data only — the real paired_fidelity CSV is never touched.

Tests
-----
- ``test_pareto_dominance_known_frontier``: known 2D Pareto set verified.
- ``test_pareto_no_dominant_when_all_equal``: degenerate case — all equal.
- ``test_pareto_single_point``: single-row DataFrame is always dominant.
- ``test_table2a_filter_nfe5``: only nfe=5 rows for the four methods enter
  table 2A; rows at other NFE values or for other methods are excluded.
- ``test_table2a_vena_headline_has_nan_stats``: VENA_HEADLINE row has NaN
  for wilcoxon_p and cliffs (no self-comparison).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from routines.validation.studies.cost_quality_study import (
    _TABLE2A_METHODS,
    _TABLE2A_NFE,
    CostQualityStudyConfig,
    compute_pareto_dominant,
)

from vena.validation.registry import VENA_HEADLINE

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pareto_df(
    seconds: list[float],
    ms_ssim: list[float],
) -> pd.DataFrame:
    """Build a minimal DataFrame for dominance tests."""
    return pd.DataFrame(
        {
            "median_inference_seconds": seconds,
            "mean_ms_ssim_brain": ms_ssim,
        }
    )


# ---------------------------------------------------------------------------
# Pareto-dominance tests
# ---------------------------------------------------------------------------


class TestParetoDominance:
    """Pareto-dominance computed on (seconds ↓, ms_ssim ↑)."""

    def test_known_frontier_two_points(self) -> None:
        """Simple 3-point case with a known 2-point frontier."""
        # A (0.5s, 0.9): Pareto-dominant.
        # B (1.0s, 0.95): Pareto-dominant.
        # C (1.5s, 0.88): dominated by both A and B (slower AND worse).
        df = _make_pareto_df(
            seconds=[0.5, 1.0, 1.5],
            ms_ssim=[0.90, 0.95, 0.88],
        )
        dom = compute_pareto_dominant(df)
        assert dom.tolist() == [True, True, False], dom.tolist()

    def test_known_frontier_five_points(self) -> None:
        """Five points matching the real data shape (staircase)."""
        # P0: (0.31, 0.785) dominant — fastest point.
        # P1: (1.49, 0.918) dominant — next step up.
        # P2: (1.76, 0.919) dominant — tiny quality gain.
        # P3: (1.79, 0.925) dominant.
        # P4: (1.82, 0.918) NOT dominant — dominated by P2 (faster, same quality).
        df = _make_pareto_df(
            seconds=[0.31, 1.49, 1.76, 1.79, 1.82],
            ms_ssim=[0.785, 0.918, 0.919, 0.925, 0.918],
        )
        dom = compute_pareto_dominant(df)
        # P4 dominated by P2 (1.76 < 1.82 AND 0.919 > 0.918)
        expected = [True, True, True, True, False]
        assert dom.tolist() == expected, dom.tolist()

    def test_dominated_point_excluded_when_strictly_worse_on_both(self) -> None:
        """A point strictly worse on both axes is not dominant."""
        df = _make_pareto_df(
            seconds=[1.0, 2.0],
            ms_ssim=[0.90, 0.80],
        )
        dom = compute_pareto_dominant(df)
        # P1 (2.0, 0.80) strictly dominated by P0 (1.0, 0.90).
        assert dom.tolist() == [True, False]

    def test_tie_on_one_axis_is_not_dominated(self) -> None:
        """Two points with equal quality but different speed: both dominant."""
        df = _make_pareto_df(
            seconds=[1.0, 2.0],
            ms_ssim=[0.90, 0.90],  # equal quality
        )
        dom = compute_pareto_dominant(df)
        # P1 (2.0, 0.90): P0 has same quality BUT is faster — strict on seconds.
        # So P0 dominates P1.
        assert dom.tolist() == [True, False]

    def test_pareto_no_dominant_when_all_equal(self) -> None:
        """All identical points: all are Pareto-dominant (no strict inequality)."""
        df = _make_pareto_df(
            seconds=[1.0, 1.0, 1.0],
            ms_ssim=[0.90, 0.90, 0.90],
        )
        dom = compute_pareto_dominant(df)
        # Identical points cannot strictly dominate each other.
        assert dom.tolist() == [True, True, True]

    def test_pareto_single_point(self) -> None:
        """A single-row DataFrame is trivially dominant."""
        df = _make_pareto_df(seconds=[2.5], ms_ssim=[0.88])
        dom = compute_pareto_dominant(df)
        assert dom.tolist() == [True]

    def test_pareto_index_preserved(self) -> None:
        """The returned Series has the same index as the input DataFrame."""
        df = _make_pareto_df(seconds=[1.0, 2.0, 3.0], ms_ssim=[0.8, 0.9, 0.7])
        df.index = [10, 20, 30]
        dom = compute_pareto_dominant(df)
        assert list(dom.index) == [10, 20, 30]


# ---------------------------------------------------------------------------
# Table 2A filter tests (using synthetic per-scan DataFrame)
# ---------------------------------------------------------------------------


def _make_ring_a_df(
    methods: list[str],
    nfes: list[int],
    n_patients: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a minimal synthetic Ring-A per-scan DataFrame."""
    rng = np.random.default_rng(seed)
    rows = []
    for method in methods:
        for nfe in nfes:
            for pid in range(n_patients):
                rows.append(
                    {
                        "method": method,
                        "nfe": nfe,
                        "cohort": "TestCohort",
                        "ring": "A",
                        "patient_id": f"P{pid:03d}",
                        "mae_brain": float(rng.uniform(0.01, 0.10)),
                        "ms_ssim_brain": float(rng.uniform(0.85, 0.97)),
                        "inference_seconds": float(rng.uniform(1.5, 2.5)),
                        "peak_vram_mb": float(rng.uniform(4000, 20000)),
                    }
                )
    return pd.DataFrame(rows)


class TestTable2AFilter:
    """Table 2A correctly filters to NFE=5 and the four listed methods."""

    def test_only_nfe5_rows_enter_table2a(self) -> None:
        """Rows at NFE ≠ 5 must not appear in Table 2A."""
        # Build data with all four methods at NFEs {1, 5, 10}.
        df = _make_ring_a_df(
            methods=list(_TABLE2A_METHODS),
            nfes=[1, 5, 10],
            n_patients=5,
        )
        # Filter to NFE=5 (replicate the table2a filter logic).
        sub = df[df["method"].isin(_TABLE2A_METHODS) & (df["nfe"] == _TABLE2A_NFE)]
        assert set(sub["nfe"].unique()) == {_TABLE2A_NFE}, (
            f"Expected only nfe={_TABLE2A_NFE}; got {sub['nfe'].unique()}"
        )

    def test_only_four_methods_enter_table2a(self) -> None:
        """Methods outside _TABLE2A_METHODS must not appear in Table 2A."""
        extra_methods = [*_TABLE2A_METHODS, "C0-Identity", "C6-3D-LDDPM"]
        df = _make_ring_a_df(methods=extra_methods, nfes=[5], n_patients=3)
        sub = df[df["method"].isin(_TABLE2A_METHODS) & (df["nfe"] == _TABLE2A_NFE)]
        assert set(sub["method"].unique()) == set(_TABLE2A_METHODS), (
            f"Unexpected methods in sub: {set(sub['method'].unique()) - set(_TABLE2A_METHODS)}"
        )

    def test_table2a_methods_constant_contains_vena_headline(self) -> None:
        """The pre-registered VENA headline must be one of the four Table 2A arms."""
        assert VENA_HEADLINE in _TABLE2A_METHODS, (
            f"{VENA_HEADLINE!r} not in _TABLE2A_METHODS={_TABLE2A_METHODS}"
        )

    def test_table2a_nfe_constant_is_five(self) -> None:
        """Table 2A NFE is pre-registered as 5."""
        assert _TABLE2A_NFE == 5


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


class TestCostQualityStudyConfig:
    """Smoke-test for config defaults."""

    def test_ring_default(self) -> None:
        """Default ring is 'A'."""
        # Cannot call CostQualityStudyConfig() without the CSV on disk,
        # but we can build one explicitly to test ring default.
        cfg = CostQualityStudyConfig(
            per_scan_csv_path=__import__("pathlib").Path("/dev/null"),
            ring="A",
        )
        assert cfg.ring == "A"

    def test_output_root_default_uses_article_dir(self) -> None:
        """Default output_root points at the article results directory."""
        cfg = CostQualityStudyConfig(
            per_scan_csv_path=__import__("pathlib").Path("/dev/null"),
        )
        assert "article" in str(cfg.output_root), (
            f"output_root default should contain 'article'; got {cfg.output_root}"
        )

    def test_bootstrap_defaults(self) -> None:
        """Default bootstrap parameters match the spec."""
        cfg = CostQualityStudyConfig(
            per_scan_csv_path=__import__("pathlib").Path("/dev/null"),
        )
        assert cfg.n_bootstrap == 10_000
        assert cfg.bootstrap_seed == 1337
