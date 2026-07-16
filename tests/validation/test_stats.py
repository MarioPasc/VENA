"""Tests for vena.validation.stats.

Covers: collapse_to_patient, holm_bonferroni (textbook), cliffs_delta
(degenerate cases), bootstrap_ci (reproducibility), paired_wilcoxon.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.validation


# ---------------------------------------------------------------------------
# collapse_to_patient
# ---------------------------------------------------------------------------


def test_collapse_to_patient_lumiere_like() -> None:
    """LUMIERE-like: 72 scans from 11 patients → 11 patient rows."""
    from vena.validation.stats import collapse_to_patient

    rng = np.random.default_rng(0)
    # 11 patients: 6 patients × 6 scans + 5 patients × 36/5 ≈ 7 scans.
    # Distribution: patients 0-5 → 6 scans each (36), patients 6-10 → 7+7+7+7+8 (36) = 72.
    scans_per_patient = [6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 8]  # sum = 72
    patient_ids = []
    ssim_vals = []
    for pid, n_scans in enumerate(scans_per_patient):
        patient_ids.extend([f"pt_{pid:02d}"] * n_scans)
        ssim_vals.extend(rng.uniform(0.7, 0.9, size=n_scans).tolist())

    df = pd.DataFrame(
        {
            "method": ["VENA-test"] * 72,
            "cohort": ["LUMIERE"] * 72,
            "nfe": [5] * 72,
            "patient_id": patient_ids,
            "ssim": ssim_vals,
        }
    )

    out = collapse_to_patient(df, ["ssim"])
    assert len(out) == len(set(patient_ids)), "expected one row per unique patient"
    assert list(out.columns) == ["method", "cohort", "nfe", "patient_id", "ssim"]


def test_collapse_to_patient_mean_is_correct() -> None:
    """Patient mean is the arithmetic mean of its scans."""
    from vena.validation.stats import collapse_to_patient

    df = pd.DataFrame(
        {
            "method": ["M"] * 4,
            "cohort": ["C"] * 4,
            "nfe": [1] * 4,
            "patient_id": ["p1", "p1", "p2", "p2"],
            "ssim": [0.8, 0.6, 0.9, 0.7],
        }
    )
    out = collapse_to_patient(df, ["ssim"])
    out = out.set_index("patient_id")
    assert abs(float(out.loc["p1", "ssim"]) - 0.7) < 1e-6
    assert abs(float(out.loc["p2", "ssim"]) - 0.8) < 1e-6


# ---------------------------------------------------------------------------
# holm_bonferroni
# ---------------------------------------------------------------------------


def test_holm_bonferroni_textbook_example() -> None:
    """Reproduce a standard Holm (1979) worked example.

    With k=4 hypotheses and α=0.05:
      Sorted p: 0.001, 0.013, 0.025, 0.450
      Holm thresholds: 0.05/4, 0.05/3, 0.05/2, 0.05/1 = 0.0125, 0.0167, 0.025, 0.05
      Decisions: reject, reject, reject-threshold (0.025 ≤ 0.025), fail-to-reject

    statsmodels holm gives adjusted p-values; we check reject flags.
    """
    from vena.validation.stats import holm_bonferroni

    pvalues = {
        "H1": 0.001,
        "H2": 0.013,
        "H3": 0.025,
        "H4": 0.450,
    }
    result = holm_bonferroni(pvalues)

    assert result["H1"].reject is True
    assert result["H2"].reject is True
    assert result["H3"].reject is True  # 0.025 × 2 = 0.05 — borderline reject
    assert result["H4"].reject is False


def test_holm_bonferroni_empty_input() -> None:
    """Empty input returns empty dict."""
    from vena.validation.stats import holm_bonferroni

    assert holm_bonferroni({}) == {}


def test_holm_bonferroni_single_hypothesis() -> None:
    """Single hypothesis: adjusted p = raw p, reject iff raw p < 0.05."""
    from vena.validation.stats import holm_bonferroni

    result = holm_bonferroni({"only": 0.03})
    assert result["only"].reject is True

    result2 = holm_bonferroni({"only": 0.1})
    assert result2["only"].reject is False


# ---------------------------------------------------------------------------
# cliffs_delta
# ---------------------------------------------------------------------------


def test_cliffs_delta_fully_dominant() -> None:
    """All a > all b → δ = +1."""
    from vena.validation.stats import cliffs_delta

    a = np.array([0.9, 0.8, 0.7])
    b = np.array([0.1, 0.2, 0.3])
    assert abs(cliffs_delta(a, b) - 1.0) < 1e-9


def test_cliffs_delta_fully_dominated() -> None:
    """All a < all b → δ = -1."""
    from vena.validation.stats import cliffs_delta

    a = np.array([0.1, 0.2, 0.3])
    b = np.array([0.9, 0.8, 0.7])
    assert abs(cliffs_delta(a, b) + 1.0) < 1e-9


def test_cliffs_delta_identical() -> None:
    """Identical arrays → δ = 0."""
    from vena.validation.stats import cliffs_delta

    a = np.array([0.5, 0.5, 0.5])
    assert abs(cliffs_delta(a, a.copy()) - 0.0) < 1e-9


def test_cliffs_delta_antisymmetry() -> None:
    """cliffs_delta(a, b) == -cliffs_delta(b, a)."""
    from vena.validation.stats import cliffs_delta

    rng = np.random.default_rng(7)
    a = rng.uniform(0, 1, 20)
    b = rng.uniform(0, 1, 20)
    assert abs(cliffs_delta(a, b) + cliffs_delta(b, a)) < 1e-9


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


def test_bootstrap_ci_deterministic() -> None:
    """Same seed → same CI bounds across calls."""
    from vena.validation.stats import bootstrap_ci

    vals = np.array([0.7, 0.8, 0.75, 0.85, 0.9])
    lo1, hi1 = bootstrap_ci(vals, seed=1337)
    lo2, hi2 = bootstrap_ci(vals, seed=1337)
    assert lo1 == lo2
    assert hi1 == hi2


def test_bootstrap_ci_different_seeds() -> None:
    """Different seeds give different (or at most coincidentally equal) results."""
    from vena.validation.stats import bootstrap_ci

    vals = np.array([0.7, 0.8, 0.75, 0.85, 0.9, 0.6, 0.72])
    lo1, hi1 = bootstrap_ci(vals, seed=1337)
    lo2, hi2 = bootstrap_ci(vals, seed=42)
    # Very unlikely to be identical with different seeds — use as a smoke test.
    assert not (lo1 == lo2 and hi1 == hi2)


def test_bootstrap_ci_stratified_deterministic() -> None:
    """Stratified bootstrap is also deterministic with a fixed seed."""
    from vena.validation.stats import bootstrap_ci

    vals = np.array([0.7, 0.8, 0.75, 0.85, 0.9, 0.6])
    strata = np.array(["A", "A", "A", "B", "B", "B"])
    lo1, hi1 = bootstrap_ci(vals, strata=strata, seed=1337)
    lo2, hi2 = bootstrap_ci(vals, strata=strata, seed=1337)
    assert lo1 == lo2
    assert hi1 == hi2


def test_bootstrap_ci_bounds_ordering() -> None:
    """lo ≤ mean ≤ hi."""
    from vena.validation.stats import bootstrap_ci

    vals = np.array([0.7, 0.8, 0.75, 0.85, 0.9])
    lo, hi = bootstrap_ci(vals, seed=1337)
    assert lo <= float(np.mean(vals)) <= hi
    assert lo <= hi


# ---------------------------------------------------------------------------
# paired_wilcoxon
# ---------------------------------------------------------------------------


def test_paired_wilcoxon_all_zero_differences() -> None:
    """All differences zero → pvalue = 1.0 without scipy raising."""
    from vena.validation.stats import paired_wilcoxon

    idx = pd.Index(["p1", "p2", "p3"], name="patient_id")
    v = pd.Series([0.8, 0.8, 0.8], index=idx)
    c = pd.Series([0.8, 0.8, 0.8], index=idx)
    result = paired_wilcoxon(v, c)
    assert result.pvalue == 1.0
    assert result.n == 3


def test_paired_wilcoxon_no_common_raises() -> None:
    """No shared patient_ids raises ValueError."""
    from vena.validation.stats import paired_wilcoxon

    v = pd.Series([0.8], index=pd.Index(["p1"]))
    c = pd.Series([0.7], index=pd.Index(["p2"]))
    with pytest.raises(ValueError, match="no common patient_ids"):
        paired_wilcoxon(v, c)


def test_paired_wilcoxon_inner_joins_on_patient_id() -> None:
    """Only the intersection of patient_ids is tested."""
    from vena.validation.stats import paired_wilcoxon

    # vena has p1, p2, p3; competitor has p2, p3, p4.
    # Inner join → p2, p3 (n=2).
    v = pd.Series([0.9, 0.85, 0.80], index=pd.Index(["p1", "p2", "p3"]))
    c = pd.Series([0.7, 0.75, 0.9], index=pd.Index(["p2", "p3", "p4"]))
    result = paired_wilcoxon(v, c)
    assert result.n == 2
