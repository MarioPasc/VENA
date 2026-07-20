"""Unit tests for vena.validation.spatial_residual (§4.3 library module).

All tests are pure NumPy — no H5, no disk, no GPU.  They verify:
- Analytic expectations: E[Conc(q)] = 1 under independence.
- Known-value Conc(q) computation.
- NaN returned on empty region.
- HOLM_FAMILY_SIZE constant.
- Shuffle convergence structure.
- Decile means output shape.
- D1 regression: n_pairs > 0 for all competitors even when VENA has multiple NFEs.
"""

# ruff: noqa: N806  — `_R` suffix mirrors the library API (R = region, §4.3).
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vena.validation.spatial_residual import (
    CONC_QUANTILES,
    HOLM_FAMILY_SIZE,
    SPATIAL_CSV_COLUMNS,
    _intensity_decile_means,
    _shuffle_null,
    aggregate_patient_tests,
    concentration_q,
    shuffle_convergence_check,
)

pytestmark = pytest.mark.unit

_RNG = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_holm_family_size() -> None:
    """HOLM_FAMILY_SIZE must be 16 = 2 stats × 8 competitors."""
    assert HOLM_FAMILY_SIZE == 16


def test_spatial_csv_columns_count() -> None:
    """Per-scan CSV header must have exactly 42 columns (40 + pred_mode + raw_p995)."""
    assert len(SPATIAL_CSV_COLUMNS) == 42


def test_conc_quantiles() -> None:
    """CONC_QUANTILES includes the primary q=0.05 value."""
    assert 0.05 in CONC_QUANTILES


# ---------------------------------------------------------------------------
# concentration_q — known-value test
# ---------------------------------------------------------------------------


def test_concentration_q_known_value() -> None:
    """Conc(0.5) on a perfectly concentrated residual equals n / ceil(0.5·n).

    With n=10 residuals and the top-5 being large, all error mass falls in the
    top 50% → Conc(0.5) = 1 / (5/10) = 2.0.
    """
    n = 10
    # Top 5 have large |r|, bottom 5 have 0.
    abs_resid = np.array([0.0] * 5 + [1.0] * 5, dtype=np.float64)
    t1c = np.ones(n, dtype=np.float64)  # uniform T1c (doesn't affect Conc)

    conc = concentration_q(abs_resid, t1c, q=0.5)
    # ceil(0.5 * 10) = 5; top-5 hold all error; denominator fraction = 5/10 = 0.5
    assert conc == pytest.approx(1.0 / 0.5, rel=1e-6)


def test_concentration_q_uniform_equals_one() -> None:
    """Conc(q) = 1 when all |r| are equal — uniform mass distribution."""
    rng = np.random.default_rng(7)
    n = 500
    abs_resid = np.ones(n, dtype=np.float64)
    t1c = rng.random(n)
    # All voxels have equal |r|, so mass in top-q% = q → Conc = 1.
    assert concentration_q(abs_resid, t1c, q=0.05) == pytest.approx(1.0, rel=1e-6)


def test_concentration_q_empty_region_is_nan() -> None:
    """concentration_q returns NaN for zero-length inputs."""
    result = concentration_q(np.array([], dtype=np.float64), np.array([], dtype=np.float64), q=0.05)
    assert np.isnan(result)


# ---------------------------------------------------------------------------
# E[Conc(q)] = 1 under independence
# ---------------------------------------------------------------------------


def test_expected_conc_is_one_under_independence() -> None:
    """E[Conc(q)] converges to 1.0 when |r| and T1c are independent.

    Under independence, the top-q fraction of |r| is a random subset of R, so
    the expected mass concentration is 1.  We estimate this empirically with
    many random draws and check |mean - 1| < 0.1 (loose tolerance for speed).
    """
    rng = np.random.default_rng(42)
    n = 2000
    q = 0.05
    n_trials = 200

    conc_vals = []
    for _ in range(n_trials):
        abs_resid = rng.random(n).astype(np.float64)
        t1c = rng.random(n).astype(np.float64)
        conc_vals.append(concentration_q(abs_resid, t1c, q=q))

    mean_conc = float(np.mean(conc_vals))
    assert abs(mean_conc - 1.0) < 0.1, f"E[Conc({q})] = {mean_conc:.4f}, expected ~1.0"


# ---------------------------------------------------------------------------
# Shuffle null — E[ρ_S] = 0 and E[Conc(q)] = 1 analytically
# ---------------------------------------------------------------------------


def test_shuffle_null_rho_mean_near_zero() -> None:
    """Shuffle-null mean Spearman ρ should be ≈ 0 (analytic expectation)."""
    rng = np.random.default_rng(1)
    n = 1000
    abs_resid_brain = rng.random(n)
    t1c_brain = rng.random(n)
    region_in_brain = np.ones(n, dtype=bool)

    result = _shuffle_null(
        abs_resid_brain,
        t1c_brain,
        region_in_brain,
        n_shuffle=200,
        rng=rng,
        q=0.05,
        domain="R",
    )
    rho_mean = result["rho_mean"]
    assert abs(rho_mean) < 0.1, f"Shuffle-null E[ρ_S] = {rho_mean:.4f}, expected ~0"


def test_shuffle_null_conc_mean_near_one() -> None:
    """Shuffle-null mean Conc(5%) should be ≈ 1 (analytic expectation)."""
    rng = np.random.default_rng(2)
    n = 1000
    abs_resid_brain = rng.random(n)
    t1c_brain = rng.random(n)
    region_in_brain = np.ones(n, dtype=bool)

    result = _shuffle_null(
        abs_resid_brain,
        t1c_brain,
        region_in_brain,
        n_shuffle=200,
        rng=rng,
        q=0.05,
        domain="R",
    )
    conc_mean = result["conc_mean"]
    assert abs(conc_mean - 1.0) < 0.15, f"Shuffle-null E[Conc(5%)] = {conc_mean:.4f}, expected ~1"


# ---------------------------------------------------------------------------
# Shuffle convergence check
# ---------------------------------------------------------------------------


def test_shuffle_convergence_keys() -> None:
    """shuffle_convergence_check returns the expected dict structure."""
    rng_seed = 99
    n = 500
    rng = np.random.default_rng(rng_seed)
    abs_resid_R = rng.random(n)
    t1c_R = rng.random(n)

    result = shuffle_convergence_check(
        abs_resid_R,
        t1c_R,
        n_list=(10, 50),
        q=0.05,
        rng_seed=rng_seed,
    )

    assert set(result.keys()) == {10, 50}
    for _, stats in result.items():
        assert set(stats.keys()) == {"rho_mean", "rho_std", "conc_mean", "conc_std"}
        assert np.isfinite(stats["rho_mean"])
        assert np.isfinite(stats["conc_mean"])
        assert stats["rho_std"] >= 0.0
        assert stats["conc_std"] >= 0.0


# ---------------------------------------------------------------------------
# Intensity decile means
# ---------------------------------------------------------------------------


def test_intensity_decile_means_shape() -> None:
    """_intensity_decile_means returns a 1D array of length n_deciles."""
    rng = np.random.default_rng(3)
    n = 1000
    n_deciles = 10
    abs_resid_R = rng.random(n)
    t1c_R = rng.random(n)

    out = _intensity_decile_means(abs_resid_R, t1c_R, n_deciles=n_deciles)
    assert len(out) == n_deciles
    assert np.all(np.isfinite(out))


def test_intensity_decile_means_empty_region() -> None:
    """Empty inputs return a NaN array of length n_deciles (not an error)."""
    n_deciles = 10
    out = _intensity_decile_means(
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
        n_deciles=n_deciles,
    )
    # Empty input → NaN array of length n_deciles (see implementation).
    assert len(out) == n_deciles
    assert np.all(np.isnan(out))


# ---------------------------------------------------------------------------
# Conc(q) — realised denominator avoids tie bias
# ---------------------------------------------------------------------------


def test_concentration_q_uses_realised_denominator() -> None:
    """The denominator fraction uses ceil(q·n)/n, not nominal q.

    With n=7 and q=0.05, ceil(0.05 × 7) = 1 voxel selected.
    Realised fraction = 1/7 ≈ 0.143.  If top-1 holds all mass:
    Conc = 1 / (1/7) = 7.0  (not 1/0.05 = 20).
    """
    n = 7
    abs_resid = np.zeros(n, dtype=np.float64)
    abs_resid[-1] = 1.0  # all mass in top-1 voxel
    t1c = np.ones(n, dtype=np.float64)

    conc = concentration_q(abs_resid, t1c, q=0.05)
    # Realised denominator: ceil(0.05 * 7) = 1; fraction = 1/7.
    expected = 1.0 / (1 / n)
    assert conc == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# D1 regression: n_pairs > 0 even when VENA covers multiple NFEs
# ---------------------------------------------------------------------------


def _make_per_scan_df(
    vena_method: str,
    competitor: str,
    n_patients: int,
    vena_nfes: list[int],
    comp_nfes: list[int],
) -> pd.DataFrame:
    """Build a minimal per_scan_df for D1 regression testing.

    Creates one row per (patient, method, nfe) combination under ring="A"
    and condition="C-noT", with random but finite metric values.
    """
    rng = np.random.default_rng(0)
    patient_ids = [f"P{i:03d}" for i in range(n_patients)]

    rows = []
    for pid in patient_ids:
        for nfe in vena_nfes:
            rows.append(
                {
                    "scan_id": f"{pid}_UCSF_{vena_method}_nfe{nfe}",
                    "patient_id": pid,
                    "cohort": "UCSF-PDGM",
                    "ring": "A",
                    "condition": "C-noT",
                    "method": vena_method,
                    "nfe": nfe,
                    "rho_s": float(rng.uniform(0.0, 0.4)),
                    "conc_01": float(rng.uniform(1.0, 5.0)),
                    "conc_05": float(rng.uniform(1.0, 3.5)),
                    "conc_10": float(rng.uniform(1.0, 2.5)),
                    "delta_brain_rho": float(rng.uniform(-0.1, 0.1)),
                    "delta_brain_conc05": float(rng.uniform(-0.2, 0.2)),
                    "delta_R_rho": float(rng.uniform(-0.1, 0.1)),
                    "delta_R_conc05": float(rng.uniform(-0.2, 0.2)),
                }
            )
        for nfe in comp_nfes:
            rows.append(
                {
                    "scan_id": f"{pid}_UCSF_{competitor}_nfe{nfe}",
                    "patient_id": pid,
                    "cohort": "UCSF-PDGM",
                    "ring": "A",
                    "condition": "C-noT",
                    "method": competitor,
                    "nfe": nfe,
                    "rho_s": float(rng.uniform(0.1, 0.5)),
                    "conc_01": float(rng.uniform(1.0, 4.0)),
                    "conc_05": float(rng.uniform(1.0, 3.0)),
                    "conc_10": float(rng.uniform(1.0, 2.0)),
                    "delta_brain_rho": float(rng.uniform(-0.1, 0.1)),
                    "delta_brain_conc05": float(rng.uniform(-0.2, 0.2)),
                    "delta_R_rho": float(rng.uniform(-0.1, 0.1)),
                    "delta_R_conc05": float(rng.uniform(-0.2, 0.2)),
                }
            )
    return pd.DataFrame(rows)


def test_n_pairs_positive_for_c0_identity() -> None:
    """D1 regression: n_pairs > 0 for C0-Identity when VENA has nfes=[1,5].

    Root cause of the original bug: smoke config has ``nfes=[1,5]``, so after
    ``collapse_to_patient`` VENA has 2 rows per patient (one per NFE) while
    C0-Identity has 1 row per patient.  ``paired_wilcoxon`` receives a VENA
    Series of length 2×n and a C0 Series of length n → shape mismatch →
    ValueError → caught → n_pairs=0.

    The fix is ``_filter_to_selection_nfe``: VENA is sliced to nfe=5 (its
    selection NFE), C0 is sliced to nfe=1 (its selection NFE), so both Series
    have exactly n rows before ``.loc[common]``.
    """
    vena_method = "VENA-S1-v3b-rw"
    competitor = "C0-Identity"
    n_patients = 10  # small for unit speed; real run uses 66

    # Reproduce the smoke scenario: VENA gets nfe=[1,5], C0 only nfe=[1].
    df = _make_per_scan_df(vena_method, competitor, n_patients, vena_nfes=[1, 5], comp_nfes=[1])

    _, test_results = aggregate_patient_tests(df, vena_method=vena_method)

    # Sanity: we got at least one result for this competitor.
    c0_results = [r for r in test_results if r.competitor == competitor]
    assert len(c0_results) > 0, "No WilcoxonTestResult produced for C0-Identity"

    for r in c0_results:
        assert r.n_pairs > 0, (
            f"D1 regression: n_pairs=0 for C0-Identity / {r.stat_name}. "
            "The _filter_to_selection_nfe fix has regressed."
        )


def test_n_pairs_positive_for_c5_rflow() -> None:
    """D1 regression: n_pairs > 0 for C5-T1C-RFlow (both methods at nfe=[1,5]).

    Both VENA and C5 have selection_nfe=5, so after filtering both arms are
    reduced to nfe=5 only → exactly n unique patient_id rows each → no shape
    mismatch.  This test guards against regressions where the filter returns
    an empty DataFrame for C5.
    """
    vena_method = "VENA-S1-v3b-rw"
    competitor = "C5-T1C-RFlow"
    n_patients = 10

    df = _make_per_scan_df(vena_method, competitor, n_patients, vena_nfes=[1, 5], comp_nfes=[1, 5])

    _, test_results = aggregate_patient_tests(df, vena_method=vena_method)

    c5_results = [r for r in test_results if r.competitor == competitor]
    assert len(c5_results) > 0, "No WilcoxonTestResult produced for C5-T1C-RFlow"

    for r in c5_results:
        assert r.n_pairs > 0, (
            f"D1 regression: n_pairs=0 for C5-T1C-RFlow / {r.stat_name}. "
            "The _filter_to_selection_nfe fix has regressed."
        )


def test_holm_families_partitioned_by_role() -> None:
    """Holm families follow registry.method_role, not "every present method".

    2026-07-19 fix: the competitor family (role="family") and the ablation family
    (role="ablation") are Holm-corrected SEPARATELY; supplementary methods are
    reported but never tested; the VENA reference arm is never tested against
    itself.  The pre-fix behaviour lumped all 15 non-VENA methods into one
    30-test family.
    """
    vena = "VENA-S1-v3b-rw"
    # (method, selection_nfe, expected role)
    methods = [
        (vena, 5),
        ("C0-Identity", 1),  # competitor
        ("C5-T1C-RFlow", 5),  # competitor
        ("VENA-S1-v3a", 5),  # ablation
        ("VENA-S3-LPL-b2c", 5),  # ablation
        ("C1-pGAN-t2", 1),  # supplementary
        ("C3-SynDiff-t2", 4),  # supplementary
    ]
    rng = np.random.default_rng(1)
    rows = []
    for pid in [f"P{i:03d}" for i in range(12)]:
        for method, nfe in methods:
            rows.append(
                {
                    "scan_id": f"{pid}_{method}_nfe{nfe}",
                    "patient_id": pid,
                    "cohort": "UCSF-PDGM",
                    "ring": "A",
                    "condition": "C-noT",
                    "method": method,
                    "nfe": nfe,
                    "rho_s": float(rng.uniform(0.0, 0.5)),
                    "conc_01": float(rng.uniform(1.0, 4.0)),
                    "conc_05": float(rng.uniform(1.0, 3.0)),
                    "conc_10": float(rng.uniform(1.0, 2.0)),
                    "delta_brain_rho": 0.0,
                    "delta_brain_conc05": 0.0,
                    "delta_R_rho": 0.0,
                    "delta_R_conc05": 0.0,
                }
            )
    df = pd.DataFrame(rows)
    _, results = aggregate_patient_tests(df, vena_method=vena)

    families: dict[str, set[str]] = {}
    for r in results:
        families.setdefault(r.competitor, set()).add(r.family)

    assert families.get("C0-Identity") == {"competitor"}
    assert families.get("C5-T1C-RFlow") == {"competitor"}
    assert families.get("VENA-S1-v3a") == {"ablation"}
    assert families.get("VENA-S3-LPL-b2c") == {"ablation"}
    # Supplementary methods and the VENA reference arm are never tested.
    assert "C1-pGAN-t2" not in families
    assert "C3-SynDiff-t2" not in families
    assert vena not in families
    # Each tested method contributes exactly 2 tests (rho_s + conc_05).
    for method in ("C0-Identity", "C5-T1C-RFlow", "VENA-S1-v3a", "VENA-S3-LPL-b2c"):
        assert sum(1 for r in results if r.competitor == method) == 2
