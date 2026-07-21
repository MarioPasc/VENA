"""Unit tests for routines.validation.studies.paired_fidelity_study.

All tests use synthetic fixtures — no real CSV, no GPU, no checkpoints.

pytestmark applies ``unit`` to the whole module so the tests run under
``-m "not slow and not gpu"``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from routines.validation.studies._shared import filter_to_selection_nfe
from routines.validation.studies.paired_fidelity_study import (
    _BOUNDED_METRICS,
    _N_INPUTS,
    _ORACLE_METHODS,
    _canonical_method_order,
)

from vena.validation.registry import (
    ABLATION_FAMILY,
    COMPETITOR_FAMILY,
    SELECTION_NFE,
    SUPPLEMENTARY,
    VENA_HEADLINE,
)
from vena.validation.stats import MCID

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Test 1 — Selection-NFE reduction is symmetric (the D1 bug guard)
# ---------------------------------------------------------------------------


def _make_multi_nfe_df(
    methods: list[tuple[str, str]],
    nfes: list[int],
    n_patients: int = 5,
) -> pd.DataFrame:
    """Build a minimal per-scan DataFrame with rows at every NFE for each method.

    Parameters
    ----------
    methods :
        List of (method_key, cohort) pairs.
    nfes :
        NFE values to include.
    n_patients :
        Number of distinct patient IDs per (method, NFE).

    Returns
    -------
    pd.DataFrame
        Synthetic tidy DataFrame with columns: method, cohort, ring, nfe,
        patient_id, mae_brain.
    """
    rows = []
    for method, cohort in methods:
        for nfe in nfes:
            for pid in range(n_patients):
                rows.append(
                    {
                        "method": method,
                        "cohort": cohort,
                        "ring": "A",
                        "nfe": nfe,
                        "patient_id": f"P{pid:03d}",
                        "mae_brain": 0.1 + nfe * 0.001,
                    }
                )
    return pd.DataFrame(rows)


def test_selection_nfe_symmetry_vena() -> None:
    """VENA-S1-v3b-rw (sel NFE=5) is reduced to only NFE=5 rows."""
    method = VENA_HEADLINE  # "VENA-S1-v3b-rw", sel NFE=5
    nfe_target = SELECTION_NFE[method]
    df = _make_multi_nfe_df(
        methods=[(method, "UCSF-PDGM")],
        nfes=[1, 2, 4, 5, 10],
    )
    df_method = df[df["method"] == method].copy()
    reduced = filter_to_selection_nfe(df_method, method)

    assert not reduced.empty, "reduced must not be empty"
    assert set(reduced["nfe"].unique()) == {nfe_target}, (
        f"VENA headline must be reduced to nfe={nfe_target}; "
        f"got nfes={set(reduced['nfe'].unique())}"
    )


def test_selection_nfe_symmetry_competitor() -> None:
    """C0-Identity (sel NFE=1) is reduced to only NFE=1 rows."""
    method = "C0-Identity"
    nfe_target = SELECTION_NFE[method]  # 1
    assert nfe_target == 1, "C0-Identity selection NFE must be 1"

    df = _make_multi_nfe_df(
        methods=[(method, "UCSF-PDGM")],
        nfes=[1, 5, 10],
    )
    df_method = df[df["method"] == method].copy()
    reduced = filter_to_selection_nfe(df_method, method)

    assert not reduced.empty
    assert set(reduced["nfe"].unique()) == {nfe_target}, (
        f"C0-Identity must be reduced to nfe={nfe_target}; got nfes={set(reduced['nfe'].unique())}"
    )


def test_selection_nfe_symmetry_both_arms_independent() -> None:
    """VENA and a competitor are each reduced to THEIR OWN sel NFE, not each other's.

    This guards against the D1 bug (HANDOFF §6) where one reducer was applied
    asymmetrically, collapsing both arms to the same NFE value.
    """
    vena_method = VENA_HEADLINE  # sel NFE=5
    comp_method = "C0-Identity"  # sel NFE=1
    assert SELECTION_NFE[vena_method] != SELECTION_NFE[comp_method], (
        "Test requires two methods with different selection NFEs"
    )

    combined_df = _make_multi_nfe_df(
        methods=[(vena_method, "UCSF-PDGM"), (comp_method, "UCSF-PDGM")],
        nfes=[1, 2, 5, 10],
    )

    vena_df = combined_df[combined_df["method"] == vena_method].copy()
    comp_df = combined_df[combined_df["method"] == comp_method].copy()

    vena_reduced = filter_to_selection_nfe(vena_df, vena_method)
    comp_reduced = filter_to_selection_nfe(comp_df, comp_method)

    vena_nfes = set(vena_reduced["nfe"].unique())
    comp_nfes = set(comp_reduced["nfe"].unique())

    assert vena_nfes == {SELECTION_NFE[vena_method]}, (
        f"VENA must be at nfe={SELECTION_NFE[vena_method]}; got {vena_nfes}"
    )
    assert comp_nfes == {SELECTION_NFE[comp_method]}, (
        f"C0 must be at nfe={SELECTION_NFE[comp_method]}; got {comp_nfes}"
    )
    # The key invariant: the two reduced sets use different NFE values.
    assert vena_nfes != comp_nfes, (
        "Symmetric reduction must produce different NFEs for methods with "
        "different selection NFEs — same NFE signals the asymmetric bug"
    )


# ---------------------------------------------------------------------------
# Test 2 — Holm family partition
# ---------------------------------------------------------------------------


def test_competitor_family_size() -> None:
    """COMPETITOR_FAMILY has exactly 8 members (pre-registered)."""
    assert len(COMPETITOR_FAMILY) == 8, (
        f"COMPETITOR_FAMILY must have 8 members; got {len(COMPETITOR_FAMILY)}"
    )


def test_ablation_family_size() -> None:
    """ABLATION_FAMILY has exactly 3 members (pre-registered)."""
    assert len(ABLATION_FAMILY) == 3, (
        f"ABLATION_FAMILY must have 3 members; got {len(ABLATION_FAMILY)}"
    )


def test_supplementary_not_in_any_family() -> None:
    """Supplementary methods must not appear in competitor or ablation families."""
    comp_set = set(COMPETITOR_FAMILY)
    abl_set = set(ABLATION_FAMILY)
    for m in SUPPLEMENTARY:
        assert m not in comp_set, f"Supplementary {m!r} must not be in COMPETITOR_FAMILY"
        assert m not in abl_set, f"Supplementary {m!r} must not be in ABLATION_FAMILY"


def test_vena_headline_not_in_families() -> None:
    """VENA headline is not in competitor or ablation families."""
    assert VENA_HEADLINE not in COMPETITOR_FAMILY, (
        f"VENA headline {VENA_HEADLINE!r} must not be in COMPETITOR_FAMILY"
    )
    assert VENA_HEADLINE not in ABLATION_FAMILY, (
        f"VENA headline {VENA_HEADLINE!r} must not be in ABLATION_FAMILY"
    )


def test_families_disjoint() -> None:
    """Competitor and ablation families must not overlap."""
    overlap = set(COMPETITOR_FAMILY) & set(ABLATION_FAMILY)
    assert len(overlap) == 0, f"Families must be disjoint; overlap = {overlap}"


def test_canonical_order_covers_all_16() -> None:
    """_canonical_method_order returns exactly 16 methods, one per METHOD_SPECS entry."""
    from vena.validation.registry import METHOD_SPECS

    order = _canonical_method_order()
    assert len(order) == len(METHOD_SPECS), (
        f"canonical order must cover all {len(METHOD_SPECS)} registered methods; got {len(order)}"
    )
    assert set(order) == {s.key for s in METHOD_SPECS}, (
        "canonical order must equal the set of all registered method keys"
    )


# ---------------------------------------------------------------------------
# Test 3 — Sub-MCID flag correctness
# ---------------------------------------------------------------------------


def _submcid(method_mean: float, v3brw_mean: float) -> bool:
    """Replicate the sub-MCID flag logic from PairedFidelityStudy.run()."""
    if np.isnan(method_mean) or np.isnan(v3brw_mean):
        return False
    return bool(abs(method_mean - v3brw_mean) < MCID)


def test_submcid_flag_below_threshold() -> None:
    """A difference of 0.009 < MCID=0.01 → submcid=True."""
    assert MCID == 0.01, "MCID must be 0.01 per pre-registration"
    # |0.109 - 0.100| = 0.009 < 0.01
    assert _submcid(0.109, 0.100) is True, "difference 0.009 < MCID=0.01 must yield submcid=True"


def test_submcid_flag_above_threshold() -> None:
    """A difference of 0.015 > MCID=0.01 → submcid=False."""
    # |0.115 - 0.100| = 0.015 > 0.01
    assert _submcid(0.115, 0.100) is False, "difference 0.015 > MCID=0.01 must yield submcid=False"


def test_submcid_flag_clearly_above_threshold() -> None:
    """A difference of 0.02 (double MCID) → submcid=False.

    Avoids the floating-point boundary: 0.110 − 0.100 ≈ 0.00999... < 0.01
    due to IEEE 754, making exact-boundary tests unreliable.  We test a
    value well above threshold instead.
    """
    # |0.120 - 0.100| = 0.02 >> MCID=0.01
    assert _submcid(0.120, 0.100) is False, "difference 0.02 > MCID=0.01 must yield submcid=False"


def test_submcid_flag_nan_returns_false() -> None:
    """NaN values produce False, not an exception."""
    assert _submcid(float("nan"), 0.100) is False
    assert _submcid(0.100, float("nan")) is False


def test_submcid_only_applied_to_bounded_metrics() -> None:
    """Only mae, ssim, ms_ssim carry the sub-MCID flag (psnr does not)."""
    assert "mae" in _BOUNDED_METRICS
    assert "ssim" in _BOUNDED_METRICS
    assert "ms_ssim" in _BOUNDED_METRICS
    assert "psnr" not in _BOUNDED_METRICS, (
        "psnr is not on a [0,1] scale; sub-MCID flag must not apply to it"
    )


# ---------------------------------------------------------------------------
# Test 4 — Oracle flag correctness
# ---------------------------------------------------------------------------


def test_oracle_methods_set() -> None:
    """Exactly the three mask-conditioned VENA variants are flagged as oracle."""
    expected = {"VENA-S1-v3b", "VENA-S1-v3b-rw", "VENA-S3-LPL-b2c"}
    assert _ORACLE_METHODS == expected, f"oracle methods must be {expected}; got {_ORACLE_METHODS}"


def test_oracle_vena_v3a_not_oracle() -> None:
    """VENA-S1-v3a (no-mask, no-oracle baseline) must not be in _ORACLE_METHODS."""
    assert "VENA-S1-v3a" not in _ORACLE_METHODS, (
        "VENA-S1-v3a has no mask conditioning; it must not be oracle"
    )


# ---------------------------------------------------------------------------
# Test 5 — n_inputs consistency with HUB
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,expected_n",
    [
        ("C0-Identity", 1),
        ("C1-pGAN-t1pre", 1),
        ("C2-ResViT", 3),
        ("C4-3D-DiT", 2),
        ("C5-T1C-RFlow", 2),
        ("VENA-S1-v3a", 3),
        ("VENA-S1-v3b-rw", 3),
    ],
)
def test_n_inputs_values(method: str, expected_n: int) -> None:
    """n_inputs matches the HUB §2.1 input-modality count for each method."""
    assert _N_INPUTS[method] == expected_n, (
        f"{method}: expected n_inputs={expected_n}, got {_N_INPUTS[method]}"
    )


def test_n_inputs_covers_all_16() -> None:
    """Every registered method has an entry in _N_INPUTS."""
    from vena.validation.registry import METHOD_SPECS

    for spec in METHOD_SPECS:
        assert spec.key in _N_INPUTS, f"_N_INPUTS missing entry for registered method {spec.key!r}"
