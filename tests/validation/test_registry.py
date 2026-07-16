"""Tests for vena.validation.registry.

Critical invariants:
- COMPETITOR_FAMILY has exactly 8 members (Holm family size).
- ABLATION_FAMILY has exactly 3 members.
- method_role returns the 4-way pre-registered role, not a 2-way name heuristic.
- Unknown methods return "supplementary" with a WARNING (fail-open for I/O).
- VENA_HEADLINE is the single reference arm of every paired test.
- SELECTION_NFE is populated at import (no load_partitions required).
"""

from __future__ import annotations

import logging

import pytest

pytestmark = pytest.mark.validation


# ---------------------------------------------------------------------------
# Family-size invariants
# ---------------------------------------------------------------------------


def test_competitor_family_has_eight_members() -> None:
    """Holm family size must be exactly 8 — pre-registration constraint."""
    from vena.validation.registry import COMPETITOR_FAMILY

    assert len(COMPETITOR_FAMILY) == 8, (
        f"Expected 8 competitor family members; got {len(COMPETITOR_FAMILY)}. "
        "Update the pre-registration if a new method is added."
    )


def test_ablation_family_has_three_members() -> None:
    """Ablation Holm family size must be exactly 3."""
    from vena.validation.registry import ABLATION_FAMILY

    assert len(ABLATION_FAMILY) == 3


def test_supplementary_has_four_members() -> None:
    from vena.validation.registry import SUPPLEMENTARY

    assert len(SUPPLEMENTARY) == 4


# ---------------------------------------------------------------------------
# Four-way role mapping
# ---------------------------------------------------------------------------


def test_method_role_vena_headline() -> None:
    from vena.validation.registry import method_role

    assert method_role("VENA-S1-v3b-rw") == "vena"


def test_method_role_ablation_members() -> None:
    from vena.validation.registry import ABLATION_FAMILY, method_role

    for m in ABLATION_FAMILY:
        assert method_role(m) == "ablation", f"{m} should be ablation"


def test_method_role_competitor_family_members() -> None:
    from vena.validation.registry import COMPETITOR_FAMILY, method_role

    for m in COMPETITOR_FAMILY:
        assert method_role(m) == "family", f"{m} should be family"


def test_method_role_supplementary_members() -> None:
    from vena.validation.registry import SUPPLEMENTARY, method_role

    for m in SUPPLEMENTARY:
        assert method_role(m) == "supplementary", f"{m} should be supplementary"


def test_method_role_vena_ablations_not_family() -> None:
    """Critical: headline and ablations must NOT be classified as family members.

    A startswith("VENA-") heuristic would pass the headline test but also
    classify ablations as "primary", silently enlarging the Holm family to
    n=12 instead of n=8.
    """
    from vena.validation.registry import method_role

    # Ablations must not be "family" — they have their own Holm correction.
    assert method_role("VENA-S1-v3b") == "ablation"
    assert method_role("VENA-S1-v3a") == "ablation"
    assert method_role("VENA-S3-LPL-b2c") == "ablation"

    # Supplementary single-source panels must not be "family".
    assert method_role("C1-pGAN-t2") == "supplementary"
    assert method_role("C1-pGAN-flair") == "supplementary"
    assert method_role("C3-SynDiff-t2") == "supplementary"
    assert method_role("C3-SynDiff-flair") == "supplementary"


def test_method_role_unknown_returns_supplementary_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unknown methods (e.g. BraTS-PED backfill) must not crash — fail-open."""
    from vena.validation.registry import method_role

    with caplog.at_level(logging.WARNING, logger="vena.validation.registry"):
        role = method_role("BRATS-PED-SomeNewMethod")

    assert role == "supplementary"
    assert any("Unknown method" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# VENA_HEADLINE constant
# ---------------------------------------------------------------------------


def test_vena_headline_is_the_region_weighted_model() -> None:
    from vena.validation.registry import VENA_HEADLINE

    assert VENA_HEADLINE == "VENA-S1-v3b-rw"


def test_vena_headline_role_is_vena() -> None:
    """The headline must have role 'vena', not 'family' or 'ablation'."""
    from vena.validation.registry import VENA_HEADLINE, method_role

    assert method_role(VENA_HEADLINE) == "vena"


# ---------------------------------------------------------------------------
# SELECTION_NFE populated at import (no I/O required)
# ---------------------------------------------------------------------------


def test_selection_nfe_populated_at_import() -> None:
    """SELECTION_NFE must not require load_partitions to be non-empty."""
    from vena.validation.registry import SELECTION_NFE

    assert len(SELECTION_NFE) >= 16  # 16 pre-registered methods


def test_selection_nfe_values_verified_on_disk() -> None:
    """NFE values verified on disk 2026-07-16 (SHARED_CONTRACTS §4)."""
    from vena.validation.registry import SELECTION_NFE

    # C0/C1-t1pre/C2/C7 → 1
    for m in ("C0-Identity", "C1-pGAN-t1pre", "C2-ResViT", "C7-3D-Latent-Pix2Pix"):
        assert SELECTION_NFE[m] == 1, f"{m}: expected nfe=1"
    # C1-t2/flair → 1
    for m in ("C1-pGAN-t2", "C1-pGAN-flair"):
        assert SELECTION_NFE[m] == 1
    # C3* → 4
    for m in ("C3-SynDiff-t1pre", "C3-SynDiff-t2", "C3-SynDiff-flair"):
        assert SELECTION_NFE[m] == 4, f"{m}: expected nfe=4"
    # C4/C5/VENA* → 5
    for m in (
        "C4-3D-DiT",
        "C5-T1C-RFlow",
        "VENA-S1-v3b-rw",
        "VENA-S1-v3b",
        "VENA-S1-v3a",
        "VENA-S3-LPL-b2c",
    ):
        assert SELECTION_NFE[m] == 5, f"{m}: expected nfe=5"
    # C6 → 1000
    assert SELECTION_NFE["C6-3D-LDDPM"] == 1000


# ---------------------------------------------------------------------------
# method_order / method_palette
# ---------------------------------------------------------------------------


def test_method_order_covers_all_specs() -> None:
    from vena.validation.registry import METHOD_SPECS, method_order

    assert set(method_order()) == {s.key for s in METHOD_SPECS}


def test_method_order_vena_first() -> None:
    from vena.validation.registry import VENA_HEADLINE, method_order

    assert method_order()[0] == VENA_HEADLINE


def test_method_palette_covers_all_specs() -> None:
    from vena.validation.registry import METHOD_SPECS, method_palette

    palette = method_palette()
    for spec in METHOD_SPECS:
        assert spec.key in palette, f"{spec.key} missing from palette"


def test_method_palette_colours_are_unique() -> None:
    """Each method gets a distinct colour (no copy-paste collision)."""
    from vena.validation.registry import method_palette

    colours = list(method_palette().values())
    assert len(colours) == len(set(colours))


# ---------------------------------------------------------------------------
# ring_of_cohort
# ---------------------------------------------------------------------------


def test_ring_of_cohort_prefers_h5_attr_over_empty_static() -> None:
    """With no partition loaded, ring_of_cohort falls back to h5_ring_attr."""
    from vena.validation.registry import ring_of_cohort

    # COHORT_RING is empty at import; h5_ring_attr provides the source of truth.
    assert ring_of_cohort("AnyNewCohort", h5_ring_attr="B") == "B"


def test_ring_of_cohort_raises_on_unknown_cohort_no_attr() -> None:
    from vena.validation.registry import ring_of_cohort

    with pytest.raises(KeyError, match="Unknown cohort"):
        ring_of_cohort("CompletelyUnknownCohort")
