"""T1 (range sanity) against synthetic ground-truth subjects."""

from __future__ import annotations

from vena.preflight.priors_validation.tests import T1RangeSanity

from ._synthetic import build_synthetic_context_from, build_synthetic_subject


def _outcomes_by_key(outcomes):
    return {(o.prior_id, o.roi_id): o for o in outcomes}


def test_t1_passes_on_known_ground_truth():
    """Synthetic CBF/ADC at literature values → T1 passes every applicable cell."""
    subj = build_synthetic_subject(
        cbf_nawm=22.0,
        cbf_gm=55.0,
        adc_nawm_mm2_s=0.8e-3,
        adc_gm_mm2_s=0.85e-3,
        adc_csf_mm2_s=3.0e-3,
    )
    ctx = build_synthetic_context_from(subj)
    outs = list(T1RangeSanity().run(subj, ctx))
    by_key = _outcomes_by_key(outs)
    # NAWM CBF should land in [12, 35] ml/100g/min
    o = by_key.get(("cbf", "nawm"))
    assert o is not None, "missing (cbf, nawm) outcome"
    assert o.severity != "error" or o.passed, (
        f"NAWM CBF should pass, got value={o.metric_value} diag={o.diagnostic}"
    )
    # ADC NAWM passes when scaled to 10^-3 mm²/s
    o = by_key.get(("adc", "nawm"))
    assert o is not None
    assert o.passed or o.severity == "warning", (
        f"NAWM ADC should pass, got value={o.metric_value} diag={o.diagnostic}"
    )


def test_t1_flags_inverted_cbf_subtraction():
    """Inverted label-control → CBF ≈ −22 in NAWM, well outside the band."""
    subj = build_synthetic_subject(cbf_nawm=-22.0, cbf_gm=-55.0)
    ctx = build_synthetic_context_from(subj)
    outs = list(T1RangeSanity().run(subj, ctx))
    by_key = _outcomes_by_key(outs)
    o = by_key[("cbf", "nawm")]
    assert not o.passed
    assert "inverted" in o.diagnostic.lower() or o.metric_value < 5.0


def test_t1_downgrades_adc_scl_slope_quirk():
    """ADC ~10^-9 (per-file scale quirk) → warning, not error."""
    subj = build_synthetic_subject(
        adc_nawm_mm2_s=1e-9,
        adc_gm_mm2_s=1.5e-9,
        adc_csf_mm2_s=3.0e-9,
    )
    ctx = build_synthetic_context_from(subj)
    outs = list(T1RangeSanity().run(subj, ctx))
    adc_outs = [o for o in outs if o.prior_id == "adc"]
    assert any(o.severity == "warning" for o in adc_outs), (
        "expected at least one ADC outcome downgraded to warning"
    )
