"""Unit tests for the V3 audit ``decision.json`` schema + write/read."""

from __future__ import annotations

import json

import pytest

from vena.preflight.normalization_audit import (
    NormalizationAuditDecisionV1,
    PerVariantMetrics,
    assert_normalization_audit_decision_valid,
    write_decision_json,
)
from vena.preflight.normalization_audit.decision import (
    DECISION_PRODUCER,
    DECISION_SCHEMA_VERSION,
    SmokeCohortVerdict,
)

pytestmark = pytest.mark.unit


def _make_metrics(
    vid: str,
    mae_whole: float,
    mae_et: float,
    image_ratio: float,
    latent_ratio: float,
    kl_max: float,
    psnr_whole: float,
) -> PerVariantMetrics:
    return PerVariantMetrics(
        variant_id=vid,
        variant_version="0.1.0",
        params={"upper": 99.5, "clip": True},
        n_patients=30,
        mae_whole=mae_whole,
        mae_et=mae_et,
        mae_netc=0.005,
        mae_ed=0.005,
        mae_bnwt=0.005,
        psnr_whole_db=psnr_whole,
        psnr_et_db=psnr_whole - 4.0,
        ssim_whole=0.98,
        image_mean_abs_diff_et=0.5,
        image_mean_abs_diff_bnwt=0.5 / max(image_ratio, 1e-6),
        image_signal_ratio_et_over_bnwt=image_ratio,
        latent_mean_abs_delta_et=0.3,
        latent_mean_abs_delta_bnwt=0.3 / max(latent_ratio, 1e-6),
        latent_signal_ratio_et_over_bnwt=latent_ratio,
        latent_mean_abs_t1c_et=0.7,
        latent_mean_abs_t1pre_et=0.4,
        kl_divergence_per_modality={"t1c": kl_max, "t1pre": 0.0, "t2": 0.0, "flair": 0.0},
        kl_divergence_max=kl_max,
        image_signal_ratio_large_et_stratum=image_ratio,
        latent_signal_ratio_large_et_stratum=latent_ratio,
        n_patients_large_et=20,
        passes_c1_mae_whole=mae_whole <= 0.010,
        passes_c2_mae_et=mae_et <= 0.015,
        passes_c3_kl=kl_max <= 1.0,
        passes_c4_image_signal=image_ratio >= 1.5,
        passes_c5_latent_signal=latent_ratio >= 1.3,
        passes_c7_psnr_whole=psnr_whole >= 35.0,
        passes_all=(
            mae_whole <= 0.010
            and mae_et <= 0.015
            and kl_max <= 1.0
            and image_ratio >= 1.5
            and latent_ratio >= 1.3
            and psnr_whole >= 35.0
        ),
    )


def test_decision_round_trips(tmp_path) -> None:
    metrics = {
        "V0": _make_metrics("V0", 0.0041, 0.0050, 0.62, 1.10, 0.00, 38.5),
        "V4": _make_metrics("V4", 0.0070, 0.0090, 2.20, 1.55, 0.42, 36.5),
    }
    decision = NormalizationAuditDecisionV1(
        produced_at="2026-06-22T12:00:00+00:00",
        producer=DECISION_PRODUCER,
        schema_version=DECISION_SCHEMA_VERSION,
        git_sha="abc1234",
        vae_checkpoint="/tmp/vae.pt",
        vae_checkpoint_sha256="deadbeef",
        main_cohort="UCSF-PDGM",
        n_patients_main=30,
        patient_seed=1337,
        variants_tested=["V0", "V4"],
        metrics_per_variant=metrics,
        acceptance_thresholds={
            "c1_mae_whole_max": 0.010,
            "c2_mae_et_max": 0.015,
            "c3_kl_max_nats": 1.0,
            "c4_image_signal_ratio_min": 1.5,
            "c5_latent_signal_ratio_min": 1.3,
            "c7_psnr_whole_min_db": 35.0,
        },
        winner="V4",
        winner_rationale="V4 passes all C1..C7.",
        fallback_used=False,
        smoke_cohorts=[
            SmokeCohortVerdict(
                cohort="BraTS-GLI",
                n_patients=5,
                mae_whole=0.007,
                mae_et=0.009,
                image_signal_ratio_et_over_bnwt=2.0,
                passes=True,
            )
        ],
        next_action="re_encode_all_cohorts",
    )
    out = tmp_path / "decision.json"
    write_decision_json(out, decision)
    parsed = assert_normalization_audit_decision_valid(out)
    assert parsed.winner == "V4"
    assert parsed.metrics_per_variant["V4"].passes_all
    assert parsed.smoke_cohorts[0].passes


def test_decision_invalid_next_action_raises(tmp_path) -> None:
    blob = {
        "schema_version": "1.0.0",
        "produced_at": "2026-06-22T12:00:00+00:00",
        "producer": DECISION_PRODUCER,
        "vae_checkpoint": "/tmp/vae.pt",
        "vae_checkpoint_sha256": "deadbeef",
        "main_cohort": "UCSF-PDGM",
        "n_patients_main": 0,
        "patient_seed": 1337,
        "variants_tested": [],
        "metrics_per_variant": {},
        "acceptance_thresholds": {},
        "winner": None,
        "winner_rationale": "",
        "fallback_used": True,
        "smoke_cohorts": [],
        "next_action": "INVALID_ACTION",
    }
    p = tmp_path / "decision.json"
    p.write_text(json.dumps(blob))
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        assert_normalization_audit_decision_valid(p)


def test_metrics_pass_all_logic() -> None:
    """A variant that passes one criterion but not all must have passes_all=False."""
    m_partial = _make_metrics("V_x", 0.005, 0.013, 1.2, 1.4, 0.5, 40.0)  # C4 fails
    assert not m_partial.passes_all

    m_full = _make_metrics("V_y", 0.005, 0.013, 1.7, 1.4, 0.5, 40.0)
    assert m_full.passes_all
