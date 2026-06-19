"""Unit tests for :class:`vena.model.fm.lpl.config.LplConfig`.

Exercises the Pydantic validator (key-set agreement between ``A`` / ``w_l``
/ ``outlier_k`` and ``region_set`` / ``alpha`` / ``p``) plus the YAML
round-trip (``from_yaml``) on a smoke config.
"""

from __future__ import annotations

import pytest
import yaml

from vena.model.fm.lpl import LplConfig

pytestmark = pytest.mark.unit


def test_defaults_validate() -> None:
    cfg = LplConfig()
    assert cfg.A == [2, 5]
    assert set(cfg.w_l) == {2, 5}
    assert cfg.region_set == ["wt", "notwt"]


def test_wl_keys_must_match_A() -> None:
    with pytest.raises(ValueError, match="w_l keys"):
        LplConfig(A=[2, 5], w_l={2: 1.0}, outlier_k={2: 5.0, 5: 5.0})


def test_outlier_k_keys_must_match_A() -> None:
    with pytest.raises(ValueError, match="outlier_k keys"):
        LplConfig(A=[2, 5], w_l={2: 1.0, 5: 2.0}, outlier_k={2: 5.0})


def test_alpha_keys_must_match_region_set() -> None:
    with pytest.raises(ValueError, match="alpha keys"):
        LplConfig(alpha={"wt": 1.0})


def test_p_must_be_1_2_or_3() -> None:
    with pytest.raises(ValueError, match=r"must be 1, 2, or 3"):
        LplConfig(p={"wt": 4, "notwt": 2})


def test_t_min_range() -> None:
    with pytest.raises(ValueError, match=r"t_min must be in"):
        LplConfig(t_min=1.5)


def test_compute_placement_b_rejected() -> None:
    """Variant B is deferred to a follow-up PR and must fail loudly."""
    with pytest.raises(ValueError, match="cross-device"):
        LplConfig(compute_placement="b")


def test_grad_checkpoint_segments_one_rejected() -> None:
    with pytest.raises(ValueError, match=r"grad_checkpoint_segments"):
        LplConfig(grad_checkpoint_segments=1)


def test_from_yaml_round_trip(tmp_path) -> None:
    path = tmp_path / "lpl.yaml"
    payload = {
        "A": [2, 5],
        "w_l": {"2": 1.0, "5": 2.0},  # str keys → coerced to int
        "t_min": 0.75,
        "lambda_img": 0.5,
        "alpha": {"wt": 2.0, "notwt": 3.0},
        "p": {"wt": 1, "notwt": 3},
        "outlier_k": {"2": 5.0, "5": 5.0},
        "soft_region": True,
        "grad_checkpoint_segments": 2,
        "compute_placement": "a",
        "region_set": ["wt", "notwt"],
    }
    path.write_text(yaml.safe_dump(payload))
    cfg = LplConfig.from_yaml(path)
    assert cfg.t_min == 0.75
    assert cfg.p == {"wt": 1, "notwt": 3}
    assert cfg.soft_region is True
    assert cfg.grad_checkpoint_segments == 2
    assert cfg.w_l == {2: 1.0, 5: 2.0}  # int keys after normalisation


def test_extra_keys_forbidden() -> None:
    """Frozen Pydantic + ``extra='forbid'`` catches typos in production YAMLs."""
    with pytest.raises(ValueError, match="extra"):
        LplConfig.model_validate(
            {"A": [2], "w_l": {2: 1.0}, "outlier_k": {2: 5.0}, "unknown_key": 1}
        )


# --------------------------------------------------------------------------
# Schedule field (lambda_img schedule)
# --------------------------------------------------------------------------


def test_schedule_field_defaults_to_none() -> None:
    cfg = LplConfig()
    assert cfg.schedule is None


def test_schedule_field_round_trip(tmp_path) -> None:
    """``loss.lpl.schedule`` YAML block round-trips into ``LplConfig.schedule``."""
    path = tmp_path / "lpl.yaml"
    payload = {
        "A": [2],
        "w_l": {2: 1.0},
        "outlier_k": {2: 5.0},
        "schedule": {
            "kind": "linear",
            "warmup_epochs": 30,
            "lambda_min": 0.0,
            "lambda_max": 1.0,
        },
    }
    path.write_text(yaml.safe_dump(payload))
    cfg = LplConfig.from_yaml(path)
    assert cfg.schedule is not None
    assert cfg.schedule.kind == "linear"
    assert cfg.schedule.warmup_epochs == 30
    assert cfg.schedule.lambda_max == 1.0


# --------------------------------------------------------------------------
# from_decision override surface
# --------------------------------------------------------------------------


@pytest.fixture
def decision_payload(tmp_path):
    """Write a minimal valid ``decoder_lpl_profile`` decision.json + return its path.

    Mirrors the 2026-06-18T20:18Z post-fix preflight: ``A=[2,3]``,
    ``w_l={2: 0.66, 3: 1.34}``, ``t_min=0.4``, ``outlier_k=5``, region recipe
    ``(2, 3)``, no per-cohort overrides.
    """
    import json

    p = tmp_path / "decision.json"
    p.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "produced_at": "2026-06-18T20:18:00Z",
                "producer": "vena.preflight.decoder_lpl_profile:1.0",
                "n_patients_run": 90,
                "patients_per_cohort": {"UCSF-PDGM": 3},
                "A_recommended": [2, 3],
                "w_l": {"2": 0.66, "3": 1.34},
                "t_min": 0.4,
                "outlier_k": {"2": 5.0, "3": 5.0},
                "region_recipe": {
                    "alpha_wt": 2.0,
                    "alpha_notwt": 3.0,
                    "soft_region": False,
                    "per_cohort_overrides": {},
                },
                "allowed_variants": ["v0", "v1", "v2", "v3", "v4"],
                "v4_brain_mask_status": "ok",
            }
        )
    )
    return p


def test_from_decision_default_uses_preflight_A(decision_payload) -> None:
    cfg = LplConfig.from_decision(decision_payload)
    assert cfg.A == [2, 3]
    assert cfg.w_l == {2: 0.66, 3: 1.34}
    assert cfg.outlier_k == {2: 5.0, 3: 5.0}
    assert cfg.alpha == {"wt": 2.0, "notwt": 3.0}


def test_from_decision_A_override_no_w_l_raises(decision_payload) -> None:
    with pytest.raises(ValueError, match="w_l_override and outlier_k_override"):
        LplConfig.from_decision(decision_payload, A_override=[2, 5])


def test_from_decision_full_override_K5(decision_payload) -> None:
    """K=5 ablation: override A=[2,5] with matching w_l and outlier_k."""
    cfg = LplConfig.from_decision(
        decision_payload,
        A_override=[2, 5],
        w_l_override={2: 1.0, 5: 2.0},
        outlier_k_override={2: 5.0, 5: 5.0},
    )
    assert cfg.A == [2, 5]
    assert cfg.w_l == {2: 1.0, 5: 2.0}
    assert cfg.outlier_k == {2: 5.0, 5: 5.0}
    # Region recipe still inherited from preflight (no alpha_override).
    assert cfg.alpha == {"wt": 2.0, "notwt": 3.0}


def test_from_decision_alpha_override_standard_LPL(decision_payload) -> None:
    """Standard LPL arm: α=(1,1) overrides the preflight's (2,3)."""
    cfg = LplConfig.from_decision(decision_payload, alpha_override={"wt": 1.0, "notwt": 1.0})
    assert cfg.alpha == {"wt": 1.0, "notwt": 1.0}


def test_from_decision_schedule_pass_through(decision_payload) -> None:
    from vena.model.fm.lpl import LambdaImgSchedule

    sched = LambdaImgSchedule(kind="linear", warmup_epochs=30, lambda_min=0.0, lambda_max=1.0)
    cfg = LplConfig.from_decision(decision_payload, schedule=sched)
    assert cfg.schedule is sched
