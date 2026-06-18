"""Pre-flight gate tests for routines.fm.train.engine._assert_preflight_gates.

Builds a synthetic FMTrainRoutineConfig with augmentations enabled and
verifies the gate raises ``PreflightGateError`` in each documented failure
mode (missing path, missing allowlist entry).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from routines.fm.train.engine import (
    FMTrainRoutineConfig,
    _assert_preflight_gates,
    _load_preflight_decision,
)
from routines.fm.train.exceptions import PreflightGateError

pytestmark = pytest.mark.unit


def _build_minimal_cfg(
    *,
    tmp_path: Path,
    augmentation_config_path: Path | None,
    preflight_decision_path: Path | None,
) -> FMTrainRoutineConfig:
    """Construct a minimal valid config for the gate to inspect.

    Most fields are placeholders; only the data subsection matters for the
    gate. We point ``corpus_registry`` at a dummy path that won't be read
    by ``_assert_preflight_gates`` (the gate is data-source-agnostic).
    """
    dummy_registry = tmp_path / "dummy_registry.json"
    dummy_registry.write_text("{}")
    dummy_ckpt = tmp_path / "trunk.pt"
    dummy_ckpt.write_bytes(b"")
    dummy_arch = tmp_path / "arch.json"
    dummy_arch.write_text("{}")
    cfg_dict = {
        "run": {"stage": "S1", "tag": "fft_cfm", "seed": 42, "device": "cpu"},
        "data": {
            "corpus_registry": str(dummy_registry),
            "augmentation_config_path": (
                str(augmentation_config_path) if augmentation_config_path else None
            ),
            "preflight_decision_path": (
                str(preflight_decision_path) if preflight_decision_path else None
            ),
        },
        "model": {
            "trunk": {
                "checkpoint": str(dummy_ckpt),
                "arch_json": str(dummy_arch),
            },
            "controlnet": {"conditioning_inputs": ["t1pre"]},
        },
        "rflow": {},
        "optim": {"lr": 1e-4},
        "ema": {"decay": 0.999},
        "training": {"total_steps": 1},
        "validation": {},
        "exhaustive_val": {"enabled": False},
        "output": {"experiments_root": str(tmp_path / "experiments")},
    }
    return FMTrainRoutineConfig.model_validate(cfg_dict)


def _write_aug_yaml(path: Path, names: list[str]) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "seed": 0,
                "augmentations": [{"name": n, "p": 0.5} for n in names],
            }
        )
    )


def _write_decision(path: Path, allowlist: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "criterion": "ssim_and_recon_floor",
                "latent_safe_augmentations": allowlist,
            }
        )
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_augmentations_skips_gate(tmp_path: Path) -> None:
    """When no augmentation YAML is set, the gate is a no-op."""
    cfg = _build_minimal_cfg(
        tmp_path=tmp_path,
        augmentation_config_path=None,
        preflight_decision_path=None,
    )
    _assert_preflight_gates(cfg)  # no raise


def test_aug_without_preflight_path_raises(tmp_path: Path) -> None:
    """Augmentations enabled but no preflight path → fail-fast."""
    aug_yaml = tmp_path / "aug.yaml"
    _write_aug_yaml(aug_yaml, ["flip_lr"])
    cfg = _build_minimal_cfg(
        tmp_path=tmp_path,
        augmentation_config_path=aug_yaml,
        preflight_decision_path=None,
    )
    with pytest.raises(PreflightGateError, match="preflight_decision_path"):
        _assert_preflight_gates(cfg)


def test_missing_decision_file_raises(tmp_path: Path) -> None:
    aug_yaml = tmp_path / "aug.yaml"
    _write_aug_yaml(aug_yaml, ["flip_lr"])
    decision_path = tmp_path / "does_not_exist" / "decision.json"
    cfg = _build_minimal_cfg(
        tmp_path=tmp_path,
        augmentation_config_path=aug_yaml,
        preflight_decision_path=decision_path,
    )
    with pytest.raises(PreflightGateError, match="missing"):
        _assert_preflight_gates(cfg)


def test_disallowed_augmentation_raises(tmp_path: Path) -> None:
    """An augmentation absent from the allowlist must trigger the gate."""
    aug_yaml = tmp_path / "aug.yaml"
    _write_aug_yaml(aug_yaml, ["flip_lr", "rotate_yaw"])
    decision_path = tmp_path / "decision.json"
    _write_decision(decision_path, allowlist=["flip_lr", "translate"])
    cfg = _build_minimal_cfg(
        tmp_path=tmp_path,
        augmentation_config_path=aug_yaml,
        preflight_decision_path=decision_path,
    )
    with pytest.raises(PreflightGateError, match=r"rotate_yaw"):
        _assert_preflight_gates(cfg)


def test_allowed_augmentation_passes(tmp_path: Path) -> None:
    aug_yaml = tmp_path / "aug.yaml"
    _write_aug_yaml(aug_yaml, ["flip_lr", "translate"])
    decision_path = tmp_path / "decision.json"
    _write_decision(decision_path, allowlist=["flip_lr", "translate", "rotate_yaw"])
    cfg = _build_minimal_cfg(
        tmp_path=tmp_path,
        augmentation_config_path=aug_yaml,
        preflight_decision_path=decision_path,
    )
    _assert_preflight_gates(cfg)  # no raise


def test_load_preflight_decision_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(PreflightGateError, match="missing"):
        _load_preflight_decision(tmp_path / "absent" / "decision.json")


# ---------------------------------------------------------------------------
# S3 decoder_lpl gate (schema 0.9.0)
# ---------------------------------------------------------------------------


def _build_s3_cfg(
    *,
    tmp_path: Path,
    decoder_lpl_decision_path: Path | None,
    variant_weights: dict[str, float] | None = None,
) -> FMTrainRoutineConfig:
    """S3-stage minimal config exercising the decoder_lpl gate."""
    dummy_registry = tmp_path / "dummy_registry.json"
    dummy_registry.write_text("{}")
    dummy_ckpt = tmp_path / "trunk.pt"
    dummy_ckpt.write_bytes(b"")
    dummy_arch = tmp_path / "arch.json"
    dummy_arch.write_text("{}")
    cfg_dict = {
        "run": {"stage": "s3", "tag": "lpl_fft", "seed": 42, "device": "cpu"},
        "data": {
            "corpus_registry": str(dummy_registry),
            "decoder_lpl_decision_path": (
                str(decoder_lpl_decision_path) if decoder_lpl_decision_path else None
            ),
            "variant_weights": variant_weights
            or {"v0": 0.2, "v1": 0.2, "v2": 0.2, "v3": 0.2, "v4": 0.2},
        },
        "model": {
            "trunk": {"checkpoint": str(dummy_ckpt), "arch_json": str(dummy_arch)},
            "controlnet": {"conditioning_inputs": ["t1pre"]},
        },
        "rflow": {},
        "optim": {"lr": 1e-4},
        "ema": {"decay": 0.999},
        "training": {"total_steps": 1},
        "validation": {},
        "exhaustive_val": {"enabled": False},
        "output": {"experiments_root": str(tmp_path / "experiments")},
    }
    return FMTrainRoutineConfig.model_validate(cfg_dict)


def _write_lpl_decision(
    path: Path,
    *,
    allowed_variants: list[str],
    A: tuple[int, ...] = (2, 3),
) -> None:
    payload = {
        "schema_version": "1.0",
        "produced_at": "2026-06-18T20:27:09Z",
        "producer": "vena.preflight.decoder_lpl_profile:1.0",
        "n_patients_run": 90,
        "patients_per_cohort": {"UCSF-PDGM": 3},
        "A_recommended": list(A),
        "w_l": {str(k): 1.0 for k in A},
        "t_min": 0.4,
        "outlier_k": {str(k): 5.0 for k in A},
        "region_recipe": {
            "alpha_wt": 2.0,
            "alpha_notwt": 3.0,
            "soft_region": False,
            "per_cohort_overrides": None,
        },
        "allowed_variants": allowed_variants,
        "v4_brain_mask_status": "ok",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_s3_requires_decoder_lpl_decision_path(tmp_path: Path) -> None:
    cfg = _build_s3_cfg(tmp_path=tmp_path, decoder_lpl_decision_path=None)
    with pytest.raises(PreflightGateError, match="decoder_lpl_decision_path"):
        _assert_preflight_gates(cfg)


def test_s3_missing_decision_file_raises(tmp_path: Path) -> None:
    cfg = _build_s3_cfg(
        tmp_path=tmp_path,
        decoder_lpl_decision_path=tmp_path / "no_such" / "decision.json",
    )
    with pytest.raises(PreflightGateError, match="does not exist"):
        _assert_preflight_gates(cfg)


def test_s3_valid_decision_passes(tmp_path: Path) -> None:
    decision = tmp_path / "decision.json"
    _write_lpl_decision(decision, allowed_variants=["v0", "v1", "v2", "v3", "v4"])
    cfg = _build_s3_cfg(tmp_path=tmp_path, decoder_lpl_decision_path=decision)
    _assert_preflight_gates(cfg)  # no raise


def test_s3_disallowed_variant_in_weights_raises(tmp_path: Path) -> None:
    """If variant_weights uses v4 but the decision rejects it, gate fails."""
    decision = tmp_path / "decision.json"
    _write_lpl_decision(decision, allowed_variants=["v0", "v1", "v2", "v3"])  # no v4
    cfg = _build_s3_cfg(
        tmp_path=tmp_path,
        decoder_lpl_decision_path=decision,
        variant_weights={"v0": 0.2, "v1": 0.2, "v2": 0.2, "v3": 0.2, "v4": 0.2},
    )
    with pytest.raises(PreflightGateError, match="v4"):
        _assert_preflight_gates(cfg)


def test_s3_disallowed_variant_with_zero_weight_passes(tmp_path: Path) -> None:
    """A zeroed disallowed variant in the weights is OK (operator masked it)."""
    decision = tmp_path / "decision.json"
    _write_lpl_decision(decision, allowed_variants=["v0", "v1", "v2", "v3"])  # no v4
    cfg = _build_s3_cfg(
        tmp_path=tmp_path,
        decoder_lpl_decision_path=decision,
        variant_weights={"v0": 0.25, "v1": 0.25, "v2": 0.25, "v3": 0.25, "v4": 0.0},
    )
    _assert_preflight_gates(cfg)  # no raise
