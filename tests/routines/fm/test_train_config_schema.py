"""Unit tests for B3/B4/B5 FMTrainRoutineConfig schema changes.

B3: ``gradient_clip_val`` default changed from 1.0 to 5.0.
B4: ``best_metric_name`` / ``best_metric_region`` / ``best_metric_nfe`` deleted.
B5: ``retention_n_checkpoints`` default changed from 3 to 40.
"""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError
from routines.fm.train.engine import FMTrainRoutineConfig

pytestmark = pytest.mark.unit

_MINIMAL_YAML = """
run:
  stage: s1
  tag: test_arm
  resume_from: null
  seed: 1337
  device: cpu
  precision: bf16-mixed

data:
  corpus_registry: /nonexistent/corpus.json

model:
  trunk:
    checkpoint: /nonexistent/trunk.pt
    arch_overrides: {}
    class_token: 9
    spacing_mm: [1.0, 1.0, 1.0]
    trainable: false
    regime: fft
  controlnet:
    enabled: false

output:
  experiments_root: /tmp/test_experiments
"""


def _parse(extra_yaml: str = "") -> FMTrainRoutineConfig:
    raw = yaml.safe_load(_MINIMAL_YAML + extra_yaml)
    return FMTrainRoutineConfig.model_validate(raw)


class TestB3GradientClipDefault:
    """B3: gradient_clip_val default is 5.0."""

    def test_default_is_5(self) -> None:
        cfg = _parse()
        assert cfg.training.gradient_clip_val == 5.0

    def test_explicit_override_respected(self) -> None:
        cfg = _parse("\ntraining:\n  gradient_clip_val: 1.0\n")
        assert cfg.training.gradient_clip_val == 1.0


class TestB4BestMetricDeleted:
    """B4: best_metric_name/region/nfe removed from _TrainingCfg; extra=forbid rejects them."""

    def test_best_metric_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _parse("\ntraining:\n  best_metric_name: mse_latent\n")

    def test_best_metric_region_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _parse("\ntraining:\n  best_metric_region: bg\n")

    def test_best_metric_nfe_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _parse("\ntraining:\n  best_metric_nfe: 5\n")

    def test_config_without_best_metric_fields_parses(self) -> None:
        cfg = _parse()
        assert not hasattr(cfg.training, "best_metric_name")
        assert not hasattr(cfg.training, "best_metric_region")
        assert not hasattr(cfg.training, "best_metric_nfe")


class TestB5RetentionDefault:
    """B5: retention_n_checkpoints default is 40."""

    def test_default_is_40(self) -> None:
        cfg = _parse()
        assert cfg.output.retention_n_checkpoints == 40

    def test_explicit_override_respected(self) -> None:
        cfg = _parse("\noutput:\n  experiments_root: /tmp\n  retention_n_checkpoints: 3\n")
        assert cfg.output.retention_n_checkpoints == 3


class TestB4PatienceNull:
    """patience: null must parse as None (EarlyStopping disabled)."""

    def test_patience_null_parses_as_none(self) -> None:
        cfg = _parse("\ntraining:\n  patience: null\n")
        assert cfg.training.patience is None

    def test_default_patience_is_none(self) -> None:
        cfg = _parse()
        assert cfg.training.patience is None


class TestNewProductionYAMLs:
    """The three new production YAML configs must parse cleanly."""

    @pytest.mark.parametrize(
        "yaml_path",
        [
            "routines/fm/train/configs/runs/picasso_s1_v4_l1_fft.yaml",
            "routines/fm/train/configs/runs/picasso_s1_v4_l2_fft.yaml",
            "routines/fm/train/configs/runs/picasso_s1_v4_huber_fft.yaml",
            "routines/fm/train/configs/smoke/loginexa_s1_v4_4ep_l1.yaml",
        ],
    )
    def test_yaml_parses(self, yaml_path: str) -> None:
        cfg = FMTrainRoutineConfig.from_yaml(yaml_path)
        # All new configs use gradient_clip_val=5.0
        assert cfg.training.gradient_clip_val == pytest.approx(5.0)
        # All new configs carry latent_preds_every_n
        assert cfg.exhaustive_val.latent_preds_every_n >= 1

    @pytest.mark.parametrize(
        "yaml_path",
        [
            "routines/fm/train/configs/runs/picasso_s1_v4_l1_fft.yaml",
            "routines/fm/train/configs/runs/picasso_s1_v4_l2_fft.yaml",
            "routines/fm/train/configs/runs/picasso_s1_v4_huber_fft.yaml",
        ],
    )
    def test_production_arms_use_runaway_guard(self, yaml_path: str) -> None:
        """Production arms set patience=150 (runaway guard only, not convergence detector)."""
        cfg = FMTrainRoutineConfig.from_yaml(yaml_path)
        assert cfg.training.patience == 150, (
            f"Expected patience=150 (runaway guard); got {cfg.training.patience}. "
            "All arms must be identical for a valid ablation (§18 Trap 2)."
        )
        assert cfg.training.total_steps == 800000
        assert cfg.training.max_epochs == 10000

    def test_smoke_disables_early_stopping(self) -> None:
        """Smoke test uses patience=null so the disabled-EarlyStopping log fires."""
        cfg = FMTrainRoutineConfig.from_yaml(
            "routines/fm/train/configs/smoke/loginexa_s1_v4_4ep_l1.yaml"
        )
        assert cfg.training.patience is None
