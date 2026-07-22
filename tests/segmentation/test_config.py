"""Tests for vena.segmentation.config.

Covers:
- Round-trip YAML loading with defaults and overrides.
- Frozen enforcement (assignment raises).
- Schema strictness: unknown keys raise ValidationError.
- Canonical defaults are exact (guards against silent drift).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from vena.segmentation.config import (
    DerivationConfig,
    MetricsConfig,
    ModelConfig,
    SegmentationConfig,
    TargetConfig,
)

pytestmark = pytest.mark.segmentation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL_YAML = textwrap.dedent(
    """\
    model:
      name: bsf_swinunetr_brats

    data:
      corpus_registry: /tmp/corpus.json
      image_h5_root: /tmp/h5
      patch_size: [128, 128, 128]
      cache_rate: 0.0
      num_workers: 2

    train:
      max_epochs: 200
      lr: 1.0e-4
      batch_size: 2
      val_every_epochs: 5
      early_stop_patience: 20
    """
)


def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


def test_from_yaml_round_trip(tmp_path: Path) -> None:
    """SegmentationConfig.from_yaml returns a valid frozen instance."""
    yaml_path = _write_yaml(tmp_path, _MINIMAL_YAML)
    cfg = SegmentationConfig.from_yaml(yaml_path)

    assert cfg.model.name == "bsf_swinunetr_brats"
    assert cfg.data.corpus_registry == Path("/tmp/corpus.json")
    assert cfg.train.max_epochs == 200
    assert cfg.seed == 1337  # default


def test_from_yaml_overrides(tmp_path: Path) -> None:
    """Overriding nested fields via YAML is reflected in the instance."""
    content = _MINIMAL_YAML + textwrap.dedent(
        """\
        seed: 42
        derivation:
          latent_grid: [48, 56, 48]
          avg_pool_stride: 4
        """
    )
    cfg = SegmentationConfig.from_yaml(_write_yaml(tmp_path, content))
    assert cfg.seed == 42
    assert cfg.derivation.latent_grid == (48, 56, 48)


def test_frozen_raises_on_assignment(tmp_path: Path) -> None:
    """Mutating any field on a frozen config raises."""
    cfg = SegmentationConfig.from_yaml(_write_yaml(tmp_path, _MINIMAL_YAML))
    with pytest.raises(ValidationError):
        cfg.seed = 99  # type: ignore[misc]


def test_nested_frozen_raises_on_assignment(tmp_path: Path) -> None:
    """Mutation on a nested sub-config also raises."""
    cfg = SegmentationConfig.from_yaml(_write_yaml(tmp_path, _MINIMAL_YAML))
    with pytest.raises(ValidationError):
        cfg.model.feature_size = 64  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Schema strictness
# ---------------------------------------------------------------------------


def test_unknown_top_level_key_raises(tmp_path: Path) -> None:
    """A typo'd top-level key raises ValidationError (extra='forbid')."""
    bad = _MINIMAL_YAML + "typo_key: 123\n"
    with pytest.raises(ValidationError):
        SegmentationConfig.from_yaml(_write_yaml(tmp_path, bad))


def test_unknown_nested_key_raises(tmp_path: Path) -> None:
    """A typo'd nested key also raises ValidationError."""
    bad = _MINIMAL_YAML + textwrap.dedent(
        """\
        derivation:
          latent_grid: [48, 56, 48]
          not_a_real_field: 99
        """
    )
    with pytest.raises(ValidationError):
        SegmentationConfig.from_yaml(_write_yaml(tmp_path, bad))


# ---------------------------------------------------------------------------
# Default-value contract tests (guards against silent drift)
# ---------------------------------------------------------------------------


def test_default_latent_grid() -> None:
    """DerivationConfig.latent_grid default is exactly (48, 56, 48)."""
    d = DerivationConfig()
    assert d.latent_grid == (48, 56, 48), (
        f"latent_grid default drifted to {d.latent_grid}; "
        "must be (48, 56, 48) — LATENT_SPATIAL from vena.data.h5.latent_domain.manifest"
    )


def test_default_out_channels() -> None:
    """ModelConfig.out_channels default is exactly 2 ([WT, NETC])."""
    m = ModelConfig(name="bsf_swinunetr_brats")
    assert m.out_channels == 2, f"out_channels default drifted to {m.out_channels}; must be 2"


def test_default_netc_operator() -> None:
    """TargetConfig.netc_operator default is 'euclidean_percomponent'."""
    t = TargetConfig()
    assert t.netc_operator == "euclidean_percomponent", (
        f"netc_operator default drifted to {t.netc_operator!r}"
    )


def test_default_tumor_region() -> None:
    """TargetConfig.tumor_region default is 'tc' (tumour core, not whole tumour).

    Guard against regression to 'wt': 81% of WT is non-enhancing edema on
    UCSF-PDGM; TC (NETC+ET) is the correct channel-0 for T1c conditioning.
    """
    t = TargetConfig()
    assert t.tumor_region == "tc", (
        f"tumor_region default drifted to {t.tumor_region!r}; "
        "must remain 'tc' — see design correction 2026-07-22"
    )


def test_default_selection_metric() -> None:
    """MetricsConfig.selection_metric default is 'dual'."""
    m = MetricsConfig()
    assert m.selection_metric == "dual", (
        f"selection_metric default drifted to {m.selection_metric!r}"
    )


def test_latent_grid_matches_served_latents() -> None:
    """DerivationConfig.latent_grid matches LATENT_SPATIAL (single source of truth).

    If this test fails the config default and the H5 manifest have drifted —
    fix DerivationConfig, not the manifest.
    """
    from vena.data.h5.latent_domain.manifest import LATENT_SPATIAL

    d = DerivationConfig()
    assert d.latent_grid == LATENT_SPATIAL == (48, 56, 48), (
        f"DerivationConfig.latent_grid={d.latent_grid} "
        f"!= LATENT_SPATIAL={LATENT_SPATIAL}; "
        "derived mask will not voxel-register with the served latents"
    )
