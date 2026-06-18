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
