"""Unit tests for :class:`AugmentationPipeline` and the YAML loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from vena.data.augment import (
    AugmentationPipeline,
    LatentAugmentationError,
    build_pipeline_from_yaml,
)
from vena.data.augment.online.pipeline import NO_AUG_TAG
from vena.data.augment.online.transforms.flip import FlipLR
from vena.data.augment.online.transforms.translate import Translate


def _make_batch() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return {
        "z_t1pre": torch.randn(4, 8, 8, 8, generator=g),
        "z_t2": torch.randn(4, 8, 8, 8, generator=g),
        "z_flair": torch.randn(4, 8, 8, 8, generator=g),
        "z_t1c": torch.randn(4, 8, 8, 8, generator=g),
        "m_wt": torch.randn(1, 8, 8, 8, generator=g),
    }


def test_empty_pipeline_rejected() -> None:
    with pytest.raises(LatentAugmentationError):
        AugmentationPipeline([])


def test_duplicate_names_rejected() -> None:
    with pytest.raises(LatentAugmentationError):
        AugmentationPipeline([FlipLR(p=0.5), FlipLR(p=0.5)])


def test_no_aug_yields_none_tag() -> None:
    pipe = AugmentationPipeline([FlipLR(p=0.0)], seed=0)
    out, combo = pipe(_make_batch())
    assert combo == NO_AUG_TAG
    assert out["_aug_combo"] == NO_AUG_TAG


def test_combo_string_sorted_join() -> None:
    pipe = AugmentationPipeline(
        [FlipLR(p=1.0), Translate(p=1.0, max_voxels=4)],
        seed=0,
    )
    _, combo = pipe(_make_batch())
    parts = combo.split("+")
    assert parts == sorted(parts)
    assert any(p.startswith("flip_lr") for p in parts)


def test_pipeline_attaches_aug_combo_key() -> None:
    pipe = AugmentationPipeline([FlipLR(p=1.0)], seed=0)
    out, combo = pipe(_make_batch())
    assert out["_aug_combo"] == combo


def test_yaml_loader_unknown_aug(tmp_path: Path) -> None:
    cfg_path = tmp_path / "aug.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "seed": 0,
                "augmentations": [{"name": "does_not_exist", "p": 0.5}],
            }
        )
    )
    with pytest.raises(LatentAugmentationError):
        build_pipeline_from_yaml(cfg_path)


def test_yaml_loader_preflight_gate(tmp_path: Path) -> None:
    cfg_path = tmp_path / "aug.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "augmentations": [
                    {"name": "flip_lr", "p": 1.0},
                    {"name": "translate", "p": 1.0, "max_voxels": 4},
                ],
            }
        )
    )
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(json.dumps({"latent_safe_augmentations": ["flip_lr"]}))
    with pytest.raises(LatentAugmentationError):
        build_pipeline_from_yaml(cfg_path, preflight_decision_path=decision_path)


def test_yaml_loader_happy_path(tmp_path: Path) -> None:
    cfg_path = tmp_path / "aug.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "augmentations": [{"name": "flip_lr", "p": 1.0}],
            }
        )
    )
    pipe = build_pipeline_from_yaml(cfg_path)
    assert pipe.names() == ("flip_lr",)
