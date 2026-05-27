"""Unit tests for the trunk arch JSON + TrunkConfig.

Does not load the actual checkpoint — those tests live under the ``fm``+``gpu``
markers and run on a workstation with a CUDA GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from vena.model.fm.maisi import TrunkConfig
from vena.model.fm.maisi.trunk import _DEFAULT_ARCH_CONFIG, _load_arch_kwargs


@pytest.mark.unit
def test_default_arch_json_exists_and_parses() -> None:
    path = _DEFAULT_ARCH_CONFIG
    assert path.is_file(), f"missing arch JSON at {path}"
    with path.open("r") as f:
        raw = json.load(f)
    assert raw["in_channels"] == 4
    assert raw["out_channels"] == 4
    assert raw["num_channels"] == [64, 128, 256, 512]


@pytest.mark.unit
def test_load_arch_kwargs_strips_comments() -> None:
    kw = _load_arch_kwargs(_DEFAULT_ARCH_CONFIG)
    assert all(not k.startswith("_") for k in kw)


@pytest.mark.unit
def test_trunk_config_class_labels_and_spacing_shapes() -> None:
    cfg = TrunkConfig(checkpoint=Path("/nonexistent.pt"), class_token=9)
    device = torch.device("cpu")
    cls = cfg.make_class_labels(batch_size=3, device=device)
    spacing = cfg.make_spacing_tensor(batch_size=3, device=device)
    assert cls.shape == (3,)
    assert cls.dtype == torch.long
    assert torch.all(cls == 9)
    assert spacing.shape == (3, 3)
    assert torch.allclose(spacing, torch.tensor([[1.0, 1.0, 1.0]] * 3))
