"""Unit tests for MaisiControlNet — instantiation + zero-init.

CPU-only; verifies the constructor is callable and the zero-init step actually
zeros the output projections so that augmented forward = trunk forward at step 0.
"""

from __future__ import annotations

import pytest
import torch

from vena.model.fm.controlnet.maisi_controlnet import MaisiControlNet


@pytest.mark.unit
def test_maisi_controlnet_constructs() -> None:
    cn = MaisiControlNet(conditioning_in_channels=13)
    assert cn.conditioning_in_channels == 13
    assert cn.arch_kwargs["conditioning_embedding_in_channels"] == 13


@pytest.mark.unit
def test_output_projections_are_zero_after_init() -> None:
    cn = MaisiControlNet(conditioning_in_channels=13)
    for name, p in cn.net.named_parameters():
        if name.startswith("controlnet_down_blocks.") or name.startswith(
            "controlnet_mid_block."
        ):
            assert torch.all(p == 0.0), f"non-zero output projection: {name}"


@pytest.mark.unit
def test_zero_init_re_zeroes_after_perturbation() -> None:
    cn = MaisiControlNet(conditioning_in_channels=5)
    for _, p in cn.net.named_parameters():
        with torch.no_grad():
            p.add_(0.1)
    cn.zero_init_output_projections()
    for name, p in cn.net.named_parameters():
        if name.startswith("controlnet_down_blocks.") or name.startswith(
            "controlnet_mid_block."
        ):
            assert torch.all(p == 0.0)
