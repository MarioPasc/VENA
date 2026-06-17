"""Tests for the §4.1 harmonisation contract."""

from __future__ import annotations

import pytest
import torch

from vena.common import percentile_normalise
from vena.inference.harmonisation import HARMONISATION_RECIPE, apply_harmonisation

pytestmark = pytest.mark.unit


def test_recipe_string_matches_encoder_call() -> None:
    assert "lower=0.0" in HARMONISATION_RECIPE
    assert "upper=99.5" in HARMONISATION_RECIPE
    assert "foreground_only=True" in HARMONISATION_RECIPE


def test_3d_input_round_trip() -> None:
    torch.manual_seed(0)
    vol = torch.rand((8, 8, 8)) * 1000.0
    out = apply_harmonisation(vol)
    assert out.shape == (8, 8, 8)
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0 + 1e-6


def test_5d_input_shape_preserved_to_3d() -> None:
    torch.manual_seed(1)
    vol = torch.rand((1, 1, 4, 4, 4))
    out = apply_harmonisation(vol)
    assert out.shape == (4, 4, 4)


def test_brain_mask_zeros_exterior() -> None:
    torch.manual_seed(2)
    vol = torch.rand((6, 6, 6)) * 1000.0
    mask = torch.zeros_like(vol)
    mask[1:5, 1:5, 1:5] = 1.0
    out = apply_harmonisation(vol, brain_mask=mask)
    # Exterior must be exactly zero.
    exterior = out[mask == 0]
    assert float(exterior.abs().max()) == 0.0


def test_parity_with_encoder_call() -> None:
    """Calling ``percentile_normalise`` directly must yield the same numbers."""
    torch.manual_seed(3)
    vol = torch.rand((8, 8, 8)) * 1000.0
    direct = percentile_normalise(
        vol[None, None].float(), lower=0.0, upper=99.5, foreground_only=True
    )[0, 0]
    via_helper = apply_harmonisation(vol)
    assert torch.allclose(direct.contiguous(), via_helper.contiguous(), atol=1e-6)


def test_invalid_shape_raises() -> None:
    bad = torch.rand((2, 4, 4, 4))  # 4-D — neither 3-D nor 5-D
    with pytest.raises(ValueError):
        apply_harmonisation(bad)


def test_mask_shape_mismatch_raises() -> None:
    vol = torch.rand((4, 4, 4))
    mask = torch.ones((3, 3, 3))
    with pytest.raises(ValueError):
        apply_harmonisation(vol, brain_mask=mask)
