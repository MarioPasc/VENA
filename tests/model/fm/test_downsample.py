"""Unit tests for the image→latent downsample registry."""

from __future__ import annotations

import pytest
import torch

from vena.model.fm.controlnet.downsample import (
    AvgPoolDownsampler,
    IdentityDownsampler,
    LiftTo4ChDownsampler,
    NearestDownsampler,
    TrilinearDownsampler,
    ZeroOutDownsampler,
    get_downsampler,
)


@pytest.mark.unit
def test_identity_passthrough() -> None:
    x = torch.randn(1, 1, 60, 60, 40)
    y = IdentityDownsampler()(x)
    assert torch.equal(x, y)


@pytest.mark.unit
def test_nearest_factor_4_shape_and_binarise() -> None:
    x = torch.rand(2, 1, 240, 240, 160)
    y = NearestDownsampler(factor=4)(x)
    assert y.shape == (2, 1, 60, 60, 40)

    y_bin = NearestDownsampler(factor=4, binarise_threshold=0.5)(x)
    assert torch.equal(y_bin, (y_bin > 0).float())  # binary


@pytest.mark.unit
def test_trilinear_shape() -> None:
    x = torch.rand(2, 1, 240, 240, 160)
    y = TrilinearDownsampler(factor=4)(x)
    assert y.shape == (2, 1, 60, 60, 40)


@pytest.mark.unit
def test_avgpool_shape() -> None:
    x = torch.rand(2, 3, 240, 240, 160)
    y = AvgPoolDownsampler(factor=4)(x)
    assert y.shape == (2, 3, 60, 60, 40)


@pytest.mark.unit
def test_zero_out_shape_and_values() -> None:
    x = torch.randn(2, 1, 60, 60, 40)
    y = ZeroOutDownsampler()(x)
    assert y.shape == x.shape
    assert torch.all(y == 0)


@pytest.mark.unit
def test_zero_out_default_out_channels_is_none() -> None:
    # ``None`` makes the assembler use the kind-based default (mask_channels=1).
    assert ZeroOutDownsampler().out_channels is None


@pytest.mark.unit
def test_lift_to_4ch_shape_and_out_channels() -> None:
    x = torch.randn(2, 1, 15, 15, 10)
    ds = LiftTo4ChDownsampler()
    y = ds(x)
    assert y.shape == (2, 4, 15, 15, 10)
    assert ds.out_channels == 4


@pytest.mark.unit
def test_lift_to_4ch_gradient_flows() -> None:
    x = torch.randn(2, 1, 6, 6, 4, requires_grad=True)
    ds = LiftTo4ChDownsampler()
    y = ds(x)
    y.sum().backward()
    assert ds.conv.weight.grad is not None
    assert torch.isfinite(ds.conv.weight.grad).all()


@pytest.mark.unit
def test_lift_to_4ch_rejects_non_positive_channels() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        LiftTo4ChDownsampler(in_channels=0)
    with pytest.raises(ValueError, match="must be positive"):
        LiftTo4ChDownsampler(out_channels=0)


@pytest.mark.unit
def test_registry_dispatch() -> None:
    assert isinstance(get_downsampler("identity"), IdentityDownsampler)
    assert isinstance(get_downsampler("nearest", factor=4), NearestDownsampler)
    assert isinstance(get_downsampler("trilinear", factor=4), TrilinearDownsampler)
    assert isinstance(get_downsampler("avg_pool", factor=4), AvgPoolDownsampler)
    assert isinstance(get_downsampler("zero_out"), ZeroOutDownsampler)
    assert isinstance(get_downsampler("lift_to_4ch"), LiftTo4ChDownsampler)


@pytest.mark.unit
def test_registry_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown downsampler"):
        get_downsampler("doesnotexist")


@pytest.mark.unit
def test_registry_vae_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        get_downsampler("vae")
