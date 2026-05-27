"""Unit tests for the image→latent downsample registry."""

from __future__ import annotations

import pytest
import torch

from vena.model.fm.controlnet.downsample import (
    AvgPoolDownsampler,
    IdentityDownsampler,
    NearestDownsampler,
    TrilinearDownsampler,
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
def test_registry_dispatch() -> None:
    assert isinstance(get_downsampler("identity"), IdentityDownsampler)
    assert isinstance(get_downsampler("nearest", factor=4), NearestDownsampler)
    assert isinstance(get_downsampler("trilinear", factor=4), TrilinearDownsampler)
    assert isinstance(get_downsampler("avg_pool", factor=4), AvgPoolDownsampler)


@pytest.mark.unit
def test_registry_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown downsampler"):
        get_downsampler("doesnotexist")


@pytest.mark.unit
def test_registry_vae_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        get_downsampler("vae")
