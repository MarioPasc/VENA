"""Unit tests for the shared decode helpers in ``vena.common.decode``.

Uses a fake decoder so no MAISI checkpoint is required.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from vena.common.decode import decode_box, decode_depth_identity
from vena.model.autoencoder.maisi.preprocessing import CropPadSpec, DepthPad

pytestmark = pytest.mark.unit


@dataclass
class _FakeDecoded:
    image: torch.Tensor


class _FakeDecoder:
    """Records the kwargs each ``decode`` call sees and returns a deterministic vol."""

    def __init__(self, target_shape: tuple[int, int, int]) -> None:
        self.target_shape = target_shape
        self.last_crop_spec: CropPadSpec | None = None
        self.last_pad: DepthPad | None = None

    def decode(
        self,
        latent: torch.Tensor,
        pad: DepthPad | None = None,
        *,
        crop_spec: CropPadSpec | None = None,
    ) -> _FakeDecoded:
        if crop_spec is not None:
            self.last_crop_spec = crop_spec
            b = latent.shape[0]
            return _FakeDecoded(image=torch.full((b, 1, *self.target_shape), 0.5))
        if pad is not None:
            self.last_pad = pad
            b = latent.shape[0]
            d = latent.shape[-1] * 4
            return _FakeDecoded(image=torch.full((b, 1, 4, 4, d), 0.5))
        raise ValueError("either pad or crop_spec must be provided")


def test_decode_box_returns_squeezed_clamped_volume() -> None:
    decoder = _FakeDecoder(target_shape=(8, 8, 8))
    latent = torch.zeros(1, 4, 2, 2, 2)
    crop_spec = CropPadSpec(crop_origin=(0, 0, 0), native_shape=(8, 8, 8), target_shape=(8, 8, 8))

    vol = decode_box(decoder, latent, crop_spec)

    assert isinstance(vol, torch.Tensor)
    assert vol.shape == (8, 8, 8)  # batch + channel squeezed
    assert vol.dtype == torch.float32
    assert torch.all((vol >= 0.0) & (vol <= 1.0))


def test_decode_box_returns_seconds_when_requested() -> None:
    decoder = _FakeDecoder(target_shape=(4, 4, 4))
    latent = torch.zeros(1, 4, 1, 1, 1)
    crop_spec = CropPadSpec(crop_origin=(0, 0, 0), native_shape=(4, 4, 4), target_shape=(4, 4, 4))

    result = decode_box(decoder, latent, crop_spec, return_seconds=True)

    assert isinstance(result, tuple)
    vol, sec = result
    assert vol.shape == (4, 4, 4)
    assert isinstance(sec, float)
    assert sec >= 0.0


def test_decode_box_skip_clamp() -> None:
    """When clamp is off, the helper must pass the raw decoder output through."""

    class _NoClampDecoder(_FakeDecoder):
        def decode(self, latent, pad=None, *, crop_spec=None):
            self.last_crop_spec = crop_spec
            return _FakeDecoded(
                image=torch.full((1, 1, *self.target_shape), 1.5)  # > 1
            )

    decoder = _NoClampDecoder(target_shape=(4, 4, 4))
    latent = torch.zeros(1, 4, 1, 1, 1)
    crop_spec = CropPadSpec(crop_origin=(0, 0, 0), native_shape=(4, 4, 4), target_shape=(4, 4, 4))

    vol = decode_box(decoder, latent, crop_spec, clamp_unit_interval=False)
    assert vol.max().item() == pytest.approx(1.5)


def test_decode_depth_identity_uses_identity_pad() -> None:
    decoder = _FakeDecoder(target_shape=(8, 8, 8))
    latent_d = 5
    latent = torch.zeros(2, 4, 1, 1, latent_d)

    out = decode_depth_identity(decoder, latent)

    assert decoder.last_pad is not None
    expected_depth = latent_d * 4
    assert decoder.last_pad.before == 0
    assert decoder.last_pad.after == 0
    assert decoder.last_pad.original_depth == expected_depth
    assert decoder.last_pad.padded_depth == expected_depth
    # Output is whatever the decoder returned; shape is decoder-defined.
    assert out.image.shape[-1] == expected_depth
