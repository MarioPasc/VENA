"""Test that ``MaisiEncoder.encode(mask=...)`` plumbs through to ``percentile_normalise``.

The encoder is built without a real autoencoder handle — we monkeypatch
``_full`` to short-circuit the forward pass and assert the pre-network
input has the expected distribution. That distribution differs depending on
whether the brain mask is consulted:

* No mask + ``foreground_only=True`` on a z-score-like volume → negative
  intra-brain voxels are clamped to 0 (the BraTS-Africa bug).
* With mask covering the whole brain → negative intra-brain voxels survive
  (rescaled into ``[0, 1]``).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vena.model.autoencoder.maisi.encode.engine import EncodeResult, MaisiEncoder
from vena.model.autoencoder.maisi.preprocessing import CropPadSpec

pytestmark = pytest.mark.unit


class _StubHandle:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.checkpoint_sha256 = "0" * 64
        self.arch_kwargs: dict = {}


def _make_encoder(monkeypatch: pytest.MonkeyPatch) -> tuple[MaisiEncoder, list[torch.Tensor]]:
    enc = MaisiEncoder(
        handle=_StubHandle(),  # type: ignore[arg-type]
        depth_pad_base=8,
        percentile_lower=0.0,
        percentile_upper=99.5,
        percentile_foreground_only=True,
        precision_mode="fp32",
    )
    captured: list[torch.Tensor] = []

    def _full_stub(x: torch.Tensor) -> torch.Tensor:
        captured.append(x.clone())
        z = torch.zeros((x.shape[0], 4, 1, 1, 1), dtype=torch.float32)
        return z

    monkeypatch.setattr(enc, "_full", _full_stub)
    monkeypatch.setattr(enc, "_sliding", _full_stub)
    return enc, captured


def _zscore_volume(shape: tuple[int, int, int]) -> torch.Tensor:
    """A z-score-like brain volume with negative intra-brain voxels."""
    rng = np.random.default_rng(0)
    vol = rng.standard_normal(shape).astype(np.float32)
    return torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)


def test_mask_keeps_negative_intra_brain(monkeypatch: pytest.MonkeyPatch) -> None:
    enc, captured = _make_encoder(monkeypatch)
    shape = (16, 16, 16)
    x = _zscore_volume(shape)
    mask = torch.ones((1, 1, *shape), dtype=torch.float32)
    spec = CropPadSpec(crop_origin=(0, 0, 0), native_shape=shape, target_shape=shape)
    _ = enc.encode(x, mode="full", crop_spec=spec, mask=mask)
    rescaled = captured[-1]
    # Even with a z-score input, mask-aware normalisation should produce a
    # range that spans most of [0, 1] (negative voxels are part of foreground
    # and percentiles are computed over them).
    assert rescaled.min().item() == pytest.approx(0.0, abs=1e-6)
    assert rescaled.max().item() == pytest.approx(1.0, abs=1e-6)


def test_no_mask_clamps_negative_with_foreground_heuristic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enc, captured = _make_encoder(monkeypatch)
    shape = (16, 16, 16)
    x = _zscore_volume(shape)
    spec = CropPadSpec(crop_origin=(0, 0, 0), native_shape=shape, target_shape=shape)
    _ = enc.encode(x, mode="full", crop_spec=spec)
    rescaled = captured[-1]
    # Negative voxels are clamped to 0 by the foreground heuristic path —
    # large fraction of zero voxels expected.
    n_zero = float((rescaled == 0).float().mean().item())
    # Half the values are negative in a z-score volume → ≥40 % zeros after clamp.
    assert n_zero > 0.4


def test_mask_shape_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    enc, _ = _make_encoder(monkeypatch)
    shape = (16, 16, 16)
    x = _zscore_volume(shape)
    bad_mask = torch.ones((1, 1, 8, 8, 8), dtype=torch.float32)  # spatial mismatch
    spec = CropPadSpec(crop_origin=(0, 0, 0), native_shape=shape, target_shape=shape)
    with pytest.raises(Exception):
        enc.encode(x, mode="full", crop_spec=spec, mask=bad_mask)


def test_returns_encode_result(monkeypatch: pytest.MonkeyPatch) -> None:
    enc, _ = _make_encoder(monkeypatch)
    shape = (16, 16, 16)
    x = _zscore_volume(shape)
    mask = torch.ones((1, 1, *shape), dtype=torch.float32)
    spec = CropPadSpec(crop_origin=(0, 0, 0), native_shape=shape, target_shape=shape)
    out = enc.encode(x, mode="full", crop_spec=spec, mask=mask)
    assert isinstance(out, EncodeResult)
    assert out.crop is spec
