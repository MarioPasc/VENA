"""Tests for the MAISI trunk's ``conv_in`` channel-expansion utility.

The load-bearing test is ``test_expansion_preserves_step0_behaviour``: with
``zero_init_new=True``, the expanded trunk's output on
``cat([x_old, X], dim=1)`` must be bit-identical to the original trunk's
output on ``x_old`` — for any ``X``. This is what guarantees the v3
trunk starts training with the pretrained MAISI behaviour intact.

The MAISI trunk is enormous (~1 B params). These tests use a tiny stand-in
``nn.Module`` that exposes a single ``nn.Conv3d`` named ``conv_in``,
mirroring the actual MAISI trunk's attribute layout. ``expand_conv_in``
only touches ``trunk.conv_in``, so a synthetic fixture exercises the full
code path.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.unit

from vena.model.fm.maisi.conv_in_expand import ConvInExpansionError, expand_conv_in


class _TinyTrunk(nn.Module):
    """Stand-in for the MAISI trunk: just a single conv_in."""

    def __init__(self, in_channels: int = 4, out_channels: int = 8) -> None:
        super().__init__()
        self.conv_in = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_in(x)


def test_expansion_preserves_step0_behaviour() -> None:
    """Bit-identical when zero-init: cat([x_old, X]) ≡ x_old for any X."""
    torch.manual_seed(0)
    trunk = _TinyTrunk(in_channels=4, out_channels=8)
    x_old = torch.randn(2, 4, 6, 6, 6)
    y_orig = trunk(x_old).detach().clone()

    trunk = expand_conv_in(trunk, new_in_channels=16, zero_init_new=True)
    # Add arbitrary "conditioning" channels — should not affect the output.
    extra = torch.randn(2, 12, 6, 6, 6) * 7.3  # large magnitude
    y_new = trunk(torch.cat([x_old, extra], dim=1))

    torch.testing.assert_close(y_new, y_orig, rtol=1e-6, atol=1e-7)


def test_expansion_zero_init_default_zeroes_new_slice() -> None:
    """``zero_init_new=True`` → new input-channel slice has |w| == 0."""
    torch.manual_seed(1)
    trunk = _TinyTrunk(in_channels=4, out_channels=8)
    trunk = expand_conv_in(trunk, new_in_channels=16, zero_init_new=True)
    new_slice = trunk.conv_in.weight[:, 4:]
    assert new_slice.abs().max().item() == 0.0


def test_expansion_preserves_bias() -> None:
    """Bias is copied verbatim from the original conv."""
    torch.manual_seed(2)
    trunk = _TinyTrunk(in_channels=4, out_channels=8)
    orig_bias = trunk.conv_in.bias.detach().clone()
    trunk = expand_conv_in(trunk, new_in_channels=16, zero_init_new=True)
    torch.testing.assert_close(trunk.conv_in.bias, orig_bias)


def test_expansion_preserves_old_weight_slice() -> None:
    """The first 4 input channels' weights are copied byte-for-byte."""
    torch.manual_seed(3)
    trunk = _TinyTrunk(in_channels=4, out_channels=8)
    orig_w = trunk.conv_in.weight.detach().clone()
    trunk = expand_conv_in(trunk, new_in_channels=16, zero_init_new=True)
    torch.testing.assert_close(trunk.conv_in.weight[:, :4], orig_w)


def test_expansion_in_channels_attribute() -> None:
    """``trunk.conv_in.in_channels`` reflects the new value."""
    trunk = _TinyTrunk(in_channels=4, out_channels=8)
    trunk = expand_conv_in(trunk, new_in_channels=17, zero_init_new=True)
    assert trunk.conv_in.in_channels == 17


def test_expansion_noop_when_equal() -> None:
    """new_in_channels == current ⇒ no-op (returns same trunk)."""
    trunk = _TinyTrunk(in_channels=4, out_channels=8)
    orig_id = id(trunk.conv_in)
    trunk = expand_conv_in(trunk, new_in_channels=4, zero_init_new=True)
    assert id(trunk.conv_in) == orig_id


def test_expansion_rejects_shrink() -> None:
    """Shrinking is not supported and must error fast."""
    trunk = _TinyTrunk(in_channels=8, out_channels=8)
    with pytest.raises(ConvInExpansionError, match="must be"):
        expand_conv_in(trunk, new_in_channels=4, zero_init_new=True)


def test_expansion_rejects_non_conv3d() -> None:
    """If the trunk's conv_in is the wrong class, surface a clear error."""

    class _BadTrunk(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv_in = nn.Linear(4, 8)  # wrong kind on purpose

    with pytest.raises(ConvInExpansionError, match="nn.Conv3d"):
        expand_conv_in(_BadTrunk(), new_in_channels=16, zero_init_new=True)


def test_expansion_kaiming_init_when_zero_init_false() -> None:
    """``zero_init_new=False`` ⇒ new channels carry non-zero weights."""
    torch.manual_seed(4)
    trunk = _TinyTrunk(in_channels=4, out_channels=8)
    trunk = expand_conv_in(trunk, new_in_channels=12, zero_init_new=False)
    new_slice = trunk.conv_in.weight[:, 4:]
    assert new_slice.abs().max().item() > 0.0


def test_expansion_dtype_preserved() -> None:
    """The new conv lives on the same dtype as the original."""
    trunk = _TinyTrunk(in_channels=4, out_channels=8).to(torch.float64)
    trunk = expand_conv_in(trunk, new_in_channels=16, zero_init_new=True)
    assert trunk.conv_in.weight.dtype == torch.float64
