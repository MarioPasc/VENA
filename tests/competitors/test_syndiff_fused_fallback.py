"""Contingency C1: the pure-PyTorch StyleGAN2 ops must match the fused kernels.

C1 swaps SynDiff's JIT-compiled CUDA ops for pure-PyTorch equivalents when the
toolchain cannot build them (the Picasso case). "Equivalent" is the whole claim,
so pin it: the shim must reproduce the fused kernels' arithmetic, keep the
parameter names the trained checkpoint expects, and not shadow the real kernels
when those do build.
"""

from __future__ import annotations

import sys

import pytest
import torch
from torch.nn import functional as F

from vena.competitors.syndiff.fused_ops_fallback import (
    FusedLeakyReLU,
    fused_leaky_relu,
    install,
    upfirdn2d,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("negative_slope", [0.2, 0.1])
@pytest.mark.parametrize("scale", [2**0.5, 1.0])
def test_fused_leaky_relu_matches_definition(negative_slope: float, scale: float) -> None:
    """The fused kernel computes scale * leaky_relu(x + bias); so must the shim."""
    x = torch.randn(2, 5, 4, 4)
    bias = torch.randn(5)

    got = fused_leaky_relu(x, bias, negative_slope, scale)
    want = F.leaky_relu(x + bias.view(1, -1, 1, 1), negative_slope=negative_slope) * scale

    torch.testing.assert_close(got, want)


def test_fused_leaky_relu_honours_negative_slope() -> None:
    """The CUDA kernel honours negative_slope; upstream's CPU branch hard-codes 0.2.

    We follow the CUDA kernel. A shim that silently used 0.2 would agree with this
    test's default case and diverge everywhere else, so check a non-default slope
    actually changes the negative half.
    """
    x = torch.full((1, 1, 1, 1), -1.0)
    bias = torch.zeros(1)

    out_02 = fused_leaky_relu(x, bias, negative_slope=0.2, scale=1.0)
    out_01 = fused_leaky_relu(x, bias, negative_slope=0.1, scale=1.0)

    assert out_02.item() == pytest.approx(-0.2)
    assert out_01.item() == pytest.approx(-0.1)


def test_fused_leaky_relu_module_exposes_bias_parameter() -> None:
    """`bias` must keep its name/shape or the trained state_dict will not load."""
    module = FusedLeakyReLU(channel=7)

    assert "bias" in dict(module.named_parameters())
    assert module.bias.shape == (7,)
    module.load_state_dict({"bias": torch.arange(7, dtype=torch.float32)})


def test_upfirdn2d_identity_kernel_is_a_noop() -> None:
    """A 1x1 unit kernel with no up/down/pad must return the input unchanged."""
    x = torch.randn(2, 3, 8, 8)
    out = upfirdn2d(x, torch.ones(1, 1), up=1, down=1, pad=(0, 0))
    torch.testing.assert_close(out, x)


def test_upfirdn2d_upsample_then_downsample_shapes() -> None:
    """Shape contract: up=2 doubles, down=2 halves (with the matching pad)."""
    x = torch.randn(1, 2, 8, 8)
    k = torch.ones(2, 2)

    up = upfirdn2d(x, k, up=2, down=1, pad=(0, 0))
    assert up.shape == (1, 2, 15, 15)  # 8*2 + 0 - 2 + 1

    down = upfirdn2d(x, k, up=1, down=2, pad=(0, 0))
    assert down.shape == (1, 2, 4, 4)  # (8 - 2)//2 + 1


def test_upfirdn2d_blur_matches_manual_conv() -> None:
    """With no resampling, upfirdn2d is a flipped-kernel valid conv (a blur)."""
    x = torch.randn(1, 1, 6, 6)
    k = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    got = upfirdn2d(x, k, up=1, down=1, pad=(0, 0))
    w = torch.flip(k, [0, 1]).view(1, 1, 2, 2)
    want = F.conv2d(x, w)

    torch.testing.assert_close(got, want)


def test_install_is_idempotent_and_seeds_sys_modules() -> None:
    """install() seeds the three module paths the vendored backbones import."""
    for name in ("utils.op", "utils.op.fused_act", "utils.op.upfirdn2d"):
        sys.modules.pop(name, None)
    try:
        install()
        install()  # idempotent

        assert sys.modules["utils.op"].FusedLeakyReLU is FusedLeakyReLU
        assert sys.modules["utils.op.fused_act"].fused_leaky_relu is fused_leaky_relu
        assert sys.modules["utils.op.upfirdn2d"].upfirdn2d is upfirdn2d
    finally:
        for name in ("utils.op", "utils.op.fused_act", "utils.op.upfirdn2d"):
            sys.modules.pop(name, None)


def test_install_does_not_shadow_real_fused_kernels() -> None:
    """If the real utils.op is already imported, the CUDA path must be left alone."""
    sentinel = object()
    sys.modules["utils.op"] = sentinel  # type: ignore[assignment]
    try:
        install()
        assert sys.modules["utils.op"] is sentinel
    finally:
        sys.modules.pop("utils.op", None)
