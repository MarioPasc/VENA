"""Contingency C1: run SynDiff's StyleGAN2 ops without the fused CUDA kernels.

SynDiff's vendored generator imports ``utils.op``, whose two modules
(``fused_act``, ``upfirdn2d``) call ``torch.utils.cpp_extension.load`` **at import
time** — so merely importing them triggers a JIT nvcc build. On Picasso that build
does not succeed:

* the `vena-syndiff` conda env was deleted after the 2026-06-15 training runs, and
  rebuilding it lands a toolchain nvcc rejects (``'timespec_get' has not been
  declared``, missing ``cusparse.h``);
* the main `vena` env has no ``ninja`` at all.

`src/external/syndiff/PATCHES.md` anticipates exactly this and specifies
contingency **C1**: alias the fused entrypoints to the pure-PyTorch reference
implementations. PATCHES.md phrases C1 as "monkey-patch ``utils/op/__init__.py``",
but `.claude/rules/external-deps.md` forbids editing anything under
``src/external/``. So C1 is applied from *this* side instead: :func:`install`
pre-seeds ``sys.modules`` with equivalent modules **before** the vendored code can
import the real ones, and the ``load()`` call never runs.

Numerics
--------
This is not an approximation. Both upstream files already ship the pure-PyTorch
path — it is the branch they take when ``input.device.type == "cpu"`` — and the
CUDA kernels are a fused, faster spelling of the same arithmetic. The functions
below are the upstream reference bodies (``upfirdn2d_native`` and the CPU branch
of ``fused_leaky_relu``), reproduced verbatim except that they run on any device.
Expect the same outputs up to floating-point association, at roughly 2-4x the
per-layer cost — irrelevant here because SynDiff runs at NFE=4.

One upstream quirk is deliberately NOT reproduced: the CPU branch of
``fused_leaky_relu`` hard-codes ``negative_slope=0.2`` in its ``F.leaky_relu``
call, ignoring the argument. The CUDA kernel honours the argument. We honour it
too, so this shim matches the *CUDA* path (SynDiff only ever uses the 0.2 default,
so the two agree in practice regardless).

``FusedLeakyReLU.bias`` keeps its name and shape, so the trained checkpoint's
state_dict loads unchanged.
"""

from __future__ import annotations

import logging
import sys
from types import ModuleType

import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

__all__ = ["FusedLeakyReLU", "fused_leaky_relu", "install", "upfirdn2d", "upfirdn2d_native"]


def upfirdn2d_native(
    input: torch.Tensor,
    kernel: torch.Tensor,
    up_x: int,
    up_y: int,
    down_x: int,
    down_y: int,
    pad_x0: int,
    pad_x1: int,
    pad_y0: int,
    pad_y1: int,
) -> torch.Tensor:
    """Upsample, FIR-filter, downsample — the upstream reference body."""
    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)

    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = F.pad(out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1

    return out.view(-1, channel, out_h, out_w)


def upfirdn2d(
    input: torch.Tensor,
    kernel: torch.Tensor,
    up: int = 1,
    down: int = 1,
    pad: tuple[int, int] = (0, 0),
) -> torch.Tensor:
    """Device-agnostic ``upfirdn2d`` — same signature as the fused entrypoint."""
    return upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])


def fused_leaky_relu(
    input: torch.Tensor,
    bias: torch.Tensor,
    negative_slope: float = 0.2,
    scale: float = 2**0.5,
) -> torch.Tensor:
    """``scale * leaky_relu(input + bias)`` — what the fused kernel computes."""
    rest_dim = [1] * (input.ndim - bias.ndim - 1)
    return F.leaky_relu(
        input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=negative_slope
    ) * scale


class FusedLeakyReLU(nn.Module):
    """Drop-in for the vendored ``FusedLeakyReLU``; ``bias`` name/shape preserved."""

    def __init__(self, channel: int, negative_slope: float = 0.2, scale: float = 2**0.5) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def _module(name: str, **attrs: object) -> ModuleType:
    mod = ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def install() -> None:
    """Pre-seed ``sys.modules`` so the vendored ops never JIT-compile.

    Idempotent, and a no-op if ``utils.op`` is already imported (i.e. the real
    fused kernels built successfully — then we must not shadow them).

    Must be called BEFORE any import of SynDiff's vendored generator, because the
    vendored modules invoke ``cpp_extension.load`` at import time.
    """
    if "utils.op" in sys.modules:
        logger.debug("utils.op already imported — leaving the fused kernels in place")
        return

    fused_act = _module(
        "utils.op.fused_act",
        FusedLeakyReLU=FusedLeakyReLU,
        fused_leaky_relu=fused_leaky_relu,
    )
    upfirdn2d_mod = _module(
        "utils.op.upfirdn2d",
        upfirdn2d=upfirdn2d,
        upfirdn2d_native=upfirdn2d_native,
    )
    op = _module(
        "utils.op",
        FusedLeakyReLU=FusedLeakyReLU,
        fused_leaky_relu=fused_leaky_relu,
        upfirdn2d=upfirdn2d,
        upfirdn2d_native=upfirdn2d_native,
        fused_act=fused_act,
        upfirdn2d_mod=upfirdn2d_mod,
    )

    sys.modules["utils.op"] = op
    sys.modules["utils.op.fused_act"] = fused_act
    sys.modules["utils.op.upfirdn2d"] = upfirdn2d_mod

    logger.info(
        "SynDiff contingency C1 active: StyleGAN2 fused CUDA ops replaced by the "
        "pure-PyTorch reference (no nvcc/ninja build; same arithmetic, ~2-4x slower)"
    )
