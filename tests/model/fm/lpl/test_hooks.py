"""Unit tests for :func:`vena.model.fm.lpl.hooks.decoder_feature_extractor`.

The context manager wraps :func:`vena.common.partial_decode` with the
``post_quant_conv`` pre-step. Tests use a stand-in handle that mimics
``AutoencoderHandle.model.{post_quant_conv, decoder}`` with two tiny
``nn.Module``s, so no real MAISI checkpoint is touched.

Coverage:

* ``extract(latent)`` returns the same features as a manual
  ``partial_decode(decoder, post_quant_conv(latent), ...)`` call.
* Hooks do not leak across two consecutive ``extract(...)`` calls.
* ``__exit__`` after an exception inside the ``with`` block does not
  leave hooks registered (hooks are per-call inside ``partial_decode``,
  so the ``__exit__`` is symbolic — but the contract must hold).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import nn

from vena.common import partial_decode
from vena.model.fm.lpl import decoder_feature_extractor

pytestmark = pytest.mark.unit


class _Block(nn.Module):
    def __init__(self, ci: int, co: int) -> None:
        super().__init__()
        self.c = nn.Conv3d(ci, co, 3, padding=1)
        self.in_channels = ci
        self.out_channels = co

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.c(x))


class _Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block(4, 8), _Block(8, 8), _Block(8, 4)])


class _Model(nn.Module):
    """Mimics ``AutoencoderHandle.model``: ``post_quant_conv`` + ``decoder``."""

    def __init__(self) -> None:
        super().__init__()
        self.post_quant_conv = nn.Conv3d(4, 4, kernel_size=1)
        self.decoder = _Decoder()


@dataclass
class _Handle:
    """Stand-in for ``AutoencoderHandle`` with just ``.model``."""

    model: _Model


def test_extract_matches_manual_partial_decode() -> None:
    torch.manual_seed(1)
    handle = _Handle(model=_Model())
    latent = torch.randn(1, 4, 4, 4, 4)

    with decoder_feature_extractor(
        handle,  # type: ignore[arg-type]
        blocks={0, 2},
        max_block=2,
    ) as extract:
        caps = extract(latent)

    # Manual reference path.
    z_post = handle.model.post_quant_conv(latent)
    expected = partial_decode(handle.model.decoder, z_post, blocks={0, 2}, max_block=2)

    assert set(caps) == set(expected) == {0, 2}
    for k in (0, 2):
        assert torch.allclose(caps[k], expected[k])


def test_consecutive_extracts_dont_leak_hooks() -> None:
    """Two ``extract`` calls inside one ``with`` block must each clean up."""
    handle = _Handle(model=_Model())
    with decoder_feature_extractor(
        handle,  # type: ignore[arg-type]
        blocks={1},
        max_block=2,
    ) as extract:
        extract(torch.randn(1, 4, 4, 4, 4))
        for b in handle.model.decoder.blocks:
            assert not b._forward_hooks, "hooks leaked between extract calls"
        extract(torch.randn(1, 4, 4, 4, 4))
        for b in handle.model.decoder.blocks:
            assert not b._forward_hooks


def test_exception_inside_with_block_clears_hooks() -> None:
    handle = _Handle(model=_Model())
    with pytest.raises(RuntimeError, match="user-side"):
        with decoder_feature_extractor(
            handle,  # type: ignore[arg-type]
            blocks={0},
            max_block=2,
        ) as extract:
            extract(torch.randn(1, 4, 4, 4, 4))
            raise RuntimeError("user-side")
    for b in handle.model.decoder.blocks:
        assert not b._forward_hooks


def test_grad_checkpoint_flag_forwarded() -> None:
    """``grad_checkpoint=True`` passes through to ``partial_decode``."""
    handle = _Handle(model=_Model())
    with decoder_feature_extractor(
        handle,  # type: ignore[arg-type]
        blocks={2},
        max_block=2,
        grad_checkpoint=True,
    ) as extract:
        caps = extract(torch.randn(1, 4, 4, 4, 4))
    assert 2 in caps
    assert torch.isfinite(caps[2]).all()
