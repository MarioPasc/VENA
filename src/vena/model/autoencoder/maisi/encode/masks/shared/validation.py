"""Model-agnostic shape and label-set checks used by mask downsamplers."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from vena.model.autoencoder.maisi.exceptions import ShapeContractError

from .exceptions import LabelCodeError


def assert_image_space_mask(mask: torch.Tensor) -> None:
    """Ensure ``mask`` is ``(B, 1, H, W, D)`` with an integral-like dtype."""
    if mask.ndim != 5:
        raise ShapeContractError(
            f"mask must be (B,1,H,W,D); got shape {tuple(mask.shape)}"
        )
    if mask.shape[1] != 1:
        raise ShapeContractError(
            f"mask must have a single label-channel; got C={mask.shape[1]}"
        )
    if mask.is_floating_point():
        # Allow float-encoded labels but warn callers via the type contract:
        # we keep this permissive because some sources cast int8 → float32
        # before reaching the downsampler.
        return


def assert_target_shape(
    latent_spatial: tuple[int, int, int],
    expected: tuple[int, int, int],
) -> None:
    if tuple(latent_spatial) != tuple(expected):
        raise ShapeContractError(
            f"downsampler produced spatial shape {latent_spatial}; expected {expected}"
        )


def assert_label_codes_subset(
    mask: torch.Tensor,
    allowed: Iterable[int],
) -> None:
    """Raise :class:`LabelCodeError` if ``mask`` contains a value outside
    ``allowed ∪ {0}``.

    Comparison is done after a defensive cast to ``int64``. ``0`` (background)
    is always allowed implicitly.
    """
    allowed_set = set(int(v) for v in allowed) | {0}
    present = torch.unique(mask.to(torch.int64))
    extras = [int(v.item()) for v in present if int(v.item()) not in allowed_set]
    if extras:
        raise LabelCodeError(
            f"mask contains unexpected label codes {sorted(extras)}; "
            f"declared label set = {sorted(allowed_set)}"
        )
