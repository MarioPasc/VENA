"""Left-right anatomical flip along the LPS L-axis."""

from __future__ import annotations

import random
from typing import Any

import torch

from vena.data.augment.base import LatentAugmentation

# All VENA H5s store volumes reoriented to LPS (see
# ``src/vena/data/h5/shared/crop.py:12``). LPS = (L, P, S), so axis 0 of
# ``(H, W, D)`` is the L → R axis and a left-right flip is ``torch.flip(.., dim=-3)``
# on a ``(C, H, W, D)`` latent or ``(H, W, D)`` image volume.
_LR_AXIS_FROM_TAIL: int = -3


class FlipLR(LatentAugmentation):
    """Hard left-right mirror flip; identical on image and latent grids.

    The operator is deterministic given that it fires (no sub-parameter), so
    :meth:`sample_params` returns an empty dict. Two consecutive applications
    cancel out, which the unit tests verify.
    """

    name = "flip_lr"

    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p=p)

    def sample_params(self, rng: random.Random) -> dict[str, Any]:
        return {}

    def apply_latent(self, batch: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        for key in self.LATENT_KEYS:
            if key in batch and isinstance(batch[key], torch.Tensor):
                batch[key] = torch.flip(batch[key], dims=[_LR_AXIS_FROM_TAIL])
        return batch

    def apply_image(self, x: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"FlipLR.apply_image expects (H,W,D); got shape {tuple(x.shape)}")
        return torch.flip(x, dims=[_LR_AXIS_FROM_TAIL])

    def param_tag(self, params: dict[str, Any]) -> str:
        return self.name
