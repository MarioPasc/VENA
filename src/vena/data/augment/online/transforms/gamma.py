"""Per-volume gamma intensity correction.

Gamma is a non-linear pixel-wise function and there is no exact latent-space
analogue for a non-linear VAE encoder. The :meth:`apply_latent` provided here
is the cheapest linear proxy: ``z → z * α`` with ``α = γ`` (a one-parameter
rescale). The equivariance preflight is expected to *reject* gamma for runtime
latent augmentation; the operator is retained so that decision can be reached
empirically rather than assumed.

For applications that want gamma in the training data distribution despite the
latent-space failure, the path is to apply gamma image-side and re-encode
through the VAE, but that costs one extra encoder forward per micro-batch and
is out of scope for the 4-epoch smoke.
"""

from __future__ import annotations

import random
from typing import Any

import torch

from vena.data.augment.online.base import LatentAugmentation, LatentAugmentationError


class Gamma(LatentAugmentation):
    """γ ∈ [γ_min, γ_max] correction on image; ``z → γ·z`` on latents."""

    name = "gamma"

    def __init__(
        self,
        p: float = 0.5,
        gamma_min: float = 0.8,
        gamma_max: float = 1.2,
    ) -> None:
        super().__init__(p=p)
        if not (0.0 < float(gamma_min) <= float(gamma_max)):
            raise LatentAugmentationError(
                f"Gamma: require 0 < gamma_min <= gamma_max; got [{gamma_min}, {gamma_max}]"
            )
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)

    def sample_params(self, rng: random.Random) -> dict[str, Any]:
        return {"gamma": float(rng.uniform(self.gamma_min, self.gamma_max))}

    def apply_latent(self, batch: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        gamma = float(params["gamma"])
        # The latent proxy: rescale every modality latent by gamma. The WT
        # mask is untouched — gamma has no geometric effect.
        for key in ("z_t1pre", "z_t2", "z_flair", "z_t1c"):
            if key in batch and isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key] * gamma
        return batch

    def apply_image(self, x: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Gamma.apply_image expects (H,W,D); got shape {tuple(x.shape)}")
        gamma = float(params["gamma"])
        # Inputs to the VAE are pre-normalised into [0, 1]. Pow on negatives
        # is undefined, so we clamp to [0, 1] first.
        return torch.pow(x.clamp(0.0, 1.0), gamma)

    def param_tag(self, params: dict[str, Any]) -> str:
        gamma = float(params["gamma"])
        # Bucket into one-decimal precision so the CSV stays bounded.
        return f"{self.name}_{gamma:.1f}"
