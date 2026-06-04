"""Abstract base class for paired image+latent augmentations.

A :class:`LatentAugmentation` exposes two operators tied to the same set of
stochastic parameters:

- :meth:`apply_latent` — applied to a sample dict during training. Receives the
  full per-sample dict produced by ``LatentH5Dataset`` (latents
  ``z_{t1pre,t2,flair,t1c}`` and the WT mask ``m_wt``) and must transform each
  spatial tensor *identically*; otherwise the conditioning latents and target
  latent decohere geometrically.
- :meth:`apply_image` — applied to a single ``(H, W, D)`` image-domain tensor.
  Used only by the equivariance preflight to verify
  ``T_image(D(z)) ≈ D(T_latent(z))``.

Stochastic parameters are drawn once per call by :meth:`sample_params`, then
passed to both apply methods so the image-space and latent-space realisations
are coupled. A :meth:`param_tag` string is appended to the per-sample
combination tag consumed by :class:`AugmentationTracker`.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import torch


class LatentAugmentationError(Exception):
    """Raised on configuration or runtime errors inside the augment library."""


class LatentAugmentation(ABC):
    """Paired image/latent augmentation operator.

    Concrete subclasses set the class-level ``name`` slug and implement
    :meth:`sample_params`, :meth:`apply_latent`, and :meth:`apply_image`. The
    constructor takes the per-sample probability ``p`` plus operator-specific
    keyword arguments.

    Parameters
    ----------
    p : float
        Per-sample probability of applying the augmentation. Must be in
        ``[0, 1]``.
    """

    name: ClassVar[str]
    # Keys of spatial tensors in the sample dict that must be transformed
    # together. Subclasses override if they need a different set (none do
    # today, but the indirection keeps the contract explicit).
    LATENT_KEYS: ClassVar[tuple[str, ...]] = (
        "z_t1pre",
        "z_t2",
        "z_flair",
        "z_t1c",
        "m_wt",
    )

    def __init__(self, p: float) -> None:
        if not 0.0 <= float(p) <= 1.0:
            raise LatentAugmentationError(f"{self.name}: probability p must lie in [0, 1]; got {p}")
        self.p = float(p)

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abstractmethod
    def sample_params(self, rng: random.Random) -> dict[str, Any]:
        """Draw stochastic parameters for one call.

        Implementations should consume ``rng`` exclusively so the augmentation
        is reproducible from a worker-level Python ``random`` state.
        """

    @abstractmethod
    def apply_latent(self, batch: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        """Apply the augmentation to a per-sample dict of latent tensors.

        Implementations must transform every key in :data:`LATENT_KEYS` that
        is present in ``batch`` with the SAME parameters and return the
        mutated dict (mutation in place is permitted; the wrapper does not
        copy).
        """

    @abstractmethod
    def apply_image(self, x: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
        """Apply the image-space counterpart of the augmentation.

        Parameters
        ----------
        x : torch.Tensor
            Volume of shape ``(H, W, D)`` (no batch / channel dims).
        params : dict
            Output of :meth:`sample_params` (or hand-crafted).

        Returns
        -------
        torch.Tensor
            Transformed volume of shape ``(H, W, D)``.
        """

    @abstractmethod
    def param_tag(self, params: dict[str, Any]) -> str:
        """Compact human-readable tag for ``params`` (no whitespace, no '+').

        Used as the augmentation's contribution to the combination string
        consumed by :class:`AugmentationTracker`. Distinct parameter draws
        should yield distinct tags so the per-epoch CSV separates them.
        """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def maybe_apply(
        self,
        batch: dict[str, Any],
        rng: random.Random,
    ) -> tuple[dict[str, Any], str | None]:
        """Bernoulli-gate the augmentation by ``self.p``.

        Returns ``(batch, tag)`` if applied, ``(batch, None)`` otherwise.
        Called by :class:`AugmentationPipeline`.
        """
        if rng.random() >= self.p:
            return batch, None
        params = self.sample_params(rng)
        batch = self.apply_latent(batch, params)
        return batch, self.param_tag(params)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover — debug helper
        return f"{type(self).__name__}(name={self.name!r}, p={self.p:g})"
