"""Running EMA of per-block, per-channel feature statistics.

The §3.3 design note pins shared, prediction-derived standardisation as the
LPL contract: the *prediction* features drive the running mean/var, and
*both* prediction and target are standardised with the same statistics
(Berrada 2025 found FID 3.79 vs 4.79 by sharing). VENA's effective batch
is small (one or two ControlNet passes), so per-batch statistics are noisy
— an EMA across optimiser steps reduces variance without bias.

The class is an :class:`nn.Module` so the running statistics are part of
the state_dict and round-trip through Lightning checkpoints natively. Per
block we keep ``mean`` and ``var`` of shape ``(C_block,)`` plus a scalar
``n_updates`` counter that drives the optional warmup gate. Statistics
are stored as buffers (no gradients).
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

_EPS: float = 1e-8


class FeatureStatsEMA(nn.Module):
    """Per-block per-channel EMA of feature mean/var.

    Parameters
    ----------
    channels : Mapping[int, int]
        Block index → channel count. Determines the per-block buffer shape.
    decay : float, default 0.99
        EMA decay. Per-batch statistics are folded in with
        ``stat <- decay * stat + (1 - decay) * batch_stat``.

    Notes
    -----
    * The first ``update`` call uses ``decay = 0`` regardless of the
      configured ``decay``: the EMA bootstraps from the first batch's
      statistics rather than mixing them with the zero-initialised
      buffers. After that, the configured decay applies.
    * ``n_updates`` is a registered buffer (int64 scalar) so it round-trips
      with the rest of the state_dict.
    """

    def __init__(
        self,
        channels: Mapping[int, int],
        decay: float = 0.99,
    ) -> None:
        super().__init__()
        if not channels:
            raise ValueError("channels mapping must be non-empty")
        if not (0.0 <= decay < 1.0):
            raise ValueError(f"decay must be in [0, 1); got {decay}")
        self._decay = float(decay)
        # Sort the keys so iteration order is reproducible.
        self._block_ids: tuple[int, ...] = tuple(sorted(channels))
        for blk in self._block_ids:
            c = int(channels[blk])
            if c <= 0:
                raise ValueError(f"channel count for block {blk} must be > 0")
            self.register_buffer(f"mean_{blk}", torch.zeros(c))
            self.register_buffer(f"var_{blk}", torch.ones(c))
        self.register_buffer("n_updates", torch.zeros((), dtype=torch.int64))

    # ------------------------------------------------------------------
    # Access helpers
    # ------------------------------------------------------------------

    @property
    def block_ids(self) -> tuple[int, ...]:
        return self._block_ids

    @property
    def decay(self) -> float:
        return self._decay

    def mean(self, block_idx: int) -> torch.Tensor:
        return self.get_buffer(f"mean_{block_idx}")

    def var(self, block_idx: int) -> torch.Tensor:
        return self.get_buffer(f"var_{block_idx}")

    def is_warmed_up(self, min_samples: int = 100) -> bool:
        """Whether enough updates have accumulated to standardise reliably."""
        return int(self.n_updates.item()) >= int(min_samples)

    # ------------------------------------------------------------------
    # EMA update + standardisation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(self, features: Mapping[int, torch.Tensor]) -> None:
        """Fold a batch of features into the running statistics.

        Each tensor must be ``(B, C_block, ...)``; the channel-axis mean
        and variance are computed over all non-channel dims of the entire
        batch (the same statistic the standardisation will undo).
        """
        is_first = int(self.n_updates.item()) == 0
        effective_decay = 0.0 if is_first else self._decay
        for blk, feat in features.items():
            if blk not in self._block_ids:
                continue
            # Flatten everything but the channel axis: (B*..., C) → mean/var
            # along dim 0 produces (C,).
            c = feat.shape[1]
            flat = feat.movedim(1, -1).reshape(-1, c).float()
            mean_b = flat.mean(dim=0)
            # Use unbiased=False so the var of a 1-element batch is well-defined.
            var_b = flat.var(dim=0, unbiased=False)
            mean_buf = self.get_buffer(f"mean_{blk}")
            var_buf = self.get_buffer(f"var_{blk}")
            mean_buf.mul_(effective_decay).add_((1.0 - effective_decay) * mean_b)
            var_buf.mul_(effective_decay).add_((1.0 - effective_decay) * var_b)
        self.n_updates.add_(1)

    def standardise(self, feat: torch.Tensor, block_idx: int) -> torch.Tensor:
        """Return ``(feat - mean) / sqrt(var + eps)`` broadcast over space.

        Parameters
        ----------
        feat : torch.Tensor
            Shape ``(B, C, ...)``.
        block_idx : int
            Block whose statistics to apply. Must have been declared at
            construction time.
        """
        if block_idx not in self._block_ids:
            raise KeyError(f"block_idx {block_idx} not registered; known: {self._block_ids}")
        mean_buf = self.get_buffer(f"mean_{block_idx}")
        var_buf = self.get_buffer(f"var_{block_idx}")
        # Broadcast (C,) over (B, C, ...) by reshaping to (1, C, 1, 1, ...).
        view = (1, -1) + (1,) * (feat.ndim - 2)
        mean = mean_buf.to(feat.device).view(view)
        std = var_buf.to(feat.device).add(_EPS).sqrt().view(view)
        return (feat - mean) / std
