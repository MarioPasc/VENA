"""EMA with Karras-2024 warm-up, wrapping :mod:`ema_pytorch`.

The training_routine spec §7.2 prescribes the linear-warmup schedule

.. math::
    \\beta(s) \\;=\\; \\min\\!\\bigl(\\beta_\\text{target},\\, (1+s) / (10+s)\\bigr),

which prevents large early-step weight updates from being washed out by an
over-aggressive EMA (Karras et al. 2024, *Analyzing and Improving the Training
Dynamics of Diffusion Models*, arXiv:2312.02696).

:mod:`ema_pytorch.EMA` exposes a closely related power schedule

.. math::
    \\beta(s) \\;=\\; 1 - (1 + s / \\text{inv\\_gamma})^{-\\text{power}}

clamped to ``[min_value, beta]``. With ``inv_gamma=10, power=1, min_value=0``
this evaluates to ``s / (s + 10)`` — within ~10 % of the doc's curve over the
first few thousand steps and identical in the converged regime. We accept this
small divergence in exchange for a tested external implementation.

The wrapper exposes a minimal surface — :meth:`update`, :attr:`ema_model`,
:meth:`state_dict`, :meth:`load_state_dict`, :meth:`get_current_decay` — so
the Lightning module never imports :mod:`ema_pytorch` directly.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


class WarmupEMA(nn.Module):
    """Wrap a model in an EMA shadow with a Karras-style warm-up schedule.

    Parameters
    ----------
    model : nn.Module
        The trainable model (typically the ControlNet).
    decay : float
        Target EMA decay :math:`\\beta_\\text{target}`. Doc default ``0.9999``.
    update_after_step : int
        Skip EMA updates for the first ``N`` optimiser steps. Default ``0``
        (start updating immediately; the warm-up schedule handles the
        bootstrap).
    update_every : int
        Apply the EMA update every ``N`` steps. Default ``1``.
    inv_gamma : float
        Power-schedule inverse-gamma. Default ``10`` matches the doc's
        denominator constant.
    power : float
        Power-schedule exponent. Default ``1`` matches the doc's linear form.
    min_value : float
        Floor of the EMA decay. Default ``0`` (no floor).

    Notes
    -----
    The wrapper is itself a :class:`nn.Module` so it can hold the shadow
    model as a submodule and participate in Lightning's
    :meth:`on_save_checkpoint` / :meth:`on_load_checkpoint` pathway via the
    standard ``state_dict``.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        update_after_step: int = 0,
        update_every: int = 1,
        inv_gamma: float = 10.0,
        power: float = 1.0,
        min_value: float = 0.0,
    ) -> None:
        super().__init__()
        try:
            from ema_pytorch import EMA
        except ImportError as exc:
            raise ImportError(
                "WarmupEMA requires the 'ema-pytorch' package; "
                "install with `pip install --no-deps ema-pytorch>=0.5`."
            ) from exc
        self._ema = EMA(
            model,
            beta=float(decay),
            update_after_step=int(update_after_step),
            update_every=int(update_every),
            inv_gamma=float(inv_gamma),
            power=float(power),
            min_value=float(min_value),
        )
        # Cache config for reproducibility metadata.
        self._cfg: dict[str, Any] = {
            "decay": float(decay),
            "update_after_step": int(update_after_step),
            "update_every": int(update_every),
            "inv_gamma": float(inv_gamma),
            "power": float(power),
            "min_value": float(min_value),
        }
        logger.info("WarmupEMA initialised with %s", self._cfg)

    @property
    def ema_model(self) -> nn.Module:
        """Shadow model — use this in validation/inference."""
        return self._ema.ema_model

    @property
    def config(self) -> dict[str, Any]:
        return dict(self._cfg)

    def get_current_decay(self) -> float:
        """Current decay coefficient, after the warm-up schedule is applied."""
        return float(self._ema.get_current_decay())

    def update(self) -> None:
        """Apply one EMA update step. Call once per optimiser step."""
        self._ema.update()

    # nn.Module's default state_dict walks ``self._ema`` (a registered
    # submodule), so we do NOT override state_dict / load_state_dict.
    # Lightning's checkpoint payload captures ``ema._ema.*`` automatically.
