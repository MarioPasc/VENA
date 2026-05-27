"""Exponential moving average of the ControlNet weights.

Thin adapter around :mod:`ema_pytorch` (lucidrains, MIT) configured with a
Karras-2024-style warmup. The shadow model is exposed as
:attr:`WarmupEMA.ema_model` so validation can run on the EMA weights directly
without manually swapping the trainable copy.
"""

from .warmup_ema import WarmupEMA

__all__ = ["WarmupEMA"]
