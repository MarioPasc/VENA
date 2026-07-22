"""Latent-space mask derivation for vena.segmentation (task 16).

Public API
----------
- :class:`ClassTemperatures` — frozen dataclass holding T_WT and T_NETC.
- :func:`fit_temperatures` — per-class NLL minimisation on calib split.
- :func:`apply_temperature` — ``sigmoid(logit / T)``, argmax-preserving.
- :func:`pool_to_latent` — crop-then-avg-pool to ``(2, 48, 56, 48)``.
- :func:`ensemble_soft` — K-fold mean (+ optional k-fold disagreement std).
"""

from __future__ import annotations

from vena.segmentation.derivation.ensemble import ensemble_soft
from vena.segmentation.derivation.pool import pool_to_latent
from vena.segmentation.derivation.temperature import (
    ClassTemperatures,
    apply_temperature,
    fit_temperatures,
)

__all__ = [
    "ClassTemperatures",
    "apply_temperature",
    "ensemble_soft",
    "fit_temperatures",
    "pool_to_latent",
]
