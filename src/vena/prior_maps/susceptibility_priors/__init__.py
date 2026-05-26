"""Susceptibility-prior extraction subsystem (SWAN magnitude → sus / itss).

`soft_priors_sources.md` §4.3 sub-option A. Sub-options B (QSM) and C
(χ-separation) are out of scope until phase data is available.
"""

from __future__ import annotations

from .abc_model import (
    REQUIRED_CHANNELS,
    AbstractSusceptibilityModel,
    PriorOutput,
    SusceptibilityInput,
    SusceptibilityPriorError,
)
from .engine import (
    AlgorithmSpec,
    SusceptibilityPriorsEngine,
    SusceptibilityPriorsRoutineConfig,
)
from .models import MODEL_REGISTRY

__all__ = [
    "MODEL_REGISTRY",
    "REQUIRED_CHANNELS",
    "AbstractSusceptibilityModel",
    "AlgorithmSpec",
    "PriorOutput",
    "SusceptibilityInput",
    "SusceptibilityPriorError",
    "SusceptibilityPriorsEngine",
    "SusceptibilityPriorsRoutineConfig",
]
