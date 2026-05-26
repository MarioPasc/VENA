"""Cellularity-prior extraction subsystem (ADC → cell / adc_rel channels).

`soft_priors_sources.md` §4.2.
"""

from __future__ import annotations

from .abc_model import (
    REQUIRED_CHANNELS,
    AbstractCellularityModel,
    CellularityInput,
    CellularityPriorError,
    PriorOutput,
)
from .engine import (
    AlgorithmSpec,
    CellularityPriorsEngine,
    CellularityPriorsRoutineConfig,
)
from .models import MODEL_REGISTRY

__all__ = [
    "MODEL_REGISTRY",
    "REQUIRED_CHANNELS",
    "AbstractCellularityModel",
    "AlgorithmSpec",
    "CellularityInput",
    "CellularityPriorError",
    "CellularityPriorsEngine",
    "CellularityPriorsRoutineConfig",
    "PriorOutput",
]
