"""Perfusion-prior extraction subsystem (ASL → CBF conditioning channels).

`soft_priors_sources.md` §4.1 — emits the ``cbf_rel`` and ``cbf`` channels for
the latent flow-matching trunk.
"""

from __future__ import annotations

from .abc_model import (
    REQUIRED_CHANNELS,
    AbstractPerfusionModel,
    PerfusionInput,
    PerfusionPriorError,
    PriorOutput,
)
from .engine import (
    AlgorithmSpec,
    PerfusionPriorsEngine,
    PerfusionPriorsRoutineConfig,
)
from .models import MODEL_REGISTRY

__all__ = [
    "MODEL_REGISTRY",
    "REQUIRED_CHANNELS",
    "AbstractPerfusionModel",
    "AlgorithmSpec",
    "PerfusionInput",
    "PerfusionPriorError",
    "PerfusionPriorsEngine",
    "PerfusionPriorsRoutineConfig",
    "PriorOutput",
]
