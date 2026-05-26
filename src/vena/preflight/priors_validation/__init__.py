"""Priors validation preflight (``priors_validation_protocol.md`` v0.1.2).

Five-test panel verifying that the soft conditioning channels emitted by
:mod:`vena.prior_maps` carry the physical content they are designed to carry,
*before* they enter the latent rectified-flow training trunk.

Public entrypoints:

* :class:`vena.preflight.priors_validation.engine.PriorsValidationEngine`
* :class:`vena.preflight.priors_validation.engine.PriorsValidationRoutineConfig`
"""

from __future__ import annotations

from .core.dataclasses import (
    AtlasSpec,
    CohortReport,
    SubjectInputs,
    SubjectMetadata,
    TestOutcome,
    ValidationResult,
)
from .core.exceptions import (
    AtlasRegistrationError,
    InsufficientCohortError,
    InvalidThresholdError,
    PriorMissingError,
    ValidationException,
)
from .core.interfaces import ValidationTest
from .engine import PriorsValidationEngine, PriorsValidationRoutineConfig

__all__ = [
    "AtlasRegistrationError",
    "AtlasSpec",
    "CohortReport",
    "InsufficientCohortError",
    "InvalidThresholdError",
    "PriorMissingError",
    "PriorsValidationEngine",
    "PriorsValidationRoutineConfig",
    "SubjectInputs",
    "SubjectMetadata",
    "TestOutcome",
    "ValidationException",
    "ValidationResult",
    "ValidationTest",
]
