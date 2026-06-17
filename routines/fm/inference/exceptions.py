"""Routine-local exception hierarchy."""

from __future__ import annotations


class InferenceRoutineError(Exception):
    """Base class for routine-side failures."""


class InferenceConfigError(InferenceRoutineError):
    """Malformed YAML config or invalid filter."""


class ModelRegistryError(InferenceRoutineError):
    """Malformed models-YAML or unknown model type."""


class CohortFilterError(InferenceRoutineError):
    """A YAML-requested cohort is missing from the corpus registry."""


class ModelSetupError(InferenceRoutineError):
    """An adapter's ``setup()`` raised — re-wrapped for engine-level logging."""
