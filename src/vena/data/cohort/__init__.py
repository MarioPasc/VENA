"""Cohort abstraction layer for VENA.

Provides a typed protocol :class:`CohortProtocol` and a decorator-based
:class:`CohortRegistry` so that adding a new pathology cohort (BraTS-MEN,
Málaga, …) is a < 100-line change that conforms to a documented contract
rather than a copy-paste of an existing reader.

See :mod:`vena.data.cohort.protocol` for the protocol definition and
``src/vena/data/cohort/HOWTO.md`` for the step-by-step recipe.
"""

from __future__ import annotations

from .protocol import (
    CohortPatient,
    CohortProtocol,
    Pathology,
)
from .registry import CohortRegistry, get_cohort_registry, register_cohort

__all__ = [
    "CohortPatient",
    "CohortProtocol",
    "CohortRegistry",
    "Pathology",
    "get_cohort_registry",
    "register_cohort",
]
