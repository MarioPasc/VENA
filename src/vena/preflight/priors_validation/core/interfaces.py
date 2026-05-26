"""Interfaces for validation tests (spec §2.1).

A :class:`ValidationTest` consumes a :class:`SubjectInputs`, possibly with
additional context provided via ``with_context``, and emits an iterable of
:class:`TestOutcome` records. Tests must be stateless across subjects; any
per-cohort state lives in the :class:`TestRunner`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import ClassVar

from .dataclasses import SubjectInputs, TestOutcome


class ValidationTest(ABC):
    """Contract for one of the five tests in the protocol panel."""

    test_id: ClassVar[str]
    """Stable identifier (e.g. ``"T1_range_sanity"``). Used in TestOutcome rows."""

    name: ClassVar[str]
    """Short human-readable name (e.g. ``"T1 range sanity"``)."""

    @abstractmethod
    def applicable(self, inputs: SubjectInputs) -> bool:
        """Return True iff this test has enough data to produce any outcome."""

    @abstractmethod
    def run(self, inputs: SubjectInputs) -> Iterable[TestOutcome]:
        """Run the test and yield one :class:`TestOutcome` per assertion cell."""
