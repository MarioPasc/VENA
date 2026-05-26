"""Test T5 — test-retest reproducibility (spec §5.5).

Gated ``applicable=False`` whenever the subject has no ``repeat_scan_id`` in
its metadata (i.e. always False for UCSF-PDGM v0; activates for Málaga when
the manifest carries paired-scan ids).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar

from ..core.dataclasses import SubjectInputs, TestOutcome
from ..core.interfaces import ValidationTest
from .base import TestContext


class T5Reproducibility(ValidationTest):
    """ICC(2,1) per (prior, ROI) across paired scans of the same subject."""

    test_id: ClassVar[str] = "T5_reproducibility"
    name: ClassVar[str] = "T5 test-retest reproducibility"

    def applicable(self, inputs: SubjectInputs) -> bool:
        return inputs.metadata.repeat_scan_id is not None

    def run(self, inputs: SubjectInputs, ctx: TestContext | None = None) -> Iterable[TestOutcome]:  # type: ignore[override]
        # v0 stub: no UCSF-PDGM subject has paired scans, so applicable()
        # always returns False and this method is not invoked. When Málaga
        # lands, fetch the paired-scan derived priors and call icc_2_1 per
        # (prior, ROI) cell against the targets in core.config.T5_ICC_TARGETS.
        return iter(())
