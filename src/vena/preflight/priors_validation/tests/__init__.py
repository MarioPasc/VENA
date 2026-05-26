"""Five validation tests (spec §5).

Tests are stateless; the test runner provides them with the per-subject
context (warped atlas masks, NAWM mask, etc.) via the :class:`TestContext`
helper in :mod:`base`.
"""

from __future__ import annotations

from .base import TestContext
from .t1_range_sanity import T1RangeSanity
from .t2_atlas_localisation import T2AtlasLocalisation
from .t3_t1gd_coherence import T3T1GdCoherence
from .t4_cross_modal import T4CrossModal
from .t5_reproducibility import T5Reproducibility

__all__ = [
    "T1RangeSanity",
    "T2AtlasLocalisation",
    "T3T1GdCoherence",
    "T4CrossModal",
    "T5Reproducibility",
    "TestContext",
]
