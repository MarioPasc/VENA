"""Test T2 — anatomical localisation (spec §5.2).

For each prior, compute the median per ROI in the expected ordering list,
then Spearman-rank-correlate observed magnitudes against the expected rank
(strictly decreasing). Pass: ρ ≥ T2_RHO_MIN.

Operates on the raw physical priors *and* the derived NAWM-relative ones
that preserve a meaningful ordering by construction (``adc_rel``).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar

import numpy as np
from scipy.stats import spearmanr

from ..core.config import T2_EXPECTED_ORDERINGS, T2_RHO_MIN
from ..core.dataclasses import SubjectInputs, TestOutcome
from ..core.interfaces import ValidationTest
from .base import TestContext, safe_median
from .t1_range_sanity import _prior_volume_for, _roi_mask_for_t1


def _derived_volume(prior: str, inputs: SubjectInputs) -> np.ndarray | None:
    arr = inputs.derived_priors.get(prior)
    return None if arr is None else np.asarray(arr.array, dtype=np.float32)


class T2AtlasLocalisation(ValidationTest):
    """Per-prior Spearman ρ between expected and observed ROI magnitudes."""

    test_id: ClassVar[str] = "T2_atlas_localisation"
    name: ClassVar[str] = "T2 atlas localisation"

    def applicable(self, inputs: SubjectInputs) -> bool:
        return any(
            (p in inputs.derived_priors)
            or (getattr(inputs, p, None) is not None)
            for p in T2_EXPECTED_ORDERINGS
        )

    def run(self, inputs: SubjectInputs, ctx: TestContext | None = None) -> Iterable[TestOutcome]:  # type: ignore[override]
        if ctx is None:
            raise RuntimeError("T2AtlasLocalisation requires a TestContext")

        for prior, ordering in T2_EXPECTED_ORDERINGS.items():
            arr = _prior_volume_for(prior, inputs)
            if arr is None:
                arr = _derived_volume(prior, inputs)
            if arr is None:
                yield TestOutcome(
                    test_id=self.test_id,
                    subject_id=inputs.subject_id,
                    prior_id=prior,
                    roi_id=None,
                    metric_name=f"rank_corr_{prior}",
                    metric_value=None,
                    threshold=T2_RHO_MIN,
                    passed=False,
                    severity="info",
                    diagnostic=f"prior {prior!r} not provided",
                )
                continue

            observed: list[tuple[str, float]] = []
            missing: list[str] = []
            for roi_id in ordering:
                mask, why = _roi_mask_for_t1(ctx, roi_id)
                if mask is None or not mask.any():
                    missing.append(roi_id)
                    continue
                val = safe_median(arr, mask)
                if np.isfinite(val):
                    observed.append((roi_id, float(val)))

            if len(observed) < 3:
                yield TestOutcome(
                    test_id=self.test_id,
                    subject_id=inputs.subject_id,
                    prior_id=prior,
                    roi_id=None,
                    metric_name=f"rank_corr_{prior}",
                    metric_value=None,
                    threshold=T2_RHO_MIN,
                    passed=False,
                    severity="info",
                    diagnostic=(
                        f"only {len(observed)} ROI(s) available out of {len(ordering)}; "
                        f"missing: {missing}"
                    ),
                )
                continue

            # Expected ranks: ordering high→low ⇒ rank N (=highest) for the first entry.
            roi_to_rank = {
                roi_id: float(len(ordering) - idx)
                for idx, roi_id in enumerate(ordering)
            }
            expected_ranks = np.array([roi_to_rank[r] for r, _ in observed])
            observed_values = np.array([v for _, v in observed])
            res = spearmanr(expected_ranks, observed_values)
            rho = float(res.statistic) if np.isfinite(res.statistic) else float("nan")
            passed = np.isfinite(rho) and rho >= T2_RHO_MIN

            diag = (
                f"observed order: {[r for r, _ in sorted(observed, key=lambda kv: kv[1], reverse=True)]}; "
                f"expected order (high→low): {ordering[:len(observed)]}"
            )
            if missing:
                diag += f"; missing: {missing}"

            yield TestOutcome(
                test_id=self.test_id,
                subject_id=inputs.subject_id,
                prior_id=prior,
                roi_id=None,
                metric_name=f"rank_corr_{prior}",
                metric_value=rho,
                threshold=T2_RHO_MIN,
                passed=bool(passed),
                severity="error" if not passed else "info",
                diagnostic=diag,
            )
