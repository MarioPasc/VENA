"""Test T4 — cross-modal physical-coupling checks (spec §5.4).

Three coupling tests:
1. ITSS–CBF inside tumour mask (uses ``sus``↔``cbf`` in sub-A).
2. cellularity–CBF inside tumour, stratified by WHO grade.
3. calcification–enhancement anti-coupling (``chi_neg`` channel) — applicable
   only when QSM data is available (v0: never).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar

import numpy as np

from ..core.config import (
    T4_CELL_CBF_RHO_MIN_HGG,
    T4_CELL_CBF_RHO_MIN_LGG,
    T4_ITSS_CBF_RHO_MIN,
)
from ..core.dataclasses import SubjectInputs, TestOutcome
from ..core.interfaces import ValidationTest
from ..statistics import spearman_with_bootstrap_ci
from .base import TestContext


def _voxels_in_mask(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return arr[mask].astype(np.float64)


class T4CrossModal(ValidationTest):
    """Three coupling assertions per spec §5.4."""

    test_id: ClassVar[str] = "T4_cross_modal"
    name: ClassVar[str] = "T4 cross-modal coupling"

    def __init__(self, n_boot: int = 1000, seed: int = 1337) -> None:
        self.n_boot = int(n_boot)
        self.seed = int(seed)

    def applicable(self, inputs: SubjectInputs) -> bool:
        # At least one of the coupling pairs has both ends present.
        priors = set(inputs.derived_priors.keys())
        return ("cbf" in priors and "sus" in priors) or ("cbf" in priors and "cell" in priors)

    def run(self, inputs: SubjectInputs, ctx: TestContext | None = None) -> Iterable[TestOutcome]:  # type: ignore[override]
        if ctx is None:
            raise RuntimeError("T4 requires a TestContext")
        sub = inputs
        if sub.tumour_mask is None:
            yield TestOutcome(
                test_id=self.test_id,
                subject_id=sub.subject_id,
                prior_id=None,
                roi_id="tumour",
                metric_name="t4_status",
                metric_value=None,
                threshold=None,
                passed=False,
                severity="info",
                diagnostic="tumour mask absent — T4 not applicable",
            )
            return

        tumour = (np.asarray(sub.tumour_mask.array) > 0) & (np.asarray(sub.brain_mask.array) > 0)
        if tumour.sum() < 50:
            yield TestOutcome(
                test_id=self.test_id,
                subject_id=sub.subject_id,
                prior_id=None,
                roi_id="tumour",
                metric_name="t4_status",
                metric_value=None,
                threshold=None,
                passed=False,
                severity="info",
                diagnostic=f"tumour mask has only {int(tumour.sum())} voxels",
            )
            return

        cbf = sub.derived_priors.get("cbf")
        sus = sub.derived_priors.get("sus")
        cell = sub.derived_priors.get("cell")

        # ---- T4-1: ITSS / sus ↔ CBF inside tumour ----
        if cbf is not None and sus is not None:
            x = _voxels_in_mask(np.asarray(sus.array), tumour)
            y = _voxels_in_mask(np.asarray(cbf.array), tumour)
            res = spearman_with_bootstrap_ci(x, y, n_boot=self.n_boot, seed=self.seed)
            passed = (
                np.isfinite(res.rho)
                and res.rho >= T4_ITSS_CBF_RHO_MIN
                and (res.rho_lo > 0 if np.isfinite(res.rho_lo) else False)
            )
            yield TestOutcome(
                test_id=self.test_id,
                subject_id=sub.subject_id,
                prior_id="sus|cbf",
                roi_id="tumour",
                metric_name="spearman_sus_cbf_tumour",
                metric_value=res.rho,
                threshold=T4_ITSS_CBF_RHO_MIN,
                passed=bool(passed),
                severity="warning" if not passed else "info",
                diagnostic=(
                    f"ρ(sus, cbf | tumour) = {res.rho:.3f} "
                    f"[95% CI {res.rho_lo:.3f}, {res.rho_hi:.3f}], n={res.n}; "
                    f"expected ≥ {T4_ITSS_CBF_RHO_MIN}"
                ),
                extras={"n": res.n, "ci": (res.rho_lo, res.rho_hi)},
            )

        # ---- T4-2: cellularity ↔ CBF inside tumour, stratified by WHO grade ----
        if cbf is not None and cell is not None:
            x = _voxels_in_mask(np.asarray(cell.array), tumour)
            y = _voxels_in_mask(np.asarray(cbf.array), tumour)
            res = spearman_with_bootstrap_ci(x, y, n_boot=self.n_boot, seed=self.seed)
            grade = sub.metadata.who_grade
            is_hgg = grade is not None and grade >= 3
            threshold = T4_CELL_CBF_RHO_MIN_HGG if is_hgg else T4_CELL_CBF_RHO_MIN_LGG
            passed = np.isfinite(res.rho) and res.rho >= threshold
            stratum = "HGG" if is_hgg else "LGG" if grade is not None else "ungraded"
            yield TestOutcome(
                test_id=self.test_id,
                subject_id=sub.subject_id,
                prior_id="cell|cbf",
                roi_id="tumour",
                metric_name=f"spearman_cell_cbf_tumour_{stratum.lower()}",
                metric_value=res.rho,
                threshold=threshold,
                passed=bool(passed),
                severity="warning" if not passed else "info",
                diagnostic=(
                    f"ρ(cell, cbf | tumour, {stratum}) = {res.rho:.3f} "
                    f"[95% CI {res.rho_lo:.3f}, {res.rho_hi:.3f}], n={res.n}; "
                    f"expected ≥ {threshold} for {stratum}"
                ),
                extras={
                    "n": res.n,
                    "ci": (res.rho_lo, res.rho_hi),
                    "who_grade": grade,
                    "stratum": stratum,
                },
            )

        # T4-3 (calcification anti-coupling) requires chi_neg (QSM), absent in v0.
        yield TestOutcome(
            test_id=self.test_id,
            subject_id=sub.subject_id,
            prior_id="chi_neg|delta_t1",
            roi_id="tumour",
            metric_name="calcification_antico",
            metric_value=None,
            threshold=None,
            passed=False,
            severity="info",
            diagnostic="QSM chi_neg channel not produced in v0 (no phase data)",
        )
