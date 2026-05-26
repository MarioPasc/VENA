"""Test T1 — quantitative range sanity (spec §5.1).

Operates on the raw physical-units inputs (``cbf``, ``adc``, ``chi``,
``swan_mag``). Per-(prior, ROI) median compared against literature ranges
in ``core.config.RANGE_TABLE``. Diagnostic hints are emitted on failure
per ``DIAGNOSTIC_HINTS``.

The ``cbf``/``adc``/``chi`` keys in this test refer to UCSF-PDGM raw
modality slots: ``cbf`` ← ``ASL.nii.gz``, ``adc`` ← ``ADC.nii.gz``, ``chi``
is universally ``None`` in v0 (no phase data). ``swan_mag`` is not in the
range table — it is arbitrary-unit and only constrained by T2 / T3.

UCSF-PDGM ADC scl_slope quirk: when the median of ADC inside the brain
mask is < 1e-6 (i.e. orders of magnitude below the expected ~10^-3 mm²/s),
the test downgrades to ``severity="warning"`` rather than ``"error"`` and
emits a diagnostic naming the cause. T3/T4 still consume the NAWM-relative
``adc_rel`` channel where the unit cancels.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar

import numpy as np

from ..core.config import DIAGNOSTIC_HINTS, RANGE_TABLE, RangeThreshold
from ..core.dataclasses import SubjectInputs, TestOutcome
from ..core.interfaces import ValidationTest
from .base import TestContext, roi_mask_from_atlas, safe_median

# Map (prior, roi) → atlas-registry ROI id used to slice the atlas label map.
_T1_PRIOR_ROI_TO_ATLAS_ROI: dict[tuple[str, str], str] = {
    ("cbf", "cortical_gm"): "cortical_gm",
    ("cbf", "nawm"): "nawm",
    ("cbf", "whole_brain"): "whole_brain",  # synthetic — full brain mask
    ("cbf", "cerebellum"): "cerebellum",  # not in HO; will skip
    ("cbf", "hgg_tumour_core"): "tumour",  # subject's tumour mask
    ("adc", "nawm"): "nawm",
    ("adc", "cortical_gm"): "cortical_gm",
    ("adc", "ventricles"): "ventricles",
    ("adc", "hgg_cellular"): "tumour",
    ("adc", "necrotic_core"): "tumour",  # uses tumour mask + ADC > threshold
    ("chi", "nawm"): "nawm",
    ("chi", "cortical_gm"): "cortical_gm",
    ("chi", "globus_pallidus"): "globus_pallidus",
    ("chi", "venous_sinus"): "sinus",
}


def _prior_volume_for(name: str, inputs: SubjectInputs) -> np.ndarray | None:
    """Resolve a raw-physical prior name to its NumPy array on the subject."""
    vol = {"cbf": inputs.cbf, "adc": inputs.adc, "chi": inputs.chi}.get(name)
    return None if vol is None else np.asarray(vol.array, dtype=np.float32)


def _roi_mask_for_t1(ctx: TestContext, atlas_roi_id: str) -> tuple[np.ndarray | None, str | None]:
    """Build the binary ROI mask for one T1 cell in subject space."""
    sub = ctx.subject
    brain = np.asarray(sub.brain_mask.array) > 0
    if atlas_roi_id == "whole_brain":
        return brain, None
    if atlas_roi_id == "tumour":
        if sub.tumour_mask is None:
            return None, "tumour mask absent"
        return (np.asarray(sub.tumour_mask.array) > 0) & brain, None
    if atlas_roi_id == "nawm":
        return ctx.nawm_mask, None if ctx.nawm_mask is not None else "NAWM mask not built"
    if atlas_roi_id == "ventricles":
        if ctx.ventricle_mask is None:
            return None, "ventricle mask not built"
        return ctx.ventricle_mask & brain, None
    # Other ROIs come from atlas-warped HO subcortical labels
    from ..atlases.registry import ATLAS_REGISTRY

    if atlas_roi_id not in ATLAS_REGISTRY:
        return None, f"ROI {atlas_roi_id!r} not in atlas registry"
    spec = ATLAS_REGISTRY[atlas_roi_id]
    if spec.atlas_id not in ctx.atlas_labels:
        return None, f"atlas {spec.atlas_id} not warped for this subject"
    return (
        roi_mask_from_atlas(ctx.atlas_labels[spec.atlas_id], spec.label_values, extra_mask=brain),
        None,
    )


def _adc_scale_quirk(adc_arr: np.ndarray, brain_mask: np.ndarray) -> bool:
    """Detect the UCSF-PDGM ADC per-file scl_slope quirk."""
    in_brain = adc_arr[brain_mask > 0]
    if in_brain.size == 0:
        return False
    return float(np.median(in_brain)) < 1e-6


def _diagnostic_for(prior: str, roi: str, value: float, t: RangeThreshold) -> str:
    if value < t.lo:
        hint = DIAGNOSTIC_HINTS.get((prior, roi, "below"))
        side = f"below lower bound {t.lo} {t.unit}"
    else:
        hint = DIAGNOSTIC_HINTS.get((prior, roi, "above"))
        side = f"above upper bound {t.hi} {t.unit}"
    if hint:
        return f"median {value:.3g} {t.unit} {side}; {hint}"
    return f"median {value:.3g} {t.unit} {side}"


class T1RangeSanity(ValidationTest):
    """Per-(prior, ROI) median falls within literature acceptable range."""

    test_id: ClassVar[str] = "T1_range_sanity"
    name: ClassVar[str] = "T1 range sanity"

    def __init__(self) -> None:
        self._cells: list[tuple[str, str, RangeThreshold]] = []
        for prior, roi_table in RANGE_TABLE.items():
            for roi_id, threshold in roi_table.items():
                self._cells.append((prior, roi_id, threshold))

    def applicable(self, inputs: SubjectInputs) -> bool:
        # At least one of CBF / ADC must be present.
        return any(getattr(inputs, k) is not None for k in ("cbf", "adc", "chi"))

    def run(self, inputs: SubjectInputs, ctx: TestContext | None = None) -> Iterable[TestOutcome]:  # type: ignore[override]
        if ctx is None:
            raise RuntimeError("T1RangeSanity requires a TestContext")
        sub = inputs
        brain = np.asarray(sub.brain_mask.array) > 0

        # Detect the ADC scl_slope quirk once so we can downgrade ADC outcomes
        adc_arr = _prior_volume_for("adc", sub)
        adc_quirk = adc_arr is not None and _adc_scale_quirk(adc_arr, brain)

        for prior, roi_id, threshold in self._cells:
            prior_arr = _prior_volume_for(prior, sub)
            if prior_arr is None:
                yield TestOutcome(
                    test_id=self.test_id,
                    subject_id=sub.subject_id,
                    prior_id=prior,
                    roi_id=roi_id,
                    metric_name=f"median_{prior}_{roi_id}",
                    metric_value=None,
                    threshold=(threshold.lo, threshold.hi),
                    passed=False,
                    severity="info",
                    diagnostic=f"prior {prior!r} not provided for this subject",
                )
                continue

            atlas_roi_id = _T1_PRIOR_ROI_TO_ATLAS_ROI.get((prior, roi_id), roi_id)
            mask, why = _roi_mask_for_t1(ctx, atlas_roi_id)
            if mask is None:
                yield TestOutcome(
                    test_id=self.test_id,
                    subject_id=sub.subject_id,
                    prior_id=prior,
                    roi_id=roi_id,
                    metric_name=f"median_{prior}_{roi_id}",
                    metric_value=None,
                    threshold=(threshold.lo, threshold.hi),
                    passed=False,
                    severity="info",
                    diagnostic=f"ROI mask unavailable: {why}",
                )
                continue

            value = safe_median(prior_arr, mask)
            if not np.isfinite(value):
                yield TestOutcome(
                    test_id=self.test_id,
                    subject_id=sub.subject_id,
                    prior_id=prior,
                    roi_id=roi_id,
                    metric_name=f"median_{prior}_{roi_id}",
                    metric_value=None,
                    threshold=(threshold.lo, threshold.hi),
                    passed=False,
                    severity="warning",
                    diagnostic="ROI mask empty in subject space",
                )
                continue

            # ADC needs to be scaled to the literature unit. Spec §4.2 gives
            # the threshold in 10^-3 mm²/s; on-disk ADC is in mm²/s for
            # well-scaled subjects. Multiply by 1e3 to compare apples to apples.
            scale = 1e3 if prior == "adc" else 1.0
            value_scaled = value * scale

            passed = threshold.lo <= value_scaled <= threshold.hi
            severity = "error"
            diagnostic = (
                "within acceptable range"
                if passed
                else _diagnostic_for(prior, roi_id, value_scaled, threshold)
            )
            if prior == "adc" and adc_quirk:
                severity = "warning"
                diagnostic = (
                    f"UCSF-PDGM ADC per-file scl_slope quirk detected "
                    f"(median {value:.3g}); raw value not in mm²/s. "
                    "Downstream tests use the NAWM-relative adc_rel channel."
                )

            yield TestOutcome(
                test_id=self.test_id,
                subject_id=sub.subject_id,
                prior_id=prior,
                roi_id=roi_id,
                metric_name=f"median_{prior}_{roi_id}",
                metric_value=float(value_scaled),
                threshold=(threshold.lo, threshold.hi),
                passed=passed if severity != "warning" else True,
                severity=severity,
                diagnostic=diagnostic,
            )
