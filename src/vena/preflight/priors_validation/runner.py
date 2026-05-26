"""Per-subject + cohort orchestrator with error containment.

The :class:`TestRunner` wires preprocessing + the five tests together. Per
spec §8.4, a failure in any test is recorded as an outcome (severity
``error``) but never crashes the routine for other subjects.
``AtlasRegistrationError`` aborts the *subject's* remaining tests but the
cohort run continues.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from .atlases import ensure_atlases
from .atlases.fetch import AtlasBundle
from .atlases.registry import HO_SUB, VENOUS_INHOUSE
from .core.config import (
    COHORT_PASS_RATE_T1,
    COHORT_PASS_RATE_T2,
    COHORT_PASS_RATE_T3,
    COHORT_PASS_RATE_T4,
    EFFECT_SIZE_MIN_FOR_INFORMATIVE,
    ROUTINE_VERSION,
    TRAINING_CLEARANCE_THRESHOLD_DEFAULT,
)
from .core.dataclasses import (
    CohortReport,
    PriorClearance,
    SubjectInputs,
    TestOutcome,
    ValidationResult,
)
from .core.exceptions import AtlasRegistrationError, ValidationException
from .core.interfaces import ValidationTest
from .preprocessing import (
    build_nawm_mask,
    build_ventricle_mask,
    register_mni_to_subject,
    robust_zscore,
    warp_label_to_subject,
)
from .preprocessing.atlas import RegistrationKind
from .tests.base import TestContext, roi_mask_from_atlas

logger = logging.getLogger(__name__)


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    """Sørensen–Dice on two boolean masks."""
    a = np.asarray(a) > 0
    b = np.asarray(b) > 0
    denom = a.sum() + b.sum()
    if denom == 0:
        return float("nan")
    return float(2 * (a & b).sum() / denom)


def _build_context(
    subject: SubjectInputs,
    bundle: AtlasBundle,
    cache_root: Path,
    *,
    registration_kind: RegistrationKind = "affine",
) -> TestContext:
    """Per-subject pre-computation: atlas warps, NAWM, ΔT1, registration QC."""
    warp = register_mni_to_subject(
        subject.t1pre,
        bundle.mni152_t1w,
        cache_root,
        kind=registration_kind,
        subject_id=subject.subject_id,
    )
    atlas_labels: dict[str, np.ndarray] = {}
    atlas_labels[HO_SUB] = warp_label_to_subject(warp, bundle.ho_sub_path)
    if bundle.venous_inhouse is not None:
        atlas_labels[VENOUS_INHOUSE] = warp_label_to_subject(warp, bundle.venous_inhouse)

    brain = np.asarray(subject.brain_mask.array) > 0
    parenchyma = (
        np.asarray(subject.parenchyma_mask.array) > 0
        if subject.parenchyma_mask is not None
        else None
    )
    tumour = np.asarray(subject.tumour_mask.array) > 0 if subject.tumour_mask is not None else None
    from .atlases.registry import ATLAS_REGISTRY

    atlas_wm = roi_mask_from_atlas(
        atlas_labels[HO_SUB], ATLAS_REGISTRY["nawm"].label_values, extra_mask=brain
    )
    atlas_ventr = roi_mask_from_atlas(
        atlas_labels[HO_SUB],
        ATLAS_REGISTRY["ventricles"].label_values,
        extra_mask=brain,
    )
    nawm = build_nawm_mask(parenchyma, tumour, atlas_wm, brain)
    ventricles = build_ventricle_mask(parenchyma, brain, atlas_ventr)

    # Registration QC — Dice between the warped MNI brain template (binarised)
    # and the subject brain mask. The original spec wanted FAST-segmented
    # ventricles vs warped atlas ventricles, but UCSF-PDGM ships no FAST
    # output; the "brain ∖ parenchyma" CSF proxy includes sulcal CSF and is
    # much larger than the lateral-ventricle ROI, biasing the Dice low by
    # construction. Switching to the warped-brain Dice is the standard
    # registration QC metric (Avants et al. *Insight J* 2009) and is robust
    # to the absence of explicit ventricle segmentations.
    warped_brain = warp_label_to_subject(warp, bundle.mni152_t1w) > 0
    dice = _dice(warped_brain, brain)
    logger.info(
        "[%s] registration QC Dice (warped MNI brain vs subject brain): %.3f",
        subject.subject_id,
        dice,
    )

    # ΔT1 = zscore(T1c) − zscore(T1pre) per spec §5.3
    delta_t1 = (
        robust_zscore(np.asarray(subject.t1gd.array), brain)
        - robust_zscore(np.asarray(subject.t1pre.array), brain)
    ).astype(np.float32)

    return TestContext(
        subject=subject,
        atlas_labels=atlas_labels,
        nawm_mask=nawm,
        ventricle_mask=ventricles,
        delta_t1=delta_t1,
        atlas_registration_dice=dice,
    )


def _run_tests_for_subject(
    subject: SubjectInputs,
    tests: Sequence[ValidationTest],
    bundle: AtlasBundle,
    cache_root: Path,
    *,
    registration_kind: RegistrationKind = "affine",
    dice_error_threshold: float = 0.5,
    dice_warning_threshold: float = 0.7,
) -> ValidationResult:
    """Run all applicable tests for one subject, with error containment."""
    outcomes: list[TestOutcome] = []
    aborted = False
    abort_reason: str | None = None
    try:
        ctx = _build_context(subject, bundle, cache_root, registration_kind=registration_kind)
    except Exception as exc:
        logger.exception("[%s] failed to build TestContext", subject.subject_id)
        return ValidationResult(
            subject_id=subject.subject_id,
            outcomes=(
                TestOutcome(
                    test_id="bootstrap",
                    subject_id=subject.subject_id,
                    prior_id=None,
                    roi_id=None,
                    metric_name="context_build",
                    metric_value=None,
                    threshold=None,
                    passed=False,
                    severity="error",
                    diagnostic=f"context build failed: {exc}",
                ),
            ),
            overall_passed=False,
            failed_priors=frozenset(),
            aborted=True,
            abort_reason=f"context: {exc}",
        )

    # Atlas registration QC gate (spec §3 + §10.3)
    dice = ctx.atlas_registration_dice or 0.0
    if dice < dice_error_threshold:
        return ValidationResult(
            subject_id=subject.subject_id,
            outcomes=(
                TestOutcome(
                    test_id="bootstrap",
                    subject_id=subject.subject_id,
                    prior_id=None,
                    roi_id="ventricles",
                    metric_name="atlas_registration_dice",
                    metric_value=float(dice),
                    threshold=dice_error_threshold,
                    passed=False,
                    severity="error",
                    diagnostic=f"atlas registration Dice {dice:.2f} below abort "
                    f"threshold {dice_error_threshold}; remaining tests skipped",
                ),
            ),
            overall_passed=False,
            failed_priors=frozenset(),
            aborted=True,
            abort_reason=f"atlas_registration_dice={dice:.2f}",
        )
    if dice < dice_warning_threshold:
        outcomes.append(
            TestOutcome(
                test_id="bootstrap",
                subject_id=subject.subject_id,
                prior_id=None,
                roi_id="ventricles",
                metric_name="atlas_registration_dice",
                metric_value=float(dice),
                threshold=dice_warning_threshold,
                passed=False,
                severity="warning",
                diagnostic=f"atlas registration Dice {dice:.2f} below warning "
                f"threshold {dice_warning_threshold}; results may be unreliable",
            )
        )

    for test in tests:
        try:
            if not test.applicable(subject):
                outcomes.append(
                    TestOutcome(
                        test_id=test.test_id,
                        subject_id=subject.subject_id,
                        prior_id=None,
                        roi_id=None,
                        metric_name=f"{test.test_id}_applicable",
                        metric_value=None,
                        threshold=None,
                        passed=True,
                        severity="info",
                        diagnostic="test not applicable for this subject",
                    )
                )
                continue
            outcomes.extend(test.run(subject, ctx))  # type: ignore[arg-type]
        except AtlasRegistrationError as exc:
            aborted = True
            abort_reason = f"{test.test_id}: {exc}"
            outcomes.append(
                TestOutcome(
                    test_id=test.test_id,
                    subject_id=subject.subject_id,
                    prior_id=None,
                    roi_id=None,
                    metric_name=f"{test.test_id}_error",
                    metric_value=None,
                    threshold=None,
                    passed=False,
                    severity="error",
                    diagnostic=f"atlas registration failure: {exc}",
                )
            )
            break
        except ValidationException as exc:
            outcomes.append(
                TestOutcome(
                    test_id=test.test_id,
                    subject_id=subject.subject_id,
                    prior_id=exc.prior_id,
                    roi_id=None,
                    metric_name=f"{test.test_id}_error",
                    metric_value=None,
                    threshold=None,
                    passed=False,
                    severity="error",
                    diagnostic=f"{type(exc).__name__}: {exc.message}",
                )
            )
        except Exception as exc:
            logger.exception("[%s] unexpected error in %s", subject.subject_id, test.test_id)
            outcomes.append(
                TestOutcome(
                    test_id=test.test_id,
                    subject_id=subject.subject_id,
                    prior_id=None,
                    roi_id=None,
                    metric_name=f"{test.test_id}_error",
                    metric_value=None,
                    threshold=None,
                    passed=False,
                    severity="error",
                    diagnostic=f"unexpected exception: {type(exc).__name__}: {exc}",
                )
            )

    failed_priors = frozenset(o.prior_id for o in outcomes if o.severity == "error" and o.prior_id)
    error_outcomes = [o for o in outcomes if o.severity == "error" and not o.passed]
    overall_passed = (not aborted) and (len(error_outcomes) == 0)

    return ValidationResult(
        subject_id=subject.subject_id,
        outcomes=tuple(outcomes),
        overall_passed=overall_passed,
        failed_priors=failed_priors,
        aborted=aborted,
        abort_reason=abort_reason,
    )


class TestRunner:
    """Run every test on every subject, with per-subject parallelism."""

    def __init__(
        self,
        tests: Sequence[ValidationTest],
        atlases_root: Path,
        cache_root: Path,
        *,
        venous_inhouse_path: Path | None = None,
        registration_kind: RegistrationKind = "affine",
        n_workers: int = 1,
        training_clearance_threshold: float = TRAINING_CLEARANCE_THRESHOLD_DEFAULT,
    ) -> None:
        self.tests = list(tests)
        self.atlases_root = Path(atlases_root)
        self.cache_root = Path(cache_root)
        self.venous_inhouse_path = venous_inhouse_path
        self.registration_kind: RegistrationKind = registration_kind
        self.n_workers = int(n_workers)
        self.training_clearance_threshold = float(training_clearance_threshold)
        self._bundle: AtlasBundle | None = None

    @property
    def bundle(self) -> AtlasBundle:
        if self._bundle is None:
            self._bundle = ensure_atlases(
                self.atlases_root, venous_inhouse_path=self.venous_inhouse_path
            )
        return self._bundle

    def run(self, subjects: Iterable[SubjectInputs]) -> CohortReport:
        subjects = list(subjects)
        bundle = self.bundle  # eagerly fetch atlases before fanning out
        t0 = time.time()
        if self.n_workers <= 1:
            results = [
                _run_tests_for_subject(
                    s,
                    self.tests,
                    bundle,
                    self.cache_root,
                    registration_kind=self.registration_kind,
                )
                for s in subjects
            ]
        else:
            results = Parallel(n_jobs=self.n_workers, backend="loky")(
                delayed(_run_tests_for_subject)(
                    s,
                    self.tests,
                    bundle,
                    self.cache_root,
                    registration_kind=self.registration_kind,
                )
                for s in subjects
            )
        logger.info(
            "cohort run completed in %.1fs (%d subjects, %d workers)",
            time.time() - t0,
            len(subjects),
            self.n_workers,
        )
        return self._aggregate(results, bundle)

    def _aggregate(self, results: Sequence[ValidationResult], bundle: AtlasBundle) -> CohortReport:
        per_test_pass: dict[str, list[bool]] = {}
        per_prior_outcomes: dict[str, list[TestOutcome]] = {}
        warnings: list[str] = []

        for vr in results:
            for o in vr.outcomes:
                # Skip ``info`` cells for cohort pass-rate computation
                if o.severity == "info":
                    continue
                per_test_pass.setdefault(o.test_id, []).append(o.passed)
                if o.prior_id:
                    per_prior_outcomes.setdefault(o.prior_id, []).append(o)

        per_test_pass_rate: dict[str, float | None] = {}
        for test_id, flags in per_test_pass.items():
            per_test_pass_rate[test_id] = float(np.mean(flags)) if flags else None
        # Tests with zero applicable outcomes ⇒ None (e.g. T5)
        for test in self.tests:
            per_test_pass_rate.setdefault(test.test_id, None)

        per_prior_clearance: dict[str, PriorClearance] = {}
        for prior_id, outs in per_prior_outcomes.items():
            ok = [o for o in outs if o.severity != "error"]
            errors = [o for o in outs if o.severity == "error" and not o.passed]
            if errors and len(errors) >= max(1, len(outs) // 3):
                per_prior_clearance[prior_id] = "failed"
            elif any(o.severity == "warning" for o in outs):
                per_prior_clearance[prior_id] = "warning"
            elif ok:
                per_prior_clearance[prior_id] = "passed"
            else:
                per_prior_clearance[prior_id] = "not_evaluated"

        passes = [vr.overall_passed for vr in results]
        cohort_pass_rate = float(np.mean(passes)) if passes else 0.0

        # Per-test cohort thresholds (spec §§5.1–5.4) — flag if missed
        thresholds = {
            "T1_range_sanity": COHORT_PASS_RATE_T1,
            "T2_atlas_localisation": COHORT_PASS_RATE_T2,
            "T3_t1gd_coherence": COHORT_PASS_RATE_T3,
            "T4_cross_modal": COHORT_PASS_RATE_T4,
        }
        for tid, thr in thresholds.items():
            rate = per_test_pass_rate.get(tid)
            if rate is not None and rate < thr:
                warnings.append(f"{tid} cohort pass rate {rate:.2%} < spec target {thr:.0%}")

        training_clearance = cohort_pass_rate >= self.training_clearance_threshold

        atlas_versions = {
            "mni152_nlin2009c": "Fonov2011 (templateflow)",
            "harvard_oxford_cort": "FSL maxprob-thr25-1mm",
            "harvard_oxford_sub": "FSL maxprob-thr25-1mm",
            "venous_inhouse": (
                str(bundle.venous_inhouse) if bundle.venous_inhouse else "not_built"
            ),
        }

        return CohortReport(
            n_subjects=len(results),
            n_subjects_applicable=sum(1 for vr in results if not vr.aborted),
            per_test_pass_rate=per_test_pass_rate,
            per_prior_clearance=per_prior_clearance,
            cohort_pass_rate_overall=cohort_pass_rate,
            training_clearance=training_clearance,
            subjects=tuple(results),
            atlas_versions=atlas_versions,
            routine_version=ROUTINE_VERSION,
            warnings=tuple(warnings),
        )


# Re-exported here for caller convenience.
__all__ = ["EFFECT_SIZE_MIN_FOR_INFORMATIVE", "TestRunner"]
