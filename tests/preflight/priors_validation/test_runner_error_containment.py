"""TestRunner must contain per-subject errors without poisoning the cohort."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from unittest import mock

from vena.preflight.priors_validation.atlases.fetch import AtlasBundle
from vena.preflight.priors_validation.core.dataclasses import (
    SubjectInputs,
    TestOutcome,
)
from vena.preflight.priors_validation.core.interfaces import ValidationTest
from vena.preflight.priors_validation.runner import (
    _run_tests_for_subject,
)

from ._synthetic import build_synthetic_subject


class _AlwaysApplicableYieldsError(ValidationTest):
    test_id = "synth_raise"
    name = "synthetic raising test"

    def applicable(self, inputs: SubjectInputs) -> bool:
        return True

    def run(self, inputs: SubjectInputs, ctx=None) -> Iterable[TestOutcome]:
        raise RuntimeError("intentional explosion")


class _AlwaysApplicableYieldsOK(ValidationTest):
    test_id = "synth_ok"
    name = "synthetic OK test"

    def applicable(self, inputs: SubjectInputs) -> bool:
        return True

    def run(self, inputs: SubjectInputs, ctx=None) -> Iterable[TestOutcome]:
        yield TestOutcome(
            test_id=self.test_id,
            subject_id=inputs.subject_id,
            prior_id=None,
            roi_id=None,
            metric_name="ok",
            metric_value=1.0,
            threshold=None,
            passed=True,
            severity="info",
            diagnostic="ok",
        )


def test_runner_contains_test_exception():
    subj = build_synthetic_subject()
    fake_bundle = AtlasBundle(
        mni152_t1w=Path("/dev/null"),
        ho_cort_path=Path("/dev/null"),
        ho_sub_path=Path("/dev/null"),
        ho_cort_labels=[],
        ho_sub_labels=[],
        venous_inhouse=None,
        extras={},
    )
    # Patch context build so the runner does not try ANTsPy on a synthetic NIfTI.
    from ._synthetic import build_synthetic_context_from

    ctx = build_synthetic_context_from(subj)
    with mock.patch("vena.preflight.priors_validation.runner._build_context", return_value=ctx):
        result = _run_tests_for_subject(
            subj,
            [_AlwaysApplicableYieldsError(), _AlwaysApplicableYieldsOK()],
            fake_bundle,
            Path("/tmp/none"),
        )
    err_outs = [o for o in result.outcomes if o.test_id == "synth_raise"]
    ok_outs = [o for o in result.outcomes if o.test_id == "synth_ok"]
    assert err_outs, "expected an error outcome for the raising test"
    assert err_outs[0].severity == "error"
    assert ok_outs and ok_outs[0].passed, "second test should still run after the first one raises"
