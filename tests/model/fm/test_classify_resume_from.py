"""Unit tests for the ``resume_from`` classifier.

The classifier is the *only* surface that decides which of the three resume
modes (BASELINE / CONTINUE / WARM_START) a YAML value belongs to. Path
existence is not checked here — that's the resolver's job — so these tests
are pure string-shape coverage.
"""

from __future__ import annotations

import pytest

from routines.fm.train.engine import ResumeMode, _classify_resume_from
from routines.fm.train.exceptions import InvalidResumeFromError

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("rf", [None, "", "baseline"])
def test_baseline_keywords(rf: str | None) -> None:
    assert _classify_resume_from(rf) is ResumeMode.BASELINE


@pytest.mark.parametrize("rf", ["latest", "best"])
def test_continue_keywords(rf: str) -> None:
    assert _classify_resume_from(rf) is ResumeMode.CONTINUE


@pytest.mark.parametrize(
    "rf",
    [
        "2026-06-10_10-24-10_s1_fft_cfm_9441bf91",
        "2026-06-10_10-24-10_s2_lora_r16_contrastive_9441bf91",
        "2026-06-10_10-24-10_s2_lora_r16_contrastive_cfg_aaaaaaaa",
    ],
)
def test_warm_start_run_id(rf: str) -> None:
    assert _classify_resume_from(rf) is ResumeMode.WARM_START


def test_warm_start_absolute_path() -> None:
    assert _classify_resume_from("/abs/path/to/last.ckpt") is ResumeMode.WARM_START


@pytest.mark.parametrize(
    "rf",
    [
        "garbage",
        "2026-06-10",  # incomplete run_id
        "2026-06-10_10-24-10_s1_aaa",  # legacy 3-field format
        "relative/path.ckpt",
        "latest_or_best",  # superstring of a keyword
    ],
)
def test_unrecognised_raises(rf: str) -> None:
    with pytest.raises(InvalidResumeFromError):
        _classify_resume_from(rf)
