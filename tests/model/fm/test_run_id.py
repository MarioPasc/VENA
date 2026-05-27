"""Unit tests for run_id generation."""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone

import pytest

from routines.fm.train.runner.run_id import generate_run_id


RUN_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_[a-z0-9_]+_[0-9a-f]{8}$")


@pytest.mark.unit
def test_run_id_format() -> None:
    rid = generate_run_id("S1")
    assert RUN_ID_RE.match(rid), rid


@pytest.mark.unit
def test_run_id_stage_normalised() -> None:
    rid = generate_run_id("Skip-S1")
    assert "_skip_s1_" in rid


@pytest.mark.unit
def test_run_id_unique_across_short_interval() -> None:
    a = generate_run_id("s1")
    time.sleep(1.001)
    b = generate_run_id("s1")
    assert a != b


@pytest.mark.unit
def test_run_id_deterministic_for_fixed_now() -> None:
    now = datetime(2026, 5, 27, 12, 0, 0, tzinfo=timezone.utc)
    # The hash includes pid+hostname so it's not byte-stable across processes,
    # but the timestamp portion must be fixed.
    rid = generate_run_id("s2", now=now)
    assert rid.startswith("2026-05-27_12-00-00_s2_")
