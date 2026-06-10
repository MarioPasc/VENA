"""Unit tests for run_id generation."""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone

import pytest

from routines.fm.train.runner.run_id import generate_run_id, normalise_tag


# 4-field format: <UTC>_<stage>_<tag>_<8-hex>.
RUN_ID_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_[a-z0-9_]+_[a-z0-9_]+_[0-9a-f]{8}$"
)


@pytest.mark.unit
def test_run_id_format() -> None:
    rid = generate_run_id("S1", "fft_cfm")
    assert RUN_ID_RE.match(rid), rid


@pytest.mark.unit
def test_run_id_stage_normalised() -> None:
    rid = generate_run_id("Skip-S1", "fft_cfm")
    assert "_skip_s1_" in rid


@pytest.mark.unit
def test_run_id_tag_embedded() -> None:
    rid = generate_run_id("s2", "lora_r16_contrastive")
    assert "_s2_lora_r16_contrastive_" in rid


@pytest.mark.unit
def test_run_id_tag_hyphen_normalised() -> None:
    # YAML authors may write hyphens; we rewrite to underscores so glob
    # patterns stay simple.
    rid = generate_run_id("s2", "lora-r16-cfg")
    assert "_s2_lora_r16_cfg_" in rid


@pytest.mark.unit
def test_run_id_unique_across_short_interval() -> None:
    a = generate_run_id("s1", "fft_cfm")
    time.sleep(1.001)
    b = generate_run_id("s1", "fft_cfm")
    assert a != b


@pytest.mark.unit
def test_run_id_deterministic_for_fixed_now() -> None:
    now = datetime(2026, 5, 27, 12, 0, 0, tzinfo=timezone.utc)
    # The hash includes pid+hostname so it's not byte-stable across processes,
    # but the timestamp portion must be fixed.
    rid = generate_run_id("s2", "fft_contrastive", now=now)
    assert rid.startswith("2026-05-27_12-00-00_s2_fft_contrastive_")


@pytest.mark.unit
def test_normalise_tag_accepts_safe_chars() -> None:
    assert normalise_tag("FFT_CFM") == "fft_cfm"
    assert normalise_tag("lora-r16-cfg") == "lora_r16_cfg"
    assert normalise_tag("v0_3") == "v0_3"


@pytest.mark.unit
def test_normalise_tag_rejects_unsafe_chars() -> None:
    for bad in ("", "  ", "tag with space", "tag.dot", "tag/slash", "tag:colon"):
        with pytest.raises(ValueError):
            normalise_tag(bad)
