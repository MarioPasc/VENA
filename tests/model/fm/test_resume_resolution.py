"""Unit tests for ``FMTrainRoutineEngine._resolve_resume_ckpt`` (no GPU).

Covers the 2026-06 three-mode semantics:

* BASELINE      — ``None`` / ``"baseline"`` → no checkpoint, mode BASELINE.
* CONTINUE      — ``latest`` / ``best`` → newest sibling matching the
                  same recipe (``*_{stage}_{tag}_*``); a sibling of a
                  *different* recipe is invisible (regression test for
                  the Picasso s1→s2 cross-contamination bug).
* WARM_START    — explicit ``<run_id>`` (resolved under ``experiments_root``)
                  or absolute ``.ckpt`` path.
* Unrecognised  — raises :class:`InvalidResumeFromError`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from routines.fm.train.engine import (
    FMTrainRoutineConfig,
    FMTrainRoutineEngine,
    ResumeMode,
)
from routines.fm.train.exceptions import InvalidResumeFromError

pytestmark = pytest.mark.unit


def _cfg(
    experiments_root: Path,
    resume_from: str | None,
    *,
    stage: str = "s1",
    tag: str = "fft_cfm",
) -> FMTrainRoutineConfig:
    # The single-cohort ``data.latents_h5`` key was retired in the pre-long-run
    # hardening pass. The resume-resolver does not touch the registry, so any
    # placeholder path passes config validation.
    return FMTrainRoutineConfig.model_validate(
        {
            "run": {"stage": stage, "tag": tag, "resume_from": resume_from},
            "data": {"corpus_registry": "/nonexistent/registry.json"},
            "model": {
                "trunk": {"checkpoint": "/nonexistent/trunk.pt"},
                "controlnet": {"conditioning_inputs": ["latent:t1pre"]},
            },
            "output": {"experiments_root": str(experiments_root)},
        }
    )


def _make_run(root: Path, name: str, ckpts: list[str]) -> Path:
    d = root / name / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    for c in ckpts:
        (d / c).write_text("x")
    return root / name


# ---------------------------------------------------------------------------
# BASELINE
# ---------------------------------------------------------------------------


def test_baseline_when_none(tmp_path: Path) -> None:
    eng = FMTrainRoutineEngine(_cfg(tmp_path / "experiments", None))
    path, mode = eng._resolve_resume_ckpt()
    assert path is None
    assert mode is ResumeMode.BASELINE


def test_baseline_when_literal_baseline(tmp_path: Path) -> None:
    eng = FMTrainRoutineEngine(_cfg(tmp_path / "experiments", "baseline"))
    path, mode = eng._resolve_resume_ckpt()
    assert path is None
    assert mode is ResumeMode.BASELINE


def test_baseline_when_empty_string(tmp_path: Path) -> None:
    eng = FMTrainRoutineEngine(_cfg(tmp_path / "experiments", ""))
    path, mode = eng._resolve_resume_ckpt()
    assert path is None
    assert mode is ResumeMode.BASELINE


# ---------------------------------------------------------------------------
# CONTINUE — scoped to {stage, tag}
# ---------------------------------------------------------------------------


def test_continue_picks_same_recipe(tmp_path: Path) -> None:
    root = tmp_path / "experiments"
    prior = _make_run(root, "2026-01-01_00-00-00_s1_fft_cfm_aaaaaaaa", ["last.ckpt"])
    eng = FMTrainRoutineEngine(_cfg(root, "latest", stage="s1", tag="fft_cfm"))
    path, mode = eng._resolve_resume_ckpt()
    assert path == str(prior / "checkpoints" / "last.ckpt")
    assert mode is ResumeMode.CONTINUE


def test_continue_ignores_different_tag(tmp_path: Path) -> None:
    """Regression test for the Picasso bug.

    An s1 job with tag ``lora_r16_cfm`` must NOT inherit the ``last.ckpt`` of
    an s1 ``fft_cfm`` sibling — they are different recipes despite sharing the
    stage prefix and the same ``experiments_root``.
    """
    root = tmp_path / "experiments"
    # Sibling of a different recipe exists with a last.ckpt.
    _make_run(root, "2026-01-01_00-00-00_s1_fft_cfm_aaaaaaaa", ["last.ckpt"])
    eng = FMTrainRoutineEngine(_cfg(root, "latest", stage="s1", tag="lora_r16_cfm"))
    path, mode = eng._resolve_resume_ckpt()
    assert path is None
    # Falls back to BASELINE so the engine mints a fresh run dir.
    assert mode is ResumeMode.BASELINE


def test_continue_ignores_different_stage(tmp_path: Path) -> None:
    """An s2 job with the same tag must NOT inherit an s1 ``last.ckpt``."""
    root = tmp_path / "experiments"
    _make_run(root, "2026-01-01_00-00-00_s1_fft_cfm_aaaaaaaa", ["last.ckpt"])
    eng = FMTrainRoutineEngine(_cfg(root, "latest", stage="s2", tag="fft_cfm"))
    path, mode = eng._resolve_resume_ckpt()
    assert path is None
    assert mode is ResumeMode.BASELINE


def test_continue_skips_empty_current_dir(tmp_path: Path) -> None:
    root = tmp_path / "experiments"
    prior = _make_run(root, "2026-01-01_00-00-00_s1_fft_cfm_aaaaaaaa", ["last.ckpt"])
    current = _make_run(root, "2026-01-02_00-00-00_s1_fft_cfm_bbbbbbbb", [])
    eng = FMTrainRoutineEngine(_cfg(root, "latest", stage="s1", tag="fft_cfm"))
    path, mode = eng._resolve_resume_ckpt(exclude_dir=current)
    assert path == str(prior / "checkpoints" / "last.ckpt")
    assert mode is ResumeMode.CONTINUE


def test_continue_falls_back_to_baseline_when_no_ckpt(tmp_path: Path) -> None:
    root = tmp_path / "experiments"
    _make_run(root, "2026-01-02_00-00-00_s1_fft_cfm_bbbbbbbb", [])
    eng = FMTrainRoutineEngine(_cfg(root, "latest", stage="s1", tag="fft_cfm"))
    path, mode = eng._resolve_resume_ckpt()
    assert path is None
    assert mode is ResumeMode.BASELINE


def test_continue_best_picks_ema_best(tmp_path: Path) -> None:
    root = tmp_path / "experiments"
    # Newest sibling has no ema_best — older one does. ``best`` must pick the
    # older one (it's the only one with the requested checkpoint).
    _make_run(root, "2026-01-02_00-00-00_s1_fft_cfm_bbbbbbbb", ["last.ckpt"])
    older = _make_run(
        root,
        "2026-01-01_00-00-00_s1_fft_cfm_aaaaaaaa",
        ["ema_best.ckpt", "last.ckpt"],
    )
    eng = FMTrainRoutineEngine(_cfg(root, "best", stage="s1", tag="fft_cfm"))
    path, mode = eng._resolve_resume_ckpt()
    assert path == str(older / "checkpoints" / "ema_best.ckpt")
    assert mode is ResumeMode.CONTINUE


def test_continue_ignores_legacy_format_dirs(tmp_path: Path) -> None:
    """Pre-v0.8 run dirs (no tag in the name) must be invisible to the glob."""
    root = tmp_path / "experiments"
    _make_run(root, "2026-01-01_00-00-00_s1_aaa", ["last.ckpt"])  # legacy 3-field
    eng = FMTrainRoutineEngine(_cfg(root, "latest", stage="s1", tag="fft_cfm"))
    path, mode = eng._resolve_resume_ckpt()
    assert path is None
    assert mode is ResumeMode.BASELINE


# ---------------------------------------------------------------------------
# WARM_START
# ---------------------------------------------------------------------------


def test_warm_start_from_run_id(tmp_path: Path) -> None:
    """Pass a literal run_id — the resolver maps it under experiments_root."""
    root = tmp_path / "experiments"
    src_run_id = "2026-01-01_00-00-00_s1_fft_cfm_aaaaaaaa"
    src = _make_run(root, src_run_id, ["last.ckpt"])
    # Destination recipe is different (s2 + lora_r16_contrastive) — that's the
    # whole point of WARM_START (s1→s2 experiment).
    eng = FMTrainRoutineEngine(_cfg(root, src_run_id, stage="s2", tag="lora_r16_contrastive"))
    path, mode = eng._resolve_resume_ckpt()
    assert path == str(src / "checkpoints" / "last.ckpt")
    assert mode is ResumeMode.WARM_START


def test_warm_start_from_absolute_path(tmp_path: Path) -> None:
    ckpt = tmp_path / "external.ckpt"
    ckpt.write_text("x")
    eng = FMTrainRoutineEngine(_cfg(tmp_path / "experiments", str(ckpt)))
    path, mode = eng._resolve_resume_ckpt()
    assert path == str(ckpt)
    assert mode is ResumeMode.WARM_START


def test_warm_start_missing_run_id_raises(tmp_path: Path) -> None:
    """A run_id that doesn't exist under experiments_root must FAIL LOUD."""
    root = tmp_path / "experiments"
    root.mkdir(parents=True, exist_ok=True)
    eng = FMTrainRoutineEngine(_cfg(root, "2026-01-01_00-00-00_s1_fft_cfm_aaaaaaaa"))
    with pytest.raises(InvalidResumeFromError):
        eng._resolve_resume_ckpt()


def test_warm_start_missing_explicit_path_raises(tmp_path: Path) -> None:
    eng = FMTrainRoutineEngine(_cfg(tmp_path / "experiments", "/nope/x.ckpt"))
    with pytest.raises(InvalidResumeFromError):
        eng._resolve_resume_ckpt()


# ---------------------------------------------------------------------------
# Invalid resume_from values
# ---------------------------------------------------------------------------


def test_unrecognised_string_raises(tmp_path: Path) -> None:
    eng = FMTrainRoutineEngine(_cfg(tmp_path / "experiments", "garbage"))
    with pytest.raises(InvalidResumeFromError):
        eng._resolve_resume_ckpt()


# ---------------------------------------------------------------------------
# WARM_START → CONTINUE auto-promotion
#
# Picasso walltime resubmit case: the first launch of a WARM_START YAML
# creates a new dir + loads the external source; subsequent launches of the
# same YAML find the just-created recipe-matching sibling and continue from
# its ``last.ckpt`` rather than re-warm-starting from the source.
# ---------------------------------------------------------------------------


def test_warm_start_promoted_to_continue_when_sibling_has_last(tmp_path: Path) -> None:
    root = tmp_path / "experiments"
    ckpt = tmp_path / "external.ckpt"
    ckpt.write_text("x")
    # The first launch already minted this dir + wrote last.ckpt.
    sibling = _make_run(
        root,
        "2026-02-01_00-00-00_s2_fft_contrastive_s1warm_cccccccc",
        ["last.ckpt"],
    )
    eng = FMTrainRoutineEngine(_cfg(root, str(ckpt), stage="s2", tag="fft_contrastive_s1warm"))
    path, mode = eng._resolve_resume_ckpt()
    assert path == str(sibling / "checkpoints" / "last.ckpt")
    assert mode is ResumeMode.CONTINUE


def test_warm_start_not_promoted_when_sibling_missing_last(tmp_path: Path) -> None:
    """Sibling dir exists but has no last.ckpt → genuine first-time WARM_START."""
    root = tmp_path / "experiments"
    ckpt = tmp_path / "external.ckpt"
    ckpt.write_text("x")
    _make_run(
        root,
        "2026-02-01_00-00-00_s2_fft_contrastive_s1warm_cccccccc",
        [],  # no last.ckpt → empty dir does not satisfy promotion
    )
    eng = FMTrainRoutineEngine(_cfg(root, str(ckpt), stage="s2", tag="fft_contrastive_s1warm"))
    path, mode = eng._resolve_resume_ckpt()
    assert path == str(ckpt)
    assert mode is ResumeMode.WARM_START


def test_warm_start_not_promoted_by_different_recipe_sibling(tmp_path: Path) -> None:
    """A sibling of a DIFFERENT recipe must not trigger promotion.

    Regression for the tag-isolation contract: the ``*_s1warm`` tag used by
    Picasso warm-start YAMLs is intentionally distinct from the BASELINE tag
    (e.g. ``s2_fft_contrastive``); the promotion glob must respect that.
    """
    root = tmp_path / "experiments"
    ckpt = tmp_path / "external.ckpt"
    ckpt.write_text("x")
    _make_run(
        root,
        "2026-01-01_00-00-00_s2_fft_contrastive_aaaaaaaa",
        ["last.ckpt"],
    )
    eng = FMTrainRoutineEngine(_cfg(root, str(ckpt), stage="s2", tag="fft_contrastive_s1warm"))
    path, mode = eng._resolve_resume_ckpt()
    assert path == str(ckpt)
    assert mode is ResumeMode.WARM_START


def test_warm_start_promotion_picks_newest_sibling(tmp_path: Path) -> None:
    import os

    root = tmp_path / "experiments"
    ckpt = tmp_path / "external.ckpt"
    ckpt.write_text("x")
    older = _make_run(
        root,
        "2026-01-01_00-00-00_s2_fft_contrastive_s1warm_aaaaaaaa",
        ["last.ckpt"],
    )
    newer = _make_run(
        root,
        "2026-02-01_00-00-00_s2_fft_contrastive_s1warm_bbbbbbbb",
        ["last.ckpt"],
    )
    # Pin mtimes so the test is independent of FS resolution.
    os.utime(older, (1000.0, 1000.0))
    os.utime(newer, (2000.0, 2000.0))
    eng = FMTrainRoutineEngine(_cfg(root, str(ckpt), stage="s2", tag="fft_contrastive_s1warm"))
    path, mode = eng._resolve_resume_ckpt()
    assert path == str(newer / "checkpoints" / "last.ckpt")
    assert mode is ResumeMode.CONTINUE


def test_warm_start_promotion_skips_excluded_dir(tmp_path: Path) -> None:
    """The just-minted (still-empty) run dir is excluded; promotion still works
    against the older sibling that has last.ckpt."""
    root = tmp_path / "experiments"
    ckpt = tmp_path / "external.ckpt"
    ckpt.write_text("x")
    sibling = _make_run(
        root,
        "2026-01-01_00-00-00_s2_fft_contrastive_s1warm_aaaaaaaa",
        ["last.ckpt"],
    )
    current = _make_run(
        root,
        "2026-02-01_00-00-00_s2_fft_contrastive_s1warm_bbbbbbbb",
        [],  # freshly-minted, no checkpoint yet
    )
    eng = FMTrainRoutineEngine(_cfg(root, str(ckpt), stage="s2", tag="fft_contrastive_s1warm"))
    path, mode = eng._resolve_resume_ckpt(exclude_dir=current)
    assert path == str(sibling / "checkpoints" / "last.ckpt")
    assert mode is ResumeMode.CONTINUE


def test_warm_start_promotion_applies_to_run_id_source_too(tmp_path: Path) -> None:
    """Auto-promotion fires regardless of the WARM_START source form
    (abs path vs. literal run_id)."""
    root = tmp_path / "experiments"
    src_run_id = "2026-01-01_00-00-00_s1_fft_cfm_aaaaaaaa"
    _make_run(root, src_run_id, ["last.ckpt"])
    # A recipe-matching sibling already exists in the destination recipe.
    sibling = _make_run(
        root,
        "2026-02-01_00-00-00_s2_fft_contrastive_s1warm_cccccccc",
        ["last.ckpt"],
    )
    eng = FMTrainRoutineEngine(_cfg(root, src_run_id, stage="s2", tag="fft_contrastive_s1warm"))
    path, mode = eng._resolve_resume_ckpt()
    assert path == str(sibling / "checkpoints" / "last.ckpt")
    assert mode is ResumeMode.CONTINUE
