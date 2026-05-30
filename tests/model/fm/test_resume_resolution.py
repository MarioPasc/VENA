"""Unit tests for ``FMTrainRoutineEngine._resolve_resume_ckpt`` (no GPU).

The resolver must, for ``resume_from: latest|best``, skip the just-created
current run directory (empty ``checkpoints/``) and pick the most recent prior
run that actually holds the target checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from routines.fm.train.engine import FMTrainRoutineConfig, FMTrainRoutineEngine

pytestmark = pytest.mark.unit


def _cfg(experiments_root: Path, resume_from: str) -> FMTrainRoutineConfig:
    # The single-cohort ``data.latents_h5`` key was retired in the pre-long-run
    # hardening pass. The resume-resolver does not touch the registry, so any
    # placeholder path passes config validation.
    return FMTrainRoutineConfig.model_validate(
        {
            "run": {"resume_from": resume_from},
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


def test_latest_skips_empty_current_dir(tmp_path: Path) -> None:
    root = tmp_path / "experiments"
    prior = _make_run(root, "2026-01-01_00-00-00_s1_aaa", ["last.ckpt"])
    current = _make_run(root, "2026-01-02_00-00-00_s1_bbb", [])  # newest, empty
    eng = FMTrainRoutineEngine(_cfg(root, "latest"))
    resolved = eng._resolve_resume_ckpt(exclude_dir=current)
    assert resolved == str(prior / "checkpoints" / "last.ckpt")


def test_latest_returns_none_when_no_ckpt_anywhere(tmp_path: Path) -> None:
    root = tmp_path / "experiments"
    current = _make_run(root, "2026-01-02_00-00-00_s1_bbb", [])
    eng = FMTrainRoutineEngine(_cfg(root, "latest"))
    assert eng._resolve_resume_ckpt(exclude_dir=current) is None


def test_latest_falls_back_to_ema_epoch(tmp_path: Path) -> None:
    root = tmp_path / "experiments"
    prior = _make_run(
        root, "2026-01-01_00-00-00_s1_aaa", ["ema_epoch_003.ckpt", "ema_epoch_007.ckpt"]
    )
    eng = FMTrainRoutineEngine(_cfg(root, "latest"))
    resolved = eng._resolve_resume_ckpt(exclude_dir=None)
    assert resolved == str(prior / "checkpoints" / "ema_epoch_007.ckpt")


def test_best_picks_ema_best(tmp_path: Path) -> None:
    root = tmp_path / "experiments"
    _make_run(root, "2026-01-01_00-00-00_s1_aaa", ["last.ckpt"])  # no ema_best
    prior_best = _make_run(root, "2026-01-02_00-00-00_s1_bbb", ["ema_best.ckpt", "last.ckpt"])
    eng = FMTrainRoutineEngine(_cfg(root, "best"))
    resolved = eng._resolve_resume_ckpt(exclude_dir=None)
    assert resolved == str(prior_best / "checkpoints" / "ema_best.ckpt")


def test_explicit_path_passthrough(tmp_path: Path) -> None:
    ckpt = tmp_path / "some.ckpt"
    ckpt.write_text("x")
    eng = FMTrainRoutineEngine(_cfg(tmp_path / "experiments", str(ckpt)))
    assert eng._resolve_resume_ckpt() == str(ckpt)


def test_none_when_resume_from_empty(tmp_path: Path) -> None:
    eng = FMTrainRoutineEngine(_cfg(tmp_path / "experiments", ""))
    assert eng._resolve_resume_ckpt() is None
