"""Train engine post-training hook — failure isolation."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from routines.fm.train.engine import _run_post_train

pytestmark = pytest.mark.unit


def test_hook_swallows_runtime_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failing post-train must NOT propagate — the training run wins."""

    def _boom(run_dir: Path, *, formats: tuple[str, ...]) -> Path:
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(
        "routines.fm.post_train.engine.render_for_run_dir",
        _boom,
    )

    with caplog.at_level(logging.WARNING, logger="routines.fm.train.engine"):
        _run_post_train(tmp_path, formats=("png",))

    messages = [r.getMessage() for r in caplog.records]
    assert any("post-train plotting failed" in m for m in messages), messages


def test_hook_succeeds_on_no_op(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Path] = []

    def _noop(run_dir: Path, *, formats: tuple[str, ...]) -> Path:
        calls.append(run_dir)
        return run_dir / "plots"

    monkeypatch.setattr(
        "routines.fm.post_train.engine.render_for_run_dir",
        _noop,
    )

    _run_post_train(tmp_path, formats=("png",))
    assert calls == [tmp_path]
