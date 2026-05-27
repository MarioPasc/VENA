"""Write env / git / hostname provenance files for a run directory."""

from __future__ import annotations

import logging
import platform
import subprocess
from pathlib import Path

from vena.data.h5.shared import resolve_git_sha

logger = logging.getLogger(__name__)


def _pip_freeze() -> str:
    try:
        out = subprocess.run(
            ["pip", "freeze"], capture_output=True, text=True, check=True, timeout=30,
        )
        return out.stdout
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        logger.warning("pip freeze failed: %s", exc)
        return ""


def _git_diff_stat(repo: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "diff", "--stat"],
            capture_output=True, text=True, check=False, timeout=10,
        )
        return out.stdout
    except (subprocess.SubprocessError, FileNotFoundError):
        return ""


def write_provenance(run_dir: Path, repo: Path | None = None) -> None:
    """Write ``env.txt``, ``git_commit.txt``, ``hostname.txt`` into ``run_dir``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "env.txt").write_text(_pip_freeze())
    sha = resolve_git_sha(repo) or "unknown"
    diff_stat = _git_diff_stat(repo) if repo else ""
    (run_dir / "git_commit.txt").write_text(f"sha: {sha}\n\n{diff_stat}")
    (run_dir / "hostname.txt").write_text(f"{platform.node()}\n{platform.platform()}\n")
