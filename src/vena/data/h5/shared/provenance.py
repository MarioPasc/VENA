"""Provenance helpers: git SHA, file SHA-256, timestamps."""

from __future__ import annotations

import datetime as _dt
import hashlib
import subprocess
from pathlib import Path


def resolve_git_sha(start: Path | None = None) -> str | None:
    """Return the current ``HEAD`` SHA of the repo containing ``start``.

    Walks upward from ``start`` (defaulting to this file) looking for a git
    repository. Returns ``None`` outside any repository or when ``git`` is
    unavailable; callers should write a placeholder rather than failing.
    """
    anchor = (start or Path(__file__)).resolve()
    if anchor.is_file():
        anchor = anchor.parent
    try:
        out = subprocess.check_output(
            ["git", "-C", str(anchor), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def now_iso_utc() -> str:
    """Return the current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return _dt.datetime.now(tz=_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()
