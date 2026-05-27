"""Generate the canonical run identifier ``{timestamp}_{stage}_{short_hash}``.

Per training_routine.md §2.1:

* timestamp: ``YYYY-MM-DD_HH-MM-SS`` UTC.
* stage: ``s1``/``s2``/``s3``/``skip_s1``.
* short_hash: first 8 hex characters of
  ``sha256(timestamp + pid + hostname)``.
"""

from __future__ import annotations

import hashlib
import os
import platform
from datetime import datetime, timezone


def generate_run_id(stage: str, now: datetime | None = None) -> str:
    """Build the run identifier string.

    Parameters
    ----------
    stage : str
        Curriculum stage in any case; lower-cased in the output.
    now : datetime | None
        Override clock for tests; default ``datetime.now(timezone.utc)``.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    ts = now.strftime("%Y-%m-%d_%H-%M-%S")
    stage_norm = stage.strip().lower().replace("-", "_")
    pid = os.getpid()
    host = platform.node()
    seed = f"{ts}{pid}{host}".encode()
    short_hash = hashlib.sha256(seed).hexdigest()[:8]
    return f"{ts}_{stage_norm}_{short_hash}"
