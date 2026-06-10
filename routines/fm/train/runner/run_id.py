"""Generate the canonical run identifier ``{timestamp}_{stage}_{tag}_{short_hash}``.

Per training_routine.md §2.1, extended in the 2026-06 resume-semantics overhaul:

* timestamp: ``YYYY-MM-DD_HH-MM-SS`` UTC.
* stage: ``s1``/``s2``/``s3``/``skip_s1``.
* tag: recipe identifier within a stage (``fft_cfm``, ``lora_r16_contrastive``,
  ``lora_r16_contrastive_cfg``, …). Embedding the tag in the run_id is what
  lets ``resume_from: latest`` glob ``*_{stage}_{tag}_*/`` and pick up only
  sibling runs of the *same* recipe — sibling jobs of a different recipe in
  the same ``experiments_root`` no longer collide.
* short_hash: first 8 hex characters of
  ``sha256(timestamp + pid + hostname)``.

The legacy 3-field format (``{timestamp}_{stage}_{short_hash}``) is no longer
produced; legacy run directories on disk are not picked up by the new glob
(they don't match ``*_{stage}_{tag}_*/``), which is the intended migration
path — old artefacts stay readable, but a fresh ``resume_from: latest`` will
not silently latch onto them.
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
from datetime import datetime, timezone

_TAG_RE = re.compile(r"^[a-z0-9_]+$")


def normalise_tag(tag: str) -> str:
    """Lower-case + ``-`` → ``_`` normalisation; reject anything else.

    Tags ride inside a glob pattern and a filesystem path, so they must be
    safe for both: ASCII lower-case letters, digits, and underscores only.
    Hyphens are accepted in the input and rewritten to underscores so YAML
    authors can write ``lora-r16-cfg`` if they prefer.
    """
    norm = tag.strip().lower().replace("-", "_")
    if not norm:
        raise ValueError("tag must be a non-empty string")
    if not _TAG_RE.match(norm):
        raise ValueError(
            f"tag={tag!r} → {norm!r} contains characters outside [a-z0-9_]; "
            "use lower-case letters, digits, '_' (or '-' which is auto-converted)."
        )
    return norm


def generate_run_id(stage: str, tag: str, now: datetime | None = None) -> str:
    """Build the run identifier string.

    Parameters
    ----------
    stage : str
        Curriculum stage in any case; lower-cased in the output.
    tag : str
        Recipe identifier within a stage. Lower-cased; ``-`` rewritten to
        ``_``; rejected if it contains anything outside ``[a-z0-9_]``.
    now : datetime | None
        Override clock for tests; default ``datetime.now(timezone.utc)``.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    ts = now.strftime("%Y-%m-%d_%H-%M-%S")
    stage_norm = stage.strip().lower().replace("-", "_")
    tag_norm = normalise_tag(tag)
    pid = os.getpid()
    host = platform.node()
    seed = f"{ts}{pid}{host}".encode()
    short_hash = hashlib.sha256(seed).hexdigest()[:8]
    return f"{ts}_{stage_norm}_{tag_norm}_{short_hash}"
