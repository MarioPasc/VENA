"""Phase-2 artifact writer utilities.

All Phase-2 routines emit self-contained artifact folders following the
pattern established in ``routines/preflights/latent_aug_equivariance``.
These helpers eliminate boilerplate and guarantee the folder layout is
identical across routines.
"""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Run directory
# ---------------------------------------------------------------------------


def make_run_dir(output_root: Path, routine_name: str) -> Path:
    """Create a UTC-stamped run directory under ``<output_root>/<routine_name>/``.

    Parameters
    ----------
    output_root :
        Root for all Phase-2 analyses (e.g.
        ``inference_root / "analyses"``).
    routine_name :
        Routine identifier, e.g. ``"preregister"`` or ``"paired_fidelity"``.

    Returns
    -------
    Path
        Newly created directory ``<output_root>/<routine_name>/<UTC-stamp>/``.
    """
    stamp = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_dir = Path(output_root) / routine_name / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tables").mkdir(exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    (run_dir / "per_scan").mkdir(exist_ok=True)
    return run_dir


# ---------------------------------------------------------------------------
# decision.json
# ---------------------------------------------------------------------------


def _git_sha() -> str | None:
    """Return the current HEAD SHA, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def write_decision_json(run_dir: Path, payload: dict) -> Path:
    """Write ``decision.json`` into *run_dir*.

    Adds ``produced_at`` (ISO-8601 UTC) and ``git_sha`` to *payload* if not
    already present.

    Parameters
    ----------
    run_dir :
        The run directory returned by :func:`make_run_dir`.
    payload :
        Arbitrary JSON-serialisable dict.  Will be mutated in place with
        ``produced_at`` and ``git_sha``.

    Returns
    -------
    Path
        Path to the written ``decision.json``.
    """
    payload.setdefault("produced_at", datetime.now(tz=UTC).isoformat())
    payload.setdefault("git_sha", _git_sha())

    out = Path(run_dir) / "decision.json"
    out.write_text(json.dumps(payload, indent=2, default=str))
    logger.debug("wrote %s", out)
    return out


# ---------------------------------------------------------------------------
# per_scan CSV
# ---------------------------------------------------------------------------


def write_per_scan_csv(run_dir: Path, df: pd.DataFrame, *, name: str = "per_scan.csv") -> Path:
    """Write the tidy per-scan CSV into ``<run_dir>/per_scan/<name>``.

    This is the primary deliverable consumed by all downstream statistics.
    Long/tidy format: one row per ``(method, cohort, nfe, scan_id)``.

    Parameters
    ----------
    run_dir :
        The run directory returned by :func:`make_run_dir`.
    df :
        Tidy DataFrame.
    name :
        Filename within ``per_scan/``.  Defaults to ``per_scan.csv``.

    Returns
    -------
    Path
        Written file path.
    """
    out = Path(run_dir) / "per_scan" / name
    df.to_csv(out, index=False)
    logger.debug("wrote %s (%d rows)", out, len(df))
    return out


# ---------------------------------------------------------------------------
# LATEST symlink
# ---------------------------------------------------------------------------


def symlink_latest(run_dir: Path) -> None:
    """Update the ``LATEST`` relative symlink to point at *run_dir*.

    The symlink lives in ``run_dir.parent`` (the routine directory).

    Parameters
    ----------
    run_dir :
        The run directory returned by :func:`make_run_dir`.
    """
    routine_dir = Path(run_dir).parent
    latest = routine_dir / "LATEST"
    target = Path(run_dir.name)  # relative
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(target)
    logger.debug("LATEST -> %s", target)
