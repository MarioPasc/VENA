"""Guards for the paired_fidelity sweep's SLURM launcher and merge worker.

Both defects pinned here fired together on 2026-07-17 (jobs 1604488/1604489)
and produced a complete-LOOKING sweep artifact built from 54 of 405 prediction
files, published as LATEST.  Neither is reachable from Python, so these are
source-level guards -- crude, but they make a regression a CI failure instead
of a plausible number in a paper table.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.validation

_SLURM = Path(__file__).resolve().parents[2] / "routines/validation/paired_fidelity/slurm"


def test_merge_worker_does_not_auto_enable_allow_partial() -> None:
    """The worker must not downgrade "shards missing" to a warning.

    cli_merge deliberately refuses to merge a partial grid ("Resubmit failed
    tasks before merging, or pass --allow-partial").  The worker used to set
    --allow-partial itself the instant that guard tripped, which is a safety
    net wired to its own off switch: the merge then emits full tables, figures
    and a plausible n_patients from a fraction of the methods.
    """
    text = (_SLURM / "worker_paired_fidelity_merge.sh").read_text()

    assert 'ALLOW_PARTIAL="--allow-partial"' not in text, (
        "merge worker auto-enables --allow-partial when shards are missing; "
        "missing shards must be a stop-the-line exit 1, and --allow-partial "
        "must stay a deliberate human decision (ALLOW_PARTIAL=1)."
    )
    # It must actually fail rather than proceed.
    assert "exit 1" in text, "merge worker must exit non-zero when shards are missing."


def test_sweep_launcher_sanitises_sbatch_job_ids() -> None:
    """Job IDs from Picasso's sbatch wrapper carry ANSI colour codes.

    Interpolated raw, "--dependency=afterok:<ESC>[31m<ESC>[0m1604488" is
    ACCEPTED by sbatch and recorded as Dependency=(null) -- so the merge runs
    immediately, against whatever shards happen to exist.  The launcher must
    strip the escape codes and verify the ID is bare digits.
    """
    text = (_SLURM / "launcher_paired_fidelity_sweep.sh").read_text()

    assert "_clean_job_id" in text, "launcher must sanitise sbatch --parsable output."
    assert "=~ ^[0-9]+$" in text, "launcher must assert the parsed array job ID is bare digits."


def test_sweep_launcher_verifies_the_dependency_was_recorded() -> None:
    """A dependency sbatch silently dropped is worse than no merge job at all.

    sbatch accepting the flag proves nothing -- it accepted the ANSI-corrupted
    form too.  The launcher must read the dependency back from scontrol and
    refuse to leave an unguarded merge job queued.
    """
    text = (_SLURM / "launcher_paired_fidelity_sweep.sh").read_text()

    assert "scontrol show job" in text, "launcher must read the dependency back from SLURM."
    assert "Dependency=(null)" in text, "launcher must detect a dropped dependency."
    assert "scancel" in text, "launcher must cancel a merge job left without a dependency."
