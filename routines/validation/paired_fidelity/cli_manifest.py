"""Generate the sweep manifest — one row per (method, cohort, nfe) prediction file.

The manifest is the authoritative task list for the SLURM array sweep.
``SLURM_ARRAY_TASK_ID`` indexes into it; the merge step reads it for
provenance and completeness checks.

Usage::

    python -m routines.validation.paired_fidelity.cli_manifest \\
        --data-root /mnt/.../execs/vena/inference \\
        --output /mnt/.../paired_fidelity_sweep/manifest.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from vena.validation.io import build_index, discover_shards

logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entry point — writes manifest.csv to --output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(
        description="Generate paired_fidelity sweep manifest from the inference tree.",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        type=Path,
        help="Root of the inference tree (contains shard sub-directories).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output path for manifest.csv.",
    )
    args = parser.parse_args()

    data_root: Path = args.data_root
    if not data_root.is_dir():
        logger.error("data-root does not exist or is not a directory: %s", data_root)
        sys.exit(1)

    # Smoke shards are excluded at discovery time (content-based, not name-based).
    discovery = discover_shards(data_root)
    if discovery.skipped_smoke:
        logger.info(
            "Skipping %d smoke shard(s): %s",
            len(discovery.skipped_smoke),
            discovery.skipped_smoke,
        )

    index = build_index(data_root)
    if index.empty:
        logger.error("No prediction H5 files found under %s — aborting.", data_root)
        sys.exit(1)

    # Insert a stable task_id column (0-based, matches SLURM_ARRAY_TASK_ID).
    index = index.reset_index(drop=True)
    index.insert(0, "task_id", range(len(index)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(args.output, index=False)

    logger.info(
        "Manifest written: %d tasks → %s",
        len(index),
        args.output,
    )
    # Print to stdout for the launcher to capture.
    print(f"manifest_tasks={len(index)}")
    print(f"manifest_path={args.output}")


if __name__ == "__main__":
    main()
