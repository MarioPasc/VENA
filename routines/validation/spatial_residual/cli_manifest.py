"""Generate the sweep manifest — one row per (method, cohort) at each method's selection_nfe.

The manifest is the authoritative task list for the SLURM array sweep.
``SLURM_ARRAY_TASK_ID`` indexes into it; the shard step reads exactly one row
per task; the merge step reads it for provenance and completeness checks.

Unlike the paired_fidelity manifest (which includes every NFE), this manifest
filters each method to its pre-registered ``selection_nfe`` from
``vena.validation.registry.SELECTION_NFE``.  Methods absent from the registry
are skipped with a WARNING; they may be submitted separately if needed.

Usage::

    python -m routines.validation.spatial_residual.cli_manifest \\
        --data-root /mnt/.../inference \\
        --output    /mnt/.../spatial_residual_sweep/manifest.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from vena.validation import registry
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
        description=(
            "Generate spatial_residual sweep manifest from the inference tree. "
            "Each method contributes exactly one prediction file: the one at its "
            "selection_nfe (from vena.validation.registry.SELECTION_NFE)."
        ),
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

    # build_index already filters smoke shards internally via discover_shards.
    index = build_index(data_root)
    if index.empty:
        logger.error("No prediction H5 files found under %s — aborting.", data_root)
        sys.exit(1)

    logger.info("Raw index: %d rows before selection_nfe filter.", len(index))

    # Filter: keep each method only at its pre-registered selection_nfe.
    # Reference registry.SELECTION_NFE as a module attribute — never via a
    # from-import, because load_partitions() rebinds the module-level name and
    # a from-import would capture the stale reference.
    sel_nfe = registry.SELECTION_NFE  # dict[str, int]

    keep_rows = []
    skipped_unregistered: list[str] = []
    for _, row in index.iterrows():
        method = str(row["method"])
        nfe = int(row["nfe"])
        if method not in sel_nfe:
            if method not in skipped_unregistered:
                logger.warning(
                    "Method %r not in registry.SELECTION_NFE — skipping all its rows.",
                    method,
                )
                skipped_unregistered.append(method)
            continue
        if nfe == sel_nfe[method]:
            keep_rows.append(row)

    if not keep_rows:
        logger.error(
            "No rows remain after selection_nfe filter. "
            "Check that the inference tree contains files at the registered NFEs: %s",
            dict(sel_nfe),
        )
        sys.exit(1)

    import pandas as pd

    filtered = pd.DataFrame(keep_rows).reset_index(drop=True)

    # Insert a stable task_id column (0-based, matches SLURM_ARRAY_TASK_ID).
    filtered.insert(0, "task_id", range(len(filtered)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(args.output, index=False)

    if skipped_unregistered:
        logger.warning(
            "%d method(s) had no selection_nfe and were excluded: %s",
            len(skipped_unregistered),
            skipped_unregistered,
        )

    logger.info(
        "Manifest written: %d tasks (from %d raw rows) → %s",
        len(filtered),
        len(index),
        args.output,
    )
    # Structured output lines for the launcher to parse.
    print(f"manifest_tasks={len(filtered)}")
    print(f"manifest_path={args.output}")


if __name__ == "__main__":
    main()
