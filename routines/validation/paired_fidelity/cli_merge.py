"""Merge CLI — concatenate sweep shards and run the full analysis pass.

Runs after the SLURM array completes.  Reads all ``shard_NNNN.csv`` files
from ``--shard-dir``, concatenates them into a single per-scan DataFrame, and
calls :meth:`PairedFidelityEngine.run_postprocess` once — the only place
where patient collapse, Holm-Bonferroni correction, figures, and
``decision.json`` are produced.

**Never run per shard** — the Holm correction is family-wide and must see all
methods simultaneously, otherwise adjusted p-values are wrong.

Usage::

    python -m routines.validation.paired_fidelity.cli_merge \\
        --manifest /path/to/manifest.csv \\
        --shard-dir /path/to/shards \\
        --config /path/to/smoke_picasso.yaml \\
        --output-root /path/to/sweep_output \\
        [--allow-partial]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from routines.validation.paired_fidelity.engine import PairedFidelityConfig, PairedFidelityEngine
from vena.validation.artifacts import make_run_dir
from vena.validation.io import discover_shards

logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(
        description="Merge sweep shards and run the full paired_fidelity analysis.",
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--shard-dir", required=True, type=Path, dest="shard_dir")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        dest="output_root",
        default=None,
        help="Override cfg.output_root for the merged artifact directory.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        dest="allow_partial",
        help=(
            "Proceed even when shard files are missing (logs WARNING per missing task). "
            "By default the merge aborts when any shard is absent."
        ),
    )
    args = parser.parse_args()

    t_start = time.perf_counter()

    # ---- Load config ----
    cfg = PairedFidelityConfig.from_yaml(args.config)
    if args.output_root is not None:
        # Build an updated config with the overridden output_root.
        # PairedFidelityConfig is frozen; use dataclasses.replace.
        import dataclasses

        cfg = dataclasses.replace(cfg, output_root=args.output_root)

    # ---- Discover shards ----
    manifest = pd.read_csv(args.manifest)
    n_tasks = len(manifest)

    shard_paths = sorted(args.shard_dir.glob("shard_*.csv"))
    n_found = len(shard_paths)

    if n_found < n_tasks:
        missing = n_tasks - n_found
        # Find which task IDs are missing by comparing shard_NNNN.csv filenames.
        found_ids = {int(p.stem.split("_")[1]) for p in shard_paths}
        missing_ids = sorted(set(range(n_tasks)) - found_ids)
        if args.allow_partial:
            logger.warning(
                "%d/%d shards missing (task IDs: %s) — proceeding with %d shards.",
                missing,
                n_tasks,
                missing_ids[:20],
                n_found,
            )
        else:
            logger.error(
                "%d/%d shards missing (task IDs: %s). "
                "Resubmit failed tasks before merging, or pass --allow-partial.",
                missing,
                n_tasks,
                missing_ids[:20],
            )
            sys.exit(1)

    if n_found == 0:
        logger.error("No shard files found in %s — aborting.", args.shard_dir)
        sys.exit(1)

    logger.info("Concatenating %d shard files …", n_found)
    dfs: list[pd.DataFrame] = []
    for p in shard_paths:
        try:
            df = pd.read_csv(p)
            if df.empty:
                logger.warning("Empty shard (zero rows): %s — skipping.", p.name)
            else:
                dfs.append(df)
        except Exception as exc:
            logger.warning("Failed to read shard %s: %s — skipping.", p.name, exc)

    if not dfs:
        logger.error("All shards empty or unreadable — aborting.")
        sys.exit(1)

    per_scan_df = pd.concat(dfs, ignore_index=True)
    n_scans = len(per_scan_df)
    elapsed_concat = time.perf_counter() - t_start
    logger.info(
        "Concatenated %d scan rows from %d shards in %.1f s.", n_scans, n_found, elapsed_concat
    )

    # ---- Create run_dir and run the analysis pass ----
    run_dir = make_run_dir(cfg.output_root, "paired_fidelity")
    logger.info("Run directory: %s", run_dir)

    # Re-derive the skipped smoke shards from the data root.  The exclusion
    # itself happened at manifest-generation time, but the merged decision.json
    # is the artifact an auditor reads, and reporting [] there would assert that
    # nothing was excluded -- i.e. that the stale smoke_loginexa shard was NOT
    # filtered, which is the exact contamination the discovery contract exists
    # to prevent (SHARED_CONTRACTS §3.1).  Recording an unlogged exclusion as
    # "none" reads as "we covered everything".  discover_shards only reads each
    # shard's decision.json, so this is cheap and cannot disagree with the
    # manifest: both call the same function on the same root.
    discovery = discover_shards(cfg.data_root)
    if discovery.skipped_smoke:
        logger.info(
            "Recording %d skipped smoke shard(s) in decision.json: %s",
            len(discovery.skipped_smoke),
            discovery.skipped_smoke,
        )

    engine = PairedFidelityEngine(cfg)
    engine.run_postprocess(
        run_dir,
        per_scan_df=per_scan_df,
        # n_files = number of shard files (each is one prediction file).
        n_files=n_found,
        n_scans=n_scans,
        # elapsed_s covers concat + analysis; shard wall-times are in shard logs.
        elapsed_s=time.perf_counter() - t_start,
        skipped_smoke_shards=discovery.skipped_smoke,
    )

    logger.info(
        "Merge complete — %d shards / %d scans → %s",
        n_found,
        n_scans,
        run_dir,
    )
    print(f"merge_run_dir={run_dir}")
    print(f"merge_n_shards={n_found}")
    print(f"merge_n_scans={n_scans}")


if __name__ == "__main__":
    main()
