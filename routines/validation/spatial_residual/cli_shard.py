"""Shard CLI — one SLURM array task per (method, cohort) prediction file.

Reads one row from the manifest by ``SLURM_ARRAY_TASK_ID``, streams all scans
from that prediction H5 file, computes per-scan spatial residual rows, and
writes ``shard_<NNNN>.csv`` to the shard directory.  No patient collapse, no
stats, no Holm correction, no figures — those run exactly once in the merge
step.

Usage::

    python -m routines.validation.spatial_residual.cli_shard \\
        --manifest /path/to/manifest.csv \\
        --task-id  "$SLURM_ARRAY_TASK_ID" \\
        --shard-dir /path/to/shards \\
        --config    /path/to/picasso_sweep.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from routines.validation.spatial_residual.engine import SpatialResidualConfig
from vena.validation.io import ReferenceCache, iter_scans
from vena.validation.spatial_residual import SPATIAL_CSV_COLUMNS, compute_scan_rows

logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Process one (method, cohort, nfe) prediction file → shard CSV. "
            "Called once per SLURM_ARRAY_TASK_ID by the sweep array job."
        ),
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--task-id", required=True, type=int, dest="task_id")
    parser.add_argument("--shard-dir", required=True, type=Path, dest="shard_dir")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    cfg = SpatialResidualConfig.from_yaml(args.config)

    manifest = pd.read_csv(args.manifest)
    if args.task_id >= len(manifest):
        logger.error("task_id %d out of range (manifest has %d rows)", args.task_id, len(manifest))
        sys.exit(1)

    row = manifest.iloc[args.task_id]
    pred_path = Path(row["path"])
    method = str(row["method"])
    cohort = str(row["cohort"])
    nfe = int(row["nfe"])

    logger.info(
        "Task %d: method=%s cohort=%s nfe=%d file=%s",
        args.task_id,
        method,
        cohort,
        nfe,
        pred_path.name,
    )

    if not pred_path.is_file():
        logger.error("Prediction file not found: %s", pred_path)
        sys.exit(1)

    t0 = time.perf_counter()
    ref_cache = ReferenceCache(maxsize=40)
    rows_out: list[dict] = []
    n_scans = 0

    try:
        for sample in iter_scans(pred_path, reference_cache=ref_cache):
            scan_rows = compute_scan_rows(
                sample,
                dilate_k=cfg.dilate_k,
                n_shuffles=cfg.n_shuffles,
                n_boot=cfg.n_boot,
                rng_seed=cfg.rng_seed,
                mi_n_voxels=cfg.mi_n_voxels,
                n_deciles=cfg.n_deciles,
            )
            rows_out.extend(scan_rows)
            n_scans += 1
    except Exception as exc:
        logger.error("Fatal error on task %d (%s): %s", args.task_id, pred_path, exc)
        raise

    elapsed = time.perf_counter() - t0
    logger.info(
        "Task %d: %d scans, %d rows in %.1f s (%.2f s/scan)",
        args.task_id,
        n_scans,
        len(rows_out),
        elapsed,
        elapsed / max(n_scans, 1),
    )

    if not rows_out:
        logger.warning("Task %d produced zero rows — writing empty shard.", args.task_id)

    args.shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = args.shard_dir / f"shard_{args.task_id:04d}.csv"

    df = pd.DataFrame(rows_out)
    # Enforce SPATIAL_CSV_COLUMNS order; tolerant of missing columns (future
    # metric additions won't break existing shards in a partial rerun).
    ordered_cols = [c for c in SPATIAL_CSV_COLUMNS if c in df.columns]
    if not df.empty:
        df = df[ordered_cols]

    df.to_csv(shard_path, index=False)
    logger.info("Wrote %s (%d rows)", shard_path, len(df))

    # Structured output lines for the SLURM log parser.
    print(f"shard_task_id={args.task_id}")
    print(f"shard_n_scans={n_scans}")
    print(f"shard_elapsed_s={elapsed:.1f}")
    print(f"shard_path={shard_path}")


if __name__ == "__main__":
    main()
