"""Shard CLI — one SLURM array task per (method, cohort, nfe) prediction file.

Reads one row from the manifest by ``SLURM_ARRAY_TASK_ID``, streams all
scan pairs from that prediction H5 file, computes paired metrics, and
writes ``shard_<NNNN>.csv`` to the shard directory.  No patient collapse,
no stats, no figures — those run once in the merge step.

Usage::

    python -m routines.validation.paired_fidelity.cli_shard \\
        --manifest /path/to/manifest.csv \\
        --task-id "$SLURM_ARRAY_TASK_ID" \\
        --shard-dir /path/to/shards \\
        --config /path/to/smoke_picasso.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from routines.validation.paired_fidelity.engine import PairedFidelityConfig
from vena.validation.io import ReferenceCache, iter_scans
from vena.validation.metrics_paired import MetricConfig, compute_paired_metrics

logger = logging.getLogger(__name__)

# Frozen column order for the shard CSV — must match engine._METRIC_COLS + id_cols.
_ID_COLS = ["method", "cohort", "ring", "nfe", "scan_id", "patient_id", "pred_mode"]
_METRIC_COLS = [
    "mae_brain",
    "mae_wt",
    "mae_bg_undilated",
    "rmse_brain",
    "rmse_wt",
    "rmse_bg_undilated",
    "psnr_brain",
    "psnr_wt",
    "psnr_bg_undilated",
    "ssim_brain",
    "ssim_wt",
    "ssim_bg_undilated",
    "ms_ssim_brain",
    "ms_ssim_wt",
    "ms_ssim_bg_undilated",
    "zgd",
    "inference_seconds",
    "peak_vram_mb",
    "n_brain_voxels",
    "n_wt_voxels",
    "n_bg_undilated_voxels",
    "raw_p995",
]


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(
        description="Process one (method, cohort, nfe) prediction file → shard CSV.",
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--task-id", required=True, type=int, dest="task_id")
    parser.add_argument("--shard-dir", required=True, type=Path, dest="shard_dir")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    cfg = PairedFidelityConfig.from_yaml(args.config)

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

    metric_cfg = MetricConfig(
        data_range=1.0,
        ssim_window_size=cfg.ssim_window_size,
        ssim_window_sigma=cfg.ssim_window_sigma,
        ms_ssim_weights=cfg.ms_ssim_weights,
        ms_ssim_bbox_margin=cfg.ms_ssim_bbox_margin,
        dilate_k=cfg.dilate_k,
    )

    t0 = time.perf_counter()
    ref_cache = ReferenceCache()
    rows_out: list[dict[str, object]] = []
    n_scans = 0

    try:
        for scan in iter_scans(pred_path, reference_cache=ref_cache):
            m = compute_paired_metrics(scan, metric_cfg)
            rows_out.append(m.to_flat_dict())
            n_scans += 1
    except Exception as exc:
        logger.error("Fatal error on task %d (%s): %s", args.task_id, pred_path, exc)
        raise

    elapsed = time.perf_counter() - t0
    logger.info(
        "Task %d: %d scans in %.1f s (%.2f s/scan)",
        args.task_id,
        n_scans,
        elapsed,
        elapsed / max(n_scans, 1),
    )

    if not rows_out:
        logger.warning("Task %d produced zero scans — writing empty shard.", args.task_id)

    args.shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = args.shard_dir / f"shard_{args.task_id:04d}.csv"

    df = pd.DataFrame(rows_out)
    # Enforce column order; tolerant of missing columns (new metrics added later).
    ordered_cols = [c for c in _ID_COLS + _METRIC_COLS if c in df.columns]
    if not df.empty:
        df = df[ordered_cols]

    df.to_csv(shard_path, index=False)
    logger.info("Wrote %s (%d rows)", shard_path, len(df))
    # Structured output for the SLURM log parser.
    print(f"shard_task_id={args.task_id}")
    print(f"shard_n_scans={n_scans}")
    print(f"shard_elapsed_s={elapsed:.1f}")
    print(f"shard_path={shard_path}")


if __name__ == "__main__":
    main()
