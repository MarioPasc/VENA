"""Select the best checkpoint from a VENA training run.

Post-hoc checkpoint selection for fixed-length training runs (B4 of
v3a_retraining.md). Replaces Lightning EarlyStopping + ``ema_best.ckpt``
with an offline scan of the exhaustive-validation ``aggregate_cv.csv`` files
written after each cadence epoch.

Usage::

    python scripts/select_checkpoint.py \\
        /path/to/experiments/<run_id> \\
        --metric ssim_brain \\
        --region brain \\
        --nfe 10 \\
        --out /path/to/selected_checkpoint.pt   # optional symlink target

The script scans all ``exhaustive_val/epoch_*/aggregate_cv.csv`` files, reads
the requested (metric, region, nfe) cell, and prints the epoch with the best
value together with the corresponding checkpoint path.

The ``aggregate_cv.csv`` schema (from ``_write_aggregate_cv_csv``) has columns:
    cohort, epoch, nfe, region, psnr_db_mean, ssim_mean, n_patients, ...

``--region`` selects rows with ``region == <value>`` (e.g. ``brain``, ``whole``).
``--nfe`` selects rows with ``nfe == <value>``; omit to aggregate over all NFEs.
``--metric`` may be any column in the CSV (default: ``ssim_mean``).

Exit code 0 when a checkpoint is found, 1 otherwise.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Select the best checkpoint from exhaustive-val aggregate CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("run_dir", type=Path, help="Path to the experiments/<run_id> directory.")
    p.add_argument(
        "--metric",
        default="ssim_mean",
        help="Column in aggregate_cv.csv to maximise.",
    )
    p.add_argument(
        "--region",
        default="brain",
        help="Row filter: keep only rows where the 'region' column equals this value.",
    )
    p.add_argument(
        "--nfe",
        type=int,
        default=None,
        help="Row filter: keep only rows where 'nfe' equals this integer. "
        "Omit to average over all NFEs before comparing epochs.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="If set, create a symlink at this path pointing at the selected checkpoint.",
    )
    p.add_argument(
        "--mode",
        choices=["max", "min"],
        default="max",
        help="Whether to maximise or minimise the metric.",
    )
    return p.parse_args()


def _collect_rows(run_dir: Path, metric: str, region: str, nfe: int | None) -> list[dict]:
    """Return one dict per (epoch, csv_path) with the mean metric value."""
    ev_root = run_dir / "exhaustive_val"
    if not ev_root.is_dir():
        print(f"[select_checkpoint] No exhaustive_val/ directory in {run_dir}", file=sys.stderr)
        return []

    records: list[dict] = []
    for epoch_dir in sorted(ev_root.glob("epoch_*")):
        csv_path = epoch_dir / "aggregate_cv.csv"
        if not csv_path.exists():
            continue
        epoch_str = epoch_dir.name.replace("epoch_", "")
        try:
            epoch_int = int(epoch_str)
        except ValueError:
            continue

        values: list[float] = []
        try:
            with csv_path.open(newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("region", "") != region:
                        continue
                    if nfe is not None and row.get("nfe", "") != str(nfe):
                        continue
                    raw = row.get(metric, "")
                    if raw in ("", None):
                        continue
                    try:
                        values.append(float(raw))
                    except ValueError:
                        continue
        except OSError as exc:
            print(f"[select_checkpoint] Cannot read {csv_path}: {exc}", file=sys.stderr)
            continue

        if not values:
            continue
        mean_val = sum(values) / len(values)
        records.append({"epoch": epoch_int, "value": mean_val, "csv": csv_path})

    return records


def _find_checkpoint(run_dir: Path, epoch: int) -> Path | None:
    """Return the checkpoint path for the given epoch, or None if absent."""
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return None
    # Lightning names: epoch=NNN-step=NNNN.ckpt  or  epoch=NNN.ckpt
    candidates = sorted(ckpt_dir.glob(f"epoch={epoch}*.ckpt"))
    if candidates:
        return candidates[0]
    # Fallback: scan all and pick closest epoch number
    all_ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    for ckpt in all_ckpts:
        name = ckpt.name
        if f"epoch={epoch}" in name:
            return ckpt
    return None


def main() -> None:
    args = _parse_args()
    run_dir: Path = args.run_dir.resolve()

    if not run_dir.is_dir():
        print(f"[select_checkpoint] run_dir does not exist: {run_dir}", file=sys.stderr)
        sys.exit(1)

    records = _collect_rows(run_dir, args.metric, args.region, args.nfe)
    if not records:
        nfe_desc = f", nfe={args.nfe}" if args.nfe is not None else ""
        print(
            f"[select_checkpoint] No rows found for metric={args.metric!r}, "
            f"region={args.region!r}{nfe_desc} in {run_dir / 'exhaustive_val'}",
            file=sys.stderr,
        )
        sys.exit(1)

    best = (
        max(records, key=lambda r: r["value"])
        if args.mode == "max"
        else min(records, key=lambda r: r["value"])
    )
    best_epoch: int = best["epoch"]
    best_value: float = best["value"]

    ckpt = _find_checkpoint(run_dir, best_epoch)
    print(
        f"[select_checkpoint] Best epoch={best_epoch}  {args.metric}={best_value:.6g}  (mode={args.mode})"
    )
    if ckpt is not None:
        print(f"[select_checkpoint] Checkpoint: {ckpt}")
    else:
        print(
            f"[select_checkpoint] WARNING: no checkpoint file found for epoch={best_epoch} "
            f"in {run_dir / 'checkpoints'}",
            file=sys.stderr,
        )

    if args.out is not None and ckpt is not None:
        args.out.unlink(missing_ok=True)
        args.out.symlink_to(ckpt)
        print(f"[select_checkpoint] Symlink created: {args.out} -> {ckpt}")

    if ckpt is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
