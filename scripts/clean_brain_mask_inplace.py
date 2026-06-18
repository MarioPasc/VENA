"""In-place CC-clean of ``masks/brain`` in cohort image H5 files.

Re-applies :func:`vena.data.h5.shared.brain_mask.clean_brain_mask` to every row
of ``masks/brain`` and writes the cleaned mask back. The MR `images/*` and
`masks/tumor` datasets are not touched. After running this script, re-run
``vena-encode-brain-to-latent`` against the affected latent H5 to refresh
``masks/brain_latent`` (max-pool-4 of the cleaned mask).

Idempotent: re-running on an already-clean H5 leaves it byte-identical.

Usage::

    python scripts/clean_brain_mask_inplace.py \
        --image-h5 /path/to/IvyGAP_image.h5 \
        --image-h5 /path/to/BraTS_GLI_image.h5 \
        --min-component-voxels 1000

The script writes the per-row delta count to a summary CSV alongside each
H5 (``<basename>.brain_cc_clean.csv``).
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import h5py
import numpy as np

# Path bootstrap so the script runs from the repo root without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vena.data.h5.shared.brain_mask import clean_brain_mask

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _process_one(path: Path, min_component_voxels: int, dry_run: bool) -> dict[str, int]:
    stats = {"n_rows": 0, "n_rows_changed": 0, "voxels_dropped_total": 0}
    summary_rows: list[dict[str, int]] = []
    with h5py.File(path, "r+") as f:
        if "masks/brain" not in f:
            raise KeyError(f"{path}: missing masks/brain")
        ds = f["masks/brain"]
        n = ds.shape[0]
        stats["n_rows"] = n
        ids_raw = f["ids"][:]
        ids = [v.decode() if isinstance(v, bytes) else str(v) for v in ids_raw]
        for i in range(n):
            arr = np.asarray(ds[i])
            before = int((arr > 0).sum())
            cleaned = clean_brain_mask(arr, min_component_voxels=min_component_voxels)
            after = int((cleaned > 0).sum())
            dropped = before - after
            if dropped > 0:
                stats["n_rows_changed"] += 1
                stats["voxels_dropped_total"] += dropped
                summary_rows.append(
                    {
                        "row": i,
                        "id": ids[i],
                        "brain_before": before,
                        "brain_after": after,
                        "voxels_dropped": dropped,
                    }
                )
                if not dry_run:
                    ds[i] = cleaned
        if not dry_run:
            ds.attrs["brain_cc_cleaned"] = True
            ds.attrs["brain_cc_min_component_voxels"] = int(min_component_voxels)
    # Write the per-row delta CSV next to the H5.
    out_csv = path.with_suffix(path.suffix + ".brain_cc_clean.csv")
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["row", "id", "brain_before", "brain_after", "voxels_dropped"]
        )
        w.writeheader()
        w.writerows(summary_rows)
    logger.info(
        "%s: %d/%d rows changed, %d voxels dropped (summary → %s)",
        path.name,
        stats["n_rows_changed"],
        stats["n_rows"],
        stats["voxels_dropped_total"],
        out_csv.name,
    )
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-h5",
        action="append",
        required=True,
        type=Path,
        help="Path to a cohort image H5 (repeatable).",
    )
    parser.add_argument("--min-component-voxels", type=int, default=1000)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute deltas without writing back to the H5.",
    )
    args = parser.parse_args(argv)

    total = {"n_rows": 0, "n_rows_changed": 0, "voxels_dropped_total": 0}
    for path in args.image_h5:
        if not path.exists():
            logger.error("missing: %s", path)
            return 2
        stats = _process_one(path, args.min_component_voxels, args.dry_run)
        for k, v in stats.items():
            total[k] += v

    logger.info(
        "TOTAL: %d/%d rows changed across %d files (%d voxels dropped)",
        total["n_rows_changed"],
        total["n_rows"],
        len(args.image_h5),
        total["voxels_dropped_total"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
