"""In-place harmonisation of ``masks/brain`` to union-of-4 + CC clean.

Two cohorts (BraTS-GLI, IvyGAP) historically derived ``masks/brain`` from
``t1pre > 0`` only; every other VENA-computed cohort uses the union of all
four modalities' nonzero voxels followed by ``clean_brain_mask``. The audit
at ``.claude/notes/data/2026-06-18_data_audit.md`` §2.2/§4.3 flags this as a
silent inconsistency. This script rewrites the affected H5s in place:

* Re-derives ``masks/brain`` row-by-row as the union of the four modalities.
* Applies ``clean_brain_mask`` with the project default (1000-voxel CC floor).
* Writes the new mask back into the image H5.
* Records ``brain_source_unified`` / ``brain_source_modalities`` / SHA-256 of
  the run on the dataset attrs.

After this script, re-run ``vena-encode-brain-to-latent`` against the latent
H5 to refresh ``masks/brain_latent``. The MR latents are not affected.

Usage::

    python scripts/harmonize_brain_source_inplace.py \
        --image-h5 /path/to/BraTS_GLI_image.h5 \
        --image-h5 /path/to/IvyGAP_image.h5 \
        --dry-run

The per-row delta CSV lands at ``<basename>.brain_source_harmonize.csv``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import h5py

# Path bootstrap so the script runs from the repo root without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vena.data.h5.shared.brain_mask import recompute_union_of_four

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _process_one(path: Path, min_component_voxels: int, dry_run: bool) -> dict[str, int]:
    stats = {"n_rows": 0, "n_rows_changed": 0, "voxels_added_total": 0, "voxels_dropped_total": 0}
    summary_rows: list[dict[str, int]] = []
    rewrites: list[tuple[int, object]] = []

    for row_idx, scan_id, old_mask, new_mask in recompute_union_of_four(
        path, min_component_voxels=min_component_voxels
    ):
        stats["n_rows"] += 1
        added = int(((new_mask > 0) & ~(old_mask > 0)).sum())
        dropped = int((~(new_mask > 0) & (old_mask > 0)).sum())
        if added or dropped:
            stats["n_rows_changed"] += 1
            stats["voxels_added_total"] += added
            stats["voxels_dropped_total"] += dropped
            summary_rows.append(
                {
                    "row": row_idx,
                    "id": scan_id,
                    "brain_before": int((old_mask > 0).sum()),
                    "brain_after": int((new_mask > 0).sum()),
                    "voxels_added": added,
                    "voxels_dropped": dropped,
                }
            )
            rewrites.append((row_idx, new_mask))

    if not dry_run and rewrites:
        with h5py.File(path, "r+") as f:
            ds = f["masks/brain"]
            for row_idx, new_mask in rewrites:
                ds[row_idx] = new_mask
            ds.attrs["brain_source_unified"] = True
            ds.attrs["brain_source_modalities"] = "t1pre,t1c,t2,flair"
            ds.attrs["brain_cc_min_component_voxels"] = int(min_component_voxels)

    out_csv = path.with_suffix(path.suffix + ".brain_source_harmonize.csv")
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "row",
                "id",
                "brain_before",
                "brain_after",
                "voxels_added",
                "voxels_dropped",
            ],
        )
        w.writeheader()
        w.writerows(summary_rows)

    logger.info(
        "%s: %d/%d rows changed (+%d /-%d voxels) → %s",
        path.name,
        stats["n_rows_changed"],
        stats["n_rows"],
        stats["voxels_added_total"],
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

    total = {"n_rows": 0, "n_rows_changed": 0, "voxels_added_total": 0, "voxels_dropped_total": 0}
    for path in args.image_h5:
        if not path.exists():
            logger.error("missing: %s", path)
            return 2
        stats = _process_one(path, args.min_component_voxels, args.dry_run)
        for k, v in stats.items():
            total[k] += v

    logger.info(
        "TOTAL: %d/%d rows changed across %d files (+%d /-%d voxels)",
        total["n_rows_changed"],
        total["n_rows"],
        len(args.image_h5),
        total["voxels_added_total"],
        total["voxels_dropped_total"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
