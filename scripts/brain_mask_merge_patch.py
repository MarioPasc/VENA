"""Merge a brain-mask patch H5 into the canonical image H5 on the destination.

Sibling to ``brain_latent_merge_patch.py`` but for the image-domain
``masks/brain`` dataset. The patch carries ``ids`` + ``masks/brain``
(int8, native cohort shape); the merger aligns by ``ids`` and rewrites the
target rows in place. The MR ``images/*`` are not touched.

After running this on Picasso, the cohort's latent ``masks/brain_latent``
must be refreshed by re-running ``vena-encode-brain-to-latent`` (which is
already CPU-only post-encode and finishes in <10 minutes per cohort).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _ids(f: h5py.File) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in f["ids"][:]]


def _merge(patch: Path, target: Path) -> dict[str, int]:
    if not patch.exists():
        raise FileNotFoundError(patch)
    if not target.exists():
        raise FileNotFoundError(target)
    with h5py.File(patch, "r") as f_patch, h5py.File(target, "a") as f_target:
        patch_ids = _ids(f_patch)
        target_ids = _ids(f_target)
        patch_brain = np.asarray(f_patch["masks/brain"])
        cleaned = bool(f_patch.attrs.get("brain_cc_cleaned", False))
        unified = bool(f_patch.attrs.get("brain_source_unified", False))

        if "masks/brain" not in f_target:
            raise KeyError(f"{target}: target lacks masks/brain — refusing to create from patch")
        target_ds = f_target["masks/brain"]
        if patch_brain.shape[1:] != target_ds.shape[1:]:
            raise ValueError(
                f"shape mismatch: patch {patch_brain.shape[1:]} vs target {target_ds.shape[1:]}"
            )

        patch_lookup = {pid: i for i, pid in enumerate(patch_ids)}
        target_rows = []
        patch_rows = []
        missing = []
        for j, tid in enumerate(target_ids):
            if tid not in patch_lookup:
                missing.append(tid)
                continue
            target_rows.append(j)
            patch_rows.append(patch_lookup[tid])
        if missing:
            raise KeyError(f"patch missing {len(missing)} target rows; first few: {missing[:5]}")

        # h5py requires monotonic indices for fancy indexing into a dataset,
        # so write row-by-row (cheap: ~501 small writes per cohort).
        for j, p in zip(target_rows, patch_rows, strict=True):
            target_ds[j] = patch_brain[p]
        if cleaned:
            target_ds.attrs["brain_cc_cleaned"] = True
        if unified:
            target_ds.attrs["brain_source_unified"] = True
            target_ds.attrs["brain_source_modalities"] = "t1pre,t1c,t2,flair"
        return {"n_written": len(target_rows), "n_target_rows": len(target_ids)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs",
        nargs="+",
        required=True,
        help="Pairs of <patch_h5> <target_image_h5>",
    )
    args = parser.parse_args(argv)
    if len(args.pairs) % 2 != 0:
        parser.error("--pairs requires an even number of arguments")
    for i in range(0, len(args.pairs), 2):
        patch = Path(args.pairs[i])
        target = Path(args.pairs[i + 1])
        logger.info("merge: %s -> %s", patch, target)
        stats = _merge(patch, target)
        logger.info("  n_written=%d n_target_rows=%d", stats["n_written"], stats["n_target_rows"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
