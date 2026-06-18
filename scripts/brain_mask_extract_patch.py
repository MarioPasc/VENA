"""Extract ``masks/brain`` from image H5s into compact patch files.

Sibling to :mod:`brain_latent_extract_patch` but for the IMAGE-domain brain
mask, used after Phase 1 (CC-clean + brain-source harmonization) to move
the masks/brain delta to Picasso without retransferring a multi-GB image
H5. Patch is keyed by ``ids`` so a row-shuffled destination still merges
correctly.

Sizes: ``masks/brain`` is int8 at native cohort shape; gzip-4 typically
compresses to <500 MB per cohort even for BraTS-GLI.

Pair with ``brain_mask_merge_patch.py`` on the destination.
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


def _extract(src: Path, dst: Path) -> dict[str, int]:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(src, "r") as f_in:
        if "masks/brain" not in f_in:
            raise KeyError(f"{src} has no masks/brain")
        ids = np.asarray(f_in["ids"][:])
        brain = np.asarray(f_in["masks/brain"][:])
        cleaned = bool(f_in["masks/brain"].attrs.get("brain_cc_cleaned", False))
        unified = bool(f_in["masks/brain"].attrs.get("brain_source_unified", False))
    with h5py.File(dst, "w") as f_out:
        f_out.attrs["source_path"] = str(src)
        f_out.attrs["brain_cc_cleaned"] = cleaned
        f_out.attrs["brain_source_unified"] = unified
        str_dt = h5py.string_dtype()
        f_out.create_dataset("ids", data=ids, dtype=str_dt)
        f_out.create_dataset(
            "masks/brain",
            data=brain,
            dtype=np.int8,
            chunks=(1, *brain.shape[1:]),
            compression="gzip",
            compression_opts=4,
        )
    return {"n_rows": int(brain.shape[0])}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs",
        nargs="+",
        required=True,
        help="Pairs of <source_image_h5> <patch_h5_out> (space-separated).",
    )
    args = parser.parse_args(argv)
    if len(args.pairs) % 2 != 0:
        parser.error("--pairs requires an even number of arguments")
    for i in range(0, len(args.pairs), 2):
        src = Path(args.pairs[i])
        dst = Path(args.pairs[i + 1])
        logger.info("extract: %s -> %s", src, dst)
        stats = _extract(src, dst)
        logger.info(
            "  n_rows=%d size=%.1f MB",
            stats["n_rows"],
            dst.stat().st_size / 1e6,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
