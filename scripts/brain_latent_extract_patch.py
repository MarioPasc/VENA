"""Extract ``masks/brain_latent`` from a list of H5 files into compact patch files.

Used to transfer the 2026-06-09 brain-mask encoding (CHANGE 2 of the overhaul
note) from server-3 to Picasso without re-uploading the whole latent H5
(~85 GB total). The patches carry only the new dataset and the
``ids`` array needed to align rows on the destination — typical sizes
≈10–50 MB per cohort after gzip-4.

Pair with ``brain_latent_merge_patch.py`` on the destination.
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
        if "masks/brain_latent" not in f_in:
            raise KeyError(f"{src} has no masks/brain_latent")
        ids = np.asarray(f_in["ids"][:])
        brain = np.asarray(f_in["masks/brain_latent"][:])
        v4_flag = bool(f_in["masks/brain_latent"].attrs.get("v4_brain_synthesised_ones", False))
        variants = None
        if "variants" in f_in:
            variants = np.asarray(f_in["variants"][:])
    with h5py.File(dst, "w") as f_out:
        f_out.attrs["source_path"] = str(src)
        f_out.attrs["v4_brain_synthesised_ones"] = v4_flag
        str_dt = h5py.string_dtype()
        f_out.create_dataset("ids", data=ids, dtype=str_dt)
        f_out.create_dataset(
            "masks/brain_latent",
            data=brain,
            dtype=np.int8,
            chunks=(1, *brain.shape[1:]),
            compression="gzip",
            compression_opts=4,
        )
        if variants is not None:
            f_out.create_dataset("variants", data=variants, dtype=str_dt)
    return {"n_rows": int(brain.shape[0]), "v4_ones": int(v4_flag)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs",
        nargs="+",
        required=True,
        help="Pairs of <source_h5> <patch_h5_out> (space-separated).",
    )
    args = parser.parse_args(argv)
    if len(args.pairs) % 2 != 0:
        parser.error("--pairs requires an even number of arguments")
    summary = []
    for i in range(0, len(args.pairs), 2):
        src = Path(args.pairs[i])
        dst = Path(args.pairs[i + 1])
        logger.info("extract: %s -> %s", src, dst)
        stats = _extract(src, dst)
        summary.append({"src": str(src), "dst": str(dst), **stats})
        logger.info(
            "  n_rows=%d v4_ones=%s size=%.1f MB",
            stats["n_rows"],
            stats["v4_ones"],
            dst.stat().st_size / 1e6,
        )
    logger.info("done: %d files written", len(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
