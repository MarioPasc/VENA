"""Merge a brain-latent patch H5 into the canonical latent H5 on the destination.

Pairs with ``brain_latent_extract_patch.py``. The merger:

1. Opens the patch H5 and the target latent H5.
2. Aligns rows by ``ids`` (the latent H5 is the authoritative ordering).
3. Writes ``masks/brain_latent`` into the target (creating the dataset when
   absent; refusing to overwrite unless ``--overwrite`` is passed and the
   existing data has the wrong shape).

Idempotent on re-run when the dataset already matches the patch.
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


def _maybe_variants(f: h5py.File) -> list[str] | None:
    if "variants" not in f:
        return None
    return [x.decode() if isinstance(x, bytes) else str(x) for x in f["variants"][:]]


def _ensure_dataset(
    f_target: h5py.File,
    n_rows: int,
    shape_tail: tuple[int, ...],
    overwrite: bool,
) -> h5py.Dataset:
    path = "masks/brain_latent"
    if path in f_target:
        ds = f_target[path]
        if ds.shape == (n_rows, *shape_tail) and not overwrite:
            return ds
        if overwrite:
            del f_target[path]
        else:
            raise RuntimeError(
                f"target {path} already exists with shape {ds.shape}; pass --overwrite"
            )
    return f_target.create_dataset(
        path,
        shape=(n_rows, *shape_tail),
        dtype=np.int8,
        chunks=(1, *shape_tail),
        compression="gzip",
        compression_opts=4,
    )


def _merge(patch: Path, target: Path, overwrite: bool) -> dict[str, int]:
    if not patch.exists():
        raise FileNotFoundError(patch)
    if not target.exists():
        raise FileNotFoundError(target)
    with h5py.File(patch, "r") as f_patch, h5py.File(target, "a") as f_target:
        patch_ids = _ids(f_patch)
        target_ids = _ids(f_target)
        patch_brain = np.asarray(f_patch["masks/brain_latent"])
        v4_flag = bool(f_patch.attrs.get("v4_brain_synthesised_ones", False))

        # Aug schema: align by (id, variant) tuple.
        patch_variants = _maybe_variants(f_patch)
        target_variants = _maybe_variants(f_target)
        if (patch_variants is None) != (target_variants is None):
            raise RuntimeError(
                f"variants presence mismatch: patch={patch_variants is not None} "
                f"target={target_variants is not None}"
            )

        if patch_variants is None:
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
                raise KeyError(
                    f"patch missing {len(missing)} target rows; first few: {missing[:5]}"
                )
        else:
            patch_lookup = {
                (pid, pv): i
                for i, (pid, pv) in enumerate(zip(patch_ids, patch_variants, strict=True))
            }
            target_rows = []
            patch_rows = []
            missing = []
            for j, (tid, tv) in enumerate(zip(target_ids, target_variants, strict=True)):
                key = (tid, tv)
                if key not in patch_lookup:
                    missing.append(key)
                    continue
                target_rows.append(j)
                patch_rows.append(patch_lookup[key])
            if missing:
                raise KeyError(
                    f"patch missing {len(missing)} target (id, variant) rows; "
                    f"first few: {missing[:5]}"
                )

        ds = _ensure_dataset(
            f_target,
            n_rows=len(target_ids),
            shape_tail=patch_brain.shape[1:],
            overwrite=overwrite,
        )
        # Write in patch order, indexed by target rows. h5py supports
        # advanced indexing for writes.
        ds[target_rows] = patch_brain[patch_rows]
        ds.attrs["units"] = "binary"
        ds.attrs["description"] = "brain mask in latent space (max-pool 4 of masks/brain)"
        ds.attrs["producer"] = "scripts.brain_latent_merge_patch:0.1.0"
        ds.attrs["v4_brain_synthesised_ones"] = bool(v4_flag)
        # Aug-latent H5s gain `produced_by_brain_to_latent=True` so the
        # conditional validator (vena.data.h5.augmented.latent_domain) picks
        # up the dataset.
        f_target.attrs["produced_by_brain_to_latent"] = True
        return {"n_written": len(target_rows), "n_target_rows": len(target_ids)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs",
        nargs="+",
        required=True,
        help="Pairs of <patch_h5> <target_latent_h5>",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if len(args.pairs) % 2 != 0:
        parser.error("--pairs requires an even number of arguments")
    for i in range(0, len(args.pairs), 2):
        patch = Path(args.pairs[i])
        target = Path(args.pairs[i + 1])
        logger.info("merge: %s -> %s", patch, target)
        stats = _merge(patch, target, overwrite=args.overwrite)
        logger.info("  n_written=%d n_target_rows=%d", stats["n_written"], stats["n_target_rows"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
