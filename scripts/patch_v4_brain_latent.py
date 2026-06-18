"""Seed-replay v4 brain mask → ``masks/brain_latent`` patch for aug latent H5s.

Phase 5 fallback for the 24h-budget path: when the audit-§6.1.1 "Option B
end-to-end rebuild" exceeds the budget, this script reproduces the correct
``masks/brain_latent`` per v4 row using **the same TorchIO transform class
and the same per-row seed** stored in the aug-image H5 (`bank_builder.py`
seed formula). For v1/v2/v3 rows the brain mask is the unwarped image-domain
mask, max-pooled to latent shape.

Inputs (per cohort):
  --source-image-h5     Clean image H5 (post-Phase-1; carries the cleaned
                        ``masks/brain``).
  --aug-image-h5        Bank's image-domain aug H5 (carries ``ids``,
                        ``source_row_index``, ``variants``, root ``seed``,
                        ``world_size``, ``rank``).
  --aug-pipeline-yaml   The bank's aug-pipeline YAML (e.g.
                        ``routines/offline_aug/maisi/configs/aug_pipelines/k4_v1.yaml``).
                        Provides v4's elastic + affine hyperparameters.
  --patch-out           Destination H5 with ``ids``, ``variants``,
                        ``masks/brain_latent`` (consumed by
                        ``brain_latent_merge_patch.py``).

The script is CPU-only; per-row time ~0.5 s (TorchIO warp + max-pool).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torchio as tio
import yaml

# Path bootstrap so the script runs from the repo root without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vena.data.augment.offline.variants import make_variant
from vena.data.h5.augmented import AUG_IMAGE_CROP_BOX
from vena.data.h5.latent_domain.manifest import LATENT_SPATIAL
from vena.model.autoencoder.maisi.preprocessing import CropPadSpec, apply_crop_pad

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _variant_seed(base_seed: int, rank: int, source_row_index: int, variant: str) -> int:
    """Mirror ``bank_builder._variant_seed`` byte for byte."""
    variant_hash = int.from_bytes(hashlib.blake2b(variant.encode(), digest_size=4).digest(), "big")
    return int(np.uint32(base_seed ^ (rank << 16) ^ source_row_index ^ variant_hash))


def _box_native(arr: np.ndarray, crop_origin: tuple[int, int, int]) -> np.ndarray:
    """Crop/pad a 3-D volume onto the bank's common crop box."""
    spec = CropPadSpec(
        crop_origin=tuple(int(v) for v in crop_origin),  # type: ignore[arg-type]
        native_shape=tuple(int(v) for v in arr.shape),  # type: ignore[arg-type]
        target_shape=AUG_IMAGE_CROP_BOX,
    )
    t = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0).unsqueeze(0).float()
    return apply_crop_pad(t, spec)[0, 0].numpy()


def _max_pool4(brain_box: np.ndarray) -> np.ndarray:
    """Image-domain brain box → latent-grid binary mask (1, 48, 56, 48) int8."""
    t = torch.from_numpy(brain_box.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    pooled = F.max_pool3d(t, kernel_size=4, stride=4)
    out = (pooled > 0).to(torch.int8).numpy()[0]
    expected = (1, *LATENT_SPATIAL)
    if out.shape != expected:
        raise RuntimeError(f"latent brain mask shape {out.shape} != {expected}")
    return out


def _load_brain_cache(source_image_h5: Path) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Read ``masks/brain`` and ``crop/origin`` once, key by scan id."""
    cache: dict[str, np.ndarray] = {}
    origins: dict[str, int] = {}
    with h5py.File(source_image_h5, "r") as f:
        ids_raw = f["ids"][:]
        ids = [v.decode() if isinstance(v, bytes) else str(v) for v in ids_raw]
        for row, pid in enumerate(ids):
            cache[pid] = np.asarray(f["masks/brain"][row], dtype=np.int8)
            origins[pid] = row
        crop_origins = f["crop/origin"][:]
    return cache, {pid: tuple(int(v) for v in crop_origins[origins[pid]]) for pid in ids}


def _replay_v4(brain_native: np.ndarray, crop_origin, seed: int, variant_cfg: dict) -> np.ndarray:
    """Re-apply the v4 TorchIO transform to the brain mask using the stored seed."""
    boxed = _box_native(brain_native, crop_origin).astype(np.int8)
    subject = tio.Subject(brain=tio.LabelMap(tensor=torch.from_numpy(boxed).unsqueeze(0).long()))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    transform = make_variant("v4", variant_cfg)
    augmented = transform(subject)
    warped = augmented["brain"].data[0].numpy().astype(np.int8, copy=False)
    return warped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-image-h5", type=Path, required=True)
    parser.add_argument("--aug-image-h5", type=Path, required=True)
    parser.add_argument("--aug-pipeline-yaml", type=Path, required=True)
    parser.add_argument("--patch-out", type=Path, required=True)
    parser.add_argument(
        "--original-world-size",
        type=int,
        default=None,
        help=(
            "If the aug-image H5 was produced by post-merge of N rank shards, "
            "pass N here so per-row seeds use the original `rank = src_idx % N`. "
            "Auto-detected from `merged_from` root attr when omitted."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    for p in (args.source_image_h5, args.aug_image_h5, args.aug_pipeline_yaml):
        if not p.exists():
            logger.error("missing: %s", p)
            return 2
    args.patch_out.parent.mkdir(parents=True, exist_ok=True)

    pipeline = yaml.safe_load(args.aug_pipeline_yaml.read_text())
    v4_cfg = pipeline.get("v4", {})

    brain_cache, origin_cache = _load_brain_cache(args.source_image_h5)
    logger.info("brain cache: %d ids from %s", len(brain_cache), args.source_image_h5.name)

    with h5py.File(args.aug_image_h5, "r") as fa:
        base_seed = int(fa.attrs["seed"])
        post_merge_rank = int(fa.attrs["rank"])
        post_merge_world = int(fa.attrs["world_size"])
        merged_from = fa.attrs.get("merged_from", None)
        ids_raw = fa["ids"][:]
        ids = [v.decode() if isinstance(v, bytes) else str(v) for v in ids_raw]
        src_idx = fa["source_row_index"][:].astype(int)
        var_raw = fa["variants"][:]
        variants = [v.decode() if isinstance(v, bytes) else str(v) for v in var_raw]
        n = len(ids)

    # Resolve the original world_size used at bank-build time. If the H5 was
    # not produced by post-merge, the post-merge attrs are correct.
    if args.original_world_size is not None:
        original_world = int(args.original_world_size)
    elif merged_from is not None:
        try:
            import json as _json

            shards = _json.loads(
                merged_from.decode() if isinstance(merged_from, bytes) else str(merged_from)
            )
            original_world = max(1, len(shards))
        except (TypeError, ValueError, AttributeError):
            original_world = post_merge_world
    else:
        original_world = post_merge_world

    def _rank_for(src_index: int) -> int:
        if original_world <= 1:
            return post_merge_rank
        return int(src_index) % original_world

    logger.info(
        "seeds: base_seed=%d, post_merge=(%d/%d), original_world=%d",
        base_seed,
        post_merge_rank,
        post_merge_world,
        original_world,
    )

    brain_latent = np.zeros((n, 1, *LATENT_SPATIAL), dtype=np.int8)
    n_v4 = 0
    n_other = 0
    for row, (pid, sidx, variant) in enumerate(zip(ids, src_idx, variants, strict=True)):
        if pid not in brain_cache:
            raise KeyError(f"aug row {row} pid {pid!r} missing from source image H5")
        brain_native = brain_cache[pid]
        crop_origin = origin_cache[pid]
        if variant == "v4":
            seed = _variant_seed(base_seed, _rank_for(int(sidx)), int(sidx), variant)
            warped = _replay_v4(brain_native, crop_origin, seed, v4_cfg)
            brain_latent[row] = _max_pool4(warped)
            n_v4 += 1
        else:
            boxed = _box_native(brain_native, crop_origin).astype(np.int8)
            brain_latent[row] = _max_pool4(boxed)
            n_other += 1
        if (row + 1) % 200 == 0:
            logger.info("%d/%d rows (v4=%d others=%d)", row + 1, n, n_v4, n_other)

    if args.dry_run:
        logger.info(
            "dry-run done: would write %s (n=%d v4=%d others=%d)",
            args.patch_out,
            n,
            n_v4,
            n_other,
        )
        return 0

    str_dt = h5py.string_dtype()
    with h5py.File(args.patch_out, "w") as fp:
        fp.attrs["source_aug_image_h5"] = str(args.aug_image_h5)
        fp.attrs["source_image_h5"] = str(args.source_image_h5)
        fp.attrs["base_seed"] = base_seed
        fp.attrs["post_merge_rank"] = post_merge_rank
        fp.attrs["original_world_size"] = original_world
        fp.attrs["n_rows"] = n
        fp.attrs["v4_rows"] = n_v4
        fp.attrs["v4_brain_synthesised_ones"] = False
        fp.create_dataset("ids", data=np.asarray(ids, dtype=object), dtype=str_dt)
        fp.create_dataset("variants", data=np.asarray(variants, dtype=object), dtype=str_dt)
        ds = fp.create_dataset(
            "masks/brain_latent",
            data=brain_latent,
            dtype="int8",
            chunks=(1, 1, *LATENT_SPATIAL),
            compression="gzip",
            compression_opts=4,
        )
        ds.attrs["units"] = "binary"
        ds.attrs["description"] = (
            "Brain mask in latent space; v4 rows obtained by TorchIO seed-replay of the "
            "elastic+affine transform applied during bank build (see scripts.patch_v4_brain_latent)."
        )
        ds.attrs["producer"] = "scripts.patch_v4_brain_latent:0.1.0"
        ds.attrs["v4_brain_synthesised_ones"] = False
    logger.info("wrote %s (n=%d v4=%d others=%d)", args.patch_out, n, n_v4, n_other)
    return 0


if __name__ == "__main__":
    sys.exit(main())
