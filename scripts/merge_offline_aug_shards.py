"""Merge per-rank image+latent aug-H5 shards for one cohort.

Reads a small ``merges/<cohort>_merge.yaml`` produced by
``scripts/gen_offline_aug_configs.py`` (just paths) and concatenates the
two ranks' shards into the cohort-level ``<COHORT>_{image,latents}_aug.h5``.

This is a thin CLI; the actual concat logic lives in
:mod:`vena.data.augment.offline.bank_builder` (image side) and in
:meth:`routines.offline_aug.maisi.engine._merge_latent_shards` (latent side).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import yaml
from rich.logging import RichHandler
from routines.offline_aug.maisi.engine import (
    OfflineAugMaisiRoutineConfig,
    OfflineAugMaisiRoutineEngine,
)

from vena.data.augment.offline.bank_builder import merge_aug_image_h5_shards
from vena.data.h5.augmented import assert_aug_latent_h5_valid
from vena.data.h5.shared import sha256_file


def _read_aug_config_sha(image_shard: Path) -> str:
    with h5py.File(image_shard, "r") as f:
        return str(f.attrs["aug_config_sha256"])


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="vena-merge-offline-aug",
        description="Merge per-rank aug-H5 shards into a cohort H5.",
    )
    parser.add_argument("merge_yaml", type=Path)
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["t1pre", "t1c", "t2", "flair"],
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    log = logging.getLogger(__name__)

    payload = yaml.safe_load(args.merge_yaml.read_text())
    cohort = payload["cohort"]
    shards_image = [Path(p) for p in payload["shards_image"]]
    shards_latent = [Path(p) for p in payload["shards_latent"]]
    merged_image = Path(payload["merged_image_aug_h5"])
    merged_latent = Path(payload["merged_latents_aug_h5"])
    aug_config_sha256 = _read_aug_config_sha(shards_image[0])

    log.info("merging %d image shards → %s", len(shards_image), merged_image)
    merge_aug_image_h5_shards(
        shards=shards_image,
        merged_path=merged_image,
        cohort=cohort,
        modalities=args.modalities,
        overwrite=args.overwrite,
    )

    # Reuse the routine engine's latent-merge by instantiating a minimal config.
    log.info("merging %d latent shards → %s", len(shards_latent), merged_latent)
    cfg = OfflineAugMaisiRoutineConfig(
        cohort=cohort,
        source_image_h5=merged_image,
        autoencoder_checkpoint=Path("/dev/null"),  # not used for merge
        aug_pipeline_yaml=Path("/dev/null"),  # not used for merge
        image_aug_h5_path=merged_image,
        latent_aug_h5_path=merged_latent,
        modalities=args.modalities,
        world_size=1,
        rank=0,
        overwrite=args.overwrite,
    )
    engine = OfflineAugMaisiRoutineEngine(cfg)
    # We only need mask_channels — read from the first latent shard.
    with h5py.File(shards_latent[0], "r") as f:
        mask_channels = int(f["masks/tumor_latent"].shape[1])
    engine._merge_latent_shards(
        shards=shards_latent,
        merged_path=merged_latent,
        cohort=cohort,
        modalities=args.modalities,
        mask_channels=mask_channels,
        aug_config_sha256=aug_config_sha256,
        overwrite=args.overwrite,
    )
    assert_aug_latent_h5_valid(merged_latent, cohort, args.modalities, mask_channels)
    log.info(
        "merge complete (cohort=%s): image=%s latent=%s",
        cohort,
        sha256_file(merged_image)[:12],
        sha256_file(merged_latent)[:12],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
