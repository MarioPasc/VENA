"""CLI entry point for the latent-augmentation equivariance preflight."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from routines.preflights.latent_aug_equivariance.engine import (
    LatentAugEquivariancePreflightRoutineEngine,
)
from vena.preflight.latent_aug_equivariance import LatentAugEquivarianceConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vena-preflight-latent-aug-equivariance",
        description=(
            "Empirically test whether candidate augmentations preserve the "
            "MAISI-V2 VAE latent space (T_image(D(z)) ≈ D(T_latent(z)))."
        ),
    )
    parser.add_argument("config", type=Path, help="Path to the YAML config.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    cfg = LatentAugEquivarianceConfig.from_yaml(args.config)
    engine = LatentAugEquivariancePreflightRoutineEngine(cfg=cfg, config_yaml_path=args.config)
    out_dir = engine.run()
    logging.getLogger(__name__).info("done — artifacts at %s", out_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
