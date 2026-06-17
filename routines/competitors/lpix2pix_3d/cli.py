"""CLI entrypoint for ``vena-competitor-lpix2pix-3d`` — one positional YAML arg.

Citation: Isola et al. 2017 + Eidex et al. 2025 §4.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.logging import RichHandler

from .engine import LPix2Pix3DCompetitorConfig, LPix2Pix3DCompetitorEngine


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vena-competitor-lpix2pix-3d",
        description=(
            "3D-Latent-Pix2Pix (Isola et al. 2017 conditional-GAN recipe + "
            "Eidex et al. 2025 §4 latent baseline) — conditional GAN for "
            "T1c synthesis from T1pre + FLAIR latents over MAISI-V2."
        ),
    )
    parser.add_argument("config", type=Path, help="YAML config file")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )

    cfg = LPix2Pix3DCompetitorConfig.from_yaml(args.config)
    engine = LPix2Pix3DCompetitorEngine(cfg, config_yaml_path=args.config)
    artifact = engine.run()
    print(f"artifact: {artifact}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
