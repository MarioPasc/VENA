"""CLI entrypoint for ``vena-competitor-lddpm-3d`` — one positional YAML arg.

Citation: Ho et al. 2020 (DDPM) + Eidex et al. 2025 §4 (3D + MAISI latents).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.logging import RichHandler

from .engine import LDDPM3DCompetitorConfig, LDDPM3DCompetitorEngine


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vena-competitor-lddpm-3d",
        description=(
            "3D-LDDPM (Ho et al. 2020 DDPM scheduler + Eidex et al. 2025 §4 "
            "baseline recipe) — latent denoising diffusion model for T1c "
            "synthesis from T1pre + FLAIR latents. Uses the paper-faithful "
            "MAISI U-Net (same backbone as T1C-RFlow) so the only delta vs "
            "T1C-RFlow is the scheduler (DDPM vs RFlow) and loss "
            "(epsilon-MSE vs velocity-L1)."
        ),
    )
    parser.add_argument("config", type=Path, help="YAML config file")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )

    cfg = LDDPM3DCompetitorConfig.from_yaml(args.config)
    engine = LDDPM3DCompetitorEngine(cfg, config_yaml_path=args.config)
    artifact = engine.run()
    print(f"artifact: {artifact}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
