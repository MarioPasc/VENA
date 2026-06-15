"""CLI entrypoint for ``vena-competitor-dit-3d`` — one positional YAML arg.

Citation: Peebles & Xie 2023 + Eidex et al. 2025 §4.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.logging import RichHandler

from .engine import DiT3DCompetitorConfig, DiT3DCompetitorEngine


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vena-competitor-dit-3d",
        description=(
            "3D-DiT (Peebles & Xie 2023 backbone, Eidex et al. 2025 §4 "
            "baseline recipe) — transformer-backbone latent rectified flow "
            "for T1c synthesis from T1pre + FLAIR latents."
        ),
    )
    parser.add_argument("config", type=Path, help="YAML config file")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )

    cfg = DiT3DCompetitorConfig.from_yaml(args.config)
    engine = DiT3DCompetitorEngine(cfg, config_yaml_path=args.config)
    artifact = engine.run()
    print(f"artifact: {artifact}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
