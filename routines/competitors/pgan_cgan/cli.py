"""CLI entrypoint for ``vena-competitor-pgan`` — one positional YAML arg."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.logging import RichHandler

from .engine import PGANCompetitorConfig, PGANCompetitorEngine


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vena-competitor-pgan",
        description="pGAN-cGAN (Dar et al., 2019) — image-domain T1c synthesis baseline.",
    )
    parser.add_argument("config", type=Path, help="YAML config file")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )

    cfg = PGANCompetitorConfig.from_yaml(args.config)
    engine = PGANCompetitorEngine(cfg, config_yaml_path=args.config)
    artifact = engine.run()
    print(f"artifact: {artifact}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
