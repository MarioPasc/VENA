"""CLI entrypoint for ``vena-competitor-resvit`` — one positional YAML arg."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.logging import RichHandler

from .engine import ResViTCompetitorConfig, ResViTCompetitorEngine


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vena-competitor-resvit",
        description="ResViT (Dalmaz, Yurt, Çukur 2022) — image-domain T1c synthesis baseline.",
    )
    parser.add_argument("config", type=Path, help="YAML config file")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )

    cfg = ResViTCompetitorConfig.from_yaml(args.config)
    engine = ResViTCompetitorEngine(cfg, config_yaml_path=args.config)
    artifact = engine.run()
    print(f"artifact: {artifact}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
