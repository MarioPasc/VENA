"""CLI entrypoint for ``vena-fm-exhaustive-val`` — one positional arg: a job YAML.

Normally launched as a subprocess by the training run's ``ExhaustiveValLauncher``
callback, but can also be run manually against any EMA snapshot for debugging.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.logging import RichHandler

from .engine import ExhaustiveValEngine, ExhaustiveValJobConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vena-fm-exhaustive-val",
        description="Async image-space validation (sample -> decode -> PSNR/SSIM vs real T1c).",
    )
    parser.add_argument("config", type=Path, help="Job YAML config file")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )

    cfg = ExhaustiveValJobConfig.from_yaml(args.config)
    engine = ExhaustiveValEngine(cfg)
    out = engine.run()
    print(f"exhaustive-val artifact: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
