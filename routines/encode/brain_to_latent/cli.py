"""Entrypoint for the brain-mask latent encoder routine.

Usage
-----
    vena-encode-brain-to-latent <config.yaml>
    python -m routines.encode.brain_to_latent.cli <config.yaml>
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.logging import RichHandler

from routines.encode.brain_to_latent.engine import (
    BrainToLatentRoutineConfig,
    BrainToLatentRoutineEngine,
)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vena-encode-brain-to-latent")
    parser.add_argument("config", type=Path, help="Path to YAML routine config")
    args = parser.parse_args(argv)

    cfg = BrainToLatentRoutineConfig.from_yaml(args.config)
    _configure_logging(cfg.log_level)
    out = BrainToLatentRoutineEngine(cfg).run()
    logging.getLogger(__name__).info("brain-latent encode done: %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
