"""Entrypoint for the MAISI latent-encoding routine.

Usage
-----
    vena-encode-maisi <config.yaml>
    python -m routines.encode.maisi.cli <config.yaml>
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.logging import RichHandler

from routines.encode.maisi.engine import (
    EncodeMaisiRoutineConfig,
    EncodeMaisiRoutineEngine,
)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vena-encode-maisi")
    parser.add_argument("config", type=Path, help="Path to YAML routine config")
    args = parser.parse_args(argv)

    cfg = EncodeMaisiRoutineConfig.from_yaml(args.config)
    _configure_logging(cfg.log_level)
    out = EncodeMaisiRoutineEngine(cfg).run()
    logging.getLogger(__name__).info("UCSF-PDGM MAISI latent H5: %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
