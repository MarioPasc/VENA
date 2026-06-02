"""Entrypoint for the REMBRANDT HD-BET skull-strip routine.

Usage
-----
    vena-preprocess-rembrandt-skullstrip <config.yaml>
    python -m routines.preprocess.rembrandt_skullstrip.cli <config.yaml>
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.logging import RichHandler

from routines.preprocess.rembrandt_skullstrip.engine import (
    REMBRANDTSkullStripRoutineConfig,
    REMBRANDTSkullStripRoutineEngine,
)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vena-preprocess-rembrandt-skullstrip")
    parser.add_argument("config", type=Path, help="Path to YAML routine config")
    args = parser.parse_args(argv)

    cfg = REMBRANDTSkullStripRoutineConfig.from_yaml(args.config)
    _configure_logging(cfg.log_level)
    out = REMBRANDTSkullStripRoutineEngine(cfg).run()
    logging.getLogger(__name__).info("REMBRANDT skull-strip output: %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
