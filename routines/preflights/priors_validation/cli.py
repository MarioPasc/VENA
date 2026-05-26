"""Entrypoint for the priors-validation preflight routine.

Usage
-----
    vena-preflight-priors-validation <config.yaml>
    python -m routines.preflights.priors_validation.cli <config.yaml>
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.logging import RichHandler

from routines.preflights.priors_validation.engine.priors_validation_engine import (
    PriorsValidationRoutineEngine,
)
from vena.preflight.priors_validation import PriorsValidationRoutineConfig


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vena-preflight-priors-validation")
    parser.add_argument("config", type=Path, help="Path to YAML routine config")
    args = parser.parse_args(argv)

    cfg = PriorsValidationRoutineConfig.from_yaml(args.config)
    _configure_logging(cfg.log_level)
    out = PriorsValidationRoutineEngine(cfg).run()
    logging.getLogger(__name__).info("Priors-validation routine artifact: %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
