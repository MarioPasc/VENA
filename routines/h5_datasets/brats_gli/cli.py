"""Entrypoint for the BraTS-GLI image-domain H5 conversion routine.

Usage
-----
    vena-h5-brats-gli <config.yaml>
    python -m routines.h5_datasets.brats_gli.cli <config.yaml>
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.logging import RichHandler

from routines.h5_datasets.brats_gli.engine import (
    BraTSGLIH5RoutineConfig,
    BraTSGLIH5RoutineEngine,
)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vena-h5-brats-gli")
    parser.add_argument("config", type=Path, help="Path to YAML routine config")
    args = parser.parse_args(argv)

    cfg = BraTSGLIH5RoutineConfig.from_yaml(args.config)
    _configure_logging(cfg.log_level)
    out = BraTSGLIH5RoutineEngine(cfg).run()
    logging.getLogger(__name__).info("BraTS-GLI image H5: %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
