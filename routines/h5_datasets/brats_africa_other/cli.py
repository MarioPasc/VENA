"""Entrypoint for the BraTS-Africa OtherNeoplasms image-domain H5 routine.

Usage
-----
    vena-h5-brats-africa-other <config.yaml>
    python -m routines.h5_datasets.brats_africa_other.cli <config.yaml>
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.logging import RichHandler

from routines.h5_datasets.brats_africa_other.engine import (
    BraTSAfricaOtherH5RoutineConfig,
    BraTSAfricaOtherH5RoutineEngine,
)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vena-h5-brats-africa-other")
    parser.add_argument("config", type=Path, help="Path to YAML routine config")
    args = parser.parse_args(argv)

    cfg = BraTSAfricaOtherH5RoutineConfig.from_yaml(args.config)
    _configure_logging(cfg.log_level)
    out = BraTSAfricaOtherH5RoutineEngine(cfg).run()
    logging.getLogger(__name__).info("BraTS-Africa-Other image H5: %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
