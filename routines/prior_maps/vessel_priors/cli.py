"""Entrypoint for the vessel-priors routine.

Usage
-----
    vena-prior-maps-vessel <config.yaml>
    python -m routines.prior_maps.vessel_priors.cli <config.yaml>
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.logging import RichHandler
from routines.prior_maps.vessel_priors.engine.vessel_priors_engine import (
    VesselPriorsRoutineEngine,
)

from vena.prior_maps.vessel_priors import VesselPriorsRoutineConfig


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vena-prior-maps-vessel")
    parser.add_argument("config", type=Path, help="Path to YAML routine config")
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Skip prediction; re-render collages from existing soft/mask NIfTIs.",
    )
    args = parser.parse_args(argv)

    cfg = VesselPriorsRoutineConfig.from_yaml(args.config)
    _configure_logging(cfg.log_level)
    out = VesselPriorsRoutineEngine(cfg).run(figures_only=args.figures_only)
    logging.getLogger(__name__).info("Vessel-priors routine artifact: %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
