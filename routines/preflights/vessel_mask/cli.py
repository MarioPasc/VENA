"""CLI entrypoint for the vessel-mask preflight routine.

Usage
-----
    vena-preflight-vessel-mask <config.yaml>
    python -m routines.preflights.vessel_mask.cli <config.yaml>
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.logging import RichHandler

from routines.preflights.vessel_mask.engine.vessel_mask_engine import (
    VesselMaskPreflightRoutineEngine,
)
from vena.preflight.vessel_mask import VesselMaskPreflightConfig


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vena-preflight-vessel-mask")
    parser.add_argument("config", type=Path, help="Path to YAML routine config")
    args = parser.parse_args(argv)

    cfg = VesselMaskPreflightConfig.from_yaml(args.config)
    _configure_logging(cfg.log_level)
    out = VesselMaskPreflightRoutineEngine(cfg).run()
    logging.getLogger(__name__).info("Vessel-mask preflight artifact: %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
