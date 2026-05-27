"""CLI entrypoint for ``vena-fm-train`` — one positional argument: a YAML config."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.logging import RichHandler

from .engine import FMTrainRoutineConfig, FMTrainRoutineEngine


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vena-fm-train",
        description="Flow-matching training routine — curriculum-aware (S1/S2/S3).",
    )
    parser.add_argument("config", type=Path, help="YAML config file")
    args = parser.parse_args(argv)

    # Logging is configured from YAML; bootstrap rich handler before we read it.
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )

    cfg = FMTrainRoutineConfig.from_yaml(args.config)
    engine = FMTrainRoutineEngine(cfg, config_yaml_path=args.config)
    artifact = engine.run()
    print(f"artifact: {artifact}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
