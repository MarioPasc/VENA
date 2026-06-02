"""CLI entry point for the cohort_dedup preflight.

Usage::

    vena-preflight-cohort-dedup <yaml>

The YAML schema is :class:`vena.preflight.cohort_dedup.CohortDedupConfig`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from routines.preflights.cohort_dedup.engine import CohortDedupPreflightRoutineEngine
from vena.preflight.cohort_dedup import CohortDedupConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vena-preflight-cohort-dedup",
        description=(
            "Build per-cohort allow-lists from the corpus registry + "
            "BraTS-2021 ↔ TCIA mapping xlsx and emit a versioned decision.json."
        ),
    )
    parser.add_argument("config", type=Path, help="Path to the YAML config.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    cfg = CohortDedupConfig.from_yaml(args.config)
    engine = CohortDedupPreflightRoutineEngine(cfg=cfg, config_yaml_path=args.config)
    out_dir = engine.run()
    logging.getLogger(__name__).info("done — artifacts at %s", out_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
