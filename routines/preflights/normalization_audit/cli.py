"""CLI entry point for the V3 normalisation audit."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

# Force the headless backend before any pyplot import elsewhere.
matplotlib.use("Agg")

import yaml

from routines.preflights.normalization_audit.engine import (
    NormalizationAuditRoutineEngine,
)
from vena.preflight.normalization_audit import (
    NormalizationAuditConfig,
)


def _load_config(path: Path) -> NormalizationAuditConfig:
    blob = yaml.safe_load(Path(path).read_text())
    return NormalizationAuditConfig.model_validate(blob)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vena-preflight-normalization-audit",
        description="V3 normalisation audit — sweep variants, decide winner.",
    )
    parser.add_argument("config", type=Path, help="YAML config path.")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase log verbosity (-v: INFO, -vv: DEBUG).",
    )
    args = parser.parse_args(argv)

    level = logging.WARNING
    if args.verbose >= 2:
        level = logging.DEBUG
    elif args.verbose >= 1:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    cfg = _load_config(args.config)
    engine = NormalizationAuditRoutineEngine(cfg=cfg)
    out_dir = engine.run()
    logging.getLogger(__name__).info("done — artefacts at %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
