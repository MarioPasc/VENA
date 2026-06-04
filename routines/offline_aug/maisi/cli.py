"""CLI entrypoint for the offline-augmentation routine (MAISI VAE).

One positional argument: the routine YAML config. Logging level comes from
``log_level`` in the YAML.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rich.logging import RichHandler

from routines.offline_aug.maisi.engine import (
    OfflineAugMaisiRoutineConfig,
    OfflineAugMaisiRoutineEngine,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="vena-offline-aug-maisi",
        description=(
            "Build the offline image-domain + latent-domain augmentation bank for "
            "one cohort × one rank (scan-level shard)."
        ),
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the routine YAML config.",
    )
    args = parser.parse_args()

    cfg = OfflineAugMaisiRoutineConfig.from_yaml(args.config)
    logging.basicConfig(
        level=cfg.log_level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    run_dir = OfflineAugMaisiRoutineEngine(cfg).run()
    logging.getLogger(__name__).info("offline-aug run complete: %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
