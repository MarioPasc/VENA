"""CLI entry-point for the paired-fidelity routine.

Usage::

    vena-validation-paired-fidelity configs/smoke.yaml
    python -m routines.validation.paired_fidelity.cli configs/smoke.yaml
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from rich.logging import RichHandler

from routines.validation.paired_fidelity.engine import (
    PairedFidelityConfig,
    PairedFidelityEngine,
)


def main() -> None:
    """Entry-point: parse one positional YAML arg and run the engine."""
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <config.yaml>", file=sys.stderr)
        sys.exit(1)

    cfg_path = Path(sys.argv[1])
    cfg = PairedFidelityConfig.from_yaml(cfg_path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    engine = PairedFidelityEngine(cfg)
    artifact_dir = engine.run()
    print(artifact_dir)


if __name__ == "__main__":
    main()
