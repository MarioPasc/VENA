"""CLI entrypoint for the validate-masks routine.

Usage::

    vena-segmentation-validate-masks <config.yaml>
    python -m routines.segmentation.validate_masks.cli <config.yaml>

Takes exactly one positional argument: the path to a YAML config file.
All parameters are read from that file.  No other flags are supported.
"""

from __future__ import annotations

import sys


def main() -> None:
    """Parse the single YAML argument and run the engine."""
    if len(sys.argv) != 2:
        print(
            f"Usage: {sys.argv[0]} <config.yaml>\n"
            "Takes exactly one positional argument: path to a YAML config.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Deferred imports — no heavy work at module scope.
    from routines.segmentation.validate_masks.engine import (
        ValidateMasksEngine,
        ValidateMasksRoutineConfig,
    )

    cfg = ValidateMasksRoutineConfig.from_yaml(sys.argv[1])
    engine = ValidateMasksEngine(cfg)
    artifact_dir = engine.run()
    print(artifact_dir)


if __name__ == "__main__":
    main()
