"""CLI entrypoint for ``vena-segmentation-train`` — one positional argument: a YAML config.

Usage::

    vena-segmentation-train <config.yaml>
    python -m routines.segmentation.train.cli <config.yaml>

Takes exactly one positional argument: the path to a YAML config file.
All parameters are read from that file.  No other flags are supported.
"""

from __future__ import annotations

import sys


def main() -> None:
    """Parse the single YAML argument and run the segmentation training engine."""
    if len(sys.argv) != 2:
        print(
            f"Usage: {sys.argv[0]} <config.yaml>\n"
            "Takes exactly one positional argument: path to a YAML config.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Deferred imports — no heavy work at module scope.
    from routines.segmentation.train.engine import SegTrainEngine
    from vena.segmentation.config import SegmentationConfig

    cfg = SegmentationConfig.from_yaml(sys.argv[1])
    engine = SegTrainEngine(cfg)
    run_dir = engine.run()
    print(f"RUN_DIR={run_dir}")


if __name__ == "__main__":
    main()
