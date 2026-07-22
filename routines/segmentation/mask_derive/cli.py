"""CLI entrypoint for the mask-derive routine.

Usage::

    vena-segmentation-mask-derive <config.yaml>
    python -m routines.segmentation.mask_derive.cli <config.yaml>

Takes exactly one positional argument: the path to a YAML config file.
All parameters (source, corpus registry, target settings, artifact dir) are
read from that file.  No other command-line flags are supported.
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

    # Imports are deferred so that `cli.py` can be imported without side
    # effects (no checkpoint loading, no CUDA, no heavy torch imports).
    from routines.segmentation.mask_derive.engine import MaskDeriveEngine, MaskDeriveRoutineConfig

    cfg = MaskDeriveRoutineConfig.from_yaml(sys.argv[1])
    engine = MaskDeriveEngine(cfg)
    artifact_dir = engine.run()
    print(artifact_dir)


if __name__ == "__main__":
    main()
