"""CLI entry point for the spatial_residual validation routine.

Usage
-----
    vena-validation-spatial-residual config.yaml
    python -m routines.validation.spatial_residual.cli config.yaml
"""

from __future__ import annotations

import sys


def main() -> None:
    """Parse one positional YAML argument and run the engine."""
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <config.yaml>", file=sys.stderr)
        sys.exit(1)

    config_path = sys.argv[1]

    # Imports deferred — no heavy work at module scope (preflight-pattern.md §6).
    from routines.validation.spatial_residual.engine import (
        SpatialResidualConfig,
        SpatialResidualEngine,
    )

    cfg = SpatialResidualConfig.from_yaml(config_path)
    engine = SpatialResidualEngine(cfg)
    run_dir = engine.run()
    print(run_dir)


if __name__ == "__main__":
    main()
