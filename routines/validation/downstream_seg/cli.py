"""CLI entry-point for the downstream-seg routine.

Usage::

    vena-validation-downstream-seg configs/smoke.yaml
    python -m routines.validation.downstream_seg.cli configs/smoke.yaml

Takes one positional argument: path to a YAML config.  All flags are read
from the YAML; no additional CLI options are supported (preflight-pattern.md §1).
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """Parse argv[1] as a YAML config path and run the engine."""
    if len(sys.argv) != 2:
        print(
            f"Usage: {sys.argv[0]} <config.yaml>",
            file=sys.stderr,
        )
        sys.exit(1)

    config_path = Path(sys.argv[1])
    if not config_path.is_file():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    # Import heavy dependencies only inside main() — never at module scope
    # (preflight-pattern.md §6).
    from routines.validation.downstream_seg.engine import (
        DownstreamSegConfig,
        DownstreamSegEngine,
    )

    cfg = DownstreamSegConfig.from_yaml(config_path)
    engine = DownstreamSegEngine(cfg=cfg)
    artifact_dir = engine.run()
    print(artifact_dir)


if __name__ == "__main__":
    main()
