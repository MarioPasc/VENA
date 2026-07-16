"""CLI entrypoint for vena-validation-preregister.

Usage::

    vena-validation-preregister path/to/config.yaml
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def main() -> None:
    """Entrypoint: freeze ring partitions from the inference tree."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <config.yaml>", file=sys.stderr)
        sys.exit(1)

    yaml_path = Path(sys.argv[1])
    if not yaml_path.is_file():
        print(f"Config not found: {yaml_path}", file=sys.stderr)
        sys.exit(1)

    # Import inside main so the module is importable without side effects.
    from routines.validation.preregister.engine import PreregisterConfig, PreregisterEngine

    cfg = PreregisterConfig.from_yaml(yaml_path)
    engine = PreregisterEngine(cfg=cfg)
    run_dir = engine.run()
    print(f"Artifact written to: {run_dir}")


if __name__ == "__main__":
    main()
