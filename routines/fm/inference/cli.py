"""CLI entrypoint for the unified validation-inference routine.

Single positional YAML argument per ``preflight-pattern.md`` invariant 1.
Registered as ``vena-fm-inference`` in ``pyproject.toml``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from routines.fm.inference.engine import InferenceEngine, InferenceJobConfig

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Unified validation-inference (VENA + 8 competitors + C0).",
    )
    parser.add_argument("config_path", type=Path, help="path to job YAML")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    cfg = InferenceJobConfig.from_yaml(args.config_path)
    engine = InferenceEngine(cfg)
    out = engine.run()
    print(f"inference complete -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
