"""CLI entrypoint for ``vena-fm-post-train``.

Usage::

    vena-fm-post-train cfg.yaml
    python -m routines.fm.post_train.cli cfg.yaml

The matplotlib ``Agg`` backend is enforced before any pyplot import so the
routine runs cleanly on Picasso (no display, no Qt/GTK dependencies).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

from rich.logging import RichHandler

from .engine import PostTrainEngine, PostTrainRoutineConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vena-fm-post-train",
        description="Render post-training plots (loss/grad + Pareto) into <run_dir>/plots/.",
    )
    parser.add_argument("config", type=Path, help="YAML config file (with `run_dir:` key)")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )

    cfg = PostTrainRoutineConfig.from_yaml(args.config)
    engine = PostTrainEngine(cfg)
    artifact = engine.run()
    print(f"artifact: {artifact}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
