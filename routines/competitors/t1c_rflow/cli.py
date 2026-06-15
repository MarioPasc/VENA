"""CLI entrypoint for ``vena-competitor-t1c-rflow`` — one positional YAML arg.

Citation: Eidex *et al.* 2025, arXiv:2509.24194.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.logging import RichHandler

from .engine import T1CRFlowCompetitorConfig, T1CRFlowCompetitorEngine


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vena-competitor-t1c-rflow",
        description=(
            "T1C-RFlow (Eidex et al., 2025, arXiv:2509.24194) — 3D latent "
            "rectified-flow synthesis of T1c from T1pre + FLAIR latents."
        ),
    )
    parser.add_argument("config", type=Path, help="YAML config file")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )

    cfg = T1CRFlowCompetitorConfig.from_yaml(args.config)
    engine = T1CRFlowCompetitorEngine(cfg, config_yaml_path=args.config)
    artifact = engine.run()
    print(f"artifact: {artifact}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
