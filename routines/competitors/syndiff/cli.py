"""CLI entrypoint for ``vena-competitor-syndiff`` — one positional YAML arg.

Citation: Özbey *et al.* 2023, IEEE TMI, arXiv:2207.08208v3.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.logging import RichHandler

from .engine import SynDiffCompetitorConfig, SynDiffCompetitorEngine


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vena-competitor-syndiff",
        description=(
            "SynDiff (Özbey et al., IEEE TMI 2023, arXiv:2207.08208v3) — "
            "adversarial diffusion for one-to-one MRI contrast translation. "
            "Trained from scratch on VENA's UCSF-PDGM + BraTS-GLI multi-cohort "
            "regime; one source modality → T1c per run."
        ),
    )
    parser.add_argument("config", type=Path, help="YAML config file")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )

    cfg = SynDiffCompetitorConfig.from_yaml(args.config)
    engine = SynDiffCompetitorEngine(cfg, config_yaml_path=args.config)
    artifact = engine.run()
    print(f"artifact: {artifact}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
