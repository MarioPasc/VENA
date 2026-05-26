"""Entrypoint for the venous-atlas build routine (in-house dural-sinus mask).

Usage
-----
    vena-preflight-venous-atlas-build <config.yaml>
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml
from rich.logging import RichHandler

from routines.preflights.venous_atlas_build.engine.venous_atlas_build_engine import (
    VenousAtlasBuildRoutineEngine,
)
from vena.preflight.priors_validation.atlases.venous_build import VenousBuildConfig


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )


def _config_from_yaml(path: Path) -> VenousBuildConfig:
    raw = yaml.safe_load(path.read_text()) or {}
    slab = raw.get("axial_slab_z_mni") or [25.0, 80.0]
    return VenousBuildConfig(
        dataset_root=Path(raw["dataset_root"]),
        metadata_csv=Path(raw["metadata_csv"]) if raw.get("metadata_csv") else None,
        output_root=Path(raw["output_root"]),
        atlases_root=Path(raw["atlases_root"]),
        cache_root=Path(raw["cache_root"]),
        n_subjects=int(raw.get("n_subjects", 30)),
        who_grade_max=raw.get("who_grade_max"),
        midline_shift_mm_max=float(raw.get("midline_shift_mm_max", 3.0)),
        intensity_percentile=float(raw.get("intensity_percentile", 95.0)),
        axial_slab_z_mni=(float(slab[0]), float(slab[1])),
        voting_threshold=float(raw.get("voting_threshold", 0.5)),
        min_component_size=int(raw.get("min_component_size", 100)),
        seed=int(raw.get("seed", 1337)),
        n_workers=int(raw.get("n_workers", 1)),
        log_level=str(raw.get("log_level", "INFO")).upper(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vena-preflight-venous-atlas-build")
    parser.add_argument("config", type=Path, help="Path to YAML routine config")
    args = parser.parse_args(argv)
    cfg = _config_from_yaml(args.config)
    _configure_logging(cfg.log_level)
    res = VenousAtlasBuildRoutineEngine(cfg).run()
    logging.getLogger(__name__).info(
        "Venous atlas built — mask volume %.1f ml from %d subjects",
        res["mask_volume_ml"],
        res["n_subjects_contributing"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
