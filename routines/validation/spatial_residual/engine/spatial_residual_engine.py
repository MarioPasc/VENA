"""Engine for the spatial_residual validation routine (§4.3).

Shardable by (method, cohort, nfe): pass ``methods``/``cohorts``/``nfes``
filters in the YAML to restrict the sweep to one slice, then merge the per_scan
CSVs from each shard before running aggregate_patient_tests.

Design notes
------------
- One YAML arg, frozen dataclass config (preflight-pattern.md §2).
- Engine.run() → Path (run_dir).
- All heavy computation delegates to ``vena.validation.spatial_residual``.
- No GPU dependencies — pure NumPy / SciPy / sklearn.
- References are resolved from the ``references_h5`` attr baked into each
  prediction H5; no separate ``reference_cache`` path is needed in the config.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import yaml

from vena.validation.artifacts import make_run_dir, symlink_latest, write_decision_json
from vena.validation.io import ReferenceCache, build_index, iter_scans
from vena.validation.spatial_residual import (
    SPATIAL_CSV_COLUMNS,
    aggregate_patient_tests,
    compute_scan_rows,
    shuffle_convergence_check,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_SCHEMA_VERSION = "1.0"
_PRODUCER = "routines.validation.spatial_residual:1.0"


@dataclass(frozen=True)
class SpatialResidualConfig:
    """Frozen configuration for the spatial residual routine.

    Parameters
    ----------
    inference_root :
        Root directory containing ``<shard>/predictions/<method>/<cohort>/nfe_*.h5``
        files. Passed to :func:`vena.validation.io.build_index`.
    output_root :
        Where to write the timestamped artifact directory.
    methods :
        Restrict sweep to these method keys.  ``null`` → all found.
    cohorts :
        Restrict sweep to these cohort keys.  ``null`` → all found.
    nfes :
        Restrict sweep to these NFE values.  ``null`` → all found.
    dilate_k :
        WT dilation kernel size for the C-noT region (must be odd; radius = k//2).
    n_shuffles :
        Shuffle-null draws per domain per condition.
    n_boot :
        Per-scan bootstrap draws for the Spearman CI.
    rng_seed :
        Global RNG seed (reproducibility).
    mi_n_voxels :
        Maximum voxels subsampled for KSG MI estimation.
    n_deciles :
        Intensity decile bins for the S3 Bland-Altman panel.
    vena_method :
        Key of the VENA headline method used in aggregate Wilcoxon tests.
    scan_limit :
        If set, stop after processing this many scans (smoke / debug).
    run_convergence_check :
        If True, run shuffle convergence check on the first valid scan.
    log_level :
        Python logging level string (e.g. ``"INFO"``, ``"DEBUG"``).
    """

    inference_root: str
    output_root: str
    methods: list[str] | None = None
    cohorts: list[str] | None = None
    nfes: list[int] | None = None
    dilate_k: int = 5
    n_shuffles: int = 100
    n_boot: int = 100
    rng_seed: int = 42
    mi_n_voxels: int = 30_000
    n_deciles: int = 10
    vena_method: str = "VENA-S1-v3b-rw"
    scan_limit: int | None = None
    run_convergence_check: bool = True
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, path: str | Path) -> SpatialResidualConfig:
        """Load config from a YAML file."""
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        return cls(**raw)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class SpatialResidualEngine:
    """Thin orchestrator wiring io → library → artifact.

    Parameters
    ----------
    cfg :
        Frozen config for this run.
    """

    def __init__(self, cfg: SpatialResidualConfig) -> None:
        self._cfg = cfg

    def run(self) -> Path:
        """Execute the spatial residual analysis and write artifacts.

        Returns
        -------
        Path
            The timestamped run directory.
        """
        cfg = self._cfg
        _setup_logging(cfg.log_level)

        inference_root = Path(cfg.inference_root)
        output_root = Path(cfg.output_root)

        # Build and filter the prediction index.
        logger.info("Building prediction index from %s", inference_root)
        index = build_index(inference_root)
        index = _apply_filters(index, cfg)
        if index.empty:
            raise SpatialResidualError(
                f"No predictions found after filtering in {inference_root}. "
                "Check inference_root, methods, cohorts, nfes filters."
            )
        logger.info("Index: %d prediction files (method × cohort × nfe)", len(index))

        # Create timestamped output directory.
        run_dir = make_run_dir(output_root, "spatial_residual")
        logger.info("Run dir: %s", run_dir)

        # Persist config for reproducibility.
        _persist_config(run_dir, cfg)

        # Shared reference cache — one ReferenceCache amortises repeated reads
        # across 16 methods that share the same cohort reference file.
        ref_cache = ReferenceCache(maxsize=40)

        # Optional shuffle convergence check on the very first scan.
        if cfg.run_convergence_check:
            _run_convergence_check(index, ref_cache, run_dir, cfg)

        # Main scan loop: iterate over every pred file in the filtered index.
        rows: list[dict] = []
        n_empty_region = 0
        n_scans_done = 0

        for _, row in index.iterrows():
            pred_path = Path(row["path"])
            for sample in iter_scans(pred_path, reference_cache=ref_cache):
                scan_rows = compute_scan_rows(
                    sample,
                    dilate_k=cfg.dilate_k,
                    n_shuffles=cfg.n_shuffles,
                    n_boot=cfg.n_boot,
                    rng_seed=cfg.rng_seed,
                    mi_n_voxels=cfg.mi_n_voxels,
                    n_deciles=cfg.n_deciles,
                )
                rows.extend(scan_rows)
                n_scans_done += 1

                for r in scan_rows:
                    if r.get("n_voxels_region", 1) == 0:
                        n_empty_region += 1

                if cfg.scan_limit and n_scans_done >= cfg.scan_limit:
                    logger.info("scan_limit=%d reached; stopping.", cfg.scan_limit)
                    break

            if cfg.scan_limit and n_scans_done >= cfg.scan_limit:
                break

            if n_scans_done % 10 == 0 and n_scans_done > 0:
                logger.info("Processed %d scans …", n_scans_done)

        logger.info(
            "Scan loop done: %d scans, %d rows, %d empty-region NaN rows",
            n_scans_done,
            len(rows),
            n_empty_region,
        )

        if not rows:
            raise SpatialResidualError(
                "No rows produced. Check prediction file schema and reference resolution."
            )

        # Write per-scan CSV with frozen 40-column header.
        per_scan_df = pd.DataFrame(rows, columns=SPATIAL_CSV_COLUMNS)
        per_scan_csv = run_dir / "tables" / "per_scan.csv"
        per_scan_df.to_csv(per_scan_csv, index=False)
        logger.info("per_scan.csv: %d rows → %s", len(per_scan_df), per_scan_csv)

        # Aggregate tests — non-blocking; a partial sweep (not all 8 competitors)
        # triggers a family-size WARNING inside aggregate_patient_tests.
        test_results: list = []
        try:
            patient_df, test_results = aggregate_patient_tests(
                per_scan_df,
                vena_method=cfg.vena_method,
                condition="C-noT",
                ring="A",
            )
            (run_dir / "tables" / "patient_stats.csv").write_text(patient_df.to_csv(index=False))
            _write_test_results(test_results, run_dir / "tables" / "wilcoxon_results.csv")
            logger.info("Wrote patient stats and Wilcoxon results.")
        except Exception as exc:
            logger.warning("Aggregate tests failed (non-blocking): %s", exc)

        # Write decision.json and LATEST symlink.
        decision = {
            "schema_version": _SCHEMA_VERSION,
            "produced_at": datetime.now(UTC).isoformat(),
            "producer": _PRODUCER,
            "git_sha": _git_sha(),
            "inference_root": str(inference_root),
            "n_pred_files": len(index),
            "n_scans": n_scans_done,
            "n_rows": len(rows),
            "n_empty_region": n_empty_region,
            "dilate_k": cfg.dilate_k,
            "n_shuffles": cfg.n_shuffles,
            "n_boot": cfg.n_boot,
            "rng_seed": cfg.rng_seed,
            "mi_n_voxels": cfg.mi_n_voxels,
            "vena_method": cfg.vena_method,
            "n_wilcoxon_tests": len(test_results),
        }
        write_decision_json(run_dir, decision)
        symlink_latest(run_dir)
        logger.info("Done → %s", run_dir)
        return run_dir


# ---------------------------------------------------------------------------
# Module-level helpers (not nested in loops — coding-standards.md §16)
# ---------------------------------------------------------------------------


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _apply_filters(index: pd.DataFrame, cfg: SpatialResidualConfig) -> pd.DataFrame:
    """Return a filtered copy of the prediction index."""
    df = index.copy()
    if cfg.methods:
        df = df[df["method"].isin(cfg.methods)]
    if cfg.cohorts:
        df = df[df["cohort"].isin(cfg.cohorts)]
    if cfg.nfes:
        df = df[df["nfe"].isin(cfg.nfes)]
    return df.reset_index(drop=True)


def _persist_config(run_dir: Path, cfg: SpatialResidualConfig) -> None:
    """Write config.yaml into the run dir for reproducibility."""
    payload = {k: v for k, v in cfg.__dict__.items() if not k.startswith("_")}
    with open(run_dir / "config.yaml", "w") as fh:
        yaml.dump(payload, fh, default_flow_style=False)


def _run_convergence_check(
    index: pd.DataFrame,
    ref_cache: ReferenceCache,
    run_dir: Path,
    cfg: SpatialResidualConfig,
) -> None:
    """Probe shuffle convergence on the first available scan; non-blocking."""
    from vena.validation.regions import region_masks

    logger.info("Running shuffle convergence check on first scan …")
    try:
        first_path = Path(index.iloc[0]["path"])
        samples = list(iter_scans(first_path, reference_cache=ref_cache))
        if not samples:
            logger.warning("No scan loaded for convergence check — skipping.")
            return

        s = samples[0]
        masks = region_masks(s.brain.astype(bool), s.wt.astype(bool), dilate_k=cfg.dilate_k)

        abs_resid_vol = np.abs(s.real.astype(np.float64) - s.pred.astype(np.float64))
        t1c_vol = s.real.astype(np.float64)

        brain_flat = masks["brain"].ravel()
        brain_indices = np.where(brain_flat)[0]
        bg_flat = masks["bg"].ravel()
        # bg gives C-noT region; region_in_brain is a bool mask into brain voxels.
        region_in_brain = bg_flat[brain_flat]

        abs_resid_brain = abs_resid_vol.ravel()[brain_indices]
        t1c_brain = t1c_vol.ravel()[brain_indices]
        abs_resid_R = abs_resid_brain[region_in_brain]  # noqa: N806
        t1c_R = t1c_brain[region_in_brain]  # noqa: N806

        conv = shuffle_convergence_check(
            abs_resid_R,
            t1c_R,
            n_list=(10, 50, 100, 500),
            q=0.05,
            rng_seed=cfg.rng_seed,
        )
        conv_path = run_dir / "tables" / "shuffle_convergence.json"
        with open(conv_path, "w") as fh:
            json.dump({str(k): v for k, v in conv.items()}, fh, indent=2)
        logger.info("Shuffle convergence → %s", conv_path)
    except Exception as exc:
        logger.warning("Convergence check failed (non-blocking): %s", exc)


def _write_test_results(test_results: list, path: Path) -> None:
    from dataclasses import asdict

    pd.DataFrame([asdict(r) for r in test_results]).to_csv(path, index=False)


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class SpatialResidualError(Exception):
    """Raised by the spatial residual engine on unrecoverable errors."""
