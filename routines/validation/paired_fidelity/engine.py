"""Paired-fidelity routine engine — §4.2 + §4.5 + §4.7.

Thin orchestrator: builds index, streams scan pairs, delegates all metric
computation to :mod:`vena.validation.metrics_paired`, writes artifact folder
per SHARED_CONTRACTS §9.

Shardable: ``filter_methods`` / ``filter_cohorts`` / ``filter_nfe`` in the
YAML narrow the index so the orchestrator can fan out across cpu_partition jobs
and merge per-scan CSVs afterwards (task spec §7 note).
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")

from vena.validation.artifacts import (
    make_run_dir,
    symlink_latest,
    write_decision_json,
    write_per_scan_csv,
)
from vena.validation.io import ReferenceCache, build_index, discover_shards, iter_scans
from vena.validation.metrics_paired import (
    MetricConfig,
    compute_paired_metrics,
)
from vena.validation.plotting import (
    annotate_significance,
    method_order,
    method_palette,
    setup_figure_style,
)
from vena.validation.registry import (
    ABLATION_FAMILY,
    COMPETITOR_FAMILY,
    SELECTION_NFE,
    VENA_HEADLINE,
)
from vena.validation.stats import (
    MCID,
    bootstrap_ci,
    cliffs_delta,
    collapse_to_patient,
    holm_bonferroni,
    paired_wilcoxon,
)

logger = logging.getLogger(__name__)

# Guard: Holm family sizes are pre-registered; a silent resize is a bug.
assert len(COMPETITOR_FAMILY) == 8, (
    f"COMPETITOR_FAMILY must have exactly 8 members; got {len(COMPETITOR_FAMILY)}"
)
assert len(ABLATION_FAMILY) == 3, (
    f"ABLATION_FAMILY must have exactly 3 members; got {len(ABLATION_FAMILY)}"
)

# Metric columns that enter the statistical pass
_METRIC_COLS: list[str] = [
    "mae_brain",
    "mae_wt",
    "mae_bg_undilated",
    "rmse_brain",
    "rmse_wt",
    "rmse_bg_undilated",
    "psnr_brain",
    "psnr_wt",
    "psnr_bg_undilated",
    "ssim_brain",
    "ssim_wt",
    "ssim_bg_undilated",
    "ms_ssim_brain",
    "ms_ssim_wt",
    "ms_ssim_bg_undilated",
    "zgd",
    "inference_seconds",
    "peak_vram_mb",
    "n_brain_voxels",
    "n_wt_voxels",
    "n_bg_undilated_voxels",
    # §4.1 scoring-space audit — float, collapsed (mean) at patient level
    "raw_p995",
]

# Columns for headline statistics
_PRIMARY_METRIC = "mae_brain"
_STAT_METRICS: list[str] = [
    "mae_brain",
    "mae_wt",
    "mae_bg_undilated",
    "ssim_brain",
    "ssim_wt",
    "ssim_bg_undilated",
    "psnr_brain",
    "zgd",
]


class PairedFidelityError(Exception):
    """Raised when the engine cannot proceed."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairedFidelityConfig:
    """Frozen config parsed from the routine YAML.

    Parameters
    ----------
    data_root :
        Inference tree root (contains shard directories with ``decision.json``).
    output_root :
        Parent of the ``paired_fidelity/`` artifact directory.
    dilate_k :
        WT dilation kernel size (YAML-configurable, auditable).
    ssim_window_size :
        SSIM Gaussian window size (must match across all runs).
    ssim_window_sigma :
        SSIM Gaussian sigma.
    ms_ssim_weights :
        4-level MS-SSIM weights (Wang 2003).
    ms_ssim_bbox_margin :
        Extra voxels added to WT bbox for MS-SSIM computation.
    n_bootstrap :
        Bootstrap replicates for 95% CI.
    bootstrap_seed :
        Fixed seed for reproducibility.
    device :
        Torch device for SSIM convolutions (default ``"cpu"``).
    filter_methods :
        Whitelist of methods to process; empty = all.
    filter_cohorts :
        Whitelist of cohorts to process; empty = all.
    filter_nfe :
        Whitelist of NFE values to process; empty = all.
    filter_rings :
        Rings to process (default: ``["A", "B"]``).
    """

    data_root: Path
    output_root: Path
    dilate_k: int = 5
    ssim_window_size: int = 7
    ssim_window_sigma: float = 1.5
    ms_ssim_weights: tuple[float, ...] = (0.0448, 0.2856, 0.3001, 0.3633)
    ms_ssim_bbox_margin: int = 8
    n_bootstrap: int = 10_000
    bootstrap_seed: int = 1337
    device: str = "cpu"
    filter_methods: tuple[str, ...] = field(default_factory=tuple)  # type: ignore[assignment]
    filter_cohorts: tuple[str, ...] = field(default_factory=tuple)  # type: ignore[assignment]
    filter_nfe: tuple[int, ...] = field(default_factory=tuple)  # type: ignore[assignment]
    filter_rings: tuple[str, ...] = ("A", "B")

    @classmethod
    def from_yaml(cls, path: Path) -> PairedFidelityConfig:
        """Parse config from YAML.

        Parameters
        ----------
        path :
            Path to the YAML config file.

        Raises
        ------
        PairedFidelityError
            When required keys are missing or the YAML is malformed.
        """
        path = Path(path)
        if not path.is_file():
            raise PairedFidelityError(f"Config not found: {path}")
        with path.open() as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}

        try:
            data_root = Path(raw["data_root"])
            output_root = Path(raw["output_root"])
        except KeyError as exc:
            raise PairedFidelityError(f"Missing required key in YAML: {exc}") from exc

        return cls(
            data_root=data_root,
            output_root=output_root,
            dilate_k=int(raw.get("dilate_k", 5)),
            ssim_window_size=int(raw.get("ssim_window_size", 7)),
            ssim_window_sigma=float(raw.get("ssim_window_sigma", 1.5)),
            ms_ssim_weights=tuple(
                float(w) for w in raw.get("ms_ssim_weights", [0.0448, 0.2856, 0.3001, 0.3633])
            ),
            ms_ssim_bbox_margin=int(raw.get("ms_ssim_bbox_margin", 8)),
            n_bootstrap=int(raw.get("n_bootstrap", 10_000)),
            bootstrap_seed=int(raw.get("bootstrap_seed", 1337)),
            device=str(raw.get("device", "cpu")),
            filter_methods=tuple(raw.get("filter_methods") or []),
            filter_cohorts=tuple(raw.get("filter_cohorts") or []),
            filter_nfe=tuple(int(x) for x in (raw.get("filter_nfe") or [])),
            filter_rings=tuple(raw.get("filter_rings") or ["A", "B"]),
        )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class PairedFidelityEngine:
    """Paired-fidelity routine engine.

    Parameters
    ----------
    cfg :
        Frozen config parsed from YAML.
    """

    def __init__(self, cfg: PairedFidelityConfig) -> None:
        self.cfg = cfg

    def run(self) -> Path:
        """Execute the routine and return the artifact directory.

        Returns
        -------
        Path
            The timestamped run directory.
        """
        t_start = time.perf_counter()
        cfg = self.cfg

        # ---- Setup ----
        run_dir = make_run_dir(cfg.output_root, "paired_fidelity")
        logger.info("Run directory: %s", run_dir)

        metric_cfg = MetricConfig(
            data_range=1.0,  # fixed — SHARED_CONTRACTS §11 trap 6
            ssim_window_size=cfg.ssim_window_size,
            ssim_window_sigma=cfg.ssim_window_sigma,
            ms_ssim_weights=cfg.ms_ssim_weights,
            ms_ssim_bbox_margin=cfg.ms_ssim_bbox_margin,
            dilate_k=cfg.dilate_k,
        )

        # ---- Discover shards (smoke shard filtering + audit trail) ----
        logger.info("Discovering shards under %s …", cfg.data_root)
        shard_discovery = discover_shards(cfg.data_root)
        logger.info(
            "Accepted %d shards; skipped %d smoke shard(s): %s",
            len(shard_discovery.accepted),
            len(shard_discovery.skipped_smoke),
            shard_discovery.skipped_smoke or "none",
        )

        # ---- Build index ----
        logger.info("Building index from %s …", cfg.data_root)
        index = build_index(cfg.data_root)
        if index.empty:
            raise PairedFidelityError(f"No prediction H5 files found under {cfg.data_root}")

        index = self._apply_filters(index)
        logger.info("Processing %d prediction files …", len(index))

        # ---- Stream all scans ----
        ref_cache = ReferenceCache()
        rows: list[dict[str, object]] = []
        n_files = 0
        n_scans = 0

        for _, file_row in index.iterrows():
            pred_path = Path(file_row["path"])
            n_files += 1
            logger.debug("  %s", pred_path.name)
            try:
                for scan in iter_scans(pred_path, reference_cache=ref_cache):
                    m = compute_paired_metrics(scan, metric_cfg)
                    rows.append(m.to_flat_dict())
                    n_scans += 1
            except Exception as exc:
                logger.warning("Failed on %s: %s", pred_path, exc)

        logger.info("Processed %d scans from %d files", n_scans, n_files)

        if not rows:
            raise PairedFidelityError("No scans processed — check filters and data paths.")

        per_scan_df = pd.DataFrame(rows)
        elapsed = time.perf_counter() - t_start

        return self.run_postprocess(
            run_dir,
            per_scan_df=per_scan_df,
            n_files=n_files,
            n_scans=n_scans,
            elapsed_s=elapsed,
            skipped_smoke_shards=shard_discovery.skipped_smoke,
        )

    def run_postprocess(
        self,
        run_dir: Path,
        *,
        per_scan_df: pd.DataFrame,
        n_files: int,
        n_scans: int,
        elapsed_s: float,
        skipped_smoke_shards: list[str],
    ) -> Path:
        """Run the analysis phase from a pre-collected per-scan DataFrame.

        Called by :meth:`run` (smoke / single-node) and by ``cli_merge`` after
        concatenating sweep shards.  The collapse and Holm correction run
        exactly once here — never per shard.

        Parameters
        ----------
        run_dir :
            Destination artifact directory (already created by the caller).
        per_scan_df :
            One row per (scan × method × cohort × nfe) with all metric columns.
        n_files :
            Number of prediction files (or shards) consumed upstream.
        n_scans :
            Total scan rows before patient collapse.
        elapsed_s :
            Wall-clock seconds for the upstream work (scan loop or shard concat).
        skipped_smoke_shards :
            Shard names excluded by ``discover_shards`` (passed through to the
            decision payload for auditability).

        Returns
        -------
        Path
            The artifact directory.
        """
        if per_scan_df.empty:
            raise PairedFidelityError("per_scan_df is empty — nothing to analyse.")

        # ---- Enforce frozen column order (no white cells) ----
        id_cols = ["method", "cohort", "ring", "nfe", "scan_id", "patient_id", "pred_mode"]
        per_scan_df = per_scan_df[[c for c in id_cols + _METRIC_COLS if c in per_scan_df.columns]]

        # ---- Scoring-space audit: per-method mode counts ----
        if "pred_mode" in per_scan_df.columns:
            _mode_pivot = (
                per_scan_df.groupby("method")["pred_mode"].value_counts().unstack(fill_value=0)
            )
            pred_mode_counts: dict[str, dict[str, int]] = {
                method: row.to_dict() for method, row in _mode_pivot.iterrows()
            }
        else:
            pred_mode_counts = {}

        write_per_scan_csv(run_dir, per_scan_df, name="paired_fidelity.csv")
        logger.info("Wrote per_scan/paired_fidelity.csv (%d rows)", len(per_scan_df))

        # ---- Patient-level collapse (ONCE — not per shard) ----
        value_cols = [c for c in _METRIC_COLS if c in per_scan_df.columns]
        per_patient_df = collapse_to_patient(
            per_scan_df,
            value_cols,
            by=("method", "cohort", "ring", "nfe", "patient_id"),
        )
        write_per_scan_csv(run_dir, per_patient_df, name="paired_fidelity_patient.csv")
        logger.info(
            "Wrote per_scan/paired_fidelity_patient.csv (%d patient-level rows)",
            len(per_patient_df),
        )

        # ---- LUMIERE collapse sanity check ----
        self._check_lumiere_collapse(per_scan_df, per_patient_df)

        # ---- Statistical pass — Ring A, Holm correction (ONCE) ----
        ring_a_pt = per_patient_df[per_patient_df["ring"] == "A"].copy()
        holm_tables = self._statistical_pass(ring_a_pt, run_dir)

        # ---- C0-Identity sanity check ----
        c0_results = self._c0_sanity(ring_a_pt, run_dir)

        # ---- Cost table (§4.5) ----
        self._write_cost_table(per_patient_df, run_dir)

        # ---- ZGD table (§4.7) ----
        self._write_zgd_table(per_patient_df, run_dir)

        # ---- Figures ----
        self._make_figures(run_dir, per_scan_df, per_patient_df, holm_tables, c0_results)

        n_patients = len(per_patient_df["patient_id"].unique())

        # ---- decision.json ----
        self._write_decision(
            run_dir,
            n_files=n_files,
            n_scans=n_scans,
            n_patients=n_patients,
            elapsed_s=elapsed_s,
            c0_results=c0_results,
            pred_mode_counts=pred_mode_counts,
            skipped_smoke_shards=skipped_smoke_shards,
        )

        self._write_report(
            run_dir,
            n_scans=n_scans,
            n_patients=n_patients,
            elapsed_s=elapsed_s,
            c0_results=c0_results,
        )
        symlink_latest(run_dir)
        logger.info("Done in %.1f s — artifact: %s", elapsed_s, run_dir)
        return run_dir

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _apply_filters(self, index: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        if cfg.filter_methods:
            index = index[index["method"].isin(cfg.filter_methods)]
        if cfg.filter_cohorts:
            index = index[index["cohort"].isin(cfg.filter_cohorts)]
        if cfg.filter_nfe:
            index = index[index["nfe"].isin(cfg.filter_nfe)]
        if cfg.filter_rings:
            index = index[index["ring"].isin(cfg.filter_rings)]
        return index.reset_index(drop=True)

    # ------------------------------------------------------------------
    # LUMIERE sanity check
    # ------------------------------------------------------------------

    def _check_lumiere_collapse(
        self, per_scan_df: pd.DataFrame, per_patient_df: pd.DataFrame
    ) -> None:
        """Assert LUMIERE collapses 72 scans → 11 patients (task spec §6)."""
        lum_scans = per_scan_df[per_scan_df["cohort"] == "LUMIERE"]
        if lum_scans.empty:
            logger.debug("LUMIERE not in index — skipping collapse check")
            return
        # Per method at any NFE
        lum_pt = per_patient_df[per_patient_df["cohort"] == "LUMIERE"]
        first_method = lum_scans["method"].iloc[0]
        first_nfe = lum_scans["nfe"].iloc[0]
        n_scans = len(
            lum_scans[(lum_scans["method"] == first_method) & (lum_scans["nfe"] == first_nfe)]
        )
        n_patients = len(lum_pt[(lum_pt["method"] == first_method) & (lum_pt["nfe"] == first_nfe)])
        logger.info(
            "LUMIERE collapse check: %d scans → %d patients (method=%s, nfe=%d)",
            n_scans,
            n_patients,
            first_method,
            first_nfe,
        )
        if n_scans > 1 and n_patients >= n_scans:
            logger.warning(
                "LUMIERE did not collapse to fewer patients than scans — "
                "patient_id collapse may be broken (n_scans=%d, n_patients=%d)",
                n_scans,
                n_patients,
            )

    # ------------------------------------------------------------------
    # Statistical pass — Ring A, per family
    # ------------------------------------------------------------------

    def _statistical_pass(
        self,
        ring_a_pt: pd.DataFrame,
        run_dir: Path,
    ) -> dict[str, dict[str, dict[str, object]]]:
        """Run Holm-corrected Wilcoxon tests for competitor and ablation families.

        Returns
        -------
        dict
            ``{family_name → {metric_col → {competitor → HolmResult}}}``.
        """
        if ring_a_pt.empty:
            logger.warning("No Ring-A patient data — skipping statistical pass.")
            return {}

        vena_pt = ring_a_pt[ring_a_pt["method"] == VENA_HEADLINE]
        if vena_pt.empty:
            logger.warning("%s not found in Ring-A data — skipping stats.", VENA_HEADLINE)
            return {}

        results: dict[str, dict[str, dict[str, object]]] = {}
        headline_rows: list[dict[str, object]] = []

        for family_name, family_members in [
            ("competitor", COMPETITOR_FAMILY),
            ("ablation", ABLATION_FAMILY),
        ]:
            results[family_name] = {}
            for metric in _STAT_METRICS:
                if metric not in ring_a_pt.columns:
                    continue
                vena_series = (
                    vena_pt.groupby("patient_id")[metric].mean()
                    if "patient_id" in vena_pt.columns
                    else vena_pt.set_index("patient_id")[metric]
                )
                wilcoxon_by_comp: dict[str, object] = {}
                for comp in family_members:
                    comp_pt = ring_a_pt[ring_a_pt["method"] == comp]
                    if comp_pt.empty:
                        continue
                    # Use selection NFE for the competitor
                    sel_nfe = SELECTION_NFE.get(comp, 1)
                    comp_at_nfe = comp_pt[comp_pt["nfe"] == sel_nfe]
                    if comp_at_nfe.empty:
                        # Fall back to any available NFE
                        comp_at_nfe = comp_pt
                    comp_series = comp_at_nfe.groupby("patient_id")[metric].mean()
                    try:
                        wilcoxon_by_comp[comp] = paired_wilcoxon(vena_series, comp_series)
                    except Exception as exc:
                        logger.debug("paired_wilcoxon failed (%s, %s): %s", metric, comp, exc)

                if not wilcoxon_by_comp:
                    continue

                # Holm-Bonferroni correction over the family
                pvalues = {c: r.pvalue for c, r in wilcoxon_by_comp.items()}  # type: ignore[union-attr]
                holm = holm_bonferroni(pvalues)
                results[family_name][metric] = {}

                for comp, wx in wilcoxon_by_comp.items():
                    from vena.validation.stats import HolmResult, WilcoxonResult

                    assert isinstance(wx, WilcoxonResult)
                    holm_r: HolmResult = holm[comp]
                    comp_pt2 = ring_a_pt[ring_a_pt["method"] == comp]
                    sel_nfe = SELECTION_NFE.get(comp, 1)
                    comp_at_nfe2 = comp_pt2[comp_pt2["nfe"] == sel_nfe]
                    if comp_at_nfe2.empty:
                        comp_at_nfe2 = comp_pt2
                    comp_vals = comp_at_nfe2.groupby("patient_id")[metric].mean().values
                    vena_vals = vena_series.reindex(
                        comp_at_nfe2.groupby("patient_id")[metric].mean().index
                    ).values
                    delta = cliffs_delta(vena_vals, comp_vals)
                    lo, hi = bootstrap_ci(
                        vena_vals - comp_vals,
                        n_boot=self.cfg.n_bootstrap,
                        seed=self.cfg.bootstrap_seed,
                    )
                    results[family_name][metric][comp] = {
                        "wilcoxon": wx,
                        "holm": holm_r,
                        "cliffs_delta": delta,
                        "ci_lo": lo,
                        "ci_hi": hi,
                    }
                    mcid_flag = abs(np.nanmean(vena_vals - comp_vals)) >= MCID
                    headline_rows.append(
                        {
                            "family": family_name,
                            "metric": metric,
                            "competitor": comp,
                            "vena_mean": float(np.nanmean(vena_vals)),
                            "comp_mean": float(np.nanmean(comp_vals)),
                            "diff_mean": float(np.nanmean(vena_vals - comp_vals)),
                            "ci_lo": lo,
                            "ci_hi": hi,
                            "pvalue_raw": wx.pvalue,
                            "pvalue_holm": holm_r.pvalue_adj,
                            "reject_h0": holm_r.reject,
                            "cliffs_delta": delta,
                            "mcid_flag": mcid_flag,
                            "n_patients": wx.n,
                        }
                    )

        if headline_rows:
            headline_df = pd.DataFrame(headline_rows)
            out = run_dir / "tables" / "headline_table.csv"
            headline_df.to_csv(out, index=False)
            logger.info("Wrote tables/headline_table.csv (%d rows)", len(headline_df))

        return results

    # ------------------------------------------------------------------
    # C0-Identity sanity check
    # ------------------------------------------------------------------

    def _c0_sanity(
        self,
        ring_a_pt: pd.DataFrame,
        run_dir: Path,
    ) -> dict[str, dict[str, float]]:
        """Verify C0-Identity is the null floor inside WT.

        Every real method must beat C0-Identity on wt-MAE.
        If any does not, report it — the metric may be wrong.

        Returns
        -------
        dict
            ``{method → {"mae_wt": float, "mae_brain": float}}``
            for C0 and VENA_HEADLINE at their selection NFE.
        """
        out: dict[str, dict[str, float]] = {}

        c0_nfe = SELECTION_NFE.get("C0-Identity", 1)
        c0_pt = ring_a_pt[(ring_a_pt["method"] == "C0-Identity") & (ring_a_pt["nfe"] == c0_nfe)]
        if c0_pt.empty:
            logger.warning("C0-Identity not found in Ring-A data — skipping sanity check.")
            return out

        for method in [VENA_HEADLINE, *list(COMPETITOR_FAMILY)]:
            if method == "C0-Identity":
                continue
            sel_nfe = SELECTION_NFE.get(method, 1)
            m_pt = ring_a_pt[(ring_a_pt["method"] == method) & (ring_a_pt["nfe"] == sel_nfe)]
            if m_pt.empty:
                continue
            out[method] = {
                "mae_wt": float(m_pt["mae_wt"].mean()),
                "mae_brain": float(m_pt["mae_brain"].mean()),
            }

        c0_mae_wt = float(c0_pt["mae_wt"].mean())
        c0_mae_brain = float(c0_pt["mae_brain"].mean())
        out["C0-Identity"] = {"mae_wt": c0_mae_wt, "mae_brain": c0_mae_brain}

        # Log violations
        for m, v in out.items():
            if m == "C0-Identity":
                continue
            if not np.isnan(v["mae_wt"]) and v["mae_wt"] >= c0_mae_wt:
                logger.warning(
                    "C0-SANITY VIOLATION: %s WT-MAE (%.4f) >= C0 WT-MAE (%.4f) "
                    "— metric may be wrong, not the model.",
                    m,
                    v["mae_wt"],
                    c0_mae_wt,
                )

        logger.info(
            "C0-Identity sanity: C0 brain-MAE=%.4f wt-MAE=%.4f; VENA brain-MAE=%.4f wt-MAE=%.4f",
            c0_mae_brain,
            c0_mae_wt,
            out.get(VENA_HEADLINE, {}).get("mae_brain", float("nan")),
            out.get(VENA_HEADLINE, {}).get("mae_wt", float("nan")),
        )

        # Write C0 sanity table
        c0_rows = [{"method": m, **v} for m, v in out.items()]
        c0_df = pd.DataFrame(c0_rows)
        c0_df.to_csv(run_dir / "tables" / "c0_sanity.csv", index=False)
        logger.info("Wrote tables/c0_sanity.csv")
        return out

    # ------------------------------------------------------------------
    # Cost table (§4.5)
    # ------------------------------------------------------------------

    def _write_cost_table(self, per_patient_df: pd.DataFrame, run_dir: Path) -> None:
        """Aggregated §4.5 inference cost table (Ring A, selection NFE)."""
        if "inference_seconds" not in per_patient_df.columns:
            return

        ring_a = per_patient_df[per_patient_df["ring"] == "A"]
        rows: list[dict[str, object]] = []
        for method in ring_a["method"].unique():
            sel_nfe = SELECTION_NFE.get(str(method), 1)
            sub = ring_a[(ring_a["method"] == method) & (ring_a["nfe"] == sel_nfe)]
            if sub.empty:
                sub = ring_a[ring_a["method"] == method]
            rows.append(
                {
                    "method": method,
                    "nfe": sel_nfe,
                    "inference_seconds_mean": float(sub["inference_seconds"].mean()),
                    "inference_seconds_std": float(sub["inference_seconds"].std()),
                    "peak_vram_mb_mean": float(sub["peak_vram_mb"].mean()),
                    "n_scans": len(sub),
                }
            )
        if rows:
            df = pd.DataFrame(rows).sort_values("inference_seconds_mean")
            df.to_csv(run_dir / "tables" / "cost_table.csv", index=False)
            logger.info("Wrote tables/cost_table.csv")

    # ------------------------------------------------------------------
    # ZGD table (§4.7)
    # ------------------------------------------------------------------

    def _write_zgd_table(self, per_patient_df: pd.DataFrame, run_dir: Path) -> None:
        """Aggregated §4.7 ZGD table."""
        if "zgd" not in per_patient_df.columns:
            return

        ring_a = per_patient_df[per_patient_df["ring"] == "A"]
        rows: list[dict[str, object]] = []
        for method in ring_a["method"].unique():
            sel_nfe = SELECTION_NFE.get(str(method), 1)
            sub = ring_a[(ring_a["method"] == method) & (ring_a["nfe"] == sel_nfe)]
            if sub.empty:
                sub = ring_a[ring_a["method"] == method]
            valid_zgd = sub["zgd"].dropna()
            rows.append(
                {
                    "method": method,
                    "nfe": sel_nfe,
                    "zgd_mean": float(valid_zgd.mean()) if len(valid_zgd) else float("nan"),
                    "zgd_std": float(valid_zgd.std()) if len(valid_zgd) else float("nan"),
                    "n": len(valid_zgd),
                }
            )
        if rows:
            df = pd.DataFrame(rows).sort_values("zgd_mean")
            df.to_csv(run_dir / "tables" / "zgd_table.csv", index=False)
            logger.info("Wrote tables/zgd_table.csv")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    def _make_figures(
        self,
        run_dir: Path,
        per_scan_df: pd.DataFrame,
        per_patient_df: pd.DataFrame,
        holm_tables: dict[str, dict[str, dict[str, object]]],
        c0_results: dict[str, dict[str, float]],
    ) -> None:
        """Generate all figures per task spec §5."""
        setup_figure_style()
        palette = method_palette()
        order = method_order()
        ring_a_pt = per_patient_df[per_patient_df["ring"] == "A"]

        # 1. Primary: MAE-on-brain, Ring A, patient distribution
        self._fig_primary_mae(run_dir, ring_a_pt, holm_tables, palette, order, c0_results)

        # 2. Region grid
        self._fig_region_grid(run_dir, ring_a_pt, palette, order)

        # 3. Cost-quality Pareto
        self._fig_cost_pareto(run_dir, ring_a_pt, per_patient_df, palette)

        # 4. ZGD per method
        self._fig_zgd(run_dir, ring_a_pt, palette, order)

    def _fig_primary_mae(
        self,
        run_dir: Path,
        ring_a_pt: pd.DataFrame,
        holm_tables: dict[str, dict[str, dict[str, object]]],
        palette: dict[str, str],
        order: list[str],
        c0_results: dict[str, dict[str, float]],
    ) -> None:
        """Box+violin+points of brain MAE, Ring A, at selection NFE."""
        metric = _PRIMARY_METRIC
        if ring_a_pt.empty or metric not in ring_a_pt.columns:
            return

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.set_facecolor("black")
        fig.patch.set_facecolor("black")

        # Filter to selection NFE per method
        rows: list[pd.DataFrame] = []
        for method in order:
            sel_nfe = SELECTION_NFE.get(method, 1)
            sub = ring_a_pt[(ring_a_pt["method"] == method) & (ring_a_pt["nfe"] == sel_nfe)]
            if sub.empty:
                sub = ring_a_pt[ring_a_pt["method"] == method]
            if not sub.empty:
                rows.append(sub.assign(_method=method))
        if not rows:
            plt.close(fig)
            return

        df = pd.concat(rows, ignore_index=True)
        methods_present = [m for m in order if m in df["_method"].values]
        x_pos = {m: i for i, m in enumerate(methods_present)}

        for i, method in enumerate(methods_present):
            vals = df[df["_method"] == method][metric].dropna().values
            if len(vals) == 0:
                continue
            color = palette.get(method, "#888888")
            # Violin
            vp = ax.violinplot([vals], positions=[i], widths=0.6, showmedians=False)
            for body in vp["bodies"]:
                body.set_facecolor(color)
                body.set_alpha(0.4)
            for key in ("cbars", "cmins", "cmaxes"):
                if key in vp:
                    vp[key].set_color(color)
                    vp[key].set_alpha(0.6)
            # Box
            bp = ax.boxplot(
                [vals],
                positions=[i],
                widths=0.3,
                patch_artist=True,
                medianprops={"color": "white", "linewidth": 2},
                boxprops={"facecolor": color, "alpha": 0.7},
                whiskerprops={"color": color},
                capprops={"color": color},
                flierprops={"marker": ".", "color": color, "alpha": 0.4},
            )
            _ = bp  # suppress unused
            # Points
            jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=8, alpha=0.5, color=color, zorder=5)

        # C0 floor line
        c0_brain = c0_results.get("C0-Identity", {}).get("mae_brain")
        if c0_brain and not np.isnan(c0_brain):
            ax.axhline(c0_brain, color="#ff4444", linewidth=1.0, linestyle="--", alpha=0.8)
            ax.text(
                len(methods_present) - 0.5,
                c0_brain,
                "C0-Identity floor",
                color="#ff4444",
                fontsize=6,
                va="bottom",
            )

        # Significance brackets (competitor family)
        comp_holm = {}
        if holm_tables.get("competitor", {}).get(metric):
            for comp, res_dict in holm_tables["competitor"][metric].items():
                if isinstance(res_dict, dict) and "holm" in res_dict:
                    comp_holm[comp] = res_dict["holm"]
        if comp_holm:
            annotate_significance(
                ax,
                pairs=[(VENA_HEADLINE, c) for c in comp_holm],
                holm_results=comp_holm,
                x_positions={m: float(x_pos.get(m, 0)) for m in comp_holm},
                y_position=None,
            )

        ax.set_xticks(range(len(methods_present)))
        ax.set_xticklabels(methods_present, rotation=45, ha="right", fontsize=7, color="white")
        ax.tick_params(colors="white")
        ax.set_ylabel("MAE (brain, Ring A)", color="white", fontsize=9)
        ax.set_title(
            f"Primary endpoint: brain MAE — Ring A — selection NFE\n"
            f"Holm-Bonferroni correction over {len(COMPETITOR_FAMILY)}-competitor family",
            color="white",
            fontsize=8,
        )
        for spine in ax.spines.values():
            spine.set_edgecolor("white")

        fig.tight_layout()
        out = run_dir / "figures" / "primary_mae_brain.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="black")
        plt.close(fig)
        logger.info("Wrote figures/primary_mae_brain.png")

    def _fig_region_grid(
        self,
        run_dir: Path,
        ring_a_pt: pd.DataFrame,
        palette: dict[str, str],
        order: list[str],
    ) -> None:
        """Metric × region small-multiples grid."""
        metrics_regions = [
            ("mae_brain", "MAE (brain)"),
            ("mae_wt", "MAE (WT)"),
            ("mae_bg_undilated", "MAE (bg)"),
            ("ssim_brain", "SSIM (brain)"),
            ("ssim_wt", "SSIM (WT)"),
            ("ssim_bg_undilated", "SSIM (bg)"),
            ("psnr_brain", "PSNR (brain)"),
            ("psnr_wt", "PSNR (WT)"),
        ]
        cols_available = [(m, lbl) for m, lbl in metrics_regions if m in ring_a_pt.columns]
        if not cols_available:
            return

        n = len(cols_available)
        ncols = 4
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.5), squeeze=False)
        fig.patch.set_facecolor("black")

        for idx, (metric, label) in enumerate(cols_available):
            ax = axes[idx // ncols][idx % ncols]
            ax.set_facecolor("black")
            methods_present = [m for m in order if m in ring_a_pt["method"].values]
            for i, method in enumerate(methods_present):
                sel_nfe = SELECTION_NFE.get(method, 1)
                sub = ring_a_pt[(ring_a_pt["method"] == method) & (ring_a_pt["nfe"] == sel_nfe)]
                if sub.empty:
                    sub = ring_a_pt[ring_a_pt["method"] == method]
                vals = sub[metric].dropna().values
                if len(vals) == 0:
                    continue
                color = palette.get(method, "#888888")
                ax.scatter(
                    [i] * len(vals),
                    vals,
                    s=4,
                    alpha=0.4,
                    color=color,
                )
                ax.scatter([i], [np.median(vals)], s=20, color=color, zorder=5, marker="_")
            ax.set_title(label, color="white", fontsize=7)
            ax.set_xticks(range(len(methods_present)))
            ax.set_xticklabels(
                [m.replace("VENA-S1-v3b-rw", "VENA") for m in methods_present],
                rotation=90,
                fontsize=5,
                color="white",
            )
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("white")

        for idx in range(len(cols_available), nrows * ncols):
            fig.delaxes(axes[idx // ncols][idx % ncols])

        fig.suptitle("Region metric grid — Ring A", color="white", fontsize=9)
        fig.tight_layout()
        out = run_dir / "figures" / "region_grid.png"
        fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="black")
        plt.close(fig)
        logger.info("Wrote figures/region_grid.png")

    def _fig_cost_pareto(
        self,
        run_dir: Path,
        ring_a_pt: pd.DataFrame,
        per_patient_df: pd.DataFrame,
        palette: dict[str, str],
    ) -> None:
        """Cost-quality Pareto: MAE vs inference_seconds."""
        if "inference_seconds" not in per_patient_df.columns:
            return

        ring_a = per_patient_df[per_patient_df["ring"] == "A"]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_facecolor("black")
        fig.patch.set_facecolor("black")

        # Group by method × nfe
        methods_with_multi_nfe: set[str] = set()
        for method, grp in ring_a.groupby("method"):
            unique_nfe = sorted(grp["nfe"].unique())
            if len(unique_nfe) > 1:
                methods_with_multi_nfe.add(str(method))

        for method, grp in ring_a.groupby("method"):
            color = palette.get(str(method), "#888888")
            unique_nfe = sorted(grp["nfe"].unique())
            xs = []
            ys = []
            for nfe in unique_nfe:
                sub = grp[grp["nfe"] == nfe]
                mae = sub["mae_brain"].mean()
                secs = sub["inference_seconds"].mean()
                xs.append(secs)
                ys.append(mae)
                ax.scatter([secs], [mae], s=30, color=color, zorder=5)
                ax.annotate(
                    f"NFE={nfe}",
                    (secs, mae),
                    textcoords="offset points",
                    xytext=(3, 3),
                    fontsize=5,
                    color=color,
                )
            if str(method) in methods_with_multi_nfe and len(xs) > 1:
                ax.plot(xs, ys, color=color, linewidth=0.8, alpha=0.7)

        ax.set_xlabel("Inference seconds (mean per scan)", color="white", fontsize=8)
        ax.set_ylabel("MAE (brain, Ring A, mean)", color="white", fontsize=8)
        ax.set_title("Cost-quality Pareto — §4.5", color="white", fontsize=8)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")

        fig.tight_layout()
        out = run_dir / "figures" / "cost_quality_pareto.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="black")
        plt.close(fig)
        logger.info("Wrote figures/cost_quality_pareto.png")

    def _fig_zgd(
        self,
        run_dir: Path,
        ring_a_pt: pd.DataFrame,
        palette: dict[str, str],
        order: list[str],
    ) -> None:
        """ZGD per method — 2D tier should separate visibly."""
        if "zgd" not in ring_a_pt.columns:
            return

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_facecolor("black")
        fig.patch.set_facecolor("black")

        methods_present = [m for m in order if m in ring_a_pt["method"].values]
        for i, method in enumerate(methods_present):
            sel_nfe = SELECTION_NFE.get(method, 1)
            sub = ring_a_pt[(ring_a_pt["method"] == method) & (ring_a_pt["nfe"] == sel_nfe)]
            if sub.empty:
                sub = ring_a_pt[ring_a_pt["method"] == method]
            vals = sub["zgd"].dropna().values
            if len(vals) == 0:
                continue
            color = palette.get(method, "#888888")
            ax.scatter(
                np.full(len(vals), i),
                vals,
                s=6,
                alpha=0.5,
                color=color,
            )
            ax.scatter([i], [np.median(vals)], s=40, color=color, zorder=5, marker="D")

        ax.axhline(1.0, color="white", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_xticks(range(len(methods_present)))
        ax.set_xticklabels(methods_present, rotation=45, ha="right", fontsize=7, color="white")
        ax.set_ylabel("ZGD (z-gradient ratio)", color="white", fontsize=8)
        ax.set_title("§4.7 ZGD — inter-slice consistency (1.0 = ideal)", color="white", fontsize=8)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")

        fig.tight_layout()
        out = run_dir / "figures" / "zgd.png"
        fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="black")
        plt.close(fig)
        logger.info("Wrote figures/zgd.png")

    # ------------------------------------------------------------------
    # decision.json
    # ------------------------------------------------------------------

    def _write_decision(
        self,
        run_dir: Path,
        *,
        n_files: int,
        n_scans: int,
        n_patients: int,
        elapsed_s: float,
        c0_results: dict[str, dict[str, float]],
        pred_mode_counts: dict[str, dict[str, int]],
        skipped_smoke_shards: list[str],
    ) -> None:
        cfg = self.cfg
        try:
            git_sha = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=Path(__file__).parent,
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
        except Exception:
            git_sha = "unknown"

        payload: dict[str, object] = {
            "schema_version": "1.0",
            "producer": "routines.validation.paired_fidelity:1.0",
            "produced_at": datetime.now(tz=UTC).isoformat(),
            "git_sha": git_sha,
            "data_root": str(cfg.data_root),
            "output_root": str(cfg.output_root),
            # Metric parameters
            "dilate_k": cfg.dilate_k,
            "ssim_window_size": cfg.ssim_window_size,
            "ssim_window_sigma": cfg.ssim_window_sigma,
            "ms_ssim_weights": list(cfg.ms_ssim_weights),
            "ms_ssim_bbox_margin": cfg.ms_ssim_bbox_margin,
            "n_bootstrap": cfg.n_bootstrap,
            "bootstrap_seed": cfg.bootstrap_seed,
            "device": cfg.device,
            # Filters
            "filter_methods": list(cfg.filter_methods),
            "filter_cohorts": list(cfg.filter_cohorts),
            "filter_nfe": list(cfg.filter_nfe),
            "filter_rings": list(cfg.filter_rings),
            # Coverage
            "n_prediction_files": n_files,
            "n_scans": n_scans,
            "n_patients": n_patients,
            "elapsed_s": round(elapsed_s, 1),
            # SSIM treatment (task spec §3)
            "ssim_treatment": (
                "principled: compute_ssim_and_cs spatial map, averaged inside "
                f"center-cropped region mask (trim={cfg.ssim_window_size // 2} voxels/side)"
            ),
            "ms_ssim_treatment": (
                "API-limited: brain=full-volume, wt=WT-bbox-crop, "
                f"bg_undilated=same-as-brain. bbox_margin={cfg.ms_ssim_bbox_margin}, "
                f"min_dim=90. NaN when bbox < min_dim."
            ),
            # Family sizes (pre-registered)
            "competitor_family": list(COMPETITOR_FAMILY),
            "ablation_family": list(ABLATION_FAMILY),
            "n_competitor": len(COMPETITOR_FAMILY),
            "n_ablation": len(ABLATION_FAMILY),
            "holm_correction": "Holm-Bonferroni per (metric, region, family)",
            # C0 sanity
            "c0_sanity": c0_results,
            # §4.1 scoring-space audit — per-method mode counts
            # Expected: 15 of 16 methods → "raw"; C0-Identity (scanner units) → "harmonised".
            # Any method appearing in "harmonised" that is NOT C0-Identity is a regression.
            "pred_mode_counts_by_method": pred_mode_counts,
            "scoring_space_note": (
                "Methods trained on percentile-normalised T1c are scored on pred_raw "
                "(brain p99.5 ≤ 1.05). Scanner-unit methods (C0-Identity) are scored on "
                "pred_harmonised. The decision is per-scan via select_scoring_volume, not "
                "a hard-coded method list. raw_p995 in per_scan CSV quantifies under-saturation "
                "(e.g. C4-3D-DiT p99.5 ≈ 0.38 vs reference ~1.0 — a reportable finding per §4.1)."
            ),
            # Shard provenance (§3.1 — smoke shards skipped at discovery time)
            "skipped_smoke_shards": skipped_smoke_shards,
        }
        write_decision_json(run_dir, payload)
        logger.info("Wrote decision.json")

    # ------------------------------------------------------------------
    # report.md
    # ------------------------------------------------------------------

    def _write_report(
        self,
        run_dir: Path,
        n_scans: int,
        n_patients: int,
        elapsed_s: float,
        c0_results: dict[str, dict[str, float]],
    ) -> None:
        cfg = self.cfg
        c0_brain = c0_results.get("C0-Identity", {}).get("mae_brain", float("nan"))
        vena_brain = c0_results.get(VENA_HEADLINE, {}).get("mae_brain", float("nan"))
        vena_wt = c0_results.get(VENA_HEADLINE, {}).get("mae_wt", float("nan"))

        md = f"""# Paired Fidelity Analysis — Report

**Produced**: {datetime.now(tz=UTC).isoformat()}
**Data root**: `{cfg.data_root}`
**Scans processed**: {n_scans} scans → {n_patients} unique patients
**Wall clock**: {elapsed_s:.1f} s

## SSIM treatment (§3 decision)

The principled "SSIM-map averaged in region" approach is used.
`monai.metrics.regression.compute_ssim_and_cs` returns the full spatial SSIM
map (valid convolution; shape `H−k+1, W−k+1, D−k+1` for k={cfg.ssim_window_size}).
The region mask is center-cropped by `k//2 = {cfg.ssim_window_size // 2}` voxels
per edge before averaging.  This avoids the degenerate mean-fill proxy
(model-coding-standards.md rule 14 / SHARED_CONTRACTS §11 trap 7).

## MS-SSIM treatment

`compute_ms_ssim` reduces spatially — no per-voxel map is available.
- **brain**: full brain volume (valid; volumes are zero outside brain).
- **wt**: WT bounding-box crop + {cfg.ms_ssim_bbox_margin} voxels; NaN when any
  dim < 90 (MONAI minimum for 4-level MS-SSIM with kernel_size=11).
- **bg_undilated**: same as brain (WT <5% of brain volume; contamination negligible).

## C0-Identity sanity check

C0-Identity (identity pass-through) is the designed null floor.
Every real method must beat C0 inside WT — if not, the metric is wrong.

| Method | brain MAE | WT MAE |
|--------|-----------|--------|
| C0-Identity | {c0_brain:.4f} | {c0_results.get("C0-Identity", {}).get("mae_wt", float("nan")):.4f} |
| {VENA_HEADLINE} | {vena_brain:.4f} | {vena_wt:.4f} |

## Parameters

| Parameter | Value |
|-----------|-------|
| `dilate_k` | {cfg.dilate_k} |
| `ssim_window_size` | {cfg.ssim_window_size} |
| `ssim_window_sigma` | {cfg.ssim_window_sigma} |
| `ms_ssim_weights` | {list(cfg.ms_ssim_weights)} |
| `n_bootstrap` | {cfg.n_bootstrap} |
| `bootstrap_seed` | {cfg.bootstrap_seed} |
| Competitor family | {len(COMPETITOR_FAMILY)} methods |
| Ablation family | {len(ABLATION_FAMILY)} methods |

## Figures

- `figures/primary_mae_brain.png` — primary endpoint: brain MAE distribution,
  Ring A, significance brackets (Holm-Bonferroni over competitor family).
- `figures/region_grid.png` — metric × region small-multiples.
- `figures/cost_quality_pareto.png` — MAE vs inference_seconds (§4.5).
- `figures/zgd.png` — §4.7 ZGD inter-slice consistency.
"""
        (run_dir / "report.md").write_text(md, encoding="utf-8")
        logger.info("Wrote report.md")
