"""Cost–quality Pareto study (Study 2) — speed vs fidelity renderer.

Consumes the frozen ``paired_fidelity_patient.csv`` per-scan CSV and
produces:

- ``tables/pareto_points.csv`` — one row per (method, nfe), Ring A.
- ``tables/table2a_matched_nfe5.csv`` — matched NFE=5 fidelity table.
- ``tables/table2b_cost_ledger.csv`` — cost at selection NFE per method.
- ``figures/fig_pareto_{metric}_{region}.png`` — 9 Pareto frontier figures
  (metrics × regions: ms_ssim/ssim/psnr × brain/wt/bg_undilated).
  x = median inference seconds (log), y = mean metric (all higher-better).
- ``decision.json`` — provenance record.

No re-scoring is performed; this is a pure aggregation and visualisation
over pre-computed per-scan metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")

from routines.validation.studies._shared import domain_of, per_scan_csv
from vena.validation.artifacts import make_run_dir, symlink_latest, write_decision_json
from vena.validation.registry import METHOD_SPECS, SELECTION_NFE, VENA_HEADLINE
from vena.validation.stats import bootstrap_ci, cliffs_delta, collapse_to_patient, paired_wilcoxon

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Generation-formulation (tier) for each method — populated at import time.
_METHOD_TIER: dict[str, str] = {s.key: s.tier for s in METHOD_SPECS}

#: Timer scope per generation domain (latent methods time includes VAE decode).
_TIMER_SCOPE: dict[str, str] = {
    "reference": "passthrough",
    "pixel": "forward_pass",
    "latent": "decode_included",
}

#: Marker shape by domain (space → shape; colour is per-method via tab20).
_DOMAIN_MARKER: dict[str, str] = {
    "reference": "D",
    "pixel": "s",
    "latent": "o",
}

#: Human-readable label for each domain (used in shape legend).
_DOMAIN_LABEL: dict[str, str] = {
    "reference": "Reference — diamond (D)",
    "pixel": "Image-space — square (s)",
    "latent": "Latent-space — circle (o)",
}

#: One colour per method from tab20 (16 methods, indices 0/20 … 15/20).
_METHOD_COLORS: dict[str, tuple[float, float, float, float]] = {
    s.key: matplotlib.cm.tab20(i / 20) for i, s in enumerate(METHOD_SPECS)
}

#: Pareto figures: (metric, region) combinations — all higher-better.
_PARETO_METRICS: tuple[str, ...] = ("ms_ssim", "ssim", "psnr")
_PARETO_REGIONS: tuple[str, ...] = ("brain", "wt", "bg_undilated")

#: Methods compared in Table 2A (matched NFE=5 sub-analysis).
_TABLE2A_METHODS: tuple[str, ...] = (
    "C4-3D-DiT",
    "C5-T1C-RFlow",
    "VENA-S1-v3a",
    VENA_HEADLINE,  # VENA-S1-v3b-rw
)
_TABLE2A_NFE: int = 5

# ---------------------------------------------------------------------------
# Pareto dominance helper (pure function — importable for testing)
# ---------------------------------------------------------------------------


def compute_pareto_dominant(
    df: pd.DataFrame,
    x_col: str = "median_inference_seconds",
    y_col: str = "mean_ms_ssim_brain",
) -> pd.Series:
    """Return a boolean Series flagging Pareto-dominant rows.

    A row is dominant iff no other row has both ``x_col`` ≤ its value *and*
    ``y_col`` ≥ its value, with at least one strict inequality.  Minimising
    ``x_col`` and maximising ``y_col`` is the assumed objective.

    Parameters
    ----------
    df :
        DataFrame with numeric columns ``x_col`` and ``y_col``.
    x_col :
        Column to minimise (lower is better — e.g. inference seconds).
    y_col :
        Column to maximise (higher is better — e.g. MS-SSIM).

    Returns
    -------
    pd.Series
        Boolean Series (same index as *df*).  ``True`` = Pareto-dominant.
    """
    xs = df[x_col].to_numpy(dtype=float)
    ys = df[y_col].to_numpy(dtype=float)
    n = len(xs)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j ≤ xi on cost AND j ≥ yi on quality (both ≤/≥)
            # with at least one strict.
            if xs[j] <= xs[i] and ys[j] >= ys[i]:
                if xs[j] < xs[i] or ys[j] > ys[i]:
                    dominated[i] = True
                    break
    return pd.Series(~dominated, index=df.index)


def _nearest_neighbor_path(
    xs_log: np.ndarray,
    ys: np.ndarray,
    x_global_range: tuple[float, float],
    y_global_range: tuple[float, float],
) -> list[int]:
    """Greedy nearest-neighbor ordering for a single method's NFE points.

    Distances are computed in normalized plot coordinates (min-max scaling of
    ``log10(seconds)`` and ``y`` each to [0, 1] using the global axis ranges)
    so neither axis dominates due to scale differences.

    Parameters
    ----------
    xs_log :
        ``log10(median_inference_seconds)`` for each NFE point of the method.
    ys :
        Fidelity metric values aligned with ``xs_log``.
    x_global_range :
        ``(min, max)`` of ``log10(seconds)`` across **all** methods/NFEs in
        the figure, for normalization.
    y_global_range :
        ``(min, max)`` of the fidelity metric across all methods/NFEs.

    Returns
    -------
    list[int]
        Indices into ``xs_log`` / ``ys`` giving the path order, starting from
        the point with the smallest ``xs_log`` value.
    """
    n = len(xs_log)
    if n <= 1:
        return list(range(n))

    x_lo, x_hi = x_global_range
    y_lo, y_hi = y_global_range
    x_scale = x_hi - x_lo if x_hi > x_lo else 1.0
    y_scale = y_hi - y_lo if y_hi > y_lo else 1.0

    xs_norm = (xs_log - x_lo) / x_scale
    ys_norm = (ys - y_lo) / y_scale

    start = int(np.argmin(xs_log))
    visited = [False] * n
    path = [start]
    visited[start] = True

    for _ in range(n - 1):
        current = path[-1]
        best_dist = float("inf")
        best_next = -1
        for j in range(n):
            if visited[j]:
                continue
            dx = xs_norm[current] - xs_norm[j]
            dy = ys_norm[current] - ys_norm[j]
            dist = dx * dx + dy * dy
            if dist < best_dist:
                best_dist = dist
                best_next = j
        path.append(best_next)
        visited[best_next] = True

    return path


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class CostQualityError(Exception):
    """Raised on configuration or data errors in the cost–quality study."""


@dataclass(frozen=True)
class CostQualityStudyConfig:
    """Frozen configuration for :class:`CostQualityStudy`.

    Parameters
    ----------
    per_scan_csv_path :
        Path to the frozen ``paired_fidelity_patient.csv``.  Defaults to
        ``_shared.per_scan_csv("paired_fidelity", "paired_fidelity_patient.csv")``.
    output_root :
        Article results root.  The study writes to
        ``<output_root>/cost_quality/<UTC-stamp>/``.
    ring :
        Ring to analyse (default ``"A"``).
    n_bootstrap :
        Bootstrap replicates for CI estimation.
    bootstrap_seed :
        Fixed seed for reproducibility.
    """

    per_scan_csv_path: Path = field(
        default_factory=lambda: per_scan_csv("paired_fidelity", "paired_fidelity_patient.csv")
    )
    output_root: Path = field(
        default_factory=lambda: Path("/media/mpascual/Sandisk2TB/research/vena/results/article")
    )
    ring: str = "A"
    n_bootstrap: int = 10_000
    bootstrap_seed: int = 1337

    @classmethod
    def from_yaml(cls, path: Path) -> CostQualityStudyConfig:
        """Parse a YAML config file.

        All keys are optional and fall back to the class defaults.

        Parameters
        ----------
        path :
            Path to the YAML config.

        Raises
        ------
        CostQualityError
            When the file is not found or the YAML is malformed.
        """
        path = Path(path)
        if not path.is_file():
            raise CostQualityError(f"Config not found: {path}")
        with path.open() as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}

        csv_path: Path
        if "per_scan_csv" in raw:
            csv_path = Path(raw["per_scan_csv"])
        else:
            csv_path = per_scan_csv("paired_fidelity", "paired_fidelity_patient.csv")

        output_root: Path
        if "output_root" in raw:
            output_root = Path(raw["output_root"])
        else:
            output_root = Path("/media/mpascual/Sandisk2TB/research/vena/results/article")

        return cls(
            per_scan_csv_path=csv_path,
            output_root=output_root,
            ring=str(raw.get("ring", "A")),
            n_bootstrap=int(raw.get("n_bootstrap", 10_000)),
            bootstrap_seed=int(raw.get("bootstrap_seed", 1337)),
        )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class CostQualityStudy:
    """Render the speed–fidelity Pareto study from a frozen per-scan CSV.

    Parameters
    ----------
    cfg :
        Frozen study configuration.

    Examples
    --------
    >>> cfg = CostQualityStudyConfig()
    >>> run_dir = CostQualityStudy(cfg).run()
    """

    def __init__(self, cfg: CostQualityStudyConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> Path:
        """Execute the full study and return the artifact directory.

        Returns
        -------
        Path
            Artifact run directory ``<output_root>/cost_quality/<UTC-stamp>/``.
            A ``LATEST`` symlink in the parent is also updated.
        """
        cfg = self.cfg
        t0 = datetime.now(tz=UTC)

        logger.info(
            "CostQualityStudy: loading %s (ring=%s)",
            cfg.per_scan_csv_path,
            cfg.ring,
        )
        df = pd.read_csv(cfg.per_scan_csv_path)
        ring_a = self._filter_ring(df)

        # Validate premise before any compute.
        self._assert_timing_populated(ring_a)

        run_dir = make_run_dir(cfg.output_root, "cost_quality")
        logger.info("Run directory: %s", run_dir)

        pareto_df = self._build_pareto_table(ring_a)
        pareto_df.to_csv(run_dir / "tables" / "pareto_points.csv", index=False)
        logger.info("Wrote tables/pareto_points.csv (%d rows)", len(pareto_df))

        t2a_df = self._build_table2a(ring_a)
        t2a_df.to_csv(run_dir / "tables" / "table2a_matched_nfe5.csv", index=False)
        logger.info("Wrote tables/table2a_matched_nfe5.csv")

        t2b_df = self._build_cost_ledger(ring_a)
        t2b_df.to_csv(run_dir / "tables" / "table2b_cost_ledger.csv", index=False)
        logger.info("Wrote tables/table2b_cost_ledger.csv")

        figs_produced = self._make_all_pareto_figs(pareto_df, run_dir)

        elapsed_s = (datetime.now(tz=UTC) - t0).total_seconds()
        self._write_decision(run_dir, df, ring_a, pareto_df, elapsed_s, figs_produced)

        symlink_latest(run_dir)
        logger.info("Done in %.1f s — artifact: %s", elapsed_s, run_dir)
        return run_dir

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _filter_ring(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to the configured ring and drop rows with NaN timing."""
        ring_a = df[df["ring"] == self.cfg.ring].copy()
        if ring_a.empty:
            raise CostQualityError(f"No rows found for ring={self.cfg.ring!r} in CSV.")
        return ring_a

    @staticmethod
    def _assert_timing_populated(ring_a: pd.DataFrame) -> None:
        """Raise if ``inference_seconds`` or ``peak_vram_mb`` are all-NaN.

        Raises
        ------
        CostQualityError
            When timing columns are missing or entirely null (PREMISE-FALSE).
        """
        for col in ("inference_seconds", "peak_vram_mb"):
            if col not in ring_a.columns:
                raise CostQualityError(
                    f"PREMISE-FALSE: column {col!r} absent from CSV — Pareto x-axis has no data."
                )
            n_nan = ring_a[col].isna().sum()
            if n_nan == len(ring_a):
                raise CostQualityError(
                    f"PREMISE-FALSE: {col!r} is all-NaN in ring-A — Pareto x-axis has no data."
                )
            if n_nan > 0:
                logger.warning(
                    "%s has %d/%d NaN in ring %s; those rows are excluded.",
                    col,
                    n_nan,
                    len(ring_a),
                    ring_a["ring"].iloc[0],
                )

    # ------------------------------------------------------------------
    # Table: pareto_points.csv
    # ------------------------------------------------------------------

    def _build_pareto_table(self, ring_a: pd.DataFrame) -> pd.DataFrame:
        """Build pareto_points.csv — one row per (method, nfe).

        Computes median/IQR of inference seconds, mean fidelity metrics (all
        9 combinations of metric × region plus MAE_brain), mean VRAM, the
        legacy ``pareto_dominant`` flag (ms_ssim_brain), and 9 per-figure
        dominance columns ``dominant_{metric}_{region}``.
        """
        rows: list[dict[str, object]] = []
        for (method, nfe), grp in ring_a.groupby(["method", "nfe"]):
            times = grp["inference_seconds"].dropna()
            dom = domain_of(str(method))
            tier = _METHOD_TIER.get(str(method), "flow")
            sel_nfe = SELECTION_NFE.get(str(method), -1)
            row: dict[str, object] = {
                "method": method,
                "nfe": int(nfe),
                "domain": dom,
                "tier": tier,
                "median_inference_seconds": float(np.median(times)),
                "iqr_lo": float(np.percentile(times, 25)),
                "iqr_hi": float(np.percentile(times, 75)),
                "mean_mae_brain": float(grp["mae_brain"].mean()),
                "mean_peak_vram_mb": float(grp["peak_vram_mb"].mean()),
                "n_patients": len(grp),
                "is_selection_nfe": bool(int(nfe) == sel_nfe),
            }
            # All 9 metric × region means.
            for metric in _PARETO_METRICS:
                for region in _PARETO_REGIONS:
                    src = f"{metric}_{region}"
                    if src in grp.columns:
                        row[f"mean_{src}"] = float(grp[src].mean())
            rows.append(row)

        df = pd.DataFrame(rows)

        # Legacy dominance flag (ms_ssim_brain) kept for backward compat.
        if "mean_ms_ssim_brain" in df.columns:
            df["pareto_dominant"] = compute_pareto_dominant(
                df,
                x_col="median_inference_seconds",
                y_col="mean_ms_ssim_brain",
            )
        else:
            df["pareto_dominant"] = False

        # Per-figure dominance columns (one per metric × region).
        for metric in _PARETO_METRICS:
            for region in _PARETO_REGIONS:
                y_col = f"mean_{metric}_{region}"
                dom_col = f"dominant_{metric}_{region}"
                if y_col in df.columns:
                    df[dom_col] = compute_pareto_dominant(
                        df,
                        x_col="median_inference_seconds",
                        y_col=y_col,
                    )
                else:
                    df[dom_col] = False

        return df

    # ------------------------------------------------------------------
    # Table 2A: matched NFE=5 fidelity comparison
    # ------------------------------------------------------------------

    def _build_table2a(self, ring_a: pd.DataFrame) -> pd.DataFrame:
        """Matched-NFE=5 fidelity table for four latent-tier methods.

        Columns: ``method, n_patients, mean_mae_brain, mae_ci_lo, mae_ci_hi,
        mean_ms_ssim_brain, msssim_ci_lo, msssim_ci_hi,
        wilcoxon_p_vs_v3brw, cliffs_vs_v3brw``.

        ``wilcoxon_p_vs_v3brw`` and ``cliffs_vs_v3brw`` are reported on
        MAE_brain (primary endpoint).  ``cliffs_delta(v3brw, method)`` is
        positive when v3b-rw tends to have higher MAE than the comparator
        (i.e. comparator is better).
        """
        cfg = self.cfg
        methods_present = set(ring_a["method"].unique())
        missing = set(_TABLE2A_METHODS) - methods_present
        if missing:
            logger.warning(
                "Table 2A: %d method(s) not found in CSV: %s",
                len(missing),
                sorted(missing),
            )

        sub = ring_a[
            ring_a["method"].isin(_TABLE2A_METHODS) & (ring_a["nfe"] == _TABLE2A_NFE)
        ].copy()

        if sub.empty:
            logger.warning("Table 2A: no rows at NFE=%d for listed methods.", _TABLE2A_NFE)
            return pd.DataFrame(
                columns=[
                    "method",
                    "n_patients",
                    "mean_mae_brain",
                    "mae_ci_lo",
                    "mae_ci_hi",
                    "mean_ms_ssim_brain",
                    "msssim_ci_lo",
                    "msssim_ci_hi",
                    "wilcoxon_p_vs_v3brw",
                    "cliffs_vs_v3brw",
                ]
            )

        # Cohort strata for stratified bootstrap (before cohort is averaged out).
        cohort_map: dict[str, str] = sub.groupby("patient_id")["cohort"].first().to_dict()

        # Collapse to patient (mean over cohorts if a patient appears in >1).
        pt = collapse_to_patient(
            sub,
            value_cols=["mae_brain", "ms_ssim_brain"],
            by=["method", "nfe", "patient_id"],
        )

        # Reference arm for paired tests.
        vena_pt = pt[pt["method"] == VENA_HEADLINE].set_index("patient_id")

        rows: list[dict[str, object]] = []
        for m in _TABLE2A_METHODS:
            arm = pt[pt["method"] == m].set_index("patient_id")
            if arm.empty:
                continue
            strata = np.array([cohort_map.get(str(pid), "") for pid in arm.index])
            mae_vals = arm["mae_brain"].to_numpy(dtype=float)
            mss_vals = arm["ms_ssim_brain"].to_numpy(dtype=float)

            mae_lo, mae_hi = bootstrap_ci(
                mae_vals,
                n_boot=cfg.n_bootstrap,
                seed=cfg.bootstrap_seed,
                strata=strata,
            )
            mss_lo, mss_hi = bootstrap_ci(
                mss_vals,
                n_boot=cfg.n_bootstrap,
                seed=cfg.bootstrap_seed,
                strata=strata,
            )

            if m == VENA_HEADLINE or vena_pt.empty:
                wilcoxon_p: float = float("nan")
                cliff: float = float("nan")
            else:
                wres = paired_wilcoxon(vena_pt["mae_brain"], arm["mae_brain"])
                wilcoxon_p = wres.pvalue
                common = vena_pt.index.intersection(arm.index)
                cliff = cliffs_delta(
                    vena_pt.loc[common, "mae_brain"].to_numpy(dtype=float),
                    arm.loc[common, "mae_brain"].to_numpy(dtype=float),
                )

            rows.append(
                {
                    "method": m,
                    "n_patients": len(arm),
                    "mean_mae_brain": float(np.mean(mae_vals)),
                    "mae_ci_lo": mae_lo,
                    "mae_ci_hi": mae_hi,
                    "mean_ms_ssim_brain": float(np.mean(mss_vals)),
                    "msssim_ci_lo": mss_lo,
                    "msssim_ci_hi": mss_hi,
                    "wilcoxon_p_vs_v3brw": wilcoxon_p,
                    "cliffs_vs_v3brw": cliff,
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Table 2B: cost ledger at selection NFE
    # ------------------------------------------------------------------

    def _build_cost_ledger(self, ring_a: pd.DataFrame) -> pd.DataFrame:
        """Per-method cost at the pre-registered selection NFE.

        Columns: ``method, selection_nfe, median_seconds, iqr_lo, iqr_hi,
        mean_peak_vram_mb, timer_scope``.
        """
        rows: list[dict[str, object]] = []
        for method in sorted(ring_a["method"].unique()):
            sel_nfe = SELECTION_NFE.get(str(method), 1)
            sub = ring_a[(ring_a["method"] == method) & (ring_a["nfe"] == sel_nfe)]
            if sub.empty:
                # Fallback: use whichever NFE is present.
                sub = ring_a[ring_a["method"] == method]
                if sub.empty:
                    continue
                sel_nfe = int(sub["nfe"].iloc[0])
                logger.warning(
                    "Cost ledger: method %r selection_nfe=%d not found; falling back to nfe=%d.",
                    method,
                    SELECTION_NFE.get(str(method), 1),
                    sel_nfe,
                )

            times = sub["inference_seconds"].dropna()
            dom = domain_of(str(method))
            rows.append(
                {
                    "method": method,
                    "selection_nfe": int(sel_nfe),
                    "median_seconds": float(np.median(times)),
                    "iqr_lo": float(np.percentile(times, 25)),
                    "iqr_hi": float(np.percentile(times, 75)),
                    "mean_peak_vram_mb": float(sub["peak_vram_mb"].mean()),
                    "timer_scope": _TIMER_SCOPE.get(dom, "decode_included"),
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Figures (9 Pareto plots: metric × region)
    # ------------------------------------------------------------------

    def _make_all_pareto_figs(self, pareto_df: pd.DataFrame, run_dir: Path) -> list[str]:
        """Produce all 9 Pareto frontier figures and return their filenames.

        One figure per (metric, region): ``fig_pareto_{metric}_{region}.png``.
        x = median inference seconds (log scale), y = mean metric (all
        higher-better).  Marker shape encodes generation domain; colour
        encodes method (tab20).
        """
        produced: list[str] = []
        for metric in _PARETO_METRICS:
            for region in _PARETO_REGIONS:
                y_col = f"mean_{metric}_{region}"
                dom_col = f"dominant_{metric}_{region}"
                if y_col not in pareto_df.columns:
                    logger.warning("Column %r missing — skipping figure.", y_col)
                    continue
                filename = f"fig_pareto_{metric}_{region}.png"
                self._render_single_pareto_fig(
                    pareto_df=pareto_df,
                    y_col=y_col,
                    dominant_col=dom_col,
                    metric=metric,
                    region=region,
                    filename=filename,
                    run_dir=run_dir,
                )
                produced.append(filename)
        return produced

    def _render_single_pareto_fig(
        self,
        pareto_df: pd.DataFrame,
        y_col: str,
        dominant_col: str,
        metric: str,
        region: str,
        filename: str,
        run_dir: Path,
    ) -> None:
        """Render one Pareto figure (white background, shape=domain, colour=method).

        Design
        ------
        - White background (fig + axes ``facecolor='white'``).
        - x = median inference seconds (log scale).  y = mean metric (↑ better).
        - Marker **shape** encodes generation domain; **colour** encodes method
          (16 colours from tab20).
        - Same-method in-scale points connected by a line via greedy NN path
          in normalised plot coords.
        - Pareto-dominant points: white star ``*`` centred in the marker; alpha=1.
        - Dominated points: alpha=0.6, except any ``VENA-*`` method = alpha=1.
        - Each in-scale point annotated with its NFE value (tiny text above).
        - Off-scale points (below the adaptive y lower-limit) drawn as ▼ at the
          clip line with "↓NFE" label so the diffusion-needs-many-steps story is
          preserved.
        - y-axis lower limit set adaptively to just below the lowest frontier
          point, expanding the populated band to fill the figure.
        - Method identity moved to a compact colour-swatch legend outside the
          axes (right side); no per-point name annotation at line endpoints.
        """
        try:
            from vena.validation.plotting import setup_figure_style

            setup_figure_style()
        except Exception:
            plt.rcParams.update({"figure.dpi": 150})

        # Disable constrained_layout for this figure so that subplots_adjust can
        # reserve right-side space for the method colour legend without warnings.
        with plt.rc_context({"figure.constrained_layout.use": False}):
            fig, ax = plt.subplots(1, 1, figsize=(13, 7), facecolor="white")
        ax.set_facecolor("white")

        # Global x range for NN normalization.
        secs = pareto_df["median_inference_seconds"].to_numpy(dtype=float)
        ys_all = pareto_df[y_col].to_numpy(dtype=float)
        xs_log_all = np.log10(secs)
        x_range = (float(xs_log_all.min()), float(xs_log_all.max()))

        # Adaptive y-axis: zoom hard to the frontier/viable band.
        # y_lo = min(frontier_y) - 0.15*(max_in_scale - min(frontier_y))
        # Fallback when frontier has <3 points: use Q75 of all points as band floor.
        dom_mask = pareto_df[dominant_col].to_numpy(dtype=bool)
        n_frontier = int(dom_mask.sum())
        max_in_scale = float(np.nanmax(ys_all))
        if n_frontier >= 3:
            y_frontier_min = float(np.nanmin(pareto_df.loc[dom_mask, y_col]))
        else:
            y_frontier_min = float(np.nanpercentile(ys_all, 75))
        span = max(max_in_scale - y_frontier_min, 1e-6)
        y_lo = y_frontier_min - 0.15 * span
        y_hi = max_in_scale + 0.05 * span
        # Clip position for off-scale markers (just above the bottom edge).
        clip_y = y_lo + 0.012 * (y_hi - y_lo)
        nn_y_range = (y_lo, y_hi)

        # Shade dominated region (step function below the frontier).
        frontier = pareto_df[dom_mask].sort_values("median_inference_seconds")
        if not frontier.empty:
            self._shade_dominated_region(ax, frontier, y_col)

        # Colour legend handles (one swatch per method, built during the loop).
        method_handles: list[mpatches.Patch] = []
        seen_methods: set[str] = set()
        off_scale_count = 0

        # Draw per-method lines + scatter points.
        for method, grp in pareto_df.groupby("method"):
            method_str = str(method)
            color = _METHOD_COLORS.get(method_str, (0.5, 0.5, 0.5, 1.0))
            domain = domain_of(method_str)
            marker = _DOMAIN_MARKER.get(domain, "o")
            is_vena = method_str.startswith("VENA-")

            grp_secs = grp["median_inference_seconds"].to_numpy(dtype=float)
            grp_ys = grp[y_col].to_numpy(dtype=float)
            grp_dom = grp[dominant_col].to_numpy(dtype=bool)
            grp_nfe = grp["nfe"].to_numpy()
            grp_log = np.log10(grp_secs)

            # Accumulate colour legend handle (deduplicated).
            if method_str not in seen_methods:
                seen_methods.add(method_str)
                method_handles.append(mpatches.Patch(facecolor=color, label=method_str))

            # NN path using only in-scale points.
            in_scale = grp_ys >= y_lo
            in_idx = np.where(in_scale)[0]
            if len(in_idx) > 1:
                path_local = _nearest_neighbor_path(
                    grp_log[in_idx], grp_ys[in_idx], x_range, nn_y_range
                )
                path_global = in_idx[path_local]
                ax.plot(
                    grp_secs[path_global],
                    grp_ys[path_global],
                    color=color,
                    alpha=0.55,
                    linewidth=1.1,
                    zorder=2,
                )

            # Scatter points — one at a time to control per-point alpha.
            for k in range(len(grp_secs)):
                is_dom = bool(grp_dom[k])
                nfe_val = int(grp_nfe[k])
                alpha = 1.0 if (is_dom or is_vena) else 0.6

                if not in_scale[k]:
                    # Off-scale: clip to bottom edge, mark with ▼ + "↓NFE".
                    off_scale_count += 1
                    ax.scatter(
                        [grp_secs[k]],
                        [clip_y],
                        s=55,
                        c=[color],
                        marker="v",
                        alpha=0.75,
                        edgecolors="white",
                        linewidths=0.5,
                        zorder=3,
                    )
                    ax.annotate(
                        f"↓{nfe_val}",
                        xy=(grp_secs[k], clip_y),
                        xytext=(0, -9),
                        textcoords="offset points",
                        fontsize=5,
                        ha="center",
                        va="top",
                        color=color,
                        alpha=0.85,
                    )
                    continue

                ax.scatter(
                    [grp_secs[k]],
                    [grp_ys[k]],
                    s=65,
                    c=[color],
                    marker=marker,
                    alpha=alpha,
                    edgecolors="white",
                    linewidths=0.8,
                    zorder=3,
                )
                # White star centred in dominant points.
                if is_dom:
                    ax.scatter(
                        [grp_secs[k]],
                        [grp_ys[k]],
                        s=45,
                        c="white",
                        marker="*",
                        alpha=1.0,
                        zorder=4,
                    )
                # NFE annotation above each in-scale point.
                ax.annotate(
                    str(nfe_val),
                    xy=(grp_secs[k], grp_ys[k]),
                    xytext=(0, 5),
                    textcoords="offset points",
                    fontsize=5,
                    ha="center",
                    va="bottom",
                    color="black",
                    alpha=0.75,
                )

        ax.set_ylim(y_lo, y_hi)
        ax.set_xscale("log")
        ax.set_xlabel("Median inference time (s/volume)", fontsize=10)
        metric_upper = metric.upper().replace("_", "-")
        region_label = region.replace("_", " ")
        ax.set_ylabel(f"Mean {metric_upper} [{region_label}]  (↑ = better)", fontsize=10)
        ax.set_title(
            f"Speed–fidelity Pareto frontier — {metric_upper} [{region_label}]  (Ring A)",
            fontsize=11,
        )
        ax.grid(True, alpha=0.25, linestyle="--", color="gray")

        # Note off-scale points at the bottom of the axes.
        if off_scale_count > 0:
            ax.text(
                0.01,
                0.015,
                f"▼ {off_scale_count} off-scale point(s) clipped to axis "
                "(low-NFE diffusion failures below y-limit)",
                transform=ax.transAxes,
                fontsize=5.5,
                color="dimgray",
                va="bottom",
            )

        # Shape legend (domain → marker) + symbol legend (star / alpha).
        shape_handles = [
            plt.scatter([], [], marker=mk, c="gray", s=70, label=_DOMAIN_LABEL[dom])
            for dom, mk in _DOMAIN_MARKER.items()
        ]
        sym_handles = [
            plt.scatter(
                [],
                [],
                marker="*",
                c="white",
                edgecolors="black",
                s=60,
                label="Pareto-dominant (white ★)",
            ),
            mpatches.Patch(facecolor="gray", alpha=0.6, label="Dominated non-VENA (α=0.6)"),
            mpatches.Patch(facecolor="orange", alpha=1.0, label="VENA-* (α=1.0 always)"),
        ]

        # Reserve right margin then attach the method colour legend to the FIGURE
        # (not the axes): fig.legend() is captured by bbox_inches="tight" even
        # when its bbox_to_anchor places it outside the axes area.
        # Disable any constrained/tight layout engine set by setup_figure_style
        # so that subplots_adjust actually takes effect.
        try:
            fig.set_layout_engine(None)
        except AttributeError:
            pass  # matplotlib < 3.6 — layout engine may override the adjust
        fig.subplots_adjust(right=0.76)
        fig.legend(
            handles=method_handles,
            fontsize=6.0,
            title="Method",
            title_fontsize=6.5,
            loc="center left",
            bbox_to_anchor=(0.77, 0.5),
            framealpha=0.92,
            ncol=1,
        )

        # Shape/symbol legend inside the axes (lower right).
        ax.legend(
            handles=[*shape_handles, *sym_handles],
            fontsize=6.5,
            loc="lower right",
            framealpha=0.85,
        )

        out = run_dir / "figures" / filename
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("Wrote %s", out)

    @staticmethod
    def _shade_dominated_region(
        ax: plt.Axes,
        frontier: pd.DataFrame,
        y_col: str,
    ) -> None:
        """Fill dominated region below the Pareto staircase (log-x axis).

        The staircase is built from Pareto-dominant points sorted by
        ``median_inference_seconds`` ascending.  The filled polygon covers the
        area below and to the right — i.e., where a hypothetical new method
        would be dominated.
        """
        xs = frontier["median_inference_seconds"].to_numpy(dtype=float)
        ys = frontier[y_col].to_numpy(dtype=float)
        if len(xs) == 0:
            return

        # Step function: horizontal-then-vertical.
        stair_x: list[float] = [xs[0]]
        stair_y: list[float] = [ys[0]]
        for i in range(1, len(xs)):
            stair_x.append(xs[i])
            stair_y.append(stair_y[-1])  # horizontal at previous y
            stair_x.append(xs[i])
            stair_y.append(ys[i])  # vertical to new y

        x_right = xs[-1] * 5.0  # extend beyond rightmost point
        stair_x.append(x_right)
        stair_y.append(ys[-1])

        y_bottom = float(np.nanmin(ys)) * 0.97
        poly_x = [*stair_x, x_right, xs[0]]
        poly_y = [*stair_y, y_bottom, y_bottom]
        ax.fill(poly_x, poly_y, color="gray", alpha=0.08, zorder=0)

    # ------------------------------------------------------------------
    # decision.json
    # ------------------------------------------------------------------

    def _write_decision(
        self,
        run_dir: Path,
        full_df: pd.DataFrame,
        ring_a: pd.DataFrame,
        pareto_df: pd.DataFrame,
        elapsed_s: float,
        figs_produced: list[str] | None = None,
    ) -> None:
        """Write decision.json to *run_dir*."""
        cfg = self.cfg
        nfe_grid: dict[str, list[int]] = {
            str(m): sorted(int(n) for n in ring_a[ring_a["method"] == m]["nfe"].unique())
            for m in ring_a["method"].unique()
        }
        payload: dict[str, object] = {
            "schema_version": "1.1",
            "producer": "routines.validation.studies.cost_quality_study:1.1",
            "source_csv": str(cfg.per_scan_csv_path),
            "ring": cfg.ring,
            "n_methods": int(ring_a["method"].nunique()),
            "methods": sorted(ring_a["method"].unique().tolist()),
            "nfe_grid_per_method": nfe_grid,
            "pareto_dominant_points_ms_ssim_brain": [
                {"method": str(r.method), "nfe": int(r.nfe)}
                for r in pareto_df[pareto_df["pareto_dominant"]].itertuples()
            ],
            "figures": figs_produced or [],
            "elapsed_s": round(elapsed_s, 1),
        }
        write_decision_json(run_dir, payload)
        logger.info("Wrote decision.json")


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = CostQualityStudyConfig()
    run_dir = CostQualityStudy(cfg).run()
    print(run_dir, file=sys.stderr)
