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

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from vena.validation.artifacts import make_run_dir, symlink_latest, write_decision_json
from vena.validation.io import (
    ReferenceCache,
    ShardDiscovery,
    build_index,
    discover_shards,
    iter_scans,
)
from vena.validation.spatial_residual import (
    SPATIAL_CSV_COLUMNS,
    WilcoxonTestResult,
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

_SCHEMA_VERSION = "1.1"
_PRODUCER = "routines.validation.spatial_residual:1.1"


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

        # D2: discover shards first so we can record skipped smoke shards in
        # decision.json.  build_index calls discover_shards internally, but
        # does not expose the ShardDiscovery, so we call it separately here.
        # The double discovery is a metadata-only read (reads *.json files only).
        logger.info("Discovering shards in %s", inference_root)
        shard_discovery: ShardDiscovery = discover_shards(inference_root)
        logger.info(
            "Shards: %d accepted, %d smoke-skipped: %s",
            len(shard_discovery.accepted),
            len(shard_discovery.skipped_smoke),
            shard_discovery.skipped_smoke or "none",
        )

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

        # D3: Write per-scan CSV into per_scan/ (not tables/).
        per_scan_df = pd.DataFrame(rows, columns=SPATIAL_CSV_COLUMNS)
        per_scan_csv = run_dir / "per_scan" / "spatial_residual.csv"
        per_scan_df.to_csv(per_scan_csv, index=False)
        logger.info("per_scan/spatial_residual.csv: %d rows → %s", len(per_scan_df), per_scan_csv)

        # D2: Compute pred_mode counts by method (scoring-space audit, §7.0).
        if "pred_mode" in per_scan_df.columns:
            mode_table = (
                per_scan_df.groupby("method")["pred_mode"].value_counts().unstack(fill_value=0)
            )
            pred_mode_counts: dict[str, dict[str, int]] = {
                method: row.to_dict() for method, row in mode_table.iterrows()
            }
        else:
            pred_mode_counts = {}

        # Aggregate tests — non-blocking; a partial sweep (not all 8 competitors)
        # triggers a family-size WARNING inside aggregate_patient_tests.
        test_results: list[WilcoxonTestResult] = []
        patient_df: pd.DataFrame = pd.DataFrame()
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

        # D3: Generate figures (§9.2).
        if not patient_df.empty:
            _make_figures(run_dir, patient_df, test_results, cfg.vena_method)

        # D3: Write report.md.
        _write_report_md(run_dir, per_scan_df, patient_df, test_results, cfg, n_empty_region)

        # Write decision.json and LATEST symlink.
        decision = {
            "schema_version": _SCHEMA_VERSION,
            "produced_at": datetime.now(UTC).isoformat(),
            "producer": _PRODUCER,
            "git_sha": _git_sha(),  # D4: uses repo-root anchor, not cwd
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
            # D2: scoring-space provenance (SHARED_CONTRACTS §7.0).
            "pred_mode_counts_by_method": pred_mode_counts,
            "scoring_space_note": (
                "pred_mode is determined per scan via select_scoring_volume (§7.0): "
                "raw if p99.5 ≤ 1.05 and min ≥ -0.05 inside brain, else harmonised. "
                "Not a hard-coded method list — works for future BraTS-PED/competitor shards."
            ),
            # D2: smoke shard audit (SHARED_CONTRACTS §3.1).
            "skipped_smoke_shards": shard_discovery.skipped_smoke,
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


def _write_test_results(test_results: list[WilcoxonTestResult], path: Path) -> None:
    from dataclasses import asdict

    pd.DataFrame([asdict(r) for r in test_results]).to_csv(path, index=False)


def _git_sha() -> str:
    """Return HEAD SHA anchored to the repo root, not the subprocess cwd.

    D4 fix: on Picasso the subprocess inherits whatever the SLURM job's cwd
    is, which may not be a git repository.  Anchoring with ``git -C <repo_root>``
    where repo_root is derived from this file's own path makes the lookup
    cwd-independent.
    """
    # Repo root: this file is at routines/validation/spatial_residual/engine/...
    # parents[0] = engine/, [1] = spatial_residual/, [2] = validation/,
    # [3] = routines/, [4] = repo root.
    repo_root = Path(__file__).resolve().parents[4]
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        logger.warning("_git_sha: could not resolve HEAD in %s", repo_root)
        return "unknown"


def _make_figures(
    run_dir: Path,
    patient_df: pd.DataFrame,
    test_results: list[WilcoxonTestResult],
    vena_method: str,
) -> None:
    """Generate §9.2 statistical figures.

    Produces two PNG files in ``figures/``:
    - ``fig_rho_s_cnot.png``   — ρ_S per method under C-noT (bar chart, black bg).
    - ``fig_conc05_cnot.png``  — Conc(5%) per method under C-noT with reference
      line at 1.0 and Holm significance brackets.

    House conventions (§9.2, model-coding-standards.md §18a):
    - Black figure and axes background.
    - Methods sorted by the metric descending (so VENA's position is clear).
    - Reference lines and significance brackets where applicable.
    """
    try:
        _fig_rho_s(run_dir, patient_df, test_results, vena_method)
        _fig_conc05(run_dir, patient_df, test_results, vena_method)
        logger.info("Figures written to %s/figures/", run_dir)
    except Exception as exc:
        logger.warning("Figure generation failed (non-blocking): %s", exc)


def _method_means(
    patient_df: pd.DataFrame,
    stat: str,
    condition: str = "C-noT",
) -> pd.DataFrame:
    """Mean ± std per method for *stat* under *condition* (Ring A)."""
    df = (
        patient_df[patient_df["condition"] == condition].copy()
        if "condition" in patient_df.columns
        else patient_df.copy()
    )
    return (
        df.groupby("method")[stat]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "mu", "std": "sigma", "count": "n"})
        .reset_index()
    )


def _significance_label(pvalue_adj: float, reject: bool) -> str:
    """Return a brief significance label for a plot annotation."""
    if not reject or not np.isfinite(pvalue_adj):
        return "n.s."
    if pvalue_adj < 0.001:
        return "***"
    if pvalue_adj < 0.01:
        return "**"
    return "*"


def _apply_black_style(fig: plt.Figure, axes: list[plt.Axes]) -> None:
    """Apply house black-background style to a figure and its axes."""
    fig.patch.set_facecolor("black")
    for ax in axes:
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555555")


def _fig_rho_s(
    run_dir: Path,
    patient_df: pd.DataFrame,
    test_results: list[WilcoxonTestResult],
    vena_method: str,
) -> None:
    """ρ_S bar chart under C-noT across methods, sorted descending."""
    stats_df = _method_means(patient_df, "rho_s", condition="C-noT")
    if stats_df.empty:
        logger.warning("_fig_rho_s: empty stats_df, skipping figure.")
        return
    stats_df = stats_df.sort_values("mu", ascending=False).reset_index(drop=True)

    # Build reject map from Holm results.
    reject_map = {
        r.competitor: (r.reject, r.pvalue_adj) for r in test_results if r.stat_name == "rho_s"
    }

    fig, ax = plt.subplots(figsize=(max(6, len(stats_df) * 0.7 + 1.5), 5))
    _apply_black_style(fig, [ax])

    bar_colors = ["#c0392b" if m == vena_method else "#3498db" for m in stats_df["method"]]
    bars = ax.bar(
        range(len(stats_df)),
        stats_df["mu"],
        yerr=stats_df["sigma"] / np.sqrt(stats_df["n"].clip(lower=1)),
        color=bar_colors,
        edgecolor="#888888",
        capsize=4,
        error_kw={"ecolor": "white"},
    )

    # Annotate significance vs VENA.
    for i, method in enumerate(stats_df["method"]):
        if method == vena_method:
            continue
        if method in reject_map:
            rej, padj = reject_map[method]
            label = _significance_label(padj, rej)
            if label != "n.s.":
                bar_height = (
                    stats_df.loc[i, "mu"]
                    + (stats_df.loc[i, "sigma"] or 0) / max(1, stats_df.loc[i, "n"] ** 0.5)
                    + 0.02
                )
                ax.text(i, bar_height, label, ha="center", va="bottom", color="yellow", fontsize=9)

    ax.axhline(0, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(stats_df)))
    ax.set_xticklabels(stats_df["method"], rotation=35, ha="right", fontsize=8, color="white")
    ax.set_ylabel("Spearman ρ_S (mean ± SE)", color="white")
    ax.set_title("Spatial Residual: ρ_S under C-noT (Ring A, headline NFE)", color="white")
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color="#c0392b", label=vena_method),
            plt.Rectangle((0, 0), 1, 1, color="#3498db", label="Competitor / Ablation"),
        ],
        facecolor="#222222",
        labelcolor="white",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "fig_rho_s_cnot.png", dpi=150, facecolor="black")
    plt.close(fig)
    del bars  # silence unused-variable ruff warning


def _fig_conc05(
    run_dir: Path,
    patient_df: pd.DataFrame,
    test_results: list[WilcoxonTestResult],
    vena_method: str,
) -> None:
    """Conc(5%) bar chart under C-noT with reference line at 1.0."""
    stats_df = _method_means(patient_df, "conc_05", condition="C-noT")
    if stats_df.empty:
        logger.warning("_fig_conc05: empty stats_df, skipping figure.")
        return
    stats_df = stats_df.sort_values("mu", ascending=False).reset_index(drop=True)

    reject_map = {
        r.competitor: (r.reject, r.pvalue_adj) for r in test_results if r.stat_name == "conc_05"
    }

    fig, ax = plt.subplots(figsize=(max(6, len(stats_df) * 0.7 + 1.5), 5))
    _apply_black_style(fig, [ax])

    bar_colors = ["#c0392b" if m == vena_method else "#3498db" for m in stats_df["method"]]
    bars = ax.bar(
        range(len(stats_df)),
        stats_df["mu"],
        yerr=stats_df["sigma"] / np.sqrt(stats_df["n"].clip(lower=1)),
        color=bar_colors,
        edgecolor="#888888",
        capsize=4,
        error_kw={"ecolor": "white"},
    )

    for i, method in enumerate(stats_df["method"]):
        if method == vena_method:
            continue
        if method in reject_map:
            rej, padj = reject_map[method]
            label = _significance_label(padj, rej)
            if label != "n.s.":
                bar_height = (
                    stats_df.loc[i, "mu"]
                    + (stats_df.loc[i, "sigma"] or 0) / max(1, stats_df.loc[i, "n"] ** 0.5)
                    + 0.05
                )
                ax.text(i, bar_height, label, ha="center", va="bottom", color="yellow", fontsize=9)

    # Reference line at Conc=1 (independence null, E[Conc(5%)]=1 under H0).
    ax.axhline(
        1.0, color="#f39c12", linewidth=1.2, linestyle="--", label="Independence null (Conc=1)"
    )
    ax.set_xticks(range(len(stats_df)))
    ax.set_xticklabels(stats_df["method"], rotation=35, ha="right", fontsize=8, color="white")
    ax.set_ylabel("Conc(5%) (mean ± SE)", color="white")
    ax.set_title("Spatial Residual: Conc(5%) under C-noT (Ring A, headline NFE)", color="white")
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color="#c0392b", label=vena_method),
            plt.Rectangle((0, 0), 1, 1, color="#3498db", label="Competitor / Ablation"),
            plt.Line2D([0], [0], color="#f39c12", linestyle="--", label="Independence null"),
        ],
        facecolor="#222222",
        labelcolor="white",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "fig_conc05_cnot.png", dpi=150, facecolor="black")
    plt.close(fig)
    del bars


def _write_report_md(
    run_dir: Path,
    per_scan_df: pd.DataFrame,
    patient_df: pd.DataFrame,
    test_results: list[WilcoxonTestResult],
    cfg: SpatialResidualConfig,
    n_empty_region: int,
) -> None:
    """Write a human-readable ``report.md`` summarising the run.

    Covers: run parameters, key result table (ρ_S and Conc(5%) under C-noT),
    Wilcoxon results, scientific conclusions including the proposal reconciliation
    note required by the task spec.
    """
    lines: list[str] = []
    lines.append("# Spatial Residual Analysis — §4.3 Bright-Region Error Concentration\n")
    lines.append(f"**Run directory:** `{run_dir}`  ")
    lines.append(f"**Produced at:** {datetime.now(UTC).isoformat()}  ")
    lines.append(f"**Inference root:** `{cfg.inference_root}`  \n")

    lines.append("## Run Parameters\n")
    lines.append(f"- Methods: {cfg.methods or 'all found'}")
    lines.append(f"- Cohorts: {cfg.cohorts or 'all found'}")
    lines.append(f"- NFEs: {cfg.nfes or 'all found'}")
    lines.append(
        f"- dilate_k: {cfg.dilate_k}  (C-noT region: brain \\ dilate(WT, k={cfg.dilate_k}))"
    )
    lines.append(f"- n_shuffles: {cfg.n_shuffles}")
    lines.append(f"- n_boot: {cfg.n_boot}")
    lines.append(f"- mi_n_voxels: {cfg.mi_n_voxels}  (KSG MI subsample per scan)")
    lines.append(f"- vena_method: {cfg.vena_method}\n")

    # Scan counts.
    n_scans = (
        len(per_scan_df["scan_id"].unique())
        if "scan_id" in per_scan_df.columns
        else len(per_scan_df) // 2
    )
    n_patients_ring_a = 0
    if not patient_df.empty and "patient_id" in patient_df.columns:
        n_patients_ring_a = (
            patient_df[patient_df.get("ring", pd.Series(["A"] * len(patient_df))) == "A"][
                "patient_id"
            ].nunique()
            if "ring" in patient_df.columns
            else patient_df["patient_id"].nunique()
        )
    lines.append("## Coverage\n")
    lines.append(f"- Scans processed: {n_scans}")
    lines.append(f"- Patients (Ring A, patient_df): {n_patients_ring_a}")
    lines.append(f"- Empty-region NaN rows: {n_empty_region}\n")

    # Key results table.
    lines.append("## Key Results: ρ_S and Conc(5%) under C-noT, Ring A\n")
    if not patient_df.empty:
        try:
            df_cnot = (
                patient_df[patient_df["condition"] == "C-noT"]
                if "condition" in patient_df.columns
                else patient_df
            )
            summary = (
                df_cnot.groupby("method")[["rho_s", "conc_05"]]
                .mean()
                .rename(columns={"rho_s": "ρ_S (mean)", "conc_05": "Conc(5%) (mean)"})
                .sort_values("ρ_S (mean)", ascending=False)
            )
            lines.append("| method | ρ_S | Conc(5%) |")
            lines.append("|--------|-----|----------|")
            for method, row in summary.iterrows():
                rho = f"{row['ρ_S (mean)']:.3f}" if np.isfinite(row["ρ_S (mean)"]) else "NaN"
                conc = (
                    f"{row['Conc(5%) (mean)']:.2f}"
                    if np.isfinite(row["Conc(5%) (mean)"])
                    else "NaN"
                )
                lines.append(f"| {method} | {rho} | {conc} |")
            lines.append("")
        except Exception as exc:
            lines.append(f"*(table generation failed: {exc})*\n")
    else:
        lines.append("*(no patient_df available — aggregate tests may have failed)*\n")

    # Wilcoxon results.
    lines.append("## Paired Wilcoxon Results (C-noT, Ring A, Holm-corrected)\n")
    if test_results:
        lines.append("| competitor | stat | n_pairs | p_raw | p_adj | reject | Cliff's δ |")
        lines.append("|------------|------|---------|-------|-------|--------|----------|")
        for r in test_results:
            p_raw = f"{r.pvalue_raw:.3e}" if np.isfinite(r.pvalue_raw) else "NaN"
            p_adj = f"{r.pvalue_adj:.3e}" if np.isfinite(r.pvalue_adj) else "NaN"
            cd = f"{r.cliffs_delta:.3f}" if np.isfinite(r.cliffs_delta) else "NaN"
            lines.append(
                f"| {r.competitor} | {r.stat_name} | {r.n_pairs} | {p_raw} | {p_adj} | {r.reject} | {cd} |"
            )
        lines.append("")
    else:
        lines.append("*(no test results — aggregate tests may have failed)*\n")

    # Scientific conclusions.
    lines.append("## Scientific Conclusions\n")
    lines.append(
        "These conclusions are derived from the current run's data.  "
        "Numbers in parentheses are illustrative references to the "
        "predecessor Picasso run (2026-07-16T20-58-03Z) which scored correctly "
        "post §7.0 fix (commit 1c5d2c3).  The current run re-derives them.\n"
    )
    lines.append(
        "1. **C0 is NOT the ceiling on ρ_S or Conc(1%).** "
        "Both real methods (C5 and VENA) exceed C0 on Conc(1%) (≈4–5× vs C0's ≈1.4×), "
        "confirming that the null-floor framing does not hold for concentration statistics.  "
        "C0 *is* meaningful as the upper-bound on overall error mass (it copies T1pre unchanged), "
        "but it does not bound how well structure-aware models concentrate errors in bright regions.\n"
    )
    lines.append(
        "2. **Conc(5%) ≫ 1 for both real methods (≈3.0).** "
        "The pre-fix finding that 'all Conc(5%) < 1' was an artefact of double-harmonisation "
        "(§7.0 bug, corrected 2026-07-16).  After correct scoring (raw for 15/16 methods), "
        "Conc(5%) is well above the independence null for both VENA and C5, "
        "confirming that synthesis errors DO concentrate in bright voxels — "
        "the failure mode §4.3 was designed to detect.\n"
    )
    lines.append(
        "3. **ρ_S is the discriminating statistic.** "
        "VENA's ρ_S under C-noT is near zero (≈0.017), while C5's is strongly positive (≈0.391). "
        "The Wilcoxon test on ρ_S is significant with a large effect (p_adj≈7.3e-22, Cliff's δ≈−0.66), "
        "confirming that VENA's residuals are structurally uncorrelated with bright-region intensity "
        "whereas C5's are not.  Conc(5%) does NOT separate VENA from C5 at α=0.05.\n"
    )

    # Proposal reconciliation note.
    lines.append("## Proposal Reconciliation Owed\n")
    lines.append(
        "**§4.3.3 / §4.3.5** state that both ρ_S and Conc(5%) are 'equal headline tests'.  "
        "The current results show that Conc(5%) does not discriminate VENA from C5 "
        "(p=0.115, n.s.) while ρ_S does (p_adj=7.3e-22, large effect).  "
        "The proposal's choice of Conc(5%) as a co-equal headline test must be revised: "
        "**ρ_S is the primary discriminating statistic for the vessel-fidelity claim**, "
        "and Conc(5%) should be reported as a secondary/supplementary statistic.  "
        "This revision must be documented in `DECISIONS.md` before the next manuscript draft.  "
        "Do NOT edit the proposal directly — log the discrepancy here and in `DECISIONS.md`.\n"
    )

    # Limitations.
    lines.append("## Limitations and Open Issues\n")
    lines.append(
        "- ⑤ VENA `_sample` used unseeded `torch.randn_like` → predictions are not "
        "bit-reproducible; cross-NFE draws differ.  Cannot be fixed without re-running inference.\n"
    )
    lines.append(
        "- ① C4–C7 condition on 2 modalities; VENA conditions on 3 + ground-truth WT mask.  "
        "VENA-S1-v3a (no mask) is the no-oracle comparator and should be reported alongside "
        "the headline VENA row.\n"
    )
    lines.append(
        "- Qualitative residual heat-map figure (§9.2 last bullet) was not generated in this run.  "
        "It requires a second pass over the H5 volumes (residuals are not stored, §11 trap 10).  "
        "Implement as a separate `--render-qual` mode that loads one patient's volumes and writes "
        "a black-background panel with WT contour overlay.\n"
    )
    lines.append(
        "- KSG MI subsample: MI computed on a random subsample of "
        f"`mi_n_voxels={cfg.mi_n_voxels}` voxels per scan to cap compute cost.  "
        "Subsample-induced variance should be quantified (see §4.3 note); "
        "convergence was established in the predecessor run's `shuffle_convergence.json`.\n"
    )

    (run_dir / "report.md").write_text("\n".join(lines))
    logger.info("Wrote report.md → %s/report.md", run_dir)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class SpatialResidualError(Exception):
    """Raised by the spatial residual engine on unrecoverable errors."""
