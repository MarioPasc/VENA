"""Paired-fidelity article study — Table 1 + supplementary tables + forest figure.

Consumes the FROZEN per-scan CSV produced by the paired-fidelity sweep and
renders the paper deliverables:

* ``tables/table1_ring_a_fidelity.csv``      — 16-method × all-(metric,region) stats table
* ``tables/table1_ring_a_fidelity.md``       — tiered human-readable version
* ``tables/tableS1_undersaturation.csv``     — raw_p995 scoring-space audit
* ``tables/tableS2_zgd.csv``                 — ZGD per method
* ``figures/fig_forest_{metric}_{region}.png`` — 12 forest plots (4 metrics × 3 regions)
* ``decision.json``                          — machine-readable provenance

Statistical protocol (pre-registered, see 00_HUB.md §2.3):
* Ring A only, patient-collapsed.
* Each arm reduced to its own selection NFE via ``filter_to_selection_nfe``
  (one shared code path — asymmetric reduction is the D1 bug).
* Bootstrap CI: patient-stratified, 10 000 resamples, seed=1337.
* Effect size: Cliff's δ (never Cohen's d).
* Multiple comparisons: Holm–Bonferroni within each (metric, region) cell;
  competitor family (8) and ablation family (3) corrected SEPARATELY;
  supplementary (4) in no family.
* Comparisons: each method vs VENA-S1-v3b-rw AND vs VENA-S1-v3a.
* MCID = 0.01 on [0,1]-scale metrics (mae, ssim, ms_ssim).
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.patches as mpatches

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from routines.validation.studies._shared import (
    DOMAIN_ORDER,
    domain_of,
    filter_to_selection_nfe,
    per_scan_csv,
)
from vena.validation.artifacts import make_run_dir, symlink_latest, write_decision_json
from vena.validation.registry import (
    ABLATION_FAMILY,
    COMPETITOR_FAMILY,
    METHOD_SPECS,
    SELECTION_NFE,
    SUPPLEMENTARY,
    VENA_HEADLINE,
)
from vena.validation.stats import (
    MCID,
    HolmResult,
    bootstrap_ci,
    cliffs_delta,
    collapse_to_patient,
    holm_bonferroni,
    paired_wilcoxon,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ── Repo root (for git rev-parse) ─────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[3]

# ── Oracle flag (from HUB §2.1) ───────────────────────────────────────────────
#: Methods that receive a ground-truth WT mask unavailable to competitors.
_ORACLE_METHODS: frozenset[str] = frozenset(
    {
        "VENA-S1-v3b",
        "VENA-S1-v3b-rw",
        "VENA-S3-LPL-b2c",
    }
)

# ── n_inputs: MRI input volumes, not counting masks (from HUB §2.1 table) ─────
_N_INPUTS: dict[str, int] = {
    "C0-Identity": 1,
    "C1-pGAN-t1pre": 1,
    "C1-pGAN-t2": 1,
    "C1-pGAN-flair": 1,
    "C2-ResViT": 3,
    "C3-SynDiff-t1pre": 1,
    "C3-SynDiff-t2": 1,
    "C3-SynDiff-flair": 1,
    "C4-3D-DiT": 2,
    "C5-T1C-RFlow": 2,
    "C6-3D-LDDPM": 2,
    "C7-3D-Latent-Pix2Pix": 2,
    "VENA-S1-v3a": 3,
    "VENA-S1-v3b": 3,
    "VENA-S1-v3b-rw": 3,
    "VENA-S3-LPL-b2c": 3,
}

# ── Metric / region combos ────────────────────────────────────────────────────
_REGIONS: tuple[str, ...] = ("brain", "wt", "bg_undilated")
_METRICS: tuple[str, ...] = ("mae", "psnr", "ssim", "ms_ssim")
# Metrics on [0,1] scale where |Δ| < MCID is clinically sub-threshold.
_BOUNDED_METRICS: frozenset[str] = frozenset({"mae", "ssim", "ms_ssim"})

# ── Comparison baseline (no-oracle counterpart) ───────────────────────────────
_VENA_V3A: str = "VENA-S1-v3a"

# ── Domain colours for forest figure ─────────────────────────────────────────
_DOMAIN_COLOR: dict[str, str] = {
    "reference": "#888888",
    "pixel": "#E69F00",
    "latent": "#0072B2",
}
_VENA_HIGHLIGHT: str = "#00BCD4"
_VENA_V3A_COLOR: str = "#00A0B8"  # slightly desaturated teal for the no-oracle arm

# ── Forest figure layout constants ────────────────────────────────────────────
#: Group display order top→bottom: pixel first, then latent, reference at bottom.
_FOREST_GROUP_ORDER: tuple[str, ...] = ("pixel", "latent", "reference")
#: Metrics where lower is better (MAE). All others: higher is better.
_BETTER_LOWER: frozenset[str] = frozenset({"mae"})


def _sig_label(padj: float, *, is_ref: bool, is_supp: bool) -> str:
    """Format a Holm-adjusted p-value as a significance star label.

    Parameters
    ----------
    padj :
        Holm-adjusted p-value.
    is_ref :
        True when the method IS the reference arm (VENA-S1-v3b-rw).
    is_supp :
        True for supplementary methods that are in no Holm family.

    Returns
    -------
    str
        ``"ref"``, ``"n/t"``, ``"***"``, ``"**"``, ``"*"``, or ``"ns"``.
    """
    if is_ref:
        return "ref"
    if is_supp or np.isnan(padj):
        return "n/t"
    if padj < 0.001:
        return "***"
    if padj < 0.01:
        return "**"
    if padj < 0.05:
        return "*"
    return "ns"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _canonical_method_order() -> list[str]:
    """Return all 16 registry methods ordered: reference → pixel → latent.

    Within each domain, registry insertion order is preserved.
    """
    buckets: dict[str, list[str]] = {d: [] for d in DOMAIN_ORDER}
    for spec in METHOD_SPECS:
        d = domain_of(spec.key)
        buckets.setdefault(d, []).append(spec.key)
    order: list[str] = []
    for d in DOMAIN_ORDER:
        order.extend(buckets.get(d, []))
    return order


def _holm_family(
    raw_pvs: dict[str, dict[tuple[str, str], float]],
    methods_in_family: tuple[str, ...],
    cell: tuple[str, str],
) -> dict[str, HolmResult]:
    """Apply Holm correction over *methods_in_family* for one (metric, region) cell."""
    pvs = {m: raw_pvs[m][cell] for m in methods_in_family if m in raw_pvs and cell in raw_pvs[m]}
    if not pvs:
        return {}
    return holm_bonferroni(pvs)


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PairedFidelityStudyConfig:
    """Frozen configuration for :class:`PairedFidelityStudy`.

    Parameters
    ----------
    per_scan_csv :
        Frozen per-scan CSV from the paired-fidelity sweep.
    output_root :
        Root directory for article study artifacts.
        Run dir is ``<output_root>/paired_fidelity/<UTC>/``.
    ring :
        Ring to analyse.  ``"A"`` = internal UCSF-PDGM test set.
    n_boot :
        Bootstrap replicates for CI estimation.
    seed :
        Fixed RNG seed for reproducibility.
    """

    per_scan_csv: Path = field(
        default_factory=lambda: per_scan_csv("paired_fidelity", "paired_fidelity_patient.csv")
    )
    output_root: Path = field(
        default_factory=lambda: Path("/media/mpascual/Sandisk2TB/research/vena/results/article")
    )
    ring: str = "A"
    n_boot: int = 10_000
    seed: int = 1337

    @classmethod
    def from_yaml(cls, path: str | Path) -> PairedFidelityStudyConfig:
        """Load config from a YAML file; unset keys fall back to field defaults.

        Parameters
        ----------
        path :
            Path to the YAML config file.

        Returns
        -------
        PairedFidelityStudyConfig
            Frozen config instance.
        """
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        kwargs: dict = {}
        if "per_scan_csv" in raw:
            kwargs["per_scan_csv"] = Path(raw["per_scan_csv"])
        if "output_root" in raw:
            kwargs["output_root"] = Path(raw["output_root"])
        if "ring" in raw:
            kwargs["ring"] = str(raw["ring"])
        if "n_boot" in raw:
            kwargs["n_boot"] = int(raw["n_boot"])
        if "seed" in raw:
            kwargs["seed"] = int(raw["seed"])
        return cls(**kwargs)


# ── Engine ────────────────────────────────────────────────────────────────────


class PairedFidelityStudy:
    """Render the paired-fidelity article study from a frozen per-scan CSV.

    Parameters
    ----------
    cfg :
        Frozen study configuration.
    """

    def __init__(self, cfg: PairedFidelityStudyConfig) -> None:
        self._cfg = cfg

    def run(self) -> Path:
        """Execute the full study and return the artifact directory.

        Returns
        -------
        Path
            Timestamped run directory ``<output_root>/paired_fidelity/<UTC>/``.
            A ``LATEST`` symlink is updated to point at it.
        """
        cfg = self._cfg
        logger.info("paired_fidelity_study: reading %s", cfg.per_scan_csv)
        df_raw = pd.read_csv(cfg.per_scan_csv)

        # ── Step 1: filter ring ───────────────────────────────────────────────
        df_ring = df_raw[df_raw["ring"] == cfg.ring].copy()
        logger.info("ring=%s: %d rows", cfg.ring, len(df_ring))

        # ── Step 2: reduce to selection NFE + collapse to patient ─────────────
        method_order = _canonical_method_order()
        value_cols_all = [f"{metric}_{region}" for metric in _METRICS for region in _REGIONS] + [
            "zgd",
            "raw_p995",
        ]

        patient_dfs: dict[str, pd.DataFrame] = {}
        for method in method_order:
            mdf = df_ring[df_ring["method"] == method].copy()
            if mdf.empty:
                logger.warning("no ring-%s rows for method=%s; skipping", cfg.ring, method)
                continue
            # Symmetric selection-NFE reduction — single shared code path.
            mdf = filter_to_selection_nfe(mdf, method)
            value_cols = [c for c in value_cols_all if c in mdf.columns]
            mdf_pat = collapse_to_patient(
                mdf,
                value_cols=value_cols,
                by=("method", "cohort", "ring", "nfe", "patient_id"),
            )
            patient_dfs[method] = mdf_pat
            logger.debug(
                "  %s → nfe=%s, %d patients",
                method,
                mdf_pat["nfe"].iloc[0] if "nfe" in mdf_pat.columns else "?",
                mdf_pat["patient_id"].nunique(),
            )

        if VENA_HEADLINE not in patient_dfs:
            raise RuntimeError(
                f"VENA headline {VENA_HEADLINE!r} absent from ring {cfg.ring!r} data"
            )

        n_patients = patient_dfs[VENA_HEADLINE]["patient_id"].nunique()
        logger.info("n_patients (Ring %s, %s): %d", cfg.ring, VENA_HEADLINE, n_patients)

        selection_nfe_used = {
            m: int(patient_dfs[m]["nfe"].iloc[0])
            for m in method_order
            if m in patient_dfs and "nfe" in patient_dfs[m].columns
        }

        # ── Step 3: compute statistics ────────────────────────────────────────
        vena_df = patient_dfs[VENA_HEADLINE].set_index("patient_id")
        v3a_df = (
            patient_dfs[_VENA_V3A].set_index("patient_id") if _VENA_V3A in patient_dfs else None
        )

        # boot_stats[method][(metric, region)] → {"mean", "ci_lo", "ci_hi"}
        boot_stats: dict[str, dict[tuple[str, str], dict[str, float]]] = {}
        # raw_pvs_vs_*[method][(metric, region)] → raw p-value
        raw_pvs_vs_v3brw: dict[str, dict[tuple[str, str], float]] = {}
        raw_pvs_vs_v3a: dict[str, dict[tuple[str, str], float]] = {}
        # delta_vs_*[method][(metric, region)] → Cliff's δ
        delta_vs_v3brw: dict[str, dict[tuple[str, str], float]] = {}
        delta_vs_v3a: dict[str, dict[tuple[str, str], float]] = {}

        for method, mdf_pat in patient_dfs.items():
            mdf_idx = mdf_pat.set_index("patient_id")
            cohort_map = mdf_idx["cohort"] if "cohort" in mdf_idx.columns else None
            boot_stats[method] = {}
            raw_pvs_vs_v3brw[method] = {}
            raw_pvs_vs_v3a[method] = {}
            delta_vs_v3brw[method] = {}
            delta_vs_v3a[method] = {}

            for metric in _METRICS:
                for region in _REGIONS:
                    col = f"{metric}_{region}"
                    if col not in mdf_idx.columns:
                        continue
                    vals = mdf_idx[col].dropna()
                    if vals.empty:
                        continue

                    strata = (
                        cohort_map.loc[vals.index].to_numpy() if cohort_map is not None else None
                    )
                    ci_lo, ci_hi = bootstrap_ci(
                        vals.to_numpy(),
                        n_boot=cfg.n_boot,
                        ci=0.95,
                        strata=strata,
                        seed=cfg.seed,
                    )
                    boot_stats[method][(metric, region)] = {
                        "mean": float(vals.mean()),
                        "ci_lo": float(ci_lo),
                        "ci_hi": float(ci_hi),
                    }

                    cell: tuple[str, str] = (metric, region)

                    # vs v3b-rw
                    if method != VENA_HEADLINE and col in vena_df.columns:
                        wr = paired_wilcoxon(vena_df[col], mdf_idx[col])
                        common = vena_df[col].index.intersection(mdf_idx[col].index)
                        cd = cliffs_delta(
                            mdf_idx[col].loc[common].to_numpy(),
                            vena_df[col].loc[common].to_numpy(),
                        )
                        raw_pvs_vs_v3brw[method][cell] = wr.pvalue
                        delta_vs_v3brw[method][cell] = cd

                    # vs v3a
                    if method != _VENA_V3A and v3a_df is not None and col in v3a_df.columns:
                        wr_a = paired_wilcoxon(v3a_df[col], mdf_idx[col])
                        common_a = v3a_df[col].index.intersection(mdf_idx[col].index)
                        cd_a = cliffs_delta(
                            mdf_idx[col].loc[common_a].to_numpy(),
                            v3a_df[col].loc[common_a].to_numpy(),
                        )
                        raw_pvs_vs_v3a[method][cell] = wr_a.pvalue
                        delta_vs_v3a[method][cell] = cd_a

        # ── Step 4: Holm correction per (metric, region) × family ─────────────
        # Maps: padj_vs_*[method][(metric, region)] → HolmResult | None
        padj_vs_v3brw: dict[str, dict[tuple[str, str], HolmResult]] = {m: {} for m in method_order}
        padj_vs_v3a: dict[str, dict[tuple[str, str], HolmResult]] = {m: {} for m in method_order}

        # Ablation family vs v3a: exclude v3a itself (comparing against itself).
        abl_family_vs_v3a = tuple(m for m in ABLATION_FAMILY if m != _VENA_V3A)

        for metric in _METRICS:
            for region in _REGIONS:
                cell = (metric, region)

                # vs v3b-rw: competitor family (8) + ablation family (3)
                comp_holm = _holm_family(raw_pvs_vs_v3brw, COMPETITOR_FAMILY, cell)
                abl_holm = _holm_family(raw_pvs_vs_v3brw, ABLATION_FAMILY, cell)
                for m, hr in {**comp_holm, **abl_holm}.items():
                    padj_vs_v3brw[m][cell] = hr

                # vs v3a: competitor family (8) + ablation family minus v3a (2)
                comp_holm_a = _holm_family(raw_pvs_vs_v3a, COMPETITOR_FAMILY, cell)
                abl_holm_a = _holm_family(raw_pvs_vs_v3a, abl_family_vs_v3a, cell)
                for m, hr in {**comp_holm_a, **abl_holm_a}.items():
                    padj_vs_v3a[m][cell] = hr

        # ── Step 5: assemble table1 DataFrame ────────────────────────────────
        spec_map = {s.key: s for s in METHOD_SPECS}
        vena_boot = boot_stats.get(VENA_HEADLINE, {})

        rows: list[dict] = []
        for method in method_order:
            if method not in patient_dfs:
                logger.warning("method %s absent from data; row will be empty", method)
            spec = spec_map.get(method)
            row: dict[str, object] = {
                "method": method,
                "display": spec.display if spec else method,
                "domain": domain_of(method),
                "tier": spec.tier if spec else "unknown",
                "role": spec.role if spec else "unknown",
                "selection_nfe": SELECTION_NFE.get(method, -1),
                "is_oracle": method in _ORACLE_METHODS,
                "n_inputs": _N_INPUTS.get(method, -1),
            }

            for metric in _METRICS:
                for region in _REGIONS:
                    col = f"{metric}_{region}"
                    cell = (metric, region)
                    bs = boot_stats.get(method, {}).get(cell, {})
                    row[f"{col}_mean"] = bs.get("mean", float("nan"))
                    row[f"{col}_ci_lo"] = bs.get("ci_lo", float("nan"))
                    row[f"{col}_ci_hi"] = bs.get("ci_hi", float("nan"))

                    # vs v3b-rw
                    row[f"{col}_p_vs_v3brw"] = raw_pvs_vs_v3brw.get(method, {}).get(
                        cell, float("nan")
                    )
                    hr_v = padj_vs_v3brw.get(method, {}).get(cell)
                    row[f"{col}_padj_vs_v3brw"] = (
                        hr_v.pvalue_adj if hr_v is not None else float("nan")
                    )
                    row[f"{col}_reject_vs_v3brw"] = bool(hr_v.reject) if hr_v is not None else False
                    row[f"{col}_cliffs_vs_v3brw"] = delta_vs_v3brw.get(method, {}).get(
                        cell, float("nan")
                    )

                    # vs v3a
                    row[f"{col}_p_vs_v3a"] = raw_pvs_vs_v3a.get(method, {}).get(cell, float("nan"))
                    hr_a = padj_vs_v3a.get(method, {}).get(cell)
                    row[f"{col}_padj_vs_v3a"] = (
                        hr_a.pvalue_adj if hr_a is not None else float("nan")
                    )
                    row[f"{col}_reject_vs_v3a"] = bool(hr_a.reject) if hr_a is not None else False
                    row[f"{col}_cliffs_vs_v3a"] = delta_vs_v3a.get(method, {}).get(
                        cell, float("nan")
                    )

                    # Sub-MCID flag (only for bounded [0,1]-scale metrics)
                    if metric in _BOUNDED_METRICS:
                        v3brw_mean = vena_boot.get(cell, {}).get("mean", float("nan"))
                        m_mean = bs.get("mean", float("nan"))
                        row[f"submcid_{col}"] = (
                            bool(abs(m_mean - v3brw_mean) < MCID)
                            if not (np.isnan(m_mean) or np.isnan(v3brw_mean))
                            else False
                        )

            rows.append(row)

        table1_df = pd.DataFrame(rows)

        # ── Step 6: write artifacts ───────────────────────────────────────────
        run_dir = make_run_dir(cfg.output_root, "paired_fidelity")

        # table1 CSV
        t1_csv = run_dir / "tables" / "table1_ring_a_fidelity.csv"
        table1_df.to_csv(t1_csv, index=False)
        logger.info("wrote %s", t1_csv)

        # table1 markdown
        t1_md = run_dir / "tables" / "table1_ring_a_fidelity.md"
        self._write_markdown_table(table1_df, t1_md, n_patients)
        logger.info("wrote %s", t1_md)

        # tableS1 — undersaturation / scoring-space audit
        s1_rows: list[dict] = []
        for method in method_order:
            if method not in patient_dfs:
                continue
            mdf = patient_dfs[method]
            p995_mean = float(mdf["raw_p995"].mean()) if "raw_p995" in mdf.columns else float("nan")
            s1_rows.append(
                {
                    "method": method,
                    "raw_p995_mean": p995_mean,
                    "scored_space": "harmonised" if p995_mean > 1.05 else "raw",
                }
            )
        ts1_csv = run_dir / "tables" / "tableS1_undersaturation.csv"
        pd.DataFrame(s1_rows).to_csv(ts1_csv, index=False)
        logger.info("wrote %s", ts1_csv)

        # tableS2 — ZGD
        s2_rows: list[dict] = []
        for method in method_order:
            if method not in patient_dfs:
                continue
            mdf = patient_dfs[method]
            if "zgd" not in mdf.columns:
                s2_rows.append(
                    {
                        "method": method,
                        "zgd_mean": float("nan"),
                        "zgd_ci_lo": float("nan"),
                        "zgd_ci_hi": float("nan"),
                    }
                )
                continue
            zgd_vals = mdf["zgd"].dropna().to_numpy()
            if len(zgd_vals) == 0:
                s2_rows.append(
                    {
                        "method": method,
                        "zgd_mean": float("nan"),
                        "zgd_ci_lo": float("nan"),
                        "zgd_ci_hi": float("nan"),
                    }
                )
                continue
            strata = (
                mdf.loc[mdf["zgd"].notna(), "cohort"].to_numpy()
                if "cohort" in mdf.columns
                else None
            )
            zi_lo, zi_hi = bootstrap_ci(
                zgd_vals, n_boot=cfg.n_boot, ci=0.95, strata=strata, seed=cfg.seed
            )
            s2_rows.append(
                {
                    "method": method,
                    "zgd_mean": float(zgd_vals.mean()),
                    "zgd_ci_lo": float(zi_lo),
                    "zgd_ci_hi": float(zi_hi),
                }
            )
        ts2_csv = run_dir / "tables" / "tableS2_zgd.csv"
        pd.DataFrame(s2_rows).to_csv(ts2_csv, index=False)
        logger.info("wrote %s", ts2_csv)

        # Forest figures (12 = 4 metrics × 3 regions)
        fig_paths = self._forest_figures(table1_df, run_dir, n_patients)
        logger.info("wrote %d forest figures → %s/figures/", len(fig_paths), run_dir)

        # decision.json
        try:
            git_sha = (
                subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],
                    cwd=_REPO_ROOT,
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
        except subprocess.CalledProcessError:
            git_sha = "unknown"

        payload = {
            "schema_version": "1.0",
            "produced_at": datetime.now(tz=UTC).isoformat(),
            "git_sha": git_sha,
            "source_csv": str(cfg.per_scan_csv.resolve()),
            "ring": cfg.ring,
            "n_patients": n_patients,
            "selection_nfe": selection_nfe_used,
            "competitor_family": list(COMPETITOR_FAMILY),
            "ablation_family": list(ABLATION_FAMILY),
        }
        write_decision_json(run_dir, payload)

        symlink_latest(run_dir)
        logger.info("paired_fidelity_study done → %s", run_dir)
        return run_dir

    # ── Private rendering helpers ─────────────────────────────────────────────

    def _write_markdown_table(
        self,
        df: pd.DataFrame,
        out_path: Path,
        n_patients: int,
    ) -> None:
        """Write a tiered, human-readable markdown table.

        Best-per-domain method (lowest MAE_brain) is **bolded**.
        Supplementary methods are included but not bolded separately.

        Parameters
        ----------
        df :
            table1 DataFrame with all methods and columns.
        out_path :
            Destination ``.md`` file.
        n_patients :
            Ring-A patient count for the footnote.
        """

        def _fmt_ci(row: pd.Series, col: str) -> str:
            m = row.get(f"{col}_mean", float("nan"))
            lo = row.get(f"{col}_ci_lo", float("nan"))
            hi = row.get(f"{col}_ci_hi", float("nan"))
            if np.isnan(m):
                return "—"
            prec = 4 if col.startswith(("mae", "ssim", "ms_ssim")) else 2
            return f"{m:.{prec}f} [{lo:.{prec}f}, {hi:.{prec}f}]"

        def _fmt_p(row: pd.Series, col: str, suffix: str) -> str:
            padj = row.get(f"{col}_padj_{suffix}", float("nan"))
            rej = row.get(f"{col}_reject_{suffix}", False)
            if np.isnan(padj):
                return "—"
            star = "✓" if rej else ""
            return f"{padj:.3f}{star}"

        # Identify best-per-domain (min MAE_brain, within methods present)
        best_by_domain: dict[str, str] = {}
        for domain in DOMAIN_ORDER:
            domain_rows = df[df["domain"] == domain]
            if domain_rows.empty:
                continue
            valid = domain_rows.dropna(subset=["mae_brain_mean"])
            if valid.empty:
                continue
            best_by_domain[domain] = str(valid.loc[valid["mae_brain_mean"].idxmin(), "method"])

        lines: list[str] = []
        lines.append(f"# Paired fidelity — Ring A (N={n_patients} patients)\n")
        lines.append(
            "> **Oracle note:** VENA-S1-v3b-rw, VENA-S1-v3b, and VENA-S3-LPL-b2c "
            "receive the ground-truth WT mask; no competitor receives this.  \n"
            "> **MCID = 0.01** on [0,1]-scale metrics (MAE, SSIM, MS-SSIM).  \n"
            "> ✓ = Holm-adjusted *p* < 0.05 vs reference arm.\n\n"
        )

        header = (
            "| Method | oracle | n_in | "
            "MAE_brain | PSNR_brain | SSIM_brain | "
            "MAE_wt | "
            "p_adj(MAE_b) vs v3b-rw | p_adj(MAE_b) vs v3a |"
        )
        sep = "|---|:---:|:---:|---|---|---|---|---|---|"

        for domain in DOMAIN_ORDER:
            domain_rows = df[df["domain"] == domain]
            if domain_rows.empty:
                continue
            lines.append(f"### {domain.capitalize()}\n")
            lines.append(header)
            lines.append(sep)

            for _, row in domain_rows.iterrows():
                method = str(row["method"])
                display = str(row.get("display", method))
                is_best = best_by_domain.get(str(row["domain"])) == method
                label = f"**{display}**" if is_best else display

                oracle_flag = "GT⊕" if bool(row.get("is_oracle", False)) else ""
                n_in = str(int(row["n_inputs"])) if not np.isnan(row["n_inputs"]) else "?"

                cols_row = [
                    label,
                    oracle_flag,
                    n_in,
                    _fmt_ci(row, "mae_brain"),
                    _fmt_ci(row, "psnr_brain"),
                    _fmt_ci(row, "ssim_brain"),
                    _fmt_ci(row, "mae_wt"),
                    _fmt_p(row, "mae_brain", "v3brw"),
                    _fmt_p(row, "mae_brain", "v3a"),
                ]
                lines.append("| " + " | ".join(cols_row) + " |")
            lines.append("")

        # NFE footnote
        nfe_parts = [
            f"{m}={v}" for m, v in sorted({s.key: s.selection_nfe for s in METHOD_SPECS}.items())
        ]
        lines.append(
            f"\n*Selection NFE per method: {', '.join(nfe_parts)}.*\n"
            f"*Full statistics (all regions × metrics, Cliff's δ, CI bounds) "
            f"in `table1_ring_a_fidelity.csv`.*"
        )

        out_path.write_text("\n".join(lines))

    def _forest_figures(
        self,
        table1_df: pd.DataFrame,
        run_dir: Path,
        n_patients: int,
    ) -> list[Path]:
        """Render one forest figure per (metric, region) combination → 12 PNGs.

        All statistics are read from *table1_df* columns already computed in
        ``run()`` — no recomputation.

        Parameters
        ----------
        table1_df :
            Full table-1 DataFrame (all methods, all stats columns).
        run_dir :
            Artifact run directory; figures go to ``run_dir/figures/``.
        n_patients :
            Patient count for the figure title.

        Returns
        -------
        list[Path]
            Paths of the 12 written PNGs, in (metric, region) iteration order.
        """
        paths: list[Path] = []
        for metric in _METRICS:
            for region in _REGIONS:
                out_path = run_dir / "figures" / f"fig_forest_{metric}_{region}.png"
                self._draw_one_forest(table1_df, metric, region, out_path, n_patients)
                paths.append(out_path)
        return paths

    def _draw_one_forest(
        self,
        df: pd.DataFrame,
        metric: str,
        region: str,
        out_path: Path,
        n_patients: int,
    ) -> None:
        """Render a single horizontal forest plot for one (metric, region) cell.

        Layout rules
        ------------
        * Groups (top→bottom): pixel → latent → reference.
        * Within each group, methods are sorted by **descending performance** for
          this metric (lowest MAE first; highest PSNR/SSIM/MS-SSIM first).
        * White figure and axes background (``facecolor='white'``).
        * Point = mean; whiskers = 95% CI from pre-computed ``_ci_lo/_ci_hi``
          columns in *df* — no recomputation.
        * Two vertical reference lines:
          - grey dashed: C0-Identity mean (floor baseline).
          - teal dotted: VENA-S1-v3b-rw mean (headline).
        * Significance stars (vs v3b-rw, Holm-adjusted, from ``_padj_vs_v3brw``):
          ``***`` p<0.001, ``**`` p<0.01, ``*`` p<0.05, ``ns``, ``ref``, ``n/t``.
        * VENA methods highlighted: v3b-rw (diamond, teal ◆) and v3a (triangle, ▲).
        * Title carries N and MCID.

        Parameters
        ----------
        df :
            table1 DataFrame with all pre-computed stat columns.
        metric :
            One of ``{"mae", "psnr", "ssim", "ms_ssim"}``.
        region :
            One of ``{"brain", "wt", "bg_undilated"}``.
        out_path :
            Destination PNG file.
        n_patients :
            Patient count for the title caption.
        """
        col_mean = f"{metric}_{region}_mean"
        col_lo = f"{metric}_{region}_ci_lo"
        col_hi = f"{metric}_{region}_ci_hi"
        col_padj = f"{metric}_{region}_padj_vs_v3brw"
        better_lower = metric in _BETTER_LOWER
        supp_set = set(SUPPLEMENTARY)
        spec_map = {s.key: s for s in METHOD_SPECS}

        # ── Build per-group sorted method lists ───────────────────────────────
        group_lists: dict[str, list[str]] = {}
        for group in _FOREST_GROUP_ORDER:
            gdf = df[df["domain"] == group].copy()
            gdf = gdf.dropna(subset=[col_mean])
            # Sort: best (lowest mae / highest others) → first entry = top of group.
            gdf = gdf.sort_values(col_mean, ascending=better_lower)
            group_lists[group] = gdf["method"].tolist()

        # Flatten to global order (pixel first → latent → reference).
        ordered: list[str] = []
        for group in _FOREST_GROUP_ORDER:
            ordered.extend(group_lists.get(group, []))

        n = len(ordered)
        if n == 0:
            return

        # y assignment: index 0 (best pixel) → y = n-1 (top); index n-1 → y = 0 (bottom).
        y_of: dict[str, int] = {m: n - 1 - i for i, m in enumerate(ordered)}

        # ── Collect per-method data from table1_df ────────────────────────────
        method_data: dict[str, dict[str, float | str]] = {}
        valid_los: list[float] = []
        valid_his: list[float] = []
        valid_means: list[float] = []

        for m in ordered:
            rows = df[df["method"] == m]
            if rows.empty:
                continue
            r = rows.iloc[0]
            mean = (
                float(r[col_mean])
                if col_mean in r.index and not pd.isna(r[col_mean])
                else float("nan")
            )
            lo = float(r[col_lo]) if col_lo in r.index and not pd.isna(r[col_lo]) else float("nan")
            hi = float(r[col_hi]) if col_hi in r.index and not pd.isna(r[col_hi]) else float("nan")
            padj = (
                float(r[col_padj])
                if col_padj in r.index and not pd.isna(r[col_padj])
                else float("nan")
            )
            domain = str(r.get("domain", "latent"))
            method_data[m] = {"mean": mean, "lo": lo, "hi": hi, "padj": padj, "domain": domain}
            if not np.isnan(mean):
                valid_means.append(mean)
            if not np.isnan(lo):
                valid_los.append(lo)
            if not np.isnan(hi):
                valid_his.append(hi)

        if not valid_means:
            return

        # ── Axis limits (leave room right for star annotations) ───────────────
        x_lo_bound = min(valid_los) if valid_los else min(valid_means)
        x_hi_bound = max(valid_his) if valid_his else max(valid_means)
        x_span = max(x_hi_bound - x_lo_bound, 1e-6)
        x_annot = x_hi_bound + 0.05 * x_span
        x_right = x_hi_bound + 0.28 * x_span

        # ── Figure setup ──────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(9.5, max(5.0, n * 0.44)), facecolor="white")
        ax.set_facecolor("white")

        # ── Group shading and dividers ────────────────────────────────────────
        _group_bg = {"pixel": "#FFF8E7", "latent": "#E8F0FA", "reference": "#F5F5F5"}
        for group in _FOREST_GROUP_ORDER:
            gms = group_lists.get(group, [])
            if not gms:
                continue
            y_top = y_of[gms[0]] + 0.5
            y_bot = y_of[gms[-1]] - 0.5
            ax.axhspan(y_bot, y_top, color=_group_bg.get(group, "#FFFFFF"), alpha=1.0, zorder=0)
            # Divider below group (except last)
            if group != _FOREST_GROUP_ORDER[-1]:
                ax.axhline(y=y_bot, color="#cccccc", linewidth=0.9, zorder=1)

        # ── Reference lines ───────────────────────────────────────────────────
        c0_data = method_data.get("C0-Identity", {})
        c0_mean_val = float(c0_data.get("mean", float("nan")))
        if not np.isnan(c0_mean_val):
            ax.axvline(
                c0_mean_val,
                color="#888888",
                linewidth=1.0,
                linestyle="--",
                zorder=1,
                label=f"C0 ({c0_mean_val:.4f})",
            )

        v3brw_data = method_data.get(VENA_HEADLINE, {})
        v3brw_mean_val = float(v3brw_data.get("mean", float("nan")))
        if not np.isnan(v3brw_mean_val):
            ax.axvline(
                v3brw_mean_val,
                color=_VENA_HIGHLIGHT,
                linewidth=1.1,
                linestyle=":",
                zorder=1,
                label=f"v3b-rw ({v3brw_mean_val:.4f})",
            )

        # ── Error bars and significance stars ─────────────────────────────────
        for m in ordered:
            if m not in method_data:
                continue
            d = method_data[m]
            mean, lo, hi = float(d["mean"]), float(d["lo"]), float(d["hi"])
            padj = float(d["padj"])
            domain = str(d["domain"])
            y = y_of[m]

            # Style
            is_headline = m == VENA_HEADLINE
            is_v3a = m == _VENA_V3A
            if is_headline:
                color, mkr, ms, lw = _VENA_HIGHLIGHT, "D", 9, 1.5
            elif is_v3a:
                color, mkr, ms, lw = _VENA_V3A_COLOR, "^", 8, 1.3
            elif domain == "reference":
                color, mkr, ms, lw = _DOMAIN_COLOR["reference"], "s", 7, 1.1
            else:
                color, mkr, ms, lw = _DOMAIN_COLOR.get(domain, "#444444"), "o", 6, 1.1

            if not (np.isnan(mean) or np.isnan(lo) or np.isnan(hi)):
                ax.errorbar(
                    mean,
                    y,
                    xerr=[[mean - lo], [hi - mean]],
                    fmt=mkr,
                    color=color,
                    markersize=ms,
                    capsize=3,
                    linewidth=lw,
                    elinewidth=0.9,
                    zorder=3,
                )

            # Significance star
            is_supp = m in supp_set
            star = _sig_label(padj, is_ref=is_headline, is_supp=is_supp)
            star_color = "#CC0000" if star in {"***", "**", "*"} else "#888888"
            if not np.isnan(hi):
                ax.text(
                    x_annot,
                    y,
                    star,
                    va="center",
                    ha="left",
                    fontsize=7.5,
                    color=star_color,
                    zorder=5,
                )

        # ── Y-axis tick labels ─────────────────────────────────────────────────
        y_ticks = [y_of[m] for m in ordered]
        y_labels: list[str] = []
        for m in ordered:
            label = spec_map[m].display if m in spec_map else m
            if m == VENA_HEADLINE:
                label = f"► {label}"
            elif m == _VENA_V3A:
                label = f"▷ {label}"
            y_labels.append(label)

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=8)
        # Bold the VENA highlight labels
        for tick_label, m in zip(ax.get_yticklabels(), ordered, strict=False):
            if m in {VENA_HEADLINE, _VENA_V3A}:
                tick_label.set_fontweight("bold")
                tick_label.set_color(_VENA_HIGHLIGHT if m == VENA_HEADLINE else _VENA_V3A_COLOR)

        # ── X-axis and title ──────────────────────────────────────────────────
        direction = "← lower is better" if better_lower else "→ higher is better"
        ax.set_xlabel(
            f"{metric.upper()} | {region}  [{direction}]  (95% CI, N={n_patients})",
            fontsize=9,
        )
        ax.set_title(
            f"Paired fidelity — {metric.upper()}_{region} — Ring A (N={n_patients}, MCID=0.01)",
            fontsize=9.5,
            pad=8,
        )

        # ── Legend ────────────────────────────────────────────────────────────
        domain_patches = [
            mpatches.Patch(color=_DOMAIN_COLOR[g], label=g.capitalize())
            for g in _FOREST_GROUP_ORDER
            if g in _DOMAIN_COLOR
        ]
        vena_patch = mpatches.Patch(color=_VENA_HIGHLIGHT, label="VENA (v3b-rw, v3a)")
        handles_extra, _ = ax.get_legend_handles_labels()
        ax.legend(
            handles=[*domain_patches, vena_patch, *handles_extra],
            fontsize=7,
            loc="best",
            framealpha=0.85,
            edgecolor="#cccccc",
        )

        ax.set_xlim(x_lo_bound - 0.05 * x_span, x_right)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
