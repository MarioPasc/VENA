"""Aggregate shard CSVs → consolidated tables, figures, decision.json, report.

The per-shard engine writes partial CSVs into
``<artifact_dir>/shard_{i}/`` (the engine's :meth:`run` step). The
aggregation step concatenates them, computes the §4.7 gates, and
emits the final deliverables.

The function is designed to run standalone after the four loginexa
shards finish (the cli's ``aggregate`` subcommand), but it can also
run in-process at the end of a single-shard smoke.
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from vena.data.h5.shared import now_iso_utc

from .decision import (
    DECISION_PRODUCER,
    DECISION_SCHEMA_VERSION,
    DecoderLplDecisionV1,
    write_decision_json,
)
from .figures import (
    channel_concentration_block2_vs_block5,
    drift_heatmap,
    inter_cohort_ratio_box,
    magnitude_curve,
    separation_per_region,
    t_min_knee,
)
from .phase3_drift import (
    decide_allowed_variants,
    detect_v4_brain_mask_status,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# CSV plumbing
# --------------------------------------------------------------------------


_INT_COLS: frozenset[str] = frozenset(
    {"block_idx", "channel_idx", "n_patients", "n_total", "n_empty_wt"}
)
_FLOAT_COLS: frozenset[str] = frozenset(
    {
        "mean_norm",
        "std_norm",
        "p99_norm",
        "mean_L_dec",
        "p99_L_dec",
        "mad",
        "mad_median",
        "recommended_k",
        "sep_dist",
        "residual_dist",
        "feature_distance_to_target",
        "t",
        "ratio_variant",
        "ratio_v0",
        "drift_value",
        "ratio_median",
        "ratio_p25",
        "ratio_p75",
        "fraction",
    }
)
_BOOL_COLS: frozenset[str] = frozenset({"passes_gate", "exceeds_global_recipe"})


def _coerce_row(row: dict) -> dict:
    """Coerce CSV-string values to numeric / bool per a fixed column map."""
    out: dict = {}
    for k, v in row.items():
        if k in _INT_COLS:
            try:
                out[k] = int(v)
            except (TypeError, ValueError):
                out[k] = v
        elif k in _FLOAT_COLS:
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                out[k] = v
        elif k in _BOOL_COLS:
            # csv.DictReader returns "True" / "False" — both truthy as raw str.
            out[k] = str(v).strip().lower() in {"true", "1"}
        else:
            out[k] = v
    return out


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return [_coerce_row(r) for r in csv.DictReader(f)]


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _concat_shards(artifact_dir: Path, filename: str) -> list[dict]:
    """Concatenate ``shard_{i}/<filename>.csv`` files into one list-of-dicts."""
    out: list[dict] = []
    for shard_dir in sorted(artifact_dir.glob("shard_*")):
        out.extend(_read_csv(shard_dir / "tables" / filename))
    # The single-shard smoke writes directly under ``tables/`` — pick up
    # that case too so the aggregator works without a shard prefix.
    direct = artifact_dir / "tables" / filename
    if direct.exists() and not out:
        out.extend(_read_csv(direct))
    return out


# --------------------------------------------------------------------------
# Decision-derivation helpers
# --------------------------------------------------------------------------


def _recommend_A_from_error_concentration(
    error_rows: list[dict],
    *,
    candidate_blocks: tuple[int, ...] = (0, 1, 2, 3, 4, 5),
    top_k: int = 2,
) -> list[int]:
    """Pick the ``top_k`` blocks whose mean per-WT residual is largest."""
    per_block: dict[int, list[float]] = defaultdict(list)
    for r in error_rows:
        if r.get("region") != "WT":
            continue
        try:
            blk = int(r["block_idx"])
            per_block[blk].append(float(r["residual_dist"]))
        except (KeyError, ValueError):
            continue
    if not per_block:
        # Fallback when phase 2 didn't run (smoke).
        return [2, 5]
    candidates = [
        (blk, float(np.mean(vals))) for blk, vals in per_block.items() if blk in candidate_blocks
    ]
    candidates.sort(key=lambda x: -x[1])
    return sorted([blk for blk, _ in candidates[:top_k]])


def _w_l_from_magnitude(
    magnitude_rows: list[dict],
    *,
    A: list[int],
) -> dict[int, float]:
    """Normalise per-block mean magnitude so ``sum(w_l) = len(A)``."""
    per_block: dict[int, list[float]] = defaultdict(list)
    for r in magnitude_rows:
        try:
            blk = int(r["block_idx"])
            per_block[blk].append(float(r["mean_norm"]))
        except (KeyError, ValueError):
            continue
    if not per_block:
        return {b: 1.0 for b in A}
    means = {b: float(np.mean(per_block.get(b, [1.0]))) for b in A}
    s = sum(means.values()) or float(len(A))
    return {b: float(means[b] * len(A) / s) for b in A}


def _outlier_k_from_distribution(
    per_channel_rows: list[dict],
    *,
    A: list[int],
    default_k: float = 5.0,
    heavy_tail_threshold: float = 10.0,
) -> dict[int, float]:
    """Per-block ``k`` from the median per-channel ``p99 / MAD`` ratio."""
    per_block_ratios: dict[int, list[float]] = defaultdict(list)
    for r in per_channel_rows:
        try:
            blk = int(r["block_idx"])
            p99 = float(r["p99_L_dec"])
            mad = float(r["mad"])
        except (KeyError, ValueError):
            continue
        if mad > 0:
            per_block_ratios[blk].append(p99 / mad)
    out: dict[int, float] = {}
    for blk in A:
        ratios = per_block_ratios.get(blk, [])
        if not ratios:
            out[blk] = float(default_k)
            continue
        median_ratio = float(np.median(ratios))
        if median_ratio > heavy_tail_threshold:
            out[blk] = float(min(default_k * (median_ratio / heavy_tail_threshold), 10.0))
        else:
            out[blk] = float(default_k)
    return out


def _t_min_from_reliability(reliability_rows: list[dict]) -> float:
    """Knee of the per-t mean-of-block reliability curve."""
    per_t_blocks: dict[float, list[float]] = defaultdict(list)
    for r in reliability_rows:
        try:
            t = float(r["t"])
            d = float(r["feature_distance_to_target"])
        except (KeyError, ValueError):
            continue
        per_t_blocks[t].append(d)
    if len(per_t_blocks) < 3:
        return 0.7
    ts = sorted(per_t_blocks)
    means = np.array([float(np.mean(per_t_blocks[t])) for t in ts])
    if means.size >= 3:
        means = np.convolve(means, np.ones(3) / 3.0, mode="same")
    curvature = np.diff(means, n=2)
    knee_idx = int(np.argmin(curvature)) + 1
    return float(ts[knee_idx])


def _region_recipe_from_drift(
    ratio_rows: list[dict],
    *,
    threshold: float = 1.5,
) -> tuple[float, float, dict[str, dict[str, float]] | None]:
    """Build the production alpha recipe.

    Default ``(alpha_wt=2.0, alpha_notwt=3.0)`` per §4.7c. When the
    inter-cohort spread of median W/nW ratios exceeds ``threshold``,
    emit per-cohort overrides — each cohort gets a scaled alpha pair so
    the effective region-weighted budget tracks its own anatomy.
    """
    cohort_medians = {}
    for r in ratio_rows:
        try:
            cohort_medians[r["cohort"]] = float(r["ratio_median"])
        except (KeyError, ValueError):
            continue
    if not cohort_medians:
        return 2.0, 3.0, None
    medians = np.array(list(cohort_medians.values()))
    spread = float(medians.max() / max(medians.min(), 1e-12))
    if spread <= threshold:
        return 2.0, 3.0, None
    # Per-cohort: scale alpha_notwt by (anchor / cohort_ratio); cohorts with
    # smaller W/nW ratio (more enhancement outside WT) get a heavier notWT
    # share, matching the §2.6 intent ("vessels live outside WT").
    anchor = float(np.median(medians))
    overrides: dict[str, dict[str, float]] = {}
    for cohort, ratio in cohort_medians.items():
        scale = float(anchor / max(ratio, 1e-12))
        overrides[cohort] = {"alpha_wt": 2.0, "alpha_notwt": float(3.0 * scale)}
    return 2.0, 3.0, overrides


# --------------------------------------------------------------------------
# Report writer
# --------------------------------------------------------------------------


def _render_report(
    out_dir: Path,
    decision: DecoderLplDecisionV1,
    *,
    figure_paths: dict[str, Path],
) -> None:
    """Write ``report.md`` with the decision summary + every figure embedded."""
    lines: list[str] = []
    lines.append("# decoder_lpl_profile — preflight report")
    lines.append("")
    lines.append(f"- Produced at: {decision.produced_at}")
    lines.append(f"- Producer: `{decision.producer}`")
    lines.append(f"- Schema: `{decision.schema_version}`")
    lines.append(f"- Patients run: {decision.n_patients_run}")
    lines.append("")
    lines.append("## Recipe pinned for S3 (decision.json keys)")
    lines.append("")
    lines.append(f"- `A_recommended` = `{decision.A_recommended}`")
    lines.append(f"- `w_l` = `{decision.w_l}`")
    lines.append(f"- `t_min` = `{decision.t_min:.3f}`")
    lines.append(f"- `outlier_k` = `{decision.outlier_k}`")
    lines.append(
        f"- `region_recipe` = `alpha_wt={decision.region_recipe.alpha_wt}, "
        f"alpha_notwt={decision.region_recipe.alpha_notwt}, "
        f"soft={decision.region_recipe.soft_region}, "
        f"overrides={decision.region_recipe.per_cohort_overrides}`"
    )
    lines.append(f"- `allowed_variants` = `{decision.allowed_variants}`")
    lines.append(f"- `v4_brain_mask_status` = `{decision.v4_brain_mask_status}`")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for name, path in figure_paths.items():
        rel = path.relative_to(out_dir) if path.is_relative_to(out_dir) else path
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"![{name}]({rel})")
        lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines))


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def aggregate(
    out_dir: Path,
    *,
    cohorts: list[str],
    variants: tuple[str, ...] = ("v0", "v1", "v2", "v3", "v4"),
    soft_region: bool = False,
) -> DecoderLplDecisionV1:
    """Build the final deliverables under ``out_dir``.

    Reads shard CSVs (or single-shard ``tables/``), writes consolidated
    tables, renders the six aggregate figures, computes the decision.json
    v1.0, and writes ``report.md``. Returns the decision payload so the
    caller can log a one-line summary.
    """
    out_dir = Path(out_dir)
    tables_dir = out_dir / "tables"
    figs_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Read + persist consolidated tables.
    consolidated: dict[str, list[dict]] = {}
    for fname in (
        "per_block_magnitude.csv",
        "per_channel_L_dec_distribution.csv",
        "outlier_threshold.csv",
        "pre_post_separation.csv",
        "error_concentration.csv",
        "x1_reliability_vs_t.csv",
        "drift_per_patient_variant.csv",
        "per_cohort_W_nW_ratio.csv",
        "empty_wt_rate.csv",
    ):
        rows = _concat_shards(out_dir, fname)
        consolidated[fname] = rows
        if rows:
            _write_csv(tables_dir / fname, rows, list(rows[0]))

    # ---- 2. Derive decision fields.
    A = _recommend_A_from_error_concentration(consolidated["error_concentration.csv"])
    w_l = _w_l_from_magnitude(consolidated["per_block_magnitude.csv"], A=A)
    t_min = _t_min_from_reliability(consolidated["x1_reliability_vs_t.csv"])
    outlier_k = _outlier_k_from_distribution(
        consolidated["per_channel_L_dec_distribution.csv"], A=A
    )

    alpha_wt, alpha_notwt, overrides = _region_recipe_from_drift(
        consolidated["per_cohort_W_nW_ratio.csv"]
    )
    drift_rows = consolidated["drift_per_patient_variant.csv"]
    allowed = decide_allowed_variants(drift_rows, variants=variants)
    v4_status = detect_v4_brain_mask_status(drift_rows)
    if v4_status == "broken_drop_v4" and "v4" in allowed:
        allowed = [v for v in allowed if v != "v4"]

    # Patient count for the decision header.
    patient_set: set[tuple[str, str, str]] = set()
    per_cohort_count: dict[str, set[str]] = defaultdict(set)
    for r in consolidated["per_block_magnitude.csv"]:
        cohort = r.get("cohort", "")
        pid = r.get("patient_id", "")
        variant = r.get("variant", "")
        patient_set.add((cohort, pid, variant))
        if cohort:
            per_cohort_count[cohort].add(pid)
    n_total = len(patient_set)
    patients_per_cohort = {c: len(s) for c, s in per_cohort_count.items()}
    # Honour the input cohorts argument: cohorts with no patients still
    # show up (count 0).
    for cohort in cohorts:
        patients_per_cohort.setdefault(cohort, 0)

    decision = DecoderLplDecisionV1(
        schema_version=DECISION_SCHEMA_VERSION,
        produced_at=now_iso_utc(),
        producer=DECISION_PRODUCER,
        n_patients_run=n_total,
        patients_per_cohort=patients_per_cohort,
        A_recommended=A,
        w_l=w_l,
        t_min=t_min,
        outlier_k=outlier_k,
        region_recipe={
            "alpha_wt": alpha_wt,
            "alpha_notwt": alpha_notwt,
            "soft_region": soft_region,
            "per_cohort_overrides": overrides,
        },
        allowed_variants=allowed,
        v4_brain_mask_status=v4_status,
    )
    write_decision_json(out_dir / "decision.json", decision)

    # ---- 3. Render the six aggregate figures.
    figure_paths: dict[str, Path] = {}
    if consolidated["per_block_magnitude.csv"]:
        p = figs_dir / "magnitude_curve.png"
        magnitude_curve(consolidated["per_block_magnitude.csv"], out_path=p)
        figure_paths["per-block magnitude (§4.1)"] = p
    if consolidated["per_channel_L_dec_distribution.csv"]:
        p = figs_dir / "channel_concentration_block2_vs_block5.png"
        channel_concentration_block2_vs_block5(
            consolidated["per_channel_L_dec_distribution.csv"], out_path=p
        )
        figure_paths["channel concentration (§4.1)"] = p
    if consolidated["pre_post_separation.csv"]:
        p = figs_dir / "separation_per_region.png"
        separation_per_region(consolidated["pre_post_separation.csv"], out_path=p)
        figure_paths["pre/post separation (§4.2)"] = p
    if consolidated["x1_reliability_vs_t.csv"]:
        p = figs_dir / "t_min_knee.png"
        t_min_knee(consolidated["x1_reliability_vs_t.csv"], out_path=p, knee_t=t_min)
        figure_paths["x̂_1 reliability vs t (§4.4)"] = p
    if drift_rows:
        p = figs_dir / "drift_heatmap.png"
        drift_heatmap(drift_rows, out_path=p)
        figure_paths["drift heatmap (§4.7b)"] = p
    if consolidated["per_cohort_W_nW_ratio.csv"]:
        p = figs_dir / "inter_cohort_ratio_box.png"
        inter_cohort_ratio_box(consolidated["per_cohort_W_nW_ratio.csv"], out_path=p)
        figure_paths["inter-cohort ratio (§4.7c)"] = p

    # ---- 4. Report.
    _render_report(out_dir, decision, figure_paths=figure_paths)

    logger.info(
        "aggregate done: n_patients=%d allowed=%s A=%s t_min=%.3f",
        decision.n_patients_run,
        decision.allowed_variants,
        decision.A_recommended,
        decision.t_min,
    )
    return decision


def update_latest_symlink(timestamp_dir: Path) -> None:
    """Point ``<root>/LATEST`` at ``timestamp_dir`` (atomic-replace)."""
    root = timestamp_dir.parent
    latest = root / "LATEST"
    tmp = root / "LATEST.tmp"
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    tmp.symlink_to(timestamp_dir.name)
    tmp.replace(latest)


__all__ = ["aggregate", "update_latest_symlink"]
