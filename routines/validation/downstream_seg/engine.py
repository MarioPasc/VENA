"""Thin engine for the downstream-seg routine (§4.4).

Wires :mod:`vena.validation.downstream_seg` to a YAML config, streams
through the shard predictions, and writes the artifact to disk.

The real arm (segmenter on real T1c) is computed **once per (cohort, scan_id)**
and cached across all 16 methods — not repeated per method.  The synthetic arm
only swaps the T1c channel; the other three harmonised volumes are identical
between arms.

Shardability: set ``methods`` and/or ``cohorts`` in the YAML to restrict a
single job to a subset; fan out to Picasso by (method, cohort) pairs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import yaml

from vena.validation.artifacts import (
    make_run_dir,
    symlink_latest,
    write_decision_json,
    write_per_scan_csv,
)
from vena.validation.downstream_seg import (
    BRATS_BUNDLE_VERSION,
    BRATS_INPUT_CHANNELS,
    BRATS_OUTPUT_CHANNELS,
    BratsSegmenter,
    CorpusLabelCache,
    SegResult,
    dice_score,
)
from vena.validation.io import (
    ReferenceCache,  # runtime use: instantiated in Engine.run()
    _decode_str_arr,
    _resolve_references_h5,
    select_scoring_volume,
)
from vena.validation.registry import SELECTION_NFE

logger = logging.getLogger(__name__)

# LUMIERE is longitudinal: 72 scans / 11 patients.  Assert after processing.
_LUMIERE_EXPECTED_SCANS = 72
_LUMIERE_EXPECTED_PATIENTS = 11


class DownstreamSegError(Exception):
    """Raised when the engine cannot proceed (missing gate, schema mismatch)."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DownstreamSegConfig:
    """Frozen configuration for the downstream-seg engine.

    Parameters
    ----------
    inference_root :
        Root of the inference shard tree.
    output_root :
        Parent directory for the artifact folder.
    bundle_path :
        Path to the downloaded ``brats_mri_segmentation`` bundle root.
    corpus_map :
        ``{cohort_name: /abs/path/to/<cohort>_image.h5}`` mapping used
        to look up multi-label tumour GT.
    ring_partitions_path :
        Path to ``ring_partitions.json`` written by ``vena-validation-preregister``.
    methods :
        Optional list of method names to run (default: all discovered).
    cohorts :
        Optional list of cohort names to run (default: all with corpus H5).
    device :
        PyTorch device string, e.g. ``"cpu"`` or ``"cuda:0"``.
    amp :
        Use AMP.  Ignored on CPU.
    selection_nfe_only :
        When True (default), run only at each method's ``SELECTION_NFE``.
    log_level :
        Python logging level name.
    """

    inference_root: Path
    output_root: Path
    bundle_path: Path
    corpus_map: dict[str, Path]
    ring_partitions_path: Path
    methods: list[str] = field(default_factory=list)
    cohorts: list[str] = field(default_factory=list)
    device: str = "cpu"
    amp: bool = False
    selection_nfe_only: bool = True
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, path: Path) -> DownstreamSegConfig:
        """Load from a YAML config file."""
        raw = yaml.safe_load(Path(path).read_text())
        corpus_raw: dict[str, str] = raw.get("corpus_map", {})
        corpus_map = {k: Path(v) for k, v in corpus_raw.items()}
        return cls(
            inference_root=Path(raw["inference_root"]),
            output_root=Path(raw["output_root"]),
            bundle_path=Path(raw["bundle_path"]),
            corpus_map=corpus_map,
            ring_partitions_path=Path(raw["ring_partitions_path"]),
            methods=list(raw.get("methods", [])),
            cohorts=list(raw.get("cohorts", [])),
            device=str(raw.get("device", "cpu")),
            amp=bool(raw.get("amp", False)),
            selection_nfe_only=bool(raw.get("selection_nfe_only", True)),
            log_level=str(raw.get("log_level", "INFO")),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_git_sha(repo_root: Path) -> str:
    """Return HEAD SHA of the repo at *repo_root*, or ``"unknown"`` on failure."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _discover_pred_files(
    inference_root: Path,
    *,
    method_filter: list[str],
    cohort_filter: list[str],
    selection_nfe: dict[str, int],
    selection_nfe_only: bool,
) -> tuple[list[tuple[str, str, int, Path]], list[str]]:
    """Glob inference_root for prediction H5 files, filtered per config.

    Smoke shards (``decision.json`` with ``smoke.enabled == True``) are
    excluded automatically so that pilot inference runs do not contaminate
    the validation analysis.

    Returns
    -------
    tuple
        ``(results, skipped_smoke_shards)`` where *results* is a list of
        ``(method, cohort, nfe, path)`` tuples and *skipped_smoke_shards*
        is a sorted list of shard names that were excluded because they
        carried ``smoke.enabled=True``.
    """
    pattern = "*/predictions/*/*/nfe_*.h5"
    files = sorted(inference_root.glob(pattern))
    results: list[tuple[str, str, int, Path]] = []
    # Cache per shard-root to avoid re-reading decision.json for every file.
    _shard_smoke: dict[Path, bool] = {}
    skipped_smoke: set[str] = set()

    for p in files:
        # Layout: <inference_root>/<shard>/predictions/<METHOD>/<COHORT>/nfe_N.h5
        # p.parents: [0]=<COHORT>, [1]=<METHOD>, [2]=predictions, [3]=<shard>
        try:
            nfe = int(p.stem.split("_")[-1])
            cohort = p.parent.name
            method = p.parent.parent.name
            shard_root = p.parents[3]
        except (ValueError, IndexError):
            logger.debug("skipping unparseable path: %s", p)
            continue

        # Smoke-shard guard: exclude shards produced by a smoke inference run.
        if shard_root not in _shard_smoke:
            dec_path = shard_root / "decision.json"
            is_smoke = False
            if dec_path.is_file():
                try:
                    dec = json.loads(dec_path.read_text())
                    smoke_cfg = dec.get("smoke", {})
                    is_smoke = bool(
                        smoke_cfg.get("enabled", False)
                        if isinstance(smoke_cfg, dict)
                        else smoke_cfg
                    )
                except (OSError, ValueError) as exc:
                    logger.debug("could not read %s: %s", dec_path, exc)
            _shard_smoke[shard_root] = is_smoke

        if _shard_smoke[shard_root]:
            if shard_root.name not in skipped_smoke:
                logger.info("skipping smoke shard: %s", shard_root.name)
            skipped_smoke.add(shard_root.name)
            continue

        if method_filter and method not in method_filter:
            continue
        if cohort_filter and cohort not in cohort_filter:
            continue
        if selection_nfe_only:
            target_nfe = selection_nfe.get(method)
            if target_nfe is None:
                logger.warning(
                    "no selection_nfe for method %s — skipping (set selection_nfe_only: false to include)",
                    method,
                )
                continue
            if nfe != target_nfe:
                continue

        results.append((method, cohort, nfe, p))

    skipped_smoke_shards = sorted(skipped_smoke)
    logger.info(
        "discovered %d prediction files; skipped %d smoke shards: %s",
        len(results),
        len(skipped_smoke_shards),
        skipped_smoke_shards,
    )
    return results, skipped_smoke_shards


def _read_pred_row(pred_path: Path, row_idx: int, *, ref_cache: ReferenceCache) -> dict[str, Any]:
    """Read one row from a prediction H5 at *row_idx*.

    Applies the commit-1c5d2c3 scoring-space fix: reads both
    ``t1c_synthetic_raw`` and ``t1c_synthetic_harmonised`` from the prediction
    H5, then calls :func:`~vena.validation.io.select_scoring_volume` to pick
    the correct volume.  For 15 of 16 methods the raw volume is already in the
    trained normalised space and must be scored as-is; only scanner-unit methods
    (``C0-Identity``) sit outside ``[0, 1.05]`` and fall back to harmonised.

    Also computes ``wt_join_dice``: Dice between ``masks/wt`` in the prediction
    H5 and ``masks/wt`` in the reference H5 (joined by scan_id).  A correct
    scan_id join produces Dice ≈ 1.0; near-zero indicates a row-index join bug
    or mismatched build pipelines.

    Parameters
    ----------
    pred_path :
        Path to a ``predictions/<METHOD>/<COHORT>/nfe_<NNN>.h5`` file.
    row_idx :
        0-based scan index within the prediction file.
    ref_cache :
        Shared :class:`~vena.validation.io.ReferenceCache` instance for
        caching the reference scan-id → row-index map across method iterations.

    Returns
    -------
    dict
        Keys: ``scan_id``, ``patient_id``, ``t1c_synth``, ``pred_mode``,
        ``wt_join_dice``, ``t1c_real``, ``t1pre``, ``t2``, ``flair``.
        All volumes are ``(H, W, D)`` float32.  ``pred_mode`` is ``"raw"``
        or ``"harmonised"`` per :func:`~vena.validation.io.select_scoring_volume`.
        ``wt_join_dice`` is a float in ``[0, 1]``.
    """
    with h5py.File(pred_path, "r") as pf:
        scan_ids = _decode_str_arr(pf["metadata/scan_id"][:])
        patient_ids = _decode_str_arr(pf["metadata/patient_id"][:])
        scan_id = scan_ids[row_idx]
        patient_id = patient_ids[row_idx]
        # Read both volumes; select_scoring_volume picks the correct one below.
        harmonised_vol = np.asarray(
            pf["predictions/t1c_synthetic_harmonised"][row_idx], dtype=np.float32
        )
        raw_vol = np.asarray(pf["predictions/t1c_synthetic_raw"][row_idx], dtype=np.float32)
        # Pred-side WT mask for scan-id join proof.
        pred_wt = np.asarray(pf["masks/wt"][row_idx], dtype=bool)

        # _resolve_references_h5 must be called while pf is open (reads attrs)
        ref_h5 = _resolve_references_h5(pf, pred_path)

    if not ref_h5.is_file():
        raise FileNotFoundError(f"reference H5 not found for {pred_path}: {ref_h5}")

    ref_idx_map = ref_cache.get_scan_index(ref_h5)
    ref_row = ref_idx_map.get(scan_id)
    if ref_row is None:
        raise KeyError(f"scan_id {scan_id!r} not in reference H5 {ref_h5}")

    # Reference H5 stores all harmonised input modalities once per cohort
    # (schema 2.0 design — verified in conftest and prod shard writer).
    with h5py.File(ref_h5, "r") as rf:
        t1c_real = rf["reference/t1c_real_harmonised"][ref_row]
        t1pre = rf["reference/t1pre_harmonised"][ref_row]
        t2 = rf["reference/t2_harmonised"][ref_row]
        flair = rf["reference/flair_harmonised"][ref_row]
        brain = np.asarray(rf["masks/brain"][ref_row], dtype=bool)
        # Ref-side WT mask for scan-id join proof.
        ref_wt = np.asarray(rf["masks/wt"][ref_row], dtype=bool)

    # commit 1c5d2c3: select the correct scoring volume per-scan.
    t1c_synth, pred_mode = select_scoring_volume(raw_vol, harmonised_vol, brain)

    # Join-proof: Dice(pred_wt, ref_wt) ≈ 1.0 iff the scan_id join is correct.
    # The inference writer copies masks/wt verbatim from the reference H5, so any
    # mismatch indicates an index-join bug or mismatched build pipelines.
    wt_join_dice = dice_score(pred_wt, ref_wt)

    return {
        "scan_id": scan_id,
        "patient_id": patient_id,
        "t1c_synth": t1c_synth,
        "pred_mode": pred_mode,
        "wt_join_dice": float(wt_join_dice),
        "t1c_real": t1c_real.astype(np.float32),
        "t1pre": t1pre.astype(np.float32),
        "t2": t2.astype(np.float32),
        "flair": flair.astype(np.float32),
    }


def _seg_to_dice(
    segmenter: BratsSegmenter,
    t1c: np.ndarray,
    t1pre: np.ndarray,
    t2: np.ndarray,
    flair: np.ndarray,
    gt_wt: np.ndarray,
    gt_tc: np.ndarray,
    gt_et: np.ndarray,
) -> tuple[float, float, float]:
    """Run segmenter and return (dice_wt, dice_tc, dice_et)."""
    tc_pred, wt_pred, et_pred = segmenter.segment(t1c, t1pre, t2, flair)
    d_wt = dice_score(wt_pred, gt_wt)
    d_tc = dice_score(tc_pred, gt_tc)
    d_et = dice_score(et_pred, gt_et)
    return d_wt, d_tc, d_et


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------


def _write_tables(run_dir: Path, df: pd.DataFrame) -> None:
    """Write per-method aggregate CSVs to tables/.

    Parameters
    ----------
    run_dir :
        Artifact run directory.
    df :
        per_scan tidy DataFrame.
    """
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(exist_ok=True)

    if df.empty:
        return

    agg_metrics = [
        "delta_wt",
        "delta_tc",
        "delta_et",
        "dice_wt_real",
        "dice_tc_real",
        "dice_et_real",
        "dice_wt_synth",
        "dice_tc_synth",
        "dice_et_synth",
        "wt_join_dice",
    ]

    # Per (method, cohort, ring, nfe)
    agg = (
        df.groupby(["method", "cohort", "ring", "nfe"])[agg_metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg.columns = ["_".join(str(c) for c in col).rstrip("_") for col in agg.columns]
    agg.to_csv(tables_dir / "method_cohort_agg.csv", index=False)

    # Per (method, ring) — Ring A and Ring B separate
    ring_agg = (
        df.groupby(["method", "ring"])[agg_metrics].agg(["mean", "std", "count"]).reset_index()
    )
    ring_agg.columns = ["_".join(str(c) for c in col).rstrip("_") for col in ring_agg.columns]
    ring_agg.to_csv(tables_dir / "method_ring_agg.csv", index=False)

    # WT-join Dice per scan (audit trail)
    df[["method", "cohort", "scan_id", "patient_id", "wt_join_dice"]].to_csv(
        tables_dir / "wt_join_dice_per_scan.csv", index=False
    )


def _write_figures(run_dir: Path, df: pd.DataFrame) -> None:
    """Write figures to figures/.

    Produces:
    - ``delta_et_per_method.png``, ``delta_wt_per_method.png``,
      ``delta_tc_per_method.png``: bar charts with Holm-corrected
      significance vs C0-Identity, black background.
    - ``wt_join_dice_histogram.png``: histogram of per-scan WT-join Dice.

    Parameters
    ----------
    run_dir :
        Artifact run directory.
    df :
        per_scan tidy DataFrame.
    """
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend; no display required
    import matplotlib.pyplot as plt
    from scipy.stats import wilcoxon
    from statsmodels.stats.multitest import multipletests

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    if df.empty:
        return

    # Collapse to patient level first (§11 — avoids anti-conservative inflation
    # for LUMIERE where 72 scans map to only 11 patients).
    patient_agg = (
        df.groupby(["method", "cohort", "patient_id"])[["delta_wt", "delta_tc", "delta_et"]]
        .mean()
        .reset_index()
    )

    methods = sorted(df["method"].unique().tolist())
    ref_method = "C0-Identity" if "C0-Identity" in methods else None

    for metric, label in [
        ("delta_et", "ΔDice ET"),
        ("delta_wt", "ΔDice WT"),
        ("delta_tc", "ΔDice TC"),
    ]:
        fig, ax = plt.subplots(figsize=(max(6, len(methods) * 1.4), 5))
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        method_means: dict[str, float] = {}
        method_pvals: dict[str, float] = {}

        for m in methods:
            vals = patient_agg[patient_agg["method"] == m][metric].dropna().values
            method_means[m] = float(np.mean(vals)) if len(vals) > 0 else float("nan")

            if ref_method and m != ref_method:
                # Paired Wilcoxon vs reference method, aligned by (cohort, patient_id).
                paired = pd.merge(
                    patient_agg[patient_agg["method"] == m][["cohort", "patient_id", metric]],
                    patient_agg[patient_agg["method"] == ref_method][
                        ["cohort", "patient_id", metric]
                    ],
                    on=["cohort", "patient_id"],
                    suffixes=("_m", "_ref"),
                ).dropna()
                if len(paired) >= 5:
                    try:
                        _, p = wilcoxon(
                            paired[f"{metric}_m"].values, paired[f"{metric}_ref"].values
                        )
                        method_pvals[m] = float(p)
                    except Exception:
                        pass

        # Sort methods by mean delta descending.
        sorted_methods = sorted(
            methods,
            key=lambda m: method_means.get(m, float("-inf")),
            reverse=True,
        )

        # Holm correction across all tested methods (same family).
        valid_methods = [m for m in sorted_methods if m in method_pvals]
        holm_sig: dict[str, tuple[bool, float]] = {}
        if valid_methods:
            pvals_arr = [method_pvals[m] for m in valid_methods]
            reject, pvals_corr, _, _ = multipletests(pvals_arr, method="holm")
            for m, rej, pc in zip(valid_methods, reject, pvals_corr, strict=True):
                holm_sig[m] = (bool(rej), float(pc))

        xs = list(range(len(sorted_methods)))
        means = [method_means.get(m, 0.0) for m in sorted_methods]
        # VENA methods in blue; competitors / C0 in orange.
        colors = ["#1a9cf0" if m.startswith("VENA") else "#e87d4d" for m in sorted_methods]
        ax.bar(xs, means, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)

        for i, m in enumerate(sorted_methods):
            if m in holm_sig:
                rej, pc = holm_sig[m]
                stars = "***" if pc < 0.001 else ("**" if pc < 0.01 else ("*" if rej else "n.s."))
                offset = 0.01 if means[i] >= 0 else -0.03
                ax.text(
                    i, means[i] + offset, stars, ha="center", va="bottom", fontsize=7, color="white"
                )

        ax.set_xticks(xs)
        ax.set_xticklabels(sorted_methods, rotation=45, ha="right", fontsize=7, color="white")
        ax.set_ylabel(label, color="white", fontsize=9)
        ax.set_title(
            f"{label} per method (patient-level mean; Holm vs {ref_method or 'none'})",
            color="white",
            fontsize=9,
        )
        ax.axhline(0, color="white", linewidth=0.5, linestyle="--")
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
        caption = (
            f"Correction: Holm-Bonferroni / test: paired Wilcoxon / "
            f"family: downstream_seg / reference: {ref_method or 'none'}"
        )
        fig.text(0.5, -0.04, caption, ha="center", fontsize=6, color="#aaaaaa")
        plt.tight_layout()
        fig.savefig(
            figures_dir / f"{metric}_per_method.png",
            dpi=150,
            bbox_inches="tight",
            facecolor="black",
        )
        plt.close(fig)

    # WT-join Dice histogram.
    wt_dice_vals = df["wt_join_dice"].dropna().values
    if len(wt_dice_vals) > 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")
        ax.hist(wt_dice_vals, bins=30, color="#1a9cf0", alpha=0.8, edgecolor="white", linewidth=0.3)
        ax.axvline(0.99, color="#ff6b6b", linewidth=1.5, linestyle="--", label="0.99 threshold")
        n_below = int((wt_dice_vals < 0.99).sum())
        ax.set_title(
            f"WT-join Dice (scan-ID join proof) — {n_below}/{len(wt_dice_vals)} below 0.99",
            color="white",
            fontsize=9,
        )
        ax.set_xlabel("Dice(pred masks/wt, ref masks/wt)", color="white", fontsize=8)
        ax.set_ylabel("Count", color="white", fontsize=8)
        ax.tick_params(colors="white")
        ax.legend(facecolor="#222222", labelcolor="white", fontsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
        plt.tight_layout()
        fig.savefig(
            figures_dir / "wt_join_dice_histogram.png",
            dpi=150,
            bbox_inches="tight",
            facecolor="black",
        )
        plt.close(fig)


def _write_report_md(
    run_dir: Path,
    df: pd.DataFrame,
    *,
    skipped_smoke_shards: list[str],
    skipped_no_corpus: set[str],
    skipped_no_scan: list[str],
    real_arm_call_count: int,
    wall_clock_s: float,
    git_sha: str,
    n_scans_lumiere: int,
    n_patients_lumiere: int,
) -> None:
    """Write report.md for the downstream-seg artifact.

    Parameters
    ----------
    run_dir :
        Artifact run directory.
    df :
        per_scan tidy DataFrame.
    skipped_smoke_shards :
        Shard names excluded because ``smoke.enabled=True``.
    skipped_no_corpus :
        Cohort names skipped because they had no corpus H5.
    skipped_no_scan :
        Per-scan skip messages.
    real_arm_call_count :
        Number of unique real-arm segmenter calls.
    wall_clock_s :
        Total wall-clock time in seconds.
    git_sha :
        Git HEAD SHA of the producing repo.
    n_scans_lumiere :
        Unique LUMIERE scan IDs processed.
    n_patients_lumiere :
        Unique LUMIERE patient IDs processed.
    """
    lines: list[str] = []
    lines.append("# Downstream-seg §4.4 — report\n")
    lines.append(f"**produced_at**: {datetime.now(tz=UTC).isoformat()}  ")
    lines.append(f"**git_sha**: `{git_sha}`  ")
    lines.append(f"**wall_clock**: {wall_clock_s:.1f} s\n")

    if df.empty:
        lines.append("**WARNING**: no rows produced — check corpus_map and inference_root.\n")
        (run_dir / "report.md").write_text("\n".join(lines))
        return

    # Summary counts
    n_methods = df["method"].nunique()
    n_cohorts = df["cohort"].nunique()
    n_scans = df.groupby(["cohort", "scan_id"]).ngroups
    n_patients = df.groupby(["cohort", "patient_id"]).ngroups
    lines.append(
        f"**Summary**: {len(df)} rows · {n_methods} methods · {n_cohorts} cohorts · "
        f"{n_scans} scans · {n_patients} patients\n"
    )

    # Oracle-mask confound documentation (mandatory per correction item 1)
    lines.append("## Oracle-mask confound\n")
    lines.append(
        "**`VENA-S1-v3b-rw`** receives the ground-truth WT mask as ControlNet conditioning. "
        "When Dice is computed against the same GT, the segmenter can partially recover the "
        "mask that was fed to the generator, inflating synthetic Dice and artificially reducing "
        "ΔDice.  **`VENA-S1-v3a`** (concat-only, no mask conditioning) is the honest comparator "
        "and must appear next to every v3b-rw number.\n"
    )

    # Negative delta count per method
    lines.append("## ΔDice summary per method\n")
    lines.append(
        "delta = Dice_real − Dice_synth. "
        "Negative delta means synthetic Dice > real Dice (oracle-confound indicator).\n"
    )

    if "delta_et" in df.columns:
        # Collapse to patient level before reporting (§11 anti-conservative guard)
        pat = (
            df.groupby(["method", "cohort", "patient_id"])[["delta_wt", "delta_tc", "delta_et"]]
            .mean()
            .reset_index()
        )
        by_method = pat.groupby("method")[["delta_wt", "delta_tc", "delta_et"]].mean()
        neg_et = (
            pat[pat["delta_et"] < 0].groupby("method")["delta_et"].count().rename("n_neg_delta_et")
        )
        by_method = by_method.join(neg_et, how="left").fillna(0)
        by_method["n_neg_delta_et"] = by_method["n_neg_delta_et"].astype(int)
        lines.append(by_method.to_markdown())
        lines.append("\n")
        total_neg_et = int((pat["delta_et"] < 0).sum())
        lines.append(
            f"**Total negative delta_et (patient-level)**: {total_neg_et} / {len(pat)} "
            f"({100 * total_neg_et / max(len(pat), 1):.1f}%).\n"
        )

    # WT-join Dice proof
    lines.append("## WT-join Dice (scan-ID join proof)\n")
    wt_vals = df["wt_join_dice"].dropna()
    if len(wt_vals) > 0:
        lines.append(
            f"Dice(pred masks/wt, ref masks/wt) — "
            f"min={wt_vals.min():.4f} mean={wt_vals.mean():.4f} "
            f"below_0.99={int((wt_vals < 0.99).sum())} / {len(wt_vals)}\n"
        )
    else:
        lines.append("**WARNING**: no wt_join_dice values computed.\n")

    # LUMIERE patient-collapse verification
    lines.append("## LUMIERE longitudinal collapse\n")
    if n_scans_lumiere > 0:
        lines.append(
            f"LUMIERE: {n_scans_lumiere} scans / {n_patients_lumiere} patients processed "
            f"(expected {_LUMIERE_EXPECTED_SCANS} / {_LUMIERE_EXPECTED_PATIENTS}).\n"
        )
        if n_scans_lumiere != _LUMIERE_EXPECTED_SCANS:
            lines.append(
                f"**WARNING**: expected {_LUMIERE_EXPECTED_SCANS} LUMIERE scans, "
                f"got {n_scans_lumiere}.\n"
            )
        if n_patients_lumiere != _LUMIERE_EXPECTED_PATIENTS:
            lines.append(
                f"**WARNING**: expected {_LUMIERE_EXPECTED_PATIENTS} LUMIERE patients, "
                f"got {n_patients_lumiere}.\n"
            )
    else:
        lines.append("LUMIERE not processed in this run.\n")

    # Known limitations (§10 open issues — state, not fix)
    lines.append("## Known limitations (inherited from Phase 1)\n")
    lines.append(
        "- **VENA `_sample` unseeded**: predictions are not bit-reproducible; "
        "cross-NFE draws differ. Cannot be fixed without re-running inference.\n"
    )
    lines.append(
        "- **Input conditioning imbalance**: C4–C7 condition on {t1pre, flair}; "
        "VENA/ResViT on {t1pre, t2, flair}; pGAN/SynDiff on {t1pre}. "
        "Disclosed in every cross-method comparison.\n"
    )
    lines.append(
        "- **Oracle mask**: VENA-S1-v3b/v3b-rw receive GT WT mask as conditioning. "
        "VENA-S1-v3a (concat-only) is the no-oracle comparator.\n"
    )
    lines.append(
        "- **Appendix A deviation**: used a fixed pretrained BraTS segmenter instead of "
        "per-cohort nnU-Net from scratch. The level confounder cancels in paired ΔDice; "
        "absolute Dice values are not directly comparable to proposal Table A1.\n"
    )

    # Skipped items
    lines.append("## Skipped items\n")
    lines.append(f"- Smoke shards excluded: {skipped_smoke_shards or 'none'}\n")
    lines.append(f"- Cohorts without corpus H5: {sorted(skipped_no_corpus) or 'none'}\n")
    lines.append(f"- Per-scan skips: {len(skipped_no_scan)}\n")
    if skipped_no_scan:
        for msg in skipped_no_scan[:10]:
            lines.append(f"  - {msg}\n")
        if len(skipped_no_scan) > 10:
            lines.append(f"  - ... and {len(skipped_no_scan) - 10} more\n")

    # Figures
    lines.append("## Figures\n")
    lines.append("- `figures/delta_et_per_method.png` — ΔDice ET per method, Holm-corrected\n")
    lines.append("- `figures/delta_wt_per_method.png` — ΔDice WT per method, Holm-corrected\n")
    lines.append("- `figures/delta_tc_per_method.png` — ΔDice TC per method, Holm-corrected\n")
    lines.append("- `figures/wt_join_dice_histogram.png` — WT-join Dice scan-ID join proof\n")

    (run_dir / "report.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


@dataclass
class DownstreamSegEngine:
    """Execute the downstream-seg routine and write the artifact."""

    cfg: DownstreamSegConfig

    def run(self) -> Path:
        """Run the routine and return the artifact directory.

        Returns
        -------
        Path
            The run directory containing ``per_scan/downstream_seg.csv``.
        """
        logging.basicConfig(
            level=getattr(logging, self.cfg.log_level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        )

        # Load ring partitions (drives COHORT_RING, used to annotate CSV)
        from vena.validation.registry import load_partitions

        load_partitions(self.cfg.ring_partitions_path)
        from vena.validation.registry import COHORT_RING

        # Build effective selection_nfe (registry + user override falls through)
        sel_nfe: dict[str, int] = dict(SELECTION_NFE)

        # Discover prediction files
        pred_files, skipped_smoke_shards = _discover_pred_files(
            self.cfg.inference_root,
            method_filter=self.cfg.methods,
            cohort_filter=self.cfg.cohorts,
            selection_nfe=sel_nfe,
            selection_nfe_only=self.cfg.selection_nfe_only,
        )
        if not pred_files:
            raise DownstreamSegError(f"No prediction files found under {self.cfg.inference_root}")

        # Instantiate the corpus label cache (lazy H5 opens)
        corpus_cache = CorpusLabelCache(self.cfg.corpus_map)

        # Instantiate the segmenter (loads checkpoint — no network access)
        segmenter = BratsSegmenter(
            self.cfg.bundle_path,
            device=self.cfg.device,
            amp=self.cfg.amp,
        )

        # Shared reference-scan-index cache: avoids re-reading metadata/scan_id
        # from the same reference H5 for each of the 16 methods.
        ref_cache = ReferenceCache()

        # Real-arm cache: (cohort, scan_id) → (dice_wt, dice_tc, dice_et)
        # Computed ONCE per scan, reused across all methods.
        real_arm_cache: dict[tuple[str, str], tuple[float, float, float]] = {}
        real_arm_call_count = 0

        run_dir = make_run_dir(self.cfg.output_root, "downstream_seg")
        logger.info("artifact dir: %s", run_dir)

        rows: list[dict[str, Any]] = []
        skipped_no_corpus: set[str] = set()
        skipped_no_scan: list[str] = []

        t_start = time.monotonic()

        for method, cohort, nfe, pred_path in pred_files:
            ring = COHORT_RING.get(cohort, "?")

            if not corpus_cache.has_cohort(cohort):
                if cohort not in skipped_no_corpus:
                    logger.warning("cohort %s has no corpus H5 with masks/tumor — skipping", cohort)
                    skipped_no_corpus.add(cohort)
                continue

            # Count scans in this prediction file
            with h5py.File(pred_path, "r") as pf:
                n_scans = pf["metadata/scan_id"].shape[0]

            logger.info(
                "processing method=%s cohort=%s nfe=%d (%d scans)",
                method,
                cohort,
                nfe,
                n_scans,
            )

            for row_idx in range(n_scans):
                try:
                    data = _read_pred_row(pred_path, row_idx, ref_cache=ref_cache)
                except (KeyError, FileNotFoundError) as exc:
                    logger.warning("skipping scan idx %d in %s: %s", row_idx, pred_path, exc)
                    skipped_no_scan.append(f"{method}/{cohort}/idx{row_idx}: {exc}")
                    continue

                scan_id = data["scan_id"]
                patient_id = data["patient_id"]

                # Fetch GT labels from corpus H5.
                try:
                    gt_wt, gt_tc, gt_et = corpus_cache.get_labels(cohort, scan_id)
                except KeyError as exc:
                    logger.warning("GT labels missing for %s/%s: %s", cohort, scan_id, exc)
                    skipped_no_scan.append(f"{method}/{cohort}/{scan_id}: no GT")
                    continue

                # T1pre / T2 / FLAIR come from the reference H5 (identical for
                # both arms; only T1c differs between real and synthetic).
                t1pre = data["t1pre"]
                t2 = data["t2"]
                flair = data["flair"]

                # Real arm — compute once per (cohort, scan_id)
                cache_key = (cohort, scan_id)
                if cache_key not in real_arm_cache:
                    real_arm_call_count += 1
                    logger.debug(
                        "real arm #%d: cohort=%s scan=%s", real_arm_call_count, cohort, scan_id
                    )
                    d_wt_r, d_tc_r, d_et_r = _seg_to_dice(
                        segmenter,
                        data["t1c_real"],
                        t1pre,
                        t2,
                        flair,
                        gt_wt,
                        gt_tc,
                        gt_et,
                    )
                    real_arm_cache[cache_key] = (d_wt_r, d_tc_r, d_et_r)
                else:
                    d_wt_r, d_tc_r, d_et_r = real_arm_cache[cache_key]

                # Synthetic arm — always recompute (only T1c channel differs)
                d_wt_s, d_tc_s, d_et_s = _seg_to_dice(
                    segmenter,
                    data["t1c_synth"],
                    t1pre,
                    t2,
                    flair,
                    gt_wt,
                    gt_tc,
                    gt_et,
                )

                pred_mode = data["pred_mode"]  # "raw" | "harmonised" — audit trail
                wt_join_dice = data["wt_join_dice"]

                result = SegResult(
                    method=method,
                    cohort=cohort,
                    ring=ring,
                    nfe=nfe,
                    scan_id=scan_id,
                    patient_id=patient_id,
                    dice_wt_real=d_wt_r,
                    dice_tc_real=d_tc_r,
                    dice_et_real=d_et_r,
                    dice_wt_synth=d_wt_s,
                    dice_tc_synth=d_tc_s,
                    dice_et_synth=d_et_s,
                )
                rows.append(
                    {
                        "method": result.method,
                        "cohort": result.cohort,
                        "ring": result.ring,
                        "nfe": result.nfe,
                        "scan_id": result.scan_id,
                        "patient_id": result.patient_id,
                        "pred_mode": pred_mode,
                        "wt_join_dice": wt_join_dice,
                        "dice_wt_real": result.dice_wt_real,
                        "dice_tc_real": result.dice_tc_real,
                        "dice_et_real": result.dice_et_real,
                        "dice_wt_synth": result.dice_wt_synth,
                        "dice_tc_synth": result.dice_tc_synth,
                        "dice_et_synth": result.dice_et_synth,
                        "delta_wt": result.delta_wt,
                        "delta_tc": result.delta_tc,
                        "delta_et": result.delta_et,
                    }
                )

        corpus_cache.close()
        wall_clock_s = time.monotonic() - t_start

        logger.info(
            "real arm calls: %d (cache hits: %d)",
            real_arm_call_count,
            len(rows) - real_arm_call_count,
        )
        logger.info("wall clock: %.1f s", wall_clock_s)

        if not rows:
            logger.warning("no rows produced — check corpus_map and inference_root")

        # LUMIERE longitudinal-collapse check (§13 — assert 72 scans / 11 patients).
        lumiere_scan_ids = {r["scan_id"] for r in rows if r["cohort"] == "LUMIERE"}
        lumiere_patient_ids = {r["patient_id"] for r in rows if r["cohort"] == "LUMIERE"}
        n_scans_lumiere = len(lumiere_scan_ids)
        n_patients_lumiere = len(lumiere_patient_ids)
        if n_scans_lumiere > 0:
            if n_scans_lumiere != _LUMIERE_EXPECTED_SCANS:
                logger.warning(
                    "LUMIERE scan count: expected %d, got %d — check shard coverage",
                    _LUMIERE_EXPECTED_SCANS,
                    n_scans_lumiere,
                )
            if n_patients_lumiere != _LUMIERE_EXPECTED_PATIENTS:
                logger.warning(
                    "LUMIERE patient count: expected %d, got %d — check patient_id join",
                    _LUMIERE_EXPECTED_PATIENTS,
                    n_patients_lumiere,
                )
            logger.info(
                "LUMIERE: %d scans / %d patients (expected %d / %d)",
                n_scans_lumiere,
                n_patients_lumiere,
                _LUMIERE_EXPECTED_SCANS,
                _LUMIERE_EXPECTED_PATIENTS,
            )

        df = pd.DataFrame(
            rows,
            columns=[
                "method",
                "cohort",
                "ring",
                "nfe",
                "scan_id",
                "patient_id",
                "pred_mode",
                "wt_join_dice",
                "dice_wt_real",
                "dice_tc_real",
                "dice_et_real",
                "dice_wt_synth",
                "dice_tc_synth",
                "dice_et_synth",
                "delta_wt",
                "delta_tc",
                "delta_et",
            ],
        )

        csv_path = write_per_scan_csv(run_dir, df, name="downstream_seg.csv")
        logger.info("wrote %s (%d rows)", csv_path, len(df))

        # Write tables, figures, and report.
        _write_tables(run_dir, df)
        _write_figures(run_dir, df)

        # Build summary stats for decision.json
        n_scans_processed = len({(r["cohort"], r["scan_id"]) for r in rows})
        n_real_arm_unique = real_arm_call_count

        bundle_sha = hashlib.sha256(
            (self.cfg.bundle_path / "models" / "model.pt").read_bytes()
        ).hexdigest()

        # WT-join Dice aggregates for the audit trail.
        wt_dice_vals = [
            r["wt_join_dice"] for r in rows if not (r["wt_join_dice"] != r["wt_join_dice"])
        ]
        wt_join_dice_min = float(np.min(wt_dice_vals)) if wt_dice_vals else float("nan")
        wt_join_dice_mean = float(np.mean(wt_dice_vals)) if wt_dice_vals else float("nan")
        wt_join_dice_below_0_99_n = sum(1 for v in wt_dice_vals if v < 0.99)

        # Resolve git SHA from the repo containing this engine file.
        _repo_root = Path(__file__).resolve().parents[3]
        git_sha = _get_git_sha(_repo_root)

        payload: dict[str, Any] = {
            "schema_version": "1.0",
            "produced_at": datetime.now(tz=UTC).isoformat(),
            "producer": "routines.validation.downstream_seg:1.0",
            "git_sha": git_sha,
            "inference_root": str(self.cfg.inference_root),
            "output_root": str(self.cfg.output_root),
            "bundle_path": str(self.cfg.bundle_path),
            "bundle_version": BRATS_BUNDLE_VERSION,
            "bundle_model_sha256": bundle_sha,
            "bundle_input_channel_order": list(BRATS_INPUT_CHANNELS),
            "bundle_output_channel_order": list(BRATS_OUTPUT_CHANNELS),
            "bundle_preprocessing": "NormalizeIntensityd(nonzero=True, channel_wise=True)",
            "bundle_inferer": "SlidingWindowInferer(roi_size=[240,240,160], overlap=0.5)",
            "device": self.cfg.device,
            "amp": self.cfg.amp,
            "selection_nfe_only": self.cfg.selection_nfe_only,
            "methods_requested": self.cfg.methods or "all",
            "cohorts_requested": self.cfg.cohorts or "all",
            "n_pred_files": len(pred_files),
            "n_scans_processed": n_scans_processed,
            "n_real_arm_unique_calls": n_real_arm_unique,
            "n_rows_csv": len(df),
            "skipped_smoke_shards": skipped_smoke_shards,
            "skipped_cohorts_no_corpus": sorted(skipped_no_corpus),
            "skipped_scans_n": len(skipped_no_scan),
            "wall_clock_s": round(wall_clock_s, 1),
            "wt_join_dice_min": wt_join_dice_min,
            "wt_join_dice_mean": wt_join_dice_mean,
            "wt_join_dice_below_0_99_n": wt_join_dice_below_0_99_n,
            "lumiere_n_scans": n_scans_lumiere,
            "lumiere_n_patients": n_patients_lumiere,
            "empty_et_convention": "NaN when both pred and GT are empty (not 0)",
            "scoring_space_fix": (
                "commit 1c5d2c3 — select_scoring_volume picks raw for 15/16 methods, "
                "harmonised only for scanner-unit methods (e.g. C0-Identity). "
                "pred_mode column records the selection per row."
            ),
            "appendix_a_deviation": (
                "Used fixed pretrained BraTS segmenter instead of per-cohort "
                "nnU-Net from scratch. Level confounder cancels in paired Δ; "
                "see 05_downstream_seg.md §2 and report.md."
            ),
            "oracle_mask_confound": (
                "VENA-S1-v3b-rw receives GT WT mask as conditioning — report "
                "VENA-S1-v3a (concat-only) alongside every v3b-rw number."
            ),
            "corpus_map": {k: str(v) for k, v in self.cfg.corpus_map.items()},
        }

        write_decision_json(run_dir, payload)

        _write_report_md(
            run_dir,
            df,
            skipped_smoke_shards=skipped_smoke_shards,
            skipped_no_corpus=skipped_no_corpus,
            skipped_no_scan=skipped_no_scan,
            real_arm_call_count=real_arm_call_count,
            wall_clock_s=wall_clock_s,
            git_sha=git_sha,
            n_scans_lumiere=n_scans_lumiere,
            n_patients_lumiere=n_patients_lumiere,
        )

        symlink_latest(run_dir)

        # Copy config for reproducibility
        shutil.copy(
            self.cfg.ring_partitions_path,
            run_dir / "ring_partitions.json",
        )

        logger.info("downstream_seg done: %s", run_dir)
        return run_dir
