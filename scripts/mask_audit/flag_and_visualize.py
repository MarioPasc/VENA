"""Merge per-cohort audit CSVs, flag outliers, and render per-patient QC figures.

Outlier policy (two tiers, because the two failure classes need different logic)
-------------------------------------------------------------------------------
**FAIL — absolute.** Logical invariants that cannot be violated at *any* tumour
size.  A single breach is a defect regardless of the population distribution.

**WARN — size-stratified robust z.** For continuous agreement metrics, "bad" is
relative.  We use the Iglewicz-Hoaglin modified z-score
``z = 0.6745 * (x - median) / MAD``, one-sided into the bad tail, threshold 3.5.
MAD (not the standard deviation) so that the outliers cannot inflate their own
cutoff.

Stratification is **essential, not cosmetic**: latent<->image IoU degrades for
small and multifocal cores purely from 4x avg-pool quantization (a single latent
voxel spans 4x4x4 image voxels).  An unstratified threshold would therefore flag
every small tumour and bury the real defects.  We bin by global quartile of
``log10(gt_tc_vox_crop)`` over TC-present scans; TC-empty scans form their own
stratum and are excluded from the statistical tier (their IoU is undefined).

Figures are rendered for every FAIL, the worst WARNs per cohort, plus a couple
of median-quality EXEMPLARs per cohort — a folder containing only outliers gives
no baseline to judge them against.  Every cap applied is logged explicitly.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812
from audit_cohort import COHORTS, LATENT_CROP_BOX, POOL_STRIDE, _crop_to_box, _decode

from vena.common import CropPadSpec
from vena.segmentation.config import TargetConfig

logger = logging.getLogger("mask_audit.flag")

# Shared visual conventions with vena.segmentation.metrics.visualize
TC_CMAP = plt.cm.YlGn
NETC_CMAP = plt.cm.RdPu
TC_RGB = (0.1, 0.9, 0.2)
NETC_RGB = (1.0, 0.1, 0.6)
CONTOUR_LEVELS = [0.25, 0.5, 0.75]
CONTOUR_LW = 0.6

MAX_COLS = 10
MIN_COLS_TARGET = 7
Z_WARN = 3.5
Z_INFO = 3.0
MAX_FAIL_FIGS_PER_COHORT = 25
MAX_WARN_FIGS_PER_COHORT = 10
N_EXEMPLARS_PER_COHORT = 2

# metric -> bad direction ("low" = small values are bad, "high" = large are bad)
STAT_METRICS: dict[str, str] = {
    "lat_iou_tc": "low",
    "lat_centroid_dist_tc": "high",
    "abs_log_volratio_tc": "high",
    "abs_log_massratio_tc": "high",
    "oracle_tc_centroid_dist": "high",
    "soft_intermediate_frac_tc": "low",
    "tc_outside_brain_frac": "high",
}

# Minimum *practically meaningful* deviation from the stratum median, per metric.
# A scan is only flagged when it is BOTH statistically unusual (z > threshold)
# AND off by at least this much. Without this gate, robust z is hypersensitive on
# tightly-concentrated or zero-inflated metrics: mass ratio clusters at ~1 and
# brain-spill at ~0, so MAD collapses toward zero and a trivial deviation scores
# z > 3.5 (measured: 67 spurious tc_outside_brain_frac and 44 mass-ratio WARNs).
MIN_EFFECT: dict[str, float] = {
    "lat_iou_tc": 0.10,  # IoU points
    "lat_centroid_dist_tc": 1.0,  # voxels (latent grid = 4 image voxels)
    "abs_log_volratio_tc": 0.05,  # ~5% volume error
    "abs_log_massratio_tc": 0.05,  # ~5% pooled-mass error
    "oracle_tc_centroid_dist": 1.0,  # voxels
    "soft_intermediate_frac_tc": 0.005,
    "tc_outside_brain_frac": 0.05,  # 5% of TC mass
}


def compute_floor() -> float:
    """Expected far-field soft value ``sigmoid(-clip_vox / sdt_sigma_vox)``."""
    t = TargetConfig()
    return 1.0 / (1.0 + math.exp(t.clip_vox / t.sdt_sigma_vox))


def hard_fail_reasons(r: pd.Series, floor: float) -> list[str]:
    """Return the list of absolute-invariant breaches for one scan."""
    out: list[str] = []
    if isinstance(r.get("error"), str) and r["error"]:
        out.append(f"error:{r['error']}")
        return out
    if r["lat_nan_count"] > 0:
        out.append("nan_in_cache")
    if not bool(r["lat_range_ok"]):
        out.append("value_out_of_range")
    if r["lat_nesting_viol_frac"] > 1e-6:
        out.append("nesting_NETC>TC")
    if r["nesting_viol_frac_img"] > 1e-6:
        out.append("nesting_img_NETC>TC")
    if r["hard_subset_soft_viol_frac_tc"] > 0.01:
        out.append("hardTC_not_subset_of_soft")
    if r["crop_clip_frac_tc"] > 0.01:
        out.append("crop_clips_tumour")
    if r["recompute_max_abs"] > 1e-5:
        out.append("cache_ne_canonical_rederive")
    # NOTE: a "far-field == floor" invariant would be WRONG here. apply_crop_pad
    # zero-pads wherever the (192,224,192) box extends past the native volume, so
    # legitimate cached values sit *below* the SDT floor in that margin
    # (measured: lat_tc_min == 0 on 148/148 scans, while a TC-empty scan's
    # lat_tc_max == floor exactly, i.e. the floor holds in the un-padded interior).
    # Non-negativity is already covered by lat_range_ok.
    if r["gt_tc_vox_crop"] > 0 and r["lat_tc_max"] <= floor + 0.01:
        out.append("tumour_lost_in_cache")
    if r["gt_tc_vox"] == 0 and r["lat_tc_max"] > floor + 0.05:
        out.append("phantom_tumour")
    # 0.25, not 0.05: at the coarse latent grid a graded halo around a
    # brain-edge tumour legitimately spills past the brain mask (measured
    # median 0.006, max 0.097 with no other defect). Only a clearly
    # pathological fraction is an absolute failure; relative outliers are
    # caught by the statistical tier via STAT_METRICS instead.
    if r["tc_outside_brain_frac"] > 0.25:
        out.append("tc_outside_brain")
    if r["gt_tc_vox_crop"] > 0 and r["dice_tc_img"] < 0.90:
        out.append("low_img_dice")
    return out


def add_robust_z(df: pd.DataFrame) -> pd.DataFrame:
    """Add size strata and per-stratum one-sided robust z for each stat metric."""
    df = df.copy()
    df["abs_log_volratio_tc"] = np.abs(np.log(df["volratio_tc_img"].clip(lower=1e-6)))
    df["abs_log_massratio_tc"] = np.abs(np.log(df["lat_tc_mass_ratio"].clip(lower=1e-6)))

    present = (df["gt_tc_vox_crop"] > 0) & (df["error"].fillna("") == "")
    df["size_stratum"] = "tc_empty"
    # Stratify WITHIN cohort as well as by size. Cohorts differ systematically
    # (voxel size, acquisition, pathology mix); with global strata those
    # differences masquerade as per-patient outliers and small OOD cohorts get
    # flagged en masse (measured: 47/243 WARN, dominated by BraTS-Africa).
    # Cohort-level effects belong in per_cohort_summary.csv instead. Bin count
    # adapts to cohort size so every stratum retains enough scans for a stable MAD.
    for cohort, g in df[present].groupby("cohort"):
        n = len(g)
        nbins = 4 if n >= 64 else (3 if n >= 36 else (2 if n >= 16 else 1))
        if nbins == 1:
            labels = ["q0"] * n
        else:
            logv = np.log10(g["gt_tc_vox_crop"].astype(float))
            try:
                b = pd.qcut(logv, nbins, labels=False, duplicates="drop")
                labels = [f"q{int(x)}" for x in b]
            except ValueError:
                labels = ["q0"] * n
        df.loc[g.index, "size_stratum"] = [f"{cohort}|{v}" for v in labels]

    for m in STAT_METRICS:
        df[f"z_{m}"] = 0.0
        df[f"dev_{m}"] = 0.0
    df["max_z"] = 0.0
    df["warn_reasons"] = ""

    for _stratum, grp in df[present].groupby("size_stratum"):
        for m, direction in STAT_METRICS.items():
            vals = grp[m].astype(float)
            # -1.0 is the sentinel for "undefined" (e.g. absent oracle group)
            valid = vals > -0.5
            v = vals[valid]
            if len(v) < 8:
                continue
            med = float(v.median())
            mad = float((v - med).abs().median())
            if mad < 1e-9:
                continue
            dev = (med - v) if direction == "low" else (v - med)
            z = 0.6745 * dev / mad
            df.loc[z.index, f"z_{m}"] = z.to_numpy()
            df.loc[dev.index, f"dev_{m}"] = dev.to_numpy()

    # Effective z: significance gated by a practically meaningful deviation.
    for m in STAT_METRICS:
        eff = np.where(df[f"dev_{m}"] > MIN_EFFECT[m], df[f"z_{m}"], 0.0)
        df[f"effz_{m}"] = eff
        df["max_z"] = np.maximum(df["max_z"], eff)
    return df


def assign_flags(df: pd.DataFrame, floor: float) -> pd.DataFrame:
    """Populate ``flag``, ``fail_reasons``, ``warn_reasons`` columns."""
    df = df.copy()
    fails, warns, flags = [], [], []
    for _, r in df.iterrows():
        fr = hard_fail_reasons(r, floor)
        wr = [m for m in STAT_METRICS if float(r.get(f"effz_{m}", 0.0)) > Z_WARN]
        ir = [m for m in STAT_METRICS if Z_INFO < float(r.get(f"effz_{m}", 0.0)) <= Z_WARN]
        fails.append(";".join(fr))
        warns.append(";".join(wr) if wr else ";".join(ir))
        if fr:
            flags.append("FAIL")
        elif wr:
            flags.append("WARN")
        elif ir:
            flags.append("INFO")
        else:
            flags.append("OK")
    df["fail_reasons"] = fails
    df["warn_reasons"] = warns
    df["flag"] = flags
    return df


def select_for_figures(df: pd.DataFrame) -> pd.DataFrame:
    """Pick which scans get a figure; logs every cap that is applied."""
    picks: list[pd.DataFrame] = []
    for cohort, g in df.groupby("cohort"):
        f = g[g["flag"] == "FAIL"].sort_values("max_z", ascending=False)
        if len(f) > MAX_FAIL_FIGS_PER_COHORT:
            logger.warning(
                "CAP: %s has %d FAIL scans; rendering only the first %d "
                "(all remain listed in audit_flags.csv)",
                cohort,
                len(f),
                MAX_FAIL_FIGS_PER_COHORT,
            )
            f = f.head(MAX_FAIL_FIGS_PER_COHORT)
        f = f.assign(fig_reason="FAIL")

        w = g[g["flag"] == "WARN"].sort_values("max_z", ascending=False)
        if len(w) > MAX_WARN_FIGS_PER_COHORT:
            logger.warning(
                "CAP: %s has %d WARN scans; rendering the %d worst by max_z",
                cohort,
                len(w),
                MAX_WARN_FIGS_PER_COHORT,
            )
            w = w.head(MAX_WARN_FIGS_PER_COHORT)
        w = w.assign(fig_reason="WARN")

        ok = g[(g["flag"] == "OK") & (g["gt_tc_vox_crop"] > 0)]
        ex = pd.DataFrame()
        if len(ok):
            med = ok["dice_tc_img"].median()
            ex = ok.assign(_d=(ok["dice_tc_img"] - med).abs()).nsmallest(
                N_EXEMPLARS_PER_COHORT, "_d"
            )
            ex = ex.drop(columns=["_d"]).assign(fig_reason="EXEMPLAR")

        empt = g[(g["gt_tc_vox"] == 0) & (g["error"].fillna("") == "")]
        if len(empt):
            empt = empt.head(1).assign(fig_reason="TC_EMPTY")

        chosen = pd.concat([f, w, ex, empt])

        # Guarantee: every cohort gets at least one figure even when nothing is
        # flagged and the EXEMPLAR pick above found no eligible scan (e.g. a
        # cohort whose scans are all TC-empty or all errored). Falls back to the
        # largest-TC scan, which is the most informative single view.
        if chosen.empty:
            usable = g[g["error"].fillna("") == ""]
            if usable.empty:
                usable = g
            fb = usable.nlargest(1, "gt_tc_vox").assign(fig_reason="COHORT_FALLBACK")
            logger.warning(
                "CAP/FALLBACK: %s had no FAIL/WARN/EXEMPLAR pick; forcing 1 figure (%s)",
                cohort,
                fb.iloc[0]["scan_id"],
            )
            chosen = fb

        picks.append(chosen)
    sel = pd.concat(picks).drop_duplicates(subset=["cohort", "scan_id"], keep="first")

    missing = set(df["cohort"].unique()) - set(sel["cohort"].unique())
    if missing:
        raise RuntimeError(f"figure selection covers no scan for cohorts: {sorted(missing)}")
    logger.info(
        "figure selection: %d scans across %d/%d cohorts",
        len(sel),
        sel["cohort"].nunique(),
        df["cohort"].nunique(),
    )
    return sel


def pick_slices(tc_hard: np.ndarray) -> np.ndarray:
    """Evenly-spaced axial indices spanning the GT-TC extent.

    Uses up to :data:`MAX_COLS` columns; a tumour spanning fewer slices yields
    that many columns (the 7-10 window in the spec, with a floor for tiny
    lesions).  Falls back to the whole box when TC is empty.
    """
    presence = tc_hard.sum(axis=(0, 1)) > 0
    z = np.where(presence)[0]
    if len(z) == 0:
        return np.linspace(0, tc_hard.shape[2] - 1, MAX_COLS, dtype=int)
    n = int(min(MAX_COLS, max(1, len(z))))
    return np.linspace(z[0], z[-1], n, dtype=int)


def _cmap_rgba(ch: np.ndarray, cmap: Any, alpha_max: float) -> np.ndarray:
    """RGBA overlay from a perceptual colormap; alpha proportional to value."""
    rgba = cmap(np.clip(ch, 0.0, 1.0)).astype(np.float32)
    rgba[:, :, 3] = np.clip(ch * alpha_max, 0.0, 1.0)
    return rgba


def _binary_rgba(ch: np.ndarray, rgb: tuple[float, float, float], alpha: float) -> np.ndarray:
    """Flat RGBA overlay for a binary mask (no gradation)."""
    h, w = ch.shape
    out = np.zeros((h, w, 4), dtype=np.float32)
    out[..., 0], out[..., 1], out[..., 2] = rgb
    out[..., 3] = (ch > 0.5).astype(np.float32) * alpha
    return out


def render_patient(
    *,
    scan_id: str,
    cohort: str,
    anat: np.ndarray,
    tc_hard: np.ndarray,
    netc_hard: np.ndarray,
    tc_soft: np.ndarray,
    netc_soft: np.ndarray,
    row: pd.Series,
    anat_name: str,
    out_png: Path,
) -> None:
    """Render the 4-row QC panel for one scan.

    Rows: GT-TC (binary) / Soft-TC (continuous) / GT-NETC (binary) /
    Soft-NETC (continuous).  Columns are evenly-spaced GT-TC tumour slices.
    All volumes are in the crop frame ``(192, 224, 192)``; the soft rows show
    the **cached latent mask upscaled x4**, i.e. the artifact the model
    actually consumes.
    """
    ks = pick_slices(tc_hard)
    ncol = len(ks)
    rows_spec = [
        ("GT TC (binary)", tc_hard, "binary", TC_RGB, TC_CMAP),
        ("Soft TC (cached latent x4)", tc_soft, "soft", TC_RGB, TC_CMAP),
        ("GT NETC (binary)", netc_hard, "binary", NETC_RGB, NETC_CMAP),
        ("Soft NETC (cached latent x4)", netc_soft, "soft", NETC_RGB, NETC_CMAP),
    ]
    # Floor the width: a small lesion yields few columns, and at 1.7 in/col the
    # 3-line metric suptitle gets clipped (observed on a 4-column FAIL panel,
    # which silently truncated the TC voxel count).
    fig_w = max(11.0, 1.7 * ncol)
    fig, axes = plt.subplots(4, ncol, figsize=(fig_w, 7.4), facecolor="black", squeeze=False)
    for ri, (label, vol, kind, rgb, cmap) in enumerate(rows_spec):
        for ci, k in enumerate(ks):
            ax = axes[ri][ci]
            ax.set_facecolor("black")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            a = np.rot90(anat[:, :, k])
            lo, hi = float(a.min()), float(a.max())
            if hi <= lo:
                lo, hi = 0.0, 1.0
            ax.imshow(a, cmap="gray", vmin=lo, vmax=hi)
            m = np.rot90(vol[:, :, k])
            if kind == "binary":
                ax.imshow(_binary_rgba(m, rgb, 0.55))
                if (m > 0.5).any():
                    ax.contour(m, levels=[0.5], colors=[rgb], linewidths=0.8, alpha=0.95)
            else:
                ax.imshow(_cmap_rgba(m, cmap, 0.80))
                if m.max() > 0.0:
                    ax.contour(
                        m, levels=CONTOUR_LEVELS, colors=[rgb], linewidths=CONTOUR_LW, alpha=0.9
                    )
            if ri == 0:
                ax.set_title(f"z={k}", color="white", fontsize=7, pad=2)
            if ci == 0:
                ax.set_ylabel(label, color="white", fontsize=7)

    flag = row["flag"]
    reasons = row["fail_reasons"] or row["warn_reasons"] or "-"
    sub = (
        f"TC={int(row['gt_tc_vox'])}vox NETC={int(row['gt_netc_vox'])}vox  "
        f"Dice_img={row['dice_tc_img']:.3f}  latIoU={row['lat_iou_tc']:.3f}  "
        f"cdist={row['lat_centroid_dist_tc']:.2f}vox  volratio={row['volratio_tc_img']:.3f}  "
        f"recompMAE={row['recompute_mae']:.2e}  maxz={row['max_z']:.1f}"
    )
    fig.suptitle(
        f"[{row['fig_reason']}/{flag}] {cohort} — {scan_id}   (anat={anat_name})\n"
        f"{sub}\nreasons: {reasons}",
        color="white",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_png, dpi=110, facecolor="black")
    plt.close(fig)


def render_all(sel: pd.DataFrame, base: Path, fig_dir: Path) -> int:
    """Render every selected scan, grouped by cohort to reuse open H5 handles."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    lut = {c[0]: (c[1], c[2]) for c in COHORTS}
    for cohort, g in sel.groupby("cohort"):
        img_rel, lat_rel = lut[cohort]
        with h5py.File(base / lat_rel, "r") as fl, h5py.File(base / img_rel, "r") as fi:
            lat_ids = [_decode(x) for x in fl["ids"][:]]
            lat_idx = {s: i for i, s in enumerate(lat_ids)}
            img_ids = [_decode(x) for x in fi["ids"][:]]
            img_idx = {s: i for i, s in enumerate(img_ids)}
            seqs = [s for s in ("t1c", "t1pre", "flair", "t2") if f"images/{s}" in fi]
            anat_name = seqs[0] if seqs else None
            for _, r in g.iterrows():
                sid = r["scan_id"]
                if sid not in lat_idx or sid not in img_idx or anat_name is None:
                    continue
                i, j = lat_idx[sid], img_idx[sid]
                label = fi["masks/tumor"][j].astype(np.int32)
                origin = fi["crop/origin"][j]
                spec = CropPadSpec(
                    crop_origin=(int(origin[0]), int(origin[1]), int(origin[2])),
                    native_shape=label.shape,
                    target_shape=LATENT_CROP_BOX,
                )
                hard = _crop_to_box(
                    np.stack([(label > 0) & (label != 2), label == 1]).astype(np.float32), spec
                )
                anat = _crop_to_box(fi[f"images/{anat_name}"][j][None].astype(np.float32), spec)[0]
                cached = fl["masks/tumor_latent_soft"][i].astype(np.float32)
                up = (
                    F.interpolate(
                        torch.from_numpy(cached).unsqueeze(0),
                        scale_factor=float(POOL_STRIDE),
                        mode="nearest",
                    )
                    .squeeze(0)
                    .numpy()
                )
                out = fig_dir / f"{cohort}__{r['fig_reason']}__{sid}.png".replace("/", "_")
                render_patient(
                    scan_id=sid,
                    cohort=cohort,
                    anat=anat,
                    tc_hard=hard[0],
                    netc_hard=hard[1],
                    tc_soft=up[0],
                    netc_soft=up[1],
                    row=r,
                    anat_name=anat_name,
                    out_png=out,
                )
                n += 1
                logger.info("figure %s", out.name)
    return n


def main() -> None:
    """Merge, flag, summarise, and render."""
    p = argparse.ArgumentParser(description="Flag mask-audit outliers and render QC figures.")
    p.add_argument("--in-dir", type=Path, required=True, help="dir with *__metrics.csv")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--base", type=str, required=True, help="dataset base dir")
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    csvs = sorted(args.in_dir.glob("*__metrics.csv"))
    if not csvs:
        raise SystemExit(f"no *__metrics.csv under {args.in_dir}")
    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    df["error"] = df["error"].fillna("")
    logger.info("loaded %d scans from %d cohorts", len(df), df["cohort"].nunique())

    floor = compute_floor()
    df = add_robust_z(df)
    df = assign_flags(df, floor)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "audit_metrics_all.csv", index=False)
    flagged = df[df["flag"] != "OK"].sort_values(["flag", "max_z"], ascending=[True, False])
    flagged.to_csv(args.out_dir / "audit_flags.csv", index=False)

    # ---- per-cohort summary ----
    recs = []
    for cohort, g in df.groupby("cohort"):
        ok = g[g["error"] == ""]
        tc = ok[ok["gt_tc_vox_crop"] > 0]
        recs.append(
            {
                "cohort": cohort,
                "n_scans": len(g),
                "n_error": int((g["error"] != "").sum()),
                "n_FAIL": int((g["flag"] == "FAIL").sum()),
                "n_WARN": int((g["flag"] == "WARN").sum()),
                "n_INFO": int((g["flag"] == "INFO").sum()),
                "n_OK": int((g["flag"] == "OK").sum()),
                "n_tc_empty": int((ok["gt_tc_vox"] == 0).sum()),
                "max_recompute_max_abs": float(ok["recompute_max_abs"].max()) if len(ok) else None,
                "median_dice_tc_img": float(tc["dice_tc_img"].median()) if len(tc) else None,
                "median_lat_iou_tc": float(tc["lat_iou_tc"].median()) if len(tc) else None,
                "median_lat_centroid_dist": float(tc["lat_centroid_dist_tc"].median())
                if len(tc)
                else None,
                "median_volratio_tc": float(tc["volratio_tc_img"].median()) if len(tc) else None,
                "median_intermediate_frac": float(tc["soft_intermediate_frac_tc"].median())
                if len(tc)
                else None,
                "max_tc_outside_brain_frac": float(ok["tc_outside_brain_frac"].max())
                if len(ok)
                else None,
            }
        )
    per_cohort = pd.DataFrame(recs).sort_values("cohort")
    per_cohort.to_csv(args.out_dir / "per_cohort_summary.csv", index=False)

    reason_counts: dict[str, int] = {}
    for s in df["fail_reasons"]:
        for tok in [t for t in str(s).split(";") if t]:
            reason_counts[tok.split(":")[0]] = reason_counts.get(tok.split(":")[0], 0) + 1
    for s in df["warn_reasons"]:
        for tok in [t for t in str(s).split(";") if t]:
            reason_counts[tok] = reason_counts.get(tok, 0) + 1

    n_figs = 0
    sel = select_for_figures(df)
    if not args.no_figures:
        n_figs = render_all(sel, Path(args.base), args.out_dir / "figures")
    sel[
        ["cohort", "scan_id", "flag", "fig_reason", "max_z", "fail_reasons", "warn_reasons"]
    ].to_csv(args.out_dir / "figures_index.csv", index=False)

    summary = {
        "n_scans": len(df),
        "n_cohorts": int(df["cohort"].nunique()),
        "expected_floor": floor,
        "thresholds": {"z_warn": Z_WARN, "z_info": Z_INFO},
        "stat_metrics": STAT_METRICS,
        "counts": {
            "FAIL": int((df["flag"] == "FAIL").sum()),
            "WARN": int((df["flag"] == "WARN").sum()),
            "INFO": int((df["flag"] == "INFO").sum()),
            "OK": int((df["flag"] == "OK").sum()),
            "error": int((df["error"] != "").sum()),
        },
        "reason_counts": reason_counts,
        "n_figures": n_figs,
        "caps": {
            "max_fail_figs_per_cohort": MAX_FAIL_FIGS_PER_COHORT,
            "max_warn_figs_per_cohort": MAX_WARN_FIGS_PER_COHORT,
        },
        "global_max_recompute_max_abs": float(df.loc[df["error"] == "", "recompute_max_abs"].max()),
    }
    (args.out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("summary: %s", json.dumps(summary["counts"]))
    logger.info("wrote %s", args.out_dir)


if __name__ == "__main__":
    main()
