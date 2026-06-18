"""Composite-figure probe: visualise every component the LPL loss consumes.

For one TRAIN-set UCSF-PDGM patient × 5 augmentation variants (v0..v4):

  1. VenaFMAdapter (src/vena/inference/adapters/vena_fm_adapter.py) loads the
     S1+FFT module from <run_dir>/config.yaml + ema_best.ckpt — trunk +
     ControlNet + EMA shadows + EulerSampler + frozen MAISI VAE all wired up.
  2. Per variant: read (z_t1pre, z_t2, z_flair, z_t1c, m_wt, m_brain) from the
     clean H5 (v0) or the augmented H5 (v1..v4), run a deterministic 10-step
     Euler RFlow sample, decode real+pred T1c with block-2/5 hooks.
  3. Per variant emit one 7-row × 7-col composite figure:

       row 1 — real T1c + WT (NETC/ED/ET) + brain contour overlay
       row 2 — real T1c, no overlay
       row 3 — predicted T1c (S1+FFT)
       row 4 — block-2 features of real T1c   (std-norm per LPL §3.3)
       row 5 — block-2 features of pred  T1c
       row 6 — block-5 features of real T1c
       row 7 — block-5 features of pred  T1c

  Columns are 7 equally-spaced depth slices (offset 10 from each end of the
  192-deep brain box). Feature rows show the resolution-corresponding slice.
  Background black; feature cmap "inferno" (black at low end).

Standardisation matches LPL spec: per-channel z-score on prediction-derived
stats, applied to both pred and target features, then channel-mean for display.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(REPO_SRC))

import h5py
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
# --------------------------------------------------------------------------
# Paths (Picasso filesystem) — cohort selectable via PROBE_COHORT env var.
# --------------------------------------------------------------------------
import os as _os

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

from vena.common.decode import decode_box
from vena.inference.adapters.vena_fm_adapter import VenaFMAdapter
from vena.model.fm.eval.exhaustive import build_crop_spec_from_h5

RUN_DIR = "/mnt/home/users/tic_163_uma/mpascual/execs/vena/experiments/2026-06-12_01-27-55_s1_fft_cfm_c9b97556"
VAE_CKPT = "/mnt/home/users/tic_163_uma/mpascual/fscratch/checkpoints/NV-Generate-MR/models/autoencoder_v2.pt"

_VENA = "/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/vena"
COHORTS = {
    "UCSF-PDGM": (
        f"{_VENA}/UCSF_PDGM/h5/UCSFPDGM_latents.h5",
        f"{_VENA}/UCSF_PDGM/h5/ucsf_pdgm_latents_aug.h5",
        f"{_VENA}/UCSF_PDGM/h5/UCSFPDGM_image.h5",
    ),
    "BraTS-GLI": (
        f"{_VENA}/BRATS_GLI/PRE_OPERATIVE/h5/BraTS_GLI_latents.h5",
        f"{_VENA}/BRATS_GLI/PRE_OPERATIVE/h5/brats_gli_latents_aug.h5",
        f"{_VENA}/BRATS_GLI/PRE_OPERATIVE/h5/BraTS_GLI_image.h5",
    ),
    "UPENN-GBM": (
        f"{_VENA}/upenn_gbm/h5/UPENN-GBM_latents.h5",
        f"{_VENA}/upenn_gbm/h5/upenn_gbm_latents_aug.h5",
        f"{_VENA}/upenn_gbm/h5/UPENN-GBM_image.h5",
    ),
    "LUMIERE": (
        f"{_VENA}/lumiere/h5/LUMIERE_latents.h5",
        f"{_VENA}/lumiere/h5/lumiere_latents_aug.h5",
        f"{_VENA}/lumiere/h5/LUMIERE_image.h5",
    ),
    "REMBRANDT": (
        f"{_VENA}/rembrandt/h5/REMBRANDT_latents.h5",
        f"{_VENA}/rembrandt/h5/rembrandt_latents_aug.h5",
        f"{_VENA}/rembrandt/h5/REMBRANDT_image.h5",
    ),
}
COHORT = _os.environ.get("PROBE_COHORT", "UCSF-PDGM")
LAT_CLEAN, LAT_AUG, IMG_H5 = COHORTS[COHORT]
OUT = Path(__file__).resolve().parent / "out_composite" / COHORT
OUT.mkdir(parents=True, exist_ok=True)

NFE = 10
SEED = 42
WT_THRESHOLD = 0.5
DEVICE = torch.device("cuda:0")

# Image-space depth slice indices (192-deep brain box; offset 10 from ends).
IMG_DEPTH = 192
LAT_DEPTH = 48
B5_DEPTH = 96
SLICE_OFFSET = 10
N_COLS = 9
IMG_SLICES = np.linspace(SLICE_OFFSET, IMG_DEPTH - SLICE_OFFSET, N_COLS).astype(int)
LAT_SLICES = np.clip((IMG_SLICES / 4).round().astype(int), 0, LAT_DEPTH - 1)
B5_SLICES = np.clip((IMG_SLICES / 2).round().astype(int), 0, B5_DEPTH - 1)


def _decode_str(x):
    return x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x


def _pick_train_patient(min_wt_voxels: int = 4000):
    """Pick a train patient that is (a) in the aug H5 and (b) carries a WT mask
    with >= ``min_wt_voxels`` at latent resolution — for visualisation clarity.

    Picks the patient with the LARGEST WT among the eligible set, so the
    NETC/ED/ET overlay on row 1 is readily visible.
    """
    with h5py.File(LAT_CLEAN, "r") as f:
        all_ids = [_decode_str(x) for x in f["ids"][:]]
        # Some cohorts have splits/cv/fold_0/train; others don't (defensively skip filter).
        try:
            train_ids_set = {_decode_str(x) for x in f["splits/cv/fold_0/train"][:]}
        except KeyError:
            print("  no splits/cv/fold_0/train — skipping train filter")
            train_ids_set = set(all_ids)
    with h5py.File(LAT_AUG, "r") as f:
        aug_ids = {_decode_str(x) for x in f["ids"][:]}

    def _is_train(scan_id, train_set):
        """BraTS-GLI / LUMIERE store SCAN-level ids in ids/ but PATIENT-level
        ids in splits/. UCSF / UPENN / REMBRANDT match directly."""
        if scan_id in train_set:
            return True
        # BraTS-GLI: 'BraTS-GLI-00000-000' → patient 'BraTS-GLI-00000'.
        parts = scan_id.rsplit("-", 1)
        if len(parts) == 2 and parts[0] in train_set:
            return True
        # LUMIERE: 'Patient-001__week-000-1' → patient 'Patient-001'.
        if "__" in scan_id and scan_id.split("__")[0] in train_set:
            return True
        return False

    eligible_rows = [
        (i, pid)
        for i, pid in enumerate(all_ids)
        if _is_train(pid, train_ids_set) and pid in aug_ids
    ]
    # Score each eligible patient by WT voxel count at latent resolution.
    best = (None, -1, -1)
    with h5py.File(LAT_CLEAN, "r") as f:
        tlat = f["masks/tumor_latent"]
        for row, pid in eligible_rows:
            soft = np.clip(tlat[row].sum(0), 0.0, 1.0)
            n_wt = int((soft >= WT_THRESHOLD).sum())
            if n_wt > best[2]:
                best = (pid, row, n_wt)
    if best[0] is None or best[2] < min_wt_voxels:
        # Fall back to the largest available, even below threshold.
        if best[0] is None:
            raise RuntimeError("No train patient in aug H5")
        print(
            f"WARN: best WT voxel count is {best[2]} < min={min_wt_voxels}; using {best[0]} anyway"
        )
    print(f"  picker: {best[0]} (row {best[1]}, n_wt_voxels={best[2]})")
    return best[0], best[1]


def _load_clean(row):
    with h5py.File(LAT_CLEAN, "r") as f:
        d = {
            "z_t1pre": f["latents/t1pre"][row].astype(np.float32),
            "z_t2": f["latents/t2"][row].astype(np.float32),
            "z_flair": f["latents/flair"][row].astype(np.float32),
            "z_t1c": f["latents/t1c"][row].astype(np.float32),
            "tumor_lat": f["masks/tumor_latent"][row].astype(np.float32),
            "brain_lat": (
                f["masks/brain_latent"][row].astype(np.float32)
                if "masks/brain_latent" in f
                else None
            ),
            "aug_params": None,
        }
    return d


def _load_aug(pid):
    out = {}
    with h5py.File(LAT_AUG, "r") as f:
        all_ids = [_decode_str(x) for x in f["ids"][:]]
        all_variants = [_decode_str(x) for x in f["variants"][:]]
        all_params = (
            [_decode_str(x) for x in f["aug_params_json"][:]]
            if "aug_params_json" in f
            else [None] * len(all_ids)
        )
        rows = [i for i, p in enumerate(all_ids) if p == pid]
        for row in rows:
            v = all_variants[row]
            d = {
                "z_t1pre": f["latents/t1pre"][row].astype(np.float32),
                "z_t2": f["latents/t2"][row].astype(np.float32),
                "z_flair": f["latents/flair"][row].astype(np.float32),
                "z_t1c": f["latents/t1c"][row].astype(np.float32),
                "tumor_lat": f["masks/tumor_latent"][row].astype(np.float32),
                "brain_lat": (
                    f["masks/brain_latent"][row].astype(np.float32)
                    if "masks/brain_latent" in f
                    else None
                ),
                "aug_params": all_params[row],
            }
            out[v] = d
    return out


def _to_batch(d):
    z_t1pre = torch.from_numpy(d["z_t1pre"]).unsqueeze(0).to(DEVICE)
    z_t2 = torch.from_numpy(d["z_t2"]).unsqueeze(0).to(DEVICE)
    z_flair = torch.from_numpy(d["z_flair"]).unsqueeze(0).to(DEVICE)
    z_t1c = torch.from_numpy(d["z_t1c"]).unsqueeze(0).to(DEVICE)
    soft_wt = np.clip(d["tumor_lat"].sum(0, keepdims=True), 0.0, 1.0)
    m_wt = torch.from_numpy((soft_wt >= WT_THRESHOLD).astype(np.float32)).unsqueeze(0).to(DEVICE)
    if d["brain_lat"] is not None:
        brain_lat = d["brain_lat"]
    else:
        brain_lat = (np.abs(d["z_t1c"]).sum(0, keepdims=True) > 0).astype(np.float32)
    m_brain = torch.from_numpy(brain_lat.astype(np.float32)).unsqueeze(0).to(DEVICE)
    return {
        "z_t1pre": z_t1pre,
        "z_t2": z_t2,
        "z_flair": z_flair,
        "z_t1c": z_t1c,
        "m_wt": m_wt,
        "m_brain": m_brain,
        "tumor_lat": torch.from_numpy(d["tumor_lat"]).unsqueeze(0).to(DEVICE),
    }


def _decode_with_hooks(vae, z, crop_spec, blocks):
    captures = {}

    def make(idx):
        def hook(_m, _i, o):
            captures[idx] = o.detach().float().clone()

        return hook

    handles = [vae.handle.model.decoder.blocks[i].register_forward_hook(make(i)) for i in blocks]
    try:
        img = decode_box(vae, z, crop_spec)
    finally:
        for h in handles:
            h.remove()
    return img, captures


def _standardize(feat_pred, feat_real):
    f_pred = feat_pred[0]
    f_real = feat_real[0]
    mean = f_pred.mean(dim=(-3, -2, -1), keepdim=True)
    std = f_pred.std(dim=(-3, -2, -1), keepdim=True).clamp_min(1e-6)
    pn = (f_pred - mean) / std
    rn = (f_real - mean) / std
    return rn.mean(0).cpu().numpy(), pn.mean(0).cpu().numpy()


def _nn_upsample(mask_lat, target_shape):
    x = mask_lat.unsqueeze(0).float()
    y = F.interpolate(x, size=target_shape, mode="nearest")
    return y[0]


def _norm01(arr):
    lo, hi = np.percentile(arr, 1.0), np.percentile(arr, 99.0)
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _overlay_masks(ax, t1c, tumor3, brain, colors, alpha=0.6):
    ax.imshow(_norm01(t1c), cmap="gray", vmin=0, vmax=1)
    rgba = np.zeros((tumor3.shape[1], tumor3.shape[2], 4), dtype=np.float32)
    for ch, color in enumerate(colors):
        m = (tumor3[ch] >= WT_THRESHOLD).astype(np.float32)
        if m.sum() == 0:
            continue
        col = np.array(to_rgba(color))
        rgba[..., 0] = np.where(m > 0, col[0], rgba[..., 0])
        rgba[..., 1] = np.where(m > 0, col[1], rgba[..., 1])
        rgba[..., 2] = np.where(m > 0, col[2], rgba[..., 2])
        rgba[..., 3] = np.where(m > 0, alpha * col[3], rgba[..., 3])
    ax.imshow(rgba, interpolation="nearest")
    if brain is not None and brain.max() > 0:
        ax.contour(brain, levels=[0.5], colors="white", linewidths=0.4, alpha=0.7)


def _build_figure(
    pid,
    v_name,
    real,
    pred,
    tumor,
    brain,
    m_wt_lat,
    cond_t1pre_lat,
    cond_t2_lat,
    cond_flair_lat,
    b2r,
    b2p,
    b5r,
    b5p,
    aug_params,
    fig_path,
):
    """13-row × N_COLS-col grouped composite figure.

    Row groups (with visual gaps):
      Group 1 (anatomy + supervisor): row 1 = real T1c + WT/brain overlay,
                                       row 2 = m_wt at LATENT resolution (binary, pre-NN-upsample)
      Group 2 (LM conditioning):     rows 3-5 = z_t1pre, z_t2, z_flair channel-mean (latent res)
      Group 3 (image-space target):  row 6 = real T1c, row 7 = predicted T1c
      Group 4 (block 2 LPL ops):     row 8 = φ′(real), row 9 = φ′(pred), row 10 = |Δφ′|
      Group 5 (block 5 LPL ops):     row 11 = φ′(real), row 12 = φ′(pred), row 13 = |Δφ′|
    """
    from matplotlib.gridspec import GridSpec

    plt.rcParams.update(
        {
            "axes.facecolor": "black",
            "figure.facecolor": "black",
            "text.color": "white",
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "axes.edgecolor": "white",
        }
    )

    # Define group structure: list of (label, content_type, slice_set, cmap, vrange)
    GAP = 0.22
    groups = [
        [  # Group 1
            ("T1c real + WT/brain overlay", "overlay"),
            ("m_wt  (binary, latent res 48×56×48)", "m_wt_lat"),
        ],
        [  # Group 2
            ("z_t1pre  (latent channel-mean)", "cond_t1pre"),
            ("z_t2     (latent channel-mean)", "cond_t2"),
            ("z_flair  (latent channel-mean)", "cond_flair"),
        ],
        [  # Group 3
            ("T1c real (decoded z_t1c)", "t1c_real"),
            ("T1c predicted (S1+FFT, Euler-10)", "t1c_pred"),
        ],
        [  # Group 4
            ("block 2  φ′(D; z_t1c)", "b2_real"),
            ("block 2  φ′(D; ẑ_t1c)", "b2_pred"),
            ("block 2  |Δφ′|", "b2_delta"),
        ],
        [  # Group 5
            ("block 5  φ′(D; z_t1c)", "b5_real"),
            ("block 5  φ′(D; ẑ_t1c)", "b5_pred"),
            ("block 5  |Δφ′|", "b5_delta"),
        ],
    ]

    # Build height_ratios with gaps between groups
    height_ratios = []
    for i, grp in enumerate(groups):
        height_ratios.extend([1.0] * len(grp))
        if i < len(groups) - 1:
            height_ratios.append(GAP)
    total_rows = len(height_ratios)

    # Figure size: 9 cols × ~2.0in each = 18in wide; rows × 1.9 in each
    n_actual_rows = sum(len(g) for g in groups)
    fig_w = 2.0 * N_COLS
    fig_h = 1.9 * n_actual_rows + 0.4 * (len(groups) - 1) + 1.2  # extra for title+cbars
    fig = plt.figure(figsize=(fig_w, fig_h))
    title = f"LPL components — {pid}, variant {v_name}"
    if aug_params:
        title += f"\naug_params={str(aug_params)[:160]}"
    fig.suptitle(title, color="white", fontsize=10, y=0.995)

    gs = GridSpec(
        total_rows,
        N_COLS,
        figure=fig,
        height_ratios=height_ratios,
        hspace=0.05,
        wspace=0.04,
        top=0.945,
        bottom=0.085,
        left=0.085,
        right=0.985,
    )

    # Resolve content type → axes mapping
    axes_by_kind = {}
    grid_row = 0
    for gi, grp in enumerate(groups):
        for label, kind in grp:
            axes_by_kind[kind] = ([fig.add_subplot(gs[grid_row, j]) for j in range(N_COLS)], label)
            grid_row += 1
        if gi < len(groups) - 1:
            grid_row += 1  # skip gap row

    colors = ["#0080ff", "#00ff80", "#ff3030"]  # NETC, ED, ET

    def _range(*arrs):
        v = np.concatenate([a.flatten() for a in arrs])
        return float(np.percentile(v, 1.0)), float(np.percentile(v, 99.0))

    # Feature range (rows 8-9 and 11-12): shared between real+pred
    b2_lo, b2_hi = _range(b2r, b2p)
    b5_lo, b5_hi = _range(b5r, b5p)
    # |Δφ| range (rows 10 and 13): shared 99-pct across both blocks for comparability
    delta_b2 = np.abs(b2r - b2p)
    delta_b5 = np.abs(b5r - b5p)
    d_b2_hi = float(np.percentile(delta_b2, 99.0))
    d_b5_hi = float(np.percentile(delta_b5, 99.0))

    cmap_feat = "inferno"
    cmap_delta = "magma"
    feat_im_handle = None
    delta_im_handle = None

    # Per-modality conditioning normalisation: percentile clip per tensor.
    def _norm_cond(arr):
        lo, hi = np.percentile(arr, 1.0), np.percentile(arr, 99.0)
        if hi - lo < 1e-9:
            return np.zeros_like(arr)
        return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)

    cond_t1pre_disp = _norm_cond(cond_t1pre_lat)
    cond_t2_disp = _norm_cond(cond_t2_lat)
    cond_flair_disp = _norm_cond(cond_flair_lat)

    for j, z_img in enumerate(IMG_SLICES):
        z_lat = LAT_SLICES[j]
        z_b5 = B5_SLICES[j]

        # --- Group 1 ---
        ax_overlay = axes_by_kind["overlay"][0][j]
        _overlay_masks(
            ax_overlay,
            real[..., z_img],
            tumor[..., z_img],
            brain[..., z_img] if brain is not None else None,
            colors,
            alpha=0.6,
        )
        ax_overlay.set_title(f"img z={z_img}  /  lat z={z_lat}", color="white", fontsize=7.5)

        ax_mwt = axes_by_kind["m_wt_lat"][0][j]
        ax_mwt.imshow(m_wt_lat[..., z_lat], cmap="gray", vmin=0, vmax=1)

        # --- Group 2 (conditioning latents, channel-mean, latent res) ---
        axes_by_kind["cond_t1pre"][0][j].imshow(
            cond_t1pre_disp[..., z_lat], cmap="inferno", vmin=0, vmax=1
        )
        axes_by_kind["cond_t2"][0][j].imshow(
            cond_t2_disp[..., z_lat], cmap="inferno", vmin=0, vmax=1
        )
        axes_by_kind["cond_flair"][0][j].imshow(
            cond_flair_disp[..., z_lat], cmap="inferno", vmin=0, vmax=1
        )

        # --- Group 3 ---
        axes_by_kind["t1c_real"][0][j].imshow(
            _norm01(real[..., z_img]), cmap="gray", vmin=0, vmax=1
        )
        axes_by_kind["t1c_pred"][0][j].imshow(
            _norm01(pred[..., z_img]), cmap="gray", vmin=0, vmax=1
        )

        # --- Group 4 (block 2 LPL) ---
        im_f = axes_by_kind["b2_real"][0][j].imshow(
            b2r[..., z_lat], cmap=cmap_feat, vmin=b2_lo, vmax=b2_hi
        )
        if feat_im_handle is None:
            feat_im_handle = im_f
        axes_by_kind["b2_pred"][0][j].imshow(
            b2p[..., z_lat], cmap=cmap_feat, vmin=b2_lo, vmax=b2_hi
        )
        im_d = axes_by_kind["b2_delta"][0][j].imshow(
            delta_b2[..., z_lat], cmap=cmap_delta, vmin=0, vmax=d_b2_hi
        )
        if delta_im_handle is None:
            delta_im_handle = im_d

        # --- Group 5 (block 5 LPL) ---
        axes_by_kind["b5_real"][0][j].imshow(b5r[..., z_b5], cmap=cmap_feat, vmin=b5_lo, vmax=b5_hi)
        axes_by_kind["b5_pred"][0][j].imshow(b5p[..., z_b5], cmap=cmap_feat, vmin=b5_lo, vmax=b5_hi)
        axes_by_kind["b5_delta"][0][j].imshow(
            delta_b5[..., z_b5], cmap=cmap_delta, vmin=0, vmax=d_b5_hi
        )

    # Strip ticks; set row labels
    for kind, (ax_row, label) in axes_by_kind.items():
        for j, ax in enumerate(ax_row):
            ax.set_xticks([])
            ax.set_yticks([])
        ax_row[0].set_ylabel(label, color="white", fontsize=8.0)

    # Legend
    legend = [
        Patch(facecolor=colors[0], edgecolor="none", label="NETC"),
        Patch(facecolor=colors[1], edgecolor="none", label="ED"),
        Patch(facecolor=colors[2], edgecolor="none", label="ET"),
        Patch(facecolor="none", edgecolor="white", label="brain contour"),
    ]
    fig.legend(
        handles=legend,
        loc="lower left",
        ncol=4,
        frameon=False,
        labelcolor="white",
        fontsize=9,
        bbox_to_anchor=(0.02, 0.012),
    )

    # Two horizontal colorbars at the bottom: features (inferno) + |Δφ| (magma)
    cax_feat = fig.add_axes([0.30, 0.044, 0.30, 0.010])
    cb1 = fig.colorbar(feat_im_handle, cax=cax_feat, orientation="horizontal")
    cb1.set_label(
        "φ′ standardised feature (z-score; pred-derived stats per LPL §3.3)",
        color="white",
        fontsize=7.5,
    )
    cb1.ax.xaxis.set_tick_params(color="white", labelsize=6.5)
    plt.setp(cb1.ax.get_xticklabels(), color="white")

    cax_delta = fig.add_axes([0.66, 0.044, 0.30, 0.010])
    cb2 = fig.colorbar(delta_im_handle, cax=cax_delta, orientation="horizontal")
    cb2.set_label("|Δφ′|  channel-mean of |φ′(D; ẑ) − φ′(D; z_t1c)|", color="white", fontsize=7.5)
    cb2.ax.xaxis.set_tick_params(color="white", labelsize=6.5)
    plt.setp(cb2.ax.get_xticklabels(), color="white")

    plt.savefig(fig_path, dpi=120, facecolor="black", bbox_inches="tight")
    plt.close()


def main():
    print("=== LPL composite figure probe ===")
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        sys.exit(1)
    print(f"cuda:0 = {torch.cuda.get_device_properties(0).name}")
    print()

    adapter = VenaFMAdapter(
        name="composite_probe",
        run_dir=RUN_DIR,
        checkpoint="ema_best.ckpt",
        vae_checkpoint=VAE_CKPT,
        device=DEVICE,
        nfe_list=(NFE,),
        selection_nfe=NFE,
    )
    print(f"Loading checkpoint from {adapter.checkpoint_path}...")
    adapter.setup()
    module = adapter._module
    sampler = adapter._sampler
    vae = adapter._vae
    print("Adapter ready.\n")

    pid, row = _pick_train_patient()
    print(f"Selected train patient: {pid} (clean row {row})")
    v0 = _load_clean(row)
    v_aug = _load_aug(pid)
    variants = {
        "v0": v0,
        **{k: v_aug[k] for k in ("v1", "v2", "v3", "v4") if k in v_aug},
    }
    print(f"Variants: {list(variants.keys())}\n")

    crop_spec = build_crop_spec_from_h5(IMG_H5, pid)
    print(f"crop_spec target_shape={crop_spec.target_shape}\n")

    out_dir = OUT / pid
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"patient_id": pid, "run_dir": RUN_DIR, "nfe": NFE, "variants": {}}

    for v_name, d in variants.items():
        print(f"--- {v_name} ---")
        batch = _to_batch(d)
        module.compute_val_conditioning(batch)
        torch.manual_seed(SEED)
        x0 = torch.randn_like(batch["z_t1c"])
        model_call = module._make_ema_call()
        with torch.inference_mode():
            z_pred = sampler.sample(model_call, x0, num_inference_steps=NFE)
        print(f"  z_pred finite={torch.isfinite(z_pred).all().item()}")

        real_img, feats_real = _decode_with_hooks(vae, batch["z_t1c"], crop_spec, [2, 5])
        pred_img, feats_pred = _decode_with_hooks(vae, z_pred, crop_spec, [2, 5])
        real = real_img.cpu().float().numpy()
        pred = pred_img.cpu().float().numpy()
        print(f"  decoded: real {real.shape}, pred {pred.shape}")

        b2r, b2p = _standardize(feats_pred[2], feats_real[2])
        b5r, b5p = _standardize(feats_pred[5], feats_real[5])
        print(f"  features: b2 {b2r.shape}, b5 {b5r.shape}")

        # Also keep the raw per-channel STANDARDISED tensors for LPL stats.
        def _std_per_channel(feat_pred, feat_real):
            """Per-channel z-score on pred stats, return (C,H,W,D) std tensors."""
            f_pred = feat_pred[0]
            f_real = feat_real[0]
            mean = f_pred.mean(dim=(-3, -2, -1), keepdim=True)
            std = f_pred.std(dim=(-3, -2, -1), keepdim=True).clamp_min(1e-6)
            return ((f_real - mean) / std).cpu().numpy(), ((f_pred - mean) / std).cpu().numpy()

        phi_b2_real_C, phi_b2_pred_C = _std_per_channel(feats_pred[2], feats_real[2])
        phi_b5_real_C, phi_b5_pred_C = _std_per_channel(feats_pred[5], feats_real[5])

        target_hwd = real.shape
        tumor_img = _nn_upsample(batch["tumor_lat"][0], target_hwd).cpu().numpy()
        brain_img = _nn_upsample(batch["m_brain"][0], target_hwd)[0].cpu().numpy()

        # Latent-resolution extras for the new figure rows.
        m_wt_lat_np = batch["m_wt"][0, 0].cpu().numpy()  # (h, w, d) binary at latent res
        cond_t1pre_lat = batch["z_t1pre"][0].mean(0).cpu().numpy()  # ch-mean (h, w, d)
        cond_t2_lat = batch["z_t2"][0].mean(0).cpu().numpy()
        cond_flair_lat = batch["z_flair"][0].mean(0).cpu().numpy()

        fig_path = out_dir / f"composite_{v_name}.png"
        _build_figure(
            pid,
            v_name,
            real,
            pred,
            tumor_img,
            brain_img,
            m_wt_lat_np,
            cond_t1pre_lat,
            cond_t2_lat,
            cond_flair_lat,
            b2r,
            b2p,
            b5r,
            b5p,
            d.get("aug_params"),
            fig_path,
        )
        print(f"  → {fig_path}")

        def _mask_at(target):
            mw = _nn_upsample(batch["m_wt"][0], target)[0].cpu().numpy()
            mb = _nn_upsample(batch["m_brain"][0], target)[0].cpu().numpy()
            return mw, mb

        delta_b2 = np.abs(b2r - b2p)
        delta_b5 = np.abs(b5r - b5p)
        mw_b2, mb_b2 = _mask_at(b2r.shape)
        mw_b5, mb_b5 = _mask_at(b5r.shape)

        def _proxy_stats(delta, wt, brain):
            """Channel-mean |Δφ′| stats — matches what the figure visualises."""
            wt_b = wt.astype(bool).flatten()
            bg_b = (brain.astype(bool) & ~wt.astype(bool)).flatten()
            flat = delta.flatten()
            r = {}
            for tag, m in (("WT", wt_b), ("brain_minus_WT", bg_b)):
                n = int(m.sum())
                if n == 0:
                    r[tag] = {"n": 0}
                    continue
                vs = flat[m]
                r[tag] = {
                    "n": n,
                    "mean": float(vs.mean()),
                    "p99": float(np.percentile(vs, 99)),
                    "max": float(vs.max()),
                }
            return r

        def _lpl_stats(phi_real_C, phi_pred_C, wt, brain):
            """Actual LPL scalars from per-channel standardised features.

            phi_*_C: (C, H, W, D) numpy arrays.
            wt, brain: (H, W, D) at feature resolution.

            Returns the scalars LPL would actually output (mean over channels
            and voxels of squared difference of standardised features) per
            region, plus per-channel signal distribution stats so the user can
            judge how much of the loss is concentrated in a few channels
            (relevant to the "block 2 looks like noise" question).
            """
            C, H, W, D = phi_real_C.shape
            wt_b = wt.astype(bool)
            bg_b = brain.astype(bool) & ~wt_b
            diff_sq = (phi_pred_C - phi_real_C) ** 2  # (C, H, W, D)

            # Per-channel scalar L_dec across all voxels (region-uniform).
            chan_L_global = diff_sq.mean(axis=(1, 2, 3))  # (C,)
            # L_dec per region (mean over channels of mean over voxels).
            if wt_b.sum() > 0:
                L_dec_WT = float(diff_sq[:, wt_b].mean())
                chan_L_WT = diff_sq[:, wt_b].mean(axis=1)  # (C,)
            else:
                L_dec_WT, chan_L_WT = None, None
            if bg_b.sum() > 0:
                L_dec_notWT = float(diff_sq[:, bg_b].mean())
                chan_L_notWT = diff_sq[:, bg_b].mean(axis=1)
            else:
                L_dec_notWT, chan_L_notWT = None, None
            L_dec_global = float(diff_sq.mean())

            # Channel concentration: fraction of total loss carried by top-K channels.
            order = np.argsort(chan_L_global)[::-1]
            sorted_L = chan_L_global[order]
            cum = np.cumsum(sorted_L) / max(sorted_L.sum(), 1e-12)
            n_top_50pct = int(np.searchsorted(cum, 0.50) + 1)  # channels for 50 % of L
            n_top_80pct = int(np.searchsorted(cum, 0.80) + 1)
            n_top_95pct = int(np.searchsorted(cum, 0.95) + 1)

            # Per-channel WT/notWT ratio (where LPL focuses inside vs outside WT)
            chan_ratio = None
            if chan_L_WT is not None and chan_L_notWT is not None:
                chan_ratio = chan_L_WT / np.clip(chan_L_notWT, 1e-12, None)

            return {
                "n_channels": int(C),
                "L_dec_global": L_dec_global,
                "L_dec_WT": L_dec_WT,
                "L_dec_notWT": L_dec_notWT,
                "ratio_L_dec_WT_over_notWT": (
                    float(L_dec_WT / L_dec_notWT) if (L_dec_WT and L_dec_notWT) else None
                ),
                "per_channel_L_dec_global": {
                    "min": float(chan_L_global.min()),
                    "p10": float(np.percentile(chan_L_global, 10)),
                    "p50": float(np.percentile(chan_L_global, 50)),
                    "p90": float(np.percentile(chan_L_global, 90)),
                    "p99": float(np.percentile(chan_L_global, 99)),
                    "max": float(chan_L_global.max()),
                    "std": float(chan_L_global.std()),
                },
                "channel_concentration": {
                    "n_top_channels_for_50pct_loss": n_top_50pct,
                    "n_top_channels_for_80pct_loss": n_top_80pct,
                    "n_top_channels_for_95pct_loss": n_top_95pct,
                    "max_over_median_ratio": float(
                        chan_L_global.max() / max(np.percentile(chan_L_global, 50), 1e-12)
                    ),
                },
                "per_channel_WT_over_notWT_ratio": (
                    {
                        "p50": float(np.percentile(chan_ratio, 50)),
                        "p90": float(np.percentile(chan_ratio, 90)),
                        "max": float(chan_ratio.max()),
                    }
                    if chan_ratio is not None
                    else None
                ),
            }

        summary["variants"][v_name] = {
            "aug_params": d.get("aug_params"),
            "block_2_proxy_absdelta_channelmean": _proxy_stats(delta_b2, mw_b2, mb_b2),
            "block_5_proxy_absdelta_channelmean": _proxy_stats(delta_b5, mw_b5, mb_b5),
            "block_2_lpl": _lpl_stats(phi_b2_real_C, phi_b2_pred_C, mw_b2, mb_b2),
            "block_5_lpl": _lpl_stats(phi_b5_real_C, phi_b5_pred_C, mw_b5, mb_b5),
        }
        del real_img, pred_img, feats_real, feats_pred, z_pred
        torch.cuda.empty_cache()

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print()
    print(f"Composite figures → {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
