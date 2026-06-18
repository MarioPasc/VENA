"""LPL feature-map visualization + per-region statistics probe.

For one real (z_T1pre, z_T1c) pair from the UCSF-PDGM corpus, decode both
through hooks on decoder blocks {0, 2, 5}, compute the enhancement-signal
proxy |phi_l(z_T1c) - phi_l(z_T1pre)| per block, then:

  1. Save axial-slice PNGs of |Delta phi| at each block, with WT mask overlaid.
  2. Compute per-region statistics (mean / p50 / p99) of |Delta phi|
     inside WT vs brain∖WT, at each block.
  3. Re-measure peak VRAM at K=2 and K=5 with the PRODUCTION latent shape
     (4, 48, 56, 48) for an updated §3.5 table.

Usage on loginexa:
  $ /mnt/.../conda_envs/vena-v100/bin/python tools/probes/lpl_feature_visualization.py

Output:
  tools/probes/out/<patient_id>/block_<k>_axial_<z>.png
  tools/probes/out/<patient_id>/summary.json
  Stdout JSON for direct parsing.

PRE_REQ: scipy installed in env (for ndimage.zoom upsampling); falls back
to F.interpolate nearest if scipy missing.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(REPO_SRC))

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from vena.common import load_autoencoder

CKPT = "/mnt/home/users/tic_163_uma/mpascual/fscratch/checkpoints/NV-Generate-MR/models/autoencoder_v2.pt"
LAT_H5 = (
    "/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/vena/UCSF_PDGM/h5/UCSFPDGM_latents.h5"
)
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

LAYERS_TO_HOOK = [0, 2, 5]  # entry, level-0 last, level-1 last
WT_THRESHOLD = 0.5
DEVICE = "cuda:0"


def _hook_blocks(decoder, layers):
    caps = {}

    def make(idx):
        def h(_m, _i, o):
            caps[idx] = o.detach()

        return h

    handles = [decoder.blocks[i].register_forward_hook(make(i)) for i in layers]
    return caps, handles


def _partial_forward(decoder, z, max_block):
    h = z
    for i, blk in enumerate(decoder.blocks):
        h = blk(h)
        if i == max_block:
            return h
    return h


def _per_region_stats(delta_abs, m_wt_at_scale, m_brain_at_scale):
    """delta_abs: (C, H, W, D) — abs(phi(z_T1c) - phi(z_T1pre)), channel-wise.
    Returns dict of stats inside WT, inside (brain∖WT), and globally."""
    d = delta_abs.float().mean(0)  # mean across channels → (H, W, D)
    flat = d.flatten()
    wt = m_wt_at_scale.bool().flatten()
    bg_brain = (m_brain_at_scale.bool() & ~m_wt_at_scale.bool()).flatten()

    def _stats(x):
        if x.numel() == 0:
            return {"n": 0}
        x = x.float().cpu().numpy()
        return {
            "n": int(x.size),
            "mean": float(np.mean(x)),
            "p50": float(np.percentile(x, 50)),
            "p95": float(np.percentile(x, 95)),
            "p99": float(np.percentile(x, 99)),
            "max": float(np.max(x)),
        }

    return {
        "WT": _stats(flat[wt]),
        "brain_minus_WT": _stats(flat[bg_brain]),
        "global": _stats(flat),
    }


def _resample_mask(mask_lat, target_hwd):
    """NN upsample (C,H,W,D) latent mask to target spatial shape."""
    x = mask_lat.unsqueeze(0).float()  # (1, C, H, W, D)
    y = F.interpolate(x, size=target_hwd, mode="nearest")
    return y[0]


def _save_axial_panel(delta_abs, m_wt_at_scale, out_path):
    """Save 5 evenly-spaced axial slices showing |Delta phi| (mean-over-C) with WT overlay."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"matplotlib missing — skipping figure {out_path}")
        return False

    d = delta_abs.float().mean(0).cpu().numpy()  # (H, W, D)
    wt = m_wt_at_scale[0].cpu().numpy() if m_wt_at_scale.dim() == 4 else m_wt_at_scale.cpu().numpy()
    D = d.shape[-1]
    n_slices = 5
    ix = np.linspace(D * 0.2, D * 0.8, n_slices).astype(int)

    fig, axes = plt.subplots(2, n_slices, figsize=(3 * n_slices, 6))
    vmax = np.percentile(d, 99.5)
    for j, k in enumerate(ix):
        axes[0, j].imshow(d[:, :, k], cmap="hot", vmin=0, vmax=vmax)
        axes[0, j].set_title(f"|Δφ| z={k}")
        axes[0, j].axis("off")
        axes[1, j].imshow(d[:, :, k], cmap="hot", vmin=0, vmax=vmax)
        axes[1, j].contour(wt[:, :, k], levels=[0.5], colors="lime", linewidths=0.6)
        axes[1, j].set_title(f"+WT overlay z={k}")
        axes[1, j].axis("off")
    plt.suptitle(out_path.stem)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close()
    return True


def _vram_probe_at_production_shape(decoder, K, dtype=torch.float16):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    B, C, H, W, D = 2, 4, 48, 56, 48
    z_pred = torch.randn(B, C, H, W, D, device=DEVICE, requires_grad=True)
    z_tgt = torch.randn(B, C, H, W, D, device=DEVICE)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.amp.autocast("cuda", dtype=dtype):
        with torch.no_grad():
            feat_tgt = _partial_forward(decoder, z_tgt, K)
    torch.cuda.synchronize()
    t_tgt = time.perf_counter() - t0
    t1 = time.perf_counter()
    with torch.amp.autocast("cuda", dtype=dtype):
        feat_pred = _partial_forward(decoder, z_pred, K)
    loss = (feat_pred.float() - feat_tgt.float()).pow(2).mean()
    torch.cuda.synchronize()
    t_fwd = time.perf_counter() - t1
    t2 = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    t_back = time.perf_counter() - t2
    peak = torch.cuda.max_memory_allocated(DEVICE) / (2**30)
    return {
        "K": K,
        "shape_out": tuple(feat_pred.shape),
        "t_target_s": round(t_tgt, 3),
        "t_fwd_s": round(t_fwd, 3),
        "t_back_s": round(t_back, 3),
        "peak_vram_gb": round(peak, 3),
    }


def main():
    print("=== LPL feature visualization probe ===")
    print(f"torch {torch.__version__}  cuda {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        sys.exit(1)
    print(f"cuda:0 = {torch.cuda.get_device_properties(0).name}")
    print()

    # ----- 1. Production-shape VRAM probe -----
    print("=== Updated VRAM at PRODUCTION latent (4, 48, 56, 48), B=2, fp16 autocast ===")
    handle = load_autoencoder(CKPT, device=DEVICE)
    ae = handle.model
    decoder = ae.decoder
    for p in ae.parameters():
        p.requires_grad_(False)
    print(
        f"AE params {sum(p.numel() for p in ae.parameters()) / 1e6:.2f}M, decoder {sum(p.numel() for p in decoder.parameters()) / 1e6:.2f}M"
    )
    vram_results = []
    for K in [2, 5]:
        try:
            r = _vram_probe_at_production_shape(decoder, K)
            vram_results.append(r)
            print(
                f"  K={K}  feat={r['shape_out']}  t_step={r['t_target_s'] + r['t_fwd_s'] + r['t_back_s']:.2f}s  peak={r['peak_vram_gb']} GB"
            )
        except torch.cuda.OutOfMemoryError:
            vram_results.append({"K": K, "result": "OOM"})
            print(f"  K={K}  OOM")
            torch.cuda.empty_cache()
    print()

    # ----- 2. Feature visualization on one real (T1pre, T1c) pair -----
    print("=== Feature visualization on UCSF-PDGM patient 0 ===")
    torch.cuda.empty_cache()

    with h5py.File(LAT_H5, "r") as f:
        pid = f["ids"][0]
        if isinstance(pid, bytes):
            pid = pid.decode("utf-8")
        z_pre_np = f["latents/t1pre"][0]  # (4, 48, 56, 48)
        z_t1c_np = f["latents/t1c"][0]
        tumor_lat = f["masks/tumor_latent"][0]  # (3, 48, 56, 48) soft
        brain_lat = f["masks/brain_latent"][0]  # (1, 48, 56, 48) int8
    print(f"patient_id = {pid}")

    out_dir = OUT / pid
    out_dir.mkdir(parents=True, exist_ok=True)

    z_pre = torch.from_numpy(z_pre_np).to(DEVICE).unsqueeze(0).float()
    z_t1c = torch.from_numpy(z_t1c_np).to(DEVICE).unsqueeze(0).float()
    soft_wt = np.clip(tumor_lat.sum(0, keepdims=True), 0.0, 1.0)
    m_wt_lat = torch.from_numpy((soft_wt >= WT_THRESHOLD).astype(np.float32)).to(DEVICE)
    m_brain_lat = torch.from_numpy(brain_lat.astype(np.float32)).to(DEVICE)
    print(
        f"  z_pre={tuple(z_pre.shape)}  z_t1c={tuple(z_t1c.shape)}  m_wt_lat={tuple(m_wt_lat.shape)}  m_brain_lat={tuple(m_brain_lat.shape)}"
    )
    print(
        f"  WT voxels at latent res: {int(m_wt_lat.sum().item())} ({100 * m_wt_lat.mean().item():.1f}% of latent grid)"
    )
    print(
        f"  brain voxels at latent res: {int(m_brain_lat.sum().item())} ({100 * m_brain_lat.mean().item():.1f}% of latent grid)"
    )
    print()

    # Hook decode blocks for both z_pre and z_t1c
    feats = {}
    for label, z in [("t1pre", z_pre), ("t1c", z_t1c)]:
        caps, handles = _hook_blocks(decoder, LAYERS_TO_HOOK)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.float16):
                _partial_forward(decoder, z, max(LAYERS_TO_HOOK))
        for h in handles:
            h.remove()
        feats[label] = caps

    summary = {
        "patient_id": pid,
        "latent_shape": tuple(z_pre.shape[1:]),
        "per_block": {},
        "vram_at_production_shape": vram_results,
    }

    for K in LAYERS_TO_HOOK:
        phi_pre = feats["t1pre"][K][0]  # (C, H, W, D)
        phi_t1c = feats["t1c"][K][0]
        delta = (phi_t1c.float() - phi_pre.float()).abs()
        target_hwd = delta.shape[-3:]
        m_wt_k = _resample_mask(m_wt_lat, target_hwd)
        m_brain_k = _resample_mask(m_brain_lat, target_hwd)
        stats = _per_region_stats(delta, m_wt_k, m_brain_k)
        summary["per_block"][f"block_{K}"] = {
            "shape": tuple(delta.shape),
            "stats": stats,
            "ratio_WT_vs_notWT_mean": stats["WT"]["mean"]
            / max(stats["brain_minus_WT"]["mean"], 1e-9),
            "ratio_WT_vs_notWT_p99": stats["WT"]["p99"] / max(stats["brain_minus_WT"]["p99"], 1e-9),
        }
        print(f"--- block {K}  shape={tuple(delta.shape)} ---")
        for r in ("WT", "brain_minus_WT", "global"):
            s = stats[r]
            print(
                f"  {r:18s}  n={s.get('n', 0):8d}  mean={s.get('mean', 0):.4f}  p50={s.get('p50', 0):.4f}  p95={s.get('p95', 0):.4f}  p99={s.get('p99', 0):.4f}  max={s.get('max', 0):.4f}"
            )
        print(
            f"  ratio WT/notWT  mean={summary['per_block'][f'block_{K}']['ratio_WT_vs_notWT_mean']:.2f}  p99={summary['per_block'][f'block_{K}']['ratio_WT_vs_notWT_p99']:.2f}"
        )
        fig_path = out_dir / f"block_{K}_axial.png"
        _save_axial_panel(delta, m_wt_k, fig_path)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print()
    print(f"figures + summary → {out_dir}")
    print()
    print("=== JSON ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
