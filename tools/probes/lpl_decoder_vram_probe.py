"""LPL decoder VRAM + timing probe.

One-off probe for the decoder-feature perceptual loss design (S3 stage).
Loads the frozen MAISI VAE-GAN decoder, runs forward + backward at each
depth K ∈ {2, 5, 8, 10}, measures peak VRAM and wall-clock, reports.

The latent shape (B=2, 4, 60, 60, 40) mirrors the brain-box used in
production training (UCSF-PDGM cohort). fp16 autocast matches the
production decode path (`norm_float16=True` in autoencoder_v2.json).

Two scenarios are tested per K:
  S1: TARGET-no-grad + PREDICTION-with-grad + backward (full LPL step)
  S2: PREDICTION-with-grad + backward only (cheaper if target features are
      pre-computed and cached)

The probe also surfaces the cross-device option (decoder on cuda:1, gradients
flowing back to cuda:0) when ≥2 GPUs are visible.

Usage on loginexa (sm_70 V100):
  $ /mnt/.../conda_envs/vena-v100/bin/python tools/probes/lpl_decoder_vram_probe.py

Output: machine-parseable JSON to stdout, plus a human-readable table.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

# Resolve repo root from this script's location so the probe is path-agnostic.
REPO_SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(REPO_SRC))

import torch  # noqa: E402

from vena.common import load_autoencoder  # noqa: E402

# Picasso checkpoint path. Same blob on server3 / local, just different roots.
CKPT_PATHS = [
    "/mnt/home/users/tic_163_uma/mpascual/fscratch/checkpoints/NV-Generate-MR/models/autoencoder_v2.pt",
    "/media/mpascual/Sandisk2TB/checkpoints/MAISI_V2_RM/NV-Generate-MR/models/autoencoder_v2.pt",
    "/media/hddb/mario/checkpoints/MAISI_V2_RM/NV-Generate-MR/models/autoencoder_v2.pt",
]

LATENT_SHAPE = (2, 4, 60, 60, 40)  # (B, C, H, W, D) — production-typical brain box.
DEPTHS_K = [2, 5, 8, 10]


def _resolve_ckpt() -> str:
    for p in CKPT_PATHS:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of the candidate MAISI VAE paths exist: {CKPT_PATHS}")


def _forward_to_block_k(decoder, z, K, autocast_dtype):
    h = z
    with torch.amp.autocast("cuda", dtype=autocast_dtype):
        for i, blk in enumerate(decoder.blocks):
            h = blk(h)
            if i == K:
                return h
    return h


def _test_depth(decoder, K, device, dtype, latent_shape, mode):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    B, C, H, W, D = latent_shape

    # Both latents on the requested device. requires_grad on the prediction.
    z_pred = torch.randn(B, C, H, W, D, device=device, dtype=torch.float32, requires_grad=True)
    z_tgt = torch.randn(B, C, H, W, D, device=device, dtype=torch.float32)

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        feat_tgt = _forward_to_block_k(decoder, z_tgt, K, dtype)
    torch.cuda.synchronize(device)
    t_tgt = time.perf_counter() - t0

    t1 = time.perf_counter()
    feat_pred = _forward_to_block_k(decoder, z_pred, K, dtype)
    loss = (feat_pred.float() - feat_tgt.float()).pow(2).mean()
    torch.cuda.synchronize(device)
    t_pred = time.perf_counter() - t1

    t2 = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize(device)
    t_back = time.perf_counter() - t2

    peak = torch.cuda.max_memory_allocated(device) / (2**30)
    return {
        "K": K,
        "mode": mode,
        "feat_shape": tuple(feat_pred.shape),
        "loss": float(loss.detach().cpu()),
        "t_target_no_grad_s": round(t_tgt, 4),
        "t_pred_fwd_s": round(t_pred, 4),
        "t_pred_back_s": round(t_back, 4),
        "t_total_step_s": round(t_tgt + t_pred + t_back, 4),
        "peak_vram_gb": round(peak, 3),
    }


def main():
    print("=== LPL decoder probe ===")
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("FATAL: no CUDA visible — probe is GPU-only.")
        sys.exit(1)
    n_gpu = torch.cuda.device_count()
    print(f"GPUs visible: {n_gpu}")
    for i in range(n_gpu):
        p = torch.cuda.get_device_properties(i)
        print(
            f"  cuda:{i} = {p.name}  total={p.total_memory / 2**30:.1f} GB  sm={p.major}.{p.minor}"
        )
    print(f"arch list: {torch.cuda.get_arch_list()}")
    print()

    ckpt = _resolve_ckpt()
    print(f"checkpoint: {ckpt}")
    device = torch.device("cuda:0")

    handle = load_autoencoder(ckpt, device=device)
    ae = handle.model
    decoder = ae.decoder
    for p in ae.parameters():
        p.requires_grad_(False)
    print(f"AE params: {sum(p.numel() for p in ae.parameters()) / 1e6:.2f} M")
    print(f"Decoder params: {sum(p.numel() for p in decoder.parameters()) / 1e6:.2f} M")
    print()

    results = []
    # V100 sm_70 has no native bf16 — autocast bf16 silently falls back to fp32.
    # fp16 is the right precision on V100 to mirror prod numerics (norm_float16=True).
    dtype = torch.float16
    print(f"latent shape: {LATENT_SHAPE}")
    print(f"autocast dtype: {dtype}")
    print()

    print(
        f"{'K':<4}{'mode':<14}{'feat_shape':<40}{'t_tgt':>10}{'t_pred':>10}{'t_back':>10}{'peak GB':>10}"
    )
    print("-" * 100)
    for K in DEPTHS_K:
        for mode in ("full_step",):  # cached-target mode is a t_pred + t_back subset → derivable
            try:
                r = _test_depth(decoder, K, device, dtype, LATENT_SHAPE, mode)
                results.append(r)
                print(
                    f"{r['K']:<4}{r['mode']:<14}{r['feat_shape']!s:<40}"
                    f"{r['t_target_no_grad_s']:>10.3f}{r['t_pred_fwd_s']:>10.3f}"
                    f"{r['t_pred_back_s']:>10.3f}{r['peak_vram_gb']:>10.3f}"
                )
            except torch.cuda.OutOfMemoryError:
                results.append({"K": K, "mode": mode, "result": "OOM", "peak_vram_gb": None})
                print(f"{K:<4}{mode:<14}{'OOM':<40}")
                torch.cuda.empty_cache()
            except Exception as e:
                tb = traceback.format_exc(limit=2)
                results.append({"K": K, "mode": mode, "result": f"ERROR: {e}", "tb": tb})
                print(f"{K:<4}{mode:<14}{'ERROR':<40}  {e}")
                torch.cuda.empty_cache()

    # --- Cross-device variant (user's proposed architecture) ---
    if n_gpu >= 2:
        print()
        print("=== cross-device probe (decoder on cuda:1, latent on cuda:0) ===")
        try:
            # Re-instantiate decoder on cuda:1 (frozen).
            handle1 = load_autoencoder(ckpt, device=torch.device("cuda:1"))
            ae1 = handle1.model
            for p in ae1.parameters():
                p.requires_grad_(False)
            decoder1 = ae1.decoder

            # Trunk-side gradient lives on cuda:0; the prediction tensor is
            # produced on cuda:0 and crossed to cuda:1 for the decode forward.
            z_pred_dev0 = torch.randn(
                *LATENT_SHAPE, device="cuda:0", dtype=torch.float32, requires_grad=True
            )
            z_tgt_dev0 = torch.randn(*LATENT_SHAPE, device="cuda:0", dtype=torch.float32)

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))
            torch.cuda.reset_peak_memory_stats(torch.device("cuda:1"))

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            # cross to cuda:1
            z_pred_dev1 = z_pred_dev0.to("cuda:1")
            z_tgt_dev1 = z_tgt_dev0.to("cuda:1")

            with torch.no_grad():
                feat_tgt = _forward_to_block_k(decoder1, z_tgt_dev1, 5, dtype)
            feat_pred = _forward_to_block_k(decoder1, z_pred_dev1, 5, dtype)
            loss_dev1 = (feat_pred.float() - feat_tgt.float()).pow(2).mean()

            # backward — grad will flow back through z_pred_dev1 → z_pred_dev0
            loss_dev0 = loss_dev1.to("cuda:0")
            torch.cuda.synchronize()
            t_fwd = time.perf_counter() - t0

            t2 = time.perf_counter()
            loss_dev0.backward()
            torch.cuda.synchronize()
            t_back = time.perf_counter() - t2

            peak0 = torch.cuda.max_memory_allocated(torch.device("cuda:0")) / (2**30)
            peak1 = torch.cuda.max_memory_allocated(torch.device("cuda:1")) / (2**30)
            print(f"K=5 cross-device  t_fwd={t_fwd:.3f}s  t_back={t_back:.3f}s")
            print(f"  peak cuda:0 = {peak0:.3f} GB   peak cuda:1 = {peak1:.3f} GB")
            print(f"  z_pred.grad on cuda:0 is set: {z_pred_dev0.grad is not None}")
            print(f"  z_pred.grad finite: {torch.isfinite(z_pred_dev0.grad).all().item()}")
            results.append(
                {
                    "K": 5,
                    "mode": "cross_device",
                    "t_fwd_s": round(t_fwd, 4),
                    "t_back_s": round(t_back, 4),
                    "peak_vram_gb_cuda0": round(peak0, 3),
                    "peak_vram_gb_cuda1": round(peak1, 3),
                    "grad_finite": bool(torch.isfinite(z_pred_dev0.grad).all().item()),
                }
            )
        except torch.cuda.OutOfMemoryError:
            print("cross-device K=5: OOM")
            results.append({"K": 5, "mode": "cross_device", "result": "OOM"})
        except Exception as e:
            print(f"cross-device K=5: ERROR {e}")
            results.append({"K": 5, "mode": "cross_device", "result": f"ERROR: {e}"})

    print()
    print("=== JSON ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
