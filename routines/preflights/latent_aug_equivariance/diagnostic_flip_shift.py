"""Diagnose the flip equivariance gap by sweeping integer-voxel shifts.

Hypothesis: MAISI-V2's transposed-convolution decoder is shift-equivariant
only modulo the upsampling phase, so ``D(flip(z))`` and ``flip(D(z))`` are
offset by ~1 latent voxel (=4 image voxels) along the flip axis. If true,
shifting one of the two paths by k ∈ {-2,...,+2} image voxels should
restore alignment and push PSNR above the 35 dB threshold.

Procedure (per patient × axis):
  1. Decode the real latent z to ``recon_image = D(z)``.
  2. Compute ``gold = torch.flip(recon_image, dims=[axis])``.
  3. Compute ``proposed = D(torch.flip(z, dims=[axis]))``.
  4. For each shift k ∈ {-2,...,+2}, compute PSNR(gold, roll(proposed, k)).
  5. Report the per-shift PSNR for each patient + axis.

If the best shift is consistently non-zero, fix the FlipLR operator with a
post-flip roll of that magnitude (in latent voxels = image voxels / 4).
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import h5py
import numpy as np
import torch

from vena.model.autoencoder.maisi.decode.engine import MaisiDecoder
from vena.model.autoencoder.maisi.loader import load_autoencoder
from vena.model.fm.eval.exhaustive import (
    build_crop_spec_from_h5,
)
from vena.model.fm.metrics import ImageMetrics

logger = logging.getLogger(__name__)


def _roll_3d(x: torch.Tensor, axis: int, shift: int) -> torch.Tensor:
    """Zero-padded integer shift on one spatial axis of a ``(H, W, D)`` volume."""
    if shift == 0:
        return x
    out = torch.zeros_like(x)
    if shift > 0:
        # Move content towards higher indices; drop the upper part.
        if axis == 0:
            out[shift:, :, :] = x[:-shift, :, :]
        elif axis == 1:
            out[:, shift:, :] = x[:, :-shift, :]
        else:
            out[:, :, shift:] = x[:, :, :-shift]
    else:
        k = -shift
        if axis == 0:
            out[:-k, :, :] = x[k:, :, :]
        elif axis == 1:
            out[:, :-k, :] = x[:, k:, :]
        else:
            out[:, :, :-k] = x[:, :, k:]
    return out


def _decode(vae: MaisiDecoder, z: torch.Tensor, crop_spec, device) -> torch.Tensor:
    with torch.inference_mode():
        out = vae.decode(z.unsqueeze(0).to(device), crop_spec=crop_spec)
    return out.image[0, 0].float().clamp(0.0, 1.0)


def _psnr(p: torch.Tensor, r: torch.Tensor, image_metrics: ImageMetrics) -> float:
    p5 = p[None, None]
    r5 = r[None, None]
    mask = torch.ones_like(p5, dtype=torch.bool)
    return float(image_metrics.psnr(p5, r5, mask).reshape(-1)[0].item())


def _ssim(p: torch.Tensor, r: torch.Tensor, image_metrics: ImageMetrics) -> float:
    p5 = p[None, None]
    r5 = r[None, None]
    mask = torch.ones_like(p5, dtype=torch.bool)
    return float(image_metrics.ssim(p5, r5, mask).reshape(-1)[0].item())


def _pick_pids(latent_h5: Path, n: int, seed: int) -> list[str]:
    with h5py.File(latent_h5, "r") as f:

        def _decode_ids(ds):
            return [b.decode() if isinstance(b, bytes) else str(b) for b in ds[:]]

        all_ids = _decode_ids(f["ids"])
        val_path = "splits/cv/fold_0/val"
        pool_keys = _decode_ids(f[val_path]) if val_path in f else list(all_ids)
        has_csr = "patients/offsets" in f and "patients/keys" in f
        offsets = f["patients/offsets"][:] if has_csr else None
        csr_keys = _decode_ids(f["patients/keys"]) if has_csr else None
    rng = np.random.default_rng(int(seed))
    n_pick = min(int(n), len(pool_keys))
    chosen = sorted(rng.choice(len(pool_keys), size=n_pick, replace=False))
    chosen_patients = [pool_keys[int(i)] for i in chosen]
    if not has_csr or set(chosen_patients).issubset(set(all_ids)):
        return [p for p in chosen_patients if p in set(all_ids)][:n_pick]
    key_to_pos = {k: i for i, k in enumerate(csr_keys)}
    scan_ids: list[str] = []
    for pk in chosen_patients:
        if pk not in key_to_pos:
            continue
        start = int(offsets[key_to_pos[pk]])
        scan_ids.append(all_ids[start])
    return scan_ids


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--vae", type=Path, required=True)
    p.add_argument("--cohort-name", type=str, required=True)
    p.add_argument("--latent-h5", type=Path, required=True)
    p.add_argument("--image-h5", type=Path, required=True)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--axes",
        type=str,
        default="0",
        help="Comma-separated axis indices in {0,1,2} to test (default 0 = L-axis)",
    )
    p.add_argument(
        "--shifts",
        type=str,
        default="-2,-1,0,1,2",
        help="Comma-separated image-voxel shifts to sweep along each axis",
    )
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if torch.cuda.is_available() and args.device.startswith("cuda"):
        device = torch.device(args.device)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    handle = load_autoencoder(args.vae, device=str(device))
    vae = MaisiDecoder(handle=handle)
    metrics = ImageMetrics(data_range=1.0)

    axes = [int(a) for a in args.axes.split(",")]
    shifts = [int(s) for s in args.shifts.split(",")]
    pids = _pick_pids(args.latent_h5, args.n, args.seed)
    logger.info(
        "diagnosing %s, n=%d, axes=%s, shifts=%s", args.cohort_name, len(pids), axes, shifts
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cohort", "patient_id", "axis", "shift_img_vox", "psnr_db", "ssim"])
        for pid in pids:
            try:
                crop_spec = build_crop_spec_from_h5(args.image_h5, pid)
                # We don't need the real T1c — the test is intrinsic: gold path
                # is flip(D(z)); proposed is D(flip(z)). VAE recon noise cancels.
                with h5py.File(args.latent_h5, "r") as h:
                    ids = [b.decode() if isinstance(b, bytes) else str(b) for b in h["ids"][:]]
                    idx = {pid_: i for i, pid_ in enumerate(ids)}
                    arr = h["latents/t1c"][idx[pid]]
                z = torch.from_numpy(np.ascontiguousarray(arr)).float().to(device)
                recon = _decode(vae, z, crop_spec, device)
            except Exception as exc:
                logger.warning("pid=%s: load/decode failed (%s); skipping.", pid, exc)
                continue
            for axis in axes:
                # axis in 0/1/2 of (H,W,D) volume; latent axis is the SAME index
                # because both grids share the same spatial-axis ordering.
                gold = torch.flip(recon, dims=[axis])
                z_flipped = torch.flip(z, dims=[axis + 1])  # +1 because latent has channel dim
                proposed = _decode(vae, z_flipped, crop_spec, device)
                for shift in shifts:
                    proposed_shift = _roll_3d(proposed, axis, shift)
                    psnr = _psnr(proposed_shift, gold, metrics)
                    ssim = _ssim(proposed_shift, gold, metrics)
                    w.writerow([args.cohort_name, pid, axis, shift, f"{psnr:.4f}", f"{ssim:.6f}"])
                    logger.info(
                        "  pid=%s axis=%d shift=%+d → PSNR=%.2f dB SSIM=%.4f",
                        pid,
                        axis,
                        shift,
                        psnr,
                        ssim,
                    )

    logger.info("wrote %s", args.out_csv)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
