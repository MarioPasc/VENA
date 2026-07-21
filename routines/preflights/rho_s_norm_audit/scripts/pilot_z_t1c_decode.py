"""Picasso GPU pilot — reproduce §14.3 ρ_S swing via z_t1c decode.

Decodes cached z_t1c from UCSFPDGM_latents.h5 using the frozen MAISI VAE
and measures ρ_S at two normalisation levels:

  P=99.5  → ρ_S expected ≈ 0.66  (scale mismatch: decoder ≈ 99.95, real at 99.5)
  P=99.95 → ρ_S expected ≈ 0.00  (both at 99.95 → near-zero residual correlation)

If you cannot reproduce this swing, the premise in §14.3 is false and the
full ρ_S audit should be aborted (report STATUS: PREMISE-FALSE).

Usage (Picasso, inside Singularity):
    python routines/preflights/rho_s_norm_audit/scripts/pilot_z_t1c_decode.py \
        --latent-h5  /mnt/.../UCSFPDGM_latents.h5 \
        --image-h5   /mnt/.../UCSFPDGM_image.h5 \
        --vae-ckpt   /mnt/.../autoencoder_v2.pt \
        --n-patients 10 \
        --device     cuda:0 \
        --out        /mnt/.../pilot_z_t1c_results.csv

Runtime: ~3 min for 10 patients on A100 (VAE decode is the bottleneck).
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy import stats

logger = logging.getLogger(__name__)


def _decode_ids(arr: h5py.Dataset) -> list[str]:
    return [s.decode() if isinstance(s, bytes) else str(s) for s in arr[:]]


def _percentile_normalise_np(
    vol: np.ndarray,
    brain: np.ndarray,
    upper: float,
) -> np.ndarray:
    """Apply per-volume percentile normalisation using foreground voxels.

    Mirrors ``vena.common.percentile_normalise`` without requiring torch.

    Parameters
    ----------
    vol :
        Raw volume ``(H, W, D)`` float32.
    brain :
        Binary brain mask ``(H, W, D)`` bool.
    upper :
        Upper percentile (99.5 or 99.95).

    Returns
    -------
    np.ndarray
        Volume in ``[0, 1]`` float32.
    """
    fg = vol[brain]
    if fg.size == 0:
        return np.zeros_like(vol)
    lo = float(np.percentile(fg, 0.0))
    hi = float(np.percentile(fg, upper))
    denom = hi - lo + 1e-8
    out = np.clip((vol - lo) / denom, 0.0, 1.0).astype(np.float32)
    return out


def compute_rho_s(
    pred: np.ndarray,
    real: np.ndarray,
    brain: np.ndarray,
    wt: np.ndarray,
    dilate_k: int = 5,
) -> float:
    """Compute ρ_S = Spearman(|pred−real|, real) over brain∖dilate(WT).

    Parameters
    ----------
    pred, real :
        Predicted and real volumes ``(H, W, D)`` float32, same normalisation.
    brain, wt :
        Boolean masks ``(H, W, D)``.
    dilate_k :
        Dilation kernel size for WT exclusion.

    Returns
    -------
    float
        Spearman ρ.
    """
    # Dilate WT with a max-pool approximation (CPU NumPy).
    from scipy.ndimage import binary_dilation

    wt_dilated = binary_dilation(wt, iterations=dilate_k // 2)
    mask = brain & ~wt_dilated

    if mask.sum() < 10:
        return float("nan")

    abs_resid = np.abs(real[mask] - pred[mask]).astype(np.float64)
    real_vals = real[mask].astype(np.float64)
    rho, _ = stats.spearmanr(abs_resid, real_vals)
    return float(rho)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s — %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-h5", required=True, type=Path)
    parser.add_argument("--image-h5", required=True, type=Path)
    parser.add_argument("--vae-ckpt", required=True, type=Path)
    parser.add_argument("--n-patients", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")

    # Load VAE decoder.
    logger.info("Loading VAE decoder from %s", args.vae_ckpt)
    from vena.common import load_autoencoder

    _encoder, decoder = load_autoencoder(args.vae_ckpt, device=device)
    decoder.eval()

    rows: list[dict] = []

    with (
        h5py.File(args.latent_h5, "r") as f_lat,
        h5py.File(args.image_h5, "r") as f_img,
    ):
        lat_ids = _decode_ids(f_lat["ids"])
        img_ids = _decode_ids(f_img["ids"])
        img_idx_map: dict[str, int] = {pid: i for i, pid in enumerate(img_ids)}

        # Pick test-split patients that are in both H5s.
        if "splits/test" in f_lat:
            test_indices = f_lat["splits/test"][:]
            candidate_ids = [lat_ids[i] for i in test_indices]
        else:
            candidate_ids = lat_ids

        selected = [pid for pid in candidate_ids if pid in img_idx_map][: args.n_patients]
        logger.info("Selected %d patients for pilot.", len(selected))

        lat_id_map: dict[str, int] = {pid: i for i, pid in enumerate(lat_ids)}

        for pid in selected:
            li = lat_id_map[pid]
            ii = img_idx_map[pid]

            # Load z_t1c from latent H5.
            z_t1c = np.asarray(f_lat["latents/t1c"][li], dtype=np.float32)

            # Load real T1c raw from image H5.
            real_raw = np.asarray(f_img["images/t1c"][ii], dtype=np.float32)

            # Load brain + WT masks from image H5.
            brain = np.asarray(f_img["masks/brain"][ii], dtype=bool)
            wt_key = "masks/tumor" if "masks/tumor" in f_img else "masks/wt"
            wt = (
                np.asarray(f_img[wt_key][ii], dtype=bool)
                if wt_key in f_img
                else np.zeros_like(brain)
            )

            # Decode z_t1c → synthetic volume (in 99.95 scale).
            z_t = torch.from_numpy(z_t1c).to(device).unsqueeze(0)  # (1, C, h, w, d)
            with torch.no_grad():
                pred_raw_t = decoder.decode(z_t)  # (1, 1, H, W, D)
            pred_raw = pred_raw_t[0, 0].cpu().float().numpy()

            # Compute ρ_S under P=99.5 (mismatch: pred near 99.95, real at 99.5).
            pred_p995 = _percentile_normalise_np(pred_raw, brain, upper=99.5)
            real_p995 = _percentile_normalise_np(real_raw, brain, upper=99.5)
            rho_995 = compute_rho_s(pred_p995, real_p995, brain, wt)

            # Compute ρ_S under P=99.95 (matched: both at 99.95).
            pred_p9995 = _percentile_normalise_np(pred_raw, brain, upper=99.95)
            real_p9995 = _percentile_normalise_np(real_raw, brain, upper=99.95)
            rho_9995 = compute_rho_s(pred_p9995, real_p9995, brain, wt)

            rows.append(
                {
                    "patient_id": pid,
                    "rho_s_P99.5": rho_995,
                    "rho_s_P99.95": rho_9995,
                    "delta": rho_9995 - rho_995,
                }
            )
            logger.info(
                "  %s: ρ_S@99.5=%.3f  ρ_S@99.95=%.3f  Δ=%.3f",
                pid,
                rho_995,
                rho_9995,
                rho_9995 - rho_995,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["patient_id", "rho_s_P99.5", "rho_s_P99.95", "delta"]
        )
        writer.writeheader()
        writer.writerows(rows)

    rho_995_vals = [r["rho_s_P99.5"] for r in rows if not np.isnan(r["rho_s_P99.5"])]
    rho_9995_vals = [r["rho_s_P99.95"] for r in rows if not np.isnan(r["rho_s_P99.95"])]
    med_995 = float(np.median(rho_995_vals)) if rho_995_vals else float("nan")
    med_9995 = float(np.median(rho_9995_vals)) if rho_9995_vals else float("nan")

    logger.info("=== PILOT SUMMARY ===")
    logger.info("Median ρ_S @ P=99.5 : %.3f  (expected ≈ 0.66)", med_995)
    logger.info("Median ρ_S @ P=99.95: %.3f  (expected ≈ 0.00)", med_9995)
    logger.info("Swing               : %.3f  (expected ≈ 0.66)", med_9995 - med_995)

    swing = med_9995 - med_995
    if abs(swing) < 0.10:
        logger.warning(
            "PREMISE-FALSE: swing=%.3f < 0.10 — normalisation confound not confirmed. "
            "Abort full audit and report PREMISE-FALSE.",
            swing,
        )
    else:
        logger.info(
            "PREMISE CONFIRMED: swing=%.3f ≥ 0.10 — proceed to full ρ_S audit.",
            swing,
        )

    logger.info("Results written to %s", args.out)


if __name__ == "__main__":
    main()
