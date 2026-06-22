#!/usr/bin/env python3
"""Compute per-region PSNR_WT for T1C-RFlow's saved NIfTI predictions.

Re-uses the saved `*_pred_t1c.nii.gz` and `*_real_t1c_normalised.nii.gz`
files. Both are already in [0, 1] (the inference module saved them under
VENA's `percentile_normalise(99.5, foreground_only)` parity contract).

For per-region metrics we need the WT mask. We pull it from the image H5
and resample to the prediction grid if needed (they should already match —
both come from the same UCSF-PDGM source).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import nibabel as nib
import numpy as np

PRED_DIR = Path("/media/hddb/mario/competitors/t1c_rflow_run/inference/epoch_best")
IMAGE_H5 = "/media/hddb/mario/data/GLIOMAS/UCSF_PDGM/h5/UCSFPDGM_image.h5"


def psnr(pred: np.ndarray, real: np.ndarray, mask: np.ndarray, data_range: float = 1.0) -> float:
    sel = mask.astype(bool)
    if not sel.any():
        return float("nan")
    mse = float(np.mean((pred[sel] - real[sel]) ** 2))
    if mse < 1e-20:
        return float("inf")
    return 10.0 * np.log10(data_range**2 / mse)


def main():
    # Build patient_id → index map for the image H5
    with h5py.File(IMAGE_H5, "r") as f:
        ids = [b.decode() if isinstance(b, (bytes, np.bytes_)) else b for b in f["ids"][:]]
        id_to_idx = {x: i for i, x in enumerate(ids)}

        rows = []
        for pred_path in sorted(PRED_DIR.glob("*_pred_t1c.nii.gz")):
            pid = pred_path.name.replace("_pred_t1c.nii.gz", "")
            if pid not in id_to_idx:
                print(f"  [skip] {pid} not in image H5")
                continue
            real_path = pred_path.parent / f"{pid}_real_t1c_normalised.nii.gz"
            pred = nib.load(str(pred_path)).get_fdata().astype(np.float32)
            real = nib.load(str(real_path)).get_fdata().astype(np.float32)

            idx = id_to_idx[pid]
            brain = f["masks/brain"][idx]
            tumor_lbl = f["masks/tumor"][idx]

            # The prediction NIfTI may be in a different crop grid than the image H5
            # (T1C-RFlow latent is 64×64×48 → decode to 256×256×192 typically).
            # The image H5 is at 240×240×155 (UCSF native, in 192×224×192 crop after re-cropping).
            print(
                f"  pid={pid} pred.shape={pred.shape} real.shape={real.shape} brain.shape={brain.shape} tumor.shape={tumor_lbl.shape}"
            )

            if pred.shape != brain.shape:
                # Resample masks to prediction grid via nearest-neighbour center-crop or zoom
                # Use scipy to zoom integer mask
                from scipy.ndimage import zoom

                zoom_factors = tuple(p / b for p, b in zip(pred.shape, brain.shape))
                brain_z = zoom(brain.astype(np.float32), zoom_factors, order=0).astype(np.int8)
                tumor_z = zoom(tumor_lbl.astype(np.float32), zoom_factors, order=0).astype(np.int8)
            else:
                brain_z = brain
                tumor_z = tumor_lbl

            wt = (tumor_z > 0).astype(np.int8)
            et = (tumor_z == 4).astype(np.int8)
            bnwt = ((brain_z > 0) & (wt == 0)).astype(np.int8)
            bg = (brain_z == 0).astype(np.int8)

            row = {
                "pid": pid,
                "psnr_whole": psnr(pred, real, np.ones_like(brain_z, dtype=bool)),
                "psnr_bg": psnr(pred, real, bg),
                "psnr_bnwt": psnr(pred, real, bnwt),
                "psnr_wt": psnr(pred, real, wt),
                "psnr_et": psnr(pred, real, et),
                "wt_vox": int(wt.sum()),
                "et_vox": int(et.sum()),
            }
            rows.append(row)
            print(
                f"  {pid}: PSNR_whole={row['psnr_whole']:.2f} PSNR_BG={row['psnr_bg']:.2f}  PSNR_WT={row['psnr_wt']:.2f}  PSNR_ET={row['psnr_et']:.2f}  wt_vox={row['wt_vox']}"
            )

    if rows:
        import statistics as s

        print()
        print(f"Aggregate over {len(rows)} patients:")
        print(
            f"  PSNR_whole = {s.mean(r['psnr_whole'] for r in rows):.2f} ± {s.stdev([r['psnr_whole'] for r in rows]):.2f}"
        )
        print(f"  PSNR_BG    = {s.mean(r['psnr_bg'] for r in rows):.2f}")
        print(f"  PSNR_BNWT  = {s.mean(r['psnr_bnwt'] for r in rows):.2f}")
        print(
            f"  PSNR_WT    = {s.mean(r['psnr_wt'] for r in rows):.2f} ± {s.stdev([r['psnr_wt'] for r in rows]):.2f}"
        )
        valid_et = [r["psnr_et"] for r in rows if not np.isnan(r["psnr_et"])]
        if valid_et:
            print(f"  PSNR_ET    = {s.mean(valid_et):.2f} (n={len(valid_et)})")


if __name__ == "__main__":
    main()
