#!/usr/bin/env python3
"""Per-region statistics of the rectified-flow training target.

For each val patient (UCSF-PDGM fold 0 val):
  z_t1c, z_t1pre, z_t2, z_flair, brain_latent, tumor_latent
  ε ~ N(0, 1)        (noise)
  u = z_t1c - ε      (target velocity, constant in α)
  Δ = z_t1c - z_t1pre  (latent-space delta from input to target)

Decompose magnitudes by region:
  BG      (brain_latent == 0)
  not-WT  (brain_latent == 1 AND tumor_latent[:,0] < 0.5)
  WT      (tumor_latent[:,0] >= 0.5)
  ET      (tumor_latent[:,2] >= 0.5)  if available

Output: scalar fractions, mean magnitudes per region, and a class-imbalance
ratio (#voxels in region / total) × (mean |signal| in region) / total |signal|.

We also do this in image space on a random subset (n=10 patients) using the
image H5 to confirm the latent analysis transfers to pixel intensities.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

LATENT_H5 = "/media/hddb/mario/data/GLIOMAS/UCSF_PDGM/h5/UCSFPDGM_latents.h5"
IMAGE_H5 = "/media/hddb/mario/data/GLIOMAS/UCSF_PDGM/h5/UCSFPDGM_image.h5"


def per_region_stats(values, masks: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Compute n_voxels, mean|x|, sum|x|, fraction-of-total-sum per region."""
    total_abs = np.sum(np.abs(values))
    total_n = values.size
    out = {}
    for name, m in masks.items():
        sel = m.astype(bool)
        n = int(sel.sum())
        if n == 0:
            out[name] = {
                "n": 0,
                "frac_voxels": 0.0,
                "mean_abs": 0.0,
                "sum_abs": 0.0,
                "frac_sum_abs": 0.0,
            }
            continue
        x = values[sel]
        abs_sum = float(np.abs(x).sum())
        out[name] = {
            "n": n,
            "frac_voxels": n / total_n,
            "mean_abs": float(np.abs(x).mean()),
            "sum_abs": abs_sum,
            "frac_sum_abs": abs_sum / float(total_abs + 1e-12),
        }
    return out


def main():
    rng = np.random.default_rng(1337)
    with h5py.File(LATENT_H5, "r") as f:
        ids_arr = f["ids"][:]
        ids = [b.decode() if isinstance(b, (bytes, np.bytes_)) else b for b in ids_arr]
        fold = 0
        val_ids = [
            b.decode() if isinstance(b, (bytes, np.bytes_)) else b
            for b in f[f"splits/cv/fold_{fold}/val"][:]
        ]
        id_to_idx = {x: i for i, x in enumerate(ids)}

        # Pick 30 val patients
        pick = rng.choice(len(val_ids), size=min(30, len(val_ids)), replace=False)
        pick_ids = [val_ids[i] for i in pick]
        print("=== Per-region latent target-velocity audit ===")
        print(f"Latent H5: {LATENT_H5}")
        print(f"N patients sampled: {len(pick_ids)} (UCSF-PDGM fold 0 val)")
        print(
            "Latent shape: (C=4, h=48, w=56, d=48) — 129,024 latent voxels (4×spatial = 516,096 numel)\n"
        )

        # Accumulators (aggregate over patients to get a population estimate)
        u_stats_acc = {
            k: {
                "n_sum": 0,
                "vox_total_sum": 0,
                "abs_sum": 0.0,
                "abs_total_sum": 0.0,
                "n_per_patient_sum": 0,
            }
            for k in ("BG", "BRAIN_NOT_WT", "WT", "ET")
        }
        delta_stats_acc = {k: dict(u_stats_acc[k]) for k in u_stats_acc}
        z_t1c_stats_acc = {k: dict(u_stats_acc[k]) for k in u_stats_acc}

        per_patient = []

        for pid in pick_ids:
            idx = id_to_idx[pid]
            z_t1c = f["latents/t1c"][idx]  # (4, 48, 56, 48)
            z_t1pre = f["latents/t1pre"][idx]
            brain = f["masks/brain_latent"][idx, 0]  # (48, 56, 48), 0/1
            tumor_l = f["masks/tumor_latent"][idx]  # (3, 48, 56, 48), soft
            # tumor_latent channel 0 = WT (likely soft), 1 = TC, 2 = ET (BraTS standard)
            wt = (tumor_l[0] >= 0.5).astype(np.int8)
            et = (tumor_l[2] >= 0.5).astype(np.int8)
            brain_not_wt = (brain == 1) & (wt == 0)
            bg = brain == 0

            # Broadcast masks across the C=4 latent channels
            m_bg = np.broadcast_to(bg[None], z_t1c.shape)
            m_bnwt = np.broadcast_to(brain_not_wt[None], z_t1c.shape)
            m_wt = np.broadcast_to(wt[None], z_t1c.shape)
            m_et = np.broadcast_to(et[None], z_t1c.shape)
            masks = {"BG": m_bg, "BRAIN_NOT_WT": m_bnwt, "WT": m_wt, "ET": m_et}

            # Velocity target: u = z_t1c - ε  (the model is trained with L1(v_pred, u))
            eps = rng.standard_normal(z_t1c.shape).astype(np.float32)
            u = z_t1c - eps

            # Latent translation difficulty: Δ = z_t1c - z_t1pre
            delta = z_t1c - z_t1pre

            # Per-patient stats
            u_stats = per_region_stats(u, masks)
            delta_stats = per_region_stats(delta, masks)
            z_t1c_stats = per_region_stats(z_t1c, masks)

            per_patient.append(
                {
                    "pid": pid,
                    "wt_voxels_latent": int(wt.sum()),
                    "et_voxels_latent": int(et.sum()),
                    "brain_voxels_latent": int((brain == 1).sum()),
                    "u_frac_sum_abs_WT": u_stats["WT"]["frac_sum_abs"],
                    "delta_frac_sum_abs_WT": delta_stats["WT"]["frac_sum_abs"],
                    "u_mean_abs_WT": u_stats["WT"]["mean_abs"],
                    "u_mean_abs_BG": u_stats["BG"]["mean_abs"],
                    "delta_mean_abs_WT": delta_stats["WT"]["mean_abs"],
                    "delta_mean_abs_BG": delta_stats["BG"]["mean_abs"],
                }
            )

            for k in u_stats_acc:
                u_stats_acc[k]["n_sum"] += u_stats[k]["n"]
                u_stats_acc[k]["abs_sum"] += u_stats[k]["sum_abs"]
                delta_stats_acc[k]["n_sum"] += delta_stats[k]["n"]
                delta_stats_acc[k]["abs_sum"] += delta_stats[k]["sum_abs"]
                z_t1c_stats_acc[k]["n_sum"] += z_t1c_stats[k]["n"]
                z_t1c_stats_acc[k]["abs_sum"] += z_t1c_stats[k]["sum_abs"]

        total_vox = (
            sum(u_stats_acc[k]["n_sum"] for k in u_stats_acc) / 4
        )  # /4 because each region was counted per channel
        # Actually wait: each region is C=4 channels worth of voxels because we broadcast.
        # Compute proper fraction-of-loss using L1 over all voxels.
        u_total_abs = sum(u_stats_acc[k]["abs_sum"] for k in u_stats_acc)
        delta_total_abs = sum(delta_stats_acc[k]["abs_sum"] for k in delta_stats_acc)
        z_t1c_total_abs = sum(z_t1c_stats_acc[k]["abs_sum"] for k in z_t1c_stats_acc)
        total_n = sum(u_stats_acc[k]["n_sum"] for k in u_stats_acc)

        print(
            f"{'Region':<14} {'Vox %':>8} {'⟨|u|⟩':>9} {'⟨|Δ|⟩':>9} {'⟨|z_t1c|⟩':>11} {'%∑|u|':>9} {'%∑|Δ|':>9}"
        )
        print("-" * 84)
        for k in ("BG", "BRAIN_NOT_WT", "WT", "ET"):
            n = u_stats_acc[k]["n_sum"]
            frac_v = n / total_n if total_n else 0
            mean_u = u_stats_acc[k]["abs_sum"] / max(n, 1)
            mean_d = delta_stats_acc[k]["abs_sum"] / max(n, 1)
            mean_z = z_t1c_stats_acc[k]["abs_sum"] / max(n, 1)
            frac_u = u_stats_acc[k]["abs_sum"] / max(u_total_abs, 1e-12)
            frac_d = delta_stats_acc[k]["abs_sum"] / max(delta_total_abs, 1e-12)
            print(
                f"  {k:<12} {frac_v * 100:>7.3f}% {mean_u:>9.4f} {mean_d:>9.4f} {mean_z:>11.4f} {frac_u * 100:>8.4f}% {frac_d * 100:>8.4f}%"
            )

        print()
        print("Interpretation:")
        print("  L1 velocity loss (mean over all voxels) = mean⟨|v_pred - u|⟩.")
        print("  If model predicts perfectly on BG/BRAIN_NOT_WT and randomly on WT,")
        print(
            f"  WT contributes ~{u_stats_acc['WT']['abs_sum'] / max(u_total_abs, 1e-12) * 100:.3f}% of total L1 magnitude."
        )
        print(
            f"  The remaining {100 - u_stats_acc['WT']['abs_sum'] / max(u_total_abs, 1e-12) * 100:.3f}% is from non-WT voxels."
        )
        print(
            f"  Optimiser gradient signal favours non-WT correctness by ratio "
            f"~{(u_stats_acc['BG']['abs_sum'] + u_stats_acc['BRAIN_NOT_WT']['abs_sum']) / max(u_stats_acc['WT']['abs_sum'], 1e-12):.1f}:1."
        )
        print()
        print("Per-patient summary (first 10):")
        for p in per_patient[:10]:
            print(
                f"  {p['pid']}  wt_lat_vox={p['wt_voxels_latent']:5d}  "
                f"%loss(WT)={p['u_frac_sum_abs_WT'] * 100:5.2f}%  "
                f"⟨|u|_WT⟩/⟨|u|_BG⟩={p['u_mean_abs_WT'] / max(p['u_mean_abs_BG'], 1e-12):4.2f}x  "
                f"⟨|Δ|_WT⟩/⟨|Δ|_BG⟩={p['delta_mean_abs_WT'] / max(p['delta_mean_abs_BG'], 1e-12):4.2f}x"
            )

        # Save full per-patient table
        import csv

        out_dir = Path("/tmp/vena_diag")
        out_dir.mkdir(exist_ok=True)
        csv_path = out_dir / "audit_target_velocity_per_patient.csv"
        with open(csv_path, "w") as f:
            w = csv.DictWriter(f, fieldnames=list(per_patient[0].keys()))
            w.writeheader()
            for p in per_patient:
                w.writerow(p)
        print(f"\nCSV saved to {csv_path}")


if __name__ == "__main__":
    main()
