#!/usr/bin/env python3
"""Per-region image-space contrast (T1c - T1pre) audit.

Mirrors VENA's `percentile_normalise(lower=0, upper=99.5, foreground_only=True)`
on each modality, then decomposes the absolute T1c-T1pre difference and the
absolute mean(T1c-T1pre) by region. Confirms that enhancement is genuinely
present and quantifies how much signal the model has to extract.
"""

from __future__ import annotations

import h5py
import numpy as np

IMAGE_H5 = "/media/hddb/mario/data/GLIOMAS/UCSF_PDGM/h5/UCSFPDGM_image.h5"


def percentile_normalise(x: np.ndarray, brain_mask: np.ndarray, lo=0.0, hi=99.5) -> np.ndarray:
    """Skull-stripped, foreground-only [lo, hi] percentile to [0, 1] with clipping."""
    foreground = x[brain_mask > 0]
    if foreground.size == 0:
        return np.zeros_like(x)
    lov, hiv = np.percentile(foreground, [lo, hi])
    if hiv <= lov:
        return np.zeros_like(x)
    out = (x - lov) / (hiv - lov)
    return np.clip(out, 0, 1).astype(np.float32)


def main():
    rng = np.random.default_rng(1337)
    with h5py.File(IMAGE_H5, "r") as f:
        ids = [b.decode() if isinstance(b, (bytes, np.bytes_)) else b for b in f["ids"][:]]
        val_ids = [
            b.decode() if isinstance(b, (bytes, np.bytes_)) else b
            for b in f["splits/cv/fold_0/val"][:]
        ]
        id_to_idx = {x: i for i, x in enumerate(ids)}
        pick = rng.choice(len(val_ids), size=min(15, len(val_ids)), replace=False)
        pick_ids = [val_ids[i] for i in pick]

        print("=== Image-space T1c-T1pre contrast audit (VENA percentile-norm parity) ===")
        print(f"Image H5: {IMAGE_H5}")
        print(f"N patients: {len(pick_ids)}\n")

        # Accumulators
        regs = ("BG", "BRAIN_NOT_WT", "WT", "ET")
        acc = {
            r: {
                "n": 0,
                "abs_sum_diff": 0.0,
                "mean_t1c": 0.0,
                "mean_t1pre": 0.0,
                "n_pat": 0,
                "max_abs": 0.0,
            }
            for r in regs
        }
        per_patient = []
        for pid in pick_ids:
            idx = id_to_idx[pid]
            t1pre_raw = f["images/t1pre"][idx].astype(np.float32)
            t1c_raw = f["images/t1c"][idx].astype(np.float32)
            brain = f["masks/brain"][idx]  # int8, 0/1
            tumor_lbl = f["masks/tumor"][idx]  # BraTS labels: 0/1/2/4

            t1pre = percentile_normalise(t1pre_raw, brain)
            t1c = percentile_normalise(t1c_raw, brain)

            wt = tumor_lbl > 0  # whole tumor (1/2/4)
            et = tumor_lbl == 4  # enhancing tumor only
            bnwt = (brain > 0) & ~wt
            bg = brain == 0

            diff = t1c - t1pre
            mks = {"BG": bg, "BRAIN_NOT_WT": bnwt, "WT": wt, "ET": et}
            row = {"pid": pid, "wt_voxels_image": int(wt.sum()), "et_voxels_image": int(et.sum())}
            for r, m in mks.items():
                if not m.any():
                    row[f"{r}_mean_t1c"] = float("nan")
                    row[f"{r}_mean_t1pre"] = float("nan")
                    row[f"{r}_mean_abs_diff"] = float("nan")
                    continue
                d = diff[m]
                row[f"{r}_mean_t1c"] = float(t1c[m].mean())
                row[f"{r}_mean_t1pre"] = float(t1pre[m].mean())
                row[f"{r}_mean_diff"] = float(d.mean())
                row[f"{r}_mean_abs_diff"] = float(np.abs(d).mean())
                row[f"{r}_max_abs"] = float(np.abs(d).max())
                acc[r]["n"] += int(m.sum())
                acc[r]["abs_sum_diff"] += float(np.abs(d).sum())
                acc[r]["mean_t1c"] += float(t1c[m].mean())
                acc[r]["mean_t1pre"] += float(t1pre[m].mean())
                acc[r]["n_pat"] += 1
                acc[r]["max_abs"] = max(acc[r]["max_abs"], float(np.abs(d).max()))
            per_patient.append(row)

        print(
            f"{'Region':<14} {'Vox':>14} {'⟨T1c⟩':>9} {'⟨T1pre⟩':>9} {'⟨|Δ|⟩':>9} {'%∑|Δ|':>9} {'maxΔ':>9}"
        )
        print("-" * 84)
        total_abs = sum(acc[r]["abs_sum_diff"] for r in regs)
        for r in regs:
            n = acc[r]["n"]
            t1c_m = acc[r]["mean_t1c"] / max(acc[r]["n_pat"], 1)
            t1p_m = acc[r]["mean_t1pre"] / max(acc[r]["n_pat"], 1)
            md = acc[r]["abs_sum_diff"] / max(n, 1)
            fp = acc[r]["abs_sum_diff"] / max(total_abs, 1e-12)
            print(
                f"  {r:<12} {n:>14,d} {t1c_m:>9.4f} {t1p_m:>9.4f} {md:>9.4f} {fp * 100:>8.4f}% {acc[r]['max_abs']:>9.4f}"
            )
        print()
        print("Interpretation:")
        wt_signal = acc["WT"]["abs_sum_diff"] / max(total_abs, 1e-12)
        et_signal = acc["ET"]["abs_sum_diff"] / max(total_abs, 1e-12)
        nbnwt_signal = acc["BRAIN_NOT_WT"]["abs_sum_diff"] / max(total_abs, 1e-12)
        bg_signal = acc["BG"]["abs_sum_diff"] / max(total_abs, 1e-12)
        print(
            f"  In IMAGE space (skull-stripped), ⟨|T1c-T1pre|⟩ in WT = {acc['WT']['abs_sum_diff'] / max(acc['WT']['n'], 1):.4f}"
        )
        print(
            f"  vs ⟨|T1c-T1pre|⟩ in BRAIN_NOT_WT = {acc['BRAIN_NOT_WT']['abs_sum_diff'] / max(acc['BRAIN_NOT_WT']['n'], 1):.4f}"
        )
        print(
            f"  Ratio (WT mean abs diff / non-WT mean abs diff) = "
            f"{(acc['WT']['abs_sum_diff'] / max(acc['WT']['n'], 1)) / max((acc['BRAIN_NOT_WT']['abs_sum_diff'] / max(acc['BRAIN_NOT_WT']['n'], 1)), 1e-12):.2f}x"
        )
        print(
            f"  Per-voxel WT signal is stronger, BUT WT covers only {acc['WT']['n'] / (acc['WT']['n'] + acc['BRAIN_NOT_WT']['n'] + acc['BG']['n']) * 100:.3f}% of voxels"
        )
        print(f"  → WT contributes {wt_signal * 100:.3f}% of total L1 magnitude in image space.")
        print(f"  → ET (enhancing only) contributes {et_signal * 100:.4f}%.")

        # Per-patient hyperintensity stats (how bright is T1c-vs-T1pre in enhancing voxels)
        print()
        print("Per-patient T1c vs T1pre in ENHANCING region (et_mean):")
        for p in per_patient[:10]:
            wt_n = p.get("wt_voxels_image", 0)
            if wt_n == 0:
                continue
            et_t1c = p.get("ET_mean_t1c", float("nan"))
            et_t1p = p.get("ET_mean_t1pre", float("nan"))
            et_d = p.get("ET_mean_diff", float("nan"))
            bnwt_t1c = p.get("BRAIN_NOT_WT_mean_t1c", float("nan"))
            print(
                f"  {p['pid']}  wt={p['wt_voxels_image']:6d}  et={p['et_voxels_image']:6d}  "
                f"⟨T1c⟩_ET={et_t1c:.3f} ⟨T1pre⟩_ET={et_t1p:.3f}  Δ_ET=+{et_d:.3f}  "
                f"⟨T1c⟩_brain={bnwt_t1c:.3f}"
            )


if __name__ == "__main__":
    main()
