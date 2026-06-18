"""Post-fix audit summary for the 2026-06-18 data audit follow-up.

Walks every cohort's image + latent + aug-latent H5 on Picasso and prints:

* For image H5: brain mask CC count, brain volume in voxels, cohort attrs.
* For latent H5: brain_latent presence + sum, produced_by_brain_to_latent
  flag, schema_version.
* For aug-latent H5: brain_latent presence + sum (per variant),
  v4_brain_synthesised_ones flag (must be False after fix), per-variant
  brain-sum range.

Used by Phase 7 of the 2026-06-18 fix-up to produce the v2 audit memo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

DEFAULT_COHORTS = [
    (
        "UCSF-PDGM",
        "cv",
        "UCSF_PDGM/h5/UCSFPDGM_image.h5",
        "UCSF_PDGM/h5/UCSFPDGM_latents.h5",
        "UCSF_PDGM/h5/ucsf_pdgm_latents_aug.h5",
    ),
    (
        "BraTS-GLI",
        "cv",
        "BRATS_GLI/PRE_OPERATIVE/h5/BraTS_GLI_image.h5",
        "BRATS_GLI/PRE_OPERATIVE/h5/BraTS_GLI_latents.h5",
        "BRATS_GLI/PRE_OPERATIVE/h5/brats_gli_latents_aug.h5",
    ),
    (
        "UPENN-GBM",
        "cv",
        "upenn_gbm/h5/UPENN-GBM_image.h5",
        "upenn_gbm/h5/UPENN-GBM_latents.h5",
        "upenn_gbm/h5/upenn_gbm_latents_aug.h5",
    ),
    (
        "IvyGAP",
        "cv",
        "ivy_gap/h5/IvyGAP_image.h5",
        "ivy_gap/h5/IvyGAP_latents.h5",
        "ivy_gap/h5/ivy_gap_latents_aug.h5",
    ),
    (
        "LUMIERE",
        "cv",
        "lumiere/h5/LUMIERE_image.h5",
        "lumiere/h5/LUMIERE_latents.h5",
        "lumiere/h5/lumiere_latents_aug.h5",
    ),
    (
        "REMBRANDT",
        "cv",
        "rembrandt/h5/REMBRANDT_image.h5",
        "rembrandt/h5/REMBRANDT_latents.h5",
        "rembrandt/h5/rembrandt_latents_aug.h5",
    ),
    (
        "BraTS-Africa-Glioma",
        "test_only",
        "brats_africa_glioma/h5/BraTS_Africa_glioma_image.h5",
        "brats_africa_glioma/h5/BraTS_Africa_glioma_latents.h5",
        None,
    ),
    (
        "BraTS-Africa-Other",
        "test_only",
        "brats_africa_other/h5/BraTS_Africa_other_image.h5",
        "brats_africa_other/h5/BraTS_Africa_other_latents.h5",
        None,
    ),
    (
        "BraTS-PED",
        "test_only",
        "brats_ped/h5/BraTS_PED_image.h5",
        "brats_ped/h5/BraTS_PED_latents.h5",
        None,
    ),
]


def _audit_image(p: Path) -> dict:
    if not p.exists():
        return {"missing": True}
    out: dict = {}
    with h5py.File(p, "r") as f:
        bs = f["masks/brain"]
        out["n_rows"] = int(bs.shape[0])
        out["brain_attrs"] = {k: str(v) for k, v in bs.attrs.items()}
        sample_idx = list(range(min(3, out["n_rows"])))
        sums = [int((np.asarray(bs[i]) > 0).sum()) for i in sample_idx]
        out["sample_brain_voxels"] = sums
        out["has_negative_t1c"] = bool(np.asarray(f["images/t1c"][0]).min() < 0)
    return out


def _audit_latent(p: Path) -> dict:
    if not p.exists():
        return {"missing": True}
    out: dict = {}
    with h5py.File(p, "r") as f:
        v = f.attrs.get("schema_version")
        out["schema_version"] = v.decode() if isinstance(v, bytes) else str(v)
        out["produced_by_brain_to_latent"] = bool(f.attrs.get("produced_by_brain_to_latent", False))
        if "masks/brain_latent" in f:
            bl = f["masks/brain_latent"]
            out["brain_latent_shape"] = list(bl.shape)
            out["brain_latent_dtype"] = str(bl.dtype)
            sample_idx = list(range(min(3, bl.shape[0])))
            out["sample_brain_latent_sums"] = [int(bl[i].sum()) for i in sample_idx]
            out["v4_brain_synthesised_ones"] = bool(
                bl.attrs.get("v4_brain_synthesised_ones", False)
            )
        else:
            out["brain_latent_shape"] = None
    return out


def _audit_aug_latent(p: Path) -> dict:
    if p is None or not p.exists():
        return {"missing": True}
    out: dict = {}
    with h5py.File(p, "r") as f:
        v = f.attrs.get("schema_version")
        out["schema_version"] = v.decode() if isinstance(v, bytes) else str(v)
        out["produced_by_brain_to_latent"] = bool(f.attrs.get("produced_by_brain_to_latent", False))
        if "masks/brain_latent" in f:
            bl = f["masks/brain_latent"]
            out["brain_latent_shape"] = list(bl.shape)
            out["v4_brain_synthesised_ones"] = bool(
                bl.attrs.get("v4_brain_synthesised_ones", False)
            )
            variants = [x.decode() if isinstance(x, bytes) else str(x) for x in f["variants"][:]]
            n = bl.shape[0]
            per_v: dict = {}
            for v_name in sorted(set(variants)):
                rows = [i for i, vv in enumerate(variants) if vv == v_name][:5]
                sums = [int(bl[i].sum()) for i in rows]
                per_v[v_name] = {"n_sample": len(sums), "sample_sums": sums}
            out["per_variant_sample_sums"] = per_v
            v4_all = [int(bl[i].sum()) for i, vv in enumerate(variants) if vv == "v4"][:200]
            if v4_all:
                out["v4_sum_stats"] = {
                    "n_v4": len([vv for vv in variants if vv == "v4"]),
                    "min": int(min(v4_all)),
                    "max": int(max(v4_all)),
                    "mean": round(float(np.mean(v4_all)), 1),
                    "all_equal_synth_ones": all(s == 129024 for s in v4_all),
                }
        else:
            out["brain_latent_shape"] = None
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/vena"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/vena/_audit_post_fix.json"
        ),
    )
    args = parser.parse_args(argv)

    import json

    report: dict = {"cohorts": {}}
    for name, role, img_rel, lat_rel, aug_rel in DEFAULT_COHORTS:
        entry = {"role": role}
        entry["image"] = _audit_image(args.root / img_rel)
        entry["latent"] = _audit_latent(args.root / lat_rel)
        entry["aug_latent"] = _audit_aug_latent(args.root / aug_rel) if aug_rel else None
        report["cohorts"][name] = entry
        print(f"=== {name} (role={role}) ===")
        print(json.dumps(entry, indent=2))
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote summary to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
