"""Generate per-cohort visual-proof PNGs for the 2026-06-18 data fix-up.

For each of the 9 cohorts, render a 3-axial-slice panel showing the
NEW brain mask on top of the t1c image. Additionally:

* For BraTS-Africa cohorts: a 4th row visualising the (z-scored) t1c
  vs the percentile-normalised version that the encoder NOW sees with
  ``mask=`` — proves intra-brain negatives survive instead of being
  clamped to 0.
* For one BraTS-GLI patient with all 4 v* variants: 4 small panels
  showing the latent-grid brain mask per variant (proves v4 is no
  longer the all-ones synth shortcut).

Outputs:

* ``figures_post_fix/<cohort>_brain_mask.png``  — image-domain brain QC
* ``figures_post_fix/brats_africa_intensity_fix.png``  — z-score fix
* ``figures_post_fix/brats_gli_v4_variants_brain.png`` — v4 brain proof
* ``figures_post_fix/index.md`` — short markdown index
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_COHORTS = [
    ("UCSF-PDGM", "UCSF_PDGM/h5/UCSFPDGM_image.h5"),
    ("BraTS-GLI", "BRATS_GLI/PRE_OPERATIVE/h5/BraTS_GLI_image.h5"),
    ("UPENN-GBM", "upenn_gbm/h5/UPENN-GBM_image.h5"),
    ("IvyGAP", "ivy_gap/h5/IvyGAP_image.h5"),
    ("LUMIERE", "lumiere/h5/LUMIERE_image.h5"),
    ("REMBRANDT", "rembrandt/h5/REMBRANDT_image.h5"),
    ("BraTS-Africa-Glioma", "brats_africa_glioma/h5/BraTS_Africa_glioma_image.h5"),
    ("BraTS-Africa-Other", "brats_africa_other/h5/BraTS_Africa_other_image.h5"),
    ("BraTS-PED", "brats_ped/h5/BraTS_PED_image.h5"),
]


def _normalise_for_display(vol: np.ndarray) -> np.ndarray:
    """Linear scale to [0,1] using 1st-99th percentile of non-zero voxels."""
    fg = vol[vol > 0]
    if fg.size == 0:
        return vol
    lo, hi = np.percentile(fg, [1, 99])
    return np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)


def _render_cohort_brain_qc(name: str, image_h5: Path, out_dir: Path) -> Path:
    """3-row × 3-col figure: axial slices at z=0.3D / 0.5D / 0.7D, t1c | brain mask | overlay."""
    with h5py.File(image_h5, "r") as f:
        t1c = np.asarray(f["images/t1c"][0])
        brain = np.asarray(f["masks/brain"][0])
        scan_id = f["ids"][0].decode() if isinstance(f["ids"][0], bytes) else str(f["ids"][0])
    H, W, D = t1c.shape
    z_idx = [int(0.3 * D), int(0.5 * D), int(0.7 * D)]
    t1c_norm = _normalise_for_display(t1c)
    fig, axes = plt.subplots(3, 3, figsize=(9, 9), gridspec_kw={"wspace": 0.02, "hspace": 0.15})
    for r, z in enumerate(z_idx):
        # T1c
        axes[r, 0].imshow(t1c_norm[:, :, z].T, cmap="gray", origin="lower")
        axes[r, 0].set_title(f"t1c (z={z}/{D})" if r == 0 else f"z={z}/{D}", fontsize=9)
        axes[r, 0].axis("off")
        # Brain mask
        axes[r, 1].imshow(brain[:, :, z].T, cmap="Reds", origin="lower", vmin=0, vmax=1)
        axes[r, 1].set_title("masks/brain (post-fix)" if r == 0 else "", fontsize=9)
        axes[r, 1].axis("off")
        # Overlay
        axes[r, 2].imshow(t1c_norm[:, :, z].T, cmap="gray", origin="lower")
        axes[r, 2].imshow(brain[:, :, z].T, cmap="autumn", origin="lower", alpha=0.35)
        axes[r, 2].set_title("overlay" if r == 0 else "", fontsize=9)
        axes[r, 2].axis("off")
    fig.suptitle(f"{name}  ·  scan={scan_id}", fontsize=11)
    out_path = out_dir / f"{name.lower().replace(' ', '_')}_brain_mask.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _render_brats_africa_intensity_fix(root: Path, out_dir: Path) -> Path:
    """Visualise the percentile_normalise(mask=) fix on the z-score cohort."""
    img_p = root / "brats_africa_glioma/h5/BraTS_Africa_glioma_image.h5"
    if not img_p.exists():
        return None
    with h5py.File(img_p, "r") as f:
        t1c = np.asarray(f["images/t1c"][0])
        brain = np.asarray(f["masks/brain"][0])
        scan_id = f["ids"][0].decode() if isinstance(f["ids"][0], bytes) else str(f["ids"][0])
    H, W, D = t1c.shape
    z = int(0.5 * D)
    # Heuristic path: foreground = x > 0 (clamps negative intra-brain to 0).
    fg_heur = t1c[t1c > 0]
    if fg_heur.size:
        lo_h, hi_h = np.percentile(fg_heur, [0, 99.5])
        t1c_heur = np.clip((t1c - lo_h) / max(hi_h - lo_h, 1e-6), 0, 1)
    else:
        t1c_heur = np.zeros_like(t1c)
    # Mask-aware path.
    fg_mask = t1c[brain > 0]
    if fg_mask.size:
        lo_m, hi_m = np.percentile(fg_mask, [0, 99.5])
        t1c_mask = np.clip((t1c - lo_m) / max(hi_m - lo_m, 1e-6), 0, 1)
    else:
        t1c_mask = np.zeros_like(t1c)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), gridspec_kw={"wspace": 0.05})
    axes[0].imshow(t1c[:, :, z].T, cmap="gray", origin="lower")
    axes[0].set_title("raw z-scored t1c\n(min<0)", fontsize=9)
    axes[0].axis("off")
    axes[1].imshow(brain[:, :, z].T, cmap="Reds", origin="lower", vmin=0, vmax=1)
    axes[1].set_title("masks/brain", fontsize=9)
    axes[1].axis("off")
    axes[2].imshow(t1c_heur[:, :, z].T, cmap="gray", origin="lower", vmin=0, vmax=1)
    axes[2].set_title(
        "OLD: foreground_only=True\n(negatives clamped → ~half\nthe brain disappears)",
        fontsize=9,
    )
    axes[2].axis("off")
    axes[3].imshow(t1c_mask[:, :, z].T, cmap="gray", origin="lower", vmin=0, vmax=1)
    axes[3].set_title(
        "NEW: mask=masks/brain\n(intra-brain negatives survive)",
        fontsize=9,
    )
    axes[3].axis("off")
    fig.suptitle(f"BraTS-Africa-Glioma intensity-fix proof  ·  scan={scan_id}", fontsize=11)
    out_path = out_dir / "brats_africa_intensity_fix.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _render_v4_brain_proof(root: Path, out_dir: Path) -> Path:
    """For one BraTS-GLI patient, render the latent-grid brain mask per variant."""
    aug_lat = root / "BRATS_GLI/PRE_OPERATIVE/h5/brats_gli_latents_aug.h5"
    if not aug_lat.exists():
        return None
    with h5py.File(aug_lat, "r") as f:
        ids = [x.decode() if isinstance(x, bytes) else str(x) for x in f["ids"][:]]
        variants = [x.decode() if isinstance(x, bytes) else str(x) for x in f["variants"][:]]
        bl = f["masks/brain_latent"]
        # Pick the first patient with all 4 variants present.
        seen: dict[str, dict[str, int]] = {}
        for i, (pid, v) in enumerate(zip(ids, variants, strict=True)):
            seen.setdefault(pid, {})[v] = i
            if len(seen[pid]) == 4:
                target_pid = pid
                rows = seen[pid]
                break
        else:
            return None
        masks = {v: np.asarray(bl[idx])[0] for v, idx in rows.items()}  # (48,56,48) each
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), gridspec_kw={"wspace": 0.05})
    for ax, v in zip(axes, ("v1", "v2", "v3", "v4"), strict=True):
        m = masks[v]
        z = m.shape[-1] // 2
        ax.imshow(m[:, :, z].T, cmap="Reds", origin="lower", vmin=0, vmax=1)
        ax.set_title(f"{v}  ·  sum={int(m.sum())}", fontsize=10)
        ax.axis("off")
    title = f"BraTS-GLI v4 brain proof  ·  patient={target_pid}\n(v4 sum != 129024 ⇒ no synth-ones)"
    fig.suptitle(title, fontsize=11)
    out_path = out_dir / "brats_gli_v4_variants_brain.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/vena"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/vena/_audit_post_fix_figures"
        ),
    )
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for name, rel in DEFAULT_COHORTS:
        p = args.root / rel
        if not p.exists():
            print(f"SKIP {name}: image H5 missing ({p})")
            continue
        out = _render_cohort_brain_qc(name, p, args.out_dir)
        paths.append(out)
        print(f"wrote {out}")
    africa = _render_brats_africa_intensity_fix(args.root, args.out_dir)
    if africa:
        paths.append(africa)
        print(f"wrote {africa}")
    v4 = _render_v4_brain_proof(args.root, args.out_dir)
    if v4:
        paths.append(v4)
        print(f"wrote {v4}")

    index = args.out_dir / "index.md"
    with index.open("w") as f:
        f.write("# 2026-06-18 post-fix visual proof\n\n")
        for p in paths:
            f.write(f"- ![{p.stem}]({p.name})\n")
    print(f"wrote index {index}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
