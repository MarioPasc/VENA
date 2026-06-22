"""Figure renderers for the V3 normalisation audit report.

Each function writes a single PNG to ``out_dir`` and returns its path.
All figures use matplotlib's headless backend (set by the routine entry
point); the renderers themselves are agnostic to backend choice.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def render_intensity_histograms(
    histograms_per_variant: dict[str, np.ndarray],
    *,
    modality: str,
    bin_centres: np.ndarray,
    v0_percentile_cuts: dict[str, float] | None = None,
    out_path: Path,
) -> Path:
    """Per-modality intensity histogram with one line per variant.

    Parameters
    ----------
    histograms_per_variant : dict[variant_id, histogram]
        1-D probability array, length ``bins``.
    bin_centres : np.ndarray
        Centre value of each histogram bin.
    v0_percentile_cuts : dict[str, float] | None
        Optional vertical dashed lines at V0's percentile cut points
        (e.g. ``{"99.5": 1.0, "99.9": 1.05}``).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for vid, hist in histograms_per_variant.items():
        ax.plot(bin_centres, hist, label=vid, linewidth=1.2)
    if v0_percentile_cuts:
        for label, x in v0_percentile_cuts.items():
            ax.axvline(x, color="grey", linestyle="--", alpha=0.5)
            ax.text(x, ax.get_ylim()[1] * 0.95, f"V0 {label}%", rotation=90, fontsize=7, va="top")
    ax.set_yscale("log")
    ax.set_xlabel(f"Normalised intensity ({modality})")
    ax.set_ylabel("Probability (log)")
    ax.set_title(f"Intensity histogram — {modality}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def render_per_region_psnr_bar(
    psnr_per_variant: dict[str, dict[str, float]],
    *,
    threshold_db: float,
    out_path: Path,
) -> Path:
    """Grouped-bar chart of per-region PSNR per variant.

    ``psnr_per_variant[variant_id][region]`` for regions ∈
    {whole, et, netc, ed, bnwt}.
    """
    regions = ["whole", "et", "netc", "ed", "bnwt"]
    variants = list(psnr_per_variant.keys())
    n_v = len(variants)
    n_r = len(regions)
    x = np.arange(n_r)
    width = 0.8 / max(n_v, 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, vid in enumerate(variants):
        vals = [psnr_per_variant[vid].get(r, np.nan) for r in regions]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width=width, label=vid)
    ax.axhline(
        threshold_db, color="red", linestyle="--", alpha=0.6, label=f"C7 ≥ {threshold_db} dB"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(regions)
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Per-region PSNR — VAE round-trip on T1c")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def render_signal_ratio_scatter(
    image_ratios: dict[str, float],
    latent_ratios: dict[str, float],
    *,
    c4_threshold: float,
    c5_threshold: float,
    out_path: Path,
) -> Path:
    """Scatter C4 (x) vs C5 (y) per variant; threshold lines at 1.5 / 1.3."""
    variants = sorted(set(image_ratios) | set(latent_ratios))
    fig, ax = plt.subplots(figsize=(7, 6))
    for vid in variants:
        x = image_ratios.get(vid, np.nan)
        y = latent_ratios.get(vid, np.nan)
        ax.scatter(x, y, s=60)
        ax.annotate(vid, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.axvline(c4_threshold, color="red", linestyle="--", alpha=0.6, label=f"C4 ≥ {c4_threshold}")
    ax.axhline(c5_threshold, color="red", linestyle="--", alpha=0.6, label=f"C5 ≥ {c5_threshold}")
    ax.axvline(1.0, color="grey", linestyle=":", alpha=0.4)
    ax.axhline(1.0, color="grey", linestyle=":", alpha=0.4)
    ax.set_xlabel("Image-space C4 ratio (⟨|T1c − T1pre|⟩_ET / ⟨|·|⟩_BNWT)")
    ax.set_ylabel("Latent-space C5 ratio (⟨|z_t1c − z_t1pre|⟩_ET / ⟨|·|⟩_BNWT)")
    ax.set_title("Signal-preservation scatter — each variant is a point")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def render_kl_divergence_bar(
    kl_per_variant_per_modality: dict[str, dict[str, float]],
    *,
    threshold_nats: float,
    out_path: Path,
) -> Path:
    """Grouped-bar chart of KL(V_i || V0) per (variant, modality)."""
    variants = list(kl_per_variant_per_modality.keys())
    if not variants:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "no KL data", ha="center", va="center")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return out_path
    modalities = sorted({m for d in kl_per_variant_per_modality.values() for m in d.keys()})
    n_v = len(variants)
    n_m = len(modalities)
    x = np.arange(n_m)
    width = 0.8 / max(n_v, 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, vid in enumerate(variants):
        vals = [kl_per_variant_per_modality[vid].get(m, np.nan) for m in modalities]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width=width, label=vid)
    ax.axhline(
        threshold_nats, color="red", linestyle="--", alpha=0.6, label=f"C3 ≤ {threshold_nats} nats"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(modalities)
    ax.set_ylabel("KL divergence (nats)")
    ax.set_title("KL(V_i || V0) per (variant, modality)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def render_recon_grid(
    real_volumes: dict[str, np.ndarray],
    recon_volumes: dict[str, np.ndarray],
    *,
    variant_id: str,
    patient_id: str,
    out_path: Path,
) -> Path:
    """Per-variant recon grid: 4 modalities × 6 columns (3 slices × {orig, recon}).

    Volumes are ``(H, W, D)`` numpy arrays in normalised intensity space.
    Mid-axial slices at indices ``D//4, D//2, 3*D//4`` are rendered.
    """
    modalities = ["t1pre", "t1c", "t2", "flair"]
    fig, axes = plt.subplots(len(modalities), 7, figsize=(15, 10), constrained_layout=True)
    fig.suptitle(f"Recon grid — variant {variant_id} — patient {patient_id}", fontsize=12)
    for r, mod in enumerate(modalities):
        if mod not in real_volumes or mod not in recon_volumes:
            for c in range(7):
                axes[r, c].axis("off")
            continue
        real = real_volumes[mod]
        rec = recon_volumes[mod]
        H, W, D = real.shape
        z_idx = [D // 4, D // 2, (3 * D) // 4]
        for ci, z in enumerate(z_idx):
            r_slice = real[:, :, z]
            p_slice = rec[:, :, z]
            vmin = float(min(r_slice.min(), p_slice.min()))
            vmax = float(max(r_slice.max(), p_slice.max()))
            axes[r, 2 * ci].imshow(r_slice.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
            axes[r, 2 * ci + 1].imshow(p_slice.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
            axes[r, 2 * ci].set_title(f"{mod} z={z} orig", fontsize=7)
            axes[r, 2 * ci + 1].set_title(f"{mod} z={z} recon", fontsize=7)
            axes[r, 2 * ci].axis("off")
            axes[r, 2 * ci + 1].axis("off")
        # 7th column: |orig - recon| ×5 (visual amplification).
        diff = np.abs(real - rec) * 5.0
        d_slice = diff[:, :, D // 2]
        axes[r, 6].imshow(d_slice.T, cmap="hot", vmin=0.0, vmax=1.0, origin="lower")
        axes[r, 6].set_title(f"{mod} |diff|×5", fontsize=7)
        axes[r, 6].axis("off")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


__all__ = [
    "render_intensity_histograms",
    "render_kl_divergence_bar",
    "render_per_region_psnr_bar",
    "render_recon_grid",
    "render_signal_ratio_scatter",
]
