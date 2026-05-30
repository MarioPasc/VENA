"""Per-transform side-by-side panels for the equivariance preflight.

For each (cohort, augmentation) pair we render a 5-row x N-slice axial panel:
real T1c, decoded(z), T_image(decoded(z)) [gold], decoded(T_latent(z))
[proposed], absolute difference between the gold and proposed paths. The
figure is the qualitative companion to the per-patient metric CSV.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def render_equivariance_panel(
    real: torch.Tensor,
    recon: torch.Tensor,
    image_path: torch.Tensor,
    latent_path: torch.Tensor,
    slice_indices: list[int],
    *,
    cohort: str,
    transform_name: str,
    param_tag: str,
    psnr_db: float,
    ssim: float,
    out_path: Path | str,
) -> Path:
    """Render a five-row qualitative panel and save it as PNG.

    Parameters
    ----------
    real : Tensor
        Reference real T1c ``(H, W, D)`` in ``[0, 1]``.
    recon : Tensor
        Decoded latent ``D(z)`` ``(H, W, D)`` in ``[0, 1]``.
    image_path : Tensor
        ``T_image(D(z))``: gold path (transform in image space).
    latent_path : Tensor
        ``D(T_latent(z))``: proposed path (transform in latent space).
    slice_indices : list[int]
        Axial indices to display.
    cohort, transform_name, param_tag : str
        Title metadata.
    psnr_db, ssim : float
        Pair-metrics summarising the equivariance gap on this volume.
    out_path : Path | str
        PNG destination.

    Returns
    -------
    Path
        ``out_path``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        ("Real T1c", real),
        ("D(z)", recon),
        ("T_img(D(z))  [gold]", image_path),
        ("D(T_lat(z))  [proposed]", latent_path),
        ("|gold − proposed|", (image_path - latent_path).abs()),
    ]
    n_rows, n_cols = len(rows), len(slice_indices)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.5 * n_cols, 1.55 * n_rows), squeeze=False)
    for r, (label, vol) in enumerate(rows):
        arr = vol.detach().cpu().float().numpy()
        # Difference row uses a tighter colour range for legibility.
        vmin, vmax = (0.0, 1.0) if label != "|gold − proposed|" else (0.0, 0.2)
        for c, k in enumerate(slice_indices):
            ax = axes[r][c]
            ax.imshow(np.rot90(arr[..., k]), cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"z={k}", fontsize=7)
            if c == 0:
                ax.set_ylabel(label, fontsize=8)
    title = (
        f"{cohort} — {transform_name} [{param_tag}]    "
        f"PSNR(gold, proposed)={psnr_db:.2f} dB, SSIM={ssim:.4f}"
    )
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_summary_boxplot(
    rows: list[dict],
    out_path: Path | str,
) -> Path:
    """Boxplot of PSNR per transform across all (cohort × patient) draws.

    Parameters
    ----------
    rows : list[dict]
        Per-patient metric rows; must carry ``transform``, ``psnr_db``,
        ``ssim`` keys.
    out_path : Path | str
        PNG destination.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_t: dict[str, list[float]] = {}
    for r in rows:
        by_t.setdefault(str(r["transform"]), []).append(float(r["psnr_db"]))
    names = sorted(by_t)
    data = [by_t[n] for n in names]

    fig, ax = plt.subplots(figsize=(max(6.0, 0.8 * len(names) + 2), 4.5))
    ax.boxplot(data, labels=names, showfliers=True)
    ax.axhline(35.0, color="red", linewidth=1.0, linestyle="--", label="pass: 35 dB")
    ax.set_ylabel("PSNR(gold, proposed) [dB]")
    ax.set_xlabel("Transform")
    ax.set_title("Equivariance gap per transform (higher = better)")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path
