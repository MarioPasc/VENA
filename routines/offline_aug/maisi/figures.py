"""QC figures for the offline-augmentation routine.

For each (cohort, variant) cell, render a 6-panel composite per sampled
patient: original | augmented | decoded D(E(aug)) | absolute diff |
SSIM-map | per-modality PSNR/SSIM bars. The figure exists to support a
qualitative call that "the autoencoder preserves the augmentation"; the
gating decision is taken in the engine on quantitative aggregates
(cohort × variant medians vs the equivariance preflight's
``vae_recon_floor``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import (  # type: ignore[import-untyped]
    peak_signal_noise_ratio,
    structural_similarity,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AugRoundtripRow:
    """One QC row: original → augmented → decoded(E(augmented)).

    All volumes are in the ``[0, 1]`` foreground-percentile-normalised space
    so PSNR/SSIM with ``data_range=1`` is the right scale.
    """

    patient_id: str
    cohort: str
    variant: str
    modality: str
    original: np.ndarray  # (H, W, D)
    augmented: np.ndarray  # (H, W, D)
    decoded: np.ndarray  # (H, W, D)


def compute_psnr_ssim(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    """Per-volume PSNR (dB) and SSIM between two ``[0, 1]`` 3-D volumes.

    Uses :func:`skimage.metrics.peak_signal_noise_ratio` and
    :func:`skimage.metrics.structural_similarity` (3-D variant). SSIM uses
    the default Gaussian-windowed implementation with ``data_range=1.0``.
    """
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape:
        raise ValueError(
            f"shape mismatch: reference {reference.shape} != candidate {candidate.shape}"
        )
    psnr = float(peak_signal_noise_ratio(reference, candidate, data_range=1.0))
    ssim = float(
        structural_similarity(
            reference,
            candidate,
            data_range=1.0,
            channel_axis=None,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
        )
    )
    return psnr, ssim


def _mid_slice(vol: np.ndarray) -> np.ndarray:
    """Mid axial slice for a (H, W, D) volume."""
    return vol[:, :, vol.shape[-1] // 2]


def render_aug_roundtrip_figure(
    rows: list[AugRoundtripRow],
    output_path: Path,
    title: str | None = None,
    dpi: int = 200,
) -> Path:
    """Render one composite per cell.

    Layout: one row per modality, five columns —
    original | augmented | decoded | |aug − decoded| | SSIM-map (a
    visual proxy: per-voxel squared error normalised). The per-modality
    PSNR/SSIM values are annotated on the third column.
    """
    if not rows:
        raise ValueError("no rows to render")
    modalities = [r.modality for r in rows]
    n_mod = len(modalities)
    fig, axes = plt.subplots(n_mod, 5, figsize=(15, 3 * n_mod), dpi=dpi)
    if n_mod == 1:
        axes = np.array([axes])

    for i, row in enumerate(rows):
        a_orig = _mid_slice(row.original)
        a_aug = _mid_slice(row.augmented)
        a_dec = _mid_slice(row.decoded)
        a_diff = np.abs(a_aug - a_dec)
        a_se = (a_aug - a_dec) ** 2
        a_se = a_se / (a_se.max() + 1e-12)
        psnr, ssim = compute_psnr_ssim(row.augmented, row.decoded)

        for ax, img, name in zip(
            axes[i, :],
            (a_orig, a_aug, a_dec, a_diff, a_se),
            ("original", f"{row.variant}", "D(E(aug))", "|aug − D(E)|", "rel SE"),
        ):
            ax.imshow(np.rot90(img), cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_axis_off()
            ax.set_title(name, fontsize=9)
        axes[i, 0].set_ylabel(row.modality, fontsize=10)
        axes[i, 2].set_title(
            f"D(E(aug))  PSNR={psnr:.1f} dB  SSIM={ssim:.3f}",
            fontsize=9,
        )

    if title:
        fig.suptitle(title, fontsize=11)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", output_path)
    return output_path


def aggregate_cohort_variant_stats(
    rows: list[AugRoundtripRow],
) -> dict[tuple[str, str], dict[str, dict[str, float]]]:
    """Aggregate per-(cohort, variant) PSNR/SSIM across modalities.

    Returns
    -------
    dict
        Keyed by ``(cohort, variant)``; each value is
        ``{"per_modality": {<mod>: {"psnr_db": .., "ssim": ..}},
            "aggregate": {"median_psnr_db": .., "median_ssim": ..,
                          "min_psnr_db": .., "min_ssim": ..,
                          "n_patients": ..}}``.
    """
    bucket: dict[tuple[str, str], dict[str, list[tuple[float, float]]]] = {}
    for row in rows:
        psnr, ssim = compute_psnr_ssim(row.augmented, row.decoded)
        cell = bucket.setdefault((row.cohort, row.variant), {})
        cell.setdefault(row.modality, []).append((psnr, ssim))
    out: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    for key, per_mod in bucket.items():
        per_modality: dict[str, dict[str, float]] = {}
        all_psnr: list[float] = []
        all_ssim: list[float] = []
        for mod, pairs in per_mod.items():
            psnrs = [p for p, _ in pairs]
            ssims = [s for _, s in pairs]
            per_modality[mod] = {
                "median_psnr_db": float(np.median(psnrs)),
                "median_ssim": float(np.median(ssims)),
                "min_psnr_db": float(min(psnrs)),
                "min_ssim": float(min(ssims)),
                "n": len(pairs),
            }
            all_psnr.extend(psnrs)
            all_ssim.extend(ssims)
        out[key] = {
            "per_modality": per_modality,
            "aggregate": {
                "median_psnr_db": float(np.median(all_psnr)),
                "median_ssim": float(np.median(all_ssim)),
                "min_psnr_db": float(min(all_psnr)),
                "min_ssim": float(min(all_ssim)),
                "n_observations": len(all_psnr),
            },
        }
    return out
