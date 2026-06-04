"""Regenerate offline-aug QC figures with the CORRECT layout.

Existing-cohort retrofit for the figure bug: the first version of
``routines.offline_aug.maisi.engine._run_qc`` passed the augmented image
as both ``original`` and ``augmented`` to :class:`AugRoundtripRow`, so
the rendered figure showed the same image twice. The H5 contents are
correct — this script just re-renders the figures using the clean source
image (cropped + percentile-normalised) as ``original``.

Usage::

    python scripts/regen_aug_qc_figures.py \
        --cohort UCSF-PDGM \
        --source-image-h5 /media/hddb/mario/data/GLIOMAS/UCSF_PDGM/h5/UCSFPDGM_image.h5 \
        --image-aug-h5 /media/hddb/mario/data/GLIOMAS/UCSF_PDGM/h5/ucsf_pdgm_image_aug.h5 \
        --latent-aug-h5 /media/hddb/mario/data/GLIOMAS/UCSF_PDGM/h5/ucsf_pdgm_latents_aug.h5 \
        --autoencoder /media/hddb/mario/checkpoints/MAISI_V2_RM/NV-Generate-MR/models/autoencoder_v2.pt \
        --output-dir /media/hddb/mario/artifacts/offline_aug/qc_regen/UCSF-PDGM \
        --n-per-variant 4 \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from rich.logging import RichHandler
from routines.offline_aug.maisi.engine.offline_aug_engine import (
    _box_native_numpy,
)
from routines.offline_aug.maisi.figures import (
    AugRoundtripRow,
    compute_psnr_ssim,
    render_aug_roundtrip_figure,
)

from vena.data.augment.offline.variants import VARIANT_NAMES
from vena.data.h5.augmented import AUG_IMAGE_CROP_BOX
from vena.model.autoencoder.maisi.decode import MaisiDecoder
from vena.model.autoencoder.maisi.loader import load_autoencoder
from vena.model.autoencoder.maisi.preprocessing import (
    CropPadSpec,
    percentile_normalise,
)

logger = logging.getLogger(__name__)


def _percentile_normalise_np(arr: np.ndarray) -> np.ndarray:
    """Foreground-percentile normalise the same way the encoder did."""
    x = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0).unsqueeze(0).float()
    normed = percentile_normalise(x, lower=0.0, upper=99.5, foreground_only=True)
    return normed[0, 0].numpy()


def _decode_one(z_np: np.ndarray, decoder: MaisiDecoder, device: torch.device) -> np.ndarray:
    z = torch.from_numpy(z_np).unsqueeze(0).to(device, dtype=torch.float32)
    crop_spec = CropPadSpec(
        crop_origin=(0, 0, 0),
        native_shape=AUG_IMAGE_CROP_BOX,
        target_shape=AUG_IMAGE_CROP_BOX,
    )
    with torch.inference_mode():
        decoded = decoder.decode(z, crop_spec=crop_spec)
    return decoded.image[0, 0].clamp(0.0, 1.0).detach().to("cpu", torch.float32).numpy()


def main() -> int:
    parser = argparse.ArgumentParser(prog="vena-regen-aug-qc")
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--source-image-h5", type=Path, required=True)
    parser.add_argument("--image-aug-h5", type=Path, required=True)
    parser.add_argument("--latent-aug-h5", type=Path, required=True)
    parser.add_argument("--autoencoder", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modalities", nargs="+", default=["t1pre", "t1c", "t2", "flair"])
    parser.add_argument("--n-per-variant", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")

    handle = load_autoencoder(
        args.autoencoder, device=device, arch_overrides={"norm_float16": True}
    )
    decoder = MaisiDecoder(handle, precision_mode="autocast")

    rng = np.random.default_rng(args.seed)

    with h5py.File(args.image_aug_h5, "r") as f_aug:
        variants_arr = np.asarray(f_aug["variants"][:], dtype=object)
        ids_arr = np.asarray(f_aug["ids"][:], dtype=object)
        source_row_arr = np.asarray(f_aug["source_row_index"][:], dtype=np.int64)

    rows_by_variant: dict[str, list[int]] = {v: [] for v in VARIANT_NAMES}
    for i, v in enumerate(variants_arr):
        v_str = v.decode() if isinstance(v, (bytes, bytearray)) else str(v)
        if v_str in rows_by_variant:
            rows_by_variant[v_str].append(i)

    summary_rows: list[str] = [
        "| cohort | variant | patient | t1pre PSNR | t1c PSNR | t2 PSNR | flair PSNR |",
        "|---|---|---|---|---|---|---|",
    ]

    with (
        h5py.File(args.image_aug_h5, "r") as f_aug,
        h5py.File(args.latent_aug_h5, "r") as f_lat,
        h5py.File(args.source_image_h5, "r") as f_src,
    ):
        for variant in VARIANT_NAMES:
            rows = rows_by_variant.get(variant, [])
            if not rows:
                logger.warning("no rows for variant %s; skipping", variant)
                continue
            picks = rng.choice(rows, size=min(args.n_per_variant, len(rows)), replace=False)
            picks = sorted(int(r) for r in picks)
            logger.info(
                "variant %s: rendering %d figures (sampled rows %s)",
                variant,
                len(picks),
                picks[:6],
            )
            for fig_idx, row_idx in enumerate(picks):
                scan_id = ids_arr[row_idx]
                scan_id_str = (
                    scan_id.decode() if isinstance(scan_id, (bytes, bytearray)) else str(scan_id)
                )
                src_row = int(source_row_arr[row_idx])
                crop_origin = tuple(int(v) for v in f_src["crop/origin"][src_row])

                rows_for_fig: list[AugRoundtripRow] = []
                for slug in args.modalities:
                    src_native = np.asarray(f_src[f"images/{slug}"][src_row], dtype=np.float32)
                    src_boxed = _box_native_numpy(src_native, crop_origin)
                    src_normed = _percentile_normalise_np(src_boxed)
                    aug_img = np.asarray(f_aug[f"images/{slug}"][row_idx], dtype=np.float32)
                    aug_normed = _percentile_normalise_np(aug_img)
                    z_np = np.asarray(f_lat[f"latents/{slug}"][row_idx], dtype=np.float32)
                    decoded = _decode_one(z_np, decoder, device)
                    rows_for_fig.append(
                        AugRoundtripRow(
                            patient_id=scan_id_str,
                            cohort=args.cohort,
                            variant=variant,
                            modality=slug,
                            original=src_normed,
                            augmented=aug_normed,
                            decoded=decoded,
                        )
                    )

                fig_path = (
                    args.output_dir
                    / f"roundtrip_{args.cohort.replace('/', '_')}_{variant}_{fig_idx:02d}_{scan_id_str}.png"
                )
                render_aug_roundtrip_figure(
                    rows_for_fig,
                    fig_path,
                    title=f"{args.cohort} — {variant} — {scan_id_str}",
                )

                psnr_orig_vs_aug = []
                for r in rows_for_fig:
                    p, _ = compute_psnr_ssim(r.original, r.augmented)
                    psnr_orig_vs_aug.append(p)
                summary_rows.append(
                    f"| {args.cohort} | {variant} | {scan_id_str} | "
                    f"{psnr_orig_vs_aug[0]:.2f} | {psnr_orig_vs_aug[1]:.2f} | "
                    f"{psnr_orig_vs_aug[2]:.2f} | {psnr_orig_vs_aug[3]:.2f} |"
                )

    (args.output_dir / "summary.md").write_text("\n".join(summary_rows))
    logger.info("done — wrote figures + summary.md to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
