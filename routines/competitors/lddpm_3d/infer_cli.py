"""CLI entrypoint for ``vena-competitor-lddpm-3d-infer``.

Usage::

    vena-competitor-lddpm-3d-infer \\
        --run-dir   /path/to/<run_id> \\
        --image-h5  /path/to/UCSFPDGM_image.h5 \\
        --latent-h5 /path/to/UCSFPDGM_latents.h5 \\
        --vae-checkpoint /path/to/autoencoder_v2.pt \\
        --epoch best --n-patients 10 --phase val \\
        --nfe 200 --nfe 500 --nfe 1000

Citation: Ho et al. 2020 (DDPM) + Eidex et al. 2025 §4 (3D + MAISI latents).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.logging import RichHandler

from vena.competitors.lddpm_3d.inference import run_inference


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vena-competitor-lddpm-3d-infer",
        description=(
            "Run 3D-LDDPM (Ho et al. 2020 + Eidex et al. 2025 §4) inference "
            "on N patients of a chosen split. Latents drive a K-step DDPM "
            "denoising loop; the frozen MAISI-V2 VAE decodes to image space; "
            "metrics are computed under VENA's percentile-norm parity "
            "contract."
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="trained run directory under experiments/competitors/lddpm_3d/",
    )
    parser.add_argument(
        "--image-h5",
        type=Path,
        required=True,
        help="image-domain H5 with the real T1c volumes",
    )
    parser.add_argument(
        "--latent-h5",
        type=Path,
        required=True,
        help="latent H5 with conditioning latents (z_T1pre, z_FLAIR)",
    )
    parser.add_argument(
        "--epoch",
        type=str,
        default="best",
        help="checkpoint epoch (default: best)",
    )
    parser.add_argument("--n-patients", type=int, default=10)
    parser.add_argument(
        "--phase",
        type=str,
        default="val",
        choices=["train", "val", "test"],
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--nfe",
        type=int,
        action="append",
        default=None,
        help="NFE per sample (repeat for a panel); default: 200, 500, 1000",
    )
    parser.add_argument(
        "--num-train-timesteps",
        type=int,
        default=1000,
        help="DDPM training timesteps (must match training-time value)",
    )
    parser.add_argument(
        "--beta-start",
        type=float,
        default=0.0015,
        help="DDPM beta_start (default 0.0015 = training-time value; "
        "upstream test_ddpm uses 0.0005 — likely a typo)",
    )
    parser.add_argument(
        "--beta-end",
        type=float,
        default=0.0195,
        help="DDPM beta_end (default 0.0195 — matches training)",
    )
    parser.add_argument(
        "--beta-schedule",
        type=str,
        default="scaled_linear_beta",
        help="DDPM beta schedule kind (default 'scaled_linear_beta')",
    )
    parser.add_argument(
        "--clip-sample",
        action="store_true",
        help="set DDPMScheduler clip_sample=True (default False — matches upstream)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: <run_dir>/inference/epoch_<epoch>/",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--vae-checkpoint",
        type=Path,
        required=True,
        help="MAISI-V2 autoencoder_v2.pt path (see src/external/LINKS.md)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )

    nfe_list = tuple(args.nfe) if args.nfe else (200, 500, 1000)

    out_dir = run_inference(
        run_dir=args.run_dir,
        image_h5=args.image_h5,
        latent_h5=args.latent_h5,
        epoch=args.epoch,
        fold=args.fold,
        phase=args.phase,
        n_patients=args.n_patients,
        nfe_list=nfe_list,
        num_train_timesteps=args.num_train_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        clip_sample=args.clip_sample,
        out_dir=args.out_dir,
        gpu_id=args.gpu_id,
        vae_checkpoint=args.vae_checkpoint,
    )
    print(f"inference: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
