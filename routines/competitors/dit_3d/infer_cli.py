"""CLI entrypoint for ``vena-competitor-dit-3d-infer``.

Usage::

    vena-competitor-dit-3d-infer \\
        --run-dir   /path/to/<run_id> \\
        --image-h5  /path/to/UCSFPDGM_image.h5 \\
        --latent-h5 /path/to/UCSFPDGM_latents.h5 \\
        --vae-checkpoint /path/to/autoencoder_v2.pt \\
        --epoch best --n-patients 10 --phase val \\
        --nfe 50 --nfe 100 --nfe 200

The DiT architecture is reconstructed from the checkpoint's ``arch_meta``
block — no separate ``--unet-arch-config`` flag is needed (unlike the
T1C-RFlow infer CLI).

Citation: Peebles & Xie 2023 + Eidex et al. 2025 §4.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.logging import RichHandler

from vena.competitors.dit_3d.inference import run_inference


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vena-competitor-dit-3d-infer",
        description=(
            "Run 3D-DiT (Peebles & Xie 2023 + Eidex et al. 2025 §4) inference "
            "on N patients of a chosen split. Latents drive Euler integration; "
            "the frozen MAISI-V2 VAE decodes to image space; metrics are "
            "computed under VENA's percentile-norm parity contract."
        ),
    )
    parser.add_argument(
        "--run-dir", type=Path, required=True,
        help="trained run directory under experiments/competitors/dit_3d/",
    )
    parser.add_argument(
        "--image-h5", type=Path, required=True,
        help="image-domain H5 with the real T1c volumes",
    )
    parser.add_argument(
        "--latent-h5", type=Path, required=True,
        help="latent H5 with conditioning latents (z_T1pre, z_FLAIR)",
    )
    parser.add_argument(
        "--epoch", type=str, default="best",
        help="checkpoint epoch (default: best)",
    )
    parser.add_argument("--n-patients", type=int, default=10)
    parser.add_argument(
        "--phase", type=str, default="val",
        choices=["train", "val", "test"],
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--nfe", type=int, action="append", default=None,
        help="NFE per sample (repeat for a panel); default: 50, 100, 200",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="default: <run_dir>/inference/epoch_<epoch>/",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--vae-checkpoint", type=Path, required=True,
        help="MAISI-V2 autoencoder_v2.pt path (see src/external/LINKS.md)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )

    nfe_list = tuple(args.nfe) if args.nfe else (50, 100, 200)

    out_dir = run_inference(
        run_dir=args.run_dir,
        image_h5=args.image_h5,
        latent_h5=args.latent_h5,
        epoch=args.epoch,
        fold=args.fold,
        phase=args.phase,
        n_patients=args.n_patients,
        nfe_list=nfe_list,
        out_dir=args.out_dir,
        gpu_id=args.gpu_id,
        vae_checkpoint=args.vae_checkpoint,
    )
    print(f"inference: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
