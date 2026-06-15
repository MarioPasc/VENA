"""CLI entrypoint for ``vena-competitor-resvit-infer``.

Usage:

    vena-competitor-resvit-infer \\
        --run-dir /path/to/<run_id> \\
        --image-h5 /path/to/UCSFPDGM_image.h5 \\
        --epoch best --n-patients 10 --phase val
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.logging import RichHandler

from vena.competitors.resvit.inference import run_inference


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vena-competitor-resvit-infer",
        description="Run ResViT inference on N patients of a chosen split.",
    )
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="trained run directory under experiments/competitors/resvit/")
    parser.add_argument("--image-h5", type=Path, required=True,
                        help="image H5 path (UCSF-PDGM-schema)")
    parser.add_argument("--epoch", type=str, default="best",
                        help="checkpoint epoch (default: best). 'latest' selects stage-2's last "
                             "epoch; 'latest_pretrain' selects stage-1's last epoch.")
    parser.add_argument("--n-patients", type=int, default=10)
    parser.add_argument("--phase", type=str, default="val",
                        choices=["train", "val", "test"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="default: <run_dir>/inference/epoch_<epoch>/")
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )

    out_dir = run_inference(
        run_dir=args.run_dir,
        epoch=args.epoch,
        image_h5=args.image_h5,
        fold=args.fold,
        phase=args.phase,
        n_patients=args.n_patients,
        out_dir=args.out_dir,
        gpu_id=args.gpu_id,
    )
    print(f"inference: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
