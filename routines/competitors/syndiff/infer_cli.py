"""CLI entrypoint for ``vena-competitor-syndiff-infer``.

Loads ``best_gen_diffusive_1.pth`` from a training run directory, synthesises
volumes for N patients of a chosen split, and dumps NIfTI + PNG + metrics CSV
+ summary JSON.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.logging import RichHandler

from vena.competitors.syndiff import run_inference


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vena-competitor-syndiff-infer",
        description=(
            "SynDiff inference — load a trained gen_diffusive_1 checkpoint and "
            "synthesise target volumes from the configured source modality."
        ),
    )
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Training run directory (contains checkpoints/).")
    parser.add_argument("--image-h5", type=Path, required=True,
                        help="VENA cohort image H5 to read source / target volumes from.")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Where to write nifti/, png/, metrics.csv, summary.json.")
    parser.add_argument("--source-modality", type=str, required=True,
                        choices=["t1pre", "t2", "flair"],
                        help="Source modality matching the training config.")
    parser.add_argument("--target-modality", type=str, default="t1c",
                        help="Target modality (default: t1c).")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--phase", type=str, default="val",
                        choices=["train", "val", "test"])
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--min-brain-voxels", type=int, default=1000)
    parser.add_argument("--max-patients", type=int, default=None,
                        help="Cap the number of patients (smoke / debug).")
    parser.add_argument("--num-timesteps", type=int, default=4,
                        help="Reverse-sampling steps (T/k from the paper; default 4).")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--which-epoch", type=str, default="best",
                        help="Which checkpoint tag to load: 'best', 'latest', or 'epoch_NNNN'.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )

    out = run_inference(
        run_dir=args.run_dir,
        image_h5=args.image_h5,
        out_dir=args.out_dir,
        source_modality=args.source_modality,
        target_modality=args.target_modality,
        fold=args.fold,
        phase=args.phase,
        image_size=args.image_size,
        min_brain_voxels=args.min_brain_voxels,
        max_patients=args.max_patients,
        num_timesteps=args.num_timesteps,
        gpu_index=args.gpu_index,
        which_epoch=args.which_epoch,
    )
    print(f"artifact: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
