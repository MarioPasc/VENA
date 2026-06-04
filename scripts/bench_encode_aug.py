"""Empirical benchmark of MAISI VAE encode throughput.

Measures wall-clock seconds per modality-volume on the server's GPU. Used to
choose between (a) offline pre-encoded augmentation bank and (b) on-the-fly
encode during training, for the augmentation proposal.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from vena.model.autoencoder.maisi.encode import MaisiEncoder
from vena.model.autoencoder.maisi.loader import load_autoencoder
from vena.model.autoencoder.maisi.preprocessing import CropPadSpec

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")


def centered_crop_spec(native: tuple[int, int, int], target: tuple[int, int, int]) -> CropPadSpec:
    """Centre the target box on the native volume (negative origin → pad)."""
    origin = tuple((n - t) // 2 for n, t in zip(native, target))
    return CropPadSpec(crop_origin=origin, native_shape=native, target_shape=target)


def load_volume(image_h5: Path, idx: int, modality: str) -> torch.Tensor:
    with h5py.File(image_h5, "r") as f:
        arr = f[f"images/{modality}"][idx, ...]
    return torch.from_numpy(np.asarray(arr, dtype=np.float32))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-h5", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--modalities", nargs="+", default=["t1pre", "t1c", "t2", "flair"])
    ap.add_argument("--n-patients", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--precision-mode", default="autocast", choices=["autocast", "fp32"])
    ap.add_argument("--inference-mode", default="full", choices=["full", "sliding"])
    ap.add_argument("--target-shape", nargs=3, type=int, default=[240, 240, 160])
    ap.add_argument("--depth-pad-base", type=int, default=8)
    ap.add_argument("--norm-float16", action="store_true", default=False)
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")

    handle = load_autoencoder(
        Path(args.checkpoint),
        device=device,
        arch_overrides={"norm_float16": args.norm_float16},
    )
    encoder = MaisiEncoder(
        handle,
        depth_pad_base=args.depth_pad_base,
        percentile_lower=0.0,
        percentile_upper=99.5,
        percentile_foreground_only=True,
        precision_mode=args.precision_mode,
    )

    image_h5 = Path(args.image_h5)
    with h5py.File(image_h5, "r") as f:
        first = args.modalities[0]
        n_total = int(f[f"images/{first}"].shape[0])
    indices = list(range(min(args.n_patients, n_total)))
    target = tuple(args.target_shape)

    v = load_volume(image_h5, indices[0], args.modalities[0])
    spec = centered_crop_spec(tuple(int(s) for s in v.shape), target)
    x = v[None, None].to(device)
    with torch.inference_mode():
        encoder.encode(x, mode=args.inference_mode, crop_spec=spec, normalise=True)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times: dict[str, list[float]] = {m: [] for m in args.modalities}
    latent_shape: tuple[int, ...] | None = None

    for idx in indices:
        for m in args.modalities:
            v = load_volume(image_h5, idx, m)
            spec = centered_crop_spec(tuple(int(s) for s in v.shape), target)
            x = v[None, None].to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                res = encoder.encode(x, mode=args.inference_mode, crop_spec=spec, normalise=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times[m].append(time.perf_counter() - t0)
            if latent_shape is None:
                latent_shape = tuple(int(s) for s in res.latent.shape)

    fp16_bytes_per_latent = int(np.prod(latent_shape) * 2) if latent_shape else None  # fp16

    out = {
        "device": str(device),
        "precision_mode": args.precision_mode,
        "norm_float16": args.norm_float16,
        "inference_mode": args.inference_mode,
        "target_shape": list(target),
        "latent_shape": list(latent_shape) if latent_shape else None,
        "fp16_bytes_per_latent": fp16_bytes_per_latent,
        "n_patients": len(indices),
        "modalities": args.modalities,
        "per_modality_sec": {
            m: {
                "mean": float(np.mean(times[m])),
                "median": float(np.median(times[m])),
                "std": float(np.std(times[m])),
                "min": float(np.min(times[m])),
                "max": float(np.max(times[m])),
            }
            for m in args.modalities
        },
        "all_modalities_per_scan_sec": {
            "mean": float(sum(np.mean(times[m]) for m in args.modalities)),
            "median": float(sum(np.median(times[m]) for m in args.modalities)),
        },
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
