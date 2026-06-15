"""3D-DiT inference + per-patient metrics.

Mirrors ``vena.competitors.t1c_rflow.inference`` end-to-end (Euler integration
over ``t: 1 → 0`` using the RFlow scheduler; decode via
``vena.common.decode.decode_box``; metrics under VENA's percentile-norm
parity contract). The only differences are:

1. Model: ``DiT3DWrapper`` rebuilt from the ``arch_meta`` block stored
   alongside the state dict in the checkpoint (so we never need to consult
   the training YAML at inference time).
2. Forward signature: ``dit(x, t, y=None)`` — the DiT accepts an optional
   class label which we never pass.

Citation: see ``src/vena/competitors/dit_3d/__init__.py``.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import h5py
import nibabel as nib
import numpy as np
import torch

from vena.common import MaisiDecoder, load_autoencoder, percentile_normalise
from vena.common.decode import decode_box
from vena.model.fm.eval.exhaustive import (
    build_crop_spec_from_h5,
    load_real_t1c_box,
)

from .dataset import DiT3DLatentDataset
from .runner import _UPSTREAM_DIR, _build_scheduler

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class InferenceError(Exception):
    """Raised on missing checkpoint, bad split, or H5 issues."""


# ---------------------------------------------------------------------------
# Whole-volume metrics — identical to t1c_rflow.inference (intentional)
# ---------------------------------------------------------------------------

def _psnr(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Whole-volume PSNR on ``[0, 1]`` arrays; optional brain mask."""
    if mask is not None:
        diff = (pred - target)[mask > 0]
    else:
        diff = (pred - target).ravel()
    mse = float(np.mean(diff * diff))
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def _ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """Whole-volume SSIM on ``[0, 1]`` arrays via scikit-image (3D)."""
    try:
        from skimage.metrics import structural_similarity as ssim_fn
    except ImportError:
        logger.warning("scikit-image not available; SSIM = nan")
        return float("nan")
    return float(ssim_fn(target, pred, data_range=1.0))


# ---------------------------------------------------------------------------
# DiT model rebuild
# ---------------------------------------------------------------------------

def _import_dit_wrapper() -> Any:
    if not _UPSTREAM_DIR.is_dir():
        raise InferenceError(
            f"vendored upstream missing at {_UPSTREAM_DIR}; "
            "re-vendor per src/external/dit_3d/UPSTREAM.md"
        )
    path_str = str(_UPSTREAM_DIR)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    from dit3d_wrapper import DiT3DWrapper  # type: ignore[import-not-found]

    return DiT3DWrapper


def _rebuild_dit_from_meta(arch_meta: dict[str, Any]) -> Any:
    DiT3DWrapper = _import_dit_wrapper()
    return DiT3DWrapper(
        in_channels=arch_meta["in_channels"],
        out_channels=arch_meta["out_channels"],
        input_size=tuple(arch_meta["input_size"]),
        patch_size=arch_meta["patch_size"],
        hidden_size=arch_meta["hidden_size"],
        depth=arch_meta["depth"],
        num_heads=arch_meta["num_heads"],
        mlp_ratio=arch_meta["mlp_ratio"],
        class_dropout_prob=0.0,
        num_classes=1,
        learn_sigma=False,
    )


# ---------------------------------------------------------------------------
# Single-patient inference
# ---------------------------------------------------------------------------

def _euler_sample(
    dit: torch.nn.Module,
    scheduler,
    z_cond: torch.Tensor,
    z_cond2: torch.Tensor,
    nfe: int,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """Euler-integrate ``t: 1 → 0`` over ``nfe`` steps."""
    z_curr = torch.randn_like(z_cond)
    b = z_curr.shape[0]
    num_vox = int(np.prod(z_curr.shape[-3:]))

    scheduler.set_timesteps(num_inference_steps=nfe, input_img_size_numel=num_vox)
    t = scheduler.timesteps.to(device)
    t_next = torch.cat((t[1:], t.new_tensor([0])))

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for ts, tsn in zip(t, t_next):
            ts_scalar = float(ts.item())
            tsn_scalar = float(tsn.item())
            ts_b = ts.expand(b)
            dit_in = torch.cat([z_curr, z_cond, z_cond2], dim=1)
            # DiT forward — y=None for unconditional class label.
            vel = dit(dit_in, ts_b, y=None) \
                if "y" in dit.forward.__code__.co_varnames \
                else dit(dit_in, ts_b)
            z_curr, _ = scheduler.step(vel, ts_scalar, z_curr, tsn_scalar)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return z_curr, time.perf_counter() - t0


def _infer_one_patient(
    dit: torch.nn.Module,
    scheduler,
    vae_decoder,
    image_h5: Path,
    latent_h5: Path,
    pid: str,
    pidx: int,
    input_latents: Sequence[str],
    nfe_list: Sequence[int],
    device: torch.device,
) -> tuple[list[dict[str, float | str]], np.ndarray, np.ndarray, np.ndarray]:
    """Run inference at each NFE in ``nfe_list`` and return one CSV row per NFE."""
    with h5py.File(latent_h5, "r") as f:
        z_cond_np = np.asarray(f[f"latents/{input_latents[0]}"][pidx], dtype=np.float32)
        z_cond2_np = np.asarray(f[f"latents/{input_latents[1]}"][pidx], dtype=np.float32)

    z_cond = torch.from_numpy(z_cond_np).unsqueeze(0).to(device)
    z_cond2 = torch.from_numpy(z_cond2_np).unsqueeze(0).to(device)

    crop_spec = build_crop_spec_from_h5(image_h5, pid)
    real_box = load_real_t1c_box(image_h5, pid, crop_spec).to(device)
    real_np = real_box.detach().float().cpu().numpy()

    # percentile_normalise expects 5-D (B, C, H, W, D); wrap then unwrap.
    real_5d = real_box[None, None]
    real_n = percentile_normalise(
        real_5d, lower=0.0, upper=99.5, foreground_only=True
    )[0, 0].detach().float().cpu().numpy()

    rows: list[dict[str, float | str]] = []
    pred_np_last: np.ndarray | None = None
    for nfe in nfe_list:
        z_pred, gen_sec = _euler_sample(
            dit, scheduler, z_cond, z_cond2, nfe, device
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dec_t0 = time.perf_counter()
        pred_img = decode_box(vae_decoder, z_pred, crop_spec)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dec_sec = time.perf_counter() - dec_t0

        pred_np = pred_img.detach().float().cpu().numpy()
        psnr = _psnr(pred_np, real_n)
        ssim = _ssim(pred_np, real_n)
        rows.append(
            {
                "patient_id": pid,
                "nfe": int(nfe),
                "psnr_db": psnr,
                "ssim": ssim,
                "gen_seconds": gen_sec,
                "decode_seconds": dec_sec,
            }
        )
        pred_np_last = pred_np
    assert pred_np_last is not None  # nfe_list is non-empty by contract
    return rows, pred_np_last, real_n, real_np


def _save_outputs(
    out_dir: Path,
    pid: str,
    pred_np: np.ndarray,
    real_normalised: np.ndarray,
    real_raw: np.ndarray,
) -> None:
    """Write per-patient NIfTI + PNG midslice triplet."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    affine = np.eye(4, dtype=np.float32)
    nib.save(nib.Nifti1Image(pred_np.astype(np.float32), affine),
             out_dir / f"{pid}_pred_t1c.nii.gz")
    nib.save(nib.Nifti1Image(real_normalised.astype(np.float32), affine),
             out_dir / f"{pid}_real_t1c_normalised.nii.gz")

    z_mid = pred_np.shape[-1] // 2
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(real_raw[..., z_mid], cmap="gray")
    axes[0].set_title("real (raw)")
    axes[0].axis("off")
    axes[1].imshow(real_normalised[..., z_mid], cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("real (percentile-norm)")
    axes[1].axis("off")
    axes[2].imshow(pred_np[..., z_mid], cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("pred T1c")
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / f"{pid}_midslice.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _resolve_checkpoint(run_dir: Path, epoch: str | int) -> Path:
    ck = run_dir / "checkpoints"
    if epoch in {"best", "latest"}:
        path = ck / f"{epoch}_net_dit.pth"
    else:
        path = ck / f"epoch_{int(epoch)}_net_dit.pth"
    if not path.is_file():
        available = sorted(p.name for p in ck.glob("*.pth"))
        raise InferenceError(
            f"checkpoint {path.name} not found in {ck}; "
            f"available: {available}"
        )
    return path


def _resolve_split_patients(
    latent_h5: Path,
    fold: int,
    phase: str,
    n_patients: int,
) -> list[tuple[str, int]]:
    ds = DiT3DLatentDataset(
        latent_h5=latent_h5,
        fold=fold,
        phase=phase,
        max_patients=n_patients,
    )
    return list(zip(ds.patient_ids, ds.patient_indices))


def run_inference(
    run_dir: Path | str,
    image_h5: Path | str,
    latent_h5: Path | str,
    *,
    epoch: str | int = "best",
    fold: int = 0,
    phase: str = "val",
    n_patients: int = 10,
    nfe_list: Sequence[int] = (50, 100, 200),
    input_latents: Sequence[str] = ("t1pre", "flair"),
    target_latent: str = "t1c",
    out_dir: Path | str | None = None,
    gpu_id: int = 0,
    vae_checkpoint: Path | str | None = None,
) -> Path:
    """Synthesise T1c for ``n_patients`` patients, write NIfTI + metrics.

    Unlike T1C-RFlow's ``run_inference``, the DiT architecture is recovered
    from the checkpoint's ``arch_meta`` block — no separate
    ``unet_arch_config`` JSON is needed.
    """
    run_dir = Path(run_dir)
    image_h5 = Path(image_h5)
    latent_h5 = Path(latent_h5)

    if not image_h5.is_file():
        raise InferenceError(f"image H5 not found at {image_h5}")
    if not latent_h5.is_file():
        raise InferenceError(f"latent H5 not found at {latent_h5}")

    ckpt_path = _resolve_checkpoint(run_dir, epoch)

    if out_dir is None:
        out_dir = run_dir / "inference" / f"epoch_{epoch}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = (
        torch.device(f"cuda:{gpu_id}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    logger.info(
        "3D-DiT inference | run_dir=%s ckpt=%s device=%s nfe=%s",
        run_dir, ckpt_path.name, device, list(nfe_list),
    )

    # -- Rebuild model from checkpoint metadata ------------------------------
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or "arch_meta" not in blob:
        raise InferenceError(
            f"checkpoint {ckpt_path} missing 'arch_meta' block — was it written "
            "by an older runner? Re-train or hand-author the arch_meta dict."
        )
    arch_meta = blob["arch_meta"]
    dit = _rebuild_dit_from_meta(arch_meta).to(device).eval()
    state_dict = blob.get("dit_state_dict", blob)
    missing, unexpected = dit.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("missing keys: %d (e.g. %s)", len(missing), missing[:3])
    if unexpected:
        logger.warning("unexpected keys: %d (e.g. %s)", len(unexpected), unexpected[:3])

    scheduler = _build_scheduler(num_train_timesteps=1000)

    if vae_checkpoint is None:
        raise InferenceError(
            "vae_checkpoint is required — pass --vae-checkpoint pointing at "
            "autoencoder_v2.pt for your platform (see src/external/LINKS.md)."
        )
    ae = load_autoencoder(vae_checkpoint, device=device)
    vae_decoder = MaisiDecoder(ae)

    # -- Resolve patients ----------------------------------------------------
    patients = _resolve_split_patients(latent_h5, fold, phase, n_patients)
    logger.info(
        "inference on %d patients from latent_h5=%s fold=%d phase=%s",
        len(patients), latent_h5.name, fold, phase,
    )

    metrics_rows: list[dict[str, float | str]] = []
    for pid, pidx in patients:
        try:
            rows, pred_np, real_n, real_raw = _infer_one_patient(
                dit, scheduler, vae_decoder,
                image_h5, latent_h5, pid, pidx,
                input_latents, nfe_list, device,
            )
        except Exception as exc:
            logger.warning("patient %s failed: %s", pid, exc)
            continue
        metrics_rows.extend(rows)
        _save_outputs(out_dir, pid, pred_np, real_n, real_raw)
        logger.info(
            "patient %s done (last NFE row: PSNR=%.2f SSIM=%.3f)",
            pid, rows[-1]["psnr_db"], rows[-1]["ssim"],
        )

    # -- Persist results -----------------------------------------------------
    metrics_csv = out_dir / "metrics.csv"
    fields = [
        "patient_id", "nfe", "psnr_db", "ssim", "gen_seconds", "decode_seconds"
    ]
    with metrics_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(metrics_rows)

    psnrs_by_nfe: dict[int, list[float]] = {}
    ssims_by_nfe: dict[int, list[float]] = {}
    for row in metrics_rows:
        psnrs_by_nfe.setdefault(int(row["nfe"]), []).append(float(row["psnr_db"]))
        ssims_by_nfe.setdefault(int(row["nfe"]), []).append(float(row["ssim"]))

    summary = {
        "schema_version": "1.0",
        "produced_at": datetime.now(UTC).isoformat(),
        "producer": "vena.competitors.dit_3d.inference",
        "run_dir": str(run_dir),
        "checkpoint": ckpt_path.name,
        "image_h5": str(image_h5),
        "latent_h5": str(latent_h5),
        "fold": fold,
        "phase": phase,
        "n_patients_requested": n_patients,
        "n_patients_succeeded": len({r["patient_id"] for r in metrics_rows}),
        "nfe_list": list(nfe_list),
        "metrics_by_nfe": {
            str(nfe): {
                "psnr_db_mean": float(np.mean(psnrs_by_nfe.get(nfe, [float("nan")]))),
                "psnr_db_std": float(np.std(psnrs_by_nfe.get(nfe, [float("nan")]))),
                "ssim_mean": float(np.nanmean(ssims_by_nfe.get(nfe, [float("nan")]))),
                "ssim_std": float(np.nanstd(ssims_by_nfe.get(nfe, [float("nan")]))),
            }
            for nfe in nfe_list
        },
        "arch_meta": arch_meta,
        "competitor": {
            "name": "dit_3d",
            "paper": "Peebles & Xie 2023 (DiT) + Eidex et al. 2025 §4 (3D adaptation)",
            "doi": "arXiv:2212.09748; arXiv:2509.24194",
        },
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info(
        "inference complete: wrote %d rows to %s", len(metrics_rows), metrics_csv
    )
    return out_dir


def run_inference_from_args(args: SimpleNamespace) -> Path:
    """SimpleNamespace adapter for the engine layer."""
    return run_inference(
        run_dir=args.run_dir,
        image_h5=args.image_h5,
        latent_h5=args.latent_h5,
        epoch=getattr(args, "epoch", "best"),
        fold=getattr(args, "fold", 0),
        phase=getattr(args, "phase", "val"),
        n_patients=getattr(args, "n_patients", 10),
        nfe_list=getattr(args, "nfe_list", (50, 100, 200)),
        input_latents=tuple(getattr(args, "input_latents", ("t1pre", "flair"))),
        target_latent=getattr(args, "target_latent", "t1c"),
        out_dir=getattr(args, "out_dir", None),
        gpu_id=getattr(args, "gpu_id", 0),
        vae_checkpoint=getattr(args, "vae_checkpoint", None),
    )
