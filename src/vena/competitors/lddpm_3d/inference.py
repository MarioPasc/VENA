"""3D-LDDPM inference and per-patient metrics (Eidex *et al.* 2025 §4 baseline).

Mirrors ``src/external/lddpm_3d/upstream/test_ddpm_t1_flair_final.py``
(SHA ``fc8314f6``) on the integration recipe:

    scheduler.set_timesteps(K)
    for t in scheduler.timesteps:
        unet_in = torch.cat([z_curr, z_T1pre, z_FLAIR], dim=1)
        eps_pred = unet(unet_in, t.expand(B))
        z_curr = scheduler.step(eps_pred, t, z_curr).prev_sample

After K-step denoising we decode via ``vena.common.decode.decode_box``
(VENA's canonical brain-box decode path, identical to what exhaustive-val
uses for FM model inference), then compute whole-volume PSNR/SSIM against
the percentile-normalised real T1c — VENA's metric parity rule
(``.claude/rules/model-coding-standards.md`` rule 15).

This deviates from upstream's ``minmax01`` intensity rescale: VENA's
percentile-norm matches what the encoder saw and keeps LDDPM's metrics
comparable to VENA's own benchmark numbers. Paper-reported absolute PSNR/SSIM
are anyway not directly comparable (different cohort, different VAE).
"""

from __future__ import annotations

import csv
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

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

from .dataset import LDDPM3DLatentDataset
from .runner import _build_scheduler, _build_unet

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class InferenceError(Exception):
    """Raised on missing checkpoint, bad split, or H5 issues."""


# ---------------------------------------------------------------------------
# Whole-volume metrics
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
    """Whole-volume SSIM on ``[0, 1]`` arrays via scikit-image (3D).

    Falls back to ``nan`` when scikit-image is unavailable so smokes do not
    explode on a thin env.
    """
    try:
        from skimage.metrics import structural_similarity as ssim_fn
    except ImportError:
        logger.warning("scikit-image not available; SSIM = nan")
        return float("nan")
    return float(ssim_fn(target, pred, data_range=1.0))


# ---------------------------------------------------------------------------
# Single-patient inference
# ---------------------------------------------------------------------------


def _ddpm_sample(
    unet: torch.nn.Module,
    scheduler,
    z_cond: torch.Tensor,
    z_cond2: torch.Tensor,
    nfe: int,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """Run K-step DDPM denoising starting from pure noise.

    Mirrors ``test_ddpm_t1_flair_final.py:269-300``. Returns the predicted
    target latent and the wall-clock seconds for the integration alone
    (CUDA-synced for accurate timing — matches VENA's exhaustive-val
    timing convention).
    """
    z_curr = torch.randn_like(z_cond)
    b = z_curr.shape[0]

    scheduler.set_timesteps(nfe)
    timesteps = scheduler.timesteps.to(device=device, dtype=torch.long)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for t in timesteps:
            t_b = t.expand(b)
            unet_in = torch.cat([z_curr, z_cond, z_cond2], dim=1)
            noise_pred = unet(unet_in, t_b)
            out = scheduler.step(noise_pred, t, z_curr)
            # MONAI DDPMScheduler.step may return:
            #   - a tuple `(prev_sample, pred_original_sample)`
            #   - an object with `.prev_sample`
            #   - a plain tensor
            # Mirrors upstream test_ddpm_t1_flair_final.py:290-298.
            if isinstance(out, tuple):
                z_curr = out[0]
            elif hasattr(out, "prev_sample"):
                z_curr = out.prev_sample
            elif hasattr(out, "sample"):
                z_curr = out.sample
            else:
                z_curr = out
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return z_curr, time.perf_counter() - t0


def _infer_one_patient(
    unet: torch.nn.Module,
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

    z_cond = torch.from_numpy(z_cond_np).unsqueeze(0).to(device)  # (1, C, h, w, d)
    z_cond2 = torch.from_numpy(z_cond2_np).unsqueeze(0).to(device)

    # crop_spec for decode: built from the image H5 metadata (the encoder side
    # stored it via the same path).
    crop_spec = build_crop_spec_from_h5(image_h5, pid)
    real_box = load_real_t1c_box(image_h5, pid, crop_spec).to(device)
    real_np = real_box.detach().float().cpu().numpy()

    # Percentile-norm the real T1c to [0, 1] — same contract as the encoder
    # applied. Decoded predictions live in the VAE's [0, 1] output range, so
    # the two arrays meet in the same intensity space.
    # ``percentile_normalise`` expects (B, C, H, W, D); ``real_box`` is (H, W, D).
    real_5d = real_box[None, None]  # → (1, 1, H, W, D)
    real_n = (
        percentile_normalise(real_5d, lower=0.0, upper=99.5, foreground_only=True)[0, 0]
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    rows: list[dict[str, float | str]] = []
    pred_np_last: np.ndarray | None = None
    for nfe in nfe_list:
        z_pred, gen_sec = _ddpm_sample(unet, scheduler, z_cond, z_cond2, nfe, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dec_t0 = time.perf_counter()
        pred_img = decode_box(vae_decoder, z_pred, crop_spec)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dec_sec = time.perf_counter() - dec_t0

        pred_np = pred_img.detach().float().cpu().numpy()
        pred_np_last = pred_np
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
    nib.save(
        nib.Nifti1Image(pred_np.astype(np.float32), affine), out_dir / f"{pid}_pred_t1c.nii.gz"
    )
    nib.save(
        nib.Nifti1Image(real_normalised.astype(np.float32), affine),
        out_dir / f"{pid}_real_t1c_normalised.nii.gz",
    )

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
    """Map ``--epoch`` to a concrete checkpoint file under ``run_dir``."""
    ck = run_dir / "checkpoints"
    if epoch in {"best", "latest"}:
        path = ck / f"{epoch}_net_unet.pth"
    else:
        path = ck / f"epoch_{int(epoch)}_net_unet.pth"
    if not path.is_file():
        available = sorted(p.name for p in ck.glob("*.pth"))
        raise InferenceError(f"checkpoint {path.name} not found in {ck}; available: {available}")
    return path


def _resolve_split_patients(
    latent_h5: Path,
    fold: int,
    phase: str,
    n_patients: int,
) -> list[tuple[str, int]]:
    """Replay the dataset's split resolution to get ``(pid, pidx)`` pairs."""
    ds = LDDPM3DLatentDataset(
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
    nfe_list: Sequence[int] = (200, 500, 1000),
    input_latents: Sequence[str] = ("t1pre", "flair"),
    target_latent: str = "t1c",
    num_train_timesteps: int = 1000,
    beta_start: float = 0.0015,
    beta_end: float = 0.0195,
    beta_schedule: str = "scaled_linear_beta",
    clip_sample: bool = False,
    out_dir: Path | str | None = None,
    gpu_id: int = 0,
    vae_checkpoint: Path | str | None = None,
) -> Path:
    """Synthesise T1c for ``n_patients`` val patients, write NIfTI + metrics.

    Parameters
    ----------
    run_dir : Path
        Training run directory containing ``checkpoints/``.
    image_h5 : Path
        Image-domain H5 (real T1c volumes + brain-box geometry).
    latent_h5 : Path
        Latent H5 (conditioning latents z_T1pre, z_FLAIR).
    epoch : str | int, default ``"best"``
        ``"best"`` / ``"latest"`` / integer epoch index.
    fold, phase : split selectors.
    n_patients : int, default 10.
    nfe_list : tuple of NFE counts, default ``(200, 500, 1000)`` — DDPM uses
        more steps than RFlow because its time-discretisation is non-adaptive.
    input_latents : tuple of conditioning latent names, default
        ``("t1pre", "flair")`` — paper-faithful for 3D-LDDPM (T1n + T2-FLAIR).
    num_train_timesteps, beta_start, beta_end, beta_schedule, clip_sample :
        DDPM scheduler kwargs (must match those used at training time).
    out_dir : Path or None.
        Default: ``<run_dir>/inference/epoch_<epoch>/``.
    gpu_id : int.
    vae_checkpoint : Path or None.
        Required — pass the absolute path to ``autoencoder_v2.pt`` for the
        platform. ``vena.common.load_autoencoder`` has no default.

    Returns
    -------
    Path
        The output directory containing ``metrics.csv``, ``summary.json``,
        and per-patient NIfTI/PNG triplets.
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

    device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(
        "3D-LDDPM inference | run_dir=%s ckpt=%s device=%s nfe=%s betas=(%g, %g) T=%d",
        run_dir,
        ckpt_path.name,
        device,
        list(nfe_list),
        beta_start,
        beta_end,
        num_train_timesteps,
    )

    # -- Build model + scheduler + VAE ---------------------------------------
    unet = (
        _build_unet(
            latent_channels=4,
            cond_latents=len(input_latents),
        )
        .to(device)
        .eval()
    )
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "unet_state_dict" in blob:
        state_dict = blob["unet_state_dict"]
    elif isinstance(blob, dict) and "unet" in blob:
        # upstream convention (train_ddpm.py:232).
        state_dict = blob["unet"]
    else:
        state_dict = blob
    missing, unexpected = unet.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("missing keys: %d (e.g. %s)", len(missing), missing[:3])
    if unexpected:
        logger.warning("unexpected keys: %d (e.g. %s)", len(unexpected), unexpected[:3])

    scheduler = _build_scheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule=beta_schedule,
        clip_sample=clip_sample,
    )

    # VENA's MAISI-V2 decoder via the canonical loader.
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
        len(patients),
        latent_h5.name,
        fold,
        phase,
    )

    # -- Loop over patients × NFE -------------------------------------------
    metrics_rows: list[dict[str, float | str]] = []
    for pid, pidx in patients:
        try:
            rows, pred_np, real_n, real_raw = _infer_one_patient(
                unet,
                scheduler,
                vae_decoder,
                image_h5,
                latent_h5,
                pid,
                pidx,
                input_latents,
                nfe_list,
                device,
            )
        except Exception as exc:
            logger.warning("patient %s failed: %s", pid, exc)
            continue
        metrics_rows.extend(rows)
        _save_outputs(out_dir, pid, pred_np, real_n, real_raw)
        logger.info(
            "patient %s done (last NFE row: PSNR=%.2f SSIM=%.3f)",
            pid,
            rows[-1]["psnr_db"],
            rows[-1]["ssim"],
        )

    # -- Persist results -----------------------------------------------------
    metrics_csv = out_dir / "metrics.csv"
    fields = ["patient_id", "nfe", "psnr_db", "ssim", "gen_seconds", "decode_seconds"]
    with metrics_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(metrics_rows)

    # NFE-aggregated summary.
    psnrs_by_nfe: dict[int, list[float]] = {}
    ssims_by_nfe: dict[int, list[float]] = {}
    for row in metrics_rows:
        psnrs_by_nfe.setdefault(int(row["nfe"]), []).append(float(row["psnr_db"]))
        ssims_by_nfe.setdefault(int(row["nfe"]), []).append(float(row["ssim"]))

    summary = {
        "schema_version": "1.0",
        "produced_at": datetime.now(UTC).isoformat(),
        "producer": "vena.competitors.lddpm_3d.inference",
        "run_dir": str(run_dir),
        "checkpoint": ckpt_path.name,
        "image_h5": str(image_h5),
        "latent_h5": str(latent_h5),
        "fold": fold,
        "phase": phase,
        "n_patients_requested": n_patients,
        "n_patients_succeeded": len({r["patient_id"] for r in metrics_rows}),
        "nfe_list": list(nfe_list),
        "scheduler": {
            "kind": "DDPMScheduler",
            "num_train_timesteps": num_train_timesteps,
            "beta_start": beta_start,
            "beta_end": beta_end,
            "schedule": beta_schedule,
            "clip_sample": clip_sample,
        },
        "metrics_by_nfe": {
            str(nfe): {
                "psnr_db_mean": float(np.mean(psnrs_by_nfe.get(nfe, [float("nan")]))),
                "psnr_db_std": float(np.std(psnrs_by_nfe.get(nfe, [float("nan")]))),
                "ssim_mean": float(np.nanmean(ssims_by_nfe.get(nfe, [float("nan")]))),
                "ssim_std": float(np.nanstd(ssims_by_nfe.get(nfe, [float("nan")]))),
            }
            for nfe in nfe_list
        },
        "competitor": {
            "name": "lddpm_3d",
            "paper": "Ho et al. 2020 (DDPM) + Eidex et al. 2025 §4 baseline (arXiv:2509.24194)",
            "doi": "arXiv:2006.11239 (DDPM); arXiv:2509.24194 (baseline)",
        },
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info("inference complete: wrote %d rows to %s", len(metrics_rows), metrics_csv)
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
        nfe_list=getattr(args, "nfe_list", (200, 500, 1000)),
        input_latents=tuple(getattr(args, "input_latents", ("t1pre", "flair"))),
        target_latent=getattr(args, "target_latent", "t1c"),
        num_train_timesteps=getattr(args, "num_train_timesteps", 1000),
        beta_start=getattr(args, "beta_start", 0.0015),
        beta_end=getattr(args, "beta_end", 0.0195),
        beta_schedule=getattr(args, "beta_schedule", "scaled_linear_beta"),
        clip_sample=getattr(args, "clip_sample", False),
        out_dir=getattr(args, "out_dir", None),
        gpu_id=getattr(args, "gpu_id", 0),
        vae_checkpoint=getattr(args, "vae_checkpoint", None),
    )
