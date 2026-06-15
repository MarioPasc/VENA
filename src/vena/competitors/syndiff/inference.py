"""Inference for a trained SynDiff model — synthesise the target volume from
the source modality on N patients of a chosen split.

For each patient:

1. Load every axial slice with brain coverage above ``min_brain_voxels``.
2. Per-patient percentile-normalise the source modality to ``[0, 1]``; pad
   to ``image_size`` (256); rescale to ``[-1, 1]``.
3. Run ``sample_from_model`` for ``num_timesteps`` reverse steps using
   ``best_gen_diffusive_1`` — concretely, start from
   ``x_init = cat([randn, source], dim=1)`` and iterate four (T/k) reverse
   steps. Output is a 2D prediction of the target's ``x_0`` per slice.
4. Crop back to native ``(H, W)`` and stack along ``z`` to form a 3D
   prediction volume.
5. Compute whole-volume PSNR/SSIM against the real target (also
   percentile-normalised to ``[0, 1]``).
6. Write NIfTI + 3-panel mid-slice PNG + ``metrics.csv`` + ``summary.json``.

We deliberately mirror the runner's normalisation pipeline so the generator
sees the same intensity distribution it was optimised against.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import h5py
import nibabel as nib
import numpy as np
import torch

from .dataset import _decode_ids, _pad_to, _percentile_thresholds_per_patient
from .runner import _PosteriorCoefficients, _import_upstream, _sample_posterior

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


class InferenceError(Exception):
    """Raised when inference cannot proceed (bad checkpoint, missing data)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _crop_to(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Reverse the centred zero-pad applied at training time."""
    H, W = x.shape[-2], x.shape[-1]
    top = (H - h) // 2
    left = (W - w) // 2
    return x[..., top : top + h, left : left + w]


def _psnr(pred: np.ndarray, real: np.ndarray, mask: np.ndarray) -> float:
    """Peak SNR over masked voxels in ``[0, 1]``."""
    diff = (pred - real) ** 2
    mse = float(diff[mask].mean()) if mask.any() else float("nan")
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def _ssim_3d(pred: np.ndarray, real: np.ndarray) -> float:
    """Whole-volume SSIM via skimage."""
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        return float("nan")
    return float(ssim(real, pred, data_range=1.0))


def _build_inference_args(cfg) -> SimpleNamespace:
    """The Namespace that NCSNpp expects — mirrors runner._build_args."""
    return SimpleNamespace(
        not_use_tanh=False,
        centered=True,
        z_emb_dim=cfg.z_emb_dim,
        num_channels_dae=cfg.num_channels_dae,
        ch_mult=list(cfg.ch_mult),
        num_res_blocks=cfg.num_res_blocks,
        attn_resolutions=tuple(cfg.attn_resolutions),
        dropout=cfg.dropout,
        resamp_with_conv=True,
        image_size=cfg.image_size,
        conditional=True,
        fir=True,
        fir_kernel=[1, 3, 3, 1],
        skip_rescale=True,
        resblock_type="biggan",
        progressive="none",
        progressive_input="residual",
        progressive_combine="sum",
        embedding_type=cfg.embedding_type,
        fourier_scale=16.0,
        num_channels=2,
        nz=cfg.nz,
        n_mlp=cfg.n_mlp,
        t_emb_dim=cfg.t_emb_dim,
        ngf=cfg.ngf,
        num_timesteps=cfg.num_timesteps,
        beta_min=cfg.beta_min,
        beta_max=cfg.beta_max,
    )


def _load_generator(checkpoint: Path, args, device: torch.device) -> torch.nn.Module:
    """Reconstruct ``gen_diffusive_1`` and load trained weights."""
    if not checkpoint.is_file():
        raise InferenceError(f"checkpoint not found at {checkpoint}")
    upstream = _import_upstream()
    gen = upstream.NCSNpp(args).to(device)
    state = torch.load(checkpoint, map_location=device)
    # Strip leading "module." if the state-dict was saved from DataParallel.
    if all(k.startswith("module.") for k in state.keys()):
        state = {k[len("module.") :]: v for k, v in state.items()}
    gen.load_state_dict(state, strict=True)
    gen.eval()
    return gen


def _sample_from_model(gen: torch.nn.Module, pos_coeff: _PosteriorCoefficients,
                       num_timesteps: int, x_init: torch.Tensor, nz: int,
                       device: torch.device) -> torch.Tensor:
    """Reverse sampling — mirrors train.sample_from_model with our local coeffs.

    ``x_init`` is ``cat([x_T, source], dim=1)``: channel 0 starts as randn,
    channel 1 holds the source slice (held constant through all reverse
    steps). Returns the predicted target in ``[-1, 1]`` shape ``(B, 1, H, W)``.
    """
    x = x_init[:, [0], :]
    source = x_init[:, [1], :]
    with torch.no_grad():
        for i in reversed(range(num_timesteps)):
            t = torch.full((x.size(0),), i, dtype=torch.int64, device=x.device)
            latent_z = torch.randn(x.size(0), nz, device=x.device)
            x_0 = gen(torch.cat((x, source), dim=1), t, latent_z)
            x = _sample_posterior(pos_coeff, x_0[:, [0], :], x, t).detach()
    return x


def _patient_native_volume(h5_path: Path, pidx: int, mod: str) -> np.ndarray:
    """Read a full patient volume for one modality at native shape ``(H, W, D)``."""
    with h5py.File(h5_path, "r") as f:
        return np.asarray(f[f"images/{mod}"][pidx], dtype=np.float32)


def _resolve_split(h5_path: Path, fold: int, phase: str) -> tuple[list[str], list[int]]:
    """Return (ids, indices) for the requested split, applying longitudinal prefix-match."""
    with h5py.File(h5_path, "r") as f:
        all_ids = _decode_ids(np.asarray(f["ids"]))
        if phase == "test":
            candidates = ["splits/test"]
        else:
            candidates = [f"splits/cv/fold_{fold}/{phase}", f"splits/{phase}"]
        key = next((c for c in candidates if c in f), None)
        if key is None:
            raise InferenceError(f"none of {candidates} present in {h5_path}")
        split_ids = _decode_ids(np.asarray(f[key]))
    id_to_idx = {pid: i for i, pid in enumerate(all_ids)}
    resolved_ids: list[str] = []
    resolved_indices: list[int] = []
    for pid in split_ids:
        if pid in id_to_idx:
            resolved_ids.append(pid)
            resolved_indices.append(id_to_idx[pid])
            continue
        prefix_dash = f"{pid}-"
        prefix_uscr = f"{pid}_"
        for full_id, idx in id_to_idx.items():
            if full_id.startswith(prefix_dash) or full_id.startswith(prefix_uscr):
                resolved_ids.append(full_id)
                resolved_indices.append(idx)
    return resolved_ids, resolved_indices


# ---------------------------------------------------------------------------
# Main inference entrypoint
# ---------------------------------------------------------------------------
def run_inference(
    run_dir: Path | str,
    image_h5: Path | str,
    out_dir: Path | str,
    source_modality: str,
    target_modality: str = "t1c",
    *,
    fold: int = 0,
    phase: str = "val",
    image_size: int = 256,
    min_brain_voxels: int = 1000,
    max_patients: int | None = None,
    num_timesteps: int = 4,
    beta_min: float = 0.1,
    beta_max: float = 20.0,
    nz: int = 100,
    z_emb_dim: int = 256,
    t_emb_dim: int = 256,
    num_channels_dae: int = 64,
    ch_mult: list[int] = (1, 1, 2, 2, 4, 4),
    num_res_blocks: int = 2,
    attn_resolutions: list[int] = (16,),
    n_mlp: int = 3,
    embedding_type: str = "positional",
    dropout: float = 0.0,
    ngf: int = 64,
    gpu_index: int = 0,
    which_epoch: str = "best",
) -> Path:
    """Run SynDiff inference end-to-end. Returns ``out_dir``.

    ``which_epoch``: ``"best"`` (default), ``"latest"``, or ``"epoch_NNNN"``.
    """
    run_dir = Path(run_dir)
    image_h5 = Path(image_h5)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(out_dir / "infer.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s :: %(message)s"))
    logging.getLogger().addHandler(fh)

    device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(gpu_index)
    logger.info("SynDiff inference — run_dir=%s device=%s", run_dir, device)

    args = SimpleNamespace(
        z_emb_dim=z_emb_dim,
        num_channels_dae=num_channels_dae,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout,
        image_size=image_size,
        embedding_type=embedding_type,
        nz=nz,
        n_mlp=n_mlp,
        t_emb_dim=t_emb_dim,
        ngf=ngf,
        num_timesteps=num_timesteps,
        beta_min=beta_min,
        beta_max=beta_max,
    )
    inference_args = _build_inference_args(args)

    ckpt = run_dir / "checkpoints" / f"{which_epoch}_gen_diffusive_1.pth"
    gen = _load_generator(ckpt, inference_args, device)
    logger.info("loaded generator from %s", ckpt)

    pos_coeff = _PosteriorCoefficients(num_timesteps, beta_min, beta_max, device)

    ids, indices = _resolve_split(image_h5, fold, phase)
    if max_patients is not None:
        ids = ids[:max_patients]
        indices = indices[:max_patients]
    logger.info("inference over %d patients (%s/fold%d)", len(ids), phase, fold)

    nii_dir = out_dir / "nifti"
    png_dir = out_dir / "png"
    nii_dir.mkdir(exist_ok=True)
    png_dir.mkdir(exist_ok=True)

    metrics_path = out_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as f_csv:
        writer = csv.DictWriter(
            f_csv,
            fieldnames=["patient_id", "psnr_brain", "ssim_volume", "n_slices", "status"],
        )
        writer.writeheader()

        n_ok = 0
        psnr_vals: list[float] = []
        ssim_vals: list[float] = []
        for pid, pidx in zip(ids, indices):
            try:
                psnr, ssim_v, n_slices = _process_patient(
                    pid=pid, pidx=pidx, image_h5=image_h5,
                    source_modality=source_modality, target_modality=target_modality,
                    image_size=image_size, min_brain_voxels=min_brain_voxels,
                    gen=gen, pos_coeff=pos_coeff, num_timesteps=num_timesteps,
                    nz=nz, device=device, nii_dir=nii_dir, png_dir=png_dir,
                )
                writer.writerow({
                    "patient_id": pid, "psnr_brain": psnr,
                    "ssim_volume": ssim_v, "n_slices": n_slices, "status": "ok",
                })
                psnr_vals.append(psnr)
                ssim_vals.append(ssim_v)
                n_ok += 1
                logger.info("patient %s: PSNR=%.2f SSIM=%.4f (%d slices)",
                            pid, psnr, ssim_v, n_slices)
            except (InferenceError, RuntimeError, KeyError) as exc:
                writer.writerow({
                    "patient_id": pid, "psnr_brain": float("nan"),
                    "ssim_volume": float("nan"), "n_slices": 0, "status": str(exc),
                })
                logger.warning("patient %s failed: %s", pid, exc)
            f_csv.flush()

    summary = {
        "n_patients_seen": len(ids),
        "n_patients_succeeded": n_ok,
        "psnr_brain_mean": float(np.mean(psnr_vals)) if psnr_vals else float("nan"),
        "psnr_brain_std": float(np.std(psnr_vals)) if psnr_vals else float("nan"),
        "ssim_volume_mean": float(np.mean(ssim_vals)) if ssim_vals else float("nan"),
        "ssim_volume_std": float(np.std(ssim_vals)) if ssim_vals else float("nan"),
        "checkpoint": str(ckpt),
        "image_h5": str(image_h5),
        "phase": phase,
        "fold": fold,
        "source_modality": source_modality,
        "target_modality": target_modality,
        "num_timesteps": num_timesteps,
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info("inference completed — %d/%d patients ok, mean PSNR=%.2f mean SSIM=%.4f",
                n_ok, len(ids), summary["psnr_brain_mean"], summary["ssim_volume_mean"])
    return out_dir


def _process_patient(
    *,
    pid: str,
    pidx: int,
    image_h5: Path,
    source_modality: str,
    target_modality: str,
    image_size: int,
    min_brain_voxels: int,
    gen: torch.nn.Module,
    pos_coeff: _PosteriorCoefficients,
    num_timesteps: int,
    nz: int,
    device: torch.device,
    nii_dir: Path,
    png_dir: Path,
) -> tuple[float, float, int]:
    """Synthesise one patient end-to-end. Returns (psnr, ssim, n_slices)."""
    thresholds = _percentile_thresholds_per_patient(
        image_h5, pidx, (source_modality, target_modality),
        upper=99.5, foreground_threshold=0.0,
    )
    src_vol = _patient_native_volume(image_h5, pidx, source_modality)
    tgt_vol = _patient_native_volume(image_h5, pidx, target_modality)
    with h5py.File(image_h5, "r") as f:
        brain = np.asarray(f["masks/brain"][pidx], dtype=np.uint8)

    H, W, D = src_vol.shape
    per_z = brain.reshape(-1, D).sum(axis=0)
    valid_z = np.flatnonzero(per_z >= min_brain_voxels).tolist()
    if not valid_z:
        raise InferenceError(f"no axial slice passes min_brain_voxels={min_brain_voxels}")

    low_s, high_s = thresholds[source_modality]
    low_t, high_t = thresholds[target_modality]

    pred_vol = np.zeros((H, W, D), dtype=np.float32)
    src_slices: list[torch.Tensor] = []
    for z in valid_z:
        s = np.clip((src_vol[:, :, z] - low_s) / (high_s - low_s), 0.0, 1.0)
        t = torch.from_numpy(s).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        t = _pad_to(t, image_size)
        t = t.mul_(2.0).sub_(1.0)  # → [-1, 1]
        src_slices.append(t)
    src_batch = torch.cat(src_slices, dim=0).to(device)  # (B, 1, size, size)

    # x_init = cat([noise, source], dim=1) — channel 0 = x_T (randn), channel 1 = source
    noise = torch.randn_like(src_batch)
    x_init = torch.cat((noise, src_batch), dim=1)
    pred = _sample_from_model(gen, pos_coeff, num_timesteps, x_init, nz, device)
    # pred: (B, 1, size, size) in [-1, 1]
    pred = (pred + 1.0) / 2.0  # → [0, 1]
    pred = pred.clamp(0.0, 1.0)
    pred = _crop_to(pred, H, W)  # (B, 1, H, W)
    pred_np = pred.squeeze(1).cpu().numpy()  # (B, H, W)
    for k, z in enumerate(valid_z):
        pred_vol[:, :, z] = pred_np[k]

    # Normalise the real target the same way for fair comparison.
    real_vol = np.clip((tgt_vol - low_t) / (high_t - low_t), 0.0, 1.0)

    mask = brain.astype(bool)
    psnr = _psnr(pred_vol, real_vol, mask)
    ssim_v = _ssim_3d(pred_vol, real_vol)

    # NIfTI: identity affine — we do not carry the original spacing through
    # the H5, and the downstream eval routines pair predictions with the
    # ground-truth NIfTIs by patient_id anyway.
    affine = np.eye(4, dtype=np.float32)
    nib.save(nib.Nifti1Image(pred_vol, affine), str(nii_dir / f"{pid}_pred_{target_modality}.nii.gz"))
    nib.save(nib.Nifti1Image(real_vol, affine), str(nii_dir / f"{pid}_real_{target_modality}_normalised.nii.gz"))

    # PNG: 3-panel of the most central valid slice.
    z_mid = valid_z[len(valid_z) // 2]
    _write_midslice_png(
        png_dir / f"{pid}_midslice.png",
        np.clip((src_vol[:, :, z_mid] - low_s) / (high_s - low_s), 0.0, 1.0),
        real_vol[:, :, z_mid],
        pred_vol[:, :, z_mid],
        source_modality, target_modality,
    )
    return psnr, ssim_v, len(valid_z)


def _write_midslice_png(path: Path, source: np.ndarray, real: np.ndarray,
                        pred: np.ndarray, source_label: str, target_label: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, im, title in zip(
        axes,
        [source, real, pred],
        [f"real {source_label}", f"real {target_label}", f"pred {target_label}"],
    ):
        ax.imshow(im.T, cmap="gray", origin="lower", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
