"""Inference for the trained ResViT model — synthesise T1c volumes from
{T1pre, T2, FLAIR} on N patients of a chosen split.

For each patient:

1. Load every axial slice with brain coverage above ``min_brain_voxels``.
2. Run the patient through the generator slice-by-slice in eval mode.
3. Re-assemble the slices into a 3D ``(H, W, D)`` volume on the original
   native grid (un-pad to the cohort-specific H, W).
4. Rescale ``[-1, 1] → [0, 1]`` to match VENA's normalised intensity space.
5. Compute whole-volume PSNR / SSIM against the real T1c (also normalised
   the same way) so we have a number to compare against VENA's FM run.
6. Save the predicted volume as NIfTI and dump a 3-panel axial PNG of the
   middle slice (source / real T1c / predicted T1c).

The implementation deliberately mirrors the training-time normalisation
pipeline so the loaded generator sees the exact same intensity statistics
it was optimised against.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import nibabel as nib
import numpy as np
import torch

from .dataset import _decode_ids, _pad_to, _percentile_thresholds_per_patient

if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)


class InferenceError(Exception):
    """Raised when inference cannot proceed (bad checkpoint, missing data)."""


_UPSTREAM_DIR = (Path(__file__).resolve().parent.parent.parent.parent
                 / "external" / "resvit" / "upstream")


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
    """Whole-volume SSIM via skimage (3D ok)."""
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        return float("nan")
    return float(ssim(real, pred, data_range=1.0))


def _build_generator(checkpoint: Path, opt) -> torch.nn.Module:
    """Reconstruct the ResViT generator and load trained weights.

    Builds the ART-block generator from scratch with `pre_trained_trans=0`
    and `pre_trained_resnet=0` (random init), then overwrites every weight
    with the trained state-dict from ``checkpoint``. The ViT ``.npz`` is NOT
    consulted at inference — the trained checkpoint already contains the
    fine-tuned transformer weights.
    """
    snapshot = list(sys.path)
    sys.path.insert(0, str(_UPSTREAM_DIR))
    try:
        from models import networks  # type: ignore[import-not-found]
    finally:
        sys.path = snapshot

    # which_model_netG='resvit' — ART blocks present. pre_trained_path is unused
    # because both pretraining flags are 0.
    netG = networks.define_G(
        opt.input_nc, opt.output_nc, opt.ngf, "resvit",
        opt.vit_name, opt.image_size, "/dev/null",
        opt.norm, not opt.no_dropout, opt.init_type, opt.gpu_ids,
        pre_trained_trans=0, pre_trained_resnet=0,
    )
    if not checkpoint.is_file():
        raise InferenceError(f"checkpoint not found: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    netG.load_state_dict(state)
    netG.eval()
    if opt.gpu_ids:
        netG.cuda(opt.gpu_ids[0])
    return netG


# ---------------------------------------------------------------------------
# Per-patient inference
# ---------------------------------------------------------------------------
def _infer_one_patient(
    image_h5: Path,
    pidx: int,
    netG: torch.nn.Module,
    input_modalities: Sequence[str],
    target_modality: str,
    image_size: int,
    min_brain_voxels: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (pred_volume, real_target, source_first_mod, brain_mask), all in [0, 1]."""
    all_mods = tuple(input_modalities) + (target_modality,)
    thresholds = _percentile_thresholds_per_patient(
        image_h5, pidx, all_mods, upper=99.5, foreground_threshold=0.0,
    )

    with h5py.File(image_h5, "r") as f:
        brain = np.asarray(f["masks/brain"][pidx]).astype(bool)
        sources_raw = {m: np.asarray(f[f"images/{m}"][pidx], dtype=np.float32)
                       for m in input_modalities}
        target_raw = np.asarray(f[f"images/{target_modality}"][pidx], dtype=np.float32)

    H, W, D = brain.shape
    pred = np.zeros_like(target_raw)
    per_z_brain = brain.reshape(-1, D).sum(axis=0)
    valid_z = np.flatnonzero(per_z_brain >= min_brain_voxels)
    logger.info("patient %d: %d valid axial slices", pidx, valid_z.size)

    # Pre-normalise once per modality.
    sources_norm: dict[str, np.ndarray] = {}
    for mod, arr in sources_raw.items():
        low, high = thresholds[mod]
        sources_norm[mod] = np.clip((arr - low) / (high - low), 0.0, 1.0)
    low_t, high_t = thresholds[target_modality]
    target_norm = np.clip((target_raw - low_t) / (high_t - low_t), 0.0, 1.0)

    with torch.no_grad():
        for z in valid_z:
            channels = [torch.from_numpy(sources_norm[m][:, :, z]) for m in input_modalities]
            A = torch.stack(channels, dim=0).unsqueeze(0)  # (1, C, H, W)
            A = _pad_to(A, image_size).mul_(2.0).sub_(1.0).to(device)
            fake_B = netG(A)
            fake_B = fake_B.add_(1.0).div_(2.0).clamp_(0.0, 1.0)  # → [0, 1]
            fake_B = _crop_to(fake_B, H, W).cpu().numpy()[0, 0]
            pred[:, :, int(z)] = fake_B

    return pred, target_norm, sources_norm[input_modalities[0]], brain.astype(np.uint8)


def _save_outputs(
    out_dir: Path,
    patient_id: str,
    pred: np.ndarray,
    real: np.ndarray,
    source: np.ndarray,
) -> dict[str, Path]:
    """Save NIfTI + 3-panel axial PNG (mid-slice)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    nii_path = out_dir / f"{patient_id}_pred_t1c.nii.gz"
    nib.save(nib.Nifti1Image(pred.astype(np.float32), affine=np.eye(4)), nii_path)
    real_nii = out_dir / f"{patient_id}_real_t1c_normalised.nii.gz"
    nib.save(nib.Nifti1Image(real.astype(np.float32), affine=np.eye(4)), real_nii)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z_mid = pred.shape[-1] // 2
    src_label = getattr(_save_outputs, "_src_label", "source")
    tgt_label = getattr(_save_outputs, "_tgt_label", "T1c")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(source[:, :, z_mid].T, cmap="gray", vmin=0, vmax=1, origin="lower")
    axes[0].set_title(f"source {src_label}"); axes[0].axis("off")
    axes[1].imshow(real[:, :, z_mid].T, cmap="gray", vmin=0, vmax=1, origin="lower")
    axes[1].set_title(f"real {tgt_label}"); axes[1].axis("off")
    axes[2].imshow(pred[:, :, z_mid].T, cmap="gray", vmin=0, vmax=1, origin="lower")
    axes[2].set_title(f"pred {tgt_label} (ResViT)"); axes[2].axis("off")
    fig.suptitle(f"{patient_id} — z={z_mid}")
    png_path = out_dir / f"{patient_id}_midslice.png"
    fig.tight_layout()
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return {"nifti": nii_path, "real_nifti": real_nii, "png": png_path}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def run_inference(
    *,
    run_dir: Path,
    epoch: str = "best",
    image_h5: Path,
    fold: int = 0,
    phase: str = "val",
    n_patients: int = 10,
    input_modalities: Sequence[str] = ("t1pre", "t2", "flair"),
    target_modality: str = "t1c",
    image_size: int = 256,
    min_brain_voxels: int = 1000,
    out_dir: Path | None = None,
    gpu_id: int = 0,
) -> Path:
    """Load ``{run_dir}/checkpoints/{epoch}_net_G.pth`` and run inference.

    Returns the output directory (``{run_dir}/inference/epoch_{epoch}/``).
    """
    run_dir = Path(run_dir)
    ckpt = run_dir / "checkpoints" / f"{epoch}_net_G.pth"
    if not ckpt.is_file():
        raise InferenceError(f"missing checkpoint {ckpt}")

    decision_path = run_dir / "decision.json"
    hp: dict = {}
    if decision_path.is_file():
        decision = json.loads(decision_path.read_text())
        hp = decision.get("hyperparams", {})
        train_data = decision.get("data", {})
        trained_mods = train_data.get("input_modalities") or list(input_modalities)
        input_modalities = tuple(trained_mods)
        target_modality = train_data.get("target_modality", target_modality)
        image_size = train_data.get("image_size", image_size)
        min_brain_voxels = train_data.get("min_brain_voxels", min_brain_voxels)
        logger.info("decision.json: input_modalities=%s → %s, image_size=%d",
                    input_modalities, target_modality, image_size)

    from types import SimpleNamespace
    opt = SimpleNamespace(
        input_nc=hp.get("input_nc", len(input_modalities)),
        output_nc=hp.get("output_nc", 1),
        ngf=hp.get("ngf", 64),
        norm=hp.get("norm", "instance"),
        no_dropout=hp.get("no_dropout", False),
        init_type=hp.get("init_type", "normal"),
        vit_name=hp.get("vit_name", "Res-ViT-B_16"),
        image_size=image_size,
        gpu_ids=[gpu_id] if torch.cuda.is_available() else [],
    )

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    netG = _build_generator(ckpt, opt)
    logger.info("Loaded generator from %s (device=%s)", ckpt, device)
    _save_outputs._src_label = "+".join(m.upper() for m in input_modalities)
    _save_outputs._tgt_label = target_modality.upper()

    out_dir = out_dir or (run_dir / "inference" / f"epoch_{epoch}")
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(image_h5, "r") as f:
        all_ids = _decode_ids(np.asarray(f["ids"]))
        candidates = ["splits/test"] if phase == "test" else [
            f"splits/cv/fold_{fold}/{phase}", f"splits/{phase}",
        ]
        key = next((c for c in candidates if c in f), None)
        if key is None:
            raise InferenceError(
                f"none of {candidates} present in {image_h5}"
            )
        split_ids = _decode_ids(np.asarray(f[key]))
    id_to_idx = {pid: i for i, pid in enumerate(all_ids)}
    # Longitudinal cohorts: scan-id may have a suffix the patient-level split id lacks.
    resolved: list[tuple[str, int]] = []
    for pid in split_ids:
        if pid in id_to_idx:
            resolved.append((pid, id_to_idx[pid]))
            continue
        for full_id, idx in id_to_idx.items():
            if full_id.startswith(f"{pid}-") or full_id.startswith(f"{pid}_"):
                resolved.append((full_id, idx))
    resolved = resolved[:n_patients]

    metrics_csv = out_dir / "metrics.csv"
    n_succeeded = 0
    n_failed = 0
    with metrics_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["patient_id", "epoch", "psnr_dB", "ssim", "n_valid_slices"]
        )
        writer.writeheader()
        for pid, pidx in resolved:
            logger.info("inference on %s (pidx=%d)", pid, pidx)
            try:
                pred, real, src, brain = _infer_one_patient(
                    image_h5=image_h5, pidx=pidx, netG=netG,
                    input_modalities=input_modalities,
                    target_modality=target_modality,
                    image_size=image_size,
                    min_brain_voxels=min_brain_voxels,
                    device=device,
                )
                psnr = _psnr(pred, real, brain.astype(bool))
                ssim = _ssim_3d(pred, real)
                n_valid = int((brain.reshape(-1, brain.shape[-1]).sum(axis=0)
                               >= min_brain_voxels).sum())
                writer.writerow({
                    "patient_id": pid,
                    "epoch": epoch,
                    "psnr_dB": f"{psnr:.4f}",
                    "ssim": f"{ssim:.4f}",
                    "n_valid_slices": n_valid,
                })
                fh.flush()
                _save_outputs(out_dir, pid, pred, real, src)
                logger.info("%s: PSNR=%.3f dB SSIM=%.3f", pid, psnr, ssim)
                n_succeeded += 1
            except (RuntimeError, KeyError, ValueError) as exc:
                logger.warning("%s: inference failed — %s", pid, exc)
                n_failed += 1

    summary = {
        "schema_version": "1.0",
        "run_dir": str(run_dir),
        "epoch": epoch,
        "n_patients": len(resolved),
        "n_patients_succeeded": n_succeeded,
        "n_patients_failed": n_failed,
        "phase": phase,
        "fold": fold,
        "metrics_csv": str(metrics_csv),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("inference summary: %d succeeded, %d failed → %s",
                n_succeeded, n_failed, out_dir / "summary.json")
    # Skill mandate: surface a clear error if every patient failed silently.
    if n_succeeded == 0:
        raise InferenceError(
            f"All {len(resolved)} inference patients failed. See logs and {metrics_csv}."
        )
    return out_dir
