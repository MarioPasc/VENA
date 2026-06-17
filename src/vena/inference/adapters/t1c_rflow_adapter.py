"""C5-T1C-RFlow adapter (Eidex *et al.* 2025, latent rectified flow).

Reuses the building blocks already vendored under
:mod:`vena.competitors.t1c_rflow`:

* ``runner._build_unet`` — paper-faithful MAISI U-Net rebuild.
* ``runner._build_scheduler`` — :class:`RFlowScheduler` with the same
  kwargs Eidex *et al.* used.
* ``inference._euler_sample`` — Euler integration of ``t: 1 → 0`` over
  ``nfe`` steps.
* ``inference._resolve_checkpoint`` — best/latest/epoch_N file lookup.

Conditioning is ``(z_T1pre, z_FLAIR)`` per paper §3.1; no T2.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import torch

from vena.common import MaisiDecoder, invert_crop_pad, load_autoencoder
from vena.common.decode import decode_box
from vena.inference.base import (
    InferenceModel,
    InferenceModelError,
    InferenceResult,
    resolve_device,
    resolve_path,
)
from vena.inference.harmonisation import apply_harmonisation
from vena.inference.image_dataset import load_image_modalities
from vena.inference.registry import register_inference_model
from vena.model.fm.eval.exhaustive import build_crop_spec_from_h5

if TYPE_CHECKING:
    from vena.data.registry import CohortEntry


class T1CRFlowAdapterError(InferenceModelError):
    """Raised on T1C-RFlow checkpoint/build failures."""


@register_inference_model("t1c_rflow")
class T1CRFlowAdapter(InferenceModel):
    """Paper-faithful T1C-RFlow inference at one (patient, NFE) at a time."""

    def __init__(
        self,
        *,
        name: str = "C5-T1C-RFlow",
        run_dir: str | Path,
        unet_arch_config: str | Path,
        vae_checkpoint: str | Path,
        checkpoint_epoch: str | int = "best",
        device: str | torch.device = "cuda:0",
        nfe_list: tuple[int, ...] = (50, 100, 200),
        selection_nfe: int = 100,
        input_latents: tuple[str, str] = ("t1pre", "flair"),
        latent_channels: int = 4,
        num_train_timesteps: int = 1000,
    ) -> None:
        super().__init__(
            name=name,
            device=resolve_device(device),
            nfe_list=nfe_list,
            selection_nfe=selection_nfe,
        )
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.unet_arch_config = Path(unet_arch_config).expanduser().resolve()
        self.vae_checkpoint = resolve_path(vae_checkpoint)
        if self.vae_checkpoint is None:
            raise T1CRFlowAdapterError(f"{name}: vae_checkpoint is required")
        self.checkpoint_epoch = checkpoint_epoch
        self.input_latents = tuple(input_latents)
        self.latent_channels = int(latent_channels)
        self.num_train_timesteps = int(num_train_timesteps)

        self._unet: torch.nn.Module | None = None
        self._scheduler: Any = None
        self._vae: MaisiDecoder | None = None
        self._checkpoint_path: Path | None = None

    def setup(self) -> None:
        if self._is_setup:
            return
        from vena.competitors.t1c_rflow.inference import _resolve_checkpoint
        from vena.competitors.t1c_rflow.runner import _build_scheduler, _build_unet

        self._checkpoint_path = _resolve_checkpoint(self.run_dir, self.checkpoint_epoch)
        unet = _build_unet(
            self.unet_arch_config,
            latent_channels=self.latent_channels,
            cond_latents=len(self.input_latents),
        )
        blob = torch.load(self._checkpoint_path, map_location="cpu", weights_only=False)
        state = blob.get("unet_state_dict") or blob.get("unet") or blob
        unet.load_state_dict(state, strict=False)
        unet = unet.to(self.device).eval()
        self._unet = unet

        self._scheduler = _build_scheduler(num_train_timesteps=self.num_train_timesteps)
        self._vae = MaisiDecoder(
            handle=load_autoencoder(self.vae_checkpoint, device=str(self.device))
        )
        super().setup()

    def predict(
        self,
        cohort: CohortEntry,
        patient_id: str,
        nfe: int,
    ) -> InferenceResult:
        from vena.competitors.t1c_rflow.inference import _euler_sample

        self._require_setup()
        assert self._unet is not None and self._scheduler is not None and self._vae is not None
        self._reset_peak_vram(self.device)
        self._sync(self.device)
        t0 = time.perf_counter()

        pidx = _patient_idx_in_latent_h5(cohort.latent_h5, patient_id)
        with h5py.File(cohort.latent_h5, "r") as f:
            z_cond_np = np.asarray(f[f"latents/{self.input_latents[0]}"][pidx], dtype=np.float32)
            z_cond2_np = np.asarray(f[f"latents/{self.input_latents[1]}"][pidx], dtype=np.float32)
        z_cond = torch.from_numpy(z_cond_np).unsqueeze(0).to(self.device)
        z_cond2 = torch.from_numpy(z_cond2_np).unsqueeze(0).to(self.device)

        z_pred, _ = _euler_sample(
            self._unet, self._scheduler, z_cond, z_cond2, int(nfe), self.device
        )
        crop_spec = build_crop_spec_from_h5(cohort.image_h5, patient_id)
        pred_box = decode_box(self._vae, z_pred, crop_spec)  # (Hbox, Wbox, Dbox) in [0,1]
        # Map box → native so the prediction lives in masks/brain coordinates.
        pred_native = invert_crop_pad(pred_box[None, None], crop_spec)[0, 0]
        raw = pred_native.detach().cpu().contiguous()

        mods = load_image_modalities(cohort.image_h5, patient_id, ())
        brain_native = torch.from_numpy(mods["brain"]).to(torch.float32)
        harmonised = apply_harmonisation(pred_native.cpu(), brain_mask=brain_native)

        self._sync(self.device)
        seconds = time.perf_counter() - t0
        return InferenceResult(
            t1c_synthetic_harmonised=harmonised,
            t1c_synthetic_raw=raw,
            inference_seconds=float(seconds),
            peak_vram_mb=self._peak_vram_mb(self.device),
        )

    def teardown(self) -> None:
        if self._unet is not None:
            del self._unet
            self._unet = None
        if self._scheduler is not None:
            del self._scheduler
            self._scheduler = None
        if self._vae is not None:
            del self._vae
            self._vae = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self._is_setup = False


def _patient_idx_in_latent_h5(latent_h5: Path | str, patient_id: str) -> int:
    """Locate a patient's row in the latent H5 by scanning ``/ids``."""
    with h5py.File(latent_h5, "r") as f:
        ids = [b.decode() if isinstance(b, bytes) else str(b) for b in f["ids"][:]]
    try:
        return ids.index(patient_id)
    except ValueError as exc:
        raise T1CRFlowAdapterError(f"patient '{patient_id}' not in {latent_h5}/ids") from exc


def _brain_to_box(brain_native: torch.Tensor, crop_spec: Any) -> torch.Tensor:
    """Crop/pad the native brain mask into the decoded-box shape."""
    from vena.common import apply_crop_pad

    box = apply_crop_pad(brain_native[None, None].float(), crop_spec)[0, 0]
    return (box > 0.5).to(torch.float32)
