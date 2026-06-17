"""C6-3D-LDDPM adapter (Eidex 2025 §4 baseline — pure conditional DDPM).

Backbone-symmetric with C5-T1C-RFlow (same MAISI U-Net build path);
the only delta is the :class:`DDPMScheduler` and the K-step DDPM
denoising loop in ``vena.competitors.lddpm_3d.inference._ddpm_sample``.
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
from vena.inference.adapters.t1c_rflow_adapter import _patient_idx_in_latent_h5
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


class LDDPM3DAdapterError(InferenceModelError):
    """Raised on LDDPM checkpoint/build failures."""


@register_inference_model("lddpm_3d")
class LDDPM3DAdapter(InferenceModel):
    """3D-Latent-DDPM inference adapter — K-step DDPM denoising + MAISI decode."""

    def __init__(
        self,
        *,
        name: str = "C6-3D-LDDPM",
        run_dir: str | Path,
        vae_checkpoint: str | Path,
        checkpoint_epoch: str | int = "best",
        device: str | torch.device = "cuda:0",
        nfe_list: tuple[int, ...] = (200, 500, 1000),
        selection_nfe: int = 500,
        input_latents: tuple[str, str] = ("t1pre", "flair"),
        latent_channels: int = 4,
    ) -> None:
        super().__init__(
            name=name,
            device=resolve_device(device),
            nfe_list=nfe_list,
            selection_nfe=selection_nfe,
        )
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.vae_checkpoint = resolve_path(vae_checkpoint)
        if self.vae_checkpoint is None:
            raise LDDPM3DAdapterError(f"{name}: vae_checkpoint is required")
        self.checkpoint_epoch = checkpoint_epoch
        self.input_latents = tuple(input_latents)
        self.latent_channels = int(latent_channels)

        self._unet: torch.nn.Module | None = None
        self._scheduler: Any = None
        self._vae: MaisiDecoder | None = None
        self._checkpoint_path: Path | None = None

    def setup(self) -> None:
        if self._is_setup:
            return
        from vena.competitors.lddpm_3d.inference import _resolve_checkpoint
        from vena.competitors.lddpm_3d.runner import _build_scheduler, _build_unet

        self._checkpoint_path = _resolve_checkpoint(self.run_dir, self.checkpoint_epoch)
        unet = _build_unet(
            latent_channels=self.latent_channels,
            cond_latents=len(self.input_latents),
        )
        blob = torch.load(self._checkpoint_path, map_location="cpu", weights_only=False)
        state = blob.get("unet_state_dict") or blob.get("unet") or blob
        unet.load_state_dict(state, strict=False)
        unet = unet.to(self.device).eval()
        self._unet = unet

        # Match training-time scheduler kwargs verbatim (per the integration
        # note in lddpm_3d.md: beta_start=0.0015 not the upstream typo 0.0005).
        self._scheduler = _build_scheduler(
            num_train_timesteps=1000,
            beta_start=0.0015,
            beta_end=0.0195,
            beta_schedule="scaled_linear_beta",
            clip_sample=False,
        )
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
        from vena.competitors.lddpm_3d.inference import _ddpm_sample

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

        z_pred, _ = _ddpm_sample(
            self._unet, self._scheduler, z_cond, z_cond2, int(nfe), self.device
        )
        crop_spec = build_crop_spec_from_h5(cohort.image_h5, patient_id)
        pred_box = decode_box(self._vae, z_pred, crop_spec)
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
