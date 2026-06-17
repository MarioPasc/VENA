"""C4-3D-DiT adapter (Peebles & Xie 2023 + Eidex 2025 3D adaptation).

Loads the architecture from the checkpoint's ``arch_meta`` block
(``vena.competitors.dit_3d.inference._rebuild_dit_from_meta``) so no
external ``--unet-arch-config`` JSON is required (one fewer YAML field
relative to T1C-RFlow). Sampling uses the same Euler path as
T1C-RFlow but with the DiT backbone in place of the U-Net.
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


class DiT3DAdapterError(InferenceModelError):
    """Raised on 3D-DiT checkpoint/build failures."""


@register_inference_model("dit_3d")
class DiT3DAdapter(InferenceModel):
    """3D-DiT inference adapter — Euler integration + MAISI decode."""

    def __init__(
        self,
        *,
        name: str = "C4-3D-DiT",
        run_dir: str | Path,
        vae_checkpoint: str | Path,
        checkpoint_epoch: str | int = "best",
        device: str | torch.device = "cuda:0",
        nfe_list: tuple[int, ...] = (50, 100, 200),
        selection_nfe: int = 100,
        input_latents: tuple[str, str] = ("t1pre", "flair"),
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
            raise DiT3DAdapterError(f"{name}: vae_checkpoint is required")
        self.checkpoint_epoch = checkpoint_epoch
        self.input_latents = tuple(input_latents)

        self._dit: torch.nn.Module | None = None
        self._scheduler: Any = None
        self._vae: MaisiDecoder | None = None
        self._checkpoint_path: Path | None = None

    def setup(self) -> None:
        if self._is_setup:
            return
        from vena.competitors.dit_3d.inference import (
            _rebuild_dit_from_meta,
            _resolve_checkpoint,
        )
        from vena.competitors.dit_3d.runner import _build_scheduler

        self._checkpoint_path = _resolve_checkpoint(self.run_dir, self.checkpoint_epoch)
        blob = torch.load(self._checkpoint_path, map_location="cpu", weights_only=False)
        if "arch_meta" not in blob:
            raise DiT3DAdapterError(
                f"{self.name}: checkpoint {self._checkpoint_path} lacks 'arch_meta' — "
                "re-train with arch_meta-aware wrapper."
            )
        dit = _rebuild_dit_from_meta(blob["arch_meta"])
        state = blob.get("dit_state_dict", blob)
        dit.load_state_dict(state, strict=False)
        dit = dit.to(self.device).eval()
        self._dit = dit

        # The DiT-3D runner builds an RFlowScheduler with paper-pinned
        # base_img_size_numel; rely on the existing helper rather than
        # re-deriving the kwargs.
        self._scheduler = _build_scheduler(num_train_timesteps=1000)
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
        from vena.competitors.dit_3d.inference import _euler_sample

        self._require_setup()
        assert self._dit is not None and self._scheduler is not None and self._vae is not None
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
            self._dit, self._scheduler, z_cond, z_cond2, int(nfe), self.device
        )
        crop_spec = build_crop_spec_from_h5(cohort.image_h5, patient_id)
        pred_box = decode_box(self._vae, z_pred, crop_spec)
        # Map the box volume back to native coordinates so the prediction
        # aligns voxel-wise with the cohort's masks/brain and reference T1c.
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
        if self._dit is not None:
            del self._dit
            self._dit = None
        if self._scheduler is not None:
            del self._scheduler
            self._scheduler = None
        if self._vae is not None:
            del self._vae
            self._vae = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self._is_setup = False
