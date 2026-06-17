"""C7-3D-Latent-Pix2Pix adapter (Isola 2017 + Eidex 2025 §4 baseline).

Single generator forward pass: ``z_pred = netG(cat([z_t1pre, z_flair]))``,
then MAISI decode. No NFE — ``nfe_list`` is fixed to ``(1,)``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

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


class LPix2Pix3DAdapterError(InferenceModelError):
    """Raised on Pix2Pix checkpoint/build failures."""


@register_inference_model("lpix2pix_3d")
class LPix2Pix3DAdapter(InferenceModel):
    """3D-Latent-Pix2Pix inference adapter — single G forward + MAISI decode."""

    def __init__(
        self,
        *,
        name: str = "C7-3D-Latent-Pix2Pix",
        run_dir: str | Path,
        vae_checkpoint: str | Path,
        checkpoint_epoch: str | int = "best",
        device: str | torch.device = "cuda:0",
        nfe_list: tuple[int, ...] = (1,),
        selection_nfe: int = 1,
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
            raise LPix2Pix3DAdapterError(f"{name}: vae_checkpoint is required")
        self.checkpoint_epoch = checkpoint_epoch
        self.input_latents = tuple(input_latents)

        self._netG: torch.nn.Module | None = None
        self._vae: MaisiDecoder | None = None
        self._checkpoint_path: Path | None = None

    def setup(self) -> None:
        if self._is_setup:
            return
        from vena.competitors.lpix2pix_3d.inference import (
            _rebuild_generator_from_meta,
            _resolve_checkpoint,
        )

        self._checkpoint_path = _resolve_checkpoint(self.run_dir, self.checkpoint_epoch)
        blob = torch.load(self._checkpoint_path, map_location="cpu", weights_only=False)
        if "arch_meta" not in blob:
            raise LPix2Pix3DAdapterError(
                f"{self.name}: checkpoint {self._checkpoint_path} lacks 'arch_meta'."
            )
        netG = _rebuild_generator_from_meta(blob["arch_meta"])
        state = blob.get("G_state_dict", blob)
        netG.load_state_dict(state, strict=False)
        netG = netG.to(self.device).eval()
        self._netG = netG
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
        del nfe  # single-shot generator; NFE ignored
        self._require_setup()
        assert self._netG is not None and self._vae is not None
        self._reset_peak_vram(self.device)
        self._sync(self.device)
        t0 = time.perf_counter()

        pidx = _patient_idx_in_latent_h5(cohort.latent_h5, patient_id)
        with h5py.File(cohort.latent_h5, "r") as f:
            cond_parts: list[torch.Tensor] = []
            for name in self.input_latents:
                arr = np.asarray(f[f"latents/{name}"][pidx], dtype=np.float32)
                cond_parts.append(torch.from_numpy(arr).unsqueeze(0).to(self.device))
        cond = torch.cat(cond_parts, dim=1)

        with torch.no_grad():
            z_pred = self._netG(cond)

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
        if self._netG is not None:
            del self._netG
            self._netG = None
        if self._vae is not None:
            del self._vae
            self._vae = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self._is_setup = False
