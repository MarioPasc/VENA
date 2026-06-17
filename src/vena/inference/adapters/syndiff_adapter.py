"""C3-SynDiff adapter (Özbey *et al.* 2023, 2D axial DDPM with adversarial bridge).

One-to-one panel of three (one trained generator per source modality).
The 4-step DDPM reverse sampler is paper-faithful; the adapter
reproduces the slice-by-slice logic from
``vena.competitors.syndiff.inference._process_patient`` inline rather
than calling it directly, because the upstream helper also writes
NIfTI/PNG to disk — we want a pure tensor return.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import h5py
import numpy as np
import torch

from vena.inference.base import (
    InferenceModel,
    InferenceModelError,
    InferenceResult,
    resolve_device,
)
from vena.inference.harmonisation import apply_harmonisation
from vena.inference.image_dataset import row_index_for_patient
from vena.inference.registry import register_inference_model

if TYPE_CHECKING:
    from vena.data.registry import CohortEntry


class SynDiffAdapterError(InferenceModelError):
    """Raised on SynDiff checkpoint/build failures."""


@register_inference_model("syndiff")
class SynDiffAdapter(InferenceModel):
    """SynDiff inference adapter — 4-step DDPM reverse + 2D slice-stack."""

    def __init__(
        self,
        *,
        name: str,
        run_dir: str | Path,
        source_modality: str,
        target_modality: str = "t1c",
        checkpoint_epoch: str | int = "best",
        device: str | torch.device = "cuda:0",
        nfe_list: tuple[int, ...] = (4,),
        selection_nfe: int = 4,
        image_size: int = 256,
        min_brain_voxels: int = 1000,
        num_timesteps: int = 4,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        nz: int = 100,
        slice_batch: int = 32,
    ) -> None:
        super().__init__(
            name=name,
            device=resolve_device(device),
            nfe_list=nfe_list,
            selection_nfe=selection_nfe,
        )
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.source_modality = source_modality
        self.target_modality = target_modality
        self.checkpoint_epoch = checkpoint_epoch
        self.image_size = int(image_size)
        self.min_brain_voxels = int(min_brain_voxels)
        self.num_timesteps = int(num_timesteps)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.nz = int(nz)
        # Slice-batch size for chunked reverse DDPM (see predict).
        # 32 keeps the working set ≤ ~6 GB on V100 32 GB; A100 40 GB can
        # afford 64+ without OOM. Set via YAML kwargs if needed.
        self.slice_batch = max(1, int(slice_batch))

        self._gen: torch.nn.Module | None = None
        self._pos_coeff: object | None = None
        self._checkpoint_path: Path | None = None

    def setup(self) -> None:
        if self._is_setup:
            return
        from vena.competitors.syndiff.inference import (
            _build_inference_args,
            _load_generator,
            _PosteriorCoefficients,
        )

        ck_dir = self.run_dir / "checkpoints"
        ck_path = ck_dir / f"{self.checkpoint_epoch}_gen_diffusive_1.pth"
        if not ck_path.is_file():
            available = sorted(p.name for p in ck_dir.glob("*.pth"))
            raise SynDiffAdapterError(
                f"{self.name}: checkpoint {ck_path.name} missing; available: {available}"
            )
        self._checkpoint_path = ck_path

        cfg = _config_from_decision(
            self.run_dir,
            image_size=self.image_size,
            num_timesteps=self.num_timesteps,
            beta_min=self.beta_min,
            beta_max=self.beta_max,
            nz=self.nz,
        )
        inference_args = _build_inference_args(cfg)
        self._gen = _load_generator(ck_path, inference_args, self.device)
        self._pos_coeff = _PosteriorCoefficients(
            self.num_timesteps, self.beta_min, self.beta_max, self.device
        )
        super().setup()

    def predict(
        self,
        cohort: CohortEntry,
        patient_id: str,
        nfe: int,
    ) -> InferenceResult:
        from vena.competitors.syndiff.inference import (
            _crop_to,
            _pad_to,
            _patient_native_volume,
            _percentile_thresholds_per_patient,
            _sample_from_model,
        )

        del nfe  # paper-faithful 4-step DDPM is fixed at training time
        self._require_setup()
        assert self._gen is not None and self._pos_coeff is not None
        self._reset_peak_vram(self.device)
        self._sync(self.device)
        t0 = time.perf_counter()

        pidx = row_index_for_patient(cohort.image_h5, patient_id)
        thresholds = _percentile_thresholds_per_patient(
            Path(cohort.image_h5),
            pidx,
            (self.source_modality,),
            upper=99.5,
            foreground_threshold=0.0,
        )
        src_vol = _patient_native_volume(Path(cohort.image_h5), pidx, self.source_modality)
        with h5py.File(cohort.image_h5, "r") as f:
            brain = np.asarray(f["masks/brain"][pidx], dtype=np.uint8)

        h, w, d = src_vol.shape
        per_z = brain.reshape(-1, d).sum(axis=0)
        valid_z = np.flatnonzero(per_z >= self.min_brain_voxels).tolist()
        if not valid_z:
            raise SynDiffAdapterError(
                f"{self.name}: patient '{patient_id}' has no axial slice with "
                f">= {self.min_brain_voxels} brain voxels"
            )
        low_s, high_s = thresholds[self.source_modality]
        # Slice-chunked DDPM reverse sampling. The vendored
        # ``_sample_from_model`` runs the full ``num_timesteps`` reverse
        # loop in-memory: at 117 axial slices × 256 × 256 × intermediate
        # activations per step, the working set is ~20 GB on V100 32 GB
        # and OOMs when the box is shared. Chunking by ``slice_batch``
        # keeps the working set under ~6 GB while preserving the per-
        # slice sampler output byte-for-byte (DDPM reverse is fully
        # independent across slices).
        pred_vol = np.zeros((h, w, d), dtype=np.float32)
        for chunk_start in range(0, len(valid_z), self.slice_batch):
            chunk = valid_z[chunk_start : chunk_start + self.slice_batch]
            slices: list[torch.Tensor] = []
            for z in chunk:
                s = np.clip((src_vol[:, :, z] - low_s) / (high_s - low_s), 0.0, 1.0)
                t = torch.from_numpy(s).unsqueeze(0).unsqueeze(0)
                t = _pad_to(t, self.image_size)
                t = t.mul_(2.0).sub_(1.0)
                slices.append(t)
            src_batch = torch.cat(slices, dim=0).to(self.device)
            noise = torch.randn_like(src_batch)
            x_init = torch.cat((noise, src_batch), dim=1)
            with torch.no_grad():
                pred_chunk = _sample_from_model(
                    self._gen,
                    self._pos_coeff,
                    self.num_timesteps,
                    x_init,
                    self.nz,
                    self.device,
                )
            pred_chunk = ((pred_chunk + 1.0) / 2.0).clamp(0.0, 1.0)
            pred_chunk = _crop_to(pred_chunk, h, w)
            pred_np_chunk = pred_chunk.squeeze(1).cpu().numpy()  # (Bc, H, W)
            for k, z in enumerate(chunk):
                pred_vol[:, :, z] = pred_np_chunk[k]
            # Release the chunk's GPU allocations before the next chunk.
            del src_batch, noise, x_init, pred_chunk, pred_np_chunk
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        pred_t = torch.from_numpy(pred_vol)
        brain_t = torch.from_numpy(brain).to(torch.float32)
        harmonised = apply_harmonisation(pred_t, brain_mask=brain_t)

        self._sync(self.device)
        seconds = time.perf_counter() - t0
        return InferenceResult(
            t1c_synthetic_harmonised=harmonised,
            t1c_synthetic_raw=pred_t.contiguous(),
            inference_seconds=float(seconds),
            peak_vram_mb=self._peak_vram_mb(self.device),
        )

    def teardown(self) -> None:
        if self._gen is not None:
            del self._gen
            self._gen = None
        if self._pos_coeff is not None:
            del self._pos_coeff
            self._pos_coeff = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self._is_setup = False


def _config_from_decision(
    run_dir: Path,
    *,
    image_size: int,
    num_timesteps: int,
    beta_min: float,
    beta_max: float,
    nz: int,
) -> SimpleNamespace:
    """Recover SynDiff architecture from ``decision.json`` with sane fallbacks."""
    hp: dict[str, object] = {}
    decision_path = run_dir / "decision.json"
    if decision_path.is_file():
        try:
            decision = json.loads(decision_path.read_text())
        except json.JSONDecodeError as exc:
            raise SynDiffAdapterError(f"{decision_path} is not valid JSON: {exc}") from exc
        hp = dict(decision.get("hyperparams", {}))
    return SimpleNamespace(
        z_emb_dim=int(hp.get("z_emb_dim", 256)),
        num_channels_dae=int(hp.get("num_channels_dae", 64)),
        ch_mult=tuple(hp.get("ch_mult", (1, 1, 2, 2, 4, 4))),
        num_res_blocks=int(hp.get("num_res_blocks", 2)),
        attn_resolutions=tuple(hp.get("attn_resolutions", (16,))),
        dropout=float(hp.get("dropout", 0.0)),
        image_size=image_size,
        embedding_type=str(hp.get("embedding_type", "positional")),
        nz=nz,
        n_mlp=int(hp.get("n_mlp", 3)),
        t_emb_dim=int(hp.get("t_emb_dim", 256)),
        ngf=int(hp.get("ngf", 64)),
        num_timesteps=num_timesteps,
        beta_min=beta_min,
        beta_max=beta_max,
    )
