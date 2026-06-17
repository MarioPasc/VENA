"""C2-ResViT adapter (Dalmaz *et al.* 2022, 2D axial residual ViT).

Panel-of-three structure identical to C1-pGAN (one trained generator per
source modality). ResViT's only additional kwargs vs pGAN are
``vit_name`` and ``image_size`` (the ART block is sized for a fixed
input shape).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

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


class ResViTAdapterError(InferenceModelError):
    """Raised on ResViT checkpoint/build failures."""


@register_inference_model("resvit")
class ResViTAdapter(InferenceModel):
    """One-to-one ResViT (single source modality → T1c)."""

    def __init__(
        self,
        *,
        name: str,
        run_dir: str | Path,
        source_modalities: tuple[str, ...] | list[str] = ("t1pre", "t2", "flair"),
        target_modality: str = "t1c",
        checkpoint_epoch: str | int = "best",
        device: str | torch.device = "cuda:0",
        nfe_list: tuple[int, ...] = (1,),
        selection_nfe: int = 1,
        image_size: int = 256,
        min_brain_voxels: int = 1000,
    ) -> None:
        super().__init__(
            name=name,
            device=resolve_device(device),
            nfe_list=nfe_list,
            selection_nfe=selection_nfe,
        )
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.source_modalities = tuple(source_modalities)
        self.target_modality = target_modality
        self.checkpoint_epoch = checkpoint_epoch
        self.image_size = int(image_size)
        self.min_brain_voxels = int(min_brain_voxels)

        self._netG: torch.nn.Module | None = None
        self._checkpoint_path: Path | None = None

    def setup(self) -> None:
        if self._is_setup:
            return
        from vena.competitors.resvit.inference import _build_generator

        ck_dir = self.run_dir / "checkpoints"
        ck_path = ck_dir / f"{self.checkpoint_epoch}_net_G.pth"
        if not ck_path.is_file():
            available = sorted(p.name for p in ck_dir.glob("*.pth"))
            raise ResViTAdapterError(
                f"{self.name}: checkpoint {ck_path.name} missing; available: {available}"
            )
        self._checkpoint_path = ck_path

        # pGAN and ResViT collide on ``models.networks`` in sys.modules — both
        # icon-lab repos ship a top-level ``models`` package. Drop the cache
        # so the upstream sys.path injection inside ``_build_generator``
        # resolves against ResViT's own tree.
        from vena.inference.adapters.pgan_adapter import _invalidate_upstream_models

        _invalidate_upstream_models()
        opt = _opt_from_decision(
            self.run_dir,
            default_input_nc=len(self.source_modalities),
            device=self.device,
            image_size=self.image_size,
        )
        self._netG = _build_generator(ck_path, opt)
        super().setup()

    def predict(
        self,
        cohort: CohortEntry,
        patient_id: str,
        nfe: int,
    ) -> InferenceResult:
        from vena.competitors.resvit.inference import _infer_one_patient

        del nfe
        self._require_setup()
        assert self._netG is not None
        self._reset_peak_vram(self.device)
        self._sync(self.device)
        t0 = time.perf_counter()

        pidx = row_index_for_patient(cohort.image_h5, patient_id)
        pred_np, _real_np, _src_np, brain_np = _infer_one_patient(
            image_h5=Path(cohort.image_h5),
            pidx=pidx,
            netG=self._netG,
            input_modalities=self.source_modalities,
            target_modality=self.target_modality,
            image_size=self.image_size,
            min_brain_voxels=self.min_brain_voxels,
            device=self.device,
        )
        pred = torch.from_numpy(pred_np)
        brain = torch.from_numpy(brain_np).to(torch.float32)
        harmonised = apply_harmonisation(pred, brain_mask=brain)
        raw = pred.contiguous()

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
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self._is_setup = False


def _opt_from_decision(
    run_dir: Path,
    *,
    default_input_nc: int,
    device: torch.device,
    image_size: int,
) -> SimpleNamespace:
    """Build the icon-lab ``opt`` Namespace for ResViT from ``decision.json``."""
    hp: dict[str, object] = {}
    decision_path = run_dir / "decision.json"
    if decision_path.is_file():
        try:
            decision = json.loads(decision_path.read_text())
        except json.JSONDecodeError as exc:
            raise ResViTAdapterError(f"{decision_path} is not valid JSON: {exc}") from exc
        hp = dict(decision.get("hyperparams", {}))
    gpu_ids = [device.index] if device.type == "cuda" else []
    return SimpleNamespace(
        input_nc=int(hp.get("input_nc", default_input_nc)),
        output_nc=int(hp.get("output_nc", 1)),
        ngf=int(hp.get("ngf", 64)),
        norm=str(hp.get("norm", "instance")),
        no_dropout=bool(hp.get("no_dropout", False)),
        init_type=str(hp.get("init_type", "normal")),
        vit_name=str(hp.get("vit_name", "Res-ViT-B_16")),
        image_size=image_size,
        gpu_ids=gpu_ids,
    )
