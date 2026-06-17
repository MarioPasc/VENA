"""C1-pGAN adapter (Dar *et al.* 2019, 2D axial slice-stacked).

pGAN is a one-to-one model: a separate generator is trained per source
modality (``t1pre→t1c``, ``t2→t1c``, ``flair→t1c``) following the
icon-lab convention. The YAML registry exposes ``source_modality`` so
the panel-of-three entries (``C1-pGAN-t1pre``, ``-t2``, ``-flair``)
share a single adapter class.

The slice-by-slice forward pass is delegated to the existing
``vena.competitors.pgan_cgan.inference._infer_one_patient`` helper —
the same path used by the per-competitor smoke runner — so the 2D→3D
stacking semantics match validation §2 rule 10 exactly (no
inter-slice smoothing, no multi-view fusion).
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
from vena.inference.image_dataset import (
    row_index_for_patient,
)
from vena.inference.registry import register_inference_model

if TYPE_CHECKING:
    from vena.data.registry import CohortEntry


class PGANAdapterError(InferenceModelError):
    """Raised on pGAN checkpoint/build failures."""


@register_inference_model("pgan")
class PGANAdapter(InferenceModel):
    """One-to-one pGAN (single source modality → T1c)."""

    def __init__(
        self,
        *,
        name: str,
        run_dir: str | Path,
        source_modality: str,
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
        self.source_modality = source_modality
        self.target_modality = target_modality
        self.checkpoint_epoch = checkpoint_epoch
        self.image_size = int(image_size)
        self.min_brain_voxels = int(min_brain_voxels)

        self._netG: torch.nn.Module | None = None
        self._checkpoint_path: Path | None = None

    def setup(self) -> None:
        if self._is_setup:
            return
        from vena.competitors.pgan_cgan.inference import _build_generator

        ck_dir = self.run_dir / "checkpoints"
        ck_path = ck_dir / f"{self.checkpoint_epoch}_net_G.pth"
        if not ck_path.is_file():
            available = sorted(p.name for p in ck_dir.glob("*.pth"))
            raise PGANAdapterError(
                f"{self.name}: checkpoint {ck_path.name} missing; available: {available}"
            )
        self._checkpoint_path = ck_path

        # pGAN and ResViT both ship a top-level ``models`` package under their
        # own upstream tree. Python's module cache keys them by the same name
        # ``models.networks``, so the *first* adapter to setup() wins and the
        # second adapter receives the wrong ``define_G`` signature. Drop the
        # shared keys here so the upstream import resolves against pGAN's
        # sys.path injection.
        _invalidate_upstream_models()
        opt = _opt_from_decision(
            self.run_dir, default_input_nc=1, device=self.device, image_size=self.image_size
        )
        self._netG = _build_generator(ck_path, opt)
        super().setup()

    def predict(
        self,
        cohort: CohortEntry,
        patient_id: str,
        nfe: int,
    ) -> InferenceResult:
        from vena.competitors.pgan_cgan.inference import _infer_one_patient

        del nfe  # single-shot 2D forward; NFE ignored
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
            input_modalities=(self.source_modality,),
            target_modality=self.target_modality,
            image_size=self.image_size,
            min_brain_voxels=self.min_brain_voxels,
            device=self.device,
        )
        # pred_np in [0, 1] over per-modality min-max normalised range.
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


def _invalidate_upstream_models() -> None:
    """Drop the ``models`` / ``models.networks`` cache entries (pGAN/ResViT collision).

    Both icon-lab repos vendor their own top-level ``models`` package, so the
    second ``from models import networks`` returns whichever one was imported
    first. Removing the cache keys here forces Python to honour the current
    ``sys.path`` injection performed inside the upstream ``_build_generator``.
    """
    import sys as _sys

    for key in [
        k for k in list(_sys.modules) if k == "models" or k.startswith("models.") or k == "networks"
    ]:
        _sys.modules.pop(key, None)


def _opt_from_decision(
    run_dir: Path,
    *,
    default_input_nc: int,
    device: torch.device,
    image_size: int,
) -> SimpleNamespace:
    """Build the icon-lab ``opt`` Namespace from the run's ``decision.json``.

    Mirrors the recovery logic in
    ``vena.competitors.pgan_cgan.inference.run_inference`` so the channel
    count and ``ngf`` match the trained generator. Falls back to the
    icon-lab defaults when ``decision.json`` is absent.
    """
    hp: dict[str, object] = {}
    decision_path = run_dir / "decision.json"
    if decision_path.is_file():
        try:
            decision = json.loads(decision_path.read_text())
        except json.JSONDecodeError as exc:
            raise PGANAdapterError(f"{decision_path} is not valid JSON: {exc}") from exc
        hp = dict(decision.get("hyperparams", {}))
    gpu_ids = [device.index] if device.type == "cuda" else []
    return SimpleNamespace(
        input_nc=int(hp.get("input_nc", default_input_nc)),
        output_nc=int(hp.get("output_nc", 1)),
        ngf=int(hp.get("ngf", 64)),
        norm=str(hp.get("norm", "instance")),
        no_dropout=bool(hp.get("no_dropout", False)),
        init_type=str(hp.get("init_type", "normal")),
        image_size=image_size,
        gpu_ids=gpu_ids,
    )
