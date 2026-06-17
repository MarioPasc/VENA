"""C0-Identity adapter — $\\widehat{T_{1c}} \\equiv T_{1\\text{pre}}$.

The null-model floor from validation §2 row 1 / §4.3.5 defence #3. No
learning. ``predict()`` reads the patient's T1pre from the image H5,
applies §4.1 harmonisation, and returns the result as the predicted
T1c. Used as the lower bound on bright-region error concentration —
any method that does not beat C0 on §4.3 metrics has added no
enhancement information beyond identity.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

from vena.inference.base import InferenceModel, InferenceResult, resolve_device
from vena.inference.harmonisation import apply_harmonisation
from vena.inference.image_dataset import load_image_modalities
from vena.inference.registry import register_inference_model

if TYPE_CHECKING:
    from vena.data.registry import CohortEntry


@register_inference_model("identity")
class IdentityAdapter(InferenceModel):
    """Pass T1pre through §4.1 harmonisation as the prediction."""

    def __init__(
        self,
        *,
        name: str = "C0-Identity",
        device: str | torch.device = "cuda:0",
        nfe_list: tuple[int, ...] = (1,),
        selection_nfe: int = 1,
    ) -> None:
        super().__init__(
            name=name,
            device=resolve_device(device),
            nfe_list=nfe_list,
            selection_nfe=selection_nfe,
        )

    def setup(self) -> None:
        self._is_setup = True

    def predict(
        self,
        cohort: CohortEntry,
        patient_id: str,
        nfe: int,
    ) -> InferenceResult:
        del nfe  # identity ignores NFE
        self._require_setup()
        self._reset_peak_vram(self.device)
        t0 = time.perf_counter()

        mods = load_image_modalities(cohort.image_h5, patient_id, ("t1pre",))
        t1pre = torch.from_numpy(mods["t1pre"])
        brain = torch.from_numpy(mods["brain"]).to(torch.float32)
        # Raw (pre-harmonisation) volume per validation §5.3 — the native
        # T1pre intensity is what we are claiming as the synthetic T1c
        # before any normalisation.
        raw = t1pre.clone()
        harmonised = apply_harmonisation(t1pre, brain_mask=brain)

        self._sync(self.device)
        seconds = time.perf_counter() - t0
        return InferenceResult(
            t1c_synthetic_harmonised=harmonised.contiguous(),
            t1c_synthetic_raw=raw.contiguous(),
            inference_seconds=float(seconds),
            peak_vram_mb=self._peak_vram_mb(self.device),
        )

    def teardown(self) -> None:
        self._is_setup = False
