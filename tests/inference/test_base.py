"""Tests for the InferenceModel ABC contract."""

from __future__ import annotations

import pytest
import torch

from vena.inference.base import (
    InferenceModel,
    InferenceModelError,
    InferenceResult,
    resolve_device,
)

pytestmark = pytest.mark.unit


class _MinimalAdapter(InferenceModel):
    """Smallest possible concrete adapter used to test the ABC plumbing."""

    model_type = "_minimal"

    def setup(self) -> None:
        super().setup()

    def predict(self, cohort, patient_id, nfe):  # type: ignore[no-untyped-def]
        self._require_setup()
        vol = torch.zeros((4, 4, 4), dtype=torch.float32)
        return InferenceResult(
            t1c_synthetic_harmonised=vol.clone(),
            t1c_synthetic_raw=vol.clone(),
            inference_seconds=0.001,
            peak_vram_mb=0.0,
        )

    def teardown(self) -> None:
        self._is_setup = False


def test_predict_before_setup_raises() -> None:
    a = _MinimalAdapter(name="m", device="cpu")
    with pytest.raises(InferenceModelError):
        a.predict(cohort=None, patient_id="x", nfe=1)  # type: ignore[arg-type]


def test_lifecycle_setup_predict_teardown() -> None:
    a = _MinimalAdapter(name="m", device="cpu")
    a.setup()
    out = a.predict(cohort=None, patient_id="x", nfe=1)  # type: ignore[arg-type]
    assert isinstance(out, InferenceResult)
    assert out.t1c_synthetic_harmonised.shape == (4, 4, 4)
    assert out.inference_seconds >= 0.0
    a.teardown()


def test_selection_nfe_must_be_in_nfe_list() -> None:
    with pytest.raises(InferenceModelError):
        _MinimalAdapter(name="m", device="cpu", nfe_list=(1, 5), selection_nfe=10)


def test_nfe_list_must_be_non_empty() -> None:
    with pytest.raises(InferenceModelError):
        _MinimalAdapter(name="m", device="cpu", nfe_list=(), selection_nfe=1)


def test_resolve_device_cpu_when_no_cuda() -> None:
    if not torch.cuda.is_available():
        assert resolve_device("cuda:0").type == "cpu"
    else:
        # If CUDA is present, the helper should return a CUDA device.
        assert resolve_device("cuda:0").type == "cuda"


def test_inference_result_is_frozen() -> None:
    vol = torch.zeros((2, 2, 2))
    r = InferenceResult(
        t1c_synthetic_harmonised=vol,
        t1c_synthetic_raw=vol,
        inference_seconds=0.0,
        peak_vram_mb=0.0,
    )
    with pytest.raises((AttributeError, TypeError)):
        r.inference_seconds = 1.0  # type: ignore[misc]
