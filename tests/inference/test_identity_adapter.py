"""End-to-end test for the C0-Identity adapter on a synthetic cohort."""

from __future__ import annotations

import pytest

from vena.inference.adapters.identity_adapter import IdentityAdapter

pytestmark = pytest.mark.unit


def test_identity_setup_predict_teardown(synthetic_cohort) -> None:
    cohort, _image_h5 = synthetic_cohort
    adapter = IdentityAdapter(name="C0-Identity", device="cpu", nfe_list=(1,), selection_nfe=1)
    adapter.setup()
    try:
        result = adapter.predict(cohort, "P001", nfe=1)
        assert result.t1c_synthetic_harmonised.shape == (16, 16, 16)
        # Within brain mask, the harmonised volume must be in [0, 1].
        h = result.t1c_synthetic_harmonised
        assert float(h.min()) >= 0.0
        assert float(h.max()) <= 1.0 + 1e-6
        # Inference time should be negligible.
        assert result.inference_seconds < 5.0
    finally:
        adapter.teardown()


def test_identity_predict_before_setup_raises(synthetic_cohort) -> None:
    from vena.inference.base import InferenceModelError

    cohort, _ = synthetic_cohort
    adapter = IdentityAdapter(name="C0", device="cpu", nfe_list=(1,), selection_nfe=1)
    with pytest.raises(InferenceModelError):
        adapter.predict(cohort, "P001", nfe=1)
