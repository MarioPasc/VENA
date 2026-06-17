"""Tests for the inference-adapter registry / factory dispatch."""

from __future__ import annotations

import pytest

from vena.inference.base import InferenceModel, InferenceResult
from vena.inference.registry import (
    InferenceRegistryError,
    get_inference_factory,
    list_registered,
    register_inference_model,
)

pytestmark = pytest.mark.unit


def test_built_in_adapter_types_are_registered() -> None:
    """Every adapter under ``vena.inference.adapters`` must self-register."""
    # Importing the package fires the decorators.
    import vena.inference  # noqa: F401

    keys = set(list_registered())
    expected = {
        "identity",
        "pgan",
        "resvit",
        "syndiff",
        "dit_3d",
        "t1c_rflow",
        "lddpm_3d",
        "lpix2pix_3d",
        "vena_fm",
    }
    assert expected.issubset(keys), f"missing: {expected - keys}"


def test_factory_unknown_type_raises() -> None:
    with pytest.raises(InferenceRegistryError):
        get_inference_factory("___does_not_exist___")


def test_duplicate_registration_raises() -> None:
    class _A(InferenceModel):
        def setup(self) -> None:
            super().setup()

        def predict(self, cohort, patient_id, nfe):  # type: ignore[no-untyped-def]
            return InferenceResult(None, None, 0.0, 0.0)  # type: ignore[arg-type]

        def teardown(self) -> None:
            self._is_setup = False

    class _B(InferenceModel):
        def setup(self) -> None:
            super().setup()

        def predict(self, cohort, patient_id, nfe):  # type: ignore[no-untyped-def]
            return InferenceResult(None, None, 0.0, 0.0)  # type: ignore[arg-type]

        def teardown(self) -> None:
            self._is_setup = False

    # Use a key unlikely to collide with built-ins.
    key = "_test_dup_key_xyz"
    register_inference_model(key)(_A)
    with pytest.raises(InferenceRegistryError):
        register_inference_model(key)(_B)
    # Re-registering the same class is a no-op.
    register_inference_model(key)(_A)
