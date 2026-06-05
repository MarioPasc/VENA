"""Registry unit tests for the PEFT adapter layer."""

from __future__ import annotations

from typing import Any

import pytest

from vena.model.fm.maisi.peft import (
    BasePEFT,
    LoRA,
    PeftConfigError,
    UnknownVariantError,
    build_peft,
    list_variants,
    register_peft,
)

pytestmark = pytest.mark.unit


def test_lora_is_registered() -> None:
    assert "lora" in list_variants()


def test_build_peft_lora_defaults() -> None:
    handler = build_peft("lora", {"r": 16})
    assert isinstance(handler, LoRA)
    payload = handler.to_dict()
    assert payload["variant"] == "lora"
    assert payload["params"]["r"] == 16
    assert payload["params"]["alpha"] == 16  # default mirrors r
    assert payload["params"]["target_modules"] == ["to_q", "to_k", "to_v", "out_proj"]


def test_build_peft_lora_full_params() -> None:
    handler = build_peft(
        "lora",
        {
            "r": 8,
            "alpha": 32,
            "dropout": 0.1,
            "target_modules": ["to_q", "to_v"],
            "bias": "lora_only",
            "init_lora_weights": "gaussian",
        },
    )
    payload = handler.to_dict()["params"]
    assert payload["r"] == 8
    assert payload["alpha"] == 32
    assert payload["dropout"] == pytest.approx(0.1)
    assert payload["target_modules"] == ["to_q", "to_v"]
    assert payload["bias"] == "lora_only"


def test_build_peft_unknown_variant() -> None:
    with pytest.raises(UnknownVariantError):
        build_peft("does-not-exist", {})


def test_build_peft_lora_rejects_unknown_keys() -> None:
    with pytest.raises(PeftConfigError):
        build_peft("lora", {"r": 16, "bogus_key": 1})


def test_build_peft_lora_rejects_bad_bias() -> None:
    with pytest.raises(PeftConfigError):
        build_peft("lora", {"r": 16, "bias": "not-an-option"})


def test_build_peft_lora_rejects_zero_rank() -> None:
    with pytest.raises(PeftConfigError):
        build_peft("lora", {"r": 0})


def test_build_peft_lora_rejects_empty_targets() -> None:
    with pytest.raises(PeftConfigError):
        build_peft("lora", {"r": 16, "target_modules": []})


def test_register_peft_collision_rejected() -> None:
    class _Other(BasePEFT):
        @classmethod
        def from_dict(cls, params: dict[str, Any]) -> _Other:
            return cls()

        def apply(self, trunk):  # type: ignore[override]
            return trunk

        def trainable_parameters(self, trunk):  # type: ignore[override]
            return []

        def extract_state(self, trunk):  # type: ignore[override]
            return {}

        def load_state(self, trunk, state):  # type: ignore[override]
            return None

        def to_dict(self):  # type: ignore[override]
            return {"variant": "lora"}

    with pytest.raises(ValueError):
        register_peft("lora")(_Other)
