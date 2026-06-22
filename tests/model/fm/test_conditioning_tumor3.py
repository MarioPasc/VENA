"""S1 v3 Variant B conditioning: three per-sub-region mask specs.

Verifies that the existing ``ConditioningAssembler`` handles
``mask:netc:identity``, ``mask:ed:identity``, ``mask:et:identity`` without
any new ``kind`` value — the parser auto-reads ``batch["m_netc"]`` etc.
via the existing ``f"m_{key}"`` convention.

The assembler concatenates them along the channel axis ⇒ a 3-channel
conditioning tensor equivalent to reading ``batch["m_tumor"]`` directly.
The S1 v3 spec doc §2.3 mentions a ``tumor3`` kind as a possible
convenience; per the scope-decision in the plan we use three independent
single-channel specs instead — simpler, no new ``kind`` value, no parser
delta. These tests pin that contract.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit

from vena.model.fm.controlnet.conditioning import ConditioningAssembler, ConditioningSpec


def _toy_batch(B: int = 1, h: int = 4, w: int = 4, d: int = 4) -> dict[str, torch.Tensor]:
    """Build a batch with the v3 mask keys."""
    torch.manual_seed(0)
    return {
        "z_t1pre": torch.randn(B, 4, h, w, d),
        "z_t2": torch.randn(B, 4, h, w, d),
        "z_flair": torch.randn(B, 4, h, w, d),
        "m_tumor": torch.rand(B, 3, h, w, d),
        "m_netc": torch.rand(B, 1, h, w, d),
        "m_ed": torch.rand(B, 1, h, w, d),
        "m_et": torch.rand(B, 1, h, w, d),
        "m_brain": torch.ones(B, 1, h, w, d),
        "m_wt": (torch.rand(B, 1, h, w, d) >= 0.5).float(),
    }


def test_three_independent_mask_specs_assemble_to_3_channels() -> None:
    """Three independent ``mask:<key>:identity`` specs ⇒ 3-channel cond tensor."""
    specs = [
        ConditioningSpec(kind="mask", key="netc", downsampler="identity"),
        ConditioningSpec(kind="mask", key="ed", downsampler="identity"),
        ConditioningSpec(kind="mask", key="et", downsampler="identity"),
    ]
    asm = ConditioningAssembler(specs, mask_channels=1)
    assert asm.channels_per_spec == [1, 1, 1]
    assert asm.total_channels == 3

    batch = _toy_batch()
    cond = asm(batch)
    assert cond.shape == torch.Size([1, 3, 4, 4, 4])
    # Channel 0 = m_netc; 1 = m_ed; 2 = m_et — concatenation order matches spec order.
    torch.testing.assert_close(cond[:, 0:1], batch["m_netc"])
    torch.testing.assert_close(cond[:, 1:2], batch["m_ed"])
    torch.testing.assert_close(cond[:, 2:3], batch["m_et"])


def test_three_specs_concatenation_matches_reading_m_tumor_directly() -> None:
    """When m_netc/m_ed/m_et are the slices of m_tumor, the assembler reproduces m_tumor.

    Mirrors the data path where the dataset emits m_netc = m_tumor[0:1] etc.
    """
    specs = [
        ConditioningSpec(kind="mask", key="netc", downsampler="identity"),
        ConditioningSpec(kind="mask", key="ed", downsampler="identity"),
        ConditioningSpec(kind="mask", key="et", downsampler="identity"),
    ]
    asm = ConditioningAssembler(specs, mask_channels=1)
    batch = _toy_batch()
    # Force the per-class views to be exact slices of m_tumor (matches dataset).
    batch["m_netc"] = batch["m_tumor"][:, 0:1]
    batch["m_ed"] = batch["m_tumor"][:, 1:2]
    batch["m_et"] = batch["m_tumor"][:, 2:3]
    cond = asm(batch)
    torch.testing.assert_close(cond, batch["m_tumor"])


def test_three_mask_specs_string_round_trip() -> None:
    """Parsing the YAML-friendly string form is supported by the existing parser."""
    asm = ConditioningAssembler(
        ["mask:netc:identity", "mask:ed:identity", "mask:et:identity"],
        mask_channels=1,
    )
    assert asm.total_channels == 3
    batch = _toy_batch()
    cond = asm(batch)
    assert cond.shape == torch.Size([1, 3, 4, 4, 4])


def test_perturb_keys_zero_only_listed_subregions() -> None:
    """CFG-style dropout on a subset of the masks.

    Useful for ablations that mask one sub-region per training step.
    """
    asm = ConditioningAssembler(
        ["mask:netc:identity", "mask:ed:identity", "mask:et:identity"],
        mask_channels=1,
    )
    batch = _toy_batch()
    cond = asm(batch, perturb_keys={"et"})
    # ET (channel 2) should be zeroed; NETC and ED untouched.
    torch.testing.assert_close(cond[:, 0:1], batch["m_netc"])
    torch.testing.assert_close(cond[:, 1:2], batch["m_ed"])
    torch.testing.assert_close(cond[:, 2:3], torch.zeros_like(batch["m_et"]))


def test_variant_a_empty_conditioning_inputs_rejected() -> None:
    """Variant A has no ControlNet at all ⇒ no ConditioningAssembler built.

    The assembler still requires at least one spec; an empty list is an
    explicit ValueError so a misconfigured Variant A is caught at startup
    rather than producing an empty tensor.
    """
    with pytest.raises(ValueError, match="at least one spec"):
        ConditioningAssembler([], mask_channels=1)
