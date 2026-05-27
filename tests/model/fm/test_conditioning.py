"""Unit tests for the ConditioningAssembler."""

from __future__ import annotations

import pytest
import torch

from vena.model.fm.controlnet.conditioning import (
    ConditioningAssembler,
    ConditioningSpec,
)


@pytest.mark.unit
def test_spec_from_string_basic() -> None:
    s = ConditioningSpec.from_string("latent:t1pre")
    assert s.kind == "latent"
    assert s.key == "t1pre"
    assert s.downsampler == "identity"
    assert s.downsampler_kwargs == {}


@pytest.mark.unit
def test_spec_from_string_with_downsampler_kwargs() -> None:
    s = ConditioningSpec.from_string("prior:vessel:trilinear:factor=4:align_corners=false")
    assert s.kind == "prior"
    assert s.key == "vessel"
    assert s.downsampler == "trilinear"
    assert s.downsampler_kwargs == {"factor": 4, "align_corners": False}


@pytest.mark.unit
def test_spec_from_string_rejects_bad_format() -> None:
    with pytest.raises(ValueError):
        ConditioningSpec.from_string("not-a-valid-spec")


@pytest.mark.unit
def test_assembler_total_channels() -> None:
    asm = ConditioningAssembler(
        [
            "latent:t1pre",
            "latent:t2",
            "latent:flair",
            "mask:wt:identity",
        ],
        latent_channels=4,
        mask_channels=1,
    )
    assert asm.total_channels == 13


@pytest.mark.unit
def test_assembler_forward_concat_order() -> None:
    asm = ConditioningAssembler(
        ["latent:t1pre", "latent:t2", "mask:wt:identity"],
        latent_channels=4,
        mask_channels=1,
    )
    B, h, w, d = 2, 60, 60, 40
    z1 = torch.full((B, 4, h, w, d), 1.0)
    z2 = torch.full((B, 4, h, w, d), 2.0)
    m_wt = torch.full((B, 1, h, w, d), 3.0)
    batch = {"z_t1pre": z1, "z_t2": z2, "m_wt": m_wt}

    c = asm(batch)
    assert c.shape == (B, 9, h, w, d)
    # Order: channels [0..3]=z1=1.0, [4..7]=z2=2.0, [8]=m_wt=3.0
    assert torch.allclose(c[:, 0:4], z1)
    assert torch.allclose(c[:, 4:8], z2)
    assert torch.allclose(c[:, 8:9], m_wt)


@pytest.mark.unit
def test_assembler_perturb_zeros_only_designated_mask() -> None:
    asm = ConditioningAssembler(
        ["latent:t1pre", "mask:wt:identity", "mask:brain:identity"],
        latent_channels=4,
        mask_channels=1,
    )
    B, h, w, d = 1, 60, 60, 40
    batch = {
        "z_t1pre": torch.full((B, 4, h, w, d), 1.0),
        "m_wt": torch.full((B, 1, h, w, d), 0.7),
        "m_brain": torch.full((B, 1, h, w, d), 0.9),
    }
    c_perturb = asm(batch, perturb_keys={"wt"})
    assert torch.allclose(c_perturb[:, 0:4], torch.full((B, 4, h, w, d), 1.0))
    assert torch.allclose(c_perturb[:, 4:5], torch.zeros((B, 1, h, w, d)))
    assert torch.allclose(c_perturb[:, 5:6], torch.full((B, 1, h, w, d), 0.9))


@pytest.mark.unit
def test_assembler_raises_on_missing_key() -> None:
    asm = ConditioningAssembler(["latent:t1pre"], latent_channels=4)
    with pytest.raises(KeyError, match="z_t1pre"):
        asm({})
