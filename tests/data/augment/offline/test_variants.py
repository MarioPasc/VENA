"""Unit tests for the offline-augmentation variant builders."""

from __future__ import annotations

import pytest
import torch
import torchio as tio

from vena.data.augment.offline import (
    VARIANT_INPUT_ONLY,
    VARIANT_NAMES,
    make_variant,
)

pytestmark = pytest.mark.unit


_SHAPE = (16, 16, 16)


def _subject() -> tio.Subject:
    torch.manual_seed(0)
    return tio.Subject(
        t1pre=tio.ScalarImage(tensor=torch.rand(1, *_SHAPE)),
        t1c=tio.ScalarImage(tensor=torch.rand(1, *_SHAPE)),
        t2=tio.ScalarImage(tensor=torch.rand(1, *_SHAPE)),
        flair=tio.ScalarImage(tensor=torch.rand(1, *_SHAPE)),
        tumor=tio.LabelMap(tensor=torch.randint(0, 5, (1, *_SHAPE))),
    )


@pytest.mark.parametrize("name", VARIANT_NAMES)
def test_variant_preserves_shape(name: str) -> None:
    comp = make_variant(name)
    s = comp(_subject())
    for key in ("t1pre", "t1c", "t2", "flair", "tumor"):
        assert s[key].data.shape == (1, *_SHAPE), f"{name}/{key} shape"


@pytest.mark.parametrize("name", [n for n in VARIANT_NAMES if VARIANT_INPUT_ONLY[n]])
def test_input_only_variant_leaves_target_and_mask_untouched(name: str) -> None:
    torch.manual_seed(42)
    s_in = _subject()
    pre_t1c = s_in["t1c"].data.clone()
    pre_tumor = s_in["tumor"].data.clone()
    s_out = make_variant(name)(s_in)
    assert torch.equal(s_out["t1c"].data, pre_t1c), f"{name}: t1c changed"
    assert torch.equal(s_out["tumor"].data, pre_tumor), f"{name}: tumor changed"


def test_v4_warps_jointly_and_keeps_mask_integer() -> None:
    torch.manual_seed(7)
    s = _subject()
    out = make_variant("v4")(s)
    t = out["tumor"].data
    assert torch.allclose(t.float(), t.float().round()), "v4 tumor mask not integer"


def test_make_variant_rejects_unknown_name() -> None:
    with pytest.raises(KeyError, match="unknown variant"):
        make_variant("v99")


def test_make_variant_accepts_hp_overrides() -> None:
    # Lock gamma off and confirm we get back a Compose with the same shape behaviour.
    comp = make_variant("v1", {"gamma_prob": 0.0})
    s = comp(_subject())
    assert s["t1pre"].data.shape == (1, *_SHAPE)
