"""Test that ``OfflineAugBankBuilder._build_subject`` exposes a brain LabelMap.

The 2026-06-18 audit found ``masks/brain_latent[v4]`` was synth-ones because
the brain mask was absent from the TorchIO Subject — the elastic+affine warp
therefore did not touch it, and the brain-to-latent post-pass had nothing
to consume. Phase 0.2 of the fix-up:

* Adds ``brain`` as a ``tio.LabelMap`` member of the Subject.
* Persists the warped brain into ``masks/brain`` of the aug-image H5
  manifest.

These tests exercise the Subject builder directly (no I/O, no GPU).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torchio as tio

from vena.data.augment.offline.bank_builder import OfflineAugBankBuilder
from vena.data.augment.offline.variants import make_variant

pytestmark = pytest.mark.unit


def _stub_builder(tmp_path: Path) -> OfflineAugBankBuilder:
    return OfflineAugBankBuilder(
        source_image_h5=tmp_path / "src.h5",
        output_path=tmp_path / "out.h5",
        cohort="UCSF-PDGM",
        modalities=["t1pre", "t1c", "t2", "flair"],
        variants=["v1", "v2", "v3", "v4"],
        aug_config_json="{}",
        aug_config_sha256="0" * 64,
        world_size=1,
        rank=0,
        seed=1337,
    )


def _planted_volumes(shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
    return {slug: np.ones(shape, dtype=np.float32) for slug in ("t1pre", "t1c", "t2", "flair")}


def _planted_tumor(shape: tuple[int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.int8)
    mask[10:20, 10:20, 10:20] = 1
    return mask


def _planted_brain(shape: tuple[int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.int8)
    mask[5:25, 5:25, 5:25] = 1
    return mask


def test_subject_includes_brain_when_provided(tmp_path: Path) -> None:
    builder = _stub_builder(tmp_path)
    shape = (32, 32, 32)
    subject = builder._build_subject(
        volumes_boxed=_planted_volumes(shape),
        mask_boxed=_planted_tumor(shape),
        brain_boxed=_planted_brain(shape),
    )
    assert "brain" in subject
    assert isinstance(subject["brain"], tio.LabelMap)
    assert subject["brain"].data.shape == (1, *shape)


def test_subject_omits_brain_when_none(tmp_path: Path) -> None:
    builder = _stub_builder(tmp_path)
    shape = (32, 32, 32)
    subject = builder._build_subject(
        volumes_boxed=_planted_volumes(shape),
        mask_boxed=_planted_tumor(shape),
        brain_boxed=None,
    )
    assert "brain" not in subject


def test_v4_warps_brain_jointly_with_images(tmp_path: Path) -> None:
    builder = _stub_builder(tmp_path)
    shape = (32, 32, 32)
    brain = _planted_brain(shape)
    subject = builder._build_subject(
        volumes_boxed=_planted_volumes(shape),
        mask_boxed=_planted_tumor(shape),
        brain_boxed=brain,
    )
    torch.manual_seed(0)
    transform = make_variant("v4")
    augmented = transform(subject)
    warped_brain = augmented["brain"].data[0].numpy().astype(np.int8)
    # Same shape, valid label set ({0, 1}), and some non-trivial change
    # (the elastic + affine warp must have moved voxels).
    assert warped_brain.shape == shape
    assert set(np.unique(warped_brain).tolist()).issubset({0, 1})
    # Joint sum can change because affine introduces zero-padding at the
    # boundary — accept any non-zero delta as evidence the warp fired.
    assert np.any(warped_brain != brain)


def test_v1_keeps_brain_unchanged(tmp_path: Path) -> None:
    builder = _stub_builder(tmp_path)
    shape = (32, 32, 32)
    brain = _planted_brain(shape)
    subject = builder._build_subject(
        volumes_boxed=_planted_volumes(shape),
        mask_boxed=_planted_tumor(shape),
        brain_boxed=brain,
    )
    torch.manual_seed(0)
    transform = make_variant("v1")
    augmented = transform(subject)
    out_brain = augmented["brain"].data[0].numpy().astype(np.int8)
    # v1 is intensity-only — every label-map member is copied verbatim.
    assert np.array_equal(out_brain, brain)
