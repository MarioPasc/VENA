"""Test that ``scripts.patch_v4_brain_latent._replay_v4`` is byte-identical
to the brain channel that the bank builder would have produced if the brain
LabelMap had been part of the Subject from day one.

The seed formula in ``bank_builder._variant_seed`` is duplicated in
``scripts.patch_v4_brain_latent`` for the same byte-level reproducibility
guarantee; this test pins that contract.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torchio as tio

from vena.data.augment.offline.bank_builder import _variant_seed as bank_seed
from vena.data.augment.offline.variants import make_variant

pytestmark = pytest.mark.unit


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def patch_module():
    return _load_module(
        "patch_v4_brain_latent",
        Path(__file__).resolve().parents[4] / "scripts" / "patch_v4_brain_latent.py",
    )


def test_variant_seed_matches_bank_builder(patch_module) -> None:
    for base in (1, 42, 99):
        for rank in (0, 1):
            for src_idx in (0, 7, 12345):
                for variant in ("v1", "v2", "v3", "v4"):
                    assert patch_module._variant_seed(base, rank, src_idx, variant) == bank_seed(
                        base, rank, src_idx, variant
                    )


def test_replay_v4_matches_bank_builder_brain(patch_module, tmp_path: Path) -> None:
    """The Subject the bank builder constructs (with brain LabelMap) and the
    Subject patch_v4_brain_latent constructs (brain only) yield byte-identical
    warped brain masks for the same seed.
    """
    # Build a simple planted volume.
    shape = patch_module.AUG_IMAGE_CROP_BOX
    brain = np.zeros(shape, dtype=np.int8)
    brain[20:170, 20:200, 20:170] = 1

    # Bank-builder path: full Subject (images + tumor + brain).
    images = {slug: torch.zeros((1, *shape)) for slug in ("t1pre", "t1c", "t2", "flair")}
    members: dict = {
        slug: tio.ScalarImage(tensor=images[slug]) for slug in ("t1pre", "t1c", "t2", "flair")
    }
    members["tumor"] = tio.LabelMap(tensor=torch.zeros((1, *shape)).long())
    members["brain"] = tio.LabelMap(tensor=torch.from_numpy(brain).unsqueeze(0).long())
    subject_bank = tio.Subject(**members)

    base_seed, rank, src_idx = 42, 0, 100
    variant = "v4"
    v4_cfg = {
        "elastic_num_control_points": 7,
        "elastic_max_displacement": 4.0,
        "elastic_locked_borders": True,
        "elastic_prob": 1.0,
        "affine_scales": [0.9, 1.1],
        "affine_degrees": 10.0,
        "affine_translation_voxels": 8.0,
        "affine_prob": 1.0,
    }
    seed = bank_seed(base_seed, rank, src_idx, variant)
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random as _r

    _r.seed(seed)
    transform = make_variant(variant, v4_cfg)
    out_bank = transform(subject_bank)["brain"].data[0].numpy().astype(np.int8)

    # Replay path: brain only, same seed.
    out_replay = patch_module._replay_v4(brain, (0, 0, 0), seed, v4_cfg)

    assert out_bank.shape == out_replay.shape == shape
    assert np.array_equal(out_bank, out_replay), (
        "seed-replay brain warp must match bank-builder brain warp byte-for-byte"
    )
