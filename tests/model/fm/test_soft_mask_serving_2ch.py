"""Tests for soft 2-channel mask serving and conditioning wiring (task-20 / S2 T-13).

Covers:
- ``_read_soft_masks`` helper (oracle_soft / missing / none / derived / invalid).
- ``ConditioningAssembler`` two-spec channel-count contract (A.8-§4).
- Batch-key wiring between data-serving keys and assembler lookup keys.

All tests are CPU-only; no MAISI checkpoints required.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.fm


# ── Fixture helpers ──────────────────────────────────────────────────────────


def _make_latent_h5(
    tmp_path: Path,
    *,
    add_soft_group: bool = True,
    h: int = 4,
    w: int = 4,
    d: int = 4,
) -> tuple[Path, np.ndarray, np.ndarray]:
    """Write a minimal synthetic latent H5 for mask-serving tests.

    Returns
    -------
    h5_path : Path
    m_netc_np : np.ndarray, shape (1, h, w, d)
    m_et_np   : np.ndarray, shape (1, h, w, d)
    """
    rng = np.random.default_rng(0)
    # tumor_latent: 3-ch (NETC, ED, ET), always present
    tumor = rng.random((1, 3, h, w, d), dtype=float).astype(np.float32)
    m_netc = tumor[:, 0:1]  # (1, h, w, d)
    m_et = tumor[:, 2:3]

    h5_path = tmp_path / "test_latent.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("masks/tumor_latent", data=tumor)
        if add_soft_group:
            # tumor_latent_soft: 2-ch (TC, NETC); TC = clip(NETC+ET, 0, 1) ≥ NETC
            tc_soft = np.clip(m_netc + m_et, 0.0, 1.0)
            soft = np.concatenate([tc_soft, m_netc], axis=1).astype(np.float32)
            f.create_dataset("masks/tumor_latent_soft", data=soft)
    return h5_path, m_netc[0:1], m_et[0:1]  # squeeze leading N dim for helper args


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_oracle_soft_present_serves_keys(tmp_path: Path) -> None:
    """oracle_soft reads masks/tumor_latent_soft → m_tc_soft + m_netc_soft.

    Shape contract: each key (1, H, W, D) float32 in [0, 1] with
    m_netc_soft <= m_tc_soft (TC includes NETC and ET; NETC-only is a subset).
    """
    from vena.model.fm.lightning.data import _read_soft_masks

    h5_path, _, _ = _make_latent_h5(tmp_path, add_soft_group=True)
    with h5py.File(h5_path, "r") as h5:
        out = _read_soft_masks(h5, row=0, h5_path=h5_path, mask_source="oracle_soft")

    assert set(out.keys()) == {"m_tc_soft", "m_netc_soft"}
    for key in ("m_tc_soft", "m_netc_soft"):
        t = out[key]
        assert t.shape == (1, 4, 4, 4), f"{key}: expected (1,4,4,4), got {t.shape}"
        assert t.dtype == torch.float32
        assert float(t.min()) >= 0.0
        assert float(t.max()) <= 1.0

    # TC ≥ NETC elementwise (TC = NETC + ET, ET ≥ 0 ⇒ TC ≥ NETC)
    assert (out["m_tc_soft"] >= out["m_netc_soft"]).all()


def test_oracle_soft_missing_raises(tmp_path: Path) -> None:
    """Absent masks/tumor_latent_soft raises MissingSoftMaskGroupError.

    The group name must appear in the error message so operators know which
    cache step is missing. Zeros / warnings are both forbidden.
    """
    from vena.model.fm.lightning.data import MissingSoftMaskGroupError, _read_soft_masks

    h5_path, _, _ = _make_latent_h5(tmp_path, add_soft_group=False)
    with h5py.File(h5_path, "r") as h5:
        with pytest.raises(MissingSoftMaskGroupError, match="masks/tumor_latent_soft"):
            _read_soft_masks(h5, row=0, h5_path=h5_path, mask_source="oracle_soft")


def test_none_source_returns_empty_dict(tmp_path: Path) -> None:
    """mask_source='none' → empty dict; back-compat with all prior run YAMLs."""
    from vena.model.fm.lightning.data import _read_soft_masks

    h5_path, _, _ = _make_latent_h5(tmp_path, add_soft_group=True)
    with h5py.File(h5_path, "r") as h5:
        out = _read_soft_masks(h5, row=0, h5_path=h5_path, mask_source="none")

    assert out == {}


def test_derived_source_builds_from_arrays(tmp_path: Path) -> None:
    """mask_source='derived' computes m_tc_soft=clip(NETC+ET,0,1) from pre-read arrays.

    The soft group does NOT need to exist — derived is the aug-H5 fallback.
    """
    from vena.model.fm.lightning.data import _read_soft_masks

    sz = 4
    rng = np.random.default_rng(1)
    m_netc_np = rng.random((1, sz, sz, sz)).astype(np.float32)
    m_et_np = rng.random((1, sz, sz, sz)).astype(np.float32)
    expected_tc = np.clip(m_netc_np + m_et_np, 0.0, 1.0)

    # H5 without the oracle_soft group — derived must not attempt to read it.
    h5_path, _, _ = _make_latent_h5(tmp_path, add_soft_group=False, h=sz, w=sz, d=sz)
    with h5py.File(h5_path, "r") as h5:
        out = _read_soft_masks(
            h5,
            row=0,
            h5_path=h5_path,
            mask_source="derived",
            m_netc_np=m_netc_np,
            m_et_np=m_et_np,
        )

    assert set(out.keys()) == {"m_tc_soft", "m_netc_soft"}
    np.testing.assert_allclose(out["m_tc_soft"].numpy(), expected_tc, rtol=1e-5)
    np.testing.assert_allclose(out["m_netc_soft"].numpy(), m_netc_np, rtol=1e-5)
    assert out["m_tc_soft"].shape == (1, sz, sz, sz)
    assert out["m_netc_soft"].shape == (1, sz, sz, sz)


def test_invalid_mask_source_raises(tmp_path: Path) -> None:
    """An unrecognised mask_source string raises ValueError immediately."""
    from vena.model.fm.lightning.data import _read_soft_masks

    h5_path, _, _ = _make_latent_h5(tmp_path, add_soft_group=True)
    with h5py.File(h5_path, "r") as h5:
        with pytest.raises(ValueError, match="Unknown mask_source"):
            _read_soft_masks(h5, row=0, h5_path=h5_path, mask_source="bad_value")


def test_two_spec_assembler_total_channels_equals_2() -> None:
    """Two 1-ch mask specs → mask-part total_channels == 2 (A.8-§4).

    This is the correct wiring for [TC, NETC] 2-channel conditioning:
        conditioning_inputs:
          - mask:tc_soft:identity
          - mask:netc_soft:identity

    A single 2-ch spec key would give total_channels == 1 (the assembler
    uses mask_channels=1 per spec, not the runtime tensor shape).
    """
    from vena.model.fm.controlnet.conditioning import ConditioningAssembler

    asm = ConditioningAssembler(
        specs=["mask:tc_soft:identity", "mask:netc_soft:identity"],
        mask_channels=1,
    )
    assert asm.total_channels == 2, (
        f"Expected total_channels=2 for two 1-ch mask specs, got {asm.total_channels}"
    )
    assert asm.channels_per_spec == [1, 1]


def test_assembler_batch_keys_match_data_serving_keys() -> None:
    """Assembler batch keys align exactly with the keys _read_soft_masks produces.

    The data path produces ``batch["m_tc_soft"]`` and ``batch["m_netc_soft"]``.
    The assembler must look up those same keys, not m_wt_soft or similar.
    This test pins the wiring so a rename in one place fails the other.
    """
    from vena.model.fm.controlnet.conditioning import ConditioningAssembler, ConditioningSpec

    specs = [
        ConditioningSpec.from_string("mask:tc_soft:identity"),
        ConditioningSpec.from_string("mask:netc_soft:identity"),
    ]
    expected_keys = {"m_tc_soft", "m_netc_soft"}
    assembler_keys = {spec.batch_key() for spec in specs}

    assert assembler_keys == expected_keys, (
        f"Assembler batch keys {assembler_keys} do not match data-serving keys {expected_keys}"
    )

    # Cross-check: assembler.forward would raise KeyError on a batch that
    # provides e.g. m_wt_soft instead of m_tc_soft.
    asm = ConditioningAssembler(specs=specs, mask_channels=1)
    sz = 4
    good_batch = {
        "m_tc_soft": torch.zeros(1, 1, sz, sz, sz),
        "m_netc_soft": torch.zeros(1, 1, sz, sz, sz),
    }
    out = asm.forward(good_batch)
    assert out.shape == (1, 2, sz, sz, sz)

    wrong_batch = {
        "m_wt_soft": torch.zeros(1, 1, sz, sz, sz),  # wrong key
        "m_netc_soft": torch.zeros(1, 1, sz, sz, sz),
    }
    with pytest.raises(KeyError, match="m_tc_soft"):
        asm.forward(wrong_batch)


def test_step0_output_scale_zero_exact_zeros() -> None:
    """output_scale=0 gates ALL residuals to exactly 0.0, not approximately.

    Proof: ``MaisiControlNet.forward`` computes::

        down_block_res_samples[i] = raw_i * output_scale
        mid_block_res_sample = raw_mid * output_scale

    where ``output_scale`` is a scalar buffer.  When that buffer is 0, every
    element is the IEEE-754 product of a finite float and 0.0, which is
    exactly 0.0 for all finite operands.

    Two implications:

    1. The ControlNet is a no-op at step 0.  The trunk forward is identical to
       the pretrained-only baseline — the warm-start identity required before
       ``OutputScaleRampCallback`` begins ramping the buffer.
    2. ``torch.equal(trunk_hidden + zeros, trunk_hidden)`` holds exactly
       (IEEE-754 float32: x + 0.0 = x), so the trunk hidden state is
       byte-identical whether or not the ControlNet residuals are added.

    The control arm (``output_scale=1.0`` with perturbed projections) confirms
    that ``torch.count_nonzero == 0`` at scale=0 is not a trivial consequence
    of zero-init weights — the residuals ARE non-zero during training.
    """
    from vena.model.fm.controlnet.maisi_controlnet import MaisiControlNet

    tiny_overrides: dict[str, object] = {
        "num_channels": [8, 8],
        "attention_levels": [False, False],
        "num_head_channels": [4, 4],
        "num_res_blocks": 1,
        "use_flash_attention": False,
        "conditioning_embedding_num_channels": [8],
        "num_class_embeds": None,
        "resblock_updown": False,
        "include_fc": False,
        "norm_num_groups": 4,
    }
    cn = MaisiControlNet(conditioning_in_channels=2, arch_overrides=tiny_overrides)
    cn.eval()

    # Simulate post-training: perturb zero-init output projections so that
    # scale=1.0 produces genuine non-zero residuals (not all-zero by zero-init).
    with torch.no_grad():
        for name, p in cn.net.named_parameters():
            if name.startswith("controlnet_down_blocks.") or name.startswith(
                "controlnet_mid_block."
            ):
                p.normal_(0, 0.1)

    x = torch.randn(1, 4, 2, 2, 2)
    ts = torch.tensor([500], dtype=torch.long)
    cond = torch.randn(1, 2, 2, 2, 2)

    # Guard: conditioning tensor and output-projection weights are non-zero,
    # so the scale=1.0 residuals are genuine and not an artifact of zero inputs.
    out_proj_max = max(
        p.abs().max().item()
        for name, p in cn.net.named_parameters()
        if name.startswith("controlnet_down_blocks.") or name.startswith("controlnet_mid_block.")
    )
    assert cond.abs().max().item() > 1e-6, "conditioning tensor must be non-zero"
    assert out_proj_max > 1e-6, "output projections must be non-zero (simulating post-training)"

    # Control: output_scale=1.0 → at least one residual element is non-zero.
    cn.output_scale.fill_(1.0)
    with torch.no_grad():
        downs1, mid1 = cn(x, ts, cond)
    assert any(torch.count_nonzero(r) > 0 for r in [*downs1, mid1]), (
        "control arm: scale=1 with perturbed weights must produce at least one non-zero residual"
    )

    # Test: output_scale=0 → EVERY element of EVERY residual is exactly 0.0.
    cn.output_scale.fill_(0.0)
    with torch.no_grad():
        downs0, mid0 = cn(x, ts, cond)
    max_abs = max(r.abs().max().item() for r in [*downs0, mid0])
    for i, r in enumerate([*downs0, mid0]):
        nz = torch.count_nonzero(r).item()
        assert nz == 0, (
            f"residual[{i}] has {nz} non-zero elements at output_scale=0; "
            f"expected exactly 0 (measured max_abs={max_abs:.6f})"
        )
    # Measured max-abs at output_scale=0: 0.0 (reported for auditing).
    assert max_abs == 0.0

    # Trunk-add identity: IEEE-754 float32 satisfies x + 0.0 = x exactly.
    # torch.equal (not allclose) is correct here: allclose has a nonzero atol
    # and would pass even if residuals were tiny but non-zero.
    trunk_hidden = torch.randn_like(downs0[0])
    assert torch.equal(trunk_hidden + downs0[0], trunk_hidden), (
        "trunk_hidden + zero_residual != trunk_hidden: IEEE-754 x+0.0=x invariant violated"
    )


def test_both_dataset_paths_agree_on_soft_masks(tmp_path: Path) -> None:
    """LatentH5Dataset and OfflineAugmentedLatentH5Dataset._read_aug agree on soft masks.

    Both code paths route through the shared ``_read_soft_masks`` helper, making
    divergence structurally impossible today.  This test acts as a CI trip-wire:
    if a future edit replaces one call site with bespoke logic that forgets to
    update the other, ``torch.equal`` on ``m_tc_soft`` / ``m_netc_soft`` fails.

    Uses ``mask_source="derived"`` (TC = clip(NETC + ET, 0, 1)) which does not
    require a ``masks/tumor_latent_soft`` group — aug H5s never carry that group,
    as documented in ``OfflineAugmentedLatentH5Dataset``.
    """
    from vena.model.fm.lightning.data import (
        LatentH5Dataset,
        OfflineAugmentedLatentH5Dataset,
    )

    sz = 4
    rng = np.random.default_rng(42)
    # 3-channel tumor latent (NETC, ED, ET) in [0, 1]; same data in both H5s.
    tumor_np = rng.random((1, 3, sz, sz, sz)).astype(np.float32)
    latent_np = rng.standard_normal((1, 4, sz, sz, sz)).astype(np.float32)
    str_dtype = h5py.special_dtype(vlen=str)

    # Clean H5: one patient, one scan.
    clean_h5 = tmp_path / "clean.h5"
    with h5py.File(clean_h5, "w") as f:
        f.create_dataset("ids", data=np.array(["p0"], dtype=object), dtype=str_dtype)
        f.create_dataset("latents/t1c", data=latent_np)
        f.create_dataset("masks/tumor_latent", data=tumor_np)

    # Aug H5: same patient, variant "v1", same tumor mask data.
    aug_h5 = tmp_path / "aug.h5"
    with h5py.File(aug_h5, "w") as f:
        f.create_dataset("ids", data=np.array(["p0"], dtype=object), dtype=str_dtype)
        f.create_dataset("variants", data=np.array(["v1"], dtype=object), dtype=str_dtype)
        f.create_dataset("latents/t1c", data=latent_np)
        f.create_dataset("masks/tumor_latent", data=tumor_np)

    # Read via the clean path.
    clean_ds = LatentH5Dataset(
        clean_h5,
        patient_ids=["p0"],
        latents=["t1c"],
        mask_source="derived",
    )
    clean_item = clean_ds[0]

    # Read via the aug path with variant_weights={"v1": 1.0} — forces _read_aug.
    aug_ds = OfflineAugmentedLatentH5Dataset(
        clean_h5_path=clean_h5,
        aug_h5_path=aug_h5,
        patient_ids=["p0"],
        variant_weights={"v1": 1.0},
        latents=["t1c"],
        mask_source="derived",
    )
    aug_item = aug_ds[0]

    # Confirm the aug path was taken (not the v0 clean fallback).
    assert aug_item.get("_aug_variant") == "v1", (
        f"Expected aug path 'v1' but got {aug_item.get('_aug_variant')!r}"
    )

    # Both masks must be bit-identical — same tumor_np through the same formula.
    assert torch.equal(clean_item["m_tc_soft"], aug_item["m_tc_soft"]), (
        "m_tc_soft differs between clean and aug paths — one path diverged from _read_soft_masks"
    )
    assert torch.equal(clean_item["m_netc_soft"], aug_item["m_netc_soft"]), (
        "m_netc_soft differs between clean and aug paths — one path diverged from _read_soft_masks"
    )
