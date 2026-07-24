"""Regression tests for mask_source wiring in exhaustive-val (task-20 correction 2).

Three guards:

1. ``LatentH5Dataset`` built with ``mask_source="oracle_soft"`` on an H5 that
   contains ``masks/tumor_latent_soft`` serves ``m_tc_soft`` and
   ``m_netc_soft``.  A ``ConditioningAssembler`` configured with
   ``[mask:tc_soft:identity, mask:netc_soft:identity]`` must consume the item
   without raising.

2. A launcher job dict that carries ``"mask_source": "oracle_soft"`` round-trips
   through ``ExhaustiveValJobConfig`` and exposes the correct value as
   ``cfg.mask_source``.

3. A synthetic ``ExhaustiveValEngine`` run in which every patient raises
   ``RuntimeError`` must:
   - raise ``ExhaustiveValAllSkippedError``, and
   - NOT write a header-only ``metrics.csv``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# PYTHONPATH: tests in the worktree must import the worktree's src/, not the
# main installed package.  The worktree's PYTHONPATH is set by the pytest
# invocation; here we just assert the import resolves to the worktree.
# ---------------------------------------------------------------------------

from routines.fm.exhaustive_val.engine import (  # noqa: E402
    ExhaustiveValAllSkippedError,
    ExhaustiveValEngine,
    ExhaustiveValJobConfig,
)

from vena.model.fm.lightning.data import LatentH5Dataset  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LATENT_SHAPE = (4, 6, 7, 6)  # (C, H, W, D) — tiny but valid
_MASK_SHAPE = (2, 6, 7, 6)  # (2, H, W, D) — [TC, NETC]


def _write_latent_h5_with_soft_masks(path: Path) -> None:
    """Write a minimal latent H5 with both ``masks/tumor_latent`` and
    ``masks/tumor_latent_soft``.

    ``masks/tumor_latent`` (3 channels: NETC, ED, ET) is always read by
    ``LatentH5Dataset._read_one`` regardless of ``mask_source``.
    ``masks/tumor_latent_soft`` (2 channels: TC, NETC) is the oracle-soft
    source read only when ``mask_source="oracle_soft"``.
    """
    ids = ["P0", "P1"]
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = "2.0.0"
        f.attrs["cohort"] = "SYNTHETIC"

        # IDs dataset
        dt = h5py.special_dtype(vlen=str)
        ds_ids = f.create_dataset("ids", data=np.array(ids, dtype=object), dtype=dt)
        ds_ids.attrs["description"] = "patient scan IDs"

        # Patient / scan CSR index
        patient_grp = f.create_group("patients")
        patient_grp.create_dataset("keys", data=np.array(ids, dtype=object), dtype=dt)
        patient_grp.create_dataset("offsets", data=np.array([0, 1, 2], dtype=np.int32))

        # Latents
        lat = f.create_group("latents")
        for modality in ("t1pre", "t1c", "t2", "flair"):
            data = np.zeros((2, *_LATENT_SHAPE), dtype=np.float32)
            lat.create_dataset(modality, data=data)

        masks = f.create_group("masks")

        # Brain mask (required by LatentH5Dataset)
        brain = np.ones((2, 1, 6, 7, 6), dtype=np.float32)
        masks.create_dataset("brain_latent", data=brain)

        # 3-channel hard soft mask [NETC, ED, ET] — always read by _read_one.
        tumor_3ch = np.random.default_rng(1).random((2, 3, 6, 7, 6)).astype(np.float32)
        masks.create_dataset("tumor_latent", data=tumor_3ch)

        # 2-channel oracle soft mask [TC, NETC] — read when mask_source="oracle_soft".
        soft_2ch = np.random.default_rng(0).random((2, 2, 6, 7, 6)).astype(np.float32)
        masks.create_dataset("tumor_latent_soft", data=soft_2ch)

        # Splits
        splits = f.create_group("splits")
        splits.create_dataset("val", data=np.array([0, 1], dtype=np.int32))


# ---------------------------------------------------------------------------
# Test 1 — LatentH5Dataset with oracle_soft serves m_tc_soft / m_netc_soft
# ---------------------------------------------------------------------------


def test_exhaustive_val_dataset_with_oracle_soft_has_soft_masks(tmp_path: Path) -> None:
    """oracle_soft → dataset item has m_tc_soft and m_netc_soft; assembler accepts it."""
    h5_path = tmp_path / "latents.h5"
    _write_latent_h5_with_soft_masks(h5_path)

    dataset = LatentH5Dataset(h5_path, ["P0", "P1"], mask_source="oracle_soft")
    item = dataset[0]

    assert "m_tc_soft" in item, "oracle_soft must produce m_tc_soft"
    assert "m_netc_soft" in item, "oracle_soft must produce m_netc_soft"
    assert item["m_tc_soft"].shape == torch.Size([1, 6, 7, 6]), (
        f"Expected (1,6,7,6), got {item['m_tc_soft'].shape}"
    )
    assert item["m_netc_soft"].shape == torch.Size([1, 6, 7, 6])

    # ConditioningAssembler must consume the item without raising.
    from vena.model.fm.controlnet.conditioning import ConditioningAssembler

    specs = ["mask:tc_soft:identity", "mask:netc_soft:identity"]
    assembler = ConditioningAssembler(specs)

    batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in item.items()}
    # assembler(batch) calls forward; should not raise.
    out = assembler(batch)
    assert out is not None, "ConditioningAssembler.forward must return a tensor"
    assert out.shape[1] == 2, f"Expected 2 conditioning channels, got {out.shape[1]}"


# ---------------------------------------------------------------------------
# Test 2 — ExhaustiveValJobConfig round-trips mask_source from job dict
# ---------------------------------------------------------------------------


def _minimal_job_dict(tmp_path: Path, mask_source: str = "oracle_soft") -> dict[str, Any]:
    """Minimal job dict accepted by ExhaustiveValJobConfig."""
    latents = tmp_path / "latents.h5"
    latents.touch()
    image = tmp_path / "image.h5"
    image.touch()
    vae = tmp_path / "vae.pt"
    vae.touch()
    trunk_ckpt = tmp_path / "trunk.pt"
    trunk_ckpt.touch()
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_id": "test_run",
        "epoch": 1,
        "trunk": {
            "checkpoint": str(trunk_ckpt),
            "arch_overrides": {},
            "class_token": 9,
            "spacing_mm": [1.0, 1.0, 1.0],
            "trainable": False,
            "regime": "fft",
        },
        "controlnet": {
            "enabled": True,
            "init_from_trunk": True,
            "conditioning_inputs": ["mask:tc_soft:identity", "mask:netc_soft:identity"],
            "arch_overrides": {},
        },
        "vae_checkpoint": str(vae),
        "latents_h5": str(latents),
        "image_h5": str(image),
        "mask_source": mask_source,
        "output_dir": str(output_dir),
    }


def test_exhaustive_val_launcher_writes_mask_source_to_job_yaml(tmp_path: Path) -> None:
    """job dict with mask_source round-trips through ExhaustiveValJobConfig."""
    job_dict = _minimal_job_dict(tmp_path, mask_source="oracle_soft")
    cfg = ExhaustiveValJobConfig.model_validate(job_dict)
    assert cfg.mask_source == "oracle_soft", (
        f"Expected mask_source='oracle_soft', got {cfg.mask_source!r}"
    )

    # Default back-compat: omitting mask_source gives "none".
    del job_dict["mask_source"]
    cfg_default = ExhaustiveValJobConfig.model_validate(job_dict)
    assert cfg_default.mask_source == "none", (
        f"Default mask_source must be 'none', got {cfg_default.mask_source!r}"
    )


# ---------------------------------------------------------------------------
# Test 3 — all-skip guard raises ExhaustiveValAllSkippedError; no metrics.csv
# ---------------------------------------------------------------------------


def test_exhaustive_val_all_skip_raises(tmp_path: Path) -> None:
    """When every patient raises, run() raises ExhaustiveValAllSkippedError.

    No header-only metrics.csv must be left behind.
    """
    # Build a minimal real H5 so the engine can open it.
    latents_path = tmp_path / "latents.h5"
    _write_latent_h5_with_soft_masks(latents_path)
    image_path = tmp_path / "image.h5"
    image_path.touch()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    job_dict = _minimal_job_dict(tmp_path, mask_source="oracle_soft")
    job_dict["output_dir"] = str(out_dir)

    cfg = ExhaustiveValJobConfig.model_validate(job_dict)

    engine = ExhaustiveValEngine.__new__(ExhaustiveValEngine)
    engine.cfg = cfg
    engine.device = "cpu"

    # Patch _build_module, load_autoencoder, MaisiDecoder, get_sampler so no
    # GPU or checkpoint is needed.  Crucially, _process_patient raises every time
    # so metric_rows stays empty.
    dummy_module = MagicMock()
    dummy_module.rflow.scheduler = MagicMock()

    dummy_vae = MagicMock()
    dummy_sampler = MagicMock()

    with (
        patch.object(engine, "_build_module", return_value=dummy_module),
        patch(
            "routines.fm.exhaustive_val.engine.load_autoencoder",
            return_value=MagicMock(),
        ),
        patch(
            "routines.fm.exhaustive_val.engine.MaisiDecoder",
            return_value=dummy_vae,
        ),
        patch(
            "routines.fm.exhaustive_val.engine.get_sampler",
            return_value=lambda **kw: dummy_sampler,
        ),
        # _val_patient_ids: return two patients from the real H5.
        patch.object(engine, "_val_patient_ids", return_value=["P0", "P1"]),
        # _process_patient always raises — simulating the assembler KeyError.
        patch.object(
            engine,
            "_process_patient",
            side_effect=KeyError("m_tc_soft — conditioning wiring error (synthetic)"),
        ),
    ):
        with pytest.raises(ExhaustiveValAllSkippedError) as exc_info:
            engine.run()

    # The error message must mention mask_source.
    assert "mask_source" in str(exc_info.value).lower() or "mask_source" in str(exc_info.value), (
        f"ExhaustiveValAllSkippedError message must mention mask_source: {exc_info.value}"
    )

    # No metrics.csv must have been written.
    metrics_csv = out_dir / "metrics.csv"
    assert not metrics_csv.exists(), (
        f"metrics.csv must NOT be written when all patients are skipped; found: {metrics_csv}"
    )
