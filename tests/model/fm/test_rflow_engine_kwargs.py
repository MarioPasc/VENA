"""Tests for :class:`RFlowEngine` `use_timestep_transform` / `base_img_size_numel` plumbing.

The 2026-06-20 baseline overhaul exposed two MONAI ``RFlowScheduler`` kwargs
that were previously inaccessible from the YAML. These tests verify the
dataclass accepts the new fields, default behaviour is byte-identical
backward-compat, and the kwargs reach the underlying scheduler.
"""

from __future__ import annotations

import pytest

from vena.model.fm.sampler.rflow import RFlowEngine


@pytest.mark.unit
def test_default_no_transform() -> None:
    engine = RFlowEngine()
    assert engine.use_timestep_transform is False
    assert engine.base_img_size_numel is None
    # Underlying scheduler builds without error.
    assert engine.scheduler is not None


@pytest.mark.unit
def test_with_timestep_transform_enabled() -> None:
    engine = RFlowEngine(
        use_timestep_transform=True,
        base_img_size_numel=129024,
    )
    assert engine.use_timestep_transform is True
    assert engine.base_img_size_numel == 129024
    # The scheduler reflects the kwarg (MONAI sets attributes from kwargs).
    sch = engine.scheduler
    assert getattr(sch, "use_timestep_transform", False) is True


@pytest.mark.unit
def test_dict_expansion_from_yaml_path() -> None:
    """Verifies the dataclass accepts the kwarg dict the engine builds.

    ``vena.model.fm.lightning.module:200`` calls ``RFlowEngine(**rflow_cfg)``
    with the YAML's ``rflow:`` block as a dict — these new fields must
    survive that round-trip.
    """
    cfg = {
        "num_train_timesteps": 1000,
        "use_discrete_timesteps": True,
        "sample_method": "logit-normal",
        "use_timestep_transform": True,
    }
    engine = RFlowEngine(**cfg)
    assert engine.use_timestep_transform is True
    assert engine.num_train_timesteps == 1000


@pytest.mark.unit
def test_backward_compat_dict_without_new_fields() -> None:
    cfg = {
        "num_train_timesteps": 1000,
        "use_discrete_timesteps": True,
        "sample_method": "uniform",
    }
    engine = RFlowEngine(**cfg)
    assert engine.use_timestep_transform is False
