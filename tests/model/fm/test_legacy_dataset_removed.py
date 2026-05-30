"""Guard test: legacy LatentH5DataModule must not reappear.

The single-cohort Lightning wrapper was removed in the pre-long-run
hardening pass; every training path now flows through
``MultiCohortLatentDataModule`` driven by a ``corpus_registry``. This test
fails if anyone re-introduces the class so the deletion does not silently
get reverted.
"""

from __future__ import annotations

import pytest

import vena.model.fm.lightning as fm_lightning
import vena.model.fm.lightning.data as fm_data

pytestmark = pytest.mark.unit


def test_lightning_package_does_not_export_legacy_datamodule() -> None:
    assert "LatentH5DataModule" not in fm_lightning.__all__
    assert not hasattr(fm_lightning, "LatentH5DataModule")


def test_data_module_class_is_removed() -> None:
    assert not hasattr(fm_data, "LatentH5DataModule")


def test_per_cohort_dataset_class_is_retained() -> None:
    """LatentH5Dataset (the per-cohort building block) must survive."""
    assert hasattr(fm_data, "LatentH5Dataset")
    assert hasattr(fm_data, "MultiCohortLatentDataset")
    assert hasattr(fm_data, "MultiCohortLatentDataModule")
