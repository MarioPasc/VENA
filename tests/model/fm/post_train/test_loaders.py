"""Unit tests for post-training loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vena.model.fm.post_train.loaders import (
    active_grad_norm_series,
    detect_active_cohorts,
    detect_active_losses,
    discover_exhaustive_val,
    load_train_epoch_csv,
)

pytestmark = pytest.mark.unit


def _make_train_epoch_df(*, with_contrastive: bool, with_trunk: bool) -> pd.DataFrame:
    epochs = np.arange(10, dtype=int)
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "epoch": epochs,
            "step": epochs * 100,
            "cfm_mean": 0.5 + 0.01 * rng.standard_normal(10),
            "cfm_std": 0.01 + 0.001 * rng.standard_normal(10),
            "contrastive_mean": (
                0.1 + 0.01 * rng.standard_normal(10) if with_contrastive else np.zeros(10)
            ),
            "contrastive_std": (
                0.005 + 0.001 * rng.standard_normal(10) if with_contrastive else np.zeros(10)
            ),
            "reconstruction_mean": np.zeros(10),
            "reconstruction_std": np.zeros(10),
            "total_mean": 0.6 + 0.01 * rng.standard_normal(10),
            "total_std": 0.01 + 0.001 * rng.standard_normal(10),
            "grad_norm_cn_postclip_mean": 1.0 + 0.1 * rng.standard_normal(10),
            "grad_norm_cn_postclip_std": 0.1 + 0.01 * rng.standard_normal(10),
            "grad_norm_trunk_postclip_mean": (
                0.5 + 0.05 * rng.standard_normal(10) if with_trunk else np.full(10, np.nan)
            ),
            "grad_norm_trunk_postclip_std": (
                0.05 + 0.005 * rng.standard_normal(10) if with_trunk else np.full(10, np.nan)
            ),
            "cfm_cohort_BraTS-GLI_mean": 0.5 + 0.01 * rng.standard_normal(10),
            "cfm_cohort_BraTS-GLI_std": 0.01 + 0.001 * rng.standard_normal(10),
            "cfm_cohort_LUMIERE_mean": 0.6 + 0.01 * rng.standard_normal(10),
            "cfm_cohort_LUMIERE_std": 0.01 + 0.001 * rng.standard_normal(10),
            "cfm_cohort_GHOST_mean": np.zeros(10),  # cohort with no data → suppressed
            "cfm_cohort_GHOST_std": np.zeros(10),
        }
    )
    return df


def test_detect_active_losses_s1() -> None:
    df = _make_train_epoch_df(with_contrastive=False, with_trunk=False)
    assert detect_active_losses(df) == ["cfm"]


def test_detect_active_losses_s2() -> None:
    df = _make_train_epoch_df(with_contrastive=True, with_trunk=True)
    assert detect_active_losses(df) == ["cfm", "contrastive"]


def test_detect_active_losses_skips_reconstruction() -> None:
    df = _make_train_epoch_df(with_contrastive=True, with_trunk=False)
    df["reconstruction_mean"] = 0.2  # non-zero, but `reconstruction` is never in the list
    assert "reconstruction" not in detect_active_losses(df)


def test_detect_active_cohorts_suppresses_all_zero() -> None:
    df = _make_train_epoch_df(with_contrastive=False, with_trunk=False)
    cohorts = detect_active_cohorts(df, loss="cfm")
    assert "BraTS-GLI" in cohorts
    assert "LUMIERE" in cohorts
    assert "GHOST" not in cohorts


def test_active_grad_norm_series_frozen_trunk() -> None:
    df = _make_train_epoch_df(with_contrastive=False, with_trunk=False)
    assert active_grad_norm_series(df) == ["cn"]


def test_active_grad_norm_series_trainable_trunk() -> None:
    df = _make_train_epoch_df(with_contrastive=True, with_trunk=True)
    assert active_grad_norm_series(df) == ["cn", "trunk"]


def test_load_train_epoch_csv_raises_on_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_train_epoch_csv(tmp_path)


def test_load_train_epoch_csv_sorts_by_epoch(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    df = _make_train_epoch_df(with_contrastive=False, with_trunk=False)
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    df.to_csv(metrics_dir / "train_epoch.csv", index=False)
    loaded = load_train_epoch_csv(tmp_path)
    assert list(loaded["epoch"]) == sorted(loaded["epoch"])


def test_discover_exhaustive_val_empty(tmp_path: Path) -> None:
    assert discover_exhaustive_val(tmp_path) == {}


def test_discover_exhaustive_val_finds_epoch_dirs(tmp_path: Path) -> None:
    base = tmp_path / "exhaustive_val"
    for epoch in (5, 10, 20):
        d = base / f"epoch_{epoch:03d}"
        d.mkdir(parents=True)
        df = pd.DataFrame(
            {
                "cohort": ["A"] * 4,
                "epoch": [epoch] * 4,
                "patient_id": [f"p{i}" for i in range(4)],
                "nfe": [1, 2, 5, 10],
                "psnr_db": [20.0, 22.0, 24.0, 25.0],
                "ssim": [0.5, 0.6, 0.7, 0.75],
            }
        )
        df.to_csv(d / "metrics.csv", index=False)
    out = discover_exhaustive_val(tmp_path)
    assert list(out.keys()) == [5, 10, 20]
    assert all(not v.empty for v in out.values())
