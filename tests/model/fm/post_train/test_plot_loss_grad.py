"""Smoke tests for the loss/grad plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import pandas as pd
import pytest

from vena.model.fm.post_train.plot_loss_grad import (
    plot_per_cohort_grad,
    plot_total_grad,
)

pytestmark = pytest.mark.unit


def _df(with_contrastive: bool, with_trunk: bool, with_cohorts: bool = True) -> pd.DataFrame:
    n = 12
    rng = np.random.default_rng(0)
    data = {
        "epoch": np.arange(n),
        "cfm_mean": 0.5 + 0.01 * rng.standard_normal(n),
        "cfm_std": 0.02 + 0.001 * rng.standard_normal(n),
        "contrastive_mean": (
            0.1 + 0.01 * rng.standard_normal(n) if with_contrastive else np.zeros(n)
        ),
        "contrastive_std": (
            0.005 + 0.001 * rng.standard_normal(n) if with_contrastive else np.zeros(n)
        ),
        "total_mean": 0.6 + 0.01 * rng.standard_normal(n),
        "total_std": 0.02 + 0.001 * rng.standard_normal(n),
        "grad_norm_cn_postclip_mean": 1.0 + 0.05 * rng.standard_normal(n),
        "grad_norm_cn_postclip_std": 0.1 + 0.01 * rng.standard_normal(n),
        "grad_norm_trunk_postclip_mean": (
            0.5 + 0.05 * rng.standard_normal(n) if with_trunk else np.full(n, np.nan)
        ),
        "grad_norm_trunk_postclip_std": (
            0.05 + 0.01 * rng.standard_normal(n) if with_trunk else np.full(n, np.nan)
        ),
    }
    if with_cohorts:
        for cohort in ("BraTS-GLI", "LUMIERE"):
            data[f"cfm_cohort_{cohort}_mean"] = 0.5 + 0.01 * rng.standard_normal(n)
            data[f"cfm_cohort_{cohort}_std"] = 0.01 + 0.001 * rng.standard_normal(n)
    return pd.DataFrame(data)


def test_plot_total_grad_s1(tmp_path: Path) -> None:
    df = _df(with_contrastive=False, with_trunk=False)
    out = plot_total_grad(df, tmp_path / "loss_total_grad.png")
    assert out.exists()
    assert out.stat().st_size > 1000  # non-trivial PNG


def test_plot_total_grad_s2(tmp_path: Path) -> None:
    df = _df(with_contrastive=True, with_trunk=True)
    out = plot_total_grad(df, tmp_path / "loss_total_grad.png")
    assert out.exists()
    assert out.stat().st_size > 1000


def test_plot_per_cohort_grad(tmp_path: Path) -> None:
    df = _df(with_contrastive=False, with_trunk=False, with_cohorts=True)
    out = plot_per_cohort_grad(df, tmp_path / "loss_per_cohort_grad.png")
    assert out.exists()
    assert out.stat().st_size > 1000


def test_plot_per_cohort_grad_no_cohorts(tmp_path: Path) -> None:
    df = _df(with_contrastive=False, with_trunk=False, with_cohorts=False)
    out = plot_per_cohort_grad(df, tmp_path / "loss_per_cohort_grad.png")
    assert out.exists()
