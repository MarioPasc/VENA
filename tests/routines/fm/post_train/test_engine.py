"""End-to-end test for the post-training routine engine."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import pandas as pd
import pytest
from routines.fm.post_train.engine import (
    PostTrainEngine,
    PostTrainRoutineConfig,
)

pytestmark = pytest.mark.unit


def _populate_run_dir(run_dir: Path) -> None:
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    n = 8
    rng = np.random.default_rng(0)
    pd.DataFrame(
        {
            "epoch": np.arange(n),
            "cfm_mean": 0.5 + 0.01 * rng.standard_normal(n),
            "cfm_std": 0.01 + 0.001 * rng.standard_normal(n),
            "contrastive_mean": np.zeros(n),
            "contrastive_std": np.zeros(n),
            "total_mean": 0.5 + 0.01 * rng.standard_normal(n),
            "total_std": 0.01 + 0.001 * rng.standard_normal(n),
            "grad_norm_cn_postclip_mean": 1.0 + 0.05 * rng.standard_normal(n),
            "grad_norm_cn_postclip_std": 0.1 + 0.01 * rng.standard_normal(n),
            "grad_norm_trunk_postclip_mean": np.full(n, np.nan),
            "grad_norm_trunk_postclip_std": np.full(n, np.nan),
            "cfm_cohort_BraTS-GLI_mean": 0.5 + 0.01 * rng.standard_normal(n),
            "cfm_cohort_BraTS-GLI_std": 0.01 + 0.001 * rng.standard_normal(n),
        }
    ).to_csv(metrics_dir / "train_epoch.csv", index=False)

    ev_dir = run_dir / "exhaustive_val" / "epoch_005"
    ev_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "cohort": ["A"] * 6,
            "epoch": [5] * 6,
            "patient_id": [f"p{i}" for i in range(6)],
            "nfe": [1, 1, 5, 5, 10, 10],
            "psnr_db": [20.0, 20.5, 22.0, 22.5, 24.0, 24.5],
            "ssim": [0.5, 0.55, 0.6, 0.62, 0.65, 0.67],
        }
    ).to_csv(ev_dir / "metrics.csv", index=False)


def test_engine_writes_all_three_plots(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _populate_run_dir(run_dir)

    cfg = PostTrainRoutineConfig(run_dir=run_dir, formats=("png",))
    plots_dir = PostTrainEngine(cfg).run()

    assert plots_dir == run_dir / "plots"
    for stem in ("loss_total_grad", "loss_per_cohort_grad", "pareto_psnr_ssim"):
        path = plots_dir / f"{stem}.png"
        assert path.exists(), f"missing {path}"
        assert path.stat().st_size > 1000


def test_engine_raises_when_run_dir_missing(tmp_path: Path) -> None:
    cfg = PostTrainRoutineConfig(run_dir=tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        PostTrainEngine(cfg).run()


def test_engine_raises_when_metrics_csv_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg = PostTrainRoutineConfig(run_dir=run_dir)
    with pytest.raises(FileNotFoundError):
        PostTrainEngine(cfg).run()


def test_engine_skips_pareto_when_no_exhaustive_val(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _populate_run_dir(run_dir)
    # Drop the exhaustive_val dir to simulate `exhaustive_val.enabled=false`
    import shutil

    shutil.rmtree(run_dir / "exhaustive_val")

    plots_dir = PostTrainEngine(PostTrainRoutineConfig(run_dir=run_dir)).run()
    assert (plots_dir / "loss_total_grad.png").exists()
    assert not (plots_dir / "pareto_psnr_ssim.png").exists()


def test_config_normalises_single_string_format(tmp_path: Path) -> None:
    # Accept a YAML where `formats: png` is a bare scalar.
    cfg = PostTrainRoutineConfig.model_validate({"run_dir": str(tmp_path), "formats": "png"})
    assert cfg.formats == ("png",)
