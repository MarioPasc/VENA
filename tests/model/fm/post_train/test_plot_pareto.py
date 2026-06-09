"""Smoke + behavioural tests for the Pareto plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import pandas as pd
import pytest

from vena.model.fm.post_train.plot_pareto import (
    _step_post_segments,
    aggregate_pareto,
    plot_pareto,
)

pytestmark = pytest.mark.unit


def _exhaustive_dfs() -> dict[int, pd.DataFrame]:
    """Two epochs, three NFE values, four patients each."""
    out: dict[int, pd.DataFrame] = {}
    rng = np.random.default_rng(0)
    for epoch in (10, 20):
        rows = []
        for nfe in (1, 5, 10):
            for pid in range(4):
                rows.append(
                    {
                        "cohort": "A",
                        "epoch": epoch,
                        "patient_id": f"p{pid}",
                        "nfe": nfe,
                        "psnr_db": 20.0 + nfe * 0.5 + 0.1 * rng.standard_normal(),
                        "ssim": 0.6 + 0.02 * nfe + 0.005 * rng.standard_normal(),
                    }
                )
        out[epoch] = pd.DataFrame(rows)
    return out


def test_aggregate_pareto_shape() -> None:
    dfs = _exhaustive_dfs()
    agg = aggregate_pareto(dfs)
    assert set(agg["epoch"].unique()) == {10, 20}
    assert set(agg["nfe"].unique()) == {1, 5, 10}
    assert len(agg) == 6
    assert (agg["n_patients"] == 4).all()


def test_aggregate_pareto_empty() -> None:
    out = aggregate_pareto({})
    assert out.empty


def test_step_post_segments_count() -> None:
    xs = np.array([1.0, 2.0, 3.0])
    ys = np.array([10.0, 20.0, 30.0])
    sx, sy = _step_post_segments(xs, ys)
    # 2*k + 1 = 5 points for k = 2 segments
    assert sx.size == 5
    assert sy.size == 5
    # First and last anchors preserved
    assert sx[0] == 1.0
    assert sy[0] == 10.0
    assert sx[-1] == 3.0
    assert sy[-1] == 30.0


def test_plot_pareto_writes_file(tmp_path: Path) -> None:
    out = plot_pareto(_exhaustive_dfs(), tmp_path / "pareto.png")
    assert out.exists()
    assert out.stat().st_size > 1000


def test_plot_pareto_raises_on_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        plot_pareto({}, tmp_path / "pareto.png")
