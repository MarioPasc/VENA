"""Sanity tests for the statistics primitives."""

from __future__ import annotations

import numpy as np

from vena.preflight.priors_validation.statistics import (
    bh_fdr,
    fisher_z,
    fisher_z_meta_analysis,
    icc_2_1,
    inverse_fisher_z,
    spearman_with_bootstrap_ci,
)


def test_spearman_strong_correlation():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    y = 0.7 * x + 0.3 * rng.normal(size=500)
    res = spearman_with_bootstrap_ci(x, y, n_boot=100, seed=0)
    assert 0.85 < res.rho < 1.0
    assert res.rho_lo > 0.7


def test_spearman_constant_returns_nan():
    res = spearman_with_bootstrap_ci(np.ones(50), np.arange(50, dtype=float), n_boot=10, seed=0)
    assert not np.isfinite(res.rho)


def test_bh_fdr_orders_pvalues():
    p = np.array([0.001, 0.01, 0.04, 0.5])
    reject, adj = bh_fdr(p, q=0.05)
    assert reject[0] and reject[1]
    assert adj[0] <= adj[1] <= adj[2] <= adj[3]


def test_icc_round_trip():
    rng = np.random.default_rng(0)
    a = rng.normal(size=30) * 5
    b = a + 0.5 * rng.normal(size=30)
    assert 0.8 < icc_2_1(np.column_stack([a, b])) <= 1.0
    assert icc_2_1(np.column_stack([a, rng.normal(size=30) * 5])) < 0.4


def test_fisher_z_meta():
    rhos = np.array([0.3, 0.4, 0.5])
    ns = np.array([100, 100, 100])
    pooled, lo, hi = fisher_z_meta_analysis(rhos, ns)
    assert 0.3 < pooled < 0.5
    assert lo < pooled < hi
    assert np.isclose(inverse_fisher_z(fisher_z(0.42)), 0.42)
