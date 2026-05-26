"""Spearman correlation with bootstrap CI + Fisher-z meta-analysis.

The bootstrap resamples voxel indices with replacement (i.i.d. voxel
sampling). Spec §10.3 flags spatial autocorrelation as a known limitation;
block-bootstrap is listed as a stretch goal and is not implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats


@dataclass(frozen=True)
class SpearmanResult:
    """Spearman ρ with bootstrap CI."""

    rho: float
    rho_lo: float
    rho_hi: float
    p_value: float
    n: int


def spearman_with_bootstrap_ci(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    *,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 1337,
    nan_policy: str = "omit",
) -> SpearmanResult:
    """Compute Spearman ρ + bootstrap (1 − α/2, α/2) CI on the same population.

    Parameters
    ----------
    x, y
        1-D arrays of the same shape.
    n_boot
        Number of bootstrap resamples for the CI.
    ci
        Two-sided coverage probability; ``0.95`` → ``[2.5, 97.5]`` percentiles.
    seed
        Seed for the bootstrap RNG.
    nan_policy
        Passed to :func:`scipy.stats.spearmanr`.

    Returns
    -------
    SpearmanResult
        With NaNs returned if ``len(x) < 3`` (correlation undefined) or if
        either vector is constant.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: x={x.shape}, y={y.shape}")
    if x.size < 3:
        return SpearmanResult(
            rho=float("nan"),
            rho_lo=float("nan"),
            rho_hi=float("nan"),
            p_value=float("nan"),
            n=int(x.size),
        )
    if nan_policy == "omit":
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return SpearmanResult(
            rho=float("nan"),
            rho_lo=float("nan"),
            rho_hi=float("nan"),
            p_value=float("nan"),
            n=int(x.size),
        )
    res = stats.spearmanr(x, y)
    rho = float(res.statistic)
    p = float(res.pvalue)
    rng = np.random.default_rng(seed)
    boot_rhos = np.empty(n_boot, dtype=np.float64)
    n = x.size
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sub_x = x[idx]
        sub_y = y[idx]
        if np.ptp(sub_x) == 0 or np.ptp(sub_y) == 0:
            boot_rhos[i] = np.nan
            continue
        boot_rhos[i] = stats.spearmanr(sub_x, sub_y).statistic
    boot_rhos = boot_rhos[np.isfinite(boot_rhos)]
    if boot_rhos.size < max(10, n_boot // 100):
        rho_lo = rho_hi = float("nan")
    else:
        alpha = 1.0 - ci
        rho_lo = float(np.percentile(boot_rhos, 100.0 * alpha / 2.0))
        rho_hi = float(np.percentile(boot_rhos, 100.0 * (1.0 - alpha / 2.0)))
    return SpearmanResult(rho=rho, rho_lo=rho_lo, rho_hi=rho_hi, p_value=p, n=int(n))


def fisher_z(rho: float | NDArray[np.floating]) -> float | NDArray[np.floating]:
    """Fisher z-transform: z = atanh(ρ). Clamped at |ρ| = 1 - 1e-7."""
    rho_arr = np.asarray(rho, dtype=np.float64)
    rho_clip = np.clip(rho_arr, -1.0 + 1e-7, 1.0 - 1e-7)
    return np.arctanh(rho_clip)


def inverse_fisher_z(z: float | NDArray[np.floating]) -> float | NDArray[np.floating]:
    """Inverse Fisher z: ρ = tanh(z)."""
    return np.tanh(np.asarray(z, dtype=np.float64))


def fisher_z_meta_analysis(
    rhos: NDArray[np.floating],
    ns: NDArray[np.integer],
) -> tuple[float, float, float]:
    """Fixed-effects meta-analysis on Fisher-z-transformed correlations.

    Returns
    -------
    (pooled_rho, lo, hi)
        Pooled correlation and its 95 % CI in correlation units.
    """
    rhos = np.asarray(rhos, dtype=np.float64)
    ns = np.asarray(ns, dtype=np.float64)
    valid = np.isfinite(rhos) & (ns > 3)
    if valid.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    z = fisher_z(rhos[valid])
    w = ns[valid] - 3.0
    z_pooled = float(np.sum(w * z) / np.sum(w))
    se = float(np.sqrt(1.0 / np.sum(w)))
    z_lo = z_pooled - 1.96 * se
    z_hi = z_pooled + 1.96 * se
    return (
        float(inverse_fisher_z(z_pooled)),
        float(inverse_fisher_z(z_lo)),
        float(inverse_fisher_z(z_hi)),
    )
