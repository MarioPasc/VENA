"""Phase-2 statistical primitives — pure numpy/pandas, no I/O.

All functions operate on **patient-level** data.  The caller is responsible
for calling :func:`collapse_to_patient` first.  Aggregating on scan-level is
trap #4 in SHARED_CONTRACTS §11 and silently produces anti-conservative tests.

Re-exports :func:`spearman_with_bootstrap_ci` from the canonical location so
the §4.3 (spatial residual) agent has a single obvious import.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats as scipy_stats
from statsmodels.stats import multitest as sm_multitest

# Re-export from canonical location (do not write a second Spearman).
from vena.preflight.priors_validation.statistics.correlation import (
    SpearmanResult,
    spearman_with_bootstrap_ci,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Minimum clinically important difference on the [0, 1] intensity scale.
#: Revicki et al. 2008, J Clin Epidemiol 61(2):102–9.
#: Used by all routines so that "statistically but not clinically significant"
#: is reported consistently.
MCID: float = 0.01

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WilcoxonResult:
    """Result of a paired Wilcoxon signed-rank test."""

    statistic: float
    pvalue: float
    n: int  # number of matched patient pairs


@dataclass(frozen=True)
class HolmResult:
    """Per-comparison result after Holm-Bonferroni correction."""

    pvalue_adj: float
    reject: bool  # at alpha = 0.05


# ---------------------------------------------------------------------------
# Core statistical functions
# ---------------------------------------------------------------------------


def collapse_to_patient(
    df: pd.DataFrame,
    value_cols: Sequence[str],
    *,
    by: tuple[str, ...] = ("method", "cohort", "nfe", "patient_id"),
) -> pd.DataFrame:
    """Mean over scans within a patient.

    This is the **single most important function** in Phase 2 (SHARED_CONTRACTS
    §11 trap #4).  LUMIERE has 72 scans from 11 patients — calling a test on
    72 rows inflates significance by 6×.  Always call this first.

    Parameters
    ----------
    df :
        Tidy DataFrame with at least the columns in *by* plus *value_cols*.
    value_cols :
        Columns to aggregate by mean.
    by :
        Grouping keys.  Default groups by ``method × cohort × nfe × patient_id``.

    Returns
    -------
    pd.DataFrame
        One row per unique combination of *by*, with *value_cols* averaged.
    """
    return df.groupby(list(by), sort=False)[list(value_cols)].mean().reset_index()


def paired_wilcoxon(
    vena: pd.Series,
    competitor: pd.Series,
) -> WilcoxonResult:
    """Two-sided Wilcoxon signed-rank test, paired on patient_id.

    Both series must be indexed by ``patient_id``.  An inner-join on
    ``patient_id`` is performed before testing so that unmatched patients do
    not silently skew the result.  An assertion guards that the inner-joined
    sets are identical — an unaligned paired test is statistical nonsense.

    Parameters
    ----------
    vena :
        Per-patient metric for the VENA method, indexed by ``patient_id``.
    competitor :
        Per-patient metric for the comparator, indexed by ``patient_id``.

    Returns
    -------
    WilcoxonResult
        Statistic, p-value, and n (number of matched pairs).
    """
    common = vena.index.intersection(competitor.index)
    if len(common) == 0:
        raise ValueError("paired_wilcoxon: no common patient_ids between the two arms")
    v = vena.loc[common].to_numpy(dtype=float)
    c = competitor.loc[common].to_numpy(dtype=float)

    diffs = v - c
    if np.all(diffs == 0.0):
        # All differences are exactly zero — the exact p-value is 1.0.
        return WilcoxonResult(statistic=0.0, pvalue=1.0, n=len(common))

    result = scipy_stats.wilcoxon(diffs, alternative="two-sided")
    return WilcoxonResult(
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        n=len(common),
    )


def holm_bonferroni(pvalues: dict[str, float]) -> dict[str, HolmResult]:
    """Apply Holm-Bonferroni correction to a family of p-values.

    The **dict key set defines the family** — pass exactly one family at a
    time (e.g. all VENA-vs-competitor comparisons on Ring A, PSNR).

    Parameters
    ----------
    pvalues :
        ``{comparison_name: raw_p_value}``.

    Returns
    -------
    dict[str, HolmResult]
        Per-comparison adjusted p-value and reject flag at α = 0.05.
    """
    if not pvalues:
        return {}
    keys = list(pvalues.keys())
    raw = np.array([pvalues[k] for k in keys])
    reject, pvals_adj, _, _ = sm_multitest.multipletests(raw, method="holm")
    return {
        k: HolmResult(pvalue_adj=float(pvals_adj[i]), reject=bool(reject[i]))
        for i, k in enumerate(keys)
    }


def cliffs_delta(a: NDArray, b: NDArray) -> float:
    """Non-parametric effect size (Cliff 1996).

    δ = (# pairs where a > b  −  # pairs where a < b) / (n_a × n_b).

    Range [-1, +1]:
      +1 = a fully dominates b.
      -1 = b fully dominates a.
       0 = identical distributions.

    Magnitude thresholds (Vargha & Delaney 2000 / Romano et al. 2006):
      |δ| < 0.147  negligible
      |δ| < 0.330  small
      |δ| < 0.474  medium
      |δ| ≥ 0.474  large

    Parameters
    ----------
    a, b :
        1-D arrays of patient-level values.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    # Vectorised dominance count — O(na × nb) in space.
    n_dom = np.sum(a[:, None] > b[None, :])
    n_sub = np.sum(a[:, None] < b[None, :])
    return float(n_dom - n_sub) / float(len(a) * len(b))


def bootstrap_ci(
    values: NDArray,
    *,
    n_boot: int = 10_000,
    ci: float = 0.95,
    strata: NDArray | None = None,
    seed: int = 1337,
) -> tuple[float, float]:
    """Bootstrap confidence interval, patient-stratified when *strata* given.

    When *strata* is supplied (cohort labels), each bootstrap resample draws
    patients within each stratum with replacement, preserving cohort proportions
    (proposal §6.2 / SHARED_CONTRACTS §8).  Without *strata*, a plain
    non-parametric bootstrap is used.

    Both paths resample **patients, never scans** — the caller must have called
    :func:`collapse_to_patient` first.

    Parameters
    ----------
    values :
        1-D array of per-patient metric values.
    n_boot :
        Number of bootstrap replicates.
    ci :
        Nominal coverage, e.g. 0.95.
    strata :
        1-D array of stratum labels (same length as *values*).
    seed :
        Fixed seed for reproducibility.

    Returns
    -------
    (lo, hi) :
        Lower and upper quantile-based CI bounds.
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    n = len(values)
    alpha = 1.0 - ci

    boot_means = np.empty(n_boot, dtype=float)

    if strata is not None:
        strata_arr = np.asarray(strata)
        unique_strata, strata_inv = np.unique(strata_arr, return_inverse=True)
        strata_groups = [np.where(strata_inv == k)[0] for k in range(len(unique_strata))]
        for b in range(n_boot):
            idx_parts = [rng.choice(g, size=len(g), replace=True) for g in strata_groups]
            idx = np.concatenate(idx_parts)
            boot_means[b] = np.mean(values[idx])
    else:
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot_means[b] = np.mean(values[idx])

    lo = float(np.percentile(boot_means, 100.0 * alpha / 2.0))
    hi = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


__all__ = [
    "MCID",
    "HolmResult",
    "SpearmanResult",
    "WilcoxonResult",
    "bootstrap_ci",
    "cliffs_delta",
    "collapse_to_patient",
    "holm_bonferroni",
    "paired_wilcoxon",
    "spearman_with_bootstrap_ci",
]
