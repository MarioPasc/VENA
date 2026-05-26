"""Benjamini–Hochberg FDR control wrapper."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from statsmodels.stats.multitest import multipletests


def bh_fdr(
    pvalues: NDArray[np.floating],
    q: float = 0.05,
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Benjamini–Hochberg FDR control at level ``q``.

    Returns
    -------
    (reject, adjusted_pvalues)
        ``reject`` is a boolean array indicating which hypotheses are rejected
        at level ``q``; ``adjusted_pvalues`` are the BH-adjusted p-values.
    """
    arr = np.asarray(pvalues, dtype=np.float64).ravel()
    if arr.size == 0:
        return np.array([], dtype=bool), np.array([], dtype=np.float64)
    valid = np.isfinite(arr)
    reject = np.zeros_like(arr, dtype=bool)
    adj = np.full_like(arr, np.nan, dtype=np.float64)
    if valid.any():
        r, p_adj, _, _ = multipletests(arr[valid], alpha=q, method="fdr_bh")
        reject[valid] = r
        adj[valid] = p_adj
    return reject, adj
