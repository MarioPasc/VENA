"""ICC(2,1) and ICC(3,1) wrappers around ``pingouin.intraclass_corr``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pingouin as pg
from numpy.typing import NDArray


def icc_2_1(values: NDArray[np.floating]) -> float:
    """Compute ICC(2,1) ("single random raters") from a ``(n_targets, n_raters)`` matrix.

    Parameters
    ----------
    values
        2-D array. Rows are targets (subjects / ROIs), columns are raters
        (paired scans). NaN rows are dropped.

    Returns
    -------
    float
        ICC(2,1). Returns NaN if the matrix has fewer than 2 valid rows or
        fewer than 2 columns.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"values must be 2-D, got shape {arr.shape}")
    if arr.shape[1] < 2:
        return float("nan")
    valid = np.all(np.isfinite(arr), axis=1)
    arr = arr[valid]
    if arr.shape[0] < 2:
        return float("nan")
    n_targets, n_raters = arr.shape
    long_form = pd.DataFrame(
        {
            "target": np.repeat(np.arange(n_targets), n_raters),
            "rater": np.tile(np.arange(n_raters), n_targets),
            "value": arr.ravel(),
        }
    )
    icc_df = pg.intraclass_corr(data=long_form, targets="target", raters="rater", ratings="value")
    # Pingouin labels two-way-random absolute-agreement single-rater as
    # "ICC(A,1)"; this corresponds to ICC(2,1) in the Shrout–Fleiss / Koo–Li
    # naming convention (Koo & Li 2016, DOI: 10.1016/j.jcm.2016.02.012).
    row = icc_df[icc_df["Type"] == "ICC(A,1)"]
    if row.empty:
        # Older pingouin used "ICC2" as the label; keep a fallback.
        row = icc_df[icc_df["Type"] == "ICC2"]
    if row.empty:
        return float("nan")
    return float(row["ICC"].values[0])
