"""Phase-2 §4.3 spatial residual analysis — bright-region error concentration.

Tests whether each method's absolute residual |r(x)| = |T1c(x) - T̂1c(x)|
is disproportionately concentrated in the bright-intensity regions of the
real T1c volume.  This is the headline vessel-fidelity claim of VENA,
evaluated label-free (no vessel segmenter required).

Two conditions per scan:
- **C-WB** (whole brain): R = brain mask.
- **C-noT** (background): R = brain \\ dilate(WT, k=5).  Bright voxels here
  are exclusively vessels, dural sinuses, choroid plexus, pituitary, pineal,
  and dural enhancement.  This is the vessel-fidelity condition.

Three headline statistics per (scan, method, condition):
- S1: Spearman ρ(|r|, T1c) over R.
- S2: Conc(q) — top-q% bright-voxel error mass concentration ratio.
- MI: KSG mutual information (exploratory, subsampled).

Both shuffle null domains are implemented:
- brain: shuffle T1c within brain, restrict to R  (proposal default, primary).
- R: shuffle T1c within R directly (secondary, for reviewer robustness check).

References
----------
Alexander-Bloch et al. 2018, NeuroImage — spatial null construction.
Bishara & Hittner 2012, Psychological Methods — Spearman over Pearson.
Bland & Altman 1986/1999 — intensity-stratified residual plots.
Abraham et al. 2003, ApJ — Conc(q) error-mass concentration.
"""

# ruff: noqa: N806, N803  — `_R` suffix and `abs_resid_R`/`t1c_R` args are
# deliberate math-notation: R = region (proposal §4.3).  Renaming would
# obscure the mapping to the LaTeX in the proposal.
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats as scipy_stats
from sklearn.feature_selection import mutual_info_regression

from vena.validation.io import ScanSample
from vena.validation.stats import (
    SpearmanResult,
    bootstrap_ci,
    cliffs_delta,
    collapse_to_patient,
    holm_bonferroni,
    paired_wilcoxon,
    spearman_with_bootstrap_ci,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Holm family size for the main spatial-residual family:
#: 2 statistics (Conc(5%) and ρ_S) × 8 competitors = 16 tests.
#: This is pre-registered and cannot change without a schema bump.
HOLM_FAMILY_SIZE: int = 16
_N_COMPETITORS: int = 8
_N_STATS: int = 2
assert HOLM_FAMILY_SIZE == _N_COMPETITORS * _N_STATS, (
    f"HOLM_FAMILY_SIZE mismatch: {_N_COMPETITORS} × {_N_STATS} ≠ {HOLM_FAMILY_SIZE}"
)

#: Quantile values for Conc(q).
CONC_QUANTILES: tuple[float, ...] = (0.01, 0.05, 0.10)

#: Number of intensity deciles for the Bland-Altman stratified plot.
N_DECILES: int = 10

#: Selection NFE per method — the single NFE used for headline-table comparisons.
#: Pre-registered in SHARED_CONTRACTS.md §4 (2026-07-16, do not re-litigate).
#: Used by :func:`_filter_to_selection_nfe` to avoid duplicate patient_id indices
#: in the Wilcoxon pairing join when a sweep covers multiple NFE values.
_SELECTION_NFE: dict[str, int] = {
    "C0-Identity": 1,
    "C1-pGAN-t1pre": 1,
    "C1-pGAN-t2": 1,
    "C1-pGAN-flair": 1,
    "C2-ResViT": 1,
    "C3-SynDiff-t1pre": 4,
    "C3-SynDiff-t2": 4,
    "C3-SynDiff-flair": 4,
    "C4-3D-DiT": 5,
    "C5-T1C-RFlow": 5,
    "C6-3D-LDDPM": 1000,
    "C7-3D-Latent-Pix2Pix": 1,
    "VENA-S1-v3a": 5,
    "VENA-S1-v3b": 5,
    "VENA-S1-v3b-rw": 5,
    "VENA-S3-LPL-b2c": 5,
}

#: Frozen CSV column order.  Any new column must be appended, never inserted,
#: to keep existing CSVs readable by old analysis scripts.
SPATIAL_CSV_COLUMNS: list[str] = [
    # --- identity ---
    "method",
    "cohort",
    "ring",
    "nfe",
    "scan_id",
    "patient_id",
    "condition",
    # --- scoring-space audit (§7.0) ---
    "pred_mode",  # "raw" | "harmonised" — which volume was scored
    "raw_p995",  # np.percentile(raw[brain], 99.5) — under-saturation audit
    # --- region size ---
    "n_voxels_brain",
    "n_voxels_region",
    # --- S1: Spearman ρ ---
    "rho_s",
    "rho_s_lo",
    "rho_s_hi",
    "rho_s_p",
    # --- S2: Conc(q) for q ∈ {1%, 5%, 10%} ---
    "conc_01",
    "conc_05",
    "conc_10",
    # --- exploratory: KSG MI ---
    "mi_ksg",
    "mi_n_sampled",
    # --- shuffle null (within-brain domain, primary) ---
    "null_brain_rho_mean",
    "null_brain_rho_std",
    "null_brain_conc05_mean",
    "null_brain_conc05_std",
    # --- delta-to-shuffle (brain domain) ---
    "delta_brain_rho",
    "delta_brain_conc05",
    # --- shuffle null (within-R domain, secondary) ---
    "null_R_rho_mean",
    "null_R_rho_std",
    "null_R_conc05_mean",
    "null_R_conc05_std",
    # --- delta-to-shuffle (R domain) ---
    "delta_R_rho",
    "delta_R_conc05",
    # --- S3: intensity decile means of |r| ---
    "decile_01",
    "decile_02",
    "decile_03",
    "decile_04",
    "decile_05",
    "decile_06",
    "decile_07",
    "decile_08",
    "decile_09",
    "decile_10",
]


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------


def concentration_q(
    abs_resid: NDArray[np.floating],
    t1c: NDArray[np.floating],
    q: float,
) -> float:
    """Top-q% bright-voxel error mass concentration ratio.

    Conc(q) = (fraction of total |r| mass in the top-q% brightest T1c voxels)
              / (realised fraction |B_q| / |R|)

    Uses the **realised** denominator |B_q| / |R| = ceil(q·n) / n rather than
    the nominal q to avoid bias from ties at the quantile boundary.

    Under independence (|r| ⊥ T1c), E[Conc(q)] = 1 analytically.
    Conc > 1 ⇒ errors disproportionately concentrated in bright voxels.

    Parameters
    ----------
    abs_resid :
        Absolute residual values in region R, flat, shape ``(n,)``.
    t1c :
        Real T1c intensity values in region R, flat, shape ``(n,)``.
    q :
        Quantile fraction, e.g. ``0.05`` for the top 5%.

    Returns
    -------
    float
        Conc(q), or ``nan`` if the region is empty or total mass is zero.
    """
    n = len(abs_resid)
    if n == 0:
        return float("nan")
    total_mass = float(abs_resid.sum())
    if total_mass == 0.0:
        return float("nan")
    n_top = max(1, int(np.ceil(q * n)))
    # argpartition is O(n); faster than full sort for large n.
    top_idx = np.argpartition(t1c, -n_top)[-n_top:]
    q_realised = n_top / n  # actual fraction, handles ties
    top_mass = float(abs_resid[top_idx].sum())
    return (top_mass / total_mass) / q_realised


def _fast_spearman_rho(x: NDArray[np.floating], y: NDArray[np.floating]) -> float:
    """Point-estimate Spearman ρ without CI or bootstrap.

    Used exclusively for the shuffle null distribution where only the
    statistic is needed (no CI overhead).

    Parameters
    ----------
    x, y :
        1-D arrays of the same shape.

    Returns
    -------
    float
        Spearman ρ, or ``nan`` if ``len(x) < 3`` or either vector is constant.
    """
    n = len(x)
    if n < 3:
        return float("nan")
    if float(np.ptp(x)) == 0.0 or float(np.ptp(y)) == 0.0:
        return float("nan")
    return float(scipy_stats.spearmanr(x, y).statistic)


def _ksg_mi(
    abs_resid: NDArray[np.floating],
    t1c: NDArray[np.floating],
    *,
    max_voxels: int,
    rng: np.random.Generator,
) -> tuple[float, int]:
    """KSG mutual information (k=5 nearest-neighbour estimator) on a subsample.

    Uses ``sklearn.feature_selection.mutual_info_regression`` which IS the
    Kraskov-Stögbauer-Grassberger estimator cited in the proposal (k=5 default).

    Parameters
    ----------
    abs_resid :
        Absolute residual, flat shape ``(n,)``.
    t1c :
        T1c intensity, flat shape ``(n,)``.
    max_voxels :
        Maximum number of voxels to subsample.  ``n`` is used when
        ``n ≤ max_voxels``.
    rng :
        NumPy random generator for reproducible subsampling.

    Returns
    -------
    (mi, n_used)
        MI estimate and the subsample size actually used.
    """
    n = len(abs_resid)
    if n < 10:
        return float("nan"), n
    if n > max_voxels:
        idx = rng.choice(n, size=max_voxels, replace=False)
        x_sub = abs_resid[idx]
        y_sub = t1c[idx]
        n_used = max_voxels
    else:
        x_sub = abs_resid
        y_sub = t1c
        n_used = n
    mi = float(
        mutual_info_regression(
            x_sub.reshape(-1, 1),
            y_sub,
            n_neighbors=5,
            random_state=int(rng.integers(0, 2**31)),
        )[0]
    )
    logger.debug("KSG MI computed on %d voxels (of %d total)", n_used, n)
    return mi, n_used


def _intensity_decile_means(
    abs_resid: NDArray[np.floating],
    t1c: NDArray[np.floating],
    *,
    n_deciles: int = 10,
) -> NDArray[np.floating]:
    """Mean |r| per T1c intensity decile (Bland-Altman adaptation).

    Partitions R into ``n_deciles`` equal-count bins by T1c intensity and
    returns the mean absolute residual per bin.

    Parameters
    ----------
    abs_resid :
        Absolute residual in region R, flat.
    t1c :
        T1c intensity in region R, flat.
    n_deciles :
        Number of bins (default 10 = deciles).

    Returns
    -------
    NDArray
        Shape ``(n_deciles,)``, mean |r| per bin, low-to-high intensity order.
        Bins with no elements return ``nan``.
    """
    n = len(abs_resid)
    if n == 0:
        return np.full(n_deciles, float("nan"), dtype=np.float64)
    # pd.qcut gives n_deciles equal-count bins
    try:
        labels = pd.qcut(t1c, q=n_deciles, labels=False, duplicates="drop")
    except ValueError:
        # Fewer unique values than n_deciles — return NaN for all
        return np.full(n_deciles, float("nan"), dtype=np.float64)
    out = np.full(n_deciles, float("nan"), dtype=np.float64)
    for d in range(n_deciles):
        mask = labels == d
        if mask.any():
            out[d] = float(abs_resid[mask].mean())
    return out


# ---------------------------------------------------------------------------
# Shuffle null
# ---------------------------------------------------------------------------


def _shuffle_null(
    abs_resid_brain: NDArray[np.floating],
    t1c_brain: NDArray[np.floating],
    region_in_brain: NDArray[np.bool_],
    *,
    n_shuffle: int,
    rng: np.random.Generator,
    q: float,
    domain: str,
) -> dict[str, float]:
    """Per-scan intensity-shuffle null for S1 (ρ) and S2 (Conc(q)).

    Destroys spatial correspondence between |r| and T1c by permuting
    T1c values within a domain, then restricts to region R to compute
    the statistics.  Under independence, E[ρ] = 0 and E[Conc] = 1.

    Parameters
    ----------
    abs_resid_brain :
        |r| for all brain voxels, flat, size ``|brain|``.
    t1c_brain :
        T1c for all brain voxels, flat, size ``|brain|``.
    region_in_brain :
        Bool mask of length ``|brain|``: True for voxels that belong to
        region R.  For C-WB, all True; for C-noT, a subset.
    n_shuffle :
        Number of shuffle draws.
    rng :
        NumPy random generator (caller's).
    q :
        Quantile for Conc(q); this null only evaluates the primary q=0.05.
    domain :
        ``"brain"`` (primary) — shuffle within brain, restrict to R.
        ``"R"`` (secondary)  — shuffle within R directly.

    Returns
    -------
    dict with keys ``rho_mean``, ``rho_std``, ``conc_mean``, ``conc_std``.
    """
    abs_resid_R = abs_resid_brain[region_in_brain]
    t1c_R = t1c_brain[region_in_brain]
    n_R = len(abs_resid_R)

    if n_R < 3:
        nan4 = float("nan")
        return {"rho_mean": nan4, "rho_std": nan4, "conc_mean": nan4, "conc_std": nan4}

    rho_nulls: list[float] = []
    conc_nulls: list[float] = []

    if domain == "brain":
        n_brain = len(abs_resid_brain)
        for _ in range(n_shuffle):
            perm = rng.permutation(n_brain)
            t1c_R_shuffled = t1c_brain[perm][region_in_brain]
            rho_nulls.append(_fast_spearman_rho(abs_resid_R, t1c_R_shuffled))
            conc_nulls.append(concentration_q(abs_resid_R, t1c_R_shuffled, q))
    elif domain == "R":
        for _ in range(n_shuffle):
            perm = rng.permutation(n_R)
            t1c_R_shuffled = t1c_R[perm]
            rho_nulls.append(_fast_spearman_rho(abs_resid_R, t1c_R_shuffled))
            conc_nulls.append(concentration_q(abs_resid_R, t1c_R_shuffled, q))
    else:
        raise ValueError(f"domain must be 'brain' or 'R', got {domain!r}")

    rho_arr = np.array([r for r in rho_nulls if np.isfinite(r)], dtype=np.float64)
    conc_arr = np.array([c for c in conc_nulls if np.isfinite(c)], dtype=np.float64)

    return {
        "rho_mean": float(rho_arr.mean()) if rho_arr.size > 0 else float("nan"),
        "rho_std": float(rho_arr.std(ddof=0)) if rho_arr.size > 1 else float("nan"),
        "conc_mean": float(conc_arr.mean()) if conc_arr.size > 0 else float("nan"),
        "conc_std": float(conc_arr.std(ddof=0)) if conc_arr.size > 1 else float("nan"),
    }


def shuffle_convergence_check(
    abs_resid_R: NDArray[np.floating],
    t1c_R: NDArray[np.floating],
    *,
    n_list: tuple[int, ...] = (10, 50, 100, 500),
    q: float = 0.05,
    rng_seed: int = 42,
) -> dict[int, dict[str, float]]:
    """Empirical shuffle-convergence study over one region.

    Runs the R-domain shuffle at each count in ``n_list`` and reports
    mean ± SD of ρ and Conc(q) to confirm ~100 shuffles is sufficient.

    Parameters
    ----------
    abs_resid_R :
        |r| in region R (flat).
    t1c_R :
        T1c in region R (flat).
    n_list :
        Tuple of shuffle counts to evaluate.
    q :
        Conc(q) quantile.
    rng_seed :
        RNG seed for reproducibility.

    Returns
    -------
    dict
        ``{n_shuffle: {"rho_mean": ..., "rho_std": ...,
                       "conc_mean": ..., "conc_std": ...}}``.
    """
    rng = np.random.default_rng(rng_seed)
    n_brain_dummy = len(abs_resid_R)
    region_all = np.ones(n_brain_dummy, dtype=bool)
    results: dict[int, dict[str, float]] = {}
    for n_s in sorted(n_list):
        results[n_s] = _shuffle_null(
            abs_resid_R,
            t1c_R,
            region_all,
            n_shuffle=n_s,
            rng=rng,
            q=q,
            domain="R",
        )
    return results


# ---------------------------------------------------------------------------
# Per-scan row computation
# ---------------------------------------------------------------------------


def _nan_row(sample: ScanSample, condition: str) -> dict:
    """Empty NaN row when a region has zero voxels."""
    row: dict = {c: float("nan") for c in SPATIAL_CSV_COLUMNS}
    row.update(
        {
            "method": sample.method,
            "cohort": sample.cohort,
            "ring": sample.ring,
            "nfe": sample.nfe,
            "scan_id": sample.scan_id,
            "patient_id": sample.patient_id,
            "condition": condition,
            "pred_mode": sample.pred_mode,
            "raw_p995": sample.raw_p995,
            "n_voxels_brain": 0,
            "n_voxels_region": 0,
            "mi_n_sampled": 0,
        }
    )
    return row


def compute_scan_rows(
    sample: ScanSample,
    *,
    dilate_k: int = 5,
    n_shuffles: int = 100,
    n_boot: int = 100,
    rng_seed: int = 42,
    mi_n_voxels: int = 30_000,
    n_deciles: int = 10,
) -> list[dict]:
    """Compute all spatial residual metrics for one scan.

    Returns two dicts (one per condition: C-WB and C-noT) ready for the
    per-scan CSV.  Empty regions return NaN rows (counted and reported).

    The residual is recomputed on the fly: ``r = real − pred``.  Neither r
    nor |r| is stored beyond this call (contracts §11 trap 10).

    Parameters
    ----------
    sample :
        One joined scan from :func:`vena.validation.io.iter_scans`.
    dilate_k :
        WT dilation kernel size (odd integer).  Default 5 = radius 2.
    n_shuffles :
        Shuffle draws per domain per condition.  Default 100; use 10 for
        smoke/fast runs.
    n_boot :
        Bootstrap draws for the per-scan Spearman CI.  Default 100; the
        per-paper CI uses 10,000 patient-stratified resamples (aggregate step).
    rng_seed :
        RNG seed for reproducibility.
    mi_n_voxels :
        Maximum voxel subsample for KSG MI estimation.
    n_deciles :
        Number of intensity decile bins (S3 Bland-Altman plot).

    Returns
    -------
    list[dict]
        Two rows: one for ``"C-WB"`` and one for ``"C-noT"``.
    """
    # Lazy import: regions.py imports torch at top level; deferring keeps
    # spatial_residual.py's module-level footprint torch-free (§2 isolation).
    from vena.validation.regions import region_masks

    brain_bool = sample.brain.astype(bool)
    wt_bool = sample.wt.astype(bool)

    masks = region_masks(brain_bool, wt_bool, dilate_k=dilate_k)

    # Compute residual on the fly — never retained beyond this scope.
    abs_resid_vol = np.abs(sample.real.astype(np.float64) - sample.pred.astype(np.float64))
    t1c_vol = sample.real.astype(np.float64)

    # Flatten to brain voxels (shared allocation for both conditions).
    brain_flat = masks["brain"].ravel()
    brain_indices = np.where(brain_flat)[0]
    n_brain = len(brain_indices)
    abs_resid_brain = abs_resid_vol.ravel()[brain_indices]
    t1c_brain = t1c_vol.ravel()[brain_indices]

    rng = np.random.default_rng(rng_seed)

    conditions: dict[str, NDArray[np.bool_]] = {
        "C-WB": masks["brain"],  # whole brain
        "C-noT": masks["bg"],  # brain \ dilate(WT, k) — DO NOT use bg_undilated
    }

    rows: list[dict] = []

    for cond_name, region_mask in conditions.items():
        region_flat = region_mask.ravel()
        # Which brain voxels belong to this region?
        region_in_brain = region_flat[brain_flat]  # bool, length n_brain
        n_region = int(region_in_brain.sum())

        if n_region < 3:
            logger.warning(
                "%s | %s | nfe=%d | scan=%s | condition=%s: n_region=%d < 3, returning NaN",
                sample.method,
                sample.cohort,
                sample.nfe,
                sample.scan_id,
                cond_name,
                n_region,
            )
            rows.append(_nan_row(sample, cond_name))
            continue

        abs_resid_R = abs_resid_brain[region_in_brain]
        t1c_R = t1c_brain[region_in_brain]

        # S1 — Spearman ρ with bootstrap CI (do not subsample).
        spr: SpearmanResult = spearman_with_bootstrap_ci(
            abs_resid_R, t1c_R, n_boot=n_boot, seed=int(rng.integers(0, 2**31))
        )

        # S2 — Conc(q) for q ∈ {1%, 5%, 10%}.
        conc_01 = concentration_q(abs_resid_R, t1c_R, 0.01)
        conc_05 = concentration_q(abs_resid_R, t1c_R, 0.05)
        conc_10 = concentration_q(abs_resid_R, t1c_R, 0.10)

        # KSG MI (exploratory, subsampled).
        mi_ksg, mi_n_used = _ksg_mi(abs_resid_R, t1c_R, max_voxels=mi_n_voxels, rng=rng)

        # Shuffle null — brain domain (primary, proposal default).
        null_brain = _shuffle_null(
            abs_resid_brain,
            t1c_brain,
            region_in_brain,
            n_shuffle=n_shuffles,
            rng=rng,
            q=0.05,
            domain="brain",
        )

        # Shuffle null — R domain (secondary, for reviewer robustness).
        null_R = _shuffle_null(
            abs_resid_brain,
            t1c_brain,
            region_in_brain,
            n_shuffle=n_shuffles,
            rng=rng,
            q=0.05,
            domain="R",
        )

        # S3 — intensity decile means.
        deciles = _intensity_decile_means(abs_resid_R, t1c_R, n_deciles=n_deciles)

        row: dict = {
            "method": sample.method,
            "cohort": sample.cohort,
            "ring": sample.ring,
            "nfe": sample.nfe,
            "scan_id": sample.scan_id,
            "patient_id": sample.patient_id,
            "condition": cond_name,
            "pred_mode": sample.pred_mode,
            "raw_p995": sample.raw_p995,
            "n_voxels_brain": n_brain,
            "n_voxels_region": n_region,
            "rho_s": spr.rho,
            "rho_s_lo": spr.rho_lo,
            "rho_s_hi": spr.rho_hi,
            "rho_s_p": spr.p_value,
            "conc_01": conc_01,
            "conc_05": conc_05,
            "conc_10": conc_10,
            "mi_ksg": mi_ksg,
            "mi_n_sampled": mi_n_used,
            "null_brain_rho_mean": null_brain["rho_mean"],
            "null_brain_rho_std": null_brain["rho_std"],
            "null_brain_conc05_mean": null_brain["conc_mean"],
            "null_brain_conc05_std": null_brain["conc_std"],
            "delta_brain_rho": (
                spr.rho - null_brain["rho_mean"]
                if np.isfinite(spr.rho) and np.isfinite(null_brain["rho_mean"])
                else float("nan")
            ),
            "delta_brain_conc05": (
                conc_05 - null_brain["conc_mean"]
                if np.isfinite(conc_05) and np.isfinite(null_brain["conc_mean"])
                else float("nan")
            ),
            "null_R_rho_mean": null_R["rho_mean"],
            "null_R_rho_std": null_R["rho_std"],
            "null_R_conc05_mean": null_R["conc_mean"],
            "null_R_conc05_std": null_R["conc_std"],
            "delta_R_rho": (
                spr.rho - null_R["rho_mean"]
                if np.isfinite(spr.rho) and np.isfinite(null_R["rho_mean"])
                else float("nan")
            ),
            "delta_R_conc05": (
                conc_05 - null_R["conc_mean"]
                if np.isfinite(conc_05) and np.isfinite(null_R["conc_mean"])
                else float("nan")
            ),
        }
        for d_i, d_val in enumerate(deciles, 1):
            row[f"decile_{d_i:02d}"] = float(d_val)

        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Aggregate statistics (patient-level)
# ---------------------------------------------------------------------------


def _filter_to_selection_nfe(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Return only the rows for *method* at its headline selection_nfe.

    After :func:`collapse_to_patient`, a sweep that covers multiple NFE values
    produces duplicate ``patient_id`` index entries (one per NFE).  Selecting
    only the selection_nfe rows guarantees unique patient_ids and makes the
    :func:`paired_wilcoxon` inner-join well-defined.

    Falls back to all NFEs for that method when the selection_nfe is not found
    in the data (e.g. a sweep that ran only nfe=1 for a method whose canonical
    nfe is 5).  Logs a warning in the fallback branch.

    Parameters
    ----------
    df :
        Patient-level DataFrame with ``method`` and ``nfe`` columns.
    method :
        Method key (e.g. ``"C0-Identity"``).

    Returns
    -------
    pd.DataFrame
        Rows for *method* filtered to its selection_nfe, or all rows for the
        method when the canonical nfe is absent from the data.
    """
    method_rows = df[df["method"] == method]
    nfe_target = _SELECTION_NFE.get(method)
    if nfe_target is None:
        # Unknown method — take rows with the highest nfe present (conservative).
        nfe_target = int(method_rows["nfe"].max()) if not method_rows.empty else None
        logger.warning(
            "_filter_to_selection_nfe: unknown method %r; falling back to nfe=%s",
            method,
            nfe_target,
        )
        return method_rows if nfe_target is None else method_rows[method_rows["nfe"] == nfe_target]

    filtered = method_rows[method_rows["nfe"] == nfe_target]
    if filtered.empty and not method_rows.empty:
        # Smoke or partial sweep: selection_nfe not present; take what we have.
        available = sorted(method_rows["nfe"].unique())
        # Pick the value closest to selection_nfe (avoids nfe=1 when nfe=5 intended).
        closest = min(available, key=lambda x: abs(x - nfe_target))
        logger.warning(
            "_filter_to_selection_nfe: %r selection_nfe=%d not found; "
            "falling back to closest available nfe=%d",
            method,
            nfe_target,
            closest,
        )
        return method_rows[method_rows["nfe"] == closest]
    return filtered


@dataclass(frozen=True)
class WilcoxonTestResult:
    """Result of one paired Wilcoxon test with Holm correction."""

    competitor: str
    stat_name: str  # "rho_s" or "conc_05"
    condition: str  # "C-noT" or "C-WB"
    statistic: float
    pvalue_raw: float
    pvalue_adj: float
    reject: bool
    n_pairs: int
    cliffs_delta: float


def aggregate_patient_tests(
    per_scan_df: pd.DataFrame,
    *,
    vena_method: str,
    condition: str = "C-noT",
    ring: str = "A",
) -> tuple[pd.DataFrame, list[WilcoxonTestResult]]:
    """Collapse scans → patients, run Wilcoxon + Holm (family of 16).

    Must be called with a DataFrame that includes VENA and all 8 competitors
    for the assertion on family size to hold.

    Parameters
    ----------
    per_scan_df :
        Tidy per-scan CSV (one row per scan × method × condition).
    vena_method :
        Key of the VENA headline method, e.g. ``"VENA-S1-v3b-rw"``.
    condition :
        Which condition to test.  ``"C-noT"`` is the primary family.
    ring :
        Test ring; only ring-``ring`` rows are included.

    Returns
    -------
    (patient_df, test_results)
        ``patient_df``: one row per patient × method × condition (collapsed).
        ``test_results``: list of :class:`WilcoxonTestResult`.
    """
    # Filter to ring and condition.
    df = per_scan_df[(per_scan_df["ring"] == ring) & (per_scan_df["condition"] == condition)].copy()

    # Collapse scans → patients (critical: LUMIERE has 72 scans / 11 patients).
    patient_df = collapse_to_patient(
        df,
        value_cols=[
            "rho_s",
            "conc_01",
            "conc_05",
            "conc_10",
            "delta_brain_rho",
            "delta_brain_conc05",
            "delta_R_rho",
            "delta_R_conc05",
        ],
        by=("method", "cohort", "ring", "nfe", "patient_id"),
    )

    # Identify competitors present.
    methods_present = set(patient_df["method"].unique())
    competitors = [m for m in methods_present if m != vena_method]

    # Build Holm family: 2 stats × all present competitors.
    stats_to_test: list[str] = ["rho_s", "conc_05"]
    pvalue_dict: dict[str, float] = {}
    raw_results: list[tuple[str, str, float, int, float]] = []

    # D1 fix: filter VENA and each competitor to their selection_nfe BEFORE
    # indexing by patient_id.  Without this, methods with multiple NFE values
    # produce duplicate patient_id index entries; vena.loc[common] then returns
    # more rows than competitor.loc[common], making (v - c) a shape-mismatch
    # that raises ValueError → silently caught → n_pairs=0.
    # Fix: each arm contributes exactly one row per patient at its headline NFE.
    vena_pat = _filter_to_selection_nfe(patient_df, vena_method).set_index("patient_id")

    for comp in competitors:
        comp_pat = _filter_to_selection_nfe(patient_df, comp).set_index("patient_id")
        for stat in stats_to_test:
            key = f"{comp}::{stat}"
            try:
                wx = paired_wilcoxon(
                    vena_pat[stat].dropna(),
                    comp_pat[stat].dropna(),
                )
                pvalue_dict[key] = wx.pvalue
                cd = cliffs_delta(
                    vena_pat[stat].dropna().to_numpy(),
                    comp_pat[stat].dropna().to_numpy(),
                )
                raw_results.append((comp, stat, wx.pvalue, wx.n, cd))
            except (ValueError, KeyError) as exc:
                logger.warning("Wilcoxon failed for %s / %s: %s", comp, stat, exc)
                pvalue_dict[key] = float("nan")
                raw_results.append((comp, stat, float("nan"), 0, float("nan")))

    # Holm-Bonferroni correction.
    holm = holm_bonferroni({k: v for k, v in pvalue_dict.items() if np.isfinite(v)})

    results: list[WilcoxonTestResult] = []
    for comp, stat, praw, n_pairs, cd in raw_results:
        key = f"{comp}::{stat}"
        holm_r = holm.get(key)
        results.append(
            WilcoxonTestResult(
                competitor=comp,
                stat_name=stat,
                condition=condition,
                statistic=float("nan"),  # Wilcoxon statistic merged in pvalue_dict
                pvalue_raw=praw,
                pvalue_adj=holm_r.pvalue_adj if holm_r else float("nan"),
                reject=holm_r.reject if holm_r else False,
                n_pairs=n_pairs,
                cliffs_delta=cd,
            )
        )

    # Diagnostic: warn if family size ≠ 16.
    family_size = len([k for k in pvalue_dict if np.isfinite(pvalue_dict[k])])
    if family_size != HOLM_FAMILY_SIZE:
        logger.warning(
            "Holm family has %d tests (expected %d = %d stats × %d competitors). "
            "Ensure all 8 competitors are present before drawing conclusions.",
            family_size,
            HOLM_FAMILY_SIZE,
            _N_STATS,
            _N_COMPETITORS,
        )

    return patient_df, results


# ---------------------------------------------------------------------------
# Bootstrap CI for table (patient-stratified, 10,000 resamples)
# ---------------------------------------------------------------------------


def patient_bootstrap_ci(
    patient_df: pd.DataFrame,
    *,
    stat: str,
    method: str,
    condition: str,
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 1337,
) -> tuple[float, float, float]:
    """Bootstrap CI for a patient-level mean (for Table 3).

    Parameters
    ----------
    patient_df :
        Collapsed patient-level DataFrame.
    stat :
        Column name (e.g. ``"rho_s"``, ``"conc_05"``).
    method :
        Method key to filter on.
    condition :
        Condition key to filter on.
    n_boot :
        Bootstrap draws (10,000 per proposal §4.3.3).
    ci :
        Coverage probability.
    seed :
        RNG seed.

    Returns
    -------
    (mean, lo, hi)
        Point estimate and bootstrap CI bounds.
    """
    vals = (
        patient_df[(patient_df["method"] == method) & (patient_df["condition"] == condition)][stat]
        .dropna()
        .to_numpy()
    )
    lo, hi = bootstrap_ci(vals, n_boot=n_boot, ci=ci, seed=seed)
    return float(vals.mean()), lo, hi
