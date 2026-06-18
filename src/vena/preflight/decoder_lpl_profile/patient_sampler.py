"""Patient selection for the ``decoder_lpl_profile`` preflight.

Picks N patients per cohort stratified by WT-volume tertile so the
§4.7b coverage requirement ("≥3 patients per cohort spanning small /
median / large WT volumes") lands on a fair sample. The WT volume is
the per-patient soft-sum of ``masks/tumor_latent`` (channels NETC + ED +
ET) at latent resolution.

The sampler operates on a single cohort's latent H5 at a time so it can
be reused without loading every registry entry. The orchestrator (engine)
wraps it in a per-cohort loop.

A deterministic ``seed`` controls in-stratum sampling so the same config
always picks the same patients, which is the load-bearing reproducibility
contract on a sweep that drives downstream w_l / region recipe pinning.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class PatientPick:
    """One chosen patient + the stratum it represents."""

    patient_id: str
    row_index: int
    wt_volume: float  # soft-sum, voxels in latent space
    stratum: str  # "small" | "median" | "large"


def _wt_soft_volumes(h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(ids, wt_volumes)`` aligned by row index.

    The per-row volume is ``sum(clip(sum_channels(tumor_latent), 0, 1))`` —
    the same proxy used by the LPL loss for the soft-WT membership.
    """
    with h5py.File(h5_path, "r") as f:
        ids = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in f["ids"][:]])
        tumor = f["masks/tumor_latent"]  # (N, 3, h, w, d)
        n = tumor.shape[0]
        volumes = np.empty(n, dtype=np.float64)
        # Read row-by-row to stay light on memory for large cohorts.
        for i in range(n):
            row = tumor[i]
            soft = np.clip(row.sum(axis=0), 0.0, 1.0)
            volumes[i] = float(soft.sum())
    return ids, volumes


def select_patients_by_strata(
    h5_path: Path,
    *,
    n_per_cohort: int = 3,
    volume_strata: Iterable[str] = ("small", "median", "large"),
    eligible_ids: Iterable[str] | None = None,
    seed: int = 42,
) -> list[PatientPick]:
    """Pick ``n_per_cohort`` patients spanning the volume tertiles.

    Parameters
    ----------
    h5_path : Path
        Cohort's clean latent H5 (``v0`` source).
    n_per_cohort : int, default 3
        Number of patients to return. When equal to ``len(volume_strata)``,
        one per stratum; otherwise ties go to the first stratum in
        ``volume_strata`` order.
    volume_strata : Iterable[str], default ("small", "median", "large")
        Tertile labels in order from smallest to largest volume. Default
        produces the §4.7b 3-way split. Custom orderings can pass e.g.
        ``("small", "large")`` to skip the median, but the implementation
        still computes tertile boundaries.
    eligible_ids : Iterable[str] | None
        If supplied, the sampler restricts to these patient ids (the
        usual case is "train + val split" — the engine pulls those from
        the cohort registry). ``None`` → use every row in the H5.
    seed : int
        Deterministic in-stratum sampling seed.

    Returns
    -------
    list[PatientPick]
        Length ≤ ``n_per_cohort`` (one stratum may be empty on tiny
        cohorts; the sampler does not over-draw to compensate).
    """
    ids, volumes = _wt_soft_volumes(h5_path)
    if eligible_ids is not None:
        eligible = set(str(x) for x in eligible_ids)
        mask = np.array([i in eligible for i in ids])
        idx = np.nonzero(mask)[0]
        ids = ids[idx]
        volumes = volumes[idx]
        row_map = idx
    else:
        row_map = np.arange(len(ids))

    if ids.size == 0:
        return []

    # Tertile boundaries (33rd / 67th percentile). When the cohort is
    # tiny (n < 3), boundaries collapse and the sampler may return all
    # rows in the first stratum.
    strata = list(volume_strata)
    n_strata = len(strata)
    quantiles = np.linspace(0.0, 1.0, n_strata + 1)[1:-1]
    cuts = np.quantile(volumes, quantiles) if quantiles.size > 0 else np.array([])

    rng = np.random.default_rng(seed)

    def _bucket_of(v: float) -> int:
        for k, cut in enumerate(cuts):
            if v < cut:
                return k
        return n_strata - 1

    # Group row indices by stratum, then sample one per stratum.
    buckets: list[list[int]] = [[] for _ in range(n_strata)]
    for k, v in enumerate(volumes):
        buckets[_bucket_of(float(v))].append(k)

    picks: list[PatientPick] = []
    for stratum_idx in range(min(n_per_cohort, n_strata)):
        members = buckets[stratum_idx]
        if not members:
            continue
        chosen = int(rng.choice(members))
        picks.append(
            PatientPick(
                patient_id=str(ids[chosen]),
                row_index=int(row_map[chosen]),
                wt_volume=float(volumes[chosen]),
                stratum=strata[stratum_idx],
            )
        )
    # If we want MORE picks than strata, fill from the largest-volume
    # stratum first (highest signal) — this is rare and only happens when
    # the operator overrides n_per_cohort above 3.
    while len(picks) < n_per_cohort and any(buckets):
        # Pull from the rightmost non-empty bucket.
        for stratum_idx in range(n_strata - 1, -1, -1):
            members = [
                m
                for m in buckets[stratum_idx]
                if int(row_map[m]) not in {p.row_index for p in picks}
            ]
            if not members:
                continue
            chosen = int(rng.choice(members))
            picks.append(
                PatientPick(
                    patient_id=str(ids[chosen]),
                    row_index=int(row_map[chosen]),
                    wt_volume=float(volumes[chosen]),
                    stratum=strata[stratum_idx] + "_extra",
                )
            )
            break
        else:
            break
    return picks


__all__ = ["PatientPick", "select_patients_by_strata"]
