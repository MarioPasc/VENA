"""Phase 3 — augmentation drift gate, per-cohort ratio, empty-WT (§4.7b).

Per (cohort, patient, variant) cell:

1. Compute the per-block WT-vs-notWT ratio ``r = mean(|delta_phi|_WT) /
   mean(|delta_phi|_notWT)`` where ``delta_phi = phi(z_t1c) -
   phi(z_t1pre)`` (the §2.6 empirical anchor's quantity).
2. Compare the variant's ratio against the same patient's ``v0`` ratio
   per the §4.7b drift formula. Variants whose drift > 0.20 on > 25% of
   patient-block pairs are rejected from the production augmentation set.
3. Track empty-WT count per cohort (the §2.6 ``max(|Omega|, 1)`` guard
   utilisation).
4. Track per-cohort median W/nW ratio for the §4.7c inter-cohort spread
   check.

This phase reads pre-computed per-block per-region distances produced
by phase 2; it does no decoder work of its own.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

# distances[(cohort, patient_id, variant)][block] = {"WT": float, "notWT": float}
DistanceTable = Mapping[tuple[str, str, str], Mapping[int, Mapping[str, float]]]


def compute_drift(
    distances: DistanceTable,
    *,
    blocks: tuple[int, ...],
) -> list[dict]:
    """For every (cohort, patient, variant ≠ v0, block), compute the §4.7b drift.

    Returns
    -------
    list of records suitable for ``drift_per_patient_variant.csv``:
    keys ``cohort``, ``patient_id``, ``variant``, ``block_idx``,
    ``ratio_variant``, ``ratio_v0``, ``drift_value``, ``passes_gate``.
    """
    out: list[dict] = []
    # Group v0 entries for quick lookup.
    v0_lookup: dict[tuple[str, str], Mapping[int, Mapping[str, float]]] = {}
    for (cohort, patient_id, variant), per_block in distances.items():
        if variant == "v0":
            v0_lookup[(cohort, patient_id)] = per_block

    for (cohort, patient_id, variant), per_block in distances.items():
        if variant == "v0":
            continue
        baseline = v0_lookup.get((cohort, patient_id))
        if baseline is None:
            continue
        for blk in blocks:
            ratio_v = _safe_ratio(per_block.get(blk, {}))
            ratio_0 = _safe_ratio(baseline.get(blk, {}))
            if ratio_0 is None or ratio_v is None or ratio_0 == 0.0:
                continue
            drift = abs(ratio_v - ratio_0) / abs(ratio_0)
            out.append(
                {
                    "cohort": cohort,
                    "patient_id": patient_id,
                    "variant": variant,
                    "block_idx": int(blk),
                    "ratio_variant": float(ratio_v),
                    "ratio_v0": float(ratio_0),
                    "drift_value": float(drift),
                    "passes_gate": bool(drift < 0.20),
                }
            )
    return out


def _safe_ratio(per_region: Mapping[str, float]) -> float | None:
    wt = per_region.get("WT")
    nw = per_region.get("notWT")
    if wt is None or nw is None or nw == 0.0:
        return None
    return wt / nw


def per_cohort_w_nw_ratio(
    distances: DistanceTable,
    *,
    block: int,
    variant: str = "v0",
) -> list[dict]:
    """For each cohort, the median W/nW ratio at a chosen block (default v0).

    Drives the §4.7c inter-cohort spread check; the aggregator emits
    per-cohort overrides when ``max / min > 1.5``.
    """
    per_cohort: dict[str, list[float]] = {}
    for (cohort, _patient_id, var), per_block in distances.items():
        if var != variant:
            continue
        ratio = _safe_ratio(per_block.get(block, {}))
        if ratio is None:
            continue
        per_cohort.setdefault(cohort, []).append(float(ratio))
    out: list[dict] = []
    for cohort, ratios in per_cohort.items():
        if not ratios:
            continue
        arr = np.array(ratios)
        out.append(
            {
                "cohort": cohort,
                "n_patients": int(arr.size),
                "ratio_median": float(np.median(arr)),
                "ratio_p25": float(np.quantile(arr, 0.25)),
                "ratio_p75": float(np.quantile(arr, 0.75)),
            }
        )
    if len(out) >= 2:
        medians = np.array([r["ratio_median"] for r in out])
        spread = float(medians.max() / max(medians.min(), 1e-12))
        for r in out:
            r["exceeds_global_recipe"] = spread > 1.5
    else:
        for r in out:
            r["exceeds_global_recipe"] = False
    return out


def empty_wt_rate(
    wt_volumes: Mapping[tuple[str, str], float],
    cohorts: Iterable[str],
) -> list[dict]:
    """For each cohort, fraction of patients with ``|m_wt| = 0``.

    Parameters
    ----------
    wt_volumes : Mapping[(cohort, patient_id), float]
        Per-patient WT-soft-sum at latent resolution.
    cohorts : Iterable[str]
        Cohorts to report (in fixed order).
    """
    out: list[dict] = []
    for cohort in cohorts:
        rows = [v for (c, _p), v in wt_volumes.items() if c == cohort]
        if not rows:
            out.append({"cohort": cohort, "n_total": 0, "n_empty_wt": 0, "fraction": 0.0})
            continue
        n_total = len(rows)
        n_empty = int(sum(1 for v in rows if v <= 0.5))
        out.append(
            {
                "cohort": cohort,
                "n_total": n_total,
                "n_empty_wt": n_empty,
                "fraction": float(n_empty) / float(n_total),
            }
        )
    return out


def decide_allowed_variants(
    drift_rows: Iterable[dict],
    *,
    variants: tuple[str, ...],
    fail_fraction: float = 0.25,
) -> list[str]:
    """A variant is allowed when ≤ ``fail_fraction`` of its patient-block
    pairs failed the §4.7b drift gate.
    """
    by_variant: dict[str, list[bool]] = {}
    for row in drift_rows:
        by_variant.setdefault(row["variant"], []).append(bool(row["passes_gate"]))
    allowed: list[str] = ["v0"]  # the clean variant is always allowed
    for v in variants:
        if v == "v0":
            continue
        results = by_variant.get(v, [])
        if not results:
            continue
        fail_rate = 1.0 - (sum(results) / len(results))
        if fail_rate <= fail_fraction:
            allowed.append(v)
    return allowed


def detect_v4_brain_mask_status(
    drift_rows: Iterable[dict],
) -> str:
    """Detect the 2026-06-18 data-audit v4 brain-mask inflation pattern.

    The audit reported a uniform ≈3× ratio inflation on v4 across every
    cohort at block 5 (BraTS-GLI 0.82→2.77, LUMIERE 1.61→5.17, etc.). If
    every v4 row at block 5 shows a ratio ≥ 2× the cohort's v0 ratio,
    flag the status as ``broken_drop_v4`` so the aggregator drops v4
    from ``allowed_variants`` regardless of the drift gate.
    """
    v4_b5 = [r for r in drift_rows if r["variant"] == "v4" and r["block_idx"] == 5]
    if not v4_b5:
        return "ok"
    inflated = sum(1 for r in v4_b5 if r["ratio_variant"] >= 2.0 * r["ratio_v0"])
    return "broken_drop_v4" if inflated >= 0.8 * len(v4_b5) else "ok"


__all__ = [
    "DistanceTable",
    "compute_drift",
    "decide_allowed_variants",
    "detect_v4_brain_mask_status",
    "empty_wt_rate",
    "per_cohort_w_nw_ratio",
]
