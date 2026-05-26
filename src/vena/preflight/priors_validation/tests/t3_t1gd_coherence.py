"""Test T3 — T1Gd coherence (spec §5.3, the central test).

ΔT1 = robust_zscore(T1Gd, brain) − robust_zscore(T1pre, brain).
For each (prior, ROI), Spearman(prior, ΔT1) with bootstrap 95 % CI.
Pass iff sign agrees with expectation and |ρ| lies in the expected band.

ROIs:
* ``tum``      — subject tumour mask
* ``sinus``    — atlas-warped venous-sinus mask (eroded by 2 mm)
* ``healthy``  — brain ∖ tumour ∖ sinus, intersected with NAWM + cortical GM
* ``pituitary`` — atlas pituitary (skipped in v0 — no HO pituitary ROI)
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar

import numpy as np
from scipy.ndimage import binary_erosion

from ..core.config import T3_EXPECTATIONS, CoherenceExpectation
from ..core.dataclasses import SubjectInputs, TestOutcome
from ..core.interfaces import ValidationTest
from ..statistics import bh_fdr, spearman_with_bootstrap_ci
from .base import TestContext, roi_mask_from_atlas
from .t1_range_sanity import _prior_volume_for
from .t2_atlas_localisation import _derived_volume

_ROIS = ("tum", "sinus", "healthy", "pituitary")


def _build_t3_roi(roi_id: str, ctx: TestContext) -> tuple[np.ndarray | None, str | None]:
    sub = ctx.subject
    brain = np.asarray(sub.brain_mask.array) > 0
    if roi_id == "tum":
        if sub.tumour_mask is None:
            return None, "tumour mask absent"
        return (np.asarray(sub.tumour_mask.array) > 0) & brain, None
    if roi_id == "sinus":
        from ..atlases.registry import ATLAS_REGISTRY, VENOUS_INHOUSE

        if VENOUS_INHOUSE not in ctx.atlas_labels:
            return None, "venous in-house atlas not warped"
        sinus = roi_mask_from_atlas(
            ctx.atlas_labels[VENOUS_INHOUSE],
            ATLAS_REGISTRY["sinus"].label_values,
            extra_mask=brain,
        )
        # Erode by 2 mm (≈ 2 voxels at 1 mm isotropic UCSF-PDGM).
        sinus = binary_erosion(sinus, iterations=2)
        if not sinus.any():
            return None, "sinus mask empty after 2-mm erosion"
        return sinus, None
    if roi_id == "healthy":
        if ctx.nawm_mask is None:
            return None, "NAWM mask not built"
        from ..atlases.registry import ATLAS_REGISTRY, HO_SUB

        if HO_SUB not in ctx.atlas_labels:
            cortex = np.zeros_like(brain, dtype=bool)
        else:
            cortex = roi_mask_from_atlas(
                ctx.atlas_labels[HO_SUB],
                ATLAS_REGISTRY["cortical_gm"].label_values,
                extra_mask=brain,
            )
        healthy = (ctx.nawm_mask | cortex) & brain
        if sub.tumour_mask is not None:
            healthy &= ~(np.asarray(sub.tumour_mask.array) > 0)
        # Subtract sinus if available
        if "venous_inhouse" in ctx.atlas_labels:
            from ..atlases.registry import ATLAS_REGISTRY as _AR

            sinus = roi_mask_from_atlas(
                ctx.atlas_labels["venous_inhouse"],
                _AR["sinus"].label_values,
                extra_mask=brain,
            )
            healthy &= ~binary_erosion(sinus, iterations=2)
        return (healthy if healthy.any() else None), (
            None if healthy.any() else "healthy ROI empty"
        )
    if roi_id == "pituitary":
        return None, "pituitary ROI not in v0 atlas bundle"
    return None, f"unknown T3 ROI {roi_id!r}"


def _classify_failure(rho: float, exp: CoherenceExpectation) -> str:
    """Spec §5.3 failure-mode taxonomy."""
    if not np.isfinite(rho):
        return "no_signal"
    sign_obs = 0 if abs(rho) < 1e-6 else (1 if rho > 0 else -1)
    sign_exp = exp.sign
    if sign_exp != 0 and sign_obs != 0 and sign_obs != sign_exp:
        return "wrong_sign"
    if abs(rho) < 0.1:
        return "near_zero_magnitude"
    if exp.sign != 0 and abs(rho) < max(exp.rho_lo, 0.1):
        return "right_sign_weak"
    return "ok"


def _expected_for(prior: str, roi: str) -> CoherenceExpectation | None:
    """Look up the T3 expectation, falling back to neighbouring channel aliases."""
    if (prior, roi) in T3_EXPECTATIONS:
        return T3_EXPECTATIONS[(prior, roi)]
    # Alias: cbf ↔ cbf_rel, adc ↔ adc_rel
    alias = {"cbf_rel": "cbf", "cbf": "cbf_rel", "adc_rel": "adc", "adc": "adc_rel"}
    if (alias.get(prior, prior), roi) in T3_EXPECTATIONS:
        return T3_EXPECTATIONS[(alias.get(prior, prior), roi)]
    return None


class T3T1GdCoherence(ValidationTest):
    """The central test: prior↔ΔT1 voxelwise Spearman per (prior, ROI)."""

    test_id: ClassVar[str] = "T3_t1gd_coherence"
    name: ClassVar[str] = "T3 T1Gd coherence"

    def __init__(self, n_boot: int = 1000, fdr_q: float = 0.05, seed: int = 1337) -> None:
        self.n_boot = int(n_boot)
        self.fdr_q = float(fdr_q)
        self.seed = int(seed)

    def applicable(self, inputs: SubjectInputs) -> bool:
        return bool(inputs.derived_priors) or any(
            getattr(inputs, p, None) is not None for p in ("cbf", "adc", "chi")
        )

    def run(self, inputs: SubjectInputs, ctx: TestContext | None = None) -> Iterable[TestOutcome]:  # type: ignore[override]
        if ctx is None or ctx.delta_t1 is None:
            raise RuntimeError("T3 requires a TestContext with delta_t1 precomputed")
        delta = ctx.delta_t1
        sub = inputs

        # Candidate priors: every (prior, roi) tuple in T3_EXPECTATIONS that we
        # can resolve to a volume.
        candidate_pairs: list[tuple[str, str, np.ndarray]] = []
        for prior, roi in T3_EXPECTATIONS:
            if prior in ("chi_pos", "chi_neg"):
                # QSM channels not produced in v0
                continue
            arr = _prior_volume_for(prior, sub)
            if arr is None:
                arr = _derived_volume(prior, sub)
            if arr is None:
                continue
            candidate_pairs.append((prior, roi, arr))

        # Pass 1 — compute all (rho, p) without yielding (so we can FDR-adjust)
        intermediate: list[dict] = []
        for prior, roi, arr in candidate_pairs:
            roi_mask, why = _build_t3_roi(roi, ctx)
            if roi_mask is None:
                intermediate.append(
                    {
                        "prior": prior,
                        "roi": roi,
                        "rho": float("nan"),
                        "rho_lo": float("nan"),
                        "rho_hi": float("nan"),
                        "p": float("nan"),
                        "n": 0,
                        "skip_reason": why,
                    }
                )
                continue
            valid = roi_mask & np.isfinite(arr) & np.isfinite(delta)
            n_valid = int(valid.sum())
            if n_valid < 50:  # below this, Spearman is noise
                intermediate.append(
                    {
                        "prior": prior,
                        "roi": roi,
                        "rho": float("nan"),
                        "rho_lo": float("nan"),
                        "rho_hi": float("nan"),
                        "p": float("nan"),
                        "n": n_valid,
                        "skip_reason": f"ROI has only {n_valid} valid voxels",
                    }
                )
                continue
            x = arr[valid].astype(np.float64)
            y = delta[valid].astype(np.float64)
            # Spearman over the full ROI is the principal estimate; the
            # bootstrap is on the same voxel population (i.i.d. resample).
            res = spearman_with_bootstrap_ci(x, y, n_boot=self.n_boot, seed=self.seed)
            intermediate.append(
                {
                    "prior": prior,
                    "roi": roi,
                    "rho": res.rho,
                    "rho_lo": res.rho_lo,
                    "rho_hi": res.rho_hi,
                    "p": res.p_value,
                    "n": res.n,
                    "skip_reason": None,
                }
            )

        # Pass 2 — Benjamini–Hochberg FDR across this subject's (prior×ROI) grid
        ps = np.array([r["p"] for r in intermediate], dtype=np.float64)
        _, p_adj = bh_fdr(ps, q=self.fdr_q)

        for entry, p_corrected in zip(intermediate, p_adj, strict=True):
            prior = entry["prior"]
            roi = entry["roi"]
            exp = _expected_for(prior, roi)
            if entry["skip_reason"] is not None:
                yield TestOutcome(
                    test_id=self.test_id,
                    subject_id=sub.subject_id,
                    prior_id=prior,
                    roi_id=roi,
                    metric_name=f"spearman_{prior}_{roi}",
                    metric_value=None,
                    threshold=((exp.rho_lo, exp.rho_hi) if exp is not None else None),
                    passed=False,
                    severity="info",
                    diagnostic=entry["skip_reason"],
                    extras={"n": entry["n"]},
                )
                continue

            rho = entry["rho"]
            if exp is None:
                yield TestOutcome(
                    test_id=self.test_id,
                    subject_id=sub.subject_id,
                    prior_id=prior,
                    roi_id=roi,
                    metric_name=f"spearman_{prior}_{roi}",
                    metric_value=rho,
                    threshold=None,
                    passed=False,
                    severity="info",
                    diagnostic="no expected band in config (orphan prior×ROI)",
                    extras={"n": entry["n"]},
                )
                continue
            # CI excludes zero with the expected sign?
            sign_band = exp.sign
            band = (exp.rho_lo, exp.rho_hi)
            ci_lo, ci_hi = entry["rho_lo"], entry["rho_hi"]

            ci_excludes_zero_with_sign = False
            if sign_band == +1 and ci_lo > 0:
                ci_excludes_zero_with_sign = True
            elif sign_band == -1 and ci_hi < 0:
                ci_excludes_zero_with_sign = True
            elif sign_band == 0:
                # Expectation is "near zero" — pass if |ρ| < 0.1
                ci_excludes_zero_with_sign = abs(rho) < 0.1

            within_band = band[0] <= abs(rho) <= band[1] if sign_band != 0 else True
            passed = bool(ci_excludes_zero_with_sign and within_band)

            failure_mode = _classify_failure(rho, exp)
            sev = (
                "error"
                if failure_mode in ("wrong_sign",)
                else ("warning" if not passed else "info")
            )
            diagnostic = (
                f"ρ={rho:.3f} [95% CI {ci_lo:.3f}, {ci_hi:.3f}], n={entry['n']}, "
                f"BH-adj p={p_corrected:.3g}; expected sign={sign_band}, "
                f"band={band}; {failure_mode}{' (PASS)' if passed else ''}"
            )

            yield TestOutcome(
                test_id=self.test_id,
                subject_id=sub.subject_id,
                prior_id=prior,
                roi_id=roi,
                metric_name=f"spearman_{prior}_{roi}",
                metric_value=rho,
                threshold=band,
                passed=passed,
                severity=sev,
                diagnostic=diagnostic,
                extras={
                    "n": entry["n"],
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "p_raw": entry["p"],
                    "p_fdr": float(p_corrected),
                    "failure_mode": failure_mode,
                },
            )
