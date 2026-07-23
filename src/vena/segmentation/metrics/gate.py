"""G-SEG gate and dual DSC+Brier model-selection for the tumour segmenter.

Public API:

- :class:`GSegResult` — frozen result of a G-SEG gate check.
- :class:`ModelScore` — (name, dsc, brier) triple for ensemble selection.
- :func:`check_gseg` — per-cohort gate; TC Dice ≥ threshold AND NETC Dice ≥
  threshold for every cohort including Ring-B OOD; healthy controls use a
  TC-volume check instead of Dice.
- :func:`select_ensemble` — dual DSC+Brier model selection (or dice-only /
  brier-only via ``mode``).

Design authority: B.f-§2 (gate), B.f-§7 (dual selection), iter-9 §a.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vena.segmentation.exceptions import SegMetricError

if TYPE_CHECKING:
    from vena.segmentation.config import MetricsConfig


@dataclass(frozen=True)
class GSegResult:
    """Result of a G-SEG gate evaluation.

    Attributes
    ----------
    passed : bool
        ``True`` iff all per-cohort checks passed.
    per_cohort : dict[str, dict[str, float]]
        The raw metric values used for each cohort.  Keys are cohort names;
        values are the metric dict passed in by the caller.
    failures : list[tuple[str, str, float]]
        List of ``(cohort, metric_name, value)`` tuples that did not meet the
        threshold.  Empty when ``passed=True``.
    """

    passed: bool
    per_cohort: dict[str, dict[str, float]] = field(default_factory=dict)
    failures: list[tuple[str, str, float]] = field(default_factory=list)


@dataclass(frozen=True)
class ModelScore:
    """Score triple for ensemble model selection.

    Attributes
    ----------
    name : str
        Unique identifier for this model checkpoint / fold / run.
    dsc : float
        Mean Dice Similarity Coefficient (higher = better).
    brier : float
        Mean Brier score (lower = better; measures calibration quality).
    """

    name: str
    dsc: float
    brier: float


# ---------------------------------------------------------------------------
# Gate logic
# ---------------------------------------------------------------------------

_HEALTHY_KEY = "tc_volume"
_TC_KEY = "tc"
_NETC_KEY = "netc"

# Maximum fraction of voxels that may be predicted as TC on a healthy control.
# Not a config field (MetricsConfig is frozen with extra="forbid" and only carries
# the three gate/selection fields); kept as a module constant because it is an
# operational guard, not a tunable research hyperparameter.
_HEALTHY_TC_VOLUME_THRESHOLD: float = 0.01


def _is_healthy_cohort(metrics: Mapping[str, float]) -> bool:
    """Return ``True`` if the cohort metric dict uses the TC-volume convention."""
    return _HEALTHY_KEY in metrics


def check_gseg(
    dice_by_cohort: Mapping[str, Mapping[str, float]],
    cfg: MetricsConfig,
) -> GSegResult:
    """Evaluate the G-SEG gate across all cohorts.

    For each cohort, two paths exist:

    **Tumour cohort** (keys ``"tc"`` and ``"netc"`` present):
      - TC Dice ≥ ``cfg.gseg_tc_dice``
      - NETC Dice ≥ ``cfg.gseg_netc_dice``

    **Healthy-control cohort** (key ``"tc_volume"`` present):
      - TC predicted volume fraction ≤ ``_HEALTHY_TC_VOLUME_THRESHOLD`` (1 %)
        (the model must NOT hallucinate tumour on tumour-free brains).
        This replaces the Dice check — Dice is undefined or misleading when
        the ground-truth tumour region is empty.

    Parameters
    ----------
    dice_by_cohort : Mapping[str, Mapping[str, float]]
        Outer key = cohort name (e.g. ``"BraTS-GLI"``, ``"UCSF-PDGM"``,
        ``"healthy_controls"``).  Inner dict for tumour cohorts contains at
        minimum ``"tc"`` (TC Dice) and ``"netc"`` (NETC Dice).  For healthy
        controls it contains ``"tc_volume"`` (fraction of voxels predicted as
        TC, in [0, 1]).
    cfg : MetricsConfig
        Thresholds; see :class:`~vena.segmentation.config.MetricsConfig`.

    Returns
    -------
    GSegResult
        ``passed=True`` iff every cohort cleared its check.
        ``failures`` lists ``(cohort, metric_name, value)`` for every violation.
    """
    failures: list[tuple[str, str, float]] = []
    per_cohort: dict[str, dict[str, float]] = {}

    for cohort, metrics in dice_by_cohort.items():
        per_cohort[cohort] = dict(metrics)

        if _is_healthy_cohort(metrics):
            # Healthy-control path: check TC predicted volume is near zero.
            # Uses the module-level constant — not a MetricsConfig field
            # (MetricsConfig has extra="forbid" with only 3 tunable fields).
            vol = float(metrics[_HEALTHY_KEY])
            if vol > _HEALTHY_TC_VOLUME_THRESHOLD:
                failures.append((cohort, _HEALTHY_KEY, vol))
        else:
            # Tumour-cohort path: check TC and NETC Dice.
            if _TC_KEY not in metrics or _NETC_KEY not in metrics:
                raise SegMetricError(
                    f"Cohort '{cohort}' metrics must contain '{_TC_KEY}' and "
                    f"'{_NETC_KEY}' keys (or '{_HEALTHY_KEY}' for healthy controls). "
                    f"Got: {list(metrics.keys())}"
                )

            tc_dice = float(metrics[_TC_KEY])
            netc_dice = float(metrics[_NETC_KEY])

            if tc_dice < cfg.gseg_tc_dice:
                failures.append((cohort, _TC_KEY, tc_dice))
            if netc_dice < cfg.gseg_netc_dice:
                failures.append((cohort, _NETC_KEY, netc_dice))

    return GSegResult(
        passed=len(failures) == 0,
        per_cohort=per_cohort,
        failures=failures,
    )


# ---------------------------------------------------------------------------
# Ensemble selection
# ---------------------------------------------------------------------------

# Absolute DSC tolerance for the "dual" selection mode: if the DSC difference
# between the best model and a candidate is below this threshold, calibration
# (Brier score) is used as the tie-breaker.
_DUAL_DSC_TOLERANCE: float = 0.01


def select_ensemble(
    models: Sequence[ModelScore],
    mode: str = "dual",
) -> str:
    """Select the best model from a sequence of scored candidates.

    Three modes:

    ``"dual"``
        Prefer the lower-Brier model when the DSC gap to the best model is
        < 1 percentage point (``|DSC_best − DSC_candidate| < 0.01``).  When
        multiple models are within 1 % DSC, pick the one with the lowest
        Brier score.  When the DSC gap is ≥ 1 %, pick the highest-DSC model
        regardless of Brier.  Rationale (B.f-§7): the generator consumes soft
        probability maps, so calibration quality is load-bearing; sacrificing
        < 1 % DSC for better calibration is worthwhile.

    ``"dice"``
        Always pick the model with the highest DSC.

    ``"brier"``
        Always pick the model with the lowest Brier score.

    Parameters
    ----------
    models : Sequence[ModelScore]
        Candidate models to compare.  At least one must be provided.
    mode : str
        One of ``"dual"``, ``"dice"``, ``"brier"``.  Default ``"dual"``.

    Returns
    -------
    str
        ``name`` of the selected :class:`ModelScore`.

    Raises
    ------
    SegMetricError
        If *models* is empty or *mode* is not recognised.
    """
    if not models:
        raise SegMetricError("select_ensemble requires at least one ModelScore")

    if mode == "dice":
        return max(models, key=lambda m: m.dsc).name

    if mode == "brier":
        return min(models, key=lambda m: m.brier).name

    if mode == "dual":
        best_dsc = max(m.dsc for m in models)
        # Candidates within _DUAL_DSC_TOLERANCE of the best DSC.
        candidates = [m for m in models if (best_dsc - m.dsc) < _DUAL_DSC_TOLERANCE]
        # Among them, prefer the best-calibrated (lowest Brier).
        return min(candidates, key=lambda m: m.brier).name

    raise SegMetricError(
        f"Unknown selection mode {mode!r}; expected one of 'dual', 'dice', 'brier'"
    )
