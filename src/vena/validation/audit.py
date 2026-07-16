"""Phase-2 §4.1 harmonisation audit — Table S1.

Confirms that the §4.1 harmonisation contract holds for every prediction:
  - inside the brain mask: intensities are in [0, 1].
  - outside the brain mask: intensities are exactly 0.

Produces one row per scan for both the synthetic and real T1c volumes.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from vena.validation.io import ScanSample


def audit_harmonisation(samples: Iterable[ScanSample]) -> pd.DataFrame:
    """Per-scan harmonisation audit for both synthetic and real T1c.

    Parameters
    ----------
    samples :
        Any iterable of :class:`~vena.validation.io.ScanSample` objects,
        e.g. the output of :func:`~vena.validation.io.iter_scans`.

    Returns
    -------
    pd.DataFrame
        One row per scan with columns:

        ``scan_id, patient_id, cohort, method, nfe,
        pred_in_range, real_in_range,
        pred_max_exterior, real_max_exterior,
        pred_min_brain, pred_max_brain,
        real_min_brain, real_max_brain``

        ``pred_in_range`` / ``real_in_range`` are ``True`` when the volume
        is in ``[0, 1]`` inside the brain mask.
        ``pred_max_exterior`` / ``real_max_exterior`` are the maximum
        absolute intensity outside the brain mask (expected to be 0).
    """
    rows: list[dict] = []
    for s in samples:
        brain = s.brain
        exterior = ~brain

        # --- synthetic ---
        pred_brain = s.pred[brain]
        pred_ext = s.pred[exterior]
        pred_in_range = (
            bool(
                np.all(np.isfinite(pred_brain))
                and float(pred_brain.min()) >= 0.0
                and float(pred_brain.max()) <= 1.0
            )
            if pred_brain.size > 0
            else True
        )
        pred_max_ext = float(np.abs(pred_ext).max()) if pred_ext.size > 0 else 0.0
        pred_min_brain = float(pred_brain.min()) if pred_brain.size > 0 else float("nan")
        pred_max_brain = float(pred_brain.max()) if pred_brain.size > 0 else float("nan")

        # --- real ---
        real_brain = s.real[brain]
        real_ext = s.real[exterior]
        real_in_range = (
            bool(
                np.all(np.isfinite(real_brain))
                and float(real_brain.min()) >= 0.0
                and float(real_brain.max()) <= 1.0
            )
            if real_brain.size > 0
            else True
        )
        real_max_ext = float(np.abs(real_ext).max()) if real_ext.size > 0 else 0.0
        real_min_brain = float(real_brain.min()) if real_brain.size > 0 else float("nan")
        real_max_brain = float(real_brain.max()) if real_brain.size > 0 else float("nan")

        rows.append(
            {
                "scan_id": s.scan_id,
                "patient_id": s.patient_id,
                "cohort": s.cohort,
                "method": s.method,
                "nfe": s.nfe,
                "pred_in_range": pred_in_range,
                "real_in_range": real_in_range,
                "pred_max_exterior": pred_max_ext,
                "real_max_exterior": real_max_ext,
                "pred_min_brain": pred_min_brain,
                "pred_max_brain": pred_max_brain,
                "real_min_brain": real_min_brain,
                "real_max_brain": real_max_brain,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "scan_id",
                "patient_id",
                "cohort",
                "method",
                "nfe",
                "pred_in_range",
                "real_in_range",
                "pred_max_exterior",
                "real_max_exterior",
                "pred_min_brain",
                "pred_max_brain",
                "real_min_brain",
                "real_max_brain",
            ]
        )
    return pd.DataFrame(rows)
