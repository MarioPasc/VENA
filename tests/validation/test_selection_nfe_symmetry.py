"""Both arms of every comparison must be taken at their pre-registered NFE.

The headline table once filtered the *competitor* to its selection NFE while
averaging *VENA* over every NFE it has (1/2/5/10/20), so each p-value compared
VENA-averaged-over-NFE against a competitor at a single NFE. Nothing crashed
and nothing looked odd; the only visible symptom was that `tables/c0_sanity.csv`
(which filtered correctly) and `tables/headline_table.csv` disagreed by 0.0005
about VENA's own mae_brain.

The asymmetry existed because the two arms were written as two code paths. These
tests pin the single path.
"""

from __future__ import annotations

import pandas as pd
import pytest
from routines.validation.paired_fidelity.engine import PairedFidelityEngine

pytestmark = pytest.mark.validation

_VENA = "VENA-S1-v3b-rw"  # selection_nfe = 5


def _multi_nfe_frame() -> pd.DataFrame:
    """VENA across all five NFEs; only nfe=5 carries the honest value.

    The other NFEs are given a deliberately better (lower) value, so averaging
    over them flatters VENA — the direction the real bug happened to take.
    """
    rows = []
    for pid in ("p1", "p2", "p3"):
        for nfe in (1, 2, 5, 10, 20):
            rows.append((_VENA, pid, nfe, 0.01 if nfe != 5 else 0.10))
        rows.append(("C2-ResViT", pid, 1, 0.08))  # selection_nfe = 1
    return pd.DataFrame(rows, columns=["method", "patient_id", "nfe", "mae_brain"])


def test_vena_is_taken_at_its_selection_nfe_not_averaged_over_all() -> None:
    """VENA must reduce to its nfe=5 value, not the mean over 1/2/5/10/20."""
    df = _multi_nfe_frame()
    s = PairedFidelityEngine._series_at_selection_nfe(df, _VENA, "mae_brain")

    assert s.mean() == pytest.approx(0.10), (
        "VENA was averaged over every NFE (would give ~0.028) instead of taken "
        "at its pre-registered nfe=5"
    )
    assert len(s) == 3, "one value per patient"


def test_competitor_is_taken_at_its_own_selection_nfe() -> None:
    """A single-NFE competitor is unaffected — the same helper still applies."""
    df = _multi_nfe_frame()
    s = PairedFidelityEngine._series_at_selection_nfe(df, "C2-ResViT", "mae_brain")
    assert s.mean() == pytest.approx(0.08)


def test_ranking_uses_selection_nfe_so_arms_are_comparable() -> None:
    """The report's ranking must not flatter VENA by averaging its NFEs.

    With the bug, VENA averages to ~0.028 and ranks ABOVE ResViT (0.08). Taken
    at nfe=5 it is 0.10 and ranks below.
    """
    md = PairedFidelityEngine._primary_ranking_md(_multi_nfe_frame())

    assert md.index("C2-ResViT") < md.index(_VENA), (
        "ranking put VENA first by averaging its NFEs; at its pre-registered "
        "nfe=5 ResViT is better and must rank first"
    )
    assert "0.1000" in md and "0.0800" in md


def test_missing_selection_nfe_falls_back_loudly(caplog: pytest.LogCaptureFixture) -> None:
    """A run lacking the pre-registered NFE must warn, not silently substitute."""
    df = pd.DataFrame(
        [(_VENA, "p1", 1, 0.05), (_VENA, "p2", 1, 0.05)],
        columns=["method", "patient_id", "nfe", "mae_brain"],
    )
    with caplog.at_level("WARNING"):
        s = PairedFidelityEngine._series_at_selection_nfe(df, _VENA, "mae_brain")

    assert s.mean() == pytest.approx(0.05)
    assert "NOT the pre-registered comparison" in caplog.text


def test_absent_method_returns_empty_not_nan() -> None:
    """An absent arm yields an empty series so the caller can skip it."""
    df = _multi_nfe_frame()
    s = PairedFidelityEngine._series_at_selection_nfe(df, "C9-Nonexistent", "mae_brain")
    assert s.empty
