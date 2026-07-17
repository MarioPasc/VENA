"""Guards for the paired_fidelity report's honesty sections.

The sweep artifact is what a reader quotes from. Two claims must survive every
refactor: that `VENA-S1-v3b-rw` is handed a ground-truth tumour mask no
competitor gets, and that the pre-registered primary endpoint ranks the methods
however it ranks them.

Both are derived from the run's own data rather than asserted, so these tests
check the derivation — a hard-coded finding that drifts out of agreement with
the tables beside it is worse than no finding.
"""

from __future__ import annotations

import pandas as pd
import pytest
from routines.validation.paired_fidelity.engine import PairedFidelityEngine

pytestmark = pytest.mark.validation

_VENA = "VENA-S1-v3b-rw"
_V3A = "VENA-S1-v3a"


#: Pre-registered selection NFE for the methods used here (SHARED_CONTRACTS §4).
#: Rows must carry it: every arm is reduced at its own selection NFE, so a
#: fixture with no `nfe` column is not a realistic stand-in for real rows.
_SEL = {_VENA: 5, _V3A: 5, "C2-ResViT": 1, "C0-Identity": 1}


def _pt(rows: list[tuple[str, str, float, float, float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["method", "patient_id", "mae_brain", "mae_wt", "ssim_wt"])
    df["nfe"] = df["method"].map(_SEL)
    return df


def test_primary_ranking_is_ordered_by_the_data_not_by_vena() -> None:
    """The ranking puts the best method first even when that is not VENA."""
    df = _pt(
        [
            (_VENA, "p1", 0.10, 0.09, 0.60),
            (_VENA, "p2", 0.10, 0.09, 0.60),
            ("C2-ResViT", "p1", 0.05, 0.20, 0.40),
            ("C2-ResViT", "p2", 0.05, 0.20, 0.40),
            ("C0-Identity", "p1", 0.30, 0.30, 0.20),
            ("C0-Identity", "p2", 0.30, 0.30, 0.20),
        ]
    )
    md = PairedFidelityEngine._primary_ranking_md(df)

    # ResViT is better on the primary endpoint here and must therefore rank 1.
    resvit_pos, vena_pos = md.index("C2-ResViT"), md.index(_VENA)
    assert resvit_pos < vena_pos, "ranking must follow the data, not favour VENA"
    assert "the pre-registered VENA arm" in md, "the VENA arm must be identifiable"
    assert "_null floor_" in md


def test_oracle_caveat_quantifies_the_gap_against_the_no_mask_arm() -> None:
    """The v3b-rw minus v3a gap is computed from the run, with its sign kept."""
    df = _pt(
        [
            (_VENA, "p1", 0.10, 0.08, 0.70),
            (_VENA, "p2", 0.10, 0.08, 0.70),
            (_V3A, "p1", 0.10, 0.13, 0.40),
            (_V3A, "p2", 0.10, 0.13, 0.40),
        ]
    )
    md = PairedFidelityEngine._oracle_caveat_md(df)

    assert _V3A in md
    # mae_wt: 0.08 - 0.13 = -0.05 (the oracle helps -> lower error)
    assert "-0.0500" in md
    # ssim_wt: 0.70 - 0.40 = +0.30
    assert "+0.3000" in md
    # mae_brain: identical -> the oracle buys nothing whole-brain
    assert "+0.0000" in md


def test_oracle_caveat_warns_loudly_when_the_no_mask_arm_is_absent() -> None:
    """A run without v3a must say so, not fall silent.

    Silence would read as "there is no caveat" — the failure mode the whole
    section exists to prevent.
    """
    df = _pt([(_VENA, "p1", 0.10, 0.08, 0.70)])
    md = PairedFidelityEngine._oracle_caveat_md(df)

    assert _V3A in md
    assert "MUST be reported" in md


def test_ranking_degrades_gracefully_on_empty_input() -> None:
    """No Ring-A rows must not raise — the report still has to be written."""
    md = PairedFidelityEngine._primary_ranking_md(pd.DataFrame())
    assert "unavailable" in md
