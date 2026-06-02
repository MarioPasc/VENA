"""Unit tests for the priority resolver."""

from __future__ import annotations

import pytest

from vena.preflight.cohort_dedup.resolver import (
    CohortClaim,
    CohortDedupResolverError,
    resolve,
)
from vena.preflight.cohort_dedup.xlsx import Brats2021Mapping, MappingRow

pytestmark = pytest.mark.unit


def _make_mapping(rows: list[tuple[str, str]]) -> Brats2021Mapping:
    """Build a minimal Brats2021Mapping from (brats21_id, data_collection) pairs."""
    mrows: list[MappingRow] = []
    by_b21: dict[str, MappingRow] = {}
    by_coll: dict[str, set[str]] = {}
    for b21, coll in rows:
        r = MappingRow(
            brats21_id=b21,
            data_collection=coll,
            site_id=None,
            portal_id=None,
            study_date=None,
            seg_cohort=None,
            mgmt_cohort=None,
            mgmt_value=None,
        )
        mrows.append(r)
        by_b21[b21] = r
        by_coll.setdefault(coll, set()).add(b21)
    return Brats2021Mapping(
        rows=tuple(mrows),
        by_brats21_id=by_b21,
        by_collection={k: frozenset(v) for k, v in by_coll.items()},
    )


def test_priority_drops_from_lower_cohort_when_umbrella_higher() -> None:
    mapping = _make_mapping(
        [
            ("BraTS2021_00000", "UCSF-PDGM"),
            ("BraTS2021_00001", "UCSF-PDGM"),
            ("BraTS2021_00002", "UCSF-PDGM"),
        ]
    )
    claims = [
        CohortClaim(
            name="BraTS-GLI",
            all_patient_ids=("BraTS-GLI-00000", "BraTS-GLI-00001"),
            implicit_brats21=True,
        ),
        CohortClaim(
            name="UCSF-PDGM",
            all_patient_ids=("UCSF-PDGM-0000", "UCSF-PDGM-0001", "UCSF-PDGM-0002"),
            pid_to_bridge={
                "UCSF-PDGM-0000": "BraTS2021_00000",
                "UCSF-PDGM-0001": "BraTS2021_00001",
                # UCSF-PDGM-0002 has no bridge -> kept regardless of priority.
            },
        ),
    ]
    out = resolve(
        claims,
        mapping=mapping,
        priority=["BraTS-GLI", "UCSF-PDGM"],
        on_unresolvable="warn",
    )
    assert set(out.rejected["UCSF-PDGM"]) == {"UCSF-PDGM-0000", "UCSF-PDGM-0001"}
    assert out.rejected["BraTS-GLI"] == ()
    assert set(out.kept["UCSF-PDGM"]) == {"UCSF-PDGM-0002"}
    assert set(out.kept["BraTS-GLI"]) == {"BraTS-GLI-00000", "BraTS-GLI-00001"}
    # Two resolved overlaps; both kept by BraTS-GLI.
    assert len(out.resolved) == 2
    for r in out.resolved:
        assert r.kept_cohort == "BraTS-GLI"
        assert r.kept_pid is None  # implicit umbrella


def test_priority_reversed_keeps_explicit_drops_implicit() -> None:
    mapping = _make_mapping([("BraTS2021_00000", "UCSF-PDGM"), ("BraTS2021_00001", "UCSF-PDGM")])
    claims = [
        CohortClaim(name="BraTS-GLI", all_patient_ids=("BraTS-GLI-00000",), implicit_brats21=True),
        CohortClaim(
            name="UCSF-PDGM",
            all_patient_ids=("UCSF-PDGM-0000", "UCSF-PDGM-0001"),
            pid_to_bridge={
                "UCSF-PDGM-0000": "BraTS2021_00000",
                "UCSF-PDGM-0001": "BraTS2021_00001",
            },
        ),
    ]
    # UCSF first -> we'd want to drop BraTS-GLI duplicates, but its claim has no
    # pid (implicit). Verify: BraTS-GLI is NOT dropped, UCSF stays whole, the
    # warning path is exercised.
    out = resolve(
        claims,
        mapping=mapping,
        priority=["UCSF-PDGM", "BraTS-GLI"],
        on_unresolvable="warn",
    )
    assert out.rejected["UCSF-PDGM"] == ()
    assert out.rejected["BraTS-GLI"] == ()
    # No concrete drop -> resolved list is empty.
    assert out.resolved == ()


def test_unresolvable_flagged_for_bridgeless_matching_collection() -> None:
    # Cohort named "IvyGAP" with NO bridge field; xlsx lists IvyGAP collection.
    mapping = _make_mapping([("BraTS2021_00100", "IvyGAP"), ("BraTS2021_00101", "IvyGAP")])
    claims = [
        CohortClaim(name="BraTS-GLI", all_patient_ids=("BraTS-GLI-00000",), implicit_brats21=True),
        CohortClaim(name="IvyGAP", all_patient_ids=("W1", "W2")),
    ]
    out = resolve(claims, mapping=mapping, priority=["BraTS-GLI", "IvyGAP"], on_unresolvable="warn")
    assert len(out.unresolvable) == 1
    u = out.unresolvable[0]
    assert u.cohort_a == "IvyGAP"
    assert u.cohort_b == "BraTS-GLI"
    assert u.n_candidate_groups == 2
    # Both IvyGAP patients stay because we cannot match them.
    assert set(out.kept["IvyGAP"]) == {"W1", "W2"}


def test_unresolvable_raises_when_on_error() -> None:
    mapping = _make_mapping([("BraTS2021_00100", "IvyGAP")])
    claims = [
        CohortClaim(name="BraTS-GLI", all_patient_ids=("BraTS-GLI-00000",), implicit_brats21=True),
        CohortClaim(name="IvyGAP", all_patient_ids=("W1",)),
    ]
    with pytest.raises(CohortDedupResolverError):
        resolve(
            claims,
            mapping=mapping,
            priority=["BraTS-GLI", "IvyGAP"],
            on_unresolvable="error",
        )


def test_cohort_outside_priority_is_lowest() -> None:
    mapping = _make_mapping([("BraTS2021_00000", "UCSF-PDGM")])
    claims = [
        CohortClaim(
            name="UCSF-PDGM",
            all_patient_ids=("UCSF-PDGM-0000",),
            pid_to_bridge={"UCSF-PDGM-0000": "BraTS2021_00000"},
        ),
        CohortClaim(name="BraTS-GLI", all_patient_ids=("BraTS-GLI-00000",), implicit_brats21=True),
    ]
    out = resolve(claims, mapping=mapping, priority=["UCSF-PDGM"], on_unresolvable="warn")
    # BraTS-GLI inherits lowest priority but has no pid to drop -> kept.
    assert out.rejected["BraTS-GLI"] == ()
    assert out.rejected["UCSF-PDGM"] == ()
