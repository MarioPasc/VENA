"""Cohort protocol + registry: registration, lookup, error paths.

Covers ``vena.data.cohort.{protocol,registry}``. Synthetic only — no real
NIfTI files needed.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from vena.data.cohort import (
    CohortProtocol,
    CohortRegistry,
    register_cohort,
)
from vena.data.cohort.registry import CohortRegistryError

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Synthetic cohort fixture
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakePatient:
    patient_id: str
    root: Path
    metadata: dict[str, Any] = field(default_factory=dict)


class _FakeCohortDataset:
    """Minimal CohortProtocol implementation backed by an in-memory list."""

    def __init__(self, source_root: Path, patients: list[str]) -> None:
        self.source_root = Path(source_root)
        self._patients = [_FakePatient(pid, self.source_root / pid) for pid in patients]
        self._by_id = {p.patient_id: i for i, p in enumerate(self._patients)}

    def __len__(self) -> int:
        return len(self._patients)

    def __iter__(self) -> Iterator[_FakePatient]:
        return iter(self._patients)

    def __getitem__(self, key: int | str) -> _FakePatient:
        if isinstance(key, int):
            return self._patients[key]
        return self._patients[self._by_id[key]]

    def ids(self) -> list[str]:
        return [p.patient_id for p in self._patients]


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_patient_protocol_is_satisfied_by_dataclass(tmp_path: Path) -> None:
    """A frozen dataclass with the three required attrs satisfies CohortPatient."""
    p = _FakePatient(patient_id="P0001", root=tmp_path / "P0001")
    # Structural attribute check.
    assert p.patient_id == "P0001"
    assert p.root == tmp_path / "P0001"
    assert p.metadata == {}
    # CohortPatient is a Protocol without runtime_checkable; verify attrs only.
    assert hasattr(p, "patient_id")
    assert hasattr(p, "root")
    assert hasattr(p, "metadata")


def test_dataset_satisfies_runtime_checkable_protocol(tmp_path: Path) -> None:
    """CohortProtocol is @runtime_checkable; verify via isinstance."""
    ds = _FakeCohortDataset(tmp_path, ["P0001", "P0002"])
    assert isinstance(ds, CohortProtocol)
    assert len(ds) == 2
    assert ds.ids() == ["P0001", "P0002"]
    assert ds[0].patient_id == "P0001"
    assert ds["P0002"].patient_id == "P0002"
    assert iter(ds) is not None


# ---------------------------------------------------------------------------
# Registry behaviour
# ---------------------------------------------------------------------------


def test_register_and_build(tmp_path: Path) -> None:
    reg = CohortRegistry()
    reg.register("fake", _FakeCohortDataset, pathology="other")
    assert "fake" in reg
    assert reg.names() == ["fake"]
    assert reg.pathology_of("fake") == "other"
    built = reg.build("fake", source_root=tmp_path, patients=["P0", "P1"])
    assert isinstance(built, _FakeCohortDataset)
    assert built.ids() == ["P0", "P1"]


def test_register_duplicate_raises(tmp_path: Path) -> None:
    reg = CohortRegistry()
    reg.register("fake", _FakeCohortDataset, pathology="other")
    with pytest.raises(CohortRegistryError, match="already registered"):
        reg.register("fake", _FakeCohortDataset, pathology="other")


def test_register_duplicate_overwrite_ok(tmp_path: Path) -> None:
    reg = CohortRegistry()
    reg.register("fake", _FakeCohortDataset, pathology="other")
    # overwrite=True swaps the factory without raising.
    reg.register(
        "fake",
        _FakeCohortDataset,
        pathology="meningioma",
        overwrite=True,
    )
    assert reg.pathology_of("fake") == "meningioma"


def test_build_unknown_raises() -> None:
    reg = CohortRegistry()
    with pytest.raises(CohortRegistryError, match="Unknown cohort"):
        reg.build("nope")


def test_pathology_of_unknown_raises() -> None:
    reg = CohortRegistry()
    with pytest.raises(CohortRegistryError, match="Unknown cohort"):
        reg.pathology_of("missing")


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def test_register_cohort_decorator(tmp_path: Path) -> None:
    """The decorator form must register and return the class unmodified."""
    # Note: this uses the global registry; pick a unique name so we do not
    # collide with the real ucsf_pdgm / brats_gli registrations.
    name = "_unittest_decorated_cohort_xyz"

    @register_cohort(name, pathology="metastasis", overwrite=True)
    class _Decorated:
        def __init__(self, source_root: Path) -> None:
            self.source_root = source_root

        def __len__(self) -> int:
            return 0

        def __iter__(self):
            return iter(())

        def __getitem__(self, key):
            raise IndexError(key)

        def ids(self) -> list[str]:
            return []

    from vena.data.cohort import get_cohort_registry

    reg = get_cohort_registry()
    assert name in reg
    assert reg.pathology_of(name) == "metastasis"
    built = reg.build(name, source_root=tmp_path)
    assert isinstance(built, _Decorated)


def test_existing_cohorts_register_on_import() -> None:
    """Importing the cohort modules must register them in the global registry."""
    import vena.data.niigz.brats_gli
    import vena.data.niigz.ucsf_pdgm  # noqa: F401
    from vena.data.cohort import get_cohort_registry

    reg = get_cohort_registry()
    assert "ucsf_pdgm" in reg
    assert "brats_gli" in reg
    assert reg.pathology_of("ucsf_pdgm") == "glioma"
    assert reg.pathology_of("brats_gli") == "glioma"
