"""Typed protocol for VENA cohort readers (NIfTI-source layer).

A cohort is a collection of patients whose imaging follows a single naming /
directory convention. Every cohort reader (UCSF-PDGM, BraTS-GLI, BraTS-MEN,
Málaga, ...) shares the same essential operations: enumerate patients,
look up by ID, expose a lightweight handle that knows where its NIfTI files
live. This module pins that contract.

This protocol describes the *image-domain* (NIfTI) reader, not the latent H5
loader. The latter is already cohort-agnostic (see
:class:`vena.model.fm.lightning.data.LatentH5Dataset`) and is shared across
cohorts via :class:`vena.data.registry.CorpusRegistry`.

Adding a new cohort
-------------------
See ``src/vena/data/cohort/HOWTO.md`` for the 4-step recipe.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable

Pathology = Literal["glioma", "meningioma", "metastasis", "healthy", "other"]
"""Allowed pathology labels.

Extend by adding a new literal here; downstream consumers will be type-checked
against the updated set."""


class CohortPatient(Protocol):
    """Minimum attributes every cohort-patient handle must expose.

    Implementations may add cohort-specific fields (e.g.
    :class:`~vena.data.niigz.brats_gli.BraTSGLISession` adds ``session_id``);
    those extras are visible to cohort-specific code but invisible to
    generic consumers, which is exactly the desired layering.

    Attributes
    ----------
    patient_id : str
        Stable, cohort-unique identifier (e.g. ``"UCSF-PDGM-0001"``,
        ``"BraTS-GLI-00001"``).
    root : pathlib.Path
        Absolute path to the per-patient directory.
    metadata : dict[str, Any]
        Optional metadata dict; empty dict is the convention when none is
        available.
    """

    patient_id: str
    root: Path
    metadata: dict[str, Any]


PatientT = TypeVar("PatientT", bound=CohortPatient)


@runtime_checkable
class CohortProtocol(Protocol[PatientT]):
    """Image-domain (NIfTI) reader contract.

    Implementations:

    * :class:`vena.data.niigz.ucsf_pdgm.UCSFPDGMDataset`
    * :class:`vena.data.niigz.brats_gli.BraTSGLIDataset`

    A new cohort is added by writing a class that satisfies this protocol
    structurally and registering it with
    :func:`vena.data.cohort.register_cohort`. No subclassing required.

    Container protocol
    ------------------
    Implementations behave like a length-indexable sequence: ``len(reader)``,
    ``for patient in reader``, ``reader[i]`` (int) or ``reader[pid]`` (str).
    """

    source_root: Path
    """Root directory containing per-patient subdirectories."""

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[PatientT]: ...

    def __getitem__(self, key: int | str) -> PatientT: ...

    def ids(self) -> list[str]:
        """Return all patient IDs in discovery order."""
        ...


__all__ = ["CohortPatient", "CohortProtocol", "Pathology", "PatientT"]
