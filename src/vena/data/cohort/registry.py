"""Decorator-based registry mapping cohort name → reader factory.

The registry is a single process-wide singleton. Cohort modules register
themselves at import time via :func:`register_cohort`; downstream consumers
look up by canonical cohort name via :meth:`CohortRegistry.build`.

Example
-------
::

    @register_cohort("ucsf_pdgm", pathology="glioma")
    class UCSFPDGMDataset:
        def __init__(self, source_root: Path, **kwargs: Any) -> None: ...


    reader = get_cohort_registry().build("ucsf_pdgm", source_root=Path("/data/ucsf_pdgm"))
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from .protocol import CohortProtocol, Pathology

logger = logging.getLogger(__name__)


class CohortRegistryError(Exception):
    """Raised on duplicate registration or unknown cohort lookup."""


T = TypeVar("T", bound=CohortProtocol[Any])


@dataclass(frozen=True)
class _CohortEntry:
    name: str
    pathology: Pathology
    factory: Callable[..., CohortProtocol[Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


class CohortRegistry:
    """Maps canonical cohort name → factory callable.

    Use :func:`get_cohort_registry` to obtain the process-wide singleton;
    construct directly only inside tests.
    """

    def __init__(self) -> None:
        self._entries: dict[str, _CohortEntry] = {}

    def register(
        self,
        name: str,
        factory: Callable[..., CohortProtocol[Any]],
        *,
        pathology: Pathology,
        metadata: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> None:
        """Register a cohort factory under ``name``.

        Parameters
        ----------
        name
            Canonical cohort key (lowercase, snake_case, e.g. ``"ucsf_pdgm"``).
        factory
            Callable that takes ``source_root`` (and arbitrary kwargs) and
            returns an object satisfying :class:`CohortProtocol`.
        pathology
            Pathology label for downstream filtering / cohort-mix analytics.
        metadata
            Optional free-form dict (release version, BIDS layout flag, etc.).
        overwrite
            If False (default), re-registering an existing name raises. Useful
            for tests that want to redefine a fixture.
        """
        if name in self._entries and not overwrite:
            raise CohortRegistryError(
                f"Cohort '{name}' is already registered. Pass overwrite=True to replace."
            )
        self._entries[name] = _CohortEntry(
            name=name,
            pathology=pathology,
            factory=factory,
            metadata=dict(metadata or {}),
        )
        logger.debug("Registered cohort '%s' (pathology=%s)", name, pathology)

    def build(self, name: str, **kwargs: Any) -> CohortProtocol[Any]:
        """Instantiate a registered cohort by name.

        Raises
        ------
        CohortRegistryError
            If ``name`` is not registered.
        """
        if name not in self._entries:
            known = sorted(self._entries)
            raise CohortRegistryError(f"Unknown cohort '{name}'. Registered: {known}")
        return self._entries[name].factory(**kwargs)

    def names(self) -> list[str]:
        """Return all registered cohort names (sorted)."""
        return sorted(self._entries)

    def pathology_of(self, name: str) -> Pathology:
        """Look up the pathology label for a registered cohort."""
        if name not in self._entries:
            raise CohortRegistryError(f"Unknown cohort '{name}'")
        return self._entries[name].pathology

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._entries

    def __len__(self) -> int:
        return len(self._entries)


_GLOBAL: CohortRegistry | None = None


def get_cohort_registry() -> CohortRegistry:
    """Return the process-wide :class:`CohortRegistry` singleton."""
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = CohortRegistry()
    return _GLOBAL


def register_cohort(
    name: str,
    *,
    pathology: Pathology,
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> Callable[[type[T]], type[T]]:
    """Class-decorator form of :meth:`CohortRegistry.register`.

    Wrap the cohort reader class to register it on import::

        @register_cohort("ucsf_pdgm", pathology="glioma")
        class UCSFPDGMDataset:
            def __init__(self, source_root: Path, **kwargs: Any) -> None: ...
    """

    def _wrap(cls: type[T]) -> type[T]:
        get_cohort_registry().register(
            name,
            cls,
            pathology=pathology,
            metadata=metadata,
            overwrite=overwrite,
        )
        return cls

    return _wrap


__all__ = [
    "CohortRegistry",
    "CohortRegistryError",
    "get_cohort_registry",
    "register_cohort",
]
