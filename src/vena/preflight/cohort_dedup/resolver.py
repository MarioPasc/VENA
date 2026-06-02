"""Priority-based duplicate resolution across cohorts.

The resolver receives, per cv cohort, a mapping ``patient_id -> bridge_value``
where the bridge value is a global identity (currently a BraTS-2021 ID). A
cohort may instead declare *implicit* membership in the bridge namespace —
for example BraTS-GLI 2023/2025 contains every BraTS-2021 patient under
renumbered IDs and we have no per-patient bridge in its H5; we model this as
``implicit_brats21=True``.

For each global identity claimed by ≥2 cohorts the patient is kept in the
highest-priority cohort and dropped from the rest. Cohorts that lack a bridge
and are not in the implicit list pass through unchanged; any potential
overlap with an implicit-umbrella cohort is logged as ``unresolvable``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

from vena.preflight.cohort_dedup.xlsx import Brats2021Mapping

logger = logging.getLogger(__name__)


class CohortDedupResolverError(Exception):
    """Raised on contradictory dedup configuration."""


OnUnresolvable = Literal["warn", "error"]


@dataclass(frozen=True)
class CohortClaim:
    """One cohort's claim on the bridge namespace.

    Parameters
    ----------
    name
        Cohort name as it appears in the corpus registry.
    pid_to_bridge
        Map ``patient_id -> bridge_value`` for patients carrying an explicit
        bridge. Empty when the cohort has no bridge field.
    all_patient_ids
        Every patient_id the cohort declares (CSR patient keys). Drives the
        kept/rejected accounting.
    implicit_brats21
        When True, the cohort is treated as claiming every BraTS-2021 ID that
        appears in the xlsx (used for the BraTS-GLI umbrella case).
    """

    name: str
    pid_to_bridge: dict[str, str] = field(default_factory=dict)
    all_patient_ids: tuple[str, ...] = ()
    implicit_brats21: bool = False


@dataclass(frozen=True)
class ResolvedOverlap:
    """One bridge value with its concrete drop decision."""

    bridge: str
    tcia_source: str | None
    kept_cohort: str
    kept_pid: str | None  # None when the kept cohort is implicit (no bridge)
    dropped: tuple[tuple[str, str], ...]  # (cohort, pid) pairs


@dataclass(frozen=True)
class UnresolvableOverlap:
    """A potential overlap that the resolver could not concretely address."""

    cohort_a: str
    cohort_b: str
    reason: str
    n_candidate_groups: int


@dataclass(frozen=True)
class ResolverOutput:
    """Result of running the resolver.

    Attributes
    ----------
    kept
        Per-cohort allow-list (patient IDs).
    rejected
        Per-cohort reject list.
    resolved
        Concrete drop decisions, one per duplicate group.
    unresolvable
        Potential overlaps the resolver could not concretely address.
    """

    kept: dict[str, tuple[str, ...]]
    rejected: dict[str, tuple[str, ...]]
    resolved: tuple[ResolvedOverlap, ...]
    unresolvable: tuple[UnresolvableOverlap, ...]


def _slug(name: str) -> str:
    return name.replace("-", "").replace("_", "").lower()


def resolve(
    claims: Iterable[CohortClaim],
    *,
    mapping: Brats2021Mapping,
    priority: list[str],
    on_unresolvable: OnUnresolvable = "warn",
) -> ResolverOutput:
    """Run the priority-based duplicate resolver.

    Parameters
    ----------
    claims
        Per-cohort claims (one per cohort in the corpus registry).
    mapping
        Parsed BraTS-2021 ↔ TCIA mapping.
    priority
        Cohort names ordered highest → lowest. Cohorts not in this list inherit
        the lowest priority (logged as a configuration warning).
    on_unresolvable
        ``"warn"`` to log potential overlaps and continue; ``"error"`` to raise.
    """
    claims = list(claims)
    by_name: dict[str, CohortClaim] = {c.name: c for c in claims}
    missing_in_priority = [c.name for c in claims if c.name not in priority]
    if missing_in_priority:
        logger.warning(
            "cohorts %s are not listed in priority %s; treating them as "
            "lowest priority (insertion order)",
            missing_in_priority,
            priority,
        )
    rank: dict[str, int] = {name: i for i, name in enumerate(priority)}
    for offset, n in enumerate(missing_in_priority):
        rank[n] = len(priority) + offset

    # bridge -> {cohort_name -> patient_id (None for implicit umbrella)}
    bridge_to_members: dict[str, dict[str, str | None]] = {}

    # Explicit bridges.
    for c in claims:
        for pid, b in c.pid_to_bridge.items():
            if not b:
                continue
            bridge_to_members.setdefault(b, {})[c.name] = pid

    # Implicit umbrella claims.
    for c in claims:
        if not c.implicit_brats21:
            continue
        for b in mapping.by_brats21_id:
            bridge_to_members.setdefault(b, {})
            bridge_to_members[b].setdefault(c.name, None)

    rejected: dict[str, set[str]] = {c.name: set() for c in claims}
    resolved: list[ResolvedOverlap] = []
    for bridge, members in bridge_to_members.items():
        if len(members) < 2:
            continue
        kept_cohort = min(members, key=lambda n: rank.get(n, 10**6))
        kept_pid = members[kept_cohort]
        dropped: list[tuple[str, str]] = []
        for name, pid in members.items():
            if name == kept_cohort:
                continue
            if pid is None:
                # Implicit umbrella ranked below a bridge-carrying cohort —
                # we have no specific patient_id to drop. This is a config
                # warning, not an error: the umbrella stays whole.
                logger.warning(
                    "bridge %s: %s is ranked below %s but has no bridge "
                    "field — cannot drop a specific patient; %s kept whole.",
                    bridge,
                    name,
                    kept_cohort,
                    name,
                )
                continue
            rejected[name].add(pid)
            dropped.append((name, pid))
        if not dropped:
            continue
        row = mapping.by_brats21_id.get(bridge)
        resolved.append(
            ResolvedOverlap(
                bridge=bridge,
                tcia_source=row.data_collection if row else None,
                kept_cohort=kept_cohort,
                kept_pid=kept_pid,
                dropped=tuple(sorted(dropped)),
            )
        )

    # Unresolvable: bridgeless cohorts whose name fuzzily matches an xlsx
    # data_collection — implies potential overlap with the implicit umbrella(s)
    # but no way to point at specific patient IDs without an external map.
    unresolvable: list[UnresolvableOverlap] = []
    implicit_cohorts = [c.name for c in claims if c.implicit_brats21]
    bridgeless_cv = [c.name for c in claims if not c.pid_to_bridge and not c.implicit_brats21]
    for cohort in bridgeless_cv:
        slug = _slug(cohort)
        candidates = [coll for coll in mapping.by_collection if slug in _slug(coll)]
        if not candidates:
            continue
        n_groups = sum(len(mapping.by_collection[c]) for c in candidates)
        for umbrella in implicit_cohorts:
            reason = (
                f"{cohort} has no bridge field; xlsx data_collection(s) "
                f"{candidates} report {n_groups} potential members of {cohort} "
                f"also in {umbrella}, but portal IDs cannot be matched to "
                f"{cohort} patient IDs without an external bridge file."
            )
            unresolvable.append(
                UnresolvableOverlap(
                    cohort_a=cohort,
                    cohort_b=umbrella,
                    reason=reason,
                    n_candidate_groups=n_groups,
                )
            )

    if unresolvable and on_unresolvable == "error":
        joined = "\n  - ".join(u.reason for u in unresolvable)
        raise CohortDedupResolverError(
            f"{len(unresolvable)} unresolvable overlap(s):\n  - {joined}"
        )

    kept: dict[str, tuple[str, ...]] = {}
    rejected_t: dict[str, tuple[str, ...]] = {}
    for c in claims:
        rej = rejected[c.name]
        kept[c.name] = tuple(sorted(pid for pid in c.all_patient_ids if pid not in rej))
        rejected_t[c.name] = tuple(sorted(rej))

    return ResolverOutput(
        kept=kept,
        rejected=rejected_t,
        resolved=tuple(resolved),
        unresolvable=tuple(unresolvable),
    )
