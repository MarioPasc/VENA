"""Cohort deduplication preflight engine.

Builds per-cohort allow-lists from a corpus registry, the BraTS-2021 ↔ TCIA
mapping xlsx, and a priority list. Run emits a versioned ``decision.json``
under ``<output_root>/<UTC-timestamp>/`` together with a human-readable
``report.md`` and a per-cohort kept-vs-rejected bar chart.

The engine never modifies the underlying H5 files; it reads ``patients/keys``
and (when configured) one metadata bridge field per cohort.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Literal

import h5py
import yaml
from pydantic import BaseModel, ConfigDict, Field

from vena.data.h5.shared import now_iso_utc, sha256_file
from vena.data.registry import load_registry
from vena.preflight.cohort_dedup._report import write_report
from vena.preflight.cohort_dedup.decision import (
    DEDUP_DECISION_SCHEMA_VERSION,
    assert_dedup_decision_valid,
    write_decision,
)
from vena.preflight.cohort_dedup.resolver import (
    CohortClaim,
    ResolverOutput,
    resolve,
)
from vena.preflight.cohort_dedup.xlsx import parse_brats2021_mapping

logger = logging.getLogger(__name__)


class CohortClaimError(Exception):
    """Raised when a cohort's H5 cannot be queried for its declared bridge."""


class CohortDedupConfig(BaseModel):
    """Pydantic config (parsed from YAML).

    Attributes
    ----------
    output_root
        Where ``<UTC>/decision.json`` is written. ``LATEST`` symlink is updated
        here on success.
    corpus_registry
        Path to the corpus registry JSON (see
        :mod:`vena.data.registry`).
    mapping_xlsx
        Path to ``BraTS2021_MappingToTCIA.xlsx``.
    priority
        Cohort names ordered highest -> lowest. The user-resolved direction
        for VENA is ``["BraTS-GLI", "UCSF-PDGM", "IvyGAP", "LUMIERE"]``.
    on_unresolvable
        ``"warn"`` (default) to log potential overlaps and continue;
        ``"error"`` to raise.
    bridge_fields
        Map ``cohort_name -> bridge_dataset_path`` (e.g. ``"metadata/brats21_id"``).
        Only cohorts listed here are expected to carry an explicit bridge.
    implicit_brats21_cohorts
        Cohorts that implicitly contain every BraTS-2021 patient under
        renumbered IDs (BraTS-GLI umbrella case).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    output_root: Path
    corpus_registry: Path
    mapping_xlsx: Path
    priority: list[str]
    on_unresolvable: Literal["warn", "error"] = "warn"
    bridge_fields: dict[str, str] = Field(default_factory=dict)
    implicit_brats21_cohorts: list[str] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path | str) -> CohortDedupConfig:
        raw = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(raw)


class CohortDedupEngine:
    """Compute per-cohort allow-lists and emit a versioned ``decision.json``."""

    def __init__(
        self,
        cfg: CohortDedupConfig,
        config_yaml_path: Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.config_yaml_path = config_yaml_path

    def run(self) -> Path:
        cfg = self.cfg
        timestamp = now_iso_utc().replace(":", "-")
        out_root = Path(cfg.output_root)
        run_dir = out_root / timestamp
        (run_dir / "figures").mkdir(parents=True, exist_ok=True)
        (run_dir / "tables").mkdir(parents=True, exist_ok=True)
        logger.info("cohort_dedup: run_dir=%s", run_dir)

        if self.config_yaml_path is not None and self.config_yaml_path.exists():
            shutil.copy2(self.config_yaml_path, run_dir / "config.original.yaml")
        (run_dir / "config.resolved.yaml").write_text(
            yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False)
        )

        registry = load_registry(cfg.corpus_registry, require_latents=False)
        mapping = parse_brats2021_mapping(cfg.mapping_xlsx)
        logger.info(
            "loaded mapping: %d rows across %d data collections",
            len(mapping.rows),
            len(mapping.by_collection),
        )

        claims: list[CohortClaim] = []
        bridge_fields_used: dict[str, str | None] = {}
        for cohort in registry.cohorts:
            all_pids, pid_to_bridge = self._read_cohort(cohort, cfg.bridge_fields)
            bridge_fields_used[cohort.name] = cfg.bridge_fields.get(cohort.name)
            claims.append(
                CohortClaim(
                    name=cohort.name,
                    pid_to_bridge=pid_to_bridge,
                    all_patient_ids=tuple(all_pids),
                    implicit_brats21=cohort.name in cfg.implicit_brats21_cohorts,
                )
            )

        outcome = resolve(
            claims,
            mapping=mapping,
            priority=list(cfg.priority),
            on_unresolvable=cfg.on_unresolvable,
        )

        payload = self._assemble_payload(
            registry_path=cfg.corpus_registry,
            mapping_path=cfg.mapping_xlsx,
            priority=cfg.priority,
            claims=claims,
            outcome=outcome,
            bridge_fields=bridge_fields_used,
        )

        decision_path = run_dir / "decision.json"
        write_decision(decision_path, payload)
        assert_dedup_decision_valid(decision_path)
        write_report(run_dir, payload)

        latest = out_root / "LATEST"
        try:
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(run_dir.name)
        except OSError as exc:
            logger.warning("could not update LATEST symlink: %s", exc)

        logger.info(
            "cohort_dedup complete: %d patients rejected across %d cohorts; report: %s",
            payload["totals"]["n_patients_total_rejected"],
            payload["totals"]["n_cohorts"],
            run_dir / "report.md",
        )
        return run_dir

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_cohort(
        cohort,
        bridge_fields: dict[str, str],
    ) -> tuple[list[str], dict[str, str]]:
        """Return ``(all_patient_ids, pid_to_bridge)`` for one cohort.

        Opens the cohort's image H5 (``patients/keys`` + ``patients/offsets``
        for the canonical patient list; the bridge field is read at the same
        scan indices and collapsed to a per-patient value). The H5 is closed
        before returning.
        """
        bridge_key = bridge_fields.get(cohort.name)
        with h5py.File(cohort.image_h5, "r", swmr=True) as f:
            keys = [b.decode() if isinstance(b, bytes) else str(b) for b in f["patients/keys"][:]]
            if bridge_key is None:
                return keys, {}
            offsets = f["patients/offsets"][:]
            try:
                raw = f[bridge_key][:]
            except KeyError as exc:
                raise CohortClaimError(
                    f"cohort {cohort.name!r} declares bridge_field "
                    f"{bridge_key!r} but {cohort.image_h5} has no such dataset"
                ) from exc
        values_str = [b.decode() if isinstance(b, bytes) else str(b) for b in raw]
        pid_to_bridge: dict[str, str] = {}
        for i, pid in enumerate(keys):
            start, end = int(offsets[i]), int(offsets[i + 1])
            seen = {values_str[j].strip() for j in range(start, end)}
            seen.discard("")
            if not seen:
                continue
            if len(seen) > 1:
                logger.warning(
                    "%s: patient %s has inconsistent bridge values %s; using first",
                    cohort.name,
                    pid,
                    sorted(seen),
                )
            pid_to_bridge[pid] = next(iter(sorted(seen)))
        logger.info(
            "%s: %d patients, %d with non-empty bridge (%s)",
            cohort.name,
            len(keys),
            len(pid_to_bridge),
            bridge_key,
        )
        return keys, pid_to_bridge

    @staticmethod
    def _assemble_payload(
        *,
        registry_path: Path,
        mapping_path: Path,
        priority: list[str],
        claims: list[CohortClaim],
        outcome: ResolverOutput,
        bridge_fields: dict[str, str | None],
    ) -> dict[str, Any]:
        per_cohort: dict[str, dict[str, Any]] = {}
        for c in claims:
            kept = list(outcome.kept[c.name])
            rejected = list(outcome.rejected[c.name])
            per_cohort[c.name] = {
                "n_total": len(c.all_patient_ids),
                "n_kept": len(kept),
                "n_rejected": len(rejected),
                "bridge_field": bridge_fields.get(c.name),
                "kept_patient_ids": kept,
                "rejected_patient_ids": rejected,
                "implicit_brats21": c.implicit_brats21,
            }
        return {
            "schema_version": DEDUP_DECISION_SCHEMA_VERSION,
            "produced_at": now_iso_utc(),
            "producer": "routines.preflights.cohort_dedup:0.1.0",
            "corpus_registry_path": str(registry_path),
            "corpus_registry_sha256": sha256_file(registry_path),
            "mapping_xlsx_path": str(mapping_path),
            "mapping_xlsx_sha256": sha256_file(mapping_path),
            "priority": list(priority),
            "policy": "drop_lower_priority",
            "totals": {
                "n_cohorts": len(claims),
                "n_patients_total_in": sum(len(c.all_patient_ids) for c in claims),
                "n_patients_total_kept": sum(len(outcome.kept[c.name]) for c in claims),
                "n_patients_total_rejected": sum(len(outcome.rejected[c.name]) for c in claims),
            },
            "cohorts": per_cohort,
            "overlap_audit": [
                {
                    "bridge": r.bridge,
                    "tcia_source": r.tcia_source,
                    "kept_cohort": r.kept_cohort,
                    "kept_pid": r.kept_pid,
                    "dropped": [list(d) for d in r.dropped],
                }
                for r in outcome.resolved
            ],
            "unresolvable_overlaps": [
                {
                    "cohort_a": u.cohort_a,
                    "cohort_b": u.cohort_b,
                    "reason": u.reason,
                    "n_candidate_groups": u.n_candidate_groups,
                }
                for u in outcome.unresolvable
            ],
        }
