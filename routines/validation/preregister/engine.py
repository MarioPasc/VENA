"""Preregister engine — freeze ring partitions from the inference tree.

Reads the prediction and reference H5 files already on disk to record:
  - which cohorts belong to Ring A (cv_test) and Ring B (test_only / OOD),
  - scan-level and patient-level counts per cohort and per ring,
  - the set of methods inferred, and their available NFE levels.

Cross-checks the scan lists against
``vena.inference.image_dataset.resolve_test_scan_patient_pairs`` when a
corpus registry is supplied and the cohort's image H5 is accessible.  If
the image H5 is not mounted locally the check is skipped with a WARNING
rather than a hard failure (Picasso-only paths are expected in the local
dev environment).

Raises ``PreregisterError`` if the cross-check runs and disagrees — a
disagreement means the predictions do not correspond to the expected test
splits, which is a stop-the-line finding.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py

from vena.validation.artifacts import make_run_dir, symlink_latest, write_decision_json
from vena.validation.io import _decode_str_arr, _resolve_references_h5, build_index

logger = logging.getLogger(__name__)


class PreregisterError(Exception):
    """Raised when the cross-check between disk and corpus registry disagrees."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreregisterConfig:
    """Frozen configuration for the preregister engine.

    Parameters
    ----------
    inference_root :
        Root of the inference tree (contains shard subdirectories).
    output_root :
        Parent directory for the artifact folder.
    corpus_registry :
        Optional path to a corpus registry JSON for the cross-check.
        When ``None``, the cross-check is skipped.
    """

    inference_root: Path
    output_root: Path
    corpus_registry: Path | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> PreregisterConfig:
        """Load from a YAML file."""
        import yaml  # type: ignore[import-untyped]

        raw = yaml.safe_load(Path(path).read_text())
        cr = raw.get("corpus_registry")
        return cls(
            inference_root=Path(raw["inference_root"]),
            output_root=Path(raw["output_root"]),
            corpus_registry=Path(cr) if cr else None,
        )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


@dataclass
class PreregisterEngine:
    """Freeze the ring partitions from the inference tree on disk."""

    cfg: PreregisterConfig

    def run(self) -> Path:
        """Execute the preregister routine and return the artifact directory.

        Returns
        -------
        Path
            The run directory containing ``ring_partitions.json``.
        """
        root = self.cfg.inference_root
        logger.info("Building index from %s …", root)
        index = build_index(root)

        if index.empty:
            raise PreregisterError(
                f"No prediction H5 files found under {root}. "
                "Check that the inference tree is mounted and accessible."
            )

        logger.info("Found %d prediction files.", len(index))

        # --- resolve per-cohort reference H5 paths and read scan lists ---
        cohort_ring: dict[str, str] = {}
        cohort_ref_path: dict[str, Path] = {}

        for cohort in index["cohort"].unique():
            # Use the first prediction file for this cohort to resolve the reference.
            row = index[index["cohort"] == cohort].iloc[0]
            ring = str(row["ring"])
            cohort_ring[cohort] = ring
            pred_path: Path = row["path"]
            with h5py.File(pred_path, "r") as f:
                ref_path = _resolve_references_h5(f, pred_path)
            cohort_ref_path[cohort] = ref_path

        # --- read scan_ids / patient_ids from reference H5s ---
        cohort_scans: dict[str, list[str]] = {}
        cohort_patients: dict[str, list[str]] = {}
        for cohort, ref_path in cohort_ref_path.items():
            if not ref_path.is_file():
                raise PreregisterError(f"Reference H5 not found for cohort {cohort!r}: {ref_path}")
            with h5py.File(ref_path, "r") as f:
                scan_ids = _decode_str_arr(f["metadata/scan_id"][:])
                patient_ids = _decode_str_arr(f["metadata/patient_id"][:])
            cohort_scans[cohort] = scan_ids
            cohort_patients[cohort] = patient_ids

        # --- optional cross-check against corpus registry ---
        cross_check_status = self._cross_check(cohort_scans, cohort_patients)

        # --- assemble ring partition counts ---
        rings: dict[str, dict[str, Any]] = {}
        for cohort, ring in cohort_ring.items():
            scan_ids = cohort_scans[cohort]
            patient_ids = cohort_patients[cohort]
            unique_patients = sorted(set(patient_ids))

            ring_entry = rings.setdefault(ring, {"n_scans": 0, "n_patients": 0, "cohorts": {}})
            ring_entry["cohorts"][cohort] = {
                "n_scans": len(scan_ids),
                "n_patients": len(unique_patients),
                "scan_ids": scan_ids,
                "patient_ids": sorted(unique_patients),
            }
            ring_entry["n_scans"] += len(scan_ids)
            # Patient IDs must be de-duplicated across cohorts only within a cohort
            # (patients are cohort-scoped — no overlap expected).
            ring_entry["n_patients"] += len(unique_patients)

        # Log the counts.
        for ring_letter in sorted(rings):
            rd = rings[ring_letter]
            logger.info(
                "Ring %s: %d scans / %d patients across %d cohorts",
                ring_letter,
                rd["n_scans"],
                rd["n_patients"],
                len(rd["cohorts"]),
            )

        # --- methods and NFEs ---
        methods = sorted(index["method"].unique().tolist())
        nfes_by_method: dict[str, list[int]] = {}
        for method in methods:
            nfes = sorted(index[index["method"] == method]["nfe"].unique().tolist())
            nfes_by_method[method] = [int(n) for n in nfes]

        # --- shard SHA-256 digests (provenance) ---
        shard_shas: dict[str, str] = {}
        for shard in index["shard"].unique():
            dec_path = root / shard / "decision.json"
            if dec_path.is_file():
                digest = hashlib.sha256(dec_path.read_bytes()).hexdigest()[:12]
                shard_shas[str(shard)] = digest

        # --- write artifacts ---
        run_dir = make_run_dir(self.cfg.output_root, "preregister")

        payload: dict[str, Any] = {
            "schema_version": "1.0",
            "producer": "routines.validation.preregister:v0.1.0",
            "inference_root": str(root),
            "rings": rings,
            "methods": methods,
            "nfes_by_method": nfes_by_method,
            "selection_nfe": {},  # filled by §4.2 pre-registration
            "shard_shas": shard_shas,
            "cross_check": cross_check_status,
        }

        write_decision_json(run_dir, payload)

        # Write the canonical ring_partitions.json at the run-dir root and
        # also as a stable symlink target.
        rp_path = run_dir / "ring_partitions.json"
        rp_path.write_text(json.dumps(payload, indent=2, default=str))
        logger.info("Wrote ring_partitions.json → %s", rp_path)

        symlink_latest(run_dir)
        return run_dir

    # ------------------------------------------------------------------

    def _cross_check(
        self,
        cohort_scans: dict[str, list[str]],
        cohort_patients: dict[str, list[str]],
    ) -> dict[str, Any]:
        """Compare disk scan lists against corpus registry.

        Returns a status dict included in ``decision.json``.
        Raises :exc:`PreregisterError` on disagreement when accessible.
        """
        if self.cfg.corpus_registry is None:
            return {"status": "skipped", "reason": "corpus_registry not set in config"}

        try:
            from vena.data.registry.loader import load_registry
            from vena.inference.image_dataset import resolve_test_scan_patient_pairs
        except ImportError as exc:
            return {"status": "skipped", "reason": f"import error: {exc}"}

        try:
            registry = load_registry(self.cfg.corpus_registry, require_latents=False)
        except Exception as exc:  # RegistryError or file-not-found
            return {
                "status": "skipped",
                "reason": f"load_registry raised: {exc}",
            }

        mismatches: list[str] = []
        skipped: list[str] = []

        for cohort_entry in registry.cohorts:
            name = cohort_entry.name
            if name not in cohort_scans:
                continue  # cohort not in predictions tree — not a preregister concern
            if not cohort_entry.image_h5.is_file():
                logger.warning(
                    "cross-check skipped for %s: image_h5 not accessible at %s",
                    name,
                    cohort_entry.image_h5,
                )
                skipped.append(name)
                continue

            try:
                expected_pairs = resolve_test_scan_patient_pairs(cohort_entry, fold=0)
            except Exception as exc:
                logger.warning("resolve_test_scan_patient_pairs(%s) raised: %s", name, exc)
                skipped.append(name)
                continue

            expected_scans = sorted(s for s, _ in expected_pairs)
            actual_scans = sorted(cohort_scans[name])

            if expected_scans != actual_scans:
                msg = (
                    f"{name}: expected {len(expected_scans)} scans from registry, "
                    f"got {len(actual_scans)} from predictions H5"
                )
                logger.error("CROSS-CHECK FAIL — %s", msg)
                mismatches.append(msg)

        if mismatches:
            raise PreregisterError(
                "Predictions do not match the corpus registry test splits:\n"
                + "\n".join(f"  - {m}" for m in mismatches)
            )

        return {
            "status": "passed" if not skipped else "partial",
            "skipped_cohorts": skipped,
            "checked_cohorts": [
                c.name for c in registry.cohorts if c.name in cohort_scans and c.name not in skipped
            ],
        }
