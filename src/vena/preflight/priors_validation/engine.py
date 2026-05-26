"""Engine for the priors-validation preflight routine.

Reads a YAML config, builds :class:`SubjectInputs` for the requested
patient subset (uses the same UCSF-PDGM sampling rules as the prior_maps
routines), runs the five-test panel via :class:`TestRunner`, then writes
per-subject + cohort reports and the ``decision.json`` contract.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from vena.data.niigz import UCSFPDGMDataset

from .core.config import (
    BOOTSTRAP_RESAMPLES_DEFAULT,
    FDR_Q_DEFAULT,
    TRAINING_CLEARANCE_THRESHOLD_DEFAULT,
)
from .io import build_subject_inputs, load_metadata_csv
from .preprocessing import robust_zscore
from .preprocessing.atlas import RegistrationKind
from .reporting import write_cohort_outputs, write_per_subject_json, write_per_subject_pdf
from .runner import TestRunner, _run_tests_for_subject
from .tests import (
    T1RangeSanity,
    T2AtlasLocalisation,
    T3T1GdCoherence,
    T4CrossModal,
    T5Reproducibility,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PriorsValidationRoutineConfig:
    """Resolved YAML config for one routine execution."""

    dataset_root: Path
    metadata_csv: Path | None
    derived_priors_roots: dict[str, Path]
    atlases_root: Path
    cache_root: Path
    output_root: Path
    n_patients: int
    seed: int
    n_workers: int = 1
    log_level: str = "INFO"
    training_clearance_threshold: float = TRAINING_CLEARANCE_THRESHOLD_DEFAULT
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES_DEFAULT
    fdr_q: float = FDR_Q_DEFAULT
    venous_atlas_path: Path | None = None
    registration_kind: RegistrationKind = "affine"
    patient_ids: tuple[str, ...] = field(default_factory=tuple)
    write_per_subject_pdf: bool = True

    @classmethod
    def from_yaml(cls, path: Path | str) -> PriorsValidationRoutineConfig:
        path = Path(path)
        raw = yaml.safe_load(path.read_text()) or {}
        derived: dict[str, Path] = {
            k: Path(v) for k, v in (raw.get("derived_priors_roots") or {}).items()
        }
        venous_path = raw.get("venous_atlas_path")
        return cls(
            dataset_root=Path(raw["dataset_root"]),
            metadata_csv=Path(raw["metadata_csv"]) if raw.get("metadata_csv") else None,
            derived_priors_roots=derived,
            atlases_root=Path(raw["atlases_root"]),
            cache_root=Path(raw["cache_root"]),
            output_root=Path(raw["output_root"]),
            n_patients=int(raw["n_patients"]),
            seed=int(raw["seed"]),
            n_workers=int(raw.get("n_workers", 1)),
            log_level=str(raw.get("log_level", "INFO")).upper(),
            training_clearance_threshold=float(
                raw.get("training_clearance_threshold", TRAINING_CLEARANCE_THRESHOLD_DEFAULT)
            ),
            bootstrap_resamples=int(raw.get("bootstrap_resamples", BOOTSTRAP_RESAMPLES_DEFAULT)),
            fdr_q=float(raw.get("fdr_q", FDR_Q_DEFAULT)),
            venous_atlas_path=Path(venous_path) if venous_path else None,
            registration_kind=str(raw.get("registration_kind", "affine")),  # type: ignore[arg-type]
            patient_ids=tuple(raw.get("patient_ids") or ()),
            write_per_subject_pdf=bool(raw.get("write_per_subject_pdf", True)),
        )


def _git_sha(repo: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


class PriorsValidationEngine:
    """Concrete engine — exposes a single ``run() -> Path`` method."""

    def __init__(self, cfg: PriorsValidationRoutineConfig) -> None:
        self.cfg = cfg

    def run(self) -> Path:
        cfg = self.cfg
        timestamp = _dt.datetime.now(tz=_dt.UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
        run_dir = cfg.output_root / "runs" / timestamp
        per_subject_dir = run_dir / "reports"
        figures_dir = run_dir / "figures"
        per_subject_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)

        # ----- cohort selection -----
        ds = UCSFPDGMDataset(cfg.dataset_root, cfg.metadata_csv)
        if cfg.patient_ids:
            patients = [ds[pid] for pid in cfg.patient_ids]
        else:
            patients = ds.sample(cfg.n_patients, seed=cfg.seed)
        logger.info(
            "Sampled %d patients (seed=%d): %s",
            len(patients),
            cfg.seed,
            ", ".join(p.patient_id for p in patients),
        )

        metadata_rows = load_metadata_csv(cfg.metadata_csv)

        # ----- tests + runner -----
        tests = [
            T1RangeSanity(),
            T2AtlasLocalisation(),
            T3T1GdCoherence(n_boot=cfg.bootstrap_resamples, fdr_q=cfg.fdr_q, seed=cfg.seed),
            T4CrossModal(n_boot=cfg.bootstrap_resamples, seed=cfg.seed),
            T5Reproducibility(),
        ]
        runner = TestRunner(
            tests=tests,
            atlases_root=cfg.atlases_root,
            cache_root=cfg.cache_root,
            venous_inhouse_path=cfg.venous_atlas_path,
            registration_kind=cfg.registration_kind,
            n_workers=cfg.n_workers,
            training_clearance_threshold=cfg.training_clearance_threshold,
        )
        bundle = runner.bundle  # forces atlas provisioning before any subject

        # ----- per-subject lazy loop -----
        # Eagerly loading all 10 subjects costs ~5 GB on a 240×240×155 cohort
        # with ≥10 NIfTI volumes per subject and triggers the OOM killer on
        # the local 8 GB workstation. Lazy loading caps the residency at one
        # subject + atlas warps (~1.5 GB peak).
        import gc

        import numpy as np

        results = []
        for p in patients:
            try:
                subj = build_subject_inputs(ds, p, cfg.derived_priors_roots, metadata_rows)
            except FileNotFoundError as exc:
                logger.warning("[%s] skipped: %s", p.patient_id, exc)
                continue
            vr = _run_tests_for_subject(
                subj,
                tests,
                bundle,
                cfg.cache_root,
                registration_kind=cfg.registration_kind,
            )
            results.append(vr)
            write_per_subject_json(vr, per_subject_dir / f"validation_report_{vr.subject_id}.json")
            if cfg.write_per_subject_pdf:
                try:
                    delta_t1 = robust_zscore(
                        np.asarray(subj.t1gd.array),
                        np.asarray(subj.brain_mask.array),
                    ) - robust_zscore(
                        np.asarray(subj.t1pre.array),
                        np.asarray(subj.brain_mask.array),
                    )
                    write_per_subject_pdf(
                        subj,
                        vr,
                        delta_t1,
                        per_subject_dir / f"{vr.subject_id}.pdf",
                        figures_dir=figures_dir,
                    )
                except Exception as exc:
                    logger.exception("[%s] per-subject PDF failed", vr.subject_id)
                    (per_subject_dir / f"{vr.subject_id}.pdf.failed.txt").write_text(str(exc))
            del subj
            gc.collect()

        if not results:
            raise RuntimeError("No subjects successfully processed; aborting.")
        report = runner._aggregate(results, bundle)

        # ----- cohort artefacts -----
        artefacts = write_cohort_outputs(report, run_dir)

        # ----- provenance -----
        (run_dir / "config_resolved.yaml").write_text(
            yaml.safe_dump(
                {
                    "dataset_root": str(cfg.dataset_root),
                    "metadata_csv": str(cfg.metadata_csv) if cfg.metadata_csv else None,
                    "derived_priors_roots": {
                        k: str(v) for k, v in cfg.derived_priors_roots.items()
                    },
                    "atlases_root": str(cfg.atlases_root),
                    "cache_root": str(cfg.cache_root),
                    "output_root": str(cfg.output_root),
                    "n_patients": cfg.n_patients,
                    "seed": cfg.seed,
                    "n_workers": cfg.n_workers,
                    "log_level": cfg.log_level,
                    "training_clearance_threshold": cfg.training_clearance_threshold,
                    "bootstrap_resamples": cfg.bootstrap_resamples,
                    "fdr_q": cfg.fdr_q,
                    "venous_atlas_path": str(cfg.venous_atlas_path)
                    if cfg.venous_atlas_path
                    else None,
                    "registration_kind": cfg.registration_kind,
                    "patient_ids": list(cfg.patient_ids),
                    "write_per_subject_pdf": cfg.write_per_subject_pdf,
                },
                sort_keys=False,
            )
        )
        sha = _git_sha(Path(__file__).resolve().parents[4])
        if sha:
            (run_dir / "git_sha.txt").write_text(sha + "\n")

        # LATEST symlink under artifacts/ as well (preflight contract)
        artifacts_latest = Path("artifacts/preflights/priors_validation")
        try:
            artifacts_latest.mkdir(parents=True, exist_ok=True)
            decision_link = artifacts_latest / "LATEST"
            if decision_link.exists() or decision_link.is_symlink():
                decision_link.unlink()
            decision_link.symlink_to(run_dir.resolve(), target_is_directory=True)
        except Exception:
            logger.exception("Failed to update artifacts/preflights/priors_validation/LATEST")

        # Summary log line
        logger.info(
            "Routine complete. Decision: %s. PDF: %s",
            artefacts["decision_json"],
            artefacts["cohort_pdf"],
        )
        with (run_dir / "manifest.json").open("w") as f:
            json.dump(
                {
                    "timestamp_utc": timestamp,
                    "n_subjects": len(results),
                    "artefacts": {k: str(v) for k, v in artefacts.items()},
                    "git_sha": sha,
                },
                f,
                indent=2,
                sort_keys=True,
            )
        return run_dir


__all__ = ["PriorsValidationEngine", "PriorsValidationRoutineConfig"]
