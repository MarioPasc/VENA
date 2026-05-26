"""Orchestrator for the vessel-prior pre-flight routine.

Glues together: cohort indexing, model instantiation from a YAML config,
optional SWI preprocessing chains, per-patient prediction, NIfTI persistence,
collage rendering, and a manifest that records the resolved configuration
(incl. git SHA and per-file SHA-256 checksums) under a UTC-timestamped run
directory.

The engine has two modes:

* **Default**: runs each ``(algorithm × patient)`` combination end-to-end.
* **figures-only** (``run(figures_only=True)``): loads previously saved
  ``vessel_soft.nii.gz`` + ``vessel_mask.nii.gz`` for each patient and only
  re-renders the collage. Useful after changes to the collage layout.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import yaml

from vena.data.niigz import UCSFPDGMDataset, save_nii
from vena.data.niigz.shared.io import load_nii
from vena.prior_maps.vessel_priors._collage import render_collage
from vena.prior_maps.vessel_priors.abc_model import VesselInput
from vena.prior_maps.vessel_priors.models import MODEL_REGISTRY
from vena.prior_maps.vessel_priors.preprocessing import PREPROCESSOR_REGISTRY

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreprocessingStepSpec:
    """One preprocessing step + its parameter dict, as parsed from YAML."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AlgorithmSpec:
    """One algorithm + an optional preprocessing chain, as parsed from YAML.

    ``tag`` overrides the output sub-namespace under ``niigz/`` and ``figures/``.
    When unset it defaults to ``name`` — so distinct preprocessing chains for the
    same vesselness operator (e.g. plain ``frangi`` vs CLAHE+frangi) need a tag
    each to avoid overwriting each other on disk.
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    preprocessing: tuple[PreprocessingStepSpec, ...] = ()
    tag: str | None = None

    @property
    def output_subdir(self) -> str:
        return self.tag or self.name


@dataclass(frozen=True)
class VesselPriorsRoutineConfig:
    """Resolved configuration for one execution of the routine."""

    dataset_root: Path
    metadata_csv: Path | None
    output_root: Path
    algorithms: tuple[AlgorithmSpec, ...]
    n_patients: int
    seed: int
    log_level: str = "INFO"
    n_slices: int = 5

    @classmethod
    def from_yaml(cls, path: Path | str) -> VesselPriorsRoutineConfig:
        path = Path(path)
        with path.open("r") as f:
            raw = yaml.safe_load(f) or {}
        algos: list[AlgorithmSpec] = []
        for a in raw.get("algorithms", []):
            pre = tuple(
                PreprocessingStepSpec(name=step["name"], params=step.get("params", {}) or {})
                for step in (a.get("preprocessing") or [])
            )
            algos.append(
                AlgorithmSpec(
                    name=a["name"],
                    params=a.get("params", {}) or {},
                    preprocessing=pre,
                    tag=a.get("tag"),
                )
            )
        if not algos:
            raise ValueError(f"No algorithms declared in {path}")
        metadata_csv = raw.get("metadata_csv")
        return cls(
            dataset_root=Path(raw["dataset_root"]),
            metadata_csv=Path(metadata_csv) if metadata_csv else None,
            output_root=Path(raw["output_root"]),
            algorithms=tuple(algos),
            n_patients=int(raw["n_patients"]),
            seed=int(raw["seed"]),
            log_level=str(raw.get("log_level", "INFO")).upper(),
            n_slices=int(raw.get("n_slices", 5)),
        )


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _git_sha(repo: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _assert_affine_preserved(saved_path: Path, source_affine: np.ndarray) -> None:
    """Re-open ``saved_path`` and verify it carries the source affine."""
    img = nib.load(str(saved_path))
    if not np.allclose(np.asarray(img.affine, dtype=np.float64), source_affine):
        raise RuntimeError(f"Affine mismatch after saving {saved_path}: physical space drifted.")


class VesselPriorsEngine:
    """Runs every ``(algorithm × patient)`` combination listed in the config."""

    def __init__(self, cfg: VesselPriorsRoutineConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------ public

    def run(self, *, figures_only: bool = False) -> Path:
        cfg = self.cfg
        timestamp = _dt.datetime.now(tz=_dt.UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
        run_dir = cfg.output_root / "runs" / timestamp
        niigz_root = cfg.output_root / "niigz"
        fig_root = cfg.output_root / "figures"
        run_dir.mkdir(parents=True, exist_ok=True)
        niigz_root.mkdir(parents=True, exist_ok=True)
        fig_root.mkdir(parents=True, exist_ok=True)

        dataset = UCSFPDGMDataset(cfg.dataset_root, cfg.metadata_csv)
        patients = dataset.sample(cfg.n_patients, seed=cfg.seed)
        logger.info(
            "Sampled %d patients (seed=%d): %s",
            len(patients),
            cfg.seed,
            ", ".join(p.patient_id for p in patients),
        )

        mode = "figures-only" if figures_only else "predict+render"
        logger.info("Engine mode: %s", mode)

        manifest: dict[str, Any] = {
            "timestamp_utc": timestamp,
            "mode": mode,
            "seed": cfg.seed,
            "n_patients": cfg.n_patients,
            "dataset_root": str(cfg.dataset_root),
            "metadata_csv": str(cfg.metadata_csv) if cfg.metadata_csv else None,
            "output_root": str(cfg.output_root),
            "git_sha": _git_sha(Path(__file__).resolve().parents[3]),
            "algorithms": [],
        }

        for algo in cfg.algorithms:
            algo_record = self._run_one_algorithm(
                algo=algo,
                patients=patients,
                dataset=dataset,
                niigz_root=niigz_root,
                fig_root=fig_root,
                n_slices=cfg.n_slices,
                figures_only=figures_only,
            )
            manifest["algorithms"].append(algo_record)

        self._persist_provenance(run_dir, manifest, cfg)
        logger.info("Run artifact: %s", run_dir)
        return run_dir

    # ------------------------------------------------------------------ helpers

    def _run_one_algorithm(
        self,
        *,
        algo: AlgorithmSpec,
        patients: list,
        dataset: UCSFPDGMDataset,
        niigz_root: Path,
        fig_root: Path,
        n_slices: int,
        figures_only: bool,
    ) -> dict[str, Any]:
        if algo.name not in MODEL_REGISTRY:
            raise KeyError(
                f"Unknown vessel model {algo.name!r}; available: {sorted(MODEL_REGISTRY)}"
            )
        model = None if figures_only else MODEL_REGISTRY[algo.name](**algo.params)

        preprocessors = []
        for step in algo.preprocessing:
            if step.name not in PREPROCESSOR_REGISTRY:
                raise KeyError(
                    f"Unknown preprocessor {step.name!r}; "
                    f"available: {sorted(PREPROCESSOR_REGISTRY)}"
                )
            preprocessors.append(PREPROCESSOR_REGISTRY[step.name](**step.params))

        algo_record: dict[str, Any] = {
            "name": algo.name,
            "tag": algo.output_subdir,
            "params": dict(algo.params),
            "preprocessing": [pre.describe() for pre in preprocessors],
            "patients": [],
        }

        for p in patients:
            logger.info("Processing %s with %s", p.patient_id, algo.name)
            swi = dataset.load_modality(p, "SWI_bias")
            brain_vol = dataset.load_brain_mask(p)
            brain_mask = (brain_vol.array > 0.5).astype(np.uint8)
            pat_dir = niigz_root / algo.output_subdir / p.patient_id
            soft_path = pat_dir / "vessel_soft.nii.gz"
            mask_path = pat_dir / "vessel_mask.nii.gz"

            if figures_only:
                if not soft_path.exists() or not mask_path.exists():
                    raise FileNotFoundError(
                        "figures-only mode requires previous outputs; missing "
                        f"{soft_path} or {mask_path}"
                    )
                soft_vol = load_nii(soft_path)
                mask_vol = load_nii(mask_path)
                soft = soft_vol.array.astype(np.float32, copy=False)
                binary = mask_vol.array.astype(np.uint8, copy=False)
                model_params: dict[str, Any] = {"source": "loaded-from-disk"}
            else:
                # Apply the preprocessing chain in declaration order. Every step
                # must preserve the SWI affine / spacing / shape.
                preprocessed = swi
                for pre in preprocessors:
                    preprocessed = pre.apply(preprocessed, brain_mask)
                    if preprocessed.array.shape != swi.array.shape:
                        raise RuntimeError(
                            f"Preprocessor {pre.name!r} changed shape "
                            f"({swi.array.shape} -> {preprocessed.array.shape})"
                        )
                    if not np.allclose(preprocessed.affine, swi.affine):
                        raise RuntimeError(f"Preprocessor {pre.name!r} altered the affine")

                out = model.predict(  # type: ignore[union-attr]
                    VesselInput(
                        swi=preprocessed,
                        brain_mask=brain_mask,
                        patient_id=p.patient_id,
                    )
                )
                soft = out.soft
                binary = out.binary
                model_params = out.params

                # Persist with the SOURCE SWI header so the saved files inherit
                # all NIfTI metadata (units, qform/sform codes, etc.) and live
                # in the same physical space.
                save_nii(soft, swi.affine, swi.header, soft_path)
                save_nii(binary, swi.affine, swi.header, mask_path)
                _assert_affine_preserved(soft_path, swi.affine)
                _assert_affine_preserved(mask_path, swi.affine)

            collage_path = render_collage(
                swi=swi.array,
                brain=brain_mask,
                soft=soft,
                binary=binary,
                out_path=fig_root / algo.output_subdir / f"{p.patient_id}.png",
                patient_id=p.patient_id,
                n_slices=n_slices,
            )

            algo_record["patients"].append(
                {
                    "patient_id": p.patient_id,
                    "swi_path": str(swi.path),
                    "swi_sha256": _sha256(swi.path),
                    "soft_path": str(soft_path),
                    "soft_sha256": _sha256(soft_path),
                    "mask_path": str(mask_path),
                    "mask_sha256": _sha256(mask_path),
                    "collage_path": str(collage_path),
                    "model_params_resolved": model_params,
                    "binary_voxels": int(np.asarray(binary).sum()),
                    "soft_max": float(np.asarray(soft).max()),
                }
            )

        return algo_record

    def _persist_provenance(
        self,
        run_dir: Path,
        manifest: dict[str, Any],
        cfg: VesselPriorsRoutineConfig,
    ) -> None:
        with (run_dir / "manifest.json").open("w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        with (run_dir / "config_resolved.yaml").open("w") as f:
            yaml.safe_dump(
                {
                    "dataset_root": str(cfg.dataset_root),
                    "metadata_csv": str(cfg.metadata_csv) if cfg.metadata_csv else None,
                    "output_root": str(cfg.output_root),
                    "n_patients": cfg.n_patients,
                    "seed": cfg.seed,
                    "log_level": cfg.log_level,
                    "n_slices": cfg.n_slices,
                    "algorithms": [
                        {
                            "name": a.name,
                            "tag": a.tag,
                            "params": dict(a.params),
                            "preprocessing": [
                                {"name": s.name, "params": dict(s.params)} for s in a.preprocessing
                            ],
                        }
                        for a in cfg.algorithms
                    ],
                },
                f,
                sort_keys=False,
            )
        if manifest["git_sha"] is not None:
            (run_dir / "git_sha.txt").write_text(manifest["git_sha"] + "\n")
