"""Orchestrator for the perfusion-prior routine.

Glues together: cohort indexing, model instantiation from a YAML config,
per-patient prediction, NIfTI persistence (one file per derived channel),
collage rendering, and a manifest that records the resolved configuration
(incl. git SHA and per-file SHA-256 checksums) under a UTC-timestamped run
directory.

Mirrors :mod:`vena.prior_maps.vessel_priors.engine` so a reader of one routine
finds the other immediately recognisable. Differences from vessel_priors:

* Source modality is ASL (already CBF-quantified upstream), not SWI.
* Two parenchyma- and tumour-derived masks are loaded per patient to build
  the NAWM proxy required by the model.
* Outputs are persisted as one NIfTI per channel name (``cbf_rel.nii.gz`` and
  ``cbf.nii.gz``); ``binary`` is ``None`` so no binary file is written.
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
from vena.prior_maps.perfusion_priors._collage import render_collage
from vena.prior_maps.perfusion_priors.abc_model import PerfusionInput
from vena.prior_maps.perfusion_priors.models import MODEL_REGISTRY

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlgorithmSpec:
    """One algorithm spec parsed from YAML."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    tag: str | None = None

    @property
    def output_subdir(self) -> str:
        return self.tag or self.name


@dataclass(frozen=True)
class PerfusionPriorsRoutineConfig:
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
    def from_yaml(cls, path: Path | str) -> PerfusionPriorsRoutineConfig:
        path = Path(path)
        with path.open("r") as f:
            raw = yaml.safe_load(f) or {}
        algos = tuple(
            AlgorithmSpec(
                name=a["name"],
                params=a.get("params", {}) or {},
                tag=a.get("tag"),
            )
            for a in raw.get("algorithms", [])
        )
        if not algos:
            raise ValueError(f"No algorithms declared in {path}")
        metadata_csv = raw.get("metadata_csv")
        return cls(
            dataset_root=Path(raw["dataset_root"]),
            metadata_csv=Path(metadata_csv) if metadata_csv else None,
            output_root=Path(raw["output_root"]),
            algorithms=algos,
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
    img = nib.load(str(saved_path))
    if not np.allclose(np.asarray(img.affine, dtype=np.float64), source_affine):
        raise RuntimeError(f"Affine mismatch after saving {saved_path}: physical space drifted.")


class PerfusionPriorsEngine:
    """Runs every ``(algorithm × patient)`` combination listed in the config."""

    SOURCE_MODALITY = "ASL"
    PRIMARY_CHANNEL = "cbf"
    SOURCE_LABEL = "ASL (CBF)"
    CHANNEL_LABEL = "cbf (tanh-squashed)"
    CHANNEL_VMIN = -1.0
    CHANNEL_VMAX = 1.0

    def __init__(self, cfg: PerfusionPriorsRoutineConfig) -> None:
        self.cfg = cfg

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
            "source_modality": self.SOURCE_MODALITY,
            "git_sha": _git_sha(Path(__file__).resolve().parents[4]),
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
                f"Unknown perfusion model {algo.name!r}; available: {sorted(MODEL_REGISTRY)}"
            )
        model = None if figures_only else MODEL_REGISTRY[algo.name](**algo.params)

        algo_record: dict[str, Any] = {
            "name": algo.name,
            "tag": algo.output_subdir,
            "params": dict(algo.params),
            "patients": [],
        }

        for p in patients:
            logger.info("Processing %s with %s", p.patient_id, algo.name)
            asl = dataset.load_modality(p, self.SOURCE_MODALITY)
            brain_vol = dataset.load_brain_mask(p)
            parenchyma_vol = dataset.load_brain_parenchyma_mask(p)
            tumour_vol = dataset.load_tumor_seg(p)
            brain_mask = (brain_vol.array > 0.5).astype(np.uint8)
            parenchyma_mask = (parenchyma_vol.array > 0.5).astype(np.uint8)
            tumour_mask = (tumour_vol.array > 0).astype(np.uint8)

            pat_dir = niigz_root / algo.output_subdir / p.patient_id
            channel_paths: dict[str, Path] = {
                "cbf_rel": pat_dir / "cbf_rel.nii.gz",
                "cbf": pat_dir / "cbf.nii.gz",
            }

            if figures_only:
                for name, path in channel_paths.items():
                    if not path.exists():
                        raise FileNotFoundError(
                            f"figures-only requires existing channel {name} at {path}"
                        )
                channels = {
                    name: load_nii(path).array.astype(np.float32, copy=False)
                    for name, path in channel_paths.items()
                }
                binary: np.ndarray | None = None
                model_params: dict[str, Any] = {"source": "loaded-from-disk"}
            else:
                out = model.predict(  # type: ignore[union-attr]
                    PerfusionInput(
                        asl=asl,
                        brain_mask=brain_mask,
                        parenchyma_mask=parenchyma_mask,
                        tumour_mask=tumour_mask,
                        patient_id=p.patient_id,
                    )
                )
                channels = out.channels
                binary = out.binary
                model_params = out.params
                for name, arr in channels.items():
                    save_nii(arr, asl.affine, asl.header, channel_paths[name])
                    _assert_affine_preserved(channel_paths[name], asl.affine)

            render_collage(
                source=asl.array,
                brain=brain_mask,
                channel=channels[self.PRIMARY_CHANNEL],
                binary=binary,
                out_path=fig_root / algo.output_subdir / f"{p.patient_id}.png",
                patient_id=p.patient_id,
                source_label=self.SOURCE_LABEL,
                channel_label=self.CHANNEL_LABEL,
                channel_vmin=self.CHANNEL_VMIN,
                channel_vmax=self.CHANNEL_VMAX,
                n_slices=n_slices,
            )

            algo_record["patients"].append(
                {
                    "patient_id": p.patient_id,
                    "asl_path": str(asl.path),
                    "asl_sha256": _sha256(asl.path),
                    "channel_paths": {n: str(pth) for n, pth in channel_paths.items()},
                    "channel_sha256": {n: _sha256(pth) for n, pth in channel_paths.items()},
                    "channel_stats": {
                        n: {
                            "min": float(arr.min()),
                            "max": float(arr.max()),
                            "mean": float(arr.mean()),
                        }
                        for n, arr in channels.items()
                    },
                    "model_params_resolved": model_params,
                }
            )

        return algo_record

    def _persist_provenance(
        self,
        run_dir: Path,
        manifest: dict[str, Any],
        cfg: PerfusionPriorsRoutineConfig,
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
                        }
                        for a in cfg.algorithms
                    ],
                },
                f,
                sort_keys=False,
            )
        if manifest["git_sha"] is not None:
            (run_dir / "git_sha.txt").write_text(manifest["git_sha"] + "\n")
