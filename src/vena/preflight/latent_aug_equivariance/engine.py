"""Equivariance preflight engine.

Workflow per patient × per transform:

1. Read the precomputed MAISI latent ``z`` from the cohort's latent H5.
2. Decode ``D(z)`` to a ``[0, 1]`` box volume.
3. Read the real T1c, crop to the same box, normalise exactly as the encoder.
4. Apply the transform to the decoded volume (image-space gold path).
5. Apply the same transform to the latent and decode (latent-space proposed).
6. Compare gold vs proposed via whole-volume PSNR/SSIM on ``[0, 1]``.

Results are aggregated to a per-transform median; ``passes`` is determined by
the configured ``psnr_db`` and ``ssim`` thresholds. Outputs follow the project
preflight conventions: ``report.md`` + ``figures/`` + ``tables/`` +
``decision.json`` under ``artifacts/latent_aug_equivariance/<UTC>/``.
"""

from __future__ import annotations

import csv
import json
import logging
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field

from vena.data.augment.base import LatentAugmentation
from vena.data.augment.transforms import REGISTRY
from vena.data.h5.shared import now_iso_utc
from vena.model.autoencoder.maisi.decode.engine import MaisiDecoder
from vena.model.autoencoder.maisi.loader import load_autoencoder
from vena.model.fm.eval import (
    full_volume_psnr_ssim,
    select_content_slices,
)
from vena.model.fm.eval.exhaustive import (
    build_crop_spec_from_h5,
    load_real_t1c_box,
)
from vena.model.fm.metrics import ImageMetrics
from vena.preflight.latent_aug_equivariance._visualize import (
    render_equivariance_panel,
    render_summary_boxplot,
)

logger = logging.getLogger(__name__)

# decision.json schema version — bump on any breaking change to keys consumed
# by downstream training routines.
DECISION_SCHEMA_VERSION: str = "1.0"


class LatentAugEquivarianceError(Exception):
    """Raised on malformed config or unrecoverable preflight failure."""


# ---------------------------------------------------------------------------
# YAML-side config (pydantic, frozen)
# ---------------------------------------------------------------------------


class _CohortCfg(BaseModel):
    """One cohort to include in the preflight."""

    model_config = ConfigDict(extra="forbid")
    name: str
    latent_h5: Path
    image_h5: Path


class _PassCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    psnr_db: float = 35.0
    ssim: float = 0.95


class _AugmentationCfg(BaseModel):
    """One transform to evaluate.

    ``name`` is looked up in :data:`vena.data.augment.transforms.REGISTRY`;
    any other key is forwarded to the operator constructor. ``param_grid``
    enumerates the explicit parameter draws to evaluate — when present each
    entry is used verbatim (no stochastic sampling) so the report can quote
    per-angle / per-shift numbers. ``param_grid`` is required for any
    transform whose effect depends on its sampled parameter (rotations,
    translations, gamma); a no-grid entry falls back to a single draw at
    the operator's default (used by :class:`FlipLR`, which is parameterless).
    """

    model_config = ConfigDict(extra="allow")
    name: str
    p: float = 1.0
    param_grid: list[dict[str, Any]] = Field(default_factory=list)


class LatentAugEquivarianceConfig(BaseModel):
    """Root config for the equivariance preflight."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    output_root: Path
    vae_checkpoint: Path
    cohorts: list[_CohortCfg]
    augmentations: list[_AugmentationCfg]
    n_patients_per_cohort: int = 10
    seed: int = 42
    device: str = "cuda:0"
    pass_threshold: _PassCfg = Field(default_factory=_PassCfg)
    n_slices_for_figure: int = 8
    figure_slice_offset: int = 10
    # Cap the number of figures we render per (cohort × transform) pair so a
    # 10-patient sweep does not produce hundreds of PNGs. The first
    # ``figures_per_pair`` patients in evaluation order get a figure.
    figures_per_pair: int = 1
    # Whole-volume metrics are computed on the entire ``[0, 1]`` box (matches
    # ``routines/fm/exhaustive_val`` semantics).
    log_every_n_patients: int = 1

    @classmethod
    def from_yaml(cls, path: Path | str) -> LatentAugEquivarianceConfig:
        path = Path(path)
        with path.open("r") as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw)


# ---------------------------------------------------------------------------
# Internal data classes
# ---------------------------------------------------------------------------


@dataclass
class _TransformInstance:
    """One concrete (operator, params, tag) triple to evaluate."""

    name: str
    operator: LatentAugmentation
    params: dict[str, Any]
    tag: str

    def label(self) -> str:
        return f"{self.name}[{self.tag}]"


@dataclass
class _Aggregate:
    """Aggregated metrics across all patients for one transform."""

    transform: str
    psnr_values: list[float] = field(default_factory=list)
    ssim_values: list[float] = field(default_factory=list)

    def add(self, psnr: float, ssim: float) -> None:
        if not np.isnan(psnr):
            self.psnr_values.append(float(psnr))
        if not np.isnan(ssim):
            self.ssim_values.append(float(ssim))

    def summary(self) -> dict[str, float | int]:
        if not self.psnr_values:
            return {
                "n": 0,
                "median_psnr_db": float("nan"),
                "median_ssim": float("nan"),
                "min_psnr_db": float("nan"),
                "min_ssim": float("nan"),
            }
        return {
            "n": len(self.psnr_values),
            "median_psnr_db": statistics.median(self.psnr_values),
            "median_ssim": statistics.median(self.ssim_values),
            "min_psnr_db": min(self.psnr_values),
            "min_ssim": min(self.ssim_values),
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class LatentAugEquivarianceEngine:
    """Run the preflight end-to-end."""

    def __init__(
        self,
        cfg: LatentAugEquivarianceConfig,
        config_yaml_path: Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.config_yaml_path = config_yaml_path
        self.device = self._resolve_device(cfg.device)

    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if not torch.cuda.is_available():
            logger.warning("CUDA unavailable; preflight falls back to CPU.")
            return torch.device("cpu")
        idx = int(device.split(":")[1]) if ":" in device else 0
        if idx >= torch.cuda.device_count():
            logger.warning(
                "requested %s but only %d GPU(s) visible; using cuda:0.",
                device,
                torch.cuda.device_count(),
            )
            idx = 0
        torch.cuda.set_device(idx)
        return torch.device(f"cuda:{idx}")

    # ------------------------------------------------------------------

    def _build_transform_instances(self) -> list[_TransformInstance]:
        """Materialise the (operator, params, tag) triples to evaluate."""
        instances: list[_TransformInstance] = []
        for aug_cfg in self.cfg.augmentations:
            if aug_cfg.name not in REGISTRY:
                raise LatentAugEquivarianceError(
                    f"unknown augmentation {aug_cfg.name!r}; available: {sorted(REGISTRY)}"
                )
            ctor_kwargs = aug_cfg.model_dump()
            ctor_kwargs.pop("name", None)
            ctor_kwargs.pop("param_grid", None)
            operator = REGISTRY[aug_cfg.name](**ctor_kwargs)
            if aug_cfg.param_grid:
                for params in aug_cfg.param_grid:
                    tag = operator.param_tag(params)
                    instances.append(
                        _TransformInstance(
                            name=aug_cfg.name,
                            operator=operator,
                            params=dict(params),
                            tag=tag,
                        )
                    )
            else:
                # Parameterless or single-default operator (FlipLR).
                default_params = operator.sample_params(random.Random(self.cfg.seed))
                instances.append(
                    _TransformInstance(
                        name=aug_cfg.name,
                        operator=operator,
                        params=default_params,
                        tag=operator.param_tag(default_params),
                    )
                )
        return instances

    # ------------------------------------------------------------------

    def _sample_patient_ids(self, latent_h5: Path, n: int, seed: int) -> list[str]:
        """Pick ``n`` patient IDs from the cohort's validation split.

        Falls back to the first ``n`` IDs in storage order when the H5 lacks
        the expected splits (test-only cohorts, smoke H5s without folds).
        """
        with h5py.File(latent_h5, "r") as f:

            def _decode(ds) -> list[str]:  # type: ignore[no-untyped-def]
                return [b.decode() if isinstance(b, bytes) else str(b) for b in ds[:]]

            all_ids = _decode(f["ids"])
            val_path = "splits/cv/fold_0/val"
            pool_patient_keys = _decode(f[val_path]) if val_path in f else list(all_ids)
            has_csr = "patients/offsets" in f and "patients/keys" in f
            offsets = f["patients/offsets"][:] if has_csr else None
            csr_keys = _decode(f["patients/keys"]) if has_csr else None
        rng = np.random.default_rng(int(seed))
        n_pick = min(int(n), len(pool_patient_keys))
        if n_pick == 0:
            return []
        chosen = sorted(rng.choice(len(pool_patient_keys), size=n_pick, replace=False))
        chosen_patients = [pool_patient_keys[int(i)] for i in chosen]
        # Cross-sectional fast path: the split keys are already scan IDs.
        if not has_csr or set(chosen_patients).issubset(set(all_ids)):
            return [p for p in chosen_patients if p in set(all_ids)][:n_pick]
        # Longitudinal CSR expansion: take the FIRST scan of each chosen
        # patient so the sample size stays at ``n`` patients (not n × scans).
        key_to_pos = {k: i for i, k in enumerate(csr_keys)}
        scan_ids: list[str] = []
        for pk in chosen_patients:
            if pk not in key_to_pos:
                continue
            start = int(offsets[key_to_pos[pk]])
            scan_ids.append(all_ids[start])
        return scan_ids

    # ------------------------------------------------------------------

    def _load_latent(self, latent_h5: Path, pid: str) -> torch.Tensor:
        """Read ``latents/t1c[pid]`` and return a ``(C, h, w, d)`` tensor."""
        with h5py.File(latent_h5, "r") as f:
            ids = [b.decode() if isinstance(b, bytes) else str(b) for b in f["ids"][:]]
            idx = {pid_: i for i, pid_ in enumerate(ids)}
            if pid not in idx:
                raise LatentAugEquivarianceError(f"patient '{pid}' not in {latent_h5}/ids")
            arr = f["latents/t1c"][idx[pid]]  # (4, h, w, d)
        return torch.from_numpy(np.ascontiguousarray(arr)).float()

    # ------------------------------------------------------------------

    def _decode(self, vae: MaisiDecoder, z: torch.Tensor, crop_spec) -> torch.Tensor:
        """Decode a latent to the ``(H, W, D)`` ``[0, 1]`` box volume."""
        with torch.inference_mode():
            out = vae.decode(z.unsqueeze(0).to(self.device), crop_spec=crop_spec)
        return out.image[0, 0].float().clamp(0.0, 1.0)

    # ------------------------------------------------------------------

    def _process_one(
        self,
        *,
        vae: MaisiDecoder,
        image_metrics: ImageMetrics,
        cohort: _CohortCfg,
        pid: str,
        z: torch.Tensor,
        recon: torch.Tensor,
        real_box: torch.Tensor,
        crop_spec,
        transform: _TransformInstance,
    ) -> tuple[float, float]:
        """Compute one (cohort, patient, transform) PSNR/SSIM pair."""
        # Latent-side transform — operate on a dict whose only key is z_t1c
        # so the operator's ``LATENT_KEYS`` filter does the right thing.
        z_batch = {"z_t1c": z.clone()}
        z_aug = transform.operator.apply_latent(z_batch, transform.params)["z_t1c"]
        latent_path = self._decode(vae, z_aug, crop_spec)
        # Image-side transform — applied to the recon so the comparison is
        # decoder-noise-invariant: both paths start from D(z).
        image_path = transform.operator.apply_image(recon.detach().clone(), transform.params)
        psnr, ssim = full_volume_psnr_ssim(
            latent_path.to(self.device),
            image_path.to(self.device),
            image_metrics,
        )
        return float(psnr), float(ssim)

    # ------------------------------------------------------------------

    def _vae_recon_floor(
        self,
        recon: torch.Tensor,
        real_box: torch.Tensor,
        image_metrics: ImageMetrics,
    ) -> tuple[float, float]:
        return full_volume_psnr_ssim(recon, real_box, image_metrics)

    # ------------------------------------------------------------------

    def run(self) -> Path:
        cfg = self.cfg
        timestamp = now_iso_utc().replace(":", "-")
        run_dir = Path(cfg.output_root) / timestamp
        (run_dir / "figures").mkdir(parents=True, exist_ok=True)
        (run_dir / "tables").mkdir(parents=True, exist_ok=True)
        logger.info("equivariance preflight: run_dir=%s device=%s", run_dir, self.device)

        # Persist a copy of the resolved config for provenance.
        (run_dir / "config.resolved.yaml").write_text(
            yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False)
        )
        if self.config_yaml_path is not None:
            import shutil

            shutil.copy2(self.config_yaml_path, run_dir / "config.original.yaml")

        # Build the VAE + metric harness (single load, reused for every
        # patient).
        handle = load_autoencoder(cfg.vae_checkpoint, device=str(self.device))
        vae = MaisiDecoder(handle=handle)
        image_metrics = ImageMetrics(data_range=1.0)

        transform_instances = self._build_transform_instances()
        logger.info(
            "evaluating %d transform instance(s): %s",
            len(transform_instances),
            [t.label() for t in transform_instances],
        )

        # Containers.
        per_row: list[dict[str, Any]] = []
        aggregates: dict[str, _Aggregate] = {
            t.label(): _Aggregate(transform=t.label()) for t in transform_instances
        }
        recon_floor: dict[str, list[tuple[float, float]]] = {c.name: [] for c in cfg.cohorts}
        figure_counts: dict[tuple[str, str], int] = {}

        for cohort_idx, cohort in enumerate(cfg.cohorts):
            patient_ids = self._sample_patient_ids(
                cohort.latent_h5,
                n=cfg.n_patients_per_cohort,
                seed=cfg.seed + cohort_idx,
            )
            logger.info("cohort=%s n_patients=%d", cohort.name, len(patient_ids))
            for p_idx, pid in enumerate(patient_ids):
                try:
                    crop_spec = build_crop_spec_from_h5(cohort.image_h5, pid)
                    real_box = load_real_t1c_box(cohort.image_h5, pid, crop_spec).to(self.device)
                    z = self._load_latent(cohort.latent_h5, pid).to(self.device)
                    recon = self._decode(vae, z, crop_spec)
                except Exception as exc:
                    logger.warning(
                        "cohort=%s pid=%s: load/decode failed (%s); skipping.",
                        cohort.name,
                        pid,
                        exc,
                    )
                    continue

                floor_psnr, floor_ssim = self._vae_recon_floor(recon, real_box, image_metrics)
                recon_floor[cohort.name].append((floor_psnr, floor_ssim))

                for ti in transform_instances:
                    try:
                        psnr, ssim = self._process_one(
                            vae=vae,
                            image_metrics=image_metrics,
                            cohort=cohort,
                            pid=pid,
                            z=z,
                            recon=recon,
                            real_box=real_box,
                            crop_spec=crop_spec,
                            transform=ti,
                        )
                    except Exception as exc:
                        logger.warning(
                            "cohort=%s pid=%s T=%s: failed (%s); skipping.",
                            cohort.name,
                            pid,
                            ti.label(),
                            exc,
                        )
                        continue
                    aggregates[ti.label()].add(psnr, ssim)
                    per_row.append(
                        {
                            "cohort": cohort.name,
                            "patient_id": pid,
                            "transform": ti.label(),
                            "transform_name": ti.name,
                            "param_tag": ti.tag,
                            "psnr_db": f"{psnr:.4f}",
                            "ssim": f"{ssim:.6f}",
                            "vae_floor_psnr_db": f"{floor_psnr:.4f}",
                            "vae_floor_ssim": f"{floor_ssim:.6f}",
                        }
                    )

                    # Figure for the first few patients of each pair.
                    key = (cohort.name, ti.label())
                    n_done = figure_counts.get(key, 0)
                    if n_done < cfg.figures_per_pair:
                        try:
                            with torch.inference_mode():
                                z_batch = {"z_t1c": z.clone()}
                                z_aug = ti.operator.apply_latent(z_batch, ti.params)["z_t1c"]
                                latent_path = self._decode(vae, z_aug, crop_spec)
                                image_path = ti.operator.apply_image(
                                    recon.detach().clone(), ti.params
                                )
                            slices = select_content_slices(
                                real_box.cpu(),
                                n_slices=cfg.n_slices_for_figure,
                                offset=cfg.figure_slice_offset,
                            )
                            safe_tag = ti.tag.replace("+", "p").replace("-", "m")
                            out_path = (
                                run_dir
                                / "figures"
                                / f"{cohort.name}_{ti.name}_{safe_tag}_{pid}.png"
                            )
                            render_equivariance_panel(
                                real=real_box.cpu(),
                                recon=recon.cpu(),
                                image_path=image_path.cpu(),
                                latent_path=latent_path.cpu(),
                                slice_indices=slices,
                                cohort=cohort.name,
                                transform_name=ti.name,
                                param_tag=ti.tag,
                                psnr_db=psnr,
                                ssim=ssim,
                                out_path=out_path,
                            )
                            figure_counts[key] = n_done + 1
                        except Exception as exc:
                            logger.warning(
                                "figure render failed for %s/%s/%s: %s",
                                cohort.name,
                                ti.label(),
                                pid,
                                exc,
                            )

                if (p_idx + 1) % cfg.log_every_n_patients == 0:
                    logger.info(
                        "cohort=%s [%d/%d] %s",
                        cohort.name,
                        p_idx + 1,
                        len(patient_ids),
                        pid,
                    )

        if not per_row:
            raise LatentAugEquivarianceError(
                "preflight produced zero metric rows — every patient failed; "
                "check the data paths and the VAE checkpoint."
            )

        # ----------------------------------------------------------
        # Per-patient + per-transform CSVs.
        # ----------------------------------------------------------
        per_patient_path = run_dir / "tables" / "per_patient_metrics.csv"
        with per_patient_path.open("w", newline="") as f:
            cols = list(per_row[0].keys())
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in per_row:
                w.writerow(r)

        summary_path = run_dir / "tables" / "per_transform_summary.csv"
        with summary_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "transform",
                    "n",
                    "median_psnr_db",
                    "median_ssim",
                    "min_psnr_db",
                    "min_ssim",
                    "passes",
                ]
            )
            for label, agg in aggregates.items():
                s = agg.summary()
                passes = bool(
                    s["n"] > 0
                    and s["median_psnr_db"] >= cfg.pass_threshold.psnr_db
                    and s["median_ssim"] >= cfg.pass_threshold.ssim
                )
                w.writerow(
                    [
                        label,
                        int(s["n"]),
                        f"{s['median_psnr_db']:.4f}",
                        f"{s['median_ssim']:.6f}",
                        f"{s['min_psnr_db']:.4f}",
                        f"{s['min_ssim']:.6f}",
                        passes,
                    ]
                )

        # Summary boxplot across all patients.
        try:
            render_summary_boxplot(per_row, run_dir / "figures" / "psnr_ssim_distribution.png")
        except Exception as exc:  # pragma: no cover — purely cosmetic
            logger.warning("summary boxplot failed: %s", exc)

        # ----------------------------------------------------------
        # decision.json — the machine-readable contract.
        # ----------------------------------------------------------
        per_transform_summary: dict[str, Any] = {}
        latent_safe_aug_names: set[str] = set()
        rejected: list[str] = []
        for label, agg in aggregates.items():
            s = agg.summary()
            passes = bool(
                s["n"] > 0
                and s["median_psnr_db"] >= cfg.pass_threshold.psnr_db
                and s["median_ssim"] >= cfg.pass_threshold.ssim
            )
            per_transform_summary[label] = {
                "n": int(s["n"]),
                "median_psnr_db": float(s["median_psnr_db"]),
                "median_ssim": float(s["median_ssim"]),
                "min_psnr_db": float(s["min_psnr_db"]),
                "min_ssim": float(s["min_ssim"]),
                "passes": passes,
            }
            # An augmentation NAME (not param-tag) is admitted iff ALL its
            # evaluated parameter draws pass. That is the strict reading of
            # equivariance and the safer choice for the training routine.
            name = label.split("[")[0]
            if passes:
                latent_safe_aug_names.add(name)
            else:
                rejected.append(label)
        # Strip names where any param-grid entry failed.
        all_names = {ti.name for ti in transform_instances}
        failing_names = {
            ti.name for ti in transform_instances if not per_transform_summary[ti.label()]["passes"]
        }
        latent_safe_aug_names -= failing_names
        image_domain_only = sorted(failing_names & {"gamma"})
        truly_rejected = sorted(failing_names - set(image_domain_only))

        recon_floor_summary: dict[str, dict[str, float]] = {}
        for cname, pairs in recon_floor.items():
            if not pairs:
                continue
            psnr_vals = [p for p, _ in pairs]
            ssim_vals = [s for _, s in pairs]
            recon_floor_summary[cname] = {
                "n": len(pairs),
                "median_psnr_db": statistics.median(psnr_vals),
                "median_ssim": statistics.median(ssim_vals),
            }

        decision = {
            "schema_version": DECISION_SCHEMA_VERSION,
            "produced_at": now_iso_utc(),
            "producer": "vena.preflight.latent_aug_equivariance:1.0",
            "n_patients_per_cohort": cfg.n_patients_per_cohort,
            "cohorts_tested": [c.name for c in cfg.cohorts],
            "modalities_tested": ["t1c"],
            "pass_threshold": cfg.pass_threshold.model_dump(),
            "latent_safe_augmentations": sorted(latent_safe_aug_names),
            "image_domain_only_augmentations": image_domain_only,
            "rejected_augmentations": truly_rejected,
            "per_transform_summary": per_transform_summary,
            "vae_recon_floor": recon_floor_summary,
            "vae_checkpoint": str(cfg.vae_checkpoint),
        }
        (run_dir / "decision.json").write_text(json.dumps(decision, indent=2))

        # ----------------------------------------------------------
        # report.md — human-readable summary.
        # ----------------------------------------------------------
        self._write_report(run_dir, decision, aggregates)

        # ----------------------------------------------------------
        # Update LATEST symlink.
        # ----------------------------------------------------------
        latest = Path(cfg.output_root) / "LATEST"
        try:
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(run_dir.name)
        except OSError as exc:
            logger.warning("could not update LATEST symlink: %s", exc)

        logger.info("equivariance preflight complete: %s", run_dir)
        return run_dir

    # ------------------------------------------------------------------

    def _write_report(
        self,
        run_dir: Path,
        decision: dict[str, Any],
        aggregates: dict[str, _Aggregate],
    ) -> None:
        lines: list[str] = []
        lines.append("# Latent-augmentation equivariance preflight\n")
        lines.append(f"Produced: {decision['produced_at']}  ")
        lines.append(f"VAE checkpoint: `{decision['vae_checkpoint']}`  ")
        lines.append(
            f"Cohorts: {', '.join(decision['cohorts_tested'])}  "
            f"N per cohort: {decision['n_patients_per_cohort']}"
        )
        lines.append("")
        lines.append(
            "## Pass threshold\n"
            f"- PSNR ≥ **{decision['pass_threshold']['psnr_db']} dB**\n"
            f"- SSIM ≥ **{decision['pass_threshold']['ssim']}**\n"
            "- Median over all evaluated patients.\n"
        )
        lines.append("## VAE reconstruction floor (D(E(x)) vs real T1c)\n")
        if decision["vae_recon_floor"]:
            lines.append("| Cohort | n | median PSNR (dB) | median SSIM |")
            lines.append("|---|---|---|---|")
            for c, s in decision["vae_recon_floor"].items():
                lines.append(
                    f"| {c} | {s['n']} | {s['median_psnr_db']:.2f} | {s['median_ssim']:.4f} |"
                )
        else:
            lines.append("(no patients succeeded)\n")
        lines.append("")
        lines.append("## Per-transform results\n")
        lines.append(
            "| Transform | n | median PSNR (dB) | median SSIM | min PSNR | min SSIM | passes |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for label, summary in decision["per_transform_summary"].items():
            lines.append(
                f"| `{label}` | {summary['n']} | "
                f"{summary['median_psnr_db']:.2f} | "
                f"{summary['median_ssim']:.4f} | "
                f"{summary['min_psnr_db']:.2f} | "
                f"{summary['min_ssim']:.4f} | "
                f"{'✅' if summary['passes'] else '❌'} |"
            )
        lines.append("")
        lines.append("## Decision\n")
        lines.append(f"- Latent-safe augmentations: `{decision['latent_safe_augmentations']}`")
        lines.append(f"- Image-domain only: `{decision['image_domain_only_augmentations']}`")
        lines.append(f"- Rejected: `{decision['rejected_augmentations']}`")
        lines.append("")
        lines.append("## Figures\n")
        for png in sorted((run_dir / "figures").glob("*.png")):
            lines.append(f"- ![{png.name}](figures/{png.name})")
        lines.append("")
        (run_dir / "report.md").write_text("\n".join(lines))
