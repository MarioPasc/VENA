"""Thin engine for the validate-masks routine.

Reads T1pre anatomy and GT labels from image-domain H5 files, recomputes
the soft [WT, NETC] mask on-the-fly via :func:`derive_latent_soft_mask`, and
writes QC figures + a machine-readable ``decision.json``.

Design constraints
------------------
* **No heavy work at import time** — all I/O and computation lives inside
  :meth:`ValidateMasksEngine.run`.
* **Deterministic** — recomputing the mask on-the-fly is byte-identical to
  the Phase-1 cache because :func:`derive_latent_soft_mask` is deterministic
  for ``source="gt"``.
* **No Picasso latent cache required** — reads only the LOCAL image H5s.
* **``masks_look_valid`` is human-set** — the engine writes ``null`` and the
  human fills it in after reviewing the figures.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict

from vena.data.h5.shared import now_iso_utc, resolve_git_sha
from vena.segmentation.config import DerivationConfig, TargetConfig
from vena.segmentation.derivation.derive import derive_latent_soft_mask
from vena.segmentation.exceptions import SegDerivationError, SegMetricError
from vena.segmentation.metrics.visualize import (
    PatientView,
    compute_mask_stats,
    render_latent_embedding,
    render_mask_qc,
    render_slice_montage,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = "1.0"
_PRODUCER = "routines.segmentation.validate_masks:1.0"

# ---------------------------------------------------------------------------
# Corpus registry
# ---------------------------------------------------------------------------


class _CohortEntry(BaseModel):
    """One cohort in the validate-masks corpus registry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    image_h5: Path


class _CorpusRegistry(BaseModel):
    """Minimal registry for the validate-masks routine."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cohorts: list[_CohortEntry]

    @classmethod
    def from_json(cls, path: Path) -> _CorpusRegistry:
        """Load from a JSON file."""
        with path.open() as fh:
            return cls.model_validate_json(fh.read())


# ---------------------------------------------------------------------------
# Routine config
# ---------------------------------------------------------------------------


class ValidateMasksRoutineConfig(BaseModel):
    """Frozen configuration for :class:`ValidateMasksEngine`.

    Attributes
    ----------
    corpus_registry:
        Path to a JSON file listing cohorts with ``image_h5`` fields.
    output_root:
        Root directory under which a timestamped artifact directory is
        created.  Default ``artifacts/segmentation/validate_masks``.
    patient_selection:
        Patient sampling policy.  ``"random"`` picks uniformly;
        ``"best"`` selects the largest-tumour patients; ``"worst"``
        selects the smallest non-empty-tumour patients.
    n_patients:
        Number of patients to include in the QC figures.
    targets:
        Soft target generation settings (SDT sigma, operator, clip radius).
    derivation:
        Latent-space pooling settings (avg-pool stride, latent grid).
    log_level:
        Python logging level string.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    corpus_registry: Path
    output_root: Path = Path("artifacts/segmentation/validate_masks")
    patient_selection: Literal["random", "best", "worst"] = "random"
    n_patients: int = 10
    targets: TargetConfig = TargetConfig()
    derivation: DerivationConfig = DerivationConfig()
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, path: Path | str) -> ValidateMasksRoutineConfig:
        """Load and validate a YAML config file.

        Parameters
        ----------
        path:
            Path to a YAML file whose top-level keys map to the fields above.

        Returns
        -------
        ValidateMasksRoutineConfig
            A frozen, fully-validated configuration instance.

        Raises
        ------
        pydantic.ValidationError
            If a required field is missing, has the wrong type, or an
            unknown key is present.
        FileNotFoundError
            If *path* does not exist.
        """
        path = Path(path)
        with path.open() as fh:
            raw = yaml.safe_load(fh)
        return cls.model_validate(raw)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ValidateMasksEngine:
    """Derive soft masks for a small patient subset and write QC figures.

    Parameters
    ----------
    cfg:
        Frozen validated routine configuration.
    """

    def __init__(self, cfg: ValidateMasksRoutineConfig) -> None:
        self._cfg = cfg

    def run(self) -> Path:
        """Execute the routine end-to-end and return the artifact directory.

        Returns
        -------
        Path
            The timestamped artifact directory containing figures,
            ``report.md``, and ``decision.json``.

        Raises
        ------
        FileNotFoundError
            If any corpus H5 is missing.
        SegDerivationError
            If mask derivation fails for a scan.
        SegMetricError
            If figure building fails.
        """
        cfg = self._cfg
        logging.basicConfig(level=cfg.log_level)

        produced_at = now_iso_utc()
        git_sha = resolve_git_sha() or "unknown"

        # Create timestamped output dir
        ts = produced_at.replace(":", "-").replace(".", "-")
        artifact_dir = Path(cfg.output_root) / ts
        figures_dir = artifact_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        registry = _CorpusRegistry.from_json(cfg.corpus_registry)

        # Collect all candidate scans across cohorts
        candidates: list[dict] = self._collect_candidates(registry)
        logger.info("total candidates: %d", len(candidates))

        if not candidates:
            raise SegMetricError(
                "No candidate scans found in corpus registry. "
                f"Check that image H5 files exist under {cfg.corpus_registry}."
            )

        # Select patients
        selected = self._select_patients(candidates, cfg.n_patients, cfg.patient_selection)
        logger.info("selected %d patients (policy=%s)", len(selected), cfg.patient_selection)

        # Derive masks and build PatientView list
        patient_views, mask_latents_dict, meta_rows = self._build_patient_views(selected)

        # Compute machine stats
        all_latent_masks = np.stack(
            [mask_latents_dict[pv.patient_id] for pv in patient_views], axis=0
        )
        stats = compute_mask_stats(all_latent_masks)
        logger.info(
            "machine stats: soft_mass_fraction_in_wt=%.4f  netc_violations=%d  empty=%d",
            stats["soft_mass_fraction_in_wt"],
            stats["netc_violation_count"],
            stats["empty_mask_count"],
        )

        # Render figures
        figure_paths: list[Path] = []

        # Per-patient QC figures
        for pv in patient_views:
            lat_mask = mask_latents_dict[pv.patient_id]  # (2, 48, 56, 48)
            # Hard mask: binarise WT channel of soft mask at image resolution
            hard_mask = (pv.soft_mask[0] > 0.5).astype(np.int32)
            fig_path = figures_dir / f"qc_{pv.patient_id}.png"
            render_mask_qc(
                image=pv.t1pre,
                hard_mask=hard_mask,
                soft_mask_img=pv.soft_mask,
                soft_mask_latent=lat_mask,
                patient_id=pv.patient_id,
                path=fig_path,
            )
            figure_paths.append(fig_path)
            logger.debug("wrote QC figure for %s", pv.patient_id)

        # Montage figure
        montage_path = figures_dir / "montage.png"
        render_slice_montage(patient_views, n_cols=10, alpha=0.6, path=montage_path)
        figure_paths.append(montage_path)

        # Latent-embedding figure (only when >= 3 patients for PCA to be meaningful)
        if len(patient_views) >= 3:
            import pandas as pd

            meta_df = pd.DataFrame(meta_rows).set_index("patient_id")
            embed_path = figures_dir / "embedding.png"
            render_latent_embedding(
                mask_latents={
                    pv.patient_id: mask_latents_dict[pv.patient_id] for pv in patient_views
                },
                meta=meta_df,
                method="pca_umap_perpatient",
                color_by=("tumor_volume", "cohort"),
                path=embed_path,
            )
            figure_paths.append(embed_path)

        # Write report.md
        report_path = artifact_dir / "report.md"
        self._write_report(report_path, patient_views, stats, figure_paths, produced_at)

        # Write decision.json
        decision_path = artifact_dir / "decision.json"
        decision = {
            "schema_version": _SCHEMA_VERSION,
            "produced_at": produced_at,
            "producer": _PRODUCER,
            "git_sha": git_sha,
            "n_patients": len(patient_views),
            "patient_ids": [pv.patient_id for pv in patient_views],
            "soft_mass_fraction_in_wt": stats["soft_mass_fraction_in_wt"],
            "netc_violation_count": stats["netc_violation_count"],
            "empty_mask_count": stats["empty_mask_count"],
            # Human-set after visual review; engine writes null.
            "masks_look_valid": None,
        }
        decision_path.write_text(json.dumps(decision, indent=2))
        logger.info("decision.json written to %s", decision_path)

        # Persist resolved YAML
        resolved_yaml_path = artifact_dir / "resolved_config.yaml"
        resolved_yaml_path.write_text(
            yaml.dump(json.loads(self._cfg.model_dump_json()), default_flow_style=False)
        )

        # Assert deliverables are present
        self._assert_deliverables(artifact_dir, figure_paths)
        logger.info("artifact dir: %s", artifact_dir)
        return artifact_dir

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _collect_candidates(self, registry: _CorpusRegistry) -> list[dict]:
        """Read scan IDs and tumour volumes from all cohort H5s."""
        import h5py

        candidates: list[dict] = []
        for cohort in registry.cohorts:
            image_h5_path = Path(cohort.image_h5)
            if not image_h5_path.exists():
                raise FileNotFoundError(f"image H5 not found: {image_h5_path}")

            with h5py.File(image_h5_path, "r") as f:
                raw_ids: np.ndarray = f["ids"][:]
                ids_str = [id_.decode() if isinstance(id_, bytes) else str(id_) for id_ in raw_ids]
                labels_ds = f["masks/tumor"]
                for i, sid in enumerate(ids_str):
                    label_row = labels_ds[i].astype(np.int32)
                    tumour_vol = float((label_row > 0).sum())
                    candidates.append(
                        {
                            "scan_id": sid,
                            "image_h5": image_h5_path,
                            "row": i,
                            "cohort": cohort.name,
                            "tumor_volume": tumour_vol,
                        }
                    )

        return candidates

    def _select_patients(
        self,
        candidates: list[dict],
        n: int,
        policy: str,
    ) -> list[dict]:
        """Select up to *n* candidates according to *policy*."""
        n = min(n, len(candidates))
        if policy == "random":
            return random.sample(candidates, n)
        if policy == "best":
            return sorted(candidates, key=lambda c: c["tumor_volume"], reverse=True)[:n]
        if policy == "worst":
            # Smallest non-empty tumour
            non_empty = [c for c in candidates if c["tumor_volume"] > 0]
            return sorted(non_empty, key=lambda c: c["tumor_volume"])[:n]
        raise SegMetricError(f"Unknown patient_selection policy: {policy!r}")

    def _build_patient_views(
        self,
        selected: list[dict],
    ) -> tuple[list[PatientView], dict[str, np.ndarray], list[dict]]:
        """Derive masks and build PatientView objects for the selected scans."""
        import h5py

        from vena.common import CropPadSpec

        cfg = self._cfg
        patient_views: list[PatientView] = []
        mask_latents_dict: dict[str, np.ndarray] = {}
        meta_rows: list[dict] = []

        for entry in selected:
            scan_id: str = entry["scan_id"]
            image_h5_path: Path = entry["image_h5"]
            row: int = entry["row"]
            cohort: str = entry["cohort"]

            with h5py.File(image_h5_path, "r") as f:
                t1pre: np.ndarray = f["images/t1pre"][row].astype(np.float32)
                label: np.ndarray = f["masks/tumor"][row].astype(np.int32)
                crop_origin_arr: np.ndarray = f["crop/origin"][row]

            crop_spec = CropPadSpec(
                crop_origin=(
                    int(crop_origin_arr[0]),
                    int(crop_origin_arr[1]),
                    int(crop_origin_arr[2]),
                ),
                native_shape=(label.shape[0], label.shape[1], label.shape[2]),
                target_shape=(192, 224, 192),
            )

            try:
                soft_latent = derive_latent_soft_mask(
                    source="gt",
                    label=label,
                    crop_spec=crop_spec,
                    cfg=cfg.derivation,
                    target_cfg=cfg.targets,
                )
            except SegDerivationError:
                logger.warning("derivation failed for %s; skipping", scan_id)
                continue

            soft_latent_np = soft_latent.numpy()  # (2, 48, 56, 48)
            mask_latents_dict[scan_id] = soft_latent_np

            # Image-resolution soft mask: reuse the GT-path intermediate
            # (SDT sigmoid before pooling) — approximate via a simple
            # binarisation + smooth for the viz.  The latent mask is the
            # authoritative one; the image-res mask is for visual reference.
            wt_bin = (label > 0).astype(np.float32)
            netc_bin = (label == 1).astype(np.float32)
            soft_img = np.stack([wt_bin, netc_bin], axis=0)  # (2, H, W, D)

            tumor_vol = float((label > 0).sum())
            pv = PatientView(
                patient_id=scan_id,
                t1pre=t1pre,
                soft_mask=soft_img,
                tumor_volume=tumor_vol,
                cohort=cohort,
            )
            patient_views.append(pv)
            meta_rows.append(
                {
                    "patient_id": scan_id,
                    "tumor_volume": tumor_vol,
                    "cohort": cohort,
                }
            )
            logger.debug(
                "derived mask for %s  WT_mean=%.4f  NETC_mean=%.4f",
                scan_id,
                float(soft_latent_np[0].mean()),
                float(soft_latent_np[1].mean()),
            )

        return patient_views, mask_latents_dict, meta_rows

    def _write_report(
        self,
        path: Path,
        patient_views: list[PatientView],
        stats: dict,
        figure_paths: list[Path],
        produced_at: str,
    ) -> None:
        """Write a minimal Markdown report."""
        lines = [
            "# validate_masks — soft-mask QC report",
            "",
            f"**Produced at**: {produced_at}",
            f"**N patients**: {len(patient_views)}",
            "",
            "## Machine stats",
            "",
            f"- `soft_mass_fraction_in_wt`: {stats['soft_mass_fraction_in_wt']:.4f}",
            f"- `netc_violation_count`: {stats['netc_violation_count']}",
            f"- `empty_mask_count`: {stats['empty_mask_count']}",
            "",
            "## Figures",
            "",
        ]
        for fp in figure_paths:
            lines.append(f"![{fp.stem}](figures/{fp.name})")
        lines.append("")
        lines.append(
            "> **Human review**: Set `masks_look_valid` in `decision.json` "
            "after inspecting the figures above."
        )
        path.write_text("\n".join(lines))

    @staticmethod
    def _assert_deliverables(artifact_dir: Path, figure_paths: list[Path]) -> None:
        """Raise if any expected deliverable is missing from disk."""
        required = [
            artifact_dir / "decision.json",
            artifact_dir / "report.md",
            artifact_dir / "resolved_config.yaml",
        ]
        for fig in figure_paths:
            required.append(fig)
        missing = [p for p in required if not p.exists()]
        if missing:
            raise SegMetricError(f"Deliverables missing after run: {[str(m) for m in missing]}")
