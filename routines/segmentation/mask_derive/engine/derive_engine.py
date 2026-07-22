"""Thin engine for the mask-derive routine.

Reads GT labels and crop origins from image-domain H5 files, derives soft
[WT, NETC] probability maps in MAISI latent space, and appends them as
``masks/tumor_latent_soft`` (GT) or ``masks/tumor_latent_pred`` (predicted)
to the corresponding latent-domain H5 files.

The engine is source-agnostic: switching from ``source: gt`` to
``source: predicted`` in the YAML config is the only change required to
swap the oracle for the segmenter output (the swap guarantee).

Design constraints
------------------
* **No heavy work at import time** — all I/O and computation lives inside
  :meth:`MaskDeriveEngine.run`.
* **ID-aligned reads** — rows are matched by scan ID, never by position;
  both H5 files may have different row orderings.
* **Idempotent** — re-running removes the target group and recreates it;
  all other groups remain byte-identical.
* **Validate before returning** — :func:`assert_latent_soft_mask_group_valid`
  is called before the artifact path is returned.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict

from vena.data.h5.latent_domain.manifest import (
    LATENT_CROP_BOX,
    LATENT_SCHEMA_VERSION_SOFT,
    PRED_MASK_GROUP,
    SOFT_MASK_GROUP,
    assert_latent_soft_mask_group_valid,
)
from vena.data.h5.shared import now_iso_utc, resolve_git_sha
from vena.segmentation.config import DerivationConfig, TargetConfig
from vena.segmentation.derivation.derive import derive_latent_soft_mask
from vena.segmentation.exceptions import SegDerivationError

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config types
# ---------------------------------------------------------------------------


class _CohortEntry(BaseModel):
    """One cohort in the corpus registry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    image_h5: Path
    latent_h5: Path


class _CorpusRegistry(BaseModel):
    """Minimal corpus registry for the mask-derive routine."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cohorts: list[_CohortEntry]

    @classmethod
    def from_json(cls, path: Path) -> _CorpusRegistry:
        """Load from a JSON file."""
        with path.open() as fh:
            return cls.model_validate_json(fh.read())


class MaskDeriveRoutineConfig(BaseModel):
    """Frozen configuration for :class:`MaskDeriveEngine`.

    Attributes
    ----------
    source:
        Which derivation path to run.  ``"gt"`` derives the oracle soft mask
        directly from BraTS integer labels (no segmenter required).
        ``"predicted"`` applies a trained segmenter (Phase-2, placeholder).
    corpus_registry:
        Path to a JSON file listing cohorts, each with ``image_h5`` and
        ``latent_h5`` paths.
    targets:
        Soft target generation settings (SDT sigma, operator, clip radius).
        Only used when ``source="gt"``.
    derivation:
        Latent-space pooling settings (avg-pool stride, latent grid).
    artifact_dir:
        Directory under which a timestamped subdirectory is created for the
        provenance artefacts (resolved YAML, decision JSON).
    log_level:
        Python logging level string (``"INFO"`` by default).
    segmenter_checkpoints:
        Paths to K-fold segmenter checkpoints.  Only used when
        ``source="predicted"``; leave empty for the GT path.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: Literal["gt", "predicted"]
    corpus_registry: Path
    targets: TargetConfig = TargetConfig()
    derivation: DerivationConfig = DerivationConfig()
    artifact_dir: Path
    log_level: str = "INFO"
    segmenter_checkpoints: list[Path] = []

    @classmethod
    def from_yaml(cls, path: Path | str) -> MaskDeriveRoutineConfig:
        """Load and validate a YAML config file.

        Parameters
        ----------
        path:
            Path to a YAML file whose top-level keys map to the fields above.

        Returns
        -------
        MaskDeriveRoutineConfig
            A frozen, fully-validated configuration instance.

        Raises
        ------
        pydantic.ValidationError
            If a required field is missing, has the wrong type, or an
            unknown key is present.
        FileNotFoundError
            If ``path`` does not exist.
        """
        path = Path(path)
        with path.open() as fh:
            raw = yaml.safe_load(fh)
        return cls.model_validate(raw)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class MaskDeriveEngine:
    """Derives and caches soft [WT, NETC] masks in MAISI latent space.

    Parameters
    ----------
    cfg:
        Frozen routine configuration.
    """

    _PRODUCER: str = "routines.segmentation.mask_derive:0.1.0"

    def __init__(self, cfg: MaskDeriveRoutineConfig) -> None:
        self._cfg = cfg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Path:
        """Derive and cache masks for all cohorts in the corpus registry.

        Returns
        -------
        Path
            Path to the timestamped artifact directory containing
            ``decision.json`` and the resolved YAML config.

        Raises
        ------
        SegDerivationError
            If any cohort H5 is unreachable, an ID cannot be aligned, or
            post-write validation fails.
        FileNotFoundError
            If the corpus registry JSON does not exist.
        """
        cfg = self._cfg

        logging.basicConfig(level=getattr(logging, cfg.log_level.upper(), logging.INFO))

        # ----------------------------------------------------------
        # Artifact directory
        # ----------------------------------------------------------
        timestamp = now_iso_utc().replace(":", "-").replace(" ", "T")
        artifact_dir = Path(cfg.artifact_dir) / timestamp
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Persist config for reproducibility.
        config_path = artifact_dir / "config.yaml"
        with config_path.open("w") as fh:
            yaml.safe_dump(cfg.model_dump(mode="json"), fh, default_flow_style=False)

        git_sha = resolve_git_sha() or "unknown"

        # ----------------------------------------------------------
        # Load corpus registry
        # ----------------------------------------------------------
        registry = _CorpusRegistry.from_json(Path(cfg.corpus_registry))

        group_name = SOFT_MASK_GROUP if cfg.source == "gt" else PRED_MASK_GROUP

        total_written = 0
        cohort_summaries: list[dict] = []

        for cohort_entry in registry.cohorts:
            n_written = self._process_cohort(
                cohort_entry=cohort_entry,
                group_name=group_name,
                git_sha=git_sha,
                config_json=config_path.read_text(),
            )
            total_written += n_written
            cohort_summaries.append({"name": cohort_entry.name, "n_written": n_written})
            logger.info("cohort=%s  n_written=%d", cohort_entry.name, n_written)

        # ----------------------------------------------------------
        # Decision JSON
        # ----------------------------------------------------------
        decision = {
            "schema_version": "0.1.0",
            "produced_at": now_iso_utc(),
            "producer": self._PRODUCER,
            "source": cfg.source,
            "group_name": group_name,
            "latent_schema_version": LATENT_SCHEMA_VERSION_SOFT,
            "corpus_registry": str(cfg.corpus_registry),
            "git_sha": git_sha,
            "total_written": total_written,
            "cohorts": cohort_summaries,
        }
        decision_path = artifact_dir / "decision.json"
        with decision_path.open("w") as fh:
            json.dump(decision, fh, indent=2)

        logger.info(
            "mask_derive done: source=%s  group=%s  total_written=%d  artifact=%s",
            cfg.source,
            group_name,
            total_written,
            artifact_dir,
        )
        return artifact_dir

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_cohort(
        self,
        *,
        cohort_entry: _CohortEntry,
        group_name: str,
        git_sha: str,
        config_json: str,
    ) -> int:
        """Process one cohort: derive masks and append to the latent H5.

        Returns
        -------
        int
            Number of scans written.
        """
        import h5py

        cfg = self._cfg
        image_h5_path = Path(cohort_entry.image_h5)
        latent_h5_path = Path(cohort_entry.latent_h5)

        if not image_h5_path.exists():
            raise FileNotFoundError(f"image H5 not found: {image_h5_path}")
        if not latent_h5_path.exists():
            raise FileNotFoundError(f"latent H5 not found: {latent_h5_path}")

        # ----------------------------------------------------------
        # Build ID → row-index map for the image H5.
        # ----------------------------------------------------------
        with h5py.File(image_h5_path, "r") as f_img:
            image_ids: np.ndarray = f_img["ids"][:]
            # Decode bytes to str if needed (vlen-str datasets).
            image_ids_str = [
                id_.decode() if isinstance(id_, bytes) else str(id_) for id_ in image_ids
            ]
            image_id_to_idx: dict[str, int] = {id_: i for i, id_ in enumerate(image_ids_str)}

        # ----------------------------------------------------------
        # Read latent IDs to know N and ordering.
        # ----------------------------------------------------------
        with h5py.File(latent_h5_path, "r") as f_lat:
            latent_ids_raw: np.ndarray = f_lat["ids"][:]
        latent_ids_str = [
            id_.decode() if isinstance(id_, bytes) else str(id_) for id_ in latent_ids_raw
        ]
        n_scans = len(latent_ids_str)

        # ----------------------------------------------------------
        # Pre-allocate output array.
        # ----------------------------------------------------------
        lat_h, lat_w, lat_d = cfg.derivation.latent_grid
        masks_out = np.zeros((n_scans, 2, lat_h, lat_w, lat_d), dtype=np.float32)

        # ----------------------------------------------------------
        # Derive per scan.
        # ----------------------------------------------------------
        for i, scan_id in enumerate(latent_ids_str):
            j = image_id_to_idx.get(scan_id)
            if j is None:
                raise SegDerivationError(
                    f"latent ID {scan_id!r} (row {i}) not found in image H5 "
                    f"{image_h5_path}; check that both H5s cover the same cohort."
                )
            mask_tensor = self._derive_one(
                image_h5_path=image_h5_path,
                image_row=j,
                scan_id=scan_id,
            )
            masks_out[i] = mask_tensor.numpy()
            if (i + 1) % 50 == 0:
                logger.debug("derived %d/%d scans for cohort=%s", i + 1, n_scans, cohort_entry.name)

        # ----------------------------------------------------------
        # Write (idempotent: remove existing group first).
        # ----------------------------------------------------------
        produced_at = now_iso_utc()
        with h5py.File(latent_h5_path, "r+") as f_lat:
            # Idempotency: delete the group if it already exists.
            if group_name in f_lat:
                del f_lat[group_name]
                logger.debug("replaced existing group %s in %s", group_name, latent_h5_path)

            dset = f_lat.create_dataset(
                group_name,
                data=masks_out,
                dtype="float32",
                chunks=(1, 2, lat_h, lat_w, lat_d),
                compression="gzip",
                compression_opts=4,
            )
            # Self-describing attrs (h5-design-principles.md principle 4).
            dset.attrs["units"] = "dimensionless"
            dset.attrs["description"] = (
                "Soft [WT, NETC] tumour probability map in MAISI latent space; "
                f"source='{cfg.source}'; channel 0 = WT, channel 1 = NETC; "
                "SDT→sigmoid at image res, avg-pooled 4× to (2, 48, 56, 48)."
            )
            dset.attrs["dtype"] = "float32"
            dset.attrs["leading_dim"] = "n_scans"

            # Bump root schema_version and stamp provenance.
            f_lat.attrs["schema_version"] = LATENT_SCHEMA_VERSION_SOFT
            f_lat.attrs["mask_source"] = cfg.source
            f_lat.attrs["mask_derive_produced_at"] = produced_at
            f_lat.attrs["mask_derive_git_sha"] = git_sha

        # ----------------------------------------------------------
        # Validate before returning.
        # ----------------------------------------------------------
        assert_latent_soft_mask_group_valid(latent_h5_path, group=group_name)
        return n_scans

    def _derive_one(
        self,
        *,
        image_h5_path: Path,
        image_row: int,
        scan_id: str,
    ) -> torch.Tensor:
        """Derive the soft mask for one scan.

        Reads the label and crop/origin from the image H5 at ``image_row``,
        builds a :class:`~vena.common.CropPadSpec`, and calls
        :func:`~vena.segmentation.derivation.derive.derive_latent_soft_mask`.

        Returns
        -------
        torch.Tensor
            Shape ``(2, 48, 56, 48)`` float32.
        """
        import h5py
        import torch  # noqa: F401 — needed for type annotation

        from vena.common import CropPadSpec

        cfg = self._cfg

        with h5py.File(image_h5_path, "r") as f_img:
            label: np.ndarray = f_img["masks/tumor"][image_row].astype(np.int32)
            crop_origin_arr: np.ndarray = f_img["crop/origin"][image_row]

        crop_spec = CropPadSpec(
            crop_origin=(
                int(crop_origin_arr[0]),
                int(crop_origin_arr[1]),
                int(crop_origin_arr[2]),
            ),
            native_shape=(label.shape[0], label.shape[1], label.shape[2]),
            target_shape=LATENT_CROP_BOX,
        )

        if cfg.source == "gt":
            mask = derive_latent_soft_mask(
                source="gt",
                label=label,
                crop_spec=crop_spec,
                cfg=cfg.derivation,
                target_cfg=cfg.targets,
            )
        else:
            # Predicted path: Phase-2 placeholder.  Segmenter checkpoints
            # are declared in cfg.segmenter_checkpoints but no segmenter
            # module exists yet.  The engine raises early here so the
            # infrastructure can be tested end-to-end for the GT path.
            raise SegDerivationError(
                "predicted path requires a trained segmenter (Phase-2). "
                "segmenter_checkpoints is set but no segmenter module is "
                "available yet.  Use source='gt' for Phase-1 caching."
            )

        logger.debug(
            "scan=%s  mask shape=%s  WT_mean=%.3f  NETC_mean=%.3f",
            scan_id,
            tuple(mask.shape),
            float(mask[0].mean()),
            float(mask[1].mean()),
        )
        return mask
