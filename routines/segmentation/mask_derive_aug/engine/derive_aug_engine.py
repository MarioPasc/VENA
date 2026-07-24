"""Thin engine for the mask-derive-aug routine.

Derives ``masks/tumor_latent_soft`` for the 6 offline-augmented latent H5s
(``*_latents_aug.h5``) that the clean ``mask_derive`` routine never touched.
The derivation operator is **identical** to :class:`MaskDeriveEngine` —
both call :func:`~vena.segmentation.derivation.derive.derive_latent_soft_mask`
with ``source="gt"``.  This is a separate routine because the alignment
strategy differs: aug banks use **row-index alignment** (not ID-dict lookup),
since aug IDs are non-unique (4 variant rows share one patient ID).

Design constraints
------------------
* **No heavy work at import time** — all I/O and computation lives inside
  :meth:`MaskDeriveAugEngine.run`.
* **Row-index alignment, not ID lookup** — aug image H5 row ``i`` ↔ aug
  latent H5 row ``i``.  Before processing, three assertions are verified:
  equal length, ``ids`` elementwise equal, ``source_row_index`` elementwise
  equal.  Any failure raises :class:`~vena.segmentation.exceptions.SegDerivationError`
  with a ``PREMISE-FALSE`` label so the caller can report it as a finding.
* **Pre-cropped labels** — aug image H5 stores labels at the LATENT_CROP_BOX
  ``(192, 224, 192)`` already; ``native_shape == target_shape`` so the
  :class:`~vena.common.CropPadSpec` is a no-op.  This matches the clean-cache
  derivation result exactly for v1/v2/v3 rows (intensity-only variants).
* **All-in-memory, write-once** — derive all rows before touching the H5,
  then write in one pass.  Idempotent: removes any existing group first.
* **Validate before returning** — :func:`assert_aug_latent_soft_mask_group_valid`
  is called before the artifact path is returned.
* **Schema bump** — root ``schema_version`` advances from ``"0.2.0"`` to
  ``"0.3.0"``; ``mask_source="gt"`` and ``tumor_region`` dataset attr are
  written to match the clean-cache convention.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict

from vena.data.h5.augmented.latent_domain import (
    AUG_LATENT_SCHEMA_VERSION_SOFT,
    assert_aug_latent_soft_mask_group_valid,
)
from vena.data.h5.latent_domain.manifest import (
    LATENT_CROP_BOX,
    SOFT_MASK_GROUP,
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


class _AugCohortEntry(BaseModel):
    """One cohort's aug H5 pair in the corpus registry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    image_aug_h5: Path
    latent_aug_h5: Path


class _AugCorpusRegistry(BaseModel):
    """Minimal corpus registry for the mask-derive-aug routine."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cohorts: list[_AugCohortEntry]

    @classmethod
    def from_json(cls, path: Path) -> _AugCorpusRegistry:
        """Load from a JSON file."""
        with path.open() as fh:
            return cls.model_validate_json(fh.read())


class MaskDeriveAugRoutineConfig(BaseModel):
    """Frozen configuration for :class:`MaskDeriveAugEngine`.

    Parameters
    ----------
    corpus_registry:
        Path to a JSON file listing cohorts, each with ``image_aug_h5``
        and ``latent_aug_h5`` paths.
    targets:
        Soft target generation settings (SDT sigma, operator, clip radius).
        Must match the settings used for the clean-cache derivation.
    derivation:
        Latent-space pooling settings (avg-pool stride, latent grid).
        Must match the clean-cache derivation settings.
    artifact_dir:
        Directory under which a timestamped subdirectory is created for
        provenance artefacts (resolved YAML, decision JSON).
    log_level:
        Python logging level string (``"INFO"`` by default).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    corpus_registry: Path
    targets: TargetConfig = TargetConfig()
    derivation: DerivationConfig = DerivationConfig()
    artifact_dir: Path
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, path: Path | str) -> MaskDeriveAugRoutineConfig:
        """Load and validate a YAML config file.

        Parameters
        ----------
        path:
            Path to a YAML file whose top-level keys map to the fields above.

        Returns
        -------
        MaskDeriveAugRoutineConfig
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


class MaskDeriveAugEngine:
    """Derives and caches ``masks/tumor_latent_soft`` into aug-latent H5 files.

    Parameters
    ----------
    cfg:
        Frozen routine configuration.
    """

    _PRODUCER: str = "routines.segmentation.mask_derive_aug:0.1.0"

    def __init__(self, cfg: MaskDeriveAugRoutineConfig) -> None:
        self._cfg = cfg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Path:
        """Derive and cache masks for all aug cohorts in the corpus registry.

        Returns
        -------
        Path
            Path to the timestamped artifact directory containing
            ``decision.json`` and the resolved YAML config.

        Raises
        ------
        SegDerivationError
            If any H5 is unreachable, row alignment fails (PREMISE-FALSE),
            or post-write validation fails.
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

        config_path = artifact_dir / "config.yaml"
        with config_path.open("w") as fh:
            yaml.safe_dump(cfg.model_dump(mode="json"), fh, default_flow_style=False)

        git_sha = resolve_git_sha() or "unknown"

        # ----------------------------------------------------------
        # Load corpus registry
        # ----------------------------------------------------------
        registry = _AugCorpusRegistry.from_json(Path(cfg.corpus_registry))

        total_written = 0
        cohort_summaries: list[dict] = []

        for cohort_entry in registry.cohorts:
            n_written = self._process_cohort(
                cohort_entry=cohort_entry,
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
            "source": "gt",
            "group_name": SOFT_MASK_GROUP,
            "aug_latent_schema_version": AUG_LATENT_SCHEMA_VERSION_SOFT,
            "corpus_registry": str(cfg.corpus_registry),
            "git_sha": git_sha,
            "total_written": total_written,
            "cohorts": cohort_summaries,
        }
        decision_path = artifact_dir / "decision.json"
        with decision_path.open("w") as fh:
            json.dump(decision, fh, indent=2)

        logger.info(
            "mask_derive_aug done: group=%s  total_written=%d  artifact=%s",
            SOFT_MASK_GROUP,
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
        cohort_entry: _AugCohortEntry,
        git_sha: str,
        config_json: str,
    ) -> int:
        """Process one aug cohort: derive masks and append to the aug-latent H5.

        Row alignment is by **index** (not ID dict), because aug IDs are
        non-unique (multiple variant rows share one patient ID).  Three
        assertions are verified before processing:

        1. ``len(aug_image_ids) == len(aug_latent_ids)``
        2. ``ids`` arrays are elementwise equal
        3. ``source_row_index`` arrays are elementwise equal

        Any failure raises :class:`~vena.segmentation.exceptions.SegDerivationError`
        with ``PREMISE-FALSE`` in the message.

        Returns
        -------
        int
            Number of rows written.
        """
        import h5py

        cfg = self._cfg
        image_aug_h5_path = Path(cohort_entry.image_aug_h5)
        latent_aug_h5_path = Path(cohort_entry.latent_aug_h5)

        if not image_aug_h5_path.exists():
            raise FileNotFoundError(f"aug image H5 not found: {image_aug_h5_path}")
        if not latent_aug_h5_path.exists():
            raise FileNotFoundError(f"aug latent H5 not found: {latent_aug_h5_path}")

        # ----------------------------------------------------------
        # Read alignment arrays from both H5s.
        # ----------------------------------------------------------
        with h5py.File(image_aug_h5_path, "r") as f_img:
            img_ids_raw: np.ndarray = f_img["ids"][:]
            img_src_idx: np.ndarray = f_img["source_row_index"][:].astype(np.int32)

        with h5py.File(latent_aug_h5_path, "r") as f_lat:
            lat_ids_raw: np.ndarray = f_lat["ids"][:]
            lat_src_idx: np.ndarray = f_lat["source_row_index"][:].astype(np.int32)

        img_ids = [id_.decode() if isinstance(id_, bytes) else str(id_) for id_ in img_ids_raw]
        lat_ids = [id_.decode() if isinstance(id_, bytes) else str(id_) for id_ in lat_ids_raw]

        # ----------------------------------------------------------
        # PREMISE-FALSE gate: verify row-index alignment.
        # ----------------------------------------------------------
        if len(img_ids) != len(lat_ids):
            raise SegDerivationError(
                f"PREMISE-FALSE: cohort={cohort_entry.name!r} — "
                f"aug image H5 has {len(img_ids)} rows but aug latent H5 has "
                f"{len(lat_ids)} rows; expected equal lengths."
            )
        mismatch_ids = [
            (i, img_ids[i], lat_ids[i]) for i in range(len(img_ids)) if img_ids[i] != lat_ids[i]
        ]
        if mismatch_ids:
            i0, a, b = mismatch_ids[0]
            raise SegDerivationError(
                f"PREMISE-FALSE: cohort={cohort_entry.name!r} — "
                f"ids mismatch at row {i0}: image={a!r} latent={b!r}. "
                f"First 3 mismatches: {mismatch_ids[:3]}"
            )
        if not np.array_equal(img_src_idx, lat_src_idx):
            diff_rows = np.where(img_src_idx != lat_src_idx)[0].tolist()
            raise SegDerivationError(
                f"PREMISE-FALSE: cohort={cohort_entry.name!r} — "
                f"source_row_index mismatch at rows {diff_rows[:5]} "
                f"(showing first 5 of {len(diff_rows)})."
            )

        n_scans = len(lat_ids)
        logger.info(
            "cohort=%s  n_scans=%d  alignment=OK",
            cohort_entry.name,
            n_scans,
        )

        # ----------------------------------------------------------
        # Pre-allocate output array (all-in-memory).
        # ----------------------------------------------------------
        lat_h, lat_w, lat_d = cfg.derivation.latent_grid
        masks_out = np.zeros((n_scans, 2, lat_h, lat_w, lat_d), dtype=np.float32)

        # ----------------------------------------------------------
        # Derive per row.
        # ----------------------------------------------------------
        for i in range(n_scans):
            mask_tensor = self._derive_one(
                image_aug_h5_path=image_aug_h5_path,
                row=i,
                scan_id=lat_ids[i],
            )
            masks_out[i] = mask_tensor.numpy()
            if (i + 1) % 50 == 0:
                logger.debug(
                    "derived %d/%d rows for cohort=%s",
                    i + 1,
                    n_scans,
                    cohort_entry.name,
                )

        # ----------------------------------------------------------
        # Write (idempotent: remove existing group first).
        # ----------------------------------------------------------
        produced_at = now_iso_utc()
        tumor_region = cfg.targets.tumor_region
        with h5py.File(latent_aug_h5_path, "r+") as f_lat:
            if SOFT_MASK_GROUP in f_lat:
                del f_lat[SOFT_MASK_GROUP]
                logger.debug(
                    "replaced existing group %s in %s",
                    SOFT_MASK_GROUP,
                    latent_aug_h5_path,
                )

            dset = f_lat.create_dataset(
                SOFT_MASK_GROUP,
                data=masks_out,
                dtype="float32",
                chunks=(1, 2, lat_h, lat_w, lat_d),
                compression="gzip",
                compression_opts=4,
            )
            # Self-describing attrs (h5-design-principles.md principle 4).
            dset.attrs["units"] = "dimensionless"
            dset.attrs["description"] = (
                f"Soft [{tumor_region.upper()}, NETC] tumour probability map in MAISI latent space; "
                f"source='gt'; channel 0 = {tumor_region.upper()}, channel 1 = NETC; "
                "SDT→sigmoid at image res (pre-cropped to crop-box), avg-pooled 4× to (2, 48, 56, 48)."
            )
            dset.attrs["dtype"] = "float32"
            dset.attrs["leading_dim"] = "n_scans"
            dset.attrs["tumor_region"] = tumor_region

            # Bump schema version and stamp provenance.
            f_lat.attrs["schema_version"] = AUG_LATENT_SCHEMA_VERSION_SOFT
            f_lat.attrs["mask_source"] = "gt"
            f_lat.attrs["mask_derive_aug_produced_at"] = produced_at
            f_lat.attrs["mask_derive_aug_git_sha"] = git_sha

        # ----------------------------------------------------------
        # Validate before returning.
        # ----------------------------------------------------------
        assert_aug_latent_soft_mask_group_valid(latent_aug_h5_path)
        return n_scans

    def _derive_one(
        self,
        *,
        image_aug_h5_path: Path,
        row: int,
        scan_id: str,
    ) -> torch.Tensor:
        """Derive the soft mask for one aug row.

        Reads ``masks/tumor`` at position ``row`` from the aug image H5.
        The label is already at the LATENT_CROP_BOX ``(192, 224, 192)``
        (aug bank is pre-cropped), so the :class:`~vena.common.CropPadSpec`
        is constructed with ``native_shape == target_shape`` — making the
        crop/pad in :func:`~vena.segmentation.derivation.derive.derive_latent_soft_mask`
        a no-op.  This produces bit-exact results matching the clean-cache
        derivation for v1/v2/v3 rows (intensity-only variants).

        Parameters
        ----------
        image_aug_h5_path:
            Path to the aug image H5.
        row:
            Row index within the aug image H5.
        scan_id:
            Scan identifier (used only for logging).

        Returns
        -------
        torch.Tensor
            Shape ``(2, 48, 56, 48)`` float32 soft probability map.
        """
        import h5py

        from vena.common import CropPadSpec

        cfg = self._cfg

        with h5py.File(image_aug_h5_path, "r") as f_img:
            label: np.ndarray = f_img["masks/tumor"][row].astype(np.int32)

        # Aug bank labels are pre-cropped to LATENT_CROP_BOX (192, 224, 192).
        # Setting native_shape == target_shape makes apply_crop_pad a no-op,
        # producing the same result as crop_spec=None while routing through
        # the identical code path used by the clean mask_derive engine.
        assert label.shape == LATENT_CROP_BOX, (
            f"Unexpected aug label shape {label.shape}; expected {LATENT_CROP_BOX}. "
            "Premise: aug labels are pre-cropped to the LATENT_CROP_BOX."
        )
        crop_spec = CropPadSpec(
            crop_origin=(0, 0, 0),
            native_shape=LATENT_CROP_BOX,
            target_shape=LATENT_CROP_BOX,
        )

        mask = derive_latent_soft_mask(
            source="gt",
            label=label,
            crop_spec=crop_spec,
            cfg=cfg.derivation,
            target_cfg=cfg.targets,
        )

        logger.debug(
            "aug row=%d scan=%s  mask_shape=%s  TC_mean=%.3f  NETC_mean=%.3f",
            row,
            scan_id,
            tuple(mask.shape),
            float(mask[0].mean()),
            float(mask[1].mean()),
        )
        return mask
