"""Image-domain H5 → MAISI latent-domain H5 converter (cohort-agnostic).

This converter is the library-level engine consumed by
``routines/encode/maisi``. It does *one* thing: read every modality row
declared in the config from the source image H5, push it through
:class:`MaisiEncoder` using the brain-centred crop box stored in the source,
downsample the tumour mask once per patient, and stack everything into the
latent H5 alongside metadata (when present) and splits/CSR copied verbatim
from the source.

What this module does NOT do:

* It does not generate QC figures. The roundtrip-fidelity and PCA plots
  live in the routine layer (``routines/encode/maisi/figures.py``).
* It does not own the device or the model lifetime. The caller passes a
  prepared :class:`MaisiEncoder` + :class:`AbstractMaskDownsampler`, so
  the same models can be reused for downstream QC without reloading.
* It does not enforce a specific modality list. The set of latents
  written is determined entirely by ``config.modalities``.

Cohort-agnostic notes:

* Metadata is optional: cohorts without ``metadata/*`` datasets (e.g.
  BraTS-GLI) produce no metadata entries in the latent H5. The source
  image H5's ``manifest_json`` attr is used to detect which metadata
  datasets are available.
* CSR (``patients/offsets``, ``patients/keys``) and ``splits/*`` are
  copied verbatim ONLY when encoding the full cohort (``src_indices ==
  list(range(n_all))``). Subset runs log a warning and skip CSR/splits
  because the patient grouping would be inconsistent.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from vena.data.h5.shared import (
    H5ConvertError,
    H5Manifest,
    H5Writer,
    assert_h5_valid,
    assign_row,
    now_iso_utc,
    resolve_git_sha,
    sha256_file,
)
from vena.model.autoencoder.maisi.encode import AbstractMaskDownsampler, MaisiEncoder
from vena.model.autoencoder.maisi.preprocessing import CropPadSpec, apply_crop_pad

from .manifest import (
    LATENT_CHANNELS,
    LATENT_SCHEMA_VERSION,
    LATENT_SEQUENCE_MAP,
    LATENT_SPATIAL,
    build_latent_manifest,
)

logger = logging.getLogger(__name__)

_PRODUCER_VERSION = "0.2.0"
_PRODUCER = f"vena.data.h5.latent_domain.convert:{_PRODUCER_VERSION}"

# UCSF-PDGM fallback metadata fields — used when the source H5 lacks
# manifest_json and the cohort attr suggests UCSF-PDGM origin.
_UCSF_PDGM_METADATA_FIELDS_FALLBACK: list[dict[str, str]] = [
    {"path": "metadata/sex", "dtype": "vlen-str", "units": "dimensionless",
     "description": "Biological sex (M/F)."},
    {"path": "metadata/age", "dtype": "float32", "units": "years",
     "description": "Age at MRI acquisition."},
    {"path": "metadata/who_grade", "dtype": "int8", "units": "WHO_grade",
     "description": "WHO CNS tumour grade (1-4); -1 if unknown."},
    {"path": "metadata/diagnosis", "dtype": "vlen-str", "units": "dimensionless",
     "description": "Final pathologic diagnosis (WHO 2021)."},
    {"path": "metadata/mgmt_status", "dtype": "vlen-str", "units": "dimensionless",
     "description": "MGMT methylation status."},
    {"path": "metadata/mgmt_index", "dtype": "vlen-str", "units": "dimensionless",
     "description": "MGMT methylation index as reported (string)."},
    {"path": "metadata/codel_1p19q", "dtype": "vlen-str", "units": "dimensionless",
     "description": "1p/19q codeletion status."},
    {"path": "metadata/idh", "dtype": "vlen-str", "units": "dimensionless",
     "description": "IDH mutation status."},
    {"path": "metadata/dead", "dtype": "int8", "units": "boolean",
     "description": "Vital status at last follow-up (1=dead, 0=alive); -1 if unknown."},
    {"path": "metadata/os_days", "dtype": "float32", "units": "days",
     "description": "Overall survival in days; NaN if unknown."},
    {"path": "metadata/eor", "dtype": "vlen-str", "units": "dimensionless",
     "description": "Extent of resection."},
    {"path": "metadata/biopsy_prior_imaging", "dtype": "vlen-str", "units": "dimensionless",
     "description": "Whether biopsy preceded MRI acquisition (Yes/No)."},
    {"path": "metadata/brats21_id", "dtype": "vlen-str", "units": "dimensionless",
     "description": "Corresponding BraTS-2021 case ID; empty if not present."},
    {"path": "metadata/brats21_seg_cohort", "dtype": "vlen-str", "units": "dimensionless",
     "description": "BraTS-2021 segmentation cohort assignment."},
    {"path": "metadata/brats21_mgmt_cohort", "dtype": "vlen-str", "units": "dimensionless",
     "description": "BraTS-2021 MGMT cohort assignment."},
]


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------


class LatentH5Config(BaseModel):
    """Resolved configuration for one execution of the latent converter."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    source_image_h5: Path
    output_path: Path
    autoencoder_checkpoint: Path

    modalities: list[str] = Field(default_factory=lambda: ["t1pre", "t1c", "t2", "flair"])
    inference_mode: str = "auto"  # auto|full|sliding
    overwrite: bool = False
    resume: bool = Field(
        default=True,
        description=(
            "When the output H5 already exists and ``overwrite`` is False, "
            "attempt to resume from ``progress/completed``. Provenance "
            "(source SHA, autoencoder SHA, modalities, encoder/downsampler "
            "attrs, ids order) must match the existing file."
        ),
    )
    checkpoint_every: int = Field(
        default=50,
        ge=1,
        description=(
            "Flush the output H5 after every ``checkpoint_every`` encoded "
            "patients. Higher values reduce I/O overhead; lower values "
            "minimise the worst-case loss on crash/SIGKILL."
        ),
    )
    limit: int | None = Field(
        default=None,
        description="Optional: encode only the first ``limit`` patients (smoke runs).",
    )
    patient_ids: list[str] | None = Field(
        default=None,
        description=(
            "Optional explicit ID list. When set, encode exactly these patients "
            "(in given order); takes precedence over ``limit``. IDs must exist "
            "in the source image H5 ``ids`` dataset."
        ),
    )

    def to_json(self) -> str:
        # Paths serialise as strings via Pydantic v2 default.
        return self.model_dump_json()


# ----------------------------------------------------------------------------
# Converter
# ----------------------------------------------------------------------------


class LatentH5Converter:
    """Run one end-to-end conversion of the image H5 to a latent H5."""

    def __init__(
        self,
        cfg: LatentH5Config,
        encoder: MaisiEncoder,
        mask_downsampler: AbstractMaskDownsampler,
    ) -> None:
        self.cfg = cfg
        self.encoder = encoder
        self.mask_downsampler = mask_downsampler

    # ------------------------------------------------------------------ public

    def run(self) -> Path:
        cfg = self.cfg
        if not cfg.source_image_h5.is_file():
            raise H5ConvertError(f"source image H5 not found: {cfg.source_image_h5}")

        with h5py.File(cfg.source_image_h5, "r") as src:
            cohort = str(src.attrs.get("cohort", "unknown"))
            metadata_fields = self._detect_metadata_fields(src)

        manifest = build_latent_manifest(
            modalities=cfg.modalities,
            mask_output_channels=self.mask_downsampler.output_channels,
            cohort=cohort,
            metadata_fields=metadata_fields,
        )

        if cfg.output_path.exists() and not cfg.overwrite:
            if not cfg.resume:
                raise FileExistsError(
                    f"Output H5 already exists: {cfg.output_path}. "
                    "Pass overwrite=True or resume=True."
                )
            return self._run_resume(manifest)
        return self._run_fresh(manifest)

    # ----- fresh run -------------------------------------------------------

    def _run_fresh(self, manifest: Any) -> Path:
        cfg = self.cfg
        with h5py.File(cfg.source_image_h5, "r") as src:
            self._assert_source_compatibility(src)
            all_ids = self._read_ids(src)
            n_all = len(all_ids)
            ids, src_indices = self._select_rows(cfg, all_ids)
            n = len(ids)
            is_full_cohort = src_indices == list(range(n_all))

            # Read crop box from source root attr once.
            box = tuple(json.loads(str(src.attrs["crop_box"])))

            logger.info(
                "%s latent-domain conversion (fresh): n_scans=%d (source has %d)",
                src.attrs.get("cohort", "unknown"),
                n,
                n_all,
            )

            timestamp = now_iso_utc()
            git_sha = resolve_git_sha()
            handle = self.encoder.handle

            # Collect extra root attrs to copy from source.
            extra_root_attrs = self._collect_extra_root_attrs(src)

            with H5Writer(
                cfg.output_path,
                manifest=manifest,
                config_json=cfg.to_json(),
                producer=_PRODUCER,
                created_at=timestamp,
                git_sha=git_sha,
                overwrite=cfg.overwrite,
                extra_root_attrs=extra_root_attrs,
            ) as w:
                self._stamp_root_attrs(w, src, manifest, handle)
                ids_dset = w.create_1d(manifest.get("ids"), n=n)
                ids_dset[:] = np.asarray(ids, dtype=object)

                latent_dsets = self._allocate_latents(w, manifest, n)
                tumor_dset = self._allocate_tumor_mask(w, manifest, n)
                completed_dset = self._allocate_progress(w, n)

                # Copy non-encoded payload upfront so a partial file is
                # structurally complete except for pending encoded rows.
                self._copy_metadata(src, w, manifest, n, src_indices)
                if is_full_cohort:
                    self._copy_csr(src, w)
                    self._copy_splits(src, w)
                else:
                    logger.warning(
                        "Subset run (n=%d < n_all=%d): skipping CSR (patients/*) and "
                        "splits copy — the subset patient grouping would be inconsistent.",
                        n,
                        n_all,
                    )
                    # Write empty placeholder CSR datasets so the manifest
                    # validator finds the declared paths (structural completeness).
                    self._write_empty_csr(w)
                self._create_priors_placeholder(w)
                w.file.flush()

                self._encode_loop(
                    src=src,
                    n=n,
                    ids=ids,
                    src_indices=src_indices,
                    out_rows=list(range(n)),
                    latent_dsets=latent_dsets,
                    tumor_dset=tumor_dset,
                    completed_dset=completed_dset,
                    h5_file=w.file,
                    checkpoint_every=cfg.checkpoint_every,
                    prior_modes={"full": 0, "sliding": 0},
                    box=box,
                )

                if not bool(np.all(completed_dset[:])):
                    raise H5ConvertError("encode loop returned but some rows are still pending")

        try:
            assert_h5_valid(cfg.output_path, manifest)
        except Exception:
            # Fresh run that fails validation is unsalvageable; remove it so
            # a future resume does not pick up a structurally broken file.
            cfg.output_path.unlink(missing_ok=True)
            raise

        logger.info("Wrote latent H5 cache: %s", cfg.output_path)
        return cfg.output_path

    # ----- resume run ------------------------------------------------------

    def _run_resume(self, manifest: Any) -> Path:
        cfg = self.cfg
        with h5py.File(cfg.source_image_h5, "r") as src:
            self._assert_source_compatibility(src)
            all_ids = self._read_ids(src)
            ids, src_indices = self._select_rows(cfg, all_ids)
            n = len(ids)
            handle = self.encoder.handle
            box = tuple(json.loads(str(src.attrs["crop_box"])))

            with h5py.File(cfg.output_path, "r+") as f:
                self._assert_resume_compatible(f, src, ids, handle)
                completed_dset = f["progress/completed"]
                done_mask = np.asarray(completed_dset[:], dtype=bool)
                pending_out_rows = [i for i, d in enumerate(done_mask) if not d]
                n_done = int(done_mask.sum())
                logger.info(
                    "%s latent-domain conversion (resume): n_scans=%d done=%d pending=%d",
                    src.attrs.get("cohort", "unknown"),
                    n,
                    n_done,
                    len(pending_out_rows),
                )

                if not pending_out_rows:
                    logger.info("resume: nothing to do (all %d rows already encoded)", n)
                else:
                    latent_dsets = {slug: f[f"latents/{slug}"] for slug in cfg.modalities}
                    tumor_dset = f["masks/tumor_latent"]
                    prior_modes = self._load_inference_modes(f)
                    f.attrs["resumed_at"] = now_iso_utc()
                    f.attrs["resume_git_sha"] = resolve_git_sha() or "unknown"

                    self._encode_loop(
                        src=src,
                        n=n,
                        ids=ids,
                        src_indices=src_indices,
                        out_rows=pending_out_rows,
                        latent_dsets=latent_dsets,
                        tumor_dset=tumor_dset,
                        completed_dset=completed_dset,
                        h5_file=f,
                        checkpoint_every=cfg.checkpoint_every,
                        prior_modes=prior_modes,
                        box=box,
                    )

                    if not bool(np.all(completed_dset[:])):
                        raise H5ConvertError("resume loop returned but some rows are still pending")

        # Validation runs on the now-complete file. If it fails, we leave the
        # file on disk (unlike the fresh path) so the user can inspect it.
        assert_h5_valid(cfg.output_path, manifest)
        logger.info("Resumed and completed latent H5 cache: %s", cfg.output_path)
        return cfg.output_path

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _detect_metadata_fields(src: h5py.File) -> list[dict[str, str]]:
        """Return the per-cohort metadata field specs for this source H5.

        Reads the source's ``manifest_json`` root attr (when present) to
        detect which ``metadata/*`` datasets are declared. Falls back to
        the UCSF-PDGM default list when the manifest is absent. Returns
        ``[]`` when the source declares no metadata datasets (e.g. BraTS-GLI).
        """
        manifest_json = src.attrs.get("manifest_json")
        if manifest_json is None:
            # Older schema: assume UCSF-PDGM metadata.
            return list(_UCSF_PDGM_METADATA_FIELDS_FALLBACK)

        try:
            src_manifest = H5Manifest.from_json(str(manifest_json))
        except Exception as exc:
            logger.warning(
                "Could not parse source manifest_json; falling back to UCSF-PDGM metadata: %s",
                exc,
            )
            return list(_UCSF_PDGM_METADATA_FIELDS_FALLBACK)

        # Collect the metadata dataset specs from the source manifest and map
        # them to our field-dict format (path, dtype, units, description).
        fields: list[dict[str, str]] = []
        for spec in src_manifest.datasets:
            if spec.kind == "metadata" and spec.path.startswith("metadata/"):
                fields.append(
                    {
                        "path": spec.path,
                        "dtype": spec.dtype,
                        "units": spec.units,
                        "description": spec.description,
                    }
                )
        return fields

    def _assert_source_compatibility(self, src: h5py.File) -> None:
        cfg = self.cfg
        if "schema_version" not in src.attrs:
            raise H5ConvertError("source image H5 lacks schema_version root attr")
        if "crop_box" not in src.attrs:
            raise H5ConvertError(
                "source image H5 lacks crop_box root attr; "
                "schema v2.0.0 is required for the box encoding path"
            )
        if "crop/origin" not in src:
            raise H5ConvertError("source image H5 lacks crop/origin dataset")
        for slug in cfg.modalities:
            path = f"images/{slug}"
            if path not in src:
                raise H5ConvertError(
                    f"source image H5 has no dataset {path!r} "
                    f"(modalities requested: {cfg.modalities})"
                )
        if "masks/tumor" not in src:
            raise H5ConvertError("source image H5 has no masks/tumor")

    def _read_ids(self, src: h5py.File) -> list[str]:
        raw = src["ids"][:]
        return [v.decode() if isinstance(v, (bytes, bytearray)) else str(v) for v in raw]

    def _select_rows(
        self,
        cfg: LatentH5Config,
        all_ids: list[str],
    ) -> tuple[list[str], list[int]]:
        """Resolve the (ids, source-indices) pair for this run."""
        if cfg.patient_ids is not None:
            index_of = {pid: i for i, pid in enumerate(all_ids)}
            missing = [p for p in cfg.patient_ids if p not in index_of]
            if missing:
                raise H5ConvertError(f"patient_ids not found in source H5: {missing}")
            src_idx = [index_of[p] for p in cfg.patient_ids]
            return list(cfg.patient_ids), src_idx
        if cfg.limit is not None:
            n = min(cfg.limit, len(all_ids))
            return all_ids[:n], list(range(n))
        return list(all_ids), list(range(len(all_ids)))

    @staticmethod
    def _collect_extra_root_attrs(src: h5py.File) -> dict[str, Any]:
        """Copy the schema-v2 semantic attrs from source into the latent H5."""
        keys = ("split_role", "longitudinal", "label_system", "crop_box", "orientation")
        out: dict[str, Any] = {}
        for k in keys:
            if k in src.attrs:
                out[k] = src.attrs[k]
        return out

    def _stamp_root_attrs(
        self,
        w: H5Writer,
        src: h5py.File,
        manifest: Any,
        handle: Any,
    ) -> None:
        cfg = self.cfg
        f = w.file
        f.attrs["source_image_h5_path"] = str(cfg.source_image_h5)
        f.attrs["source_image_h5_sha256"] = sha256_file(cfg.source_image_h5)
        f.attrs["source_image_h5_schema_version"] = src.attrs.get("schema_version", "unknown")
        f.attrs["autoencoder_checkpoint_path"] = str(cfg.autoencoder_checkpoint)
        f.attrs["autoencoder_checkpoint_sha256"] = handle.checkpoint_sha256
        f.attrs["autoencoder_arch_config_json"] = json.dumps(handle.arch_kwargs, sort_keys=True)
        f.attrs["modalities_encoded_json"] = json.dumps(list(cfg.modalities))
        f.attrs["modality_codes_json"] = json.dumps(
            {slug: LATENT_SEQUENCE_MAP[slug] for slug in cfg.modalities}
        )
        f.attrs["latent_channels"] = LATENT_CHANNELS
        f.attrs["latent_spatial_json"] = json.dumps(list(LATENT_SPATIAL))
        f.attrs["crop_box_json"] = str(src.attrs.get("crop_box", "null"))
        f.attrs["spatial_compression"] = 4
        f.attrs["padding_strategy"] = "brain_centred_crop_box"
        f.attrs["encode_runtime_attrs_json"] = json.dumps(self.encoder.to_attrs())
        f.attrs["mask_downsampler_attrs_json"] = json.dumps(self.mask_downsampler.to_attrs())
        f.attrs["scale_factor"] = "null"  # reserved; set by FM training when chosen.

    def _allocate_latents(
        self,
        w: H5Writer,
        manifest: Any,
        n: int,
    ) -> dict[str, h5py.Dataset]:
        spatial = (LATENT_CHANNELS, *LATENT_SPATIAL)
        out: dict[str, h5py.Dataset] = {}
        for slug in self.cfg.modalities:
            spec = manifest.get(f"latents/{slug}")
            out[slug] = w.create_stacked(spec, n=n, spatial_shape=spatial)
            ds = out[slug]
            ds.attrs["latent_shape"] = list(spatial)
            ds.attrs["maisi_modality_slug"] = LATENT_SEQUENCE_MAP[slug]
        return out

    def _allocate_progress(self, w: H5Writer, n: int) -> h5py.Dataset:
        """Allocate the ``progress/completed`` sidecar dataset."""
        f = w.file
        if "progress" not in f:
            f.create_group("progress")
        dset = f.create_dataset(
            "progress/completed",
            shape=(n,),
            dtype=np.bool_,
            chunks=(min(n, 64),),
        )
        dset[:] = False
        dset.attrs["units"] = "dimensionless"
        dset.attrs["description"] = (
            "Per-row completion flag for the encode loop. True iff the "
            "row's latents and tumour mask are fully written."
        )
        dset.attrs["dtype"] = "bool"
        dset.attrs["leading_dim"] = "n_scans"
        f["progress"].attrs["description"] = (
            "Internal checkpoint metadata (not part of the manifest)."
        )
        return dset

    def _assert_resume_compatible(
        self,
        f: h5py.File,
        src: h5py.File,
        ids: list[str],
        handle: Any,
    ) -> None:
        """Validate that the existing output H5 is compatible with the current config."""
        cfg = self.cfg
        violations: list[str] = []

        def _check(attr: str, expected: Any) -> None:
            got = f.attrs.get(attr)
            if got is None:
                violations.append(f"missing root attr {attr!r}")
                return
            if str(got) != str(expected):
                violations.append(f"{attr}: existing={got!r} new={expected!r}")

        _check("source_image_h5_sha256", sha256_file(cfg.source_image_h5))
        _check("autoencoder_checkpoint_sha256", handle.checkpoint_sha256)
        _check("modalities_encoded_json", json.dumps(list(cfg.modalities)))
        _check("encode_runtime_attrs_json", json.dumps(self.encoder.to_attrs()))
        _check("mask_downsampler_attrs_json", json.dumps(self.mask_downsampler.to_attrs()))

        if "progress/completed" not in f:
            violations.append(
                "existing file lacks progress/completed; cannot resume "
                "(was it produced by an older converter? re-run with overwrite=True)"
            )

        if "ids" not in f:
            violations.append("existing file lacks ids dataset")
        else:
            existing_ids = [
                v.decode() if isinstance(v, (bytes, bytearray)) else str(v) for v in f["ids"][:]
            ]
            if existing_ids != ids:
                first_diff = next(
                    (i for i, (a, b) in enumerate(zip(existing_ids, ids)) if a != b),
                    min(len(existing_ids), len(ids)),
                )
                violations.append(
                    f"ids dataset disagrees with the resolved patient selection "
                    f"(existing n={len(existing_ids)}, new n={len(ids)}, "
                    f"first divergence at row {first_diff})"
                )

        if violations:
            joined = "\n  - ".join(violations)
            raise H5ConvertError(
                "Cannot resume — existing output H5 is incompatible with "
                f"the current config:\n  - {joined}"
            )

    @staticmethod
    def _load_inference_modes(f: h5py.File) -> dict[str, int]:
        raw = f.attrs.get("inference_mode_counts_json")
        if raw is None:
            return {"full": 0, "sliding": 0}
        try:
            parsed = json.loads(str(raw))
        except json.JSONDecodeError:
            return {"full": 0, "sliding": 0}
        return {
            "full": int(parsed.get("full", 0)),
            "sliding": int(parsed.get("sliding", 0)),
        }

    def _allocate_tumor_mask(self, w: H5Writer, manifest: Any, n: int) -> h5py.Dataset:
        ds_shape = (self.mask_downsampler.output_channels, *LATENT_SPATIAL)
        spec = manifest.get("masks/tumor_latent")
        dset = w.create_stacked(spec, n=n, spatial_shape=ds_shape)
        for k, v in self.mask_downsampler.to_attrs().items():
            if isinstance(v, (str, int, float, bool)):
                dset.attrs[k] = v
            else:
                dset.attrs[k] = json.dumps(v)
        return dset

    def _encode_loop(
        self,
        *,
        src: h5py.File,
        n: int,
        ids: list[str],
        src_indices: list[int],
        out_rows: list[int],
        latent_dsets: dict[str, h5py.Dataset],
        tumor_dset: h5py.Dataset,
        completed_dset: h5py.Dataset,
        h5_file: h5py.File,
        checkpoint_every: int,
        prior_modes: dict[str, int],
        box: tuple[int, ...],
    ) -> None:
        cfg = self.cfg
        device = self.encoder.handle.device
        modes_seen = dict(prior_modes)
        t0 = time.monotonic()
        n_pending = len(out_rows)
        log_every = max(1, n_pending // 50)

        # Native shape is the same for all scans in a given cohort.
        # Read from the first image dataset.
        first_slug = cfg.modalities[0]
        native_shape: tuple[int, int, int] = tuple(src[f"images/{first_slug}"].shape[1:])  # type: ignore[assignment]

        # BraTS-2023 cohorts (e.g. BraTS-GLI) encode the enhancing-tumour class as
        # label 3 instead of UCSF/BraTS-2021's label 4. The per-class mask
        # downsampler is configured for the {0,1,2,4} set; remap 3→4 on the fly
        # so the latent tumour-mask channels (NETC, ED, ET) stay consistent
        # across cohorts. Same semantic class, just a renumbering.
        remap_brats2023_et = str(src.attrs.get("label_system", "")) == "BraTS2023"

        for k, out_row in enumerate(out_rows):
            src_row = src_indices[out_row]
            pid = ids[out_row]

            # Build the per-scan crop spec from the stored crop/origin.
            crop_origin: tuple[int, int, int] = tuple(  # type: ignore[assignment]
                int(v) for v in src["crop/origin"][src_row]
            )
            spec = CropPadSpec(
                crop_origin=crop_origin,
                native_shape=native_shape,
                target_shape=tuple(box),  # type: ignore[arg-type]
            )

            # ---- latents ---------------------------------------------------
            for slug in cfg.modalities:
                arr = src[f"images/{slug}"][src_row]  # (H, W, D), float32
                t = torch.from_numpy(np.asarray(arr, dtype=np.float32))
                t = t.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, D)
                t = t.to(device, non_blocking=True)
                res = self.encoder.encode(t, mode=cfg.inference_mode, crop_spec=spec)
                modes_seen[res.inference_mode] = modes_seen.get(res.inference_mode, 0) + 1
                z = res.latent.detach().to("cpu", dtype=torch.float32).contiguous()
                z_np = z[0].numpy()  # (C, h, w, d)
                # Assert the latent has the expected spatial shape.
                if tuple(z_np.shape) != (LATENT_CHANNELS, *LATENT_SPATIAL):
                    raise H5ConvertError(
                        f"unexpected latent shape for {pid}/{slug}: "
                        f"got {z_np.shape}, expected "
                        f"({LATENT_CHANNELS}, {LATENT_SPATIAL})"
                    )
                assign_row(latent_dsets[slug], out_row, z_np)
                del t, z, res

            # ---- tumor mask -----------------------------------------------
            # Crop the native seg to the box before downsampling so that the
            # downsampler sees (192, 224, 192) → avg_pool4 → (48, 56, 48).
            seg = src["masks/tumor"][src_row]  # (H, W, D), int8
            if remap_brats2023_et:
                seg = np.where(seg == 3, 4, seg).astype(seg.dtype, copy=False)
            seg_t = torch.from_numpy(np.asarray(seg, dtype=np.int64))
            seg_t = seg_t.unsqueeze(0).unsqueeze(0).to(device)
            # apply_crop_pad requires float; cast, crop, cast back.
            seg_float = apply_crop_pad(seg_t.float(), spec)
            seg_cropped = seg_float.to(torch.int64)
            mask_latent = self.mask_downsampler.downsample(
                seg_cropped, target_shape=LATENT_SPATIAL
            )
            mask_np = mask_latent[0].detach().to("cpu").numpy()
            assign_row(tumor_dset, out_row, mask_np)
            del seg_t, mask_latent

            if device.type == "cuda":
                torch.cuda.empty_cache()

            # Order matters: mark this row complete only after BOTH latents
            # and the tumour mask have been written.
            completed_dset[out_row] = True

            done_now = k + 1
            if done_now % checkpoint_every == 0 or done_now == n_pending:
                h5_file.attrs["inference_mode_counts_json"] = json.dumps(modes_seen)
                h5_file.flush()
                logger.info(
                    "checkpoint: flushed at %d/%d pending rows (cumulative %d/%d)",
                    done_now,
                    n_pending,
                    int(np.asarray(completed_dset[:]).sum()),
                    n,
                )

            if done_now % log_every == 0 or done_now == n_pending:
                elapsed = time.monotonic() - t0
                rate = done_now / elapsed if elapsed > 0 else 0.0
                eta = (n_pending - done_now) / rate if rate > 0 else float("inf")
                logger.info(
                    "encode %s [%d/%d pending] (%.1f%%) rate=%.2f scans/s eta=%.0fs modes=%s",
                    pid,
                    done_now,
                    n_pending,
                    100.0 * done_now / n_pending,
                    rate,
                    eta,
                    modes_seen,
                )

        h5_file.attrs["inference_mode_counts_json"] = json.dumps(modes_seen)

    # ----- metadata + CSR + splits + priors --------------------------------

    def _copy_metadata(
        self,
        src: h5py.File,
        w: H5Writer,
        manifest: Any,
        n: int,
        src_indices: list[int],
    ) -> None:
        idx = np.asarray(src_indices, dtype=np.int64)
        for spec in manifest.datasets:
            if spec.kind != "metadata":
                continue
            # Skip CSR datasets (patients/*) — those are handled by _copy_csr.
            if spec.path.startswith("patients/"):
                continue
            if spec.path not in src:
                logger.warning("source missing metadata %s; writing default fill", spec.path)
                values = self._default_metadata_values(spec.dtype, n)
            else:
                full = src[spec.path][:]
                raw = np.asarray(full)[idx]
                values = self._coerce_metadata(raw, spec.dtype)
            dset = w.create_1d(spec, n=n)
            dset[:] = values

    @staticmethod
    def _coerce_metadata(raw: NDArray[Any], dtype: str) -> NDArray[Any]:
        if dtype == "vlen-str":
            out = np.empty(raw.shape, dtype=object)
            for i, v in enumerate(raw):
                out[i] = v.decode() if isinstance(v, (bytes, bytearray)) else str(v)
            return out
        if dtype == "int8":
            return np.asarray(raw, dtype=np.int8)
        if dtype == "float32":
            return np.asarray(raw, dtype=np.float32)
        raise ValueError(f"unhandled metadata dtype {dtype!r}")

    @staticmethod
    def _default_metadata_values(dtype: str, n: int) -> NDArray[Any]:
        if dtype == "vlen-str":
            return np.asarray([""] * n, dtype=object)
        if dtype == "int8":
            return np.full(n, -1, dtype=np.int8)
        if dtype == "float32":
            return np.full(n, np.nan, dtype=np.float32)
        raise ValueError(f"unhandled metadata dtype {dtype!r}")

    @staticmethod
    def _write_empty_csr(w: H5Writer) -> None:
        """Write zero-length placeholder CSR datasets for subset runs.

        The manifest always declares ``patients/offsets`` and ``patients/keys``
        (structural completeness). Subset runs cannot copy the real CSR because
        the patient grouping would be inconsistent, so we write empty arrays
        here so ``assert_h5_valid`` does not flag missing datasets.
        """
        w.write_int_1d(
            "patients/offsets",
            np.zeros(0, dtype=np.int32),
            dtype="int32",
            units="dimensionless",
            description=(
                "CSR offsets — empty placeholder (subset run; full cohort CSR not copied)."
            ),
        )
        w.write_vlen_str_1d(
            "patients/keys",
            [],
            description="Patient keys — empty placeholder (subset run).",
        )

    @staticmethod
    def _copy_csr(src: h5py.File, w: H5Writer) -> None:
        """Copy ``patients/offsets`` (int32) and ``patients/keys`` (vlen-str) verbatim."""
        if "patients/offsets" not in src or "patients/keys" not in src:
            logger.warning("source image H5 has no patients/* CSR; skipping CSR copy")
            return
        offsets = np.asarray(src["patients/offsets"][:], dtype=np.int32)
        w.write_int_1d(
            "patients/offsets",
            offsets,
            dtype="int32",
            units="dimensionless",
            description=(
                "CSR offsets, length n_patients+1; scans of patient k are "
                "rows [offsets[k]:offsets[k+1]]."
            ),
        )
        keys_raw = src["patients/keys"][:]
        keys = [v.decode() if isinstance(v, (bytes, bytearray)) else str(v) for v in keys_raw]
        w.write_vlen_str_1d(
            "patients/keys",
            keys,
            description="Unique patient keys, length n_patients, in offset order.",
        )
        logger.info("CSR copied: %d patients, %d scans", len(keys), int(offsets[-1]))

    def _copy_splits(self, src: h5py.File, w: H5Writer) -> None:
        if "splits" not in src:
            logger.warning("source image H5 has no splits group; skipping copy")
            return

        def _visit(name: str, obj: h5py.HLObject) -> None:
            if isinstance(obj, h5py.Dataset):
                raw = obj[:]
                values = [v.decode() if isinstance(v, (bytes, bytearray)) else str(v) for v in raw]
                w.write_vlen_str_1d(name, values)

        src["splits"].visititems(lambda subname, obj: _visit(f"splits/{subname}", obj))
        if "splits" in w.file and "splits" in src:
            grp_src = src["splits"]
            grp_dst = w.file["splits"]
            for attr in ("description", "n_folds"):
                if attr in grp_src.attrs:
                    grp_dst.attrs[attr] = grp_src.attrs[attr]

    def _create_priors_placeholder(self, w: H5Writer) -> None:
        if "priors" in w.file:
            return
        g = w.file.create_group("priors")
        g.attrs["description"] = (
            "Reserved placeholder for prior maps (vessel, cellularity, perfusion, "
            "susceptibility, ...). Empty in latent v2.0.0 — future routines append "
            "datasets here with their own provenance attrs."
        )
        g.attrs["schema_version"] = LATENT_SCHEMA_VERSION
