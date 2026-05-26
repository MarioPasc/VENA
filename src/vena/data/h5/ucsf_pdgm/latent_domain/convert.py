"""Image-domain H5 → MAISI latent-domain H5 converter.

This converter is the library-level engine consumed by
``routines/encode/maisi``. It does *one* thing: read every modality row
declared in the config from the source image H5, push it through
:class:`MaisiEncoder`, downsample the tumour mask once per patient, and
stack everything into the latent H5 alongside metadata and splits copied
verbatim from the source.

What this module does NOT do:

* It does not generate QC figures. The roundtrip-fidelity and PCA plots
  live in the routine layer (``routines/encode/maisi/figures.py``).
* It does not own the device or the model lifetime. The caller passes a
  prepared :class:`MaisiEncoder` + :class:`AbstractMaskDownsampler`, so
  the same models can be reused for downstream QC without reloading.
* It does not enforce a specific modality list. The set of latents
  written is determined entirely by ``config.modalities``.
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
    H5Writer,
    assert_h5_valid,
    assign_row,
    now_iso_utc,
    resolve_git_sha,
    sha256_file,
)
from vena.model.autoencoder.maisi.encode import AbstractMaskDownsampler, MaisiEncoder

from .manifest import (
    UCSF_PDGM_IMAGE_NATIVE_SHAPE,
    UCSF_PDGM_IMAGE_PADDED_SHAPE,
    UCSF_PDGM_LATENT_CHANNELS,
    UCSF_PDGM_LATENT_SCHEMA_VERSION,
    UCSF_PDGM_LATENT_SEQUENCE_MAP,
    UCSF_PDGM_LATENT_SPATIAL,
    build_latent_manifest,
)

logger = logging.getLogger(__name__)

_PRODUCER_VERSION = "0.1.0"
_PRODUCER = f"vena.data.h5.ucsf_pdgm.latent_domain.convert:{_PRODUCER_VERSION}"


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------


class UCSFPDGMLatentH5Config(BaseModel):
    """Resolved configuration for one execution of the latent converter."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    source_image_h5: Path
    output_path: Path
    autoencoder_checkpoint: Path

    modalities: list[str] = Field(default_factory=lambda: ["t1pre", "t1c", "t2", "flair"])
    inference_mode: str = "auto"  # auto|full|sliding
    overwrite: bool = False
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


class UCSFPDGMLatentH5Converter:
    """Run one end-to-end conversion of the image H5 to a latent H5."""

    def __init__(
        self,
        cfg: UCSFPDGMLatentH5Config,
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

        manifest = build_latent_manifest(
            modalities=cfg.modalities,
            mask_output_channels=self.mask_downsampler.output_channels,
        )

        with h5py.File(cfg.source_image_h5, "r") as src:
            self._assert_source_compatibility(src)
            all_ids = self._read_ids(src)
            n_all = len(all_ids)
            ids, src_indices = self._select_rows(cfg, all_ids)
            n = len(ids)
            logger.info(
                "UCSF-PDGM latent-domain conversion: n_patients=%d (source has %d)",
                n,
                n_all,
            )

            timestamp = now_iso_utc()
            git_sha = resolve_git_sha()
            handle = self.encoder.handle

            with H5Writer(
                cfg.output_path,
                manifest=manifest,
                config_json=cfg.to_json(),
                producer=_PRODUCER,
                created_at=timestamp,
                git_sha=git_sha,
                overwrite=cfg.overwrite,
            ) as w:
                self._stamp_root_attrs(w, src, manifest, handle)
                ids_dset = w.create_1d(manifest.get("ids"), n=n)
                ids_dset[:] = np.asarray(ids, dtype=object)

                latent_dsets = self._allocate_latents(w, manifest, n)
                tumor_dset = self._allocate_tumor_mask(w, manifest, n)

                self._encode_loop(src, n, ids, src_indices, latent_dsets, tumor_dset)

                self._copy_metadata(src, w, manifest, n, src_indices)
                self._copy_splits(src, w)
                self._create_priors_placeholder(w)

        try:
            assert_h5_valid(cfg.output_path, manifest)
        except Exception:
            cfg.output_path.unlink(missing_ok=True)
            raise

        logger.info("Wrote latent H5 cache: %s", cfg.output_path)
        return cfg.output_path

    # ------------------------------------------------------------------ helpers

    def _assert_source_compatibility(self, src: h5py.File) -> None:
        cfg = self.cfg
        # Soft-check: the image H5 schema_version may evolve; require it
        # to exist but do not pin to a single value here, because the
        # image-domain manifest is bumped independently.
        if "schema_version" not in src.attrs:
            raise H5ConvertError("source image H5 lacks schema_version root attr")
        for slug in cfg.modalities:
            path = f"images/{slug}"
            if path not in src:
                raise H5ConvertError(
                    f"source image H5 has no dataset {path!r} (modalities requested: {cfg.modalities})"
                )
        if "masks/tumor" not in src:
            raise H5ConvertError("source image H5 has no masks/tumor")

    def _read_ids(self, src: h5py.File) -> list[str]:
        raw = src["ids"][:]
        # vlen-str returns numpy object array of bytes or str depending on h5py.
        return [v.decode() if isinstance(v, (bytes, bytearray)) else str(v) for v in raw]

    def _select_rows(
        self,
        cfg: UCSFPDGMLatentH5Config,
        all_ids: list[str],
    ) -> tuple[list[str], list[int]]:
        """Resolve the (ids, source-indices) pair for this run.

        ``patient_ids`` (when set) takes precedence; otherwise ``limit``
        truncates to the first ``limit`` rows of the source H5; otherwise
        the full cohort is used.
        """
        if cfg.patient_ids is not None:
            index_of = {pid: i for i, pid in enumerate(all_ids)}
            missing = [p for p in cfg.patient_ids if p not in index_of]
            if missing:
                raise H5ConvertError(
                    f"patient_ids not found in source H5: {missing}"
                )
            src_idx = [index_of[p] for p in cfg.patient_ids]
            return list(cfg.patient_ids), src_idx
        if cfg.limit is not None:
            n = min(cfg.limit, len(all_ids))
            return all_ids[:n], list(range(n))
        return list(all_ids), list(range(len(all_ids)))

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
            {slug: UCSF_PDGM_LATENT_SEQUENCE_MAP[slug] for slug in cfg.modalities}
        )
        f.attrs["latent_channels"] = UCSF_PDGM_LATENT_CHANNELS
        f.attrs["latent_spatial_json"] = json.dumps(list(UCSF_PDGM_LATENT_SPATIAL))
        f.attrs["image_native_shape_json"] = json.dumps(list(UCSF_PDGM_IMAGE_NATIVE_SHAPE))
        f.attrs["image_padded_shape_json"] = json.dumps(list(UCSF_PDGM_IMAGE_PADDED_SHAPE))
        f.attrs["spatial_compression"] = 4
        f.attrs["padding_strategy"] = "zero_pad_depth_to_multiple_of_8"
        f.attrs["encode_runtime_attrs_json"] = json.dumps(self.encoder.to_attrs())
        f.attrs["mask_downsampler_attrs_json"] = json.dumps(self.mask_downsampler.to_attrs())
        f.attrs["scale_factor"] = "null"  # reserved; set by FM training when chosen.

    def _allocate_latents(
        self,
        w: H5Writer,
        manifest: Any,
        n: int,
    ) -> dict[str, h5py.Dataset]:
        spatial = (UCSF_PDGM_LATENT_CHANNELS, *UCSF_PDGM_LATENT_SPATIAL)
        out: dict[str, h5py.Dataset] = {}
        for slug in self.cfg.modalities:
            spec = manifest.get(f"latents/{slug}")
            out[slug] = w.create_stacked(spec, n=n, spatial_shape=spatial)
            ds = out[slug]
            ds.attrs["latent_shape"] = list(spatial)
            ds.attrs["maisi_modality_slug"] = UCSF_PDGM_LATENT_SEQUENCE_MAP[slug]
        return out

    def _allocate_tumor_mask(self, w: H5Writer, manifest: Any, n: int) -> h5py.Dataset:
        ds_shape = (self.mask_downsampler.output_channels, *UCSF_PDGM_LATENT_SPATIAL)
        spec = manifest.get("masks/tumor_latent")
        dset = w.create_stacked(spec, n=n, spatial_shape=ds_shape)
        for k, v in self.mask_downsampler.to_attrs().items():
            # h5py rejects arbitrary nested dicts/lists; serialise to JSON
            # for non-scalar values to keep things robust.
            if isinstance(v, (str, int, float, bool)):
                dset.attrs[k] = v
            else:
                dset.attrs[k] = json.dumps(v)
        return dset

    def _encode_loop(
        self,
        src: h5py.File,
        n: int,
        ids: list[str],
        src_indices: list[int],
        latent_dsets: dict[str, h5py.Dataset],
        tumor_dset: h5py.Dataset,
    ) -> None:
        cfg = self.cfg
        device = self.encoder.handle.device
        modes_seen: dict[str, int] = {"full": 0, "sliding": 0}
        t0 = time.monotonic()
        log_every = max(1, n // 50)

        for out_row, src_row in enumerate(src_indices):
            pid = ids[out_row]
            # ---- latents ---------------------------------------------------
            for slug in cfg.modalities:
                arr = src[f"images/{slug}"][src_row]  # (H, W, D), float32
                t = torch.from_numpy(np.asarray(arr, dtype=np.float32))
                t = t.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, D)
                t = t.to(device, non_blocking=True)
                res = self.encoder.encode(t, mode=cfg.inference_mode)
                modes_seen[res.inference_mode] += 1
                z = res.latent.detach().to("cpu", dtype=torch.float32).contiguous()
                z_np = z[0].numpy()  # (C, h, w, d)
                assign_row(latent_dsets[slug], out_row, z_np)
                del t, z, res

            # ---- tumor mask -----------------------------------------------
            seg = src["masks/tumor"][src_row]  # (H, W, D), int8
            seg_t = torch.from_numpy(np.asarray(seg, dtype=np.int64))
            seg_t = seg_t.unsqueeze(0).unsqueeze(0).to(device)
            mask_latent = self.mask_downsampler.downsample(
                seg_t, target_shape=UCSF_PDGM_LATENT_SPATIAL
            )
            mask_np = mask_latent[0].detach().to("cpu").numpy()
            assign_row(tumor_dset, out_row, mask_np)
            del seg_t, mask_latent

            if device.type == "cuda":
                torch.cuda.empty_cache()

            if (out_row + 1) % log_every == 0 or (out_row + 1) == n:
                elapsed = time.monotonic() - t0
                rate = (out_row + 1) / elapsed if elapsed > 0 else 0.0
                eta = (n - (out_row + 1)) / rate if rate > 0 else float("inf")
                logger.info(
                    "encode %s [%d/%d] (%.1f%%) rate=%.2f scans/s eta=%.0fs modes=%s",
                    pid,
                    out_row + 1,
                    n,
                    100.0 * (out_row + 1) / n,
                    rate,
                    eta,
                    modes_seen,
                )

        # Record the mix of inference modes used over the full run.
        tumor_dset.file.attrs["inference_mode_counts_json"] = json.dumps(modes_seen)

    # ----- metadata + splits + priors --------------------------------------

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
            if spec.path not in src:
                logger.warning("source missing metadata %s; writing default fill", spec.path)
                values = self._default_metadata_values(spec.dtype, n)
            else:
                # h5py supports fancy indexing on 1-D datasets; pull only
                # the rows we encoded so the metadata aligns with ``ids``.
                full = src[spec.path][:]
                raw = np.asarray(full)[idx]
                values = self._coerce_metadata(raw, spec.dtype)
            dset = w.create_1d(spec, n=n)
            dset[:] = values

    @staticmethod
    def _coerce_metadata(raw: NDArray[Any], dtype: str) -> NDArray[Any]:
        if dtype == "vlen-str":
            # ``raw`` is object dtype; convert bytes → str defensively.
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

    def _copy_splits(self, src: h5py.File, w: H5Writer) -> None:
        if "splits" not in src:
            logger.warning("source image H5 has no splits group; skipping copy")
            return
        # Walk every dataset under splits/ and copy as-is.
        def _visit(name: str, obj: h5py.HLObject) -> None:
            if isinstance(obj, h5py.Dataset):
                # Build the values list (vlen-str expected).
                raw = obj[:]
                values = [v.decode() if isinstance(v, (bytes, bytearray)) else str(v) for v in raw]
                w.write_vlen_str_1d(name, values)

        src["splits"].visititems(lambda subname, obj: _visit(f"splits/{subname}", obj))
        # Also stamp the group-level description / n_folds if present.
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
            "susceptibility, ...). Empty in latent v0.1.0 — future routines append "
            "datasets here with their own provenance attrs."
        )
        g.attrs["schema_version"] = UCSF_PDGM_LATENT_SCHEMA_VERSION
