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
            logger.info(
                "UCSF-PDGM latent-domain conversion (fresh): n_patients=%d (source has %d)",
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
                completed_dset = self._allocate_progress(w, n)

                # Copy non-encoded payload (metadata, splits, priors) upfront
                # so a partial file is structurally complete except for the
                # rows ``progress/completed`` marks as pending.
                self._copy_metadata(src, w, manifest, n, src_indices)
                self._copy_splits(src, w)
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

            with h5py.File(cfg.output_path, "r+") as f:
                self._assert_resume_compatible(f, src, ids, handle)
                completed_dset = f["progress/completed"]
                done_mask = np.asarray(completed_dset[:], dtype=bool)
                pending_out_rows = [i for i, d in enumerate(done_mask) if not d]
                n_done = int(done_mask.sum())
                logger.info(
                    "UCSF-PDGM latent-domain conversion (resume): n_patients=%d done=%d pending=%d",
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
                    )

                    if not bool(np.all(completed_dset[:])):
                        raise H5ConvertError("resume loop returned but some rows are still pending")

        # Validation runs on the now-complete file. If it fails, we leave the
        # file on disk (unlike the fresh path) so the user can inspect it.
        assert_h5_valid(cfg.output_path, manifest)
        logger.info("Resumed and completed latent H5 cache: %s", cfg.output_path)
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
                raise H5ConvertError(f"patient_ids not found in source H5: {missing}")
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

    def _allocate_progress(self, w: H5Writer, n: int) -> h5py.Dataset:
        """Allocate the ``progress/completed`` sidecar dataset.

        Bool, shape ``(n,)``, initialised to False. Marked True per-row by
        :meth:`_encode_loop` after both latents and the tumour mask have
        been written for that row. Lives outside the manifest so the
        structural validator ignores it.
        """
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
            "row's latents and tumour mask are fully written. Used by "
            "the resume path to skip already-done patients."
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
        """Validate that the existing output H5 was produced under a config
        compatible with the current one. Raises :class:`H5ConvertError` with
        every violation listed."""
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
                # Show the first divergence for actionable diagnostics.
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
    ) -> None:
        cfg = self.cfg
        device = self.encoder.handle.device
        modes_seen = dict(prior_modes)
        t0 = time.monotonic()
        n_pending = len(out_rows)
        log_every = max(1, n_pending // 50)

        for k, out_row in enumerate(out_rows):
            src_row = src_indices[out_row]
            pid = ids[out_row]
            # ---- latents ---------------------------------------------------
            for slug in cfg.modalities:
                arr = src[f"images/{slug}"][src_row]  # (H, W, D), float32
                t = torch.from_numpy(np.asarray(arr, dtype=np.float32))
                t = t.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, D)
                t = t.to(device, non_blocking=True)
                res = self.encoder.encode(t, mode=cfg.inference_mode)
                modes_seen[res.inference_mode] = modes_seen.get(res.inference_mode, 0) + 1
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

            # Order matters: mark this row complete only after BOTH latents
            # and the tumour mask have been written. The flush below makes
            # the flag visible to a future resume.
            completed_dset[out_row] = True

            done_now = k + 1
            if done_now % checkpoint_every == 0 or done_now == n_pending:
                # Record running inference-mode tally so a resume sees the
                # full historical count even if it never re-loops.
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

        # Final stamp of the inference-mode counts.
        h5_file.attrs["inference_mode_counts_json"] = json.dumps(modes_seen)

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
