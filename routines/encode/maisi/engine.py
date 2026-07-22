"""Routine engine for encoding an image H5 to the MAISI latent H5.

Three responsibilities:

1. **Load** the MAISI VAE-GAN once and build the encoder / decoder / mask
   downsampler instances.
2. **Encode** every patient by delegating to
   :class:`LatentH5Converter` (library code).
3. **QC** the result — decode one patient per WHO grade for the roundtrip
   figure; pool every latent for the PCA figure; write ``report.md`` and
   ``decision.json``.

The engine is the only place where the routine layer touches the GPU.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field

from vena.data.h5.latent_domain.convert import LatentH5Config, LatentH5Converter
from vena.data.h5.shared import now_iso_utc, resolve_git_sha
from vena.model.autoencoder.maisi import load_autoencoder
from vena.model.autoencoder.maisi.decode import MaisiDecoder
from vena.model.autoencoder.maisi.encode import MaisiEncoder, get_downsampler
from vena.model.autoencoder.maisi.preprocessing import (
    CropPadSpec,
    apply_crop_pad,
    percentile_normalise,
)

from .figures import (
    RoundtripRow,
    render_pca_figure,
    render_per_modality_roundtrip_figures,
    render_roundtrip_figure,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------


class _SlidingWindowCfg(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    roi_size: list[int] = Field(default_factory=lambda: [80, 80, 32])
    overlap: float = 0.4
    mode: str = "gaussian"


class _MaskDownsamplerCfg(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = "per_class_avg_pool"
    params: dict[str, Any] = Field(default_factory=dict)


class _RoundtripCfg(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    figure_filename: str = "roundtrip.png"  # legacy; per-modality files override
    # If null: one patient per unique WHO grade present in the cohort.
    # Otherwise: take the first N WHO-grade-distinct patients up to this cap.
    n_patients: int | None = None
    # Subset of modalities to render per row; defaults to t1c only (the
    # synthesis target) to keep the figure compact.
    modalities: list[str] = Field(default_factory=lambda: ["t1c"])
    # When True, emit one figure per modality (named
    # ``roundtrip_<modality>.png``) instead of a single multi-modality
    # composite. Recommended; the per-modality variant is easier to read.
    per_modality_figures: bool = True
    # Dump every (patient, modality) row to a pair of .nii.gz files
    # (original + reconstructed) alongside the figures. Identity affine.
    save_nifti: bool = True


class _StabilizationCfg(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    n_per_grade: int = 10
    grades: list[int] = Field(default_factory=lambda: [1, 2, 3, 4])
    seed: int = 11
    n_bootstrap: int = 5000
    ci_level: float = 0.95
    # Group the stratified picks into ``n_per_grade`` *sets*. Set k holds
    # one patient per grade (the k-th sampled from each grade). One
    # collage figure is emitted per set under
    # ``figures/stabilization/set_{kk}.png``.
    set_figures: bool = True
    # Modalities to render per row inside each set collage.
    modalities: list[str] = Field(default_factory=lambda: ["t1pre", "t1c", "t2", "flair"])


class _PcaCfg(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    figure_filename: str = "pca.png"
    pooling: str = "global_avg"
    n_components: int = 2


class EncodeMaisiRoutineConfig(BaseModel):
    """YAML-driven config for the encode routine."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_image_h5: Path
    autoencoder_checkpoint: Path
    output_dir: Path = Path("artifacts/encode/maisi")
    output_h5_path: Path | None = None

    modalities: list[str] = Field(default_factory=lambda: ["t1pre", "t1c", "t2", "flair"])
    device: str = "cuda"
    inference_mode: str = "auto"
    sliding_window: _SlidingWindowCfg = Field(default_factory=_SlidingWindowCfg)
    depth_pad_base: int = 8
    mask_downsampler: _MaskDownsamplerCfg = Field(default_factory=_MaskDownsamplerCfg)

    # Precision policy for both encoder and decoder.
    # "autocast" — wrap forward in ``torch.autocast("cuda", fp16)``; safe
    #   default that uses less VRAM and matches MAISI's intended runtime.
    # "fp32"     — bypass autocast; force ``norm_float16=False`` in the
    #   autoencoder arch. Higher numerical precision at the cost of ~2× VRAM.
    precision_mode: str = "autocast"
    autoencoder_norm_float16: bool | None = None  # None → resolved from precision_mode

    # Intensity-normalisation knobs. MAISI's VAE_Transform uses upper=99.5;
    # foreground_only=True is the correct default for skull-stripped volumes.
    percentile_lower: float = 0.0
    percentile_upper: float = 99.5
    percentile_foreground_only: bool = True
    # When True (default), pass the source ``masks/brain`` to
    # :func:`percentile_normalise` instead of the ``x > 0`` foreground
    # heuristic. Critical for z-score cohorts (BraTS-Africa); no-op for
    # raw-intensity cohorts. Forwarded to ``LatentH5Config`` and persisted
    # in ``decision.json``.
    percentile_use_brain_mask: bool = True

    limit: int | None = None
    patient_ids: list[str] | None = None
    overwrite: bool = False
    log_level: str = "INFO"
    seed: int = 42

    # Full-cohort + checkpoint/resume controls (forwarded to the converter).
    # When ``encode_full_cohort`` is True, the converter is invoked with
    # ``patient_ids=None`` and every source patient is encoded; ``patient_ids``
    # is still consulted by the QC selection (roundtrip / NIfTI dump).
    encode_full_cohort: bool = False
    checkpoint_every: int = Field(
        default=50,
        ge=1,
        description="Flush the latent H5 after every K encoded patients.",
    )
    resume_from_run_id: str | None = Field(
        default=None,
        description=(
            "Opt-in resume pointer. When null, the routine creates a fresh "
            "timestamped run dir and the converter takes the fresh path "
            "(if the latent H5 already exists and ``overwrite`` is False, "
            "the converter raises). When set to a prior run-dir name "
            "(e.g. ``2026-05-27T07-59-22Z``), the engine: (1) looks up "
            "``<output_dir>/<resume_from_run_id>/`` — must exist; "
            "(2) reads its ``config.yaml`` to recover the latent H5 path; "
            "(3) reuses that prior run dir so all artefacts (figures, "
            "tables, decision.json) accumulate in one place; (4) drives "
            "the converter onto its resume path. The latent H5 itself "
            "carries the 'which patient ids are done' state via the "
            "combination of ``ids[i]`` and ``progress/completed[i]``."
        ),
    )

    roundtrip: _RoundtripCfg = Field(default_factory=_RoundtripCfg)
    pca: _PcaCfg = Field(default_factory=_PcaCfg)
    stabilization: _StabilizationCfg = Field(default_factory=_StabilizationCfg)

    @classmethod
    def from_yaml(cls, path: Path | str) -> EncodeMaisiRoutineConfig:
        with Path(path).open("r") as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw)

    def to_json(self) -> str:
        return self.model_dump_json()


# ----------------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------------


class EncodeMaisiRoutineEngine:
    """Orchestrate encoding + QC for one routine invocation."""

    def __init__(self, cfg: EncodeMaisiRoutineConfig) -> None:
        self.cfg = cfg
        self.run_dir: Path | None = None

    def run(self) -> Path:
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        resumed_h5_override: Path | None = None
        if cfg.resume_from_run_id is not None:
            self.run_dir, resumed_h5_override = self._resolve_resume_target(cfg)
            logger.info(
                "Resuming routine from run_id=%s (run_dir=%s, latent_h5=%s)",
                cfg.resume_from_run_id,
                self.run_dir,
                resumed_h5_override,
            )
        else:
            self.run_dir = self._make_run_dir(cfg.output_dir)
        figures_dir = self.run_dir / "figures"
        tables_dir = self.run_dir / "tables"
        figures_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Routine run dir: %s", self.run_dir)
        self._persist_resolved_config(self.run_dir)

        # ---- model load ------------------------------------------------------
        device = torch.device(cfg.device)
        # Resolve fp16/fp32 policy. If the user did not explicitly set the
        # ``norm_float16`` override, derive it from ``precision_mode``.
        norm_float16 = cfg.autoencoder_norm_float16
        if norm_float16 is None:
            norm_float16 = cfg.precision_mode == "autocast"
        handle = load_autoencoder(
            cfg.autoencoder_checkpoint,
            device=device,
            arch_overrides={"norm_float16": bool(norm_float16)},
        )
        encoder = MaisiEncoder(
            handle=handle,
            sliding_roi=tuple(cfg.sliding_window.roi_size),
            sliding_overlap=cfg.sliding_window.overlap,
            sliding_mode=cfg.sliding_window.mode,
            depth_pad_base=cfg.depth_pad_base,
            percentile_lower=cfg.percentile_lower,
            percentile_upper=cfg.percentile_upper,
            percentile_foreground_only=cfg.percentile_foreground_only,
            precision_mode=cfg.precision_mode,
        )
        decoder = MaisiDecoder(handle=handle, precision_mode=cfg.precision_mode)
        mask_ds = get_downsampler(cfg.mask_downsampler.name, **cfg.mask_downsampler.params)

        # ---- resolve the patient list to encode -----------------------------
        # The encode pass is the expensive step; do it once for every
        # patient that any QC pass will need (named + stabilisation).
        named_ids = list(cfg.patient_ids) if cfg.patient_ids else []
        stab_ids_by_grade: dict[int, list[str]] = {}
        if cfg.stabilization.enabled:
            stab_ids_by_grade = self._sample_stratified_ids(cfg)
        all_stab_ids = [
            pid for grade in cfg.stabilization.grades for pid in stab_ids_by_grade.get(grade, [])
        ]
        # Union preserving order: named first, then stabilisation, dedup.
        seen: set[str] = set()
        all_ids: list[str] = []
        for pid in named_ids + all_stab_ids:
            if pid in seen:
                continue
            seen.add(pid)
            all_ids.append(pid)

        # ---- encode (delegate to library converter) -------------------------
        if resumed_h5_override is not None:
            # When resuming, the latent H5 path is dictated by the prior
            # run's persisted config — not the current YAML — so the
            # converter writes back into the same on-disk file.
            latent_h5 = resumed_h5_override
        else:
            latent_h5 = cfg.output_h5_path or (self.run_dir / "UCSFPDGM_latent.h5")
        # Full-cohort mode: ignore the named/stabilisation union and encode
        # every patient in the source H5. The QC passes still pick the named
        # + stab IDs by string lookup, so the visual artefacts are identical.
        converter_patient_ids: list[str] | None
        if cfg.encode_full_cohort:
            converter_patient_ids = None
        else:
            converter_patient_ids = all_ids if all_ids else None
        # Resume is opt-in via ``resume_from_run_id``. On a fresh run, the
        # converter takes its fresh path and respects ``overwrite``: an
        # existing latent H5 with overwrite=False raises FileExistsError.
        conv_cfg = LatentH5Config(
            source_image_h5=cfg.source_image_h5,
            output_path=latent_h5,
            autoencoder_checkpoint=cfg.autoencoder_checkpoint,
            modalities=list(cfg.modalities),
            inference_mode=cfg.inference_mode,
            overwrite=cfg.overwrite,
            resume=cfg.resume_from_run_id is not None,
            checkpoint_every=cfg.checkpoint_every,
            limit=cfg.limit,
            patient_ids=converter_patient_ids,
            percentile_use_brain_mask=cfg.percentile_use_brain_mask,
        )
        converter = LatentH5Converter(cfg=conv_cfg, encoder=encoder, mask_downsampler=mask_ds)
        latent_path = converter.run()

        # ---- QC --------------------------------------------------------------
        decision: dict[str, Any] = {
            "schema_version": "0.1.0",
            "latent_h5_path": str(latent_path),
            "git_sha": resolve_git_sha() or "unknown",
            "created_at": now_iso_utc(),
            "percentile_use_brain_mask": bool(cfg.percentile_use_brain_mask),
            "n_patients_encoded": self._n_patients(latent_path),
            "modalities_encoded": list(cfg.modalities),
        }

        if cfg.roundtrip.enabled:
            rt = self._run_roundtrip_qc(
                latent_path=latent_path,
                decoder=decoder,
                figures_dir=figures_dir,
                tables_dir=tables_dir,
            )
            decision["roundtrip"] = rt

        if cfg.stabilization.enabled:
            stab = self._run_stabilization_qc(
                latent_path=latent_path,
                decoder=decoder,
                figures_dir=figures_dir,
                tables_dir=tables_dir,
                ids_by_grade=stab_ids_by_grade,
            )
            decision["stabilization"] = stab

        if cfg.pca.enabled:
            pca = self._run_pca_qc(
                latent_path=latent_path,
                figures_dir=figures_dir,
                tables_dir=tables_dir,
            )
            decision["pca"] = pca

        # ---- artefacts -------------------------------------------------------
        (self.run_dir / "decision.json").write_text(json.dumps(decision, indent=2))
        (self.run_dir / "git_sha.txt").write_text(decision["git_sha"] + "\n")
        self._write_report(decision)
        logger.info("Routine artefacts at %s", self.run_dir)
        return latent_path

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _make_run_dir(parent: Path) -> Path:
        parent = Path(parent).resolve()
        stamp = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
        run_dir = parent / stamp
        run_dir.mkdir(parents=True, exist_ok=True)
        # Update LATEST symlink atomically.
        latest = parent / "LATEST"
        try:
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(run_dir.name)
        except OSError as exc:
            logger.warning("could not update LATEST symlink: %s", exc)
        return run_dir

    def _resolve_resume_target(self, cfg: EncodeMaisiRoutineConfig) -> tuple[Path, Path]:
        """Locate the prior run dir and recover the latent H5 path.

        Returns ``(prior_run_dir, latent_h5_path)``. Raises
        :class:`FileNotFoundError` if the run dir is missing,
        :class:`ValueError` if the persisted config is unreadable, and
        :class:`FileNotFoundError` if the recovered latent H5 does not
        exist on disk.
        """
        assert cfg.resume_from_run_id is not None
        prior = Path(cfg.output_dir).resolve() / cfg.resume_from_run_id
        if not prior.is_dir():
            raise FileNotFoundError(
                f"resume_from_run_id={cfg.resume_from_run_id!r} not found "
                f"under {cfg.output_dir}: directory {prior} does not exist"
            )
        cfg_path = prior / "config.yaml"
        if not cfg_path.is_file():
            raise ValueError(
                f"prior run dir {prior} lacks config.yaml — cannot recover the latent H5 path"
            )
        with cfg_path.open("r") as f:
            prior_cfg_raw = yaml.safe_load(f) or {}
        prior_h5_raw = prior_cfg_raw.get("output_h5_path")
        if prior_h5_raw is None:
            # Fallback to the prior run's default (latent under run_dir).
            prior_h5 = prior / "UCSFPDGM_latent.h5"
        else:
            prior_h5 = Path(prior_h5_raw)
        if not prior_h5.is_file():
            raise FileNotFoundError(
                f"latent H5 declared by prior run not found: {prior_h5}. Cannot resume."
            )
        # Sanity: surface any mismatch between this YAML and the prior's.
        if cfg.output_h5_path is not None and Path(cfg.output_h5_path) != prior_h5:
            logger.warning(
                "resume: current YAML output_h5_path=%s differs from "
                "prior run's %s; using the prior one.",
                cfg.output_h5_path,
                prior_h5,
            )
        # Update LATEST to point at the resumed dir so consumers (e.g. the
        # report.md reader) find the live artefacts.
        latest = Path(cfg.output_dir).resolve() / "LATEST"
        try:
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(prior.name)
        except OSError as exc:
            logger.warning("could not update LATEST symlink: %s", exc)
        return prior, prior_h5

    def _persist_resolved_config(self, run_dir: Path) -> None:
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(json.loads(self.cfg.to_json()), sort_keys=False)
        )

    @staticmethod
    def _n_patients(latent_h5: Path) -> int:
        with h5py.File(latent_h5, "r") as f:
            return int(f["ids"].shape[0])

    # ----- roundtrip QC ---------------------------------------------------

    def _run_roundtrip_qc(
        self,
        latent_h5_path: Path | None = None,
        latent_path: Path | None = None,
        decoder: MaisiDecoder | None = None,
        figures_dir: Path | None = None,
        tables_dir: Path | None = None,
    ) -> dict[str, Any]:
        latent_h5 = latent_path or latent_h5_path
        assert latent_h5 is not None
        assert decoder is not None
        assert figures_dir is not None
        assert tables_dir is not None

        cfg = self.cfg
        selected = self._resolve_named_selection(latent_h5)
        if not selected:
            logger.warning("Roundtrip QC: no patients to render; skipping")
            return {"skipped": True, "reason": "no_patients"}
        logger.info(
            "Roundtrip QC: %d patients (grades=%s)",
            len(selected),
            [g for _, _, g in selected],
        )

        crop_spec_map: dict[str, CropPadSpec] = {}
        rows: list[RoundtripRow] = self._decode_rows(
            latent_h5=latent_h5,
            decoder=decoder,
            selected=selected,
            modalities=cfg.roundtrip.modalities,
            crop_spec_map=crop_spec_map,
        )
        if not rows:
            logger.warning("Roundtrip QC: no rows produced; skipping figure/CSV")
            return {"skipped": True, "reason": "no_rows"}

        # CSV of per-(patient, modality) metrics.
        metrics_lines = ["patient_id,who_grade,modality,mae,mse,lp3"]
        for r in rows:
            diff = r.reconstructed - r.original
            mae = float(np.mean(np.abs(diff)))
            mse = float(np.mean(diff**2))
            lp3 = float(np.mean(np.abs(diff) ** 3))
            metrics_lines.append(
                f"{r.patient_id},{r.who_grade},{r.modality},{mae:.6f},{mse:.6f},{lp3:.6f}"
            )
        metrics_csv = tables_dir / "roundtrip_metrics.csv"
        metrics_csv.write_text("\n".join(metrics_lines) + "\n")

        # NIfTI dump.
        nifti_paths: dict[str, list[str]] = {}
        if cfg.roundtrip.save_nifti:
            for r in rows:
                pair = self._save_nifti_pair(r, figures_dir / "nifti")
                nifti_paths.setdefault(r.patient_id, []).extend(str(p) for p in pair)

        # Per-modality figures (one figure per modality across patients).
        figure_paths: dict[str, str] = {}
        if cfg.roundtrip.per_modality_figures:
            outs = render_per_modality_roundtrip_figures(
                rows,
                output_dir=figures_dir,
                filename_template="roundtrip_{modality}.png",
                title_template="MAISI roundtrip — {modality}",
            )
            figure_paths = {m: str(p) for m, p in outs.items()}
        else:
            single = render_roundtrip_figure(rows, figures_dir / cfg.roundtrip.figure_filename)
            figure_paths = {"all": str(single)}

        # Aggregated stats per modality.
        per_modality_stats = self._aggregate_metrics(rows)
        return {
            "n_patients": len(selected),
            "grades": [g for _, _, g in selected],
            "figures": figure_paths,
            "metrics_csv": str(metrics_csv),
            "nifti": nifti_paths,
            "per_modality_stats": per_modality_stats,
        }

    # ----- shared QC primitives ------------------------------------------

    def _resolve_named_selection(
        self,
        latent_h5: Path,
    ) -> list[tuple[str, int, int]]:
        """Return ``[(pid, row_idx, who_grade), ...]`` for the roundtrip QC.

        When ``metadata/who_grade`` is absent (e.g. BraTS-GLI), all grades
        are reported as ``-1`` and the grade-stratified fallback selects the
        first ``n_patients`` rows instead.
        """
        cfg = self.cfg
        with h5py.File(latent_h5, "r") as f:
            ids = [v.decode() if isinstance(v, (bytes, bytearray)) else str(v) for v in f["ids"][:]]
            n = len(ids)
            if "metadata/who_grade" in f:
                grades = np.asarray(f["metadata/who_grade"][:], dtype=np.int8)
            else:
                grades = np.full(n, -1, dtype=np.int8)

        if cfg.patient_ids is not None:
            id_to_row = {pid: i for i, pid in enumerate(ids)}
            out: list[tuple[str, int, int]] = []
            for pid in cfg.patient_ids:
                if pid not in id_to_row:
                    logger.warning("roundtrip: %s not in latent H5; skipping", pid)
                    continue
                row = id_to_row[pid]
                out.append((pid, row, int(grades[row])))
            return out

        # Fallback: one patient per unique WHO grade (or first N when grade unavailable).
        unique_grades = sorted({int(g) for g in grades if int(g) >= 0})
        rng = np.random.default_rng(cfg.seed)
        out = []
        if unique_grades:
            for g in unique_grades:
                candidates = [i for i, gg in enumerate(grades) if int(gg) == g]
                pick = int(rng.choice(candidates))
                out.append((ids[pick], pick, g))
        else:
            # No grade info: pick first min(n_patients or 4, n) rows.
            cap = cfg.roundtrip.n_patients if cfg.roundtrip.n_patients is not None else 4
            for i in range(min(cap, n)):
                out.append((ids[i], i, -1))
        if cfg.roundtrip.n_patients is not None:
            out = out[: cfg.roundtrip.n_patients]
        return out

    @staticmethod
    def _build_crop_spec(latent_h5: Path, pid: str) -> CropPadSpec:
        """Reconstruct the per-patient :class:`CropPadSpec` from the latent H5.

        Reads ``crop/origin``, the native image shape, and ``crop_box`` from
        the source image H5 referenced by the latent H5's root attr.
        """
        import json as _json

        with h5py.File(latent_h5, "r") as f:
            src_path = Path(str(f.attrs["source_image_h5_path"]))
            crop_box_json = str(f.attrs.get("crop_box_json", f.attrs.get("crop_box", "null")))

        box: tuple[int, int, int] = tuple(_json.loads(crop_box_json))  # type: ignore[assignment]

        with h5py.File(src_path, "r") as src:
            all_ids = [
                v.decode() if isinstance(v, (bytes, bytearray)) else str(v) for v in src["ids"][:]
            ]
            src_row = all_ids.index(pid)
            crop_origin: tuple[int, int, int] = tuple(  # type: ignore[assignment]
                int(v) for v in src["crop/origin"][src_row]
            )
            # Native shape from the first image dataset in the source.
            first_ds = next(iter(src["images"].values()))
            native_shape: tuple[int, int, int] = tuple(first_ds.shape[1:])  # type: ignore[assignment]

        return CropPadSpec(
            crop_origin=crop_origin,
            native_shape=native_shape,
            target_shape=box,
        )

    def _decode_rows(
        self,
        latent_h5: Path,
        decoder: MaisiDecoder,
        selected: list[tuple[str, int, int]],
        modalities: list[str],
        crop_spec_map: dict[str, CropPadSpec] | None = None,
    ) -> list[RoundtripRow]:
        """Decode the selected rows and return roundtrip rows in box space."""
        cfg = self.cfg
        rows: list[RoundtripRow] = []
        with h5py.File(latent_h5, "r") as f:
            for pid, idx, grade in selected:
                # Build crop spec for this patient (cache in caller's map).
                if crop_spec_map is not None and pid in crop_spec_map:
                    spec = crop_spec_map[pid]
                else:
                    try:
                        spec = self._build_crop_spec(latent_h5, pid)
                    except Exception as exc:
                        logger.warning("Could not build crop spec for %s: %s; skipping", pid, exc)
                        continue
                    if crop_spec_map is not None:
                        crop_spec_map[pid] = spec

                centroid = self._tumor_centroid_box(spec)
                for mod in modalities:
                    if mod not in cfg.modalities:
                        logger.warning("decode row: modality %s not encoded; skipping", mod)
                        continue
                    z = np.asarray(f[f"latents/{mod}"][idx], dtype=np.float32)
                    z_t = torch.from_numpy(z).unsqueeze(0).to(decoder.handle.device)
                    # Box path: decode returns box volume directly.
                    rec = decoder.decode(z_t, crop_spec=spec).image
                    rec_np = rec[0, 0].detach().to("cpu", dtype=torch.float32).numpy()
                    src = self._read_normalised_source(latent_h5, pid, mod, spec)
                    if src.shape != rec_np.shape:
                        raise RuntimeError(
                            f"shape mismatch {pid}/{mod}: src={src.shape} rec={rec_np.shape}"
                        )
                    rows.append(
                        RoundtripRow(
                            patient_id=pid,
                            who_grade=int(grade),
                            modality=mod,
                            original=src,
                            reconstructed=rec_np,
                            tumor_centroid=centroid,
                        )
                    )
        return rows

    @staticmethod
    def _aggregate_metrics(rows: list[RoundtripRow]) -> dict[str, dict[str, float]]:
        """Mean / median / min / max / 95% bootstrap CI of MAE/MSE/Lp³ per modality."""
        per_mod: dict[str, dict[str, list[float]]] = {}
        for r in rows:
            diff = r.reconstructed - r.original
            d = per_mod.setdefault(r.modality, {"mae": [], "mse": [], "lp3": []})
            d["mae"].append(float(np.mean(np.abs(diff))))
            d["mse"].append(float(np.mean(diff**2)))
            d["lp3"].append(float(np.mean(np.abs(diff) ** 3)))
        out: dict[str, dict[str, float]] = {}
        for mod, d in per_mod.items():
            entry: dict[str, float] = {}
            for metric, vals in d.items():
                arr = np.asarray(vals, dtype=np.float64)
                entry[f"{metric}_mean"] = float(arr.mean())
                entry[f"{metric}_median"] = float(np.median(arr))
                entry[f"{metric}_min"] = float(arr.min())
                entry[f"{metric}_max"] = float(arr.max())
                lo, hi = EncodeMaisiRoutineEngine._bootstrap_ci(arr)
                entry[f"{metric}_ci95_lo"] = lo
                entry[f"{metric}_ci95_hi"] = hi
            entry["n"] = float(len(d["mae"]))
            out[mod] = entry
        return out

    @staticmethod
    def _bootstrap_ci(
        arr: np.ndarray,
        n_resamples: int = 5000,
        ci_level: float = 0.95,
        seed: int = 0,
    ) -> tuple[float, float]:
        """Percentile bootstrap CI for the mean of ``arr``."""
        n = arr.size
        if n < 2:
            return float(arr.mean()) if n else float("nan"), float(arr.mean()) if n else float(
                "nan"
            )
        rng = np.random.default_rng(seed)
        means = np.empty(n_resamples, dtype=np.float64)
        for i in range(n_resamples):
            means[i] = arr[rng.integers(0, n, n)].mean()
        lo = float(np.percentile(means, (1 - ci_level) / 2 * 100))
        hi = float(np.percentile(means, (1 + ci_level) / 2 * 100))
        return lo, hi

    # ----- NIfTI dump ----------------------------------------------------

    @staticmethod
    def _save_nifti_pair(row: RoundtripRow, out_dir: Path) -> tuple[Path, Path]:
        """Write the original and reconstructed volumes for ``row`` as
        ``{pid}_{modality}_{original,reconstructed}.nii.gz``. Identity
        affine, 1 mm isotropic spacing (UCSF-PDGM convention)."""
        import nibabel as nib

        out_dir.mkdir(parents=True, exist_ok=True)
        affine = np.eye(4, dtype=np.float64)  # 1 mm iso, LPS-ish
        orig_path = out_dir / f"{row.patient_id}_{row.modality}_original.nii.gz"
        rec_path = out_dir / f"{row.patient_id}_{row.modality}_reconstructed.nii.gz"
        nib.save(nib.Nifti1Image(row.original.astype(np.float32), affine), str(orig_path))
        nib.save(nib.Nifti1Image(row.reconstructed.astype(np.float32), affine), str(rec_path))
        return orig_path, rec_path

    # ----- stabilisation pass --------------------------------------------

    def _sample_stratified_ids(
        self,
        cfg: EncodeMaisiRoutineConfig,
    ) -> dict[int, list[str]]:
        """Sample ``n_per_grade`` patient IDs per requested WHO grade from
        the *source image H5*. Returns ``{grade: [ids...]}``; grades that
        are not present in the cohort yield an empty list.

        When ``metadata/who_grade`` is absent (e.g. BraTS-GLI), returns
        ``{}`` (empty dict) and logs a warning — the stabilization pass is
        effectively skipped for that cohort.
        """
        with h5py.File(cfg.source_image_h5, "r") as f:
            ids = [v.decode() if isinstance(v, (bytes, bytearray)) else str(v) for v in f["ids"][:]]
            if "metadata/who_grade" not in f:
                logger.warning(
                    "source image H5 has no metadata/who_grade; "
                    "stabilization stratification is unavailable — returning empty."
                )
                return {}
            grades = np.asarray(f["metadata/who_grade"][:], dtype=np.int8)
        rng = np.random.default_rng(cfg.stabilization.seed)
        out: dict[int, list[str]] = {}
        for g in cfg.stabilization.grades:
            candidates = [i for i, gg in enumerate(grades) if int(gg) == int(g)]
            if not candidates:
                logger.info("stabilization: no patients with WHO grade %d; skipping", g)
                out[int(g)] = []
                continue
            n_pick = min(cfg.stabilization.n_per_grade, len(candidates))
            sel = rng.choice(candidates, size=n_pick, replace=False)
            out[int(g)] = [ids[int(i)] for i in sel]
        return out

    def _run_stabilization_qc(
        self,
        latent_path: Path,
        decoder: MaisiDecoder,
        figures_dir: Path,
        tables_dir: Path,
        ids_by_grade: dict[int, list[str]],
    ) -> dict[str, Any]:
        cfg = self.cfg
        all_ids = [pid for g in cfg.stabilization.grades for pid in ids_by_grade.get(g, [])]
        if not all_ids:
            logger.warning("stabilization: empty pick; skipping")
            return {"skipped": True, "reason": "no_patients"}

        # Map patient -> latent row.
        with h5py.File(latent_path, "r") as f:
            latent_ids = [
                v.decode() if isinstance(v, (bytes, bytearray)) else str(v) for v in f["ids"][:]
            ]
            if "metadata/who_grade" in f:
                grades_in_latent = np.asarray(f["metadata/who_grade"][:], dtype=np.int8)
            else:
                grades_in_latent = np.full(len(latent_ids), -1, dtype=np.int8)
        row_of = {pid: i for i, pid in enumerate(latent_ids)}
        selected: list[tuple[str, int, int]] = []
        for pid in all_ids:
            if pid not in row_of:
                logger.warning("stabilization: %s not in latent H5", pid)
                continue
            r = row_of[pid]
            selected.append((pid, r, int(grades_in_latent[r])))

        logger.info(
            "Stabilization QC: %d patients across grades %s",
            len(selected),
            sorted({g for _, _, g in selected}),
        )

        crop_spec_map: dict[str, CropPadSpec] = {}
        rows = self._decode_rows(
            latent_h5=latent_path,
            decoder=decoder,
            selected=selected,
            modalities=cfg.stabilization.modalities,
            crop_spec_map=crop_spec_map,
        )

        # Raw per-(patient, modality) metrics.
        metrics_lines = ["patient_id,who_grade,modality,mae,mse,lp3"]
        for r in rows:
            diff = r.reconstructed - r.original
            mae = float(np.mean(np.abs(diff)))
            mse = float(np.mean(diff**2))
            lp3 = float(np.mean(np.abs(diff) ** 3))
            metrics_lines.append(
                f"{r.patient_id},{r.who_grade},{r.modality},{mae:.6f},{mse:.6f},{lp3:.6f}"
            )
        metrics_csv = tables_dir / "stabilization_metrics.csv"
        metrics_csv.write_text("\n".join(metrics_lines) + "\n")

        # Aggregated stats: pooled (all grades) and per-grade per-modality.
        agg_pooled = self._aggregate_metrics(rows)
        agg_by_grade: dict[str, dict[str, dict[str, float]]] = {}
        for g in sorted({r.who_grade for r in rows}):
            agg_by_grade[str(g)] = self._aggregate_metrics([r for r in rows if r.who_grade == g])
        # Persist a tidy long-form aggregate CSV.
        agg_lines = ["scope,grade,modality,metric,mean,median,min,max,ci95_lo,ci95_hi,n"]
        for mod, e in agg_pooled.items():
            for metric in ("mae", "mse", "lp3"):
                agg_lines.append(
                    f"pooled,all,{mod},{metric},"
                    f"{e[f'{metric}_mean']:.6f},{e[f'{metric}_median']:.6f},"
                    f"{e[f'{metric}_min']:.6f},{e[f'{metric}_max']:.6f},"
                    f"{e[f'{metric}_ci95_lo']:.6f},{e[f'{metric}_ci95_hi']:.6f},"
                    f"{int(e['n'])}"
                )
        for g, by_mod in agg_by_grade.items():
            for mod, e in by_mod.items():
                for metric in ("mae", "mse", "lp3"):
                    agg_lines.append(
                        f"per_grade,{g},{mod},{metric},"
                        f"{e[f'{metric}_mean']:.6f},{e[f'{metric}_median']:.6f},"
                        f"{e[f'{metric}_min']:.6f},{e[f'{metric}_max']:.6f},"
                        f"{e[f'{metric}_ci95_lo']:.6f},{e[f'{metric}_ci95_hi']:.6f},"
                        f"{int(e['n'])}"
                    )
        agg_csv = tables_dir / "stabilization_metrics_aggregate.csv"
        agg_csv.write_text("\n".join(agg_lines) + "\n")

        # Set-collage figures: build "set k" from the k-th sample of each
        # grade (if present). One figure per set, in figures/stabilization/.
        set_figures_dir = figures_dir / "stabilization"
        set_figure_paths: list[str] = []
        if cfg.stabilization.set_figures:
            n_sets = cfg.stabilization.n_per_grade
            rows_by_grade: dict[int, dict[str, list[RoundtripRow]]] = {}
            for r in rows:
                rows_by_grade.setdefault(r.who_grade, {}).setdefault(r.patient_id, []).append(r)
            for k in range(n_sets):
                set_rows: list[RoundtripRow] = []
                row_labels: list[str] = []
                for g in cfg.stabilization.grades:
                    pids = ids_by_grade.get(g, [])
                    if k >= len(pids):
                        continue
                    pid = pids[k]
                    rows_for_pid = rows_by_grade.get(int(g), {}).get(pid, [])
                    set_rows.extend(rows_for_pid)
                if not set_rows:
                    continue
                set_path = set_figures_dir / f"set_{k + 1:02d}.png"

                def _label(r: RoundtripRow) -> str:
                    return f"{r.patient_id}\nWHO {r.who_grade}\n{r.modality}"

                render_roundtrip_figure(
                    set_rows,
                    set_path,
                    title=f"Stabilization set #{k + 1:02d}",
                    row_label=_label,
                )
                set_figure_paths.append(str(set_path))

        return {
            "n_patients": len({r.patient_id for r in rows}),
            "n_rows": len(rows),
            "grades_used": sorted({r.who_grade for r in rows}),
            "metrics_csv": str(metrics_csv),
            "aggregate_csv": str(agg_csv),
            "pooled_per_modality_stats": agg_pooled,
            "per_grade_stats": agg_by_grade,
            "set_figures": set_figure_paths,
        }

    @staticmethod
    def _source_row_for_pid(src: h5py.File, pid: str) -> int:
        raw = src["ids"][:]
        all_ids = [v.decode() if isinstance(v, (bytes, bytearray)) else str(v) for v in raw]
        try:
            return all_ids.index(pid)
        except ValueError as exc:
            raise RuntimeError(f"patient {pid!r} not in source image H5") from exc

    @staticmethod
    def _tumor_centroid_box(spec: CropPadSpec) -> tuple[int, int, int]:
        """Return the centre of the crop box as the (approximate) tumour centroid.

        In box space we no longer have easy access to the native segmentation
        centroid without re-reading the source H5. For QC slice selection the
        box centre is a safe, cheap fallback — the tumour is expected to be
        near the box centre by construction of the brain-centred crop.
        """
        return (
            spec.target_shape[0] // 2,
            spec.target_shape[1] // 2,
            spec.target_shape[2] // 2,
        )

    def _read_normalised_source(
        self, latent_h5: Path, pid: str, modality: str, spec: CropPadSpec
    ) -> np.ndarray:
        """Crop and normalise the native source image to box space.

        Applies :func:`apply_crop_pad` with ``spec`` then
        :func:`percentile_normalise` so the box-space normalised source is
        directly comparable against the decoded box output.
        """
        cfg = self.cfg
        with h5py.File(latent_h5, "r") as f:
            src_path = Path(str(f.attrs["source_image_h5_path"]))
        with h5py.File(src_path, "r") as src:
            src_row = self._source_row_for_pid(src, pid)
            arr = np.asarray(src[f"images/{modality}"][src_row], dtype=np.float32)
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        # Crop to box first, then normalise over the box region.
        t_box = apply_crop_pad(t, spec)
        return percentile_normalise(
            t_box,
            lower=cfg.percentile_lower,
            upper=cfg.percentile_upper,
            foreground_only=cfg.percentile_foreground_only,
        )[0, 0].numpy()

    # ----- PCA QC ----------------------------------------------------------

    def _run_pca_qc(
        self,
        latent_path: Path,
        figures_dir: Path,
        tables_dir: Path,
    ) -> dict[str, Any]:
        cfg = self.cfg
        rows: list[np.ndarray] = []
        row_modality: list[str] = []
        row_volume: list[float] = []

        with h5py.File(latent_path, "r") as f:
            n = int(f["ids"].shape[0])
            ids = [v.decode() if isinstance(v, (bytes, bytearray)) else str(v) for v in f["ids"][:]]
            tumor_volumes_ml = self._tumor_volumes_ml(f, n)

            for slug in cfg.modalities:
                ds = f[f"latents/{slug}"]
                # GAP over (C, h, w, d) → (C,) per row.
                for i in range(n):
                    z = np.asarray(ds[i], dtype=np.float32)  # (C, h, w, d)
                    pooled = z.reshape(z.shape[0], -1).mean(axis=1)  # (C,)
                    rows.append(pooled)
                    row_modality.append(slug)
                    row_volume.append(float(tumor_volumes_ml[i]))

        pooled_arr = np.stack(rows, axis=0)
        vol_arr = np.asarray(row_volume, dtype=np.float64)

        fig_path = figures_dir / cfg.pca.figure_filename
        render_pca_figure(
            pooled_latents=pooled_arr,
            modalities_per_row=row_modality,
            tumor_volume_ml=vol_arr,
            output_path=fig_path,
            n_components=cfg.pca.n_components,
        )

        # Persist the raw GAP table for downstream analyses.
        header = ["patient_id", "modality", "tumor_volume_ml"] + [
            f"gap_{i}" for i in range(pooled_arr.shape[1])
        ]
        lines = [",".join(header)]
        for j in range(pooled_arr.shape[0]):
            i_patient = j % n  # since we wrote n × modalities in modality-outer order
            # actually we wrote in modality-outer order, so:
            i_patient = j % n
            line = [
                ids[i_patient],
                row_modality[j],
                f"{row_volume[j]:.4f}",
                *[f"{v:.6f}" for v in pooled_arr[j]],
            ]
            lines.append(",".join(line))
        (tables_dir / "pca_gap_table.csv").write_text("\n".join(lines) + "\n")
        return {
            "n_rows": int(pooled_arr.shape[0]),
            "n_features": int(pooled_arr.shape[1]),
            "figure_png": str(fig_path),
            "table_csv": str(tables_dir / "pca_gap_table.csv"),
        }

    @staticmethod
    def _tumor_volumes_ml(f: h5py.File, n: int) -> np.ndarray:
        """Compute the NETC+ET volume in mL from the source image H5.

        Uses 1 mm³ isotropic spacing (UCSF-PDGM convention; recorded in the
        image manifest extras). Per-row lookup uses patient IDs so the
        result is correct for both full-cohort runs and explicit-ID
        subsets.
        """
        src_path = Path(str(f.attrs["source_image_h5_path"]))
        if not src_path.is_file():
            return np.zeros(n, dtype=np.float64)
        latent_ids = [
            v.decode() if isinstance(v, (bytes, bytearray)) else str(v) for v in f["ids"][:]
        ]
        ml_per_voxel = 1e-3
        out = np.zeros(n, dtype=np.float64)
        with h5py.File(src_path, "r") as src:
            raw = src["ids"][:]
            src_ids = [v.decode() if isinstance(v, (bytes, bytearray)) else str(v) for v in raw]
            src_index = {pid: i for i, pid in enumerate(src_ids)}
            for i, pid in enumerate(latent_ids):
                if pid not in src_index:
                    continue
                seg = np.asarray(src["masks/tumor"][src_index[pid]], dtype=np.int8)
                count = int(((seg == 1) | (seg == 4)).sum())
                out[i] = count * ml_per_voxel
        return out

    # ----- report ---------------------------------------------------------

    def _write_report(self, decision: dict[str, Any]) -> None:
        assert self.run_dir is not None
        lines: list[str] = []
        lines.append("# MAISI encode routine — report")
        lines.append("")
        lines.append(f"- created_at: {decision['created_at']}")
        lines.append(f"- git_sha: {decision['git_sha']}")
        lines.append(f"- latent_h5: `{decision['latent_h5_path']}`")
        lines.append(f"- n_patients_encoded: {decision['n_patients_encoded']}")
        lines.append(f"- modalities_encoded: {decision['modalities_encoded']}")
        rt = decision.get("roundtrip")
        if rt is not None:
            lines.append("")
            lines.append("## Roundtrip fidelity (named patients)")
            if rt.get("skipped"):
                lines.append(f"_skipped: {rt.get('reason')}_")
            else:
                lines.append(f"- n_patients: {rt['n_patients']} (grades={rt['grades']})")
                lines.append("- figures:")
                for mod, path in rt.get("figures", {}).items():
                    lines.append(f"  - {mod}: `{path}`")
                lines.append("- per-modality stats (mean / median / 95% CI):")
                lines.append("")
                lines.append(
                    "| modality | MAE mean | MAE median | MAE 95% CI | MSE mean | Lp³ mean |"
                )
                lines.append(
                    "|----------|---------:|-----------:|:----------:|---------:|---------:|"
                )
                for mod, m in rt.get("per_modality_stats", {}).items():
                    lines.append(
                        f"| {mod} | {m['mae_mean']:.4f} | {m['mae_median']:.4f} | "
                        f"[{m['mae_ci95_lo']:.4f}, {m['mae_ci95_hi']:.4f}] | "
                        f"{m['mse_mean']:.5f} | {m['lp3_mean']:.6f} |"
                    )
        stab = decision.get("stabilization")
        if stab is not None:
            lines.append("")
            lines.append("## Stabilization sweep")
            if stab.get("skipped"):
                lines.append(f"_skipped: {stab.get('reason')}_")
            else:
                lines.append(f"- n_patients: {stab['n_patients']}  (rows: {stab['n_rows']})")
                lines.append(f"- grades_used: {stab['grades_used']}")
                lines.append(f"- aggregate CSV: `{stab['aggregate_csv']}`")
                lines.append(f"- raw CSV: `{stab['metrics_csv']}`")
                lines.append("- pooled per-modality MAE / MSE / Lp³ (mean ± 95% CI):")
                lines.append("")
                lines.append("| modality | MAE | MSE | Lp³ | n |")
                lines.append("|----------|---|---|---|---:|")
                for mod, m in stab.get("pooled_per_modality_stats", {}).items():
                    lines.append(
                        f"| {mod} | "
                        f"{m['mae_mean']:.4f} [{m['mae_ci95_lo']:.4f}, {m['mae_ci95_hi']:.4f}] | "
                        f"{m['mse_mean']:.5f} [{m['mse_ci95_lo']:.5f}, {m['mse_ci95_hi']:.5f}] | "
                        f"{m['lp3_mean']:.6f} [{m['lp3_ci95_lo']:.6f}, {m['lp3_ci95_hi']:.6f}] | "
                        f"{int(m['n'])} |"
                    )
                if stab.get("per_grade_stats"):
                    lines.append("")
                    lines.append("- per-WHO-grade per-modality MAE (mean [95% CI]):")
                    lines.append("")
                    grades_sorted = sorted(stab["per_grade_stats"].keys(), key=int)
                    mods_sorted = sorted(stab["pooled_per_modality_stats"].keys())
                    header = "| modality | " + " | ".join(f"WHO {g}" for g in grades_sorted) + " |"
                    sep = "|----------|" + "|".join("---" for _ in grades_sorted) + "|"
                    lines.append(header)
                    lines.append(sep)
                    for mod in mods_sorted:
                        cells: list[str] = []
                        for g in grades_sorted:
                            e = stab["per_grade_stats"][g].get(mod)
                            if e is None:
                                cells.append("—")
                            else:
                                cells.append(
                                    f"{e['mae_mean']:.4f} [{e['mae_ci95_lo']:.4f}, "
                                    f"{e['mae_ci95_hi']:.4f}] (n={int(e['n'])})"
                                )
                        lines.append(f"| {mod} | " + " | ".join(cells) + " |")
                if stab.get("set_figures"):
                    lines.append("")
                    lines.append(f"- {len(stab['set_figures'])} set collage figures:")
                    for p in stab["set_figures"]:
                        lines.append(f"  - `{p}`")
        pca = decision.get("pca")
        if pca is not None:
            lines.append("")
            lines.append("## PCA of GAP-pooled latents")
            lines.append(f"- rows: {pca['n_rows']}  (patients × modalities)")
            lines.append(f"- features: {pca['n_features']}")
            lines.append(f"- figure: `{pca['figure_png']}`")
        lines.append("")
        (self.run_dir / "report.md").write_text("\n".join(lines))


__all__ = ["EncodeMaisiRoutineConfig", "EncodeMaisiRoutineEngine"]


def _iterate_modalities(slugs: Iterable[str]) -> list[str]:
    """Helper retained for future symmetric API; currently a no-op."""
    return list(slugs)
