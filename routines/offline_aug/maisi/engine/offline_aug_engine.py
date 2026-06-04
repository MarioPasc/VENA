"""Engine for the offline-augmentation routine.

For one cohort, on one rank of a scan-level shard:

1. Build ``<cohort>_image_aug_rank{rank}.h5`` via
   :class:`vena.data.augment.offline.OfflineAugBankBuilder` — TorchIO
   composes per (scan, variant), MONAI's ``RandHistogramShift`` shim for
   variant v2.
2. Encode the aug-image H5 via
   :class:`vena.data.h5.latent_domain.LatentH5Converter` in ``aug_mode=True``
   → ``<cohort>_latents_aug_rank{rank}.h5``.
3. (Optional, when ``merge_after=True`` and per-rank shards are present)
   merge two rank shards into ``<cohort>_{image,latents}_aug.h5`` and
   validate.
4. QC: sample ``n_patients_per_variant`` rows per variant from this rank's
   shard, decode the latent rows, compute PSNR/SSIM between the stored
   augmented image and ``D(E(aug))``; aggregate per (cohort × variant),
   gate against the equivariance preflight's per-cohort
   ``vae_recon_floor``.
5. Write ``decision.json`` (schema 0.1.0) with build provenance + QC.

The engine is the only place the GPU is touched. Configs are split into
shard configs (per-rank) and a optional merge-config that just calls the
merge helpers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field

from vena.data.augment.offline.bank_builder import (
    OfflineAugBankBuilder,
    merge_aug_image_h5_shards,
)
from vena.data.augment.offline.variants import VARIANT_NAMES
from vena.data.h5.augmented import (
    AUG_IMAGE_CROP_BOX,
    assert_aug_latent_h5_valid,
)
from vena.data.h5.latent_domain.convert import LatentH5Config, LatentH5Converter
from vena.data.h5.shared import resolve_git_sha, sha256_file
from vena.model.autoencoder.maisi import (  # type: ignore[attr-defined]
    LATENT_CHANNELS,
)
from vena.model.autoencoder.maisi.decode import MaisiDecoder
from vena.model.autoencoder.maisi.encode import MaisiEncoder, get_downsampler
from vena.model.autoencoder.maisi.loader import load_autoencoder
from vena.model.autoencoder.maisi.preprocessing import (
    CropPadSpec,
    percentile_normalise,
)

from ..figures import (
    AugRoundtripRow,
    aggregate_cohort_variant_stats,
    render_aug_roundtrip_figure,
)

logger = logging.getLogger(__name__)

_PRODUCER_VERSION = "0.1.0"
_PRODUCER = f"routines.offline_aug.maisi.engine:{_PRODUCER_VERSION}"


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _box_native_numpy(
    arr: np.ndarray,
    crop_origin: tuple[int, int, int],
    target: tuple[int, int, int] = AUG_IMAGE_CROP_BOX,
) -> np.ndarray:
    """Apply the same brain-box crop the bank-builder did, in pure numpy.

    Mirrors ``vena.data.augment.offline.bank_builder._box_native`` (which uses
    the torch primitive ``apply_crop_pad``) so the QC ``original`` panel sees
    the exact same content the augmentation acted on.
    """
    o = tuple(int(v) for v in crop_origin)
    n = tuple(int(v) for v in arr.shape)
    out = np.zeros(target, dtype=arr.dtype)
    for axis in range(3):
        pass
    s_h = slice(max(0, o[0]), min(n[0], o[0] + target[0]))
    s_w = slice(max(0, o[1]), min(n[1], o[1] + target[1]))
    s_d = slice(max(0, o[2]), min(n[2], o[2] + target[2]))
    d_h = slice(max(0, -o[0]), max(0, -o[0]) + (s_h.stop - s_h.start))
    d_w = slice(max(0, -o[1]), max(0, -o[1]) + (s_w.stop - s_w.start))
    d_d = slice(max(0, -o[2]), max(0, -o[2]) + (s_d.stop - s_d.start))
    out[d_h, d_w, d_d] = arr[s_h, s_w, s_d]
    return out


class _SlidingWindowCfg(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    roi_size: list[int] = Field(default_factory=lambda: [80, 80, 32])
    overlap: float = 0.4
    mode: str = "gaussian"


class _MaskDownsamplerCfg(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str = "per_class_avg_pool"
    params: dict[str, Any] = Field(default_factory=dict)


class _QcCfg(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    enabled: bool = True
    n_patients_per_variant: int = 4
    figure_filename_template: str = "roundtrip_{cohort}_{variant}.png"
    equivariance_decision_path: Path | None = None
    psnr_tolerance_db: float = 2.0
    ssim_tolerance: float = 0.02
    psnr_floor_default_db: float = 26.0
    ssim_floor_default: float = 0.93


class _DedupCfg(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    enabled: bool = False
    decisions_path: Path | None = None


class _MergeCfg(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    enabled: bool = False
    shards: list[Path] = Field(default_factory=list)


class OfflineAugMaisiRoutineConfig(BaseModel):
    """YAML-driven config for one shard of the offline-aug routine."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cohort: str
    source_image_h5: Path
    autoencoder_checkpoint: Path

    output_dir: Path = Path("artifacts/offline_aug/maisi")
    image_aug_h5_path: Path | None = None  # auto-derived if None
    latent_aug_h5_path: Path | None = None  # auto-derived if None
    modalities: list[str] = Field(default_factory=lambda: ["t1pre", "t1c", "t2", "flair"])
    variants: list[str] = Field(default_factory=lambda: list(VARIANT_NAMES))

    aug_pipeline_yaml: Path
    """Path to the K-variant YAML (per-variant hyperparams)."""

    device: str = "cuda"
    precision_mode: str = "autocast"
    autoencoder_norm_float16: bool | None = None
    inference_mode: str = "auto"
    sliding_window: _SlidingWindowCfg = Field(default_factory=_SlidingWindowCfg)
    depth_pad_base: int = 8
    mask_downsampler: _MaskDownsamplerCfg = Field(default_factory=_MaskDownsamplerCfg)
    percentile_lower: float = 0.0
    percentile_upper: float = 99.5
    percentile_foreground_only: bool = True

    world_size: int = 1
    rank: int = 0
    seed: int = 42
    overwrite: bool = False
    log_level: str = "INFO"
    limit_source_rows: int | None = None
    """Optional cap on the per-rank source-scan list (smoke runs)."""

    dedup: _DedupCfg = Field(default_factory=_DedupCfg)
    qc: _QcCfg = Field(default_factory=_QcCfg)
    merge: _MergeCfg = Field(default_factory=_MergeCfg)

    @classmethod
    def from_yaml(cls, path: Path | str) -> OfflineAugMaisiRoutineConfig:
        with Path(path).open("r") as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw)

    def to_json(self) -> str:
        return self.model_dump_json()


class OfflineAugMaisiRoutineEngine:
    """Single-shard execution of the offline-aug routine."""

    def __init__(self, cfg: OfflineAugMaisiRoutineConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------ public

    def run(self) -> Path:
        cfg = self.cfg
        logging.getLogger().setLevel(cfg.log_level)
        torch.set_float32_matmul_precision("high")
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        run_dir = self._make_run_dir(cfg.output_dir)
        figures_dir = run_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        self._persist_resolved_config(run_dir)

        # ---- 0. resolve paths ------------------------------------------------
        image_aug_h5_path = cfg.image_aug_h5_path or self._default_image_aug_path()
        latent_aug_h5_path = cfg.latent_aug_h5_path or self._default_latent_aug_path()
        image_aug_h5_path.parent.mkdir(parents=True, exist_ok=True)

        # ---- 1. load aug pipeline + provenance ------------------------------
        aug_yaml = Path(cfg.aug_pipeline_yaml).read_text()
        aug_payload = yaml.safe_load(aug_yaml) or {}
        aug_config_sha256 = _sha256_str(aug_yaml)
        variant_hp = aug_payload.get("variants", {}) or {}

        # ---- 2. dedup allowlist (optional) ----------------------------------
        dedup_allowlist: set[str] | None = None
        if cfg.dedup.enabled:
            if cfg.dedup.decisions_path is None:
                raise ValueError("dedup.enabled=True but dedup.decisions_path is None")
            from vena.preflight.cohort_dedup import (
                assert_dedup_decision_valid,
                build_allowlists,
            )

            payload = assert_dedup_decision_valid(cfg.dedup.decisions_path)
            allowlists = build_allowlists(payload)
            dedup_allowlist = allowlists.get(cfg.cohort)
            if dedup_allowlist is None:
                raise RuntimeError(
                    f"cohort_dedup decision.json has no entry for cohort {cfg.cohort!r}"
                )
            logger.info(
                "dedup ENABLED for %s (%d allowed patients)",
                cfg.cohort,
                len(dedup_allowlist),
            )

        # ---- 3. PHASE 1 — image-domain bank shard ---------------------------
        t0 = time.monotonic()
        builder = OfflineAugBankBuilder(
            source_image_h5=cfg.source_image_h5,
            output_path=image_aug_h5_path,
            cohort=cfg.cohort,
            modalities=cfg.modalities,
            variants=cfg.variants,
            variant_hyperparams=variant_hp,
            aug_config_json=aug_yaml,
            aug_config_sha256=aug_config_sha256,
            dedup_allowlist=dedup_allowlist,
            world_size=cfg.world_size,
            rank=cfg.rank,
            seed=cfg.seed,
            overwrite=cfg.overwrite,
            limit_source_rows=cfg.limit_source_rows,
        )
        builder.build()
        bank_seconds = time.monotonic() - t0
        with h5py.File(image_aug_h5_path, "r") as f:
            n_image_rows = int(f["ids"].shape[0])
        logger.info(
            "image-aug shard: %d rows in %.1f s (avg %.2f s/row)",
            n_image_rows,
            bank_seconds,
            bank_seconds / max(n_image_rows, 1),
        )

        # ---- 4. PHASE 2 — latent shard --------------------------------------
        t0 = time.monotonic()
        device = torch.device(cfg.device)
        norm_fp16 = cfg.autoencoder_norm_float16
        if norm_fp16 is None:
            norm_fp16 = cfg.precision_mode == "autocast"
        handle = load_autoencoder(
            cfg.autoencoder_checkpoint,
            device=device,
            arch_overrides={"norm_float16": bool(norm_fp16)},
        )
        encoder = MaisiEncoder(
            handle,
            sliding_roi=tuple(cfg.sliding_window.roi_size),  # type: ignore[arg-type]
            sliding_overlap=cfg.sliding_window.overlap,
            sliding_mode=cfg.sliding_window.mode,
            depth_pad_base=cfg.depth_pad_base,
            percentile_lower=cfg.percentile_lower,
            percentile_upper=cfg.percentile_upper,
            percentile_foreground_only=cfg.percentile_foreground_only,
            precision_mode=cfg.precision_mode,
        )
        mask_ds = get_downsampler(cfg.mask_downsampler.name, **cfg.mask_downsampler.params)

        latent_cfg = LatentH5Config(
            source_image_h5=image_aug_h5_path,
            output_path=latent_aug_h5_path,
            autoencoder_checkpoint=cfg.autoencoder_checkpoint,
            modalities=cfg.modalities,
            inference_mode=cfg.inference_mode,
            overwrite=cfg.overwrite,
            resume=False,
            checkpoint_every=64,
            aug_mode=True,
            aug_config_sha256=aug_config_sha256,
        )
        LatentH5Converter(latent_cfg, encoder=encoder, mask_downsampler=mask_ds).run()
        assert_aug_latent_h5_valid(
            latent_aug_h5_path,
            cfg.cohort,
            cfg.modalities,
            mask_ds.output_channels,
        )
        encode_seconds = time.monotonic() - t0
        logger.info("latent-aug shard: encoded in %.1f s", encode_seconds)

        # ---- 5. PHASE 3 — QC -------------------------------------------------
        decoder = MaisiDecoder(handle, precision_mode=cfg.precision_mode)
        decision_qc: dict[str, Any] = {"enabled": False}
        if cfg.qc.enabled:
            decision_qc = self._run_qc(
                cfg=cfg,
                image_aug_h5=image_aug_h5_path,
                latent_aug_h5=latent_aug_h5_path,
                decoder=decoder,
                figures_dir=figures_dir,
            )

        # ---- 6. PHASE 4 — merge (only when explicitly requested) ------------
        merged_paths: dict[str, str] | None = None
        if cfg.merge.enabled:
            if not cfg.merge.shards:
                raise ValueError("merge.enabled=True but merge.shards is empty")
            merged_image_path = self._merged_path(image_aug_h5_path)
            merged_latent_path = self._merged_path(latent_aug_h5_path)
            merge_aug_image_h5_shards(
                shards=cfg.merge.shards,
                merged_path=merged_image_path,
                cohort=cfg.cohort,
                modalities=cfg.modalities,
                overwrite=cfg.overwrite,
            )
            # Note: the latent-shard merge follows the same row-concat policy;
            # implemented inline for symmetry (no shared utility yet).
            self._merge_latent_shards(
                shards=[self._latent_for_image(s) for s in cfg.merge.shards],
                merged_path=merged_latent_path,
                cohort=cfg.cohort,
                modalities=cfg.modalities,
                mask_channels=mask_ds.output_channels,
                aug_config_sha256=aug_config_sha256,
                overwrite=cfg.overwrite,
            )
            merged_paths = {
                "image_aug_h5": str(merged_image_path),
                "latent_aug_h5": str(merged_latent_path),
            }

        # ---- 7. decision.json + report.md -----------------------------------
        decision: dict[str, Any] = {
            "schema_version": "0.1.0",
            "produced_at": datetime.now(tz=UTC).isoformat(),
            "producer": _PRODUCER,
            "run_dir": str(run_dir),
            "cohort": cfg.cohort,
            "rank": cfg.rank,
            "world_size": cfg.world_size,
            "seed": cfg.seed,
            "image_aug_h5_path": str(image_aug_h5_path),
            "image_aug_h5_sha256": sha256_file(image_aug_h5_path),
            "latent_aug_h5_path": str(latent_aug_h5_path),
            "latent_aug_h5_sha256": sha256_file(latent_aug_h5_path),
            "source_image_h5_path": str(cfg.source_image_h5),
            "source_image_h5_sha256": sha256_file(cfg.source_image_h5),
            "autoencoder_checkpoint_path": str(cfg.autoencoder_checkpoint),
            "autoencoder_checkpoint_sha256": handle.checkpoint_sha256,
            "aug_config_sha256": aug_config_sha256,
            "variants": cfg.variants,
            "modalities": cfg.modalities,
            "n_image_rows_written": n_image_rows,
            "wall_clock_seconds": {
                "bank_build": bank_seconds,
                "encode": encode_seconds,
            },
            "dedup": {
                "enabled": cfg.dedup.enabled,
                "decisions_path": str(cfg.dedup.decisions_path)
                if cfg.dedup.decisions_path
                else None,
                "n_allowed_patients": len(dedup_allowlist) if dedup_allowlist else None,
            },
            "qc": decision_qc,
            "merged_outputs": merged_paths,
            "git_sha": resolve_git_sha(),
        }
        (run_dir / "decision.json").write_text(json.dumps(decision, indent=2))
        self._write_report(run_dir, decision)
        return run_dir

    # ------------------------------------------------------------------ helpers

    def _default_image_aug_path(self) -> Path:
        cfg = self.cfg
        src = cfg.source_image_h5
        suffix = f"_rank{cfg.rank}" if cfg.world_size > 1 else ""
        return src.parent / f"{cfg.cohort}_image_aug{suffix}.h5"

    def _default_latent_aug_path(self) -> Path:
        cfg = self.cfg
        src = cfg.source_image_h5
        suffix = f"_rank{cfg.rank}" if cfg.world_size > 1 else ""
        return src.parent / f"{cfg.cohort}_latents_aug{suffix}.h5"

    @staticmethod
    def _merged_path(shard_path: Path) -> Path:
        name = shard_path.name.replace("_rank0", "").replace("_rank1", "")
        return shard_path.parent / name

    @staticmethod
    def _latent_for_image(image_shard: Path) -> Path:
        return image_shard.parent / image_shard.name.replace("_image_aug", "_latents_aug")

    @staticmethod
    def _make_run_dir(parent: Path) -> Path:
        parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
        run_dir = parent / stamp
        run_dir.mkdir(parents=True, exist_ok=False)
        latest = parent / "LATEST"
        if latest.is_symlink() or latest.exists():
            try:
                latest.unlink()
            except OSError:
                pass
        try:
            latest.symlink_to(run_dir.name)
        except OSError as exc:
            logger.warning("could not update LATEST symlink: %s", exc)
        return run_dir

    def _persist_resolved_config(self, run_dir: Path) -> None:
        config_path = run_dir / "config.yaml"
        config_path.write_text(yaml.safe_dump(json.loads(self.cfg.to_json()), sort_keys=False))

    # ----- QC ----------------------------------------------------------------

    def _run_qc(
        self,
        *,
        cfg: OfflineAugMaisiRoutineConfig,
        image_aug_h5: Path,
        latent_aug_h5: Path,
        decoder: MaisiDecoder,
        figures_dir: Path,
    ) -> dict[str, Any]:
        """Sample N rows per variant, decode latents, score, render figures."""
        n_per_variant = cfg.qc.n_patients_per_variant
        device = decoder.handle.device
        rng = np.random.default_rng(cfg.seed)

        # Pick row indices per variant.
        with h5py.File(image_aug_h5, "r") as f:
            variants_arr = np.asarray(f["variants"][:], dtype=object)
            ids_arr = np.asarray(f["ids"][:], dtype=object)
            source_row_arr = np.asarray(f["source_row_index"][:], dtype=np.int64)
        rows_by_variant: dict[str, list[int]] = {v: [] for v in cfg.variants}
        for i, v in enumerate(variants_arr):
            v_str = v.decode() if isinstance(v, (bytes, bytearray)) else str(v)
            if v_str in rows_by_variant:
                rows_by_variant[v_str].append(i)
        sampled: dict[str, list[int]] = {}
        for variant, rows in rows_by_variant.items():
            if not rows:
                continue
            picks = rng.choice(rows, size=min(n_per_variant, len(rows)), replace=False)
            sampled[variant] = sorted(int(r) for r in picks)

        rows_for_figs: list[AugRoundtripRow] = []
        with (
            h5py.File(image_aug_h5, "r") as f_img,
            h5py.File(latent_aug_h5, "r") as f_lat,
            h5py.File(cfg.source_image_h5, "r") as f_src,
        ):
            for variant, row_indices in sampled.items():
                for row_idx in row_indices:
                    scan_id = ids_arr[row_idx]
                    scan_id_str = (
                        scan_id.decode()
                        if isinstance(scan_id, (bytes, bytearray))
                        else str(scan_id)
                    )
                    src_row = int(source_row_arr[row_idx])
                    crop_origin = tuple(int(v) for v in f_src["crop/origin"][src_row])
                    for slug in cfg.modalities:
                        # CLEAN source — boxed to the same (192,224,192) as the
                        # bank-builder did, then percentile-normalised.
                        src_native = np.asarray(f_src[f"images/{slug}"][src_row], dtype=np.float32)
                        src_boxed = _box_native_numpy(src_native, crop_origin)
                        src_normed = self._percentile_normalise_np(src_boxed, cfg)
                        # AUGMENTED — read what the bank-builder wrote.
                        aug_img = np.asarray(f_img[f"images/{slug}"][row_idx], dtype=np.float32)
                        aug_img_normed = self._percentile_normalise_np(aug_img, cfg)
                        # DECODED — D(E(aug)).
                        z_np = np.asarray(f_lat[f"latents/{slug}"][row_idx], dtype=np.float32)
                        decoded = self._decode_latent(z_np, decoder, device)
                        rows_for_figs.append(
                            AugRoundtripRow(
                                patient_id=scan_id_str,
                                cohort=cfg.cohort,
                                variant=variant,
                                modality=slug,
                                original=src_normed,
                                augmented=aug_img_normed,
                                decoded=decoded,
                            )
                        )
                # Render one figure per (cohort, variant) on the first sampled patient.
                first_patient_rows = [r for r in rows_for_figs if r.variant == variant][
                    : len(cfg.modalities)
                ]
                if first_patient_rows:
                    fig_name = cfg.qc.figure_filename_template.format(
                        cohort=cfg.cohort.replace("/", "_"),
                        variant=variant,
                    )
                    render_aug_roundtrip_figure(
                        first_patient_rows,
                        figures_dir / fig_name,
                        title=f"{cfg.cohort} — {variant}",
                    )

        aggregated = aggregate_cohort_variant_stats(rows_for_figs)
        gate = self._build_qc_gate(cfg)
        per_cell: dict[str, dict[str, Any]] = {}
        any_failed = False
        for (cohort, variant), stats in aggregated.items():
            agg = stats["aggregate"]
            psnr_med = agg["median_psnr_db"]
            ssim_med = agg["median_ssim"]
            floor_psnr = gate["psnr"] - cfg.qc.psnr_tolerance_db
            floor_ssim = gate["ssim"] - cfg.qc.ssim_tolerance
            status = "pass" if (psnr_med >= floor_psnr and ssim_med >= floor_ssim) else "fail"
            any_failed = any_failed or status == "fail"
            per_cell[f"{cohort}/{variant}"] = {
                **stats,
                "gate": {
                    "floor_psnr_db": floor_psnr,
                    "floor_ssim": floor_ssim,
                    "source_floor_psnr_db": gate["psnr"],
                    "source_floor_ssim": gate["ssim"],
                    "source": gate["source"],
                },
                "status": status,
            }
        return {
            "enabled": True,
            "n_patients_per_variant": n_per_variant,
            "per_cell": per_cell,
            "status": "fail" if any_failed else "pass",
            "figures_dir": str(figures_dir),
        }

    def _percentile_normalise_np(
        self,
        arr: np.ndarray,
        cfg: OfflineAugMaisiRoutineConfig,
    ) -> np.ndarray:
        """Apply the encoder's foreground-percentile normalisation as numpy."""
        x = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0).unsqueeze(0).float()
        normed = percentile_normalise(
            x,
            lower=cfg.percentile_lower,
            upper=cfg.percentile_upper,
            foreground_only=cfg.percentile_foreground_only,
        )
        return normed[0, 0].numpy()

    def _decode_latent(
        self,
        z_np: np.ndarray,
        decoder: MaisiDecoder,
        device: torch.device,
    ) -> np.ndarray:
        """Decode one ``(C, h, w, d)`` latent into the cropped box space ``(H, W, D)``."""
        z = torch.from_numpy(z_np).unsqueeze(0).to(device, dtype=torch.float32)
        crop_spec = CropPadSpec(
            crop_origin=(0, 0, 0),
            native_shape=AUG_IMAGE_CROP_BOX,
            target_shape=AUG_IMAGE_CROP_BOX,
        )
        with torch.inference_mode():
            decoded = decoder.decode(z, crop_spec=crop_spec)
        # decoded.image shape: (B, 1, H, W, D), already in [0, 1] post-VAE.
        img = decoded.image[0, 0].clamp(0.0, 1.0).detach().to("cpu", torch.float32).numpy()
        return img

    def _build_qc_gate(self, cfg: OfflineAugMaisiRoutineConfig) -> dict[str, Any]:
        """Resolve the PSNR/SSIM gate from the equivariance preflight decision."""
        psnr_default = cfg.qc.psnr_floor_default_db
        ssim_default = cfg.qc.ssim_floor_default
        if cfg.qc.equivariance_decision_path is None:
            return {"psnr": psnr_default, "ssim": ssim_default, "source": "default"}
        path = Path(cfg.qc.equivariance_decision_path)
        if not path.is_file():
            logger.warning("equivariance decision JSON not found at %s; using defaults", path)
            return {"psnr": psnr_default, "ssim": ssim_default, "source": "default"}
        payload = json.loads(path.read_text())
        floors = payload.get("vae_recon_floor", {}) or {}
        cohort_floor = floors.get(cfg.cohort)
        if cohort_floor is not None:
            return {
                "psnr": float(cohort_floor.get("median_psnr_db", psnr_default)),
                "ssim": float(cohort_floor.get("median_ssim", ssim_default)),
                "source": f"vae_recon_floor[{cfg.cohort}]",
            }
        # Cohort not in floor dict — fall back to the worst across the listed cohorts.
        if floors:
            psnrs = [v["median_psnr_db"] for v in floors.values() if "median_psnr_db" in v]
            ssims = [v["median_ssim"] for v in floors.values() if "median_ssim" in v]
            return {
                "psnr": float(min(psnrs)) if psnrs else psnr_default,
                "ssim": float(min(ssims)) if ssims else ssim_default,
                "source": "vae_recon_floor[min-of-known-cohorts]",
            }
        return {"psnr": psnr_default, "ssim": ssim_default, "source": "default"}

    # ----- merge (latent) ----------------------------------------------------

    def _merge_latent_shards(
        self,
        *,
        shards: list[Path],
        merged_path: Path,
        cohort: str,
        modalities: list[str],
        mask_channels: int,
        aug_config_sha256: str,
        overwrite: bool,
    ) -> Path:
        """Concatenate per-rank aug-latent H5s, validate, write."""
        if merged_path.exists() and not overwrite:
            raise FileExistsError(
                f"merged aug-latent already exists: {merged_path}. Pass overwrite=True."
            )
        if merged_path.exists():
            merged_path.unlink()

        total_rows = 0
        for s in shards:
            with h5py.File(s, "r") as fs:
                total_rows += int(fs["ids"].shape[0])

        # Copy schema-v2 root attrs from the first shard
        with h5py.File(shards[0], "r") as fs0:
            ref_aug_sha = str(fs0.attrs["aug_config_sha256"])

        if ref_aug_sha != aug_config_sha256:
            raise ValueError(
                f"shard aug_config_sha256 disagrees with the routine's: "
                f"{ref_aug_sha!r} != {aug_config_sha256!r}"
            )

        from vena.data.h5.augmented import build_aug_latent_manifest
        from vena.data.h5.latent_domain.manifest import LATENT_SPATIAL
        from vena.data.h5.shared import H5Writer

        manifest = build_aug_latent_manifest(cohort, modalities, mask_channels)
        with h5py.File(shards[0], "r") as fs0:
            extra_root_attrs = {
                k: fs0.attrs[k]
                for k in (
                    "split_role",
                    "longitudinal",
                    "label_system",
                    "crop_box",
                    "orientation",
                )
                if k in fs0.attrs
            }
            cfg_json = str(fs0.attrs["config_json"])
            source_aug_image_h5_path = str(fs0.attrs.get("source_aug_image_h5_path", ""))
            variants_json = str(fs0.attrs.get("variants_json", "[]"))

        with H5Writer(
            merged_path,
            manifest=manifest,
            config_json=cfg_json,
            producer=_PRODUCER,
            created_at=datetime.now(tz=UTC).isoformat(),
            git_sha=resolve_git_sha(),
            overwrite=overwrite,
            extra_root_attrs=extra_root_attrs,
        ) as w:
            f = w.file
            f.attrs["source_aug_image_h5_path"] = source_aug_image_h5_path
            f.attrs["aug_config_sha256"] = aug_config_sha256
            f.attrs["variants_json"] = variants_json
            f.attrs["merged_from"] = json.dumps([str(s) for s in shards])

            ids_dset = w.create_1d(manifest.get("ids"), n=total_rows)
            srci_dset = w.create_1d(manifest.get("source_row_index"), n=total_rows)
            variants_dset = w.create_1d(manifest.get("variants"), n=total_rows)
            params_dset = w.create_1d(manifest.get("aug_params_json"), n=total_rows)
            spatial = (LATENT_CHANNELS, *LATENT_SPATIAL)
            latent_dsets = {
                slug: w.create_stacked(
                    manifest.get(f"latents/{slug}"),
                    n=total_rows,
                    spatial_shape=spatial,
                )
                for slug in modalities
            }
            mask_spatial = (mask_channels, *LATENT_SPATIAL)
            mask_dset = w.create_stacked(
                manifest.get("masks/tumor_latent"),
                n=total_rows,
                spatial_shape=mask_spatial,
            )

            out_row = 0
            for s in shards:
                with h5py.File(s, "r") as fs:
                    n = int(fs["ids"].shape[0])
                    if n == 0:
                        continue
                    end = out_row + n
                    ids_dset[out_row:end] = np.asarray(fs["ids"][:], dtype=object)
                    srci_dset[out_row:end] = fs["source_row_index"][:]
                    variants_dset[out_row:end] = np.asarray(fs["variants"][:], dtype=object)
                    params_dset[out_row:end] = np.asarray(fs["aug_params_json"][:], dtype=object)
                    for slug in modalities:
                        latent_dsets[slug][out_row:end] = fs[f"latents/{slug}"][:]
                    mask_dset[out_row:end] = fs["masks/tumor_latent"][:]
                    out_row = end

        # Add the aug-image SHA cross-reference now that we have it.
        merged_image = self._merged_path(Path(source_aug_image_h5_path))
        if merged_image.exists():
            with h5py.File(merged_path, "r+") as f:
                f.attrs["source_aug_image_h5_sha256"] = sha256_file(merged_image)

        assert_aug_latent_h5_valid(merged_path, cohort, modalities, mask_channels)
        return merged_path

    # ----- report.md ---------------------------------------------------------

    @staticmethod
    def _write_report(run_dir: Path, decision: dict[str, Any]) -> None:
        cohort = decision["cohort"]
        rank = decision["rank"]
        ws = decision["world_size"]
        qc = decision.get("qc", {}) or {}
        per_cell = qc.get("per_cell", {}) or {}
        lines: list[str] = [
            f"# Offline aug bank — {cohort} (rank {rank}/{ws})\n",
            "",
            f"- Image-aug H5: `{decision['image_aug_h5_path']}`",
            f"- Latent-aug H5: `{decision['latent_aug_h5_path']}`",
            f"- Aug config SHA-256: `{decision['aug_config_sha256']}`",
            f"- Bank build: {decision['wall_clock_seconds']['bank_build']:.1f} s",
            f"- Encode: {decision['wall_clock_seconds']['encode']:.1f} s",
            "",
            "## QC roundtrip (PSNR/SSIM of D(E(aug)) vs aug)",
            "",
            "| Cohort | Variant | n | Median PSNR (dB) | Median SSIM | Status |",
            "|---|---|---|---|---|---|",
        ]
        for key, cell in per_cell.items():
            agg = cell.get("aggregate", {})
            lines.append(
                f"| {key.split('/')[0]} | {key.split('/')[-1]} "
                f"| {agg.get('n_observations', 0)} "
                f"| {agg.get('median_psnr_db', 0):.2f} "
                f"| {agg.get('median_ssim', 0):.3f} "
                f"| {cell.get('status', 'unknown')} |"
            )
        (run_dir / "report.md").write_text("\n".join(lines))
