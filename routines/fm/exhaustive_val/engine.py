"""Exhaustive image-space validation engine (standalone, second-GPU).

Run as a subprocess by the training run's ``ExhaustiveValLauncher`` callback
(or manually). Given an EMA-weights snapshot and the run config, it:

1. Rebuilds the trunk + ControlNet (identical to training) and loads the EMA
   shadow weights from the snapshot — so sampling matches the training EMA.
2. For each of ``n_patients`` validation patients and each ``nfe_levels`` entry:
   samples the predicted T1c latent (timed), decodes to image space (timed,
   cropped to the native 155-slice depth), loads the patient's real T1c from
   the image-domain H5 normalised *exactly* as the encoder input, and computes
   whole-volume PSNR/SSIM plus whole-volume latent MSE/L1/cosine.
3. Writes ``metrics.csv``, ``timing.csv``, ``latent_preds.h5``, and two
   qualitative panels (best/worst patient by mean SSIM across NFE levels).

Everything runs on ``device`` (default ``cuda:1``) so training continues
uninterrupted on the primary GPU.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F  # noqa: N812
import yaml
from pydantic import BaseModel, ConfigDict, Field

from vena.common import CropPadSpec, MaisiDecoder, load_autoencoder
from vena.common.decode import decode_box
from vena.model.fm.eval import (
    full_volume_psnr_ssim,
    render_comparison_figure,
    select_content_slices,
    write_latent_preds_h5,
)
from vena.model.fm.eval.exhaustive import build_crop_spec_from_h5, load_real_t1c_box
from vena.model.fm.inference import get_sampler
from vena.model.fm.lightning import FMLightningModule, LatentH5Dataset
from vena.model.fm.maisi.config import TrunkConfig
from vena.model.fm.metrics import ImageMetrics, LatentMetrics
from vena.validation.metrics_paired import ms_ssim_brain as _ms_ssim_brain
from vena.validation.metrics_paired import ms_ssim_wt_bbox as _ms_ssim_wt_bbox

logger = logging.getLogger(__name__)


def _should_write_latent_preds(pass_count: int, every_n: int) -> bool:
    """Return True if latent_preds.h5 should be written at this cadence pass.

    Fires on passes 1, 1+every_n, 1+2*every_n, … (1-indexed so the first
    cadence epoch always writes).  ``every_n <= 0`` means always write.

    Parameters
    ----------
    pass_count : int
        1-indexed cadence-pass counter populated by the launcher.
    every_n : int
        Write interval in cadence passes.  0 disables gating.
    """
    if every_n <= 0:
        return True
    return (pass_count - 1) % every_n == 0


class ExhaustiveValConfigError(Exception):
    """Raised on a malformed exhaustive-validation job config."""


class ExhaustiveValAllSkippedError(Exception):
    """Raised when every patient in an exhaustive-val epoch was skipped.

    A 100 % skip rate indicates a wiring or schema error (e.g. the
    ConditioningAssembler cannot find a batch key that the dataset was not
    configured to serve), not a transient per-patient data problem.  The engine
    raises rather than logging "complete" and leaving a header-only
    ``metrics.csv`` that looks like a successful run.
    """


# Skip fraction at which the patient-loop logs at ERROR (not yet a hard raise).
# A fraction below this produces a WARNING per skipped patient; at or above it
# an additional ERROR summarises the damage before the run continues.
_SKIP_FRACTION_ERROR_THRESHOLD: float = 0.5


class _TrunkJobCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    checkpoint: Path
    arch_json: Path | None = None
    arch_overrides: dict[str, Any] = Field(default_factory=dict)
    class_token: int = 9
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0)
    trainable: bool = True
    # Mirrors the train-time TrunkConfig fields so PEFT-trained runs can
    # reconstruct the wrapped trunk before loading the EMA snapshot.
    regime: Literal["fft", "peft"] = "fft"
    peft: dict[str, Any] | None = None
    # S1 v3 (2026-06-22) — input-concat conditioning at trunk's first conv.
    # Mirrors train-time _InputConcatCfg fields verbatim. Disabled (default)
    # is byte-identical to S1 v2 behaviour.
    input_concat: dict[str, Any] = Field(default_factory=dict)


class _CNJobCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # S1 v3 Variant A: ``enabled=False`` skips ControlNet entirely. Defaults
    # to ``True`` for back-compat with S1 v2 job YAMLs.
    enabled: bool = True
    init_from_trunk: bool = True
    conditioning_inputs: list[str] = Field(default_factory=list)
    arch_overrides: dict[str, Any] = Field(default_factory=dict)


class ExhaustiveValJobConfig(BaseModel):
    """Self-contained job spec written by the launcher (one YAML, one run).

    Supports two data paths:

    * **Single-cohort (legacy)**: set ``latents_h5`` + ``image_h5``.
    * **Multi-cohort**: set ``corpus_registry``; the engine iterates all cohorts
      (cv + test) using each cohort's own ``latent_h5`` and ``image_h5``.

    Exactly one of ``corpus_registry`` or ``latents_h5`` must be provided.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    epoch: int
    stage: str = "s1"
    seed: int = 1337

    trunk: _TrunkJobCfg
    controlnet: _CNJobCfg
    vae_checkpoint: Path
    rflow: dict[str, Any] = Field(default_factory=dict)
    ema: dict[str, Any] = Field(default_factory=dict)

    # Single-cohort (legacy) path — mutually exclusive with corpus_registry.
    latents_h5: Path | None = None
    image_h5: Path | None = None
    # Multi-cohort path — mutually exclusive with latents_h5 / image_h5.
    corpus_registry: Path | None = None

    fold: int = 0
    # S2 T-13 (task-20): soft 2-channel mask serving policy — must mirror
    # ``data.mask_source`` from the training config so the exhaustive-val
    # dataset serves the same batch keys the ConditioningAssembler expects.
    # Default ``"none"`` preserves back-compat with job YAMLs written before
    # this field was introduced.
    mask_source: str = "none"

    # S1 v3 Variant A (no ControlNet) has no CN EMA shadow to load; the
    # launcher writes a ``None`` here and the sub-process skips the load.
    ema_snapshot: Path | None = None
    # Unfrozen-trunk ablation: EMA trunk shadow saved by the launcher. When set
    # (and ``trunk.trainable``), sampling uses the fine-tuned trunk; otherwise
    # the original frozen trunk checkpoint is used.
    trunk_finetuned_snapshot: Path | None = None
    nfe_levels: list[int] = Field(default_factory=lambda: [1, 2, 5, 10, 20])
    integrator: str = "euler"
    n_patients: int = 20
    device: str = "cuda:1"
    output_dir: Path
    figure_n_slices: int = 10
    figure_slice_offset: int = 10
    # How many top-best / top-worst patients to render per epoch. Default 3
    # gives 6 panels: ``figure_best_{1,2,3}.png`` + ``figure_worst_{1,2,3}.png``.
    # Capped at half the patient count so a tiny cohort cannot have its top-K
    # and bottom-K overlap.
    figure_top_k: int = 3
    # S3 — per-block real-vs-synth decoder-feature figure for the top-K best
    # and worst patients. Off by default. Activated by K=2 picasso YAMLs;
    # K=5 keeps it off to avoid OOM on the val GPU.
    export_per_block_figures: bool = False
    # Block indices to read features from (subset of the trainer's LPL `A`).
    # Required (and only consumed) when ``export_per_block_figures`` is True.
    # The launcher populates this from the training config so the val job
    # matches the trainer's readout depth.
    lpl_A: list[int] = Field(default_factory=list)  # noqa: N815 — LPL block-index list; name is load-bearing
    # Path to the frozen MAISI VAE checkpoint. Required when
    # ``export_per_block_figures`` is True (the decoder is loaded on
    # ``device`` to extract per-block features).
    vae_checkpoint: Path | None = None
    # B12 (2026-07-29): write latent_preds.h5 only every N cadence passes to
    # avoid filling the experiment dir with 2–3 GB H5s each epoch. Pass 1 always
    # writes (pass_count=1 default).  Launcher increments latent_preds_pass_count
    # and writes it into the job YAML; the engine reads it here.
    # ``latent_preds_every_n <= 0`` disables the gate (always write).
    latent_preds_every_n: int = 4
    latent_preds_pass_count: int = 1

    @classmethod
    def from_yaml(cls, path: Path | str) -> ExhaustiveValJobConfig:
        with Path(path).open("r") as f:
            return cls.model_validate(yaml.safe_load(f))


class ExhaustiveValEngine:
    """Runs the full image-space validation for one snapshot/epoch."""

    def __init__(self, cfg: ExhaustiveValJobConfig) -> None:
        self.cfg = cfg
        self.device = self._resolve_device(cfg.device)
        # Per-patient ground-truth T1c latent — populated by
        # ``_process_patient`` so ``_render_per_block_features`` can extract
        # decoder features on both the real and the predicted latent. Empty
        # when ``export_per_block_figures`` is False.
        self._real_latents_by_pid: dict[str, torch.Tensor] = {}

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if not torch.cuda.is_available():
            logger.warning("CUDA unavailable; exhaustive validation falls back to CPU.")
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

    def _build_module(self) -> FMLightningModule:
        cfg = self.cfg
        trunk_cfg = TrunkConfig(
            checkpoint=cfg.trunk.checkpoint,
            arch_json=cfg.trunk.arch_json,
            arch_overrides=cfg.trunk.arch_overrides,
            class_token=cfg.trunk.class_token,
            spacing_mm=cfg.trunk.spacing_mm,
            trainable=cfg.trunk.trainable,
            regime=cfg.trunk.regime,
            peft=cfg.trunk.peft,
        )
        module = FMLightningModule(
            trunk_config=trunk_cfg,
            conditioning_specs=list(cfg.controlnet.conditioning_inputs),
            stage=cfg.stage.upper() if cfg.stage.startswith("s") else cfg.stage,
            controlnet_arch_overrides=cfg.controlnet.arch_overrides,
            rflow_cfg=dict(cfg.rflow),
            ema_cfg=dict(cfg.ema),
            region_resolver=None,
            vae_decoder=None,
            # S1 v3 — propagate the architecture flags so Variant A sub-jobs
            # rebuild the module with controlnet_enabled=False (and skip the
            # ConditioningAssembler) and the trunk-side input-concat.
            controlnet_enabled=cfg.controlnet.enabled,
            controlnet_init_from_trunk_enabled=cfg.controlnet.init_from_trunk,
            input_concat_cfg=cfg.trunk.input_concat,
        )
        module = module.to(self.device)
        module.setup()
        if cfg.ema_snapshot is not None and module.ema is not None:
            self._load_ema_snapshot(module, cfg.ema_snapshot)
        elif cfg.ema_snapshot is None and module.ema is None:
            logger.info(
                "S1 v3 Variant A: skipping CN EMA snapshot load (controlnet.enabled=false)."
            )
        else:
            logger.warning(
                "ema_snapshot=%s but module.ema is %s — mismatched. Skipping load.",
                cfg.ema_snapshot,
                module.ema,
            )
        if cfg.trunk.trainable and cfg.trunk_finetuned_snapshot is not None:
            self._load_trunk_ema_snapshot(module, cfg.trunk_finetuned_snapshot)
        module.eval()
        return module

    def _load_ema_snapshot(self, module: FMLightningModule, snapshot: Path) -> None:
        state = torch.load(snapshot, map_location=self.device, weights_only=True)
        # The launcher saves the EMA *shadow* model state_dict directly.
        module.ema.ema_model.load_state_dict(state)
        logger.info("loaded EMA snapshot %s", snapshot)

    def _load_trunk_ema_snapshot(self, module: FMLightningModule, snapshot: Path) -> None:
        """Load the fine-tuned trunk EMA shadow (unfrozen-trunk ablation).

        ``setup()`` builds ``module.trunk_ema`` from the *original* trunk
        checkpoint; here we overwrite its shadow with the training-time EMA
        snapshot so ``_make_ema_call`` samples with the fine-tuned trunk.
        """
        if module.trunk_ema is None:
            raise ExhaustiveValConfigError(
                "trunk_finetuned_snapshot was provided but the module has no trunk EMA "
                "(trunk.trainable must be true)"
            )
        state = torch.load(snapshot, map_location=self.device, weights_only=True)
        module.trunk_ema.ema_model.load_state_dict(state)
        logger.info("loaded fine-tuned trunk EMA snapshot %s", snapshot)

    def _val_patient_ids(self) -> list[str]:
        """Return validation patient IDs from the single-cohort latent H5."""
        import h5py

        assert self.cfg.latents_h5 is not None, "_val_patient_ids requires latents_h5"
        with h5py.File(self.cfg.latents_h5, "r") as f:
            ids = [
                b.decode() if isinstance(b, bytes) else str(b)
                for b in f[f"splits/cv/fold_{self.cfg.fold}/val"][:]
            ]
        return ids[: self.cfg.n_patients]

    @staticmethod
    def _split_n_patients(total: int, n_cohorts: int) -> list[int]:
        """Distribute a total patient budget uniformly across cohorts.

        The remainder ``total % n_cohorts`` is allocated one-extra-each to the
        first cohorts in registry order, so allocations are deterministic and
        stable across epochs / re-runs.

        Examples
        --------
        >>> ExhaustiveValEngine._split_n_patients(10, 2)
        [5, 5]
        >>> ExhaustiveValEngine._split_n_patients(10, 3)
        [4, 3, 3]
        """
        if n_cohorts <= 0:
            return []
        base, rem = divmod(int(total), int(n_cohorts))
        return [base + (1 if i < rem else 0) for i in range(n_cohorts)]

    def _cohort_val_patients(self, latents_h5: Path, budget: int, seed: int) -> list[str]:
        """Return up to ``budget`` validation scan IDs from a cohort's latent H5.

        Reads patient keys from ``splits/cv/fold_<fold>/val`` (matching the
        training data module's validation split), draws
        ``min(budget, |val|)`` keys with a seeded RNG, then expands each
        patient to its scan-level IDs via the CSR layout — so longitudinal
        cohorts contribute every scan of a selected patient.
        """
        import h5py
        import numpy as np

        with h5py.File(latents_h5, "r") as f:

            def _decode(ds) -> list[str]:  # type: ignore[no-untyped-def]
                return [b.decode() if isinstance(b, bytes) else str(b) for b in ds[:]]

            # cv cohorts carry `splits/cv/fold_<fold>/val`; test-only cohorts
            # carry only `splits/test` post-2026-06-19 schema unification (the
            # legacy `splits/cv/fold_0/val` alias on test-only cohorts was
            # dropped by normalize_splits). Try the cv path first, fall back
            # to the test pool for the OOD evaluation case.
            cv_key = f"splits/cv/fold_{self.cfg.fold}/val"
            if cv_key in f:
                val_patient_keys = _decode(f[cv_key])
            elif "splits/test" in f:
                val_patient_keys = _decode(f["splits/test"])
            else:
                raise KeyError(f"{latents_h5}: neither {cv_key!r} nor 'splits/test' present")
            offsets = f["patients/offsets"][:]
            csr_keys = _decode(f["patients/keys"])
            all_ids = _decode(f["ids"])

        n_pick = min(int(budget), len(val_patient_keys))
        if n_pick <= 0:
            return []

        rng = np.random.default_rng(int(seed))
        chosen_idx = sorted(
            int(i) for i in rng.choice(len(val_patient_keys), size=n_pick, replace=False)
        )
        chosen_patients = [val_patient_keys[i] for i in chosen_idx]

        key_to_pos = {k: i for i, k in enumerate(csr_keys)}
        scan_ids: list[str] = []
        for pk in chosen_patients:
            if pk not in key_to_pos:
                continue
            pos = key_to_pos[pk]
            start, end = int(offsets[pos]), int(offsets[pos + 1])
            for row in range(start, end):
                scan_ids.append(all_ids[row])
        return scan_ids

    # ------------------------------------------------------------------

    def run(self) -> Path:
        cfg = self.cfg
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "exhaustive-val epoch=%d device=%s nfe=%s n_patients=%d -> %s",
            cfg.epoch,
            self.device,
            cfg.nfe_levels,
            cfg.n_patients,
            out_dir,
        )

        module = self._build_module()
        vae = MaisiDecoder(handle=load_autoencoder(cfg.vae_checkpoint, device=str(self.device)))
        latent_metrics = LatentMetrics()
        image_metrics = ImageMetrics(data_range=1.0)
        # When the trained scheduler uses MONAI's SD3-style timestep transform
        # (``rflow.use_timestep_transform=True``), the sampler's
        # ``set_timesteps`` call needs ``input_img_size_numel`` or the
        # internal :func:`timestep_transform` divides ``None`` by ``int`` and
        # every per-patient sample fails with a silent WARNING. Plumb the
        # YAML's ``rflow.base_img_size_numel`` through as the inference-side
        # value; per-patient brain-box shape variations are sub-10 % deltas
        # in the cubic-root ratio used by ``timestep_transform``.
        sampler_kwargs: dict[str, Any] = {"scheduler": module.rflow.scheduler}
        if cfg.rflow.get("use_timestep_transform"):
            ts_numel = cfg.rflow.get("base_img_size_numel")
            if ts_numel is None:
                ts_numel = 48 * 56 * 48  # VENA brain-box latent (48x56x48)
            sampler_kwargs["input_img_size_numel"] = int(ts_numel)
        sampler = get_sampler(cfg.integrator)(**sampler_kwargs)

        metric_rows: list[dict[str, Any]] = []
        latent_entries: list[tuple[str, int, Any]] = []
        latents_by_pid_nfe: dict[tuple[str, int], torch.Tensor] = {}
        ssim_by_pid: dict[str, list[float]] = {}
        gen_decode_time: dict[tuple[str, int], float] = {}
        # Maps each patient_id to its image H5 path for _render_best_worst.
        pid_to_image_h5: dict[str, Path] = {}

        if cfg.corpus_registry is not None:
            self._run_multi_cohort(
                module=module,
                sampler=sampler,
                vae=vae,
                latent_metrics=latent_metrics,
                image_metrics=image_metrics,
                metric_rows=metric_rows,
                latent_entries=latent_entries,
                latents_by_pid_nfe=latents_by_pid_nfe,
                ssim_by_pid=ssim_by_pid,
                gen_decode_time=gen_decode_time,
                pid_to_image_h5=pid_to_image_h5,
            )
        else:
            self._run_single_cohort(
                module=module,
                sampler=sampler,
                vae=vae,
                latent_metrics=latent_metrics,
                image_metrics=image_metrics,
                metric_rows=metric_rows,
                latent_entries=latent_entries,
                latents_by_pid_nfe=latents_by_pid_nfe,
                ssim_by_pid=ssim_by_pid,
                gen_decode_time=gen_decode_time,
                pid_to_image_h5=pid_to_image_h5,
            )

        # Guard: if zero patients produced metric rows the run produced nothing
        # useful — a header-only CSV looks indistinguishable from a successful
        # empty run and masks wiring errors (e.g. mask_source mismatch).
        if not metric_rows:
            raise ExhaustiveValAllSkippedError(
                f"exhaustive-val epoch={cfg.epoch}: every patient was skipped — "
                "no metric rows collected. Check that conditioning wiring is "
                f"consistent: mask_source={cfg.mask_source!r} must match the "
                "conditioning_inputs specs written in the job YAML."
            )

        self._write_metrics_csv(out_dir / "metrics.csv", metric_rows)
        self._write_timing_csv(out_dir / "timing.csv", metric_rows)
        # S1 v3 (2026-06-22): per-(cohort, nfe, region) aggregate.csv summarises
        # the row-level metrics.csv for downstream convergence plots. NaN cells
        # are skipped from the mean/std so patients with empty regions (e.g.
        # 17 % of UCSF-PDGM val has no latent WT voxels) do not pull the
        # average to NaN.
        self._write_aggregate_csv(out_dir / "aggregate.csv", metric_rows)
        # B1 (2026-07-29): patient-mean then cohort-mean aggregate restricted to
        # cv cohorts. Used by select_checkpoint.py and the post-hoc monitor.
        self._write_aggregate_cv_csv(out_dir / "aggregate_cv.csv", metric_rows)
        # B12 (2026-07-29): latent_preds.h5 is 2-3 GB per epoch; gate writes to
        # every N cadence passes to avoid filling the experiment dir.
        if _should_write_latent_preds(cfg.latent_preds_pass_count, cfg.latent_preds_every_n):
            write_latent_preds_h5(
                out_dir / "latent_preds.h5",
                latent_entries,
                epoch=cfg.epoch,
                run_id=cfg.run_id,
                extra_attrs={"nfe_levels_json": json.dumps(list(cfg.nfe_levels))},
            )
        else:
            logger.info(
                "latent_preds.h5 skipped (pass_count=%d, every_n=%d)",
                cfg.latent_preds_pass_count,
                cfg.latent_preds_every_n,
            )
        # Per-(patient, NFE) PSNR/SSIM index for the comparison figure's
        # per-row annotation (2026-06-20 global figure overhaul). Built once
        # from ``metric_rows`` and passed down to ``_render_best_worst``.
        psnr_ssim_by_pid_nfe: dict[tuple[str, int], tuple[float, float]] = {
            (str(row["patient_id"]), int(row["nfe"])): (
                float(row.get("psnr_db", float("nan"))),
                float(row.get("ssim", float("nan"))),
            )
            for row in metric_rows
            if "patient_id" in row and "nfe" in row
        }
        self._render_best_worst(
            module,
            vae,
            ssim_by_pid,
            latents_by_pid_nfe,
            gen_decode_time,
            pid_to_image_h5,
            out_dir,
            psnr_ssim_by_pid_nfe=psnr_ssim_by_pid_nfe,
        )
        if cfg.export_per_block_figures and cfg.lpl_A:
            try:
                self._render_per_block_features(
                    vae=vae,
                    ssim_by_pid=ssim_by_pid,
                    latents_by_pid_nfe=latents_by_pid_nfe,
                    out_dir=out_dir,
                )
            except (RuntimeError, ValueError) as exc:
                logger.warning("per-block feature render skipped: %s", exc)
            finally:
                self._real_latents_by_pid.clear()
        logger.info("exhaustive-val epoch=%d complete -> %s", cfg.epoch, out_dir)
        return out_dir

    def _run_single_cohort(
        self,
        *,
        module: FMLightningModule,
        sampler: Any,
        vae: MaisiDecoder,
        latent_metrics: LatentMetrics,
        image_metrics: ImageMetrics,
        metric_rows: list[dict[str, Any]],
        latent_entries: list[tuple[str, int, Any]],
        latents_by_pid_nfe: dict[tuple[str, int], torch.Tensor],
        ssim_by_pid: dict[str, list[float]],
        gen_decode_time: dict[tuple[str, int], float],
        pid_to_image_h5: dict[str, Path],
    ) -> None:
        """Legacy single-cohort path: one latents_h5 + one image_h5."""
        import h5py

        cfg = self.cfg
        assert cfg.latents_h5 is not None and cfg.image_h5 is not None

        # Resolve cohort name from H5 root attr (fall back to "unknown").
        with h5py.File(cfg.latents_h5, "r") as f:
            cohort_name = str(f.attrs.get("cohort", "unknown"))

        patient_ids = self._val_patient_ids()
        dataset = LatentH5Dataset(cfg.latents_h5, patient_ids, mask_source=cfg.mask_source)
        ssim_by_pid.update({pid: [] for pid in patient_ids})
        pid_to_image_h5.update({pid: cfg.image_h5 for pid in patient_ids})

        n_ok = 0
        n_skip = 0
        for i, pid in enumerate(patient_ids):
            try:
                self._process_patient(
                    i,
                    pid,
                    dataset,
                    module,
                    sampler,
                    vae,
                    latent_metrics,
                    image_metrics,
                    metric_rows,
                    latent_entries,
                    latents_by_pid_nfe,
                    ssim_by_pid,
                    gen_decode_time,
                    image_h5=cfg.image_h5,
                    cohort=cohort_name,
                )
                n_ok += 1
            except Exception as exc:
                logger.warning("exhaustive-val: patient '%s' failed (%s); skipping.", pid, exc)
                n_skip += 1
                continue
            logger.info("  [%d/%d] %s done", i + 1, len(patient_ids), pid)

        total = n_ok + n_skip
        if n_skip > 0 and total > 0 and n_skip / total >= _SKIP_FRACTION_ERROR_THRESHOLD:
            logger.error(
                "exhaustive-val single-cohort: %.0f%% of patients skipped (%d/%d); "
                "check conditioning wiring (mask_source=%r matches conditioning_inputs?).",
                100.0 * n_skip / total,
                n_skip,
                total,
                cfg.mask_source,
            )

    def _run_multi_cohort(
        self,
        *,
        module: FMLightningModule,
        sampler: Any,
        vae: MaisiDecoder,
        latent_metrics: LatentMetrics,
        image_metrics: ImageMetrics,
        metric_rows: list[dict[str, Any]],
        latent_entries: list[tuple[str, int, Any]],
        latents_by_pid_nfe: dict[tuple[str, int], torch.Tensor],
        ssim_by_pid: dict[str, list[float]],
        gen_decode_time: dict[tuple[str, int], float],
        pid_to_image_h5: dict[str, Path],
    ) -> None:
        """Multi-cohort path: split ``n_patients`` uniformly across cohorts.

        ``cfg.n_patients`` is the TOTAL budget for this epoch. It is
        partitioned across all registered cohorts (cv + test-only) via
        :meth:`_split_n_patients` so that the total exhaustive-val cost stays
        constant as cohorts are added to the corpus. Each cohort's slice is
        drawn from its CV-fold validation split with a deterministic seed
        derived from ``cfg.seed + cohort_idx`` — so the same patients are
        re-evaluated every epoch and the metrics traces are comparable.
        """
        from vena.data.registry import load_registry

        cfg = self.cfg
        assert cfg.corpus_registry is not None

        registry = load_registry(cfg.corpus_registry)
        all_cohorts = registry.cv_cohorts() + registry.test_cohorts()
        # B2 (2026-07-29): tag each row with "cv" / "test_only" so aggregate_cv.csv
        # can restrict to cv cohorts only without post-hoc filtering on cohort names.
        cv_names: set[str] = {c.name for c in registry.cv_cohorts()}
        budgets = self._split_n_patients(int(cfg.n_patients), len(all_cohorts))
        logger.info(
            "exhaustive-val: cohort budgets %s (total=%d, n_cohorts=%d)",
            {c.name: b for c, b in zip(all_cohorts, budgets, strict=True)},
            cfg.n_patients,
            len(all_cohorts),
        )

        n_ok_total = 0
        n_skip_total = 0
        for cohort_idx, (cohort, budget) in enumerate(zip(all_cohorts, budgets, strict=True)):
            if budget <= 0:
                logger.info(
                    "exhaustive-val: cohort '%s' allocated 0 patients; skipping.",
                    cohort.name,
                )
                continue
            logger.info(
                "exhaustive-val: cohort '%s' budget=%d (seed=%d)",
                cohort.name,
                budget,
                int(cfg.seed) + cohort_idx,
            )
            patient_ids = self._cohort_val_patients(
                cohort.latent_h5, budget=budget, seed=int(cfg.seed) + cohort_idx
            )
            if not patient_ids:
                logger.warning(
                    "exhaustive-val: cohort '%s' yielded no patients; skipping.",
                    cohort.name,
                )
                continue
            dataset = LatentH5Dataset(cohort.latent_h5, patient_ids, mask_source=cfg.mask_source)
            ssim_by_pid.update({pid: [] for pid in patient_ids})
            pid_to_image_h5.update({pid: cohort.image_h5 for pid in patient_ids})

            # ``i`` indexes ``dataset`` (per-cohort, 0-based) so that
            # ``dataset[i]`` returns the row for ``pid``. The previous
            # implementation passed a global running counter here, which both
            # mislabelled the first cross-cohort patient and raised
            # ``IndexError`` on every subsequent one.
            cohort_role = "cv" if cohort.name in cv_names else "test_only"
            for i, pid in enumerate(patient_ids):
                try:
                    self._process_patient(
                        i,
                        pid,
                        dataset,
                        module,
                        sampler,
                        vae,
                        latent_metrics,
                        image_metrics,
                        metric_rows,
                        latent_entries,
                        latents_by_pid_nfe,
                        ssim_by_pid,
                        gen_decode_time,
                        image_h5=cohort.image_h5,
                        cohort=cohort.name,
                        cohort_role=cohort_role,
                    )
                    n_ok_total += 1
                except Exception as exc:
                    logger.warning(
                        "exhaustive-val: patient '%s' (cohort %s) failed (%s); skipping.",
                        pid,
                        cohort.name,
                        exc,
                    )
                    n_skip_total += 1
                    continue
                logger.info(
                    "  cohort=%s [%d/%d] %s done", cohort.name, i + 1, len(patient_ids), pid
                )

        grand_total = n_ok_total + n_skip_total
        if (
            n_skip_total > 0
            and grand_total > 0
            and (n_skip_total / grand_total >= _SKIP_FRACTION_ERROR_THRESHOLD)
        ):
            logger.error(
                "exhaustive-val multi-cohort: %.0f%% of patients skipped (%d/%d); "
                "check conditioning wiring (mask_source=%r matches conditioning_inputs?).",
                100.0 * n_skip_total / grand_total,
                n_skip_total,
                grand_total,
                cfg.mask_source,
            )

    def _process_patient(
        self,
        i: int,
        pid: str,
        dataset: LatentH5Dataset,
        module: FMLightningModule,
        sampler: Any,
        vae: MaisiDecoder,
        latent_metrics: LatentMetrics,
        image_metrics: ImageMetrics,
        metric_rows: list[dict[str, Any]],
        latent_entries: list[tuple[str, int, Any]],
        latents_by_pid_nfe: dict[tuple[str, int], torch.Tensor],
        ssim_by_pid: dict[str, list[float]],
        gen_decode_time: dict[tuple[str, int], float],
        *,
        image_h5: Path,
        cohort: str,
        cohort_role: str = "cv",
    ) -> None:
        """Process one patient: sample at each NFE, decode to box, compute metrics.

        Box-comparison path: the VAE decodes to box space ``(B,1,*target_shape)``
        and the real T1c is cropped to the same box via :func:`build_crop_spec_from_h5`
        + :func:`load_real_t1c_box` before PSNR/SSIM, ensuring intensity-space parity.
        """
        cfg = self.cfg
        item = dataset[i]
        batch = {
            k: (v.unsqueeze(0).to(self.device) if isinstance(v, torch.Tensor) else v)
            for k, v in item.items()
        }
        z_target = batch["z_t1c"]

        # Box path: build the crop spec from the image H5, then crop the real
        # T1c to the same box as the decoded prediction for a fair comparison.
        crop_spec = build_crop_spec_from_h5(image_h5, pid)
        real_box = load_real_t1c_box(image_h5, pid, crop_spec).to(self.device)

        module.compute_val_conditioning(batch)

        # WT + brain masks in image space (NN-upsampled from the latent masks
        # the dataset already loaded). Used for region-masked PSNR/SSIM below;
        # the decoded volume is already on the GPU so the extra metric calls
        # are near-free. ``m_brain_img`` is ``None`` for H5s that pre-date
        # `vena-encode-brain-to-latent`; the metric helper falls back to the
        # `real_box > 0` inference in that case.
        m_wt_img = self._wt_mask_in_image_space(batch, real_box.shape)
        m_brain_img = self._brain_mask_in_image_space(batch, real_box.shape)
        m_netc_img, m_ed_img, m_et_img = self._per_class_tumor_masks_in_image_space(
            batch, real_box.shape
        )
        brain_mask_source = "masks/brain_latent" if m_brain_img is not None else "real_box>0"
        # §15 / B2 admissibility guard: the real_box>0 fallback is derived from
        # the real T1c box and would contaminate ssim_brain used for model
        # selection.  A row with brain_mask_source=="real_box>0" silently leaks
        # target information into the selection metric — exactly the failure mode
        # §15 argues cannot be in the methods section.  Raise here so the job
        # fails loudly rather than producing poisoned aggregate_cv.csv rows.
        if m_brain_img is None:
            raise AssertionError(
                f"[§15 brain_mask_source guard] Patient {pid!r} has no "
                "'masks/brain_latent' dataset in the latent H5. "
                "The real_box>0 fallback is derived from the real T1c and would "
                "contaminate ssim_brain used for model selection. "
                "Re-encode the latent H5 with vena-encode-brain-to-latent "
                "before running exhaustive validation."
            )

        # Stash the real T1c latent for the per-block feature render. Cheap
        # (one (B=1,C,h,w,d) CPU tensor per scored patient); only retained
        # when ``export_per_block_figures`` is set.
        if cfg.export_per_block_figures:
            self._real_latents_by_pid[pid] = z_target.detach().to("cpu")

        for nfe in cfg.nfe_levels:
            z_pred, gen_t = self._sample(module, sampler, z_target, int(nfe))
            latents_by_pid_nfe[(pid, int(nfe))] = z_pred.detach().to("cpu")
            latent_entries.append((pid, int(nfe), z_pred[0].detach().cpu().numpy()))

            img_pred, dec_t = self._decode(vae, z_pred, crop_spec)
            psnr, ssim = full_volume_psnr_ssim(img_pred, real_box, image_metrics)
            psnr_wt, ssim_wt, psnr_bg, ssim_bg, psnr_nwt, ssim_nwt = self._region_psnr_ssim(
                img_pred, real_box, m_wt_img, image_metrics, m_brain_img=m_brain_img
            )
            # S1 v3 (2026-06-22) extra per-region metrics (ET / NETC / ED / BNWT
            # + MAE/MSE for every region + voxel counts). Missing m_tumor /
            # m_brain ⇒ those entries are NaN / 0.
            v3_extras = self._v3_per_region_metrics(
                img_pred,
                real_box,
                m_netc_img=m_netc_img,
                m_ed_img=m_ed_img,
                m_et_img=m_et_img,
                m_wt_img=m_wt_img,
                m_brain_img=m_brain_img,
                image_metrics=image_metrics,
            )

            mask = torch.ones_like(z_target, dtype=torch.bool)
            lat_mse = float(latent_metrics.mse(z_pred, z_target, mask)[0].item())
            lat_l1 = float(latent_metrics.l1(z_pred, z_target, mask)[0].item())
            lat_cos = float(latent_metrics.cosine(z_pred, z_target, mask)[0].item())

            ssim_by_pid[pid].append(ssim)
            gen_decode_time[(pid, int(nfe))] = gen_t + dec_t
            row: dict[str, Any] = {
                "epoch": int(cfg.epoch),
                "cohort": cohort,
                "role": cohort_role,  # B2 (2026-07-29)
                "patient_id": pid,
                "nfe": int(nfe),
                "psnr_db": psnr,
                "ssim": ssim,
                "psnr_db_wt": psnr_wt,
                "ssim_wt": ssim_wt,
                "psnr_db_bg": psnr_bg,
                "ssim_bg": ssim_bg,
                "psnr_db_nwt": psnr_nwt,
                "ssim_nwt": ssim_nwt,
                "latent_mse": lat_mse,
                "latent_l1": lat_l1,
                "latent_cosine": lat_cos,
                "gen_sec": gen_t,
                "decode_sec": dec_t,
                "brain_mask_source": brain_mask_source,
            }
            row.update(v3_extras)
            metric_rows.append(row)
            del img_pred

    # ------------------------------------------------------------------

    def _sample(
        self,
        module: FMLightningModule,
        sampler: Any,
        z_target: torch.Tensor,
        nfe: int,
    ) -> tuple[torch.Tensor, float]:
        model_call = module._make_ema_call()
        x0 = torch.randn_like(z_target)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            z_pred = sampler.sample(model_call, x0, num_inference_steps=int(nfe))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return z_pred, time.perf_counter() - t0

    def _decode(
        self,
        vae: MaisiDecoder,
        z_pred: torch.Tensor,
        crop_spec: CropPadSpec,
    ) -> tuple[torch.Tensor, float]:
        """Decode latent to the box volume using the crop_spec path.

        Delegates to :func:`vena.common.decode.decode_box` with CUDA-synced
        timing and ``[0, 1]`` clamp so the returned volume is metric-ready.
        """
        return decode_box(vae, z_pred, crop_spec, return_seconds=True)

    def _render_best_worst(
        self,
        module: FMLightningModule,
        vae: MaisiDecoder,
        ssim_by_pid: dict[str, list[float]],
        latents_by_pid_nfe: dict[tuple[str, int], torch.Tensor],
        gen_decode_time: dict[tuple[str, int], float],
        pid_to_image_h5: dict[str, Path],
        out_dir: Path,
        *,
        psnr_ssim_by_pid_nfe: dict[tuple[str, int], tuple[float, float]] | None = None,
    ) -> None:
        """Render ``figure_top_k`` best and worst patients by mean SSIM.

        With the default ``figure_top_k=3`` the artifact directory gains six
        panels per epoch (``figure_best_{1,2,3}.png`` and
        ``figure_worst_{1,2,3}.png``), where rank 1 is the most-best /
        most-worst case. ``k`` is clamped to half the number of scored patients
        so that a small cohort cannot produce overlapping best/worst lists.
        """
        mean_ssim = {pid: (sum(v) / len(v)) for pid, v in ssim_by_pid.items() if v}
        if not mean_ssim:
            logger.warning("no SSIM scores; skipping qualitative figures.")
            return
        k = max(1, min(int(self.cfg.figure_top_k), len(mean_ssim) // 2))
        if k < self.cfg.figure_top_k:
            logger.info(
                "figure_top_k clamped from %d to %d (only %d patients scored)",
                self.cfg.figure_top_k,
                k,
                len(mean_ssim),
            )
        sorted_desc = sorted(mean_ssim.items(), key=lambda kv: kv[1], reverse=True)
        best_picks = sorted_desc[:k]
        worst_picks = list(reversed(sorted_desc[-k:]))

        targets: list[tuple[str, str]] = [
            (f"best_{rank + 1}", pid) for rank, (pid, _) in enumerate(best_picks)
        ] + [(f"worst_{rank + 1}", pid) for rank, (pid, _) in enumerate(worst_picks)]
        for tag, pid in targets:
            image_h5 = pid_to_image_h5.get(pid)
            if image_h5 is None:
                logger.warning("_render_best_worst: no image_h5 for pid '%s'; skipping.", pid)
                continue
            crop_spec = build_crop_spec_from_h5(image_h5, pid)
            real = load_real_t1c_box(image_h5, pid, crop_spec).to(self.device)
            synth_by_nfe: dict[int, torch.Tensor] = {}
            time_by_nfe: dict[int, float] = {}
            for nfe in self.cfg.nfe_levels:
                z = latents_by_pid_nfe[(pid, int(nfe))].to(self.device)
                img, _ = self._decode(vae, z, crop_spec)
                synth_by_nfe[int(nfe)] = img.cpu()
                time_by_nfe[int(nfe)] = gen_decode_time[(pid, int(nfe))]
            slices = select_content_slices(
                real.cpu(), n_slices=self.cfg.figure_n_slices, offset=self.cfg.figure_slice_offset
            )
            psnr_ssim_for_pid: dict[int, tuple[float, float]] = {}
            if psnr_ssim_by_pid_nfe is not None:
                for nfe in self.cfg.nfe_levels:
                    psnr_ssim_for_pid[int(nfe)] = psnr_ssim_by_pid_nfe.get(
                        (pid, int(nfe)),
                        (float("nan"), float("nan")),
                    )
            render_comparison_figure(
                real.cpu(),
                synth_by_nfe,
                time_by_nfe,
                slices,
                patient_id=pid,
                title_tag=tag,
                out_path=out_dir / f"figure_{tag}.png",
                psnr_ssim_by_nfe=psnr_ssim_for_pid,
            )
            logger.info("wrote figure_%s.png (%s, mean SSIM=%.4f)", tag, pid, mean_ssim[pid])

    # ------------------------------------------------------------------

    def _render_per_block_features(
        self,
        *,
        vae: MaisiDecoder,
        ssim_by_pid: dict[str, list[float]],
        latents_by_pid_nfe: dict[tuple[str, int], torch.Tensor],
        out_dir: Path,
    ) -> None:
        """Render per-block real-vs-synth decoder-feature maps (S3 diagnostic).

        For each of the top-K best and bottom-K worst patients (by mean SSIM),
        for every block index in ``cfg.lpl_A``, extract decoder features from
        both the predicted latent (highest available NFE) and the real T1c
        latent via :func:`vena.model.fm.lpl.decoder_feature_extractor` (the
        same context manager ``training_step`` uses). Renders a 1×3
        matplotlib panel ``(real channel-mean, synth channel-mean, |diff|)``
        on the central axial slice. Also emits ``feature_maps.csv`` with the
        per-patient per-block feature-MSE (channel-mean) so the plot can be
        compared quantitatively across epochs.

        K=2 only on production runs — for K=5 the activation cost on the val
        GPU is prohibitive (cfg sets ``export_per_block_figures=false`` in
        that case).
        """
        import csv as _csv

        import matplotlib.pyplot as plt

        from vena.model.fm.lpl import decoder_feature_extractor

        if not self._real_latents_by_pid:
            logger.info("per-block feature render: no real latents cached; skipping.")
            return

        mean_ssim = {pid: (sum(v) / len(v)) for pid, v in ssim_by_pid.items() if v}
        if not mean_ssim:
            return
        k = max(1, min(int(self.cfg.figure_top_k), len(mean_ssim) // 2))
        sorted_desc = sorted(mean_ssim.items(), key=lambda kv: kv[1], reverse=True)
        best_picks = sorted_desc[:k]
        worst_picks = list(reversed(sorted_desc[-k:]))
        targets: list[tuple[str, str]] = [
            (f"best_{rank + 1}", pid) for rank, (pid, _) in enumerate(best_picks)
        ] + [(f"worst_{rank + 1}", pid) for rank, (pid, _) in enumerate(worst_picks)]

        blocks_sorted = sorted(int(b) for b in self.cfg.lpl_A)
        max_block = int(blocks_sorted[-1])
        blocks_fset = frozenset(blocks_sorted)
        # Highest NFE = best-quality prediction → diagnostic content is
        # maximised at top NFE rather than the noisy 1-step prediction.
        nfe_for_render = int(max(self.cfg.nfe_levels))
        fig_dir = out_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / "feature_maps.csv"
        rows: list[dict[str, Any]] = []

        for tag, pid in targets:
            real_lat = self._real_latents_by_pid.get(pid)
            pred_lat = latents_by_pid_nfe.get((pid, nfe_for_render))
            if real_lat is None or pred_lat is None:
                logger.warning(
                    "per-block render: missing latents for pid=%s (real=%s pred=%s)",
                    pid,
                    real_lat is None,
                    pred_lat is None,
                )
                continue
            real_lat = real_lat.to(self.device)
            pred_lat = pred_lat.to(self.device)

            # Use the canonical context manager (same one ``training_step``
            # uses) so ``post_quant_conv`` + the partial-decode hooks stay
            # in lock-step with the training path. The wrapper consumes
            # ``vae.handle`` (the AutoencoderHandle), not the bare
            # MaisiDecoder — ``post_quant_conv`` lives on the parent
            # autoencoder, not on the decoder submodule.
            with torch.no_grad():
                with decoder_feature_extractor(
                    vae.handle,
                    blocks=blocks_fset,
                    max_block=max_block,
                    grad_checkpoint=False,
                ) as extract:
                    phi_real = extract(real_lat)
                    phi_pred = extract(pred_lat)

            for blk in blocks_sorted:
                fr = phi_real[blk][0]  # (C, h, w, d)
                fp = phi_pred[blk][0]
                # Channel-mean projection (one scalar per voxel) — matches the
                # design-note §2.6 channel-mean panel.
                mean_real = fr.mean(dim=0)
                mean_pred = fp.mean(dim=0)
                diff = (mean_pred - mean_real).abs()
                # Quantitative: per-block feature MSE (channel-mean).
                mse = float((mean_pred - mean_real).pow(2).mean().item())
                rows.append(
                    {
                        "patient_id": pid,
                        "tag": tag,
                        "block": int(blk),
                        "nfe": nfe_for_render,
                        "feat_mse_channel_mean": mse,
                        "mean_ssim": mean_ssim.get(pid, float("nan")),
                    }
                )

                # Mid-axial slice (depth axis).
                d_idx = mean_real.shape[-1] // 2
                a = mean_real[..., d_idx].detach().cpu().numpy()
                b = mean_pred[..., d_idx].detach().cpu().numpy()
                c = diff[..., d_idx].detach().cpu().numpy()
                fig, axes = plt.subplots(1, 3, figsize=(10, 4))
                vmin = float(min(a.min(), b.min()))
                vmax = float(max(a.max(), b.max()))
                axes[0].imshow(a, cmap="gray", vmin=vmin, vmax=vmax)
                axes[0].set_title(f"real block {blk}")
                axes[0].axis("off")
                axes[1].imshow(b, cmap="gray", vmin=vmin, vmax=vmax)
                axes[1].set_title(f"synth block {blk} (NFE={nfe_for_render})")
                axes[1].axis("off")
                axes[2].imshow(c, cmap="magma")
                axes[2].set_title(f"|diff|  MSE={mse:.4f}")
                axes[2].axis("off")
                fig.suptitle(f"{tag}  {pid}", fontsize=10)
                fig.tight_layout()
                fig.savefig(fig_dir / f"block{blk}_{tag}.png", dpi=120)
                plt.close(fig)
            logger.info(
                "per-block render: pid=%s tag=%s blocks=%s (NFE=%d)",
                pid,
                tag,
                blocks_sorted,
                nfe_for_render,
            )

        if rows:
            with csv_path.open("w", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                for row in rows:
                    w.writerow(row)
            logger.info("wrote feature_maps.csv (%d rows)", len(rows))

    # ------------------------------------------------------------------

    def _wt_mask_in_image_space(
        self, batch: dict[str, Any], image_shape: tuple[int, ...]
    ) -> torch.Tensor | None:
        """NN-upsample the latent WT mask to the image box.

        Returns
        -------
        Tensor | None
            ``(B, 1, *image_shape[-3:])`` boolean mask, or ``None`` when the
            dataset did not supply ``m_wt`` for this patient. The boolean dtype
            matches what :class:`ImageMetrics` expects.
        """
        return self._upsample_latent_mask(batch.get("m_wt"), image_shape)

    def _brain_mask_in_image_space(
        self, batch: dict[str, Any], image_shape: tuple[int, ...]
    ) -> torch.Tensor | None:
        """NN-upsample the latent brain mask to the image box.

        Reads ``batch["m_brain"]`` which the dataset populates from the
        cohort's ``masks/brain_latent`` (max-pool 4 of the image-domain
        ``masks/brain``, produced by ``vena-encode-brain-to-latent``). Returns
        ``None`` when the H5 lacks the dataset; the caller then falls back
        to the ``real_box > 0`` skull-strip-foreground inference.
        """
        return self._upsample_latent_mask(batch.get("m_brain"), image_shape)

    @staticmethod
    def _upsample_latent_mask(mask: Any, image_shape: tuple[int, ...]) -> torch.Tensor | None:
        if mask is None:
            return None
        if not isinstance(mask, torch.Tensor):
            return None
        if mask.dim() == 4:
            # CSR datasets occasionally return (1,h,w,d); add the batch dim.
            mask = mask.unsqueeze(0)
        target_spatial = tuple(int(s) for s in image_shape[-3:])
        m_up = F.interpolate(mask.float(), size=target_spatial, mode="nearest")
        return m_up.bool()

    def _per_class_tumor_masks_in_image_space(
        self,
        batch: dict[str, Any],
        image_shape: tuple[int, ...],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """NN-upsample the 3-channel ``m_tumor`` to image space, per class.

        Returns ``(netc, ed, et)`` each shaped ``(B, 1, *image_shape[-3:])``,
        or ``(None, None, None)`` when ``batch["m_tumor"]`` is missing.
        Soft masks are binarised at threshold 0.5 (matches the loss / region
        resolver convention).
        """
        m_tumor = batch.get("m_tumor")
        if m_tumor is None or not isinstance(m_tumor, torch.Tensor):
            return None, None, None
        if m_tumor.dim() == 4:
            # (3,h,w,d) → (1,3,h,w,d) for the DataLoader-of-1 path.
            m_tumor = m_tumor.unsqueeze(0)
        target_spatial = tuple(int(s) for s in image_shape[-3:])
        m_up = F.interpolate(m_tumor.float(), size=target_spatial, mode="nearest")
        m_hard = m_up >= 0.5
        return (
            m_hard[:, 0:1],  # NETC
            m_hard[:, 1:2],  # ED
            m_hard[:, 2:3],  # ET
        )

    @staticmethod
    def _region_psnr_ssim(
        img_pred: torch.Tensor,
        real_box: torch.Tensor,
        m_wt_img: torch.Tensor | None,
        image_metrics: ImageMetrics,
        m_brain_img: torch.Tensor | None = None,
    ) -> tuple[float, float, float, float, float, float]:
        """Region-masked PSNR/SSIM for WT, BG, and NWT (healthy brain).

        Three regions are returned:

        * **wt** — whole-tumour mask (NN-upsampled from latent space).
        * **bg** — complement of ``wt`` (everything that is not tumour;
          historically named "background" but includes both outside-brain
          voxels and healthy brain). Kept under this semantic for
          backward-compat with existing CSV consumers.
        * **nwt** — healthy brain: precise brain mask (from
          ``masks/brain_latent`` when present in the H5, otherwise inferred
          from ``real_box > 0``) intersected with ``~wt``. The precise mask
          is preferred because the zero-foreground proxy mis-labels two
          small populations as non-brain: (i) dark CSF voxels with intensity
          exactly 0, (ii) near-zero halo around the skull-strip boundary.
          The CSV cell ``brain_mask_source`` records which path was taken
          per row.

        ``decode_box`` returns a 3-D ``(H, W, D)`` tensor and the whole-volume
        helper adds two leading dims internally; the masked metric helpers
        expect ``(B, C, H, W, D)``, so we promote both volume and mask here.
        Empty regions or a missing mask return ``nan`` for that pair so the
        CSV cell renders blank (downstream tooling reads ``""`` as NaN).
        """
        if m_wt_img is None or m_wt_img.numel() == 0:
            return (float("nan"),) * 6
        p = img_pred[None, None] if img_pred.ndim == 3 else img_pred
        r = real_box[None, None] if real_box.ndim == 3 else real_box
        wt = m_wt_img if m_wt_img.ndim == 5 else m_wt_img[None, None]
        bg = ~wt
        # Healthy brain: precise brain mask when available, otherwise
        # `real_box > 0` skull-strip-foreground inference.
        if m_brain_img is not None and m_brain_img.numel() > 0:
            brain = m_brain_img if m_brain_img.ndim == 5 else m_brain_img[None, None]
        else:
            brain = r > 0
            if brain.shape != wt.shape:
                # Defensive: only collapse when r had a multi-channel layout we did
                # not anticipate. The training pipeline always emits single-channel
                # boxes, so this branch is rarely taken.
                brain = brain.any(dim=1, keepdim=True)
        nwt = brain & ~wt

        psnr_wt = image_metrics.psnr(p, r, wt)
        ssim_wt = image_metrics.ssim(p, r, wt)
        psnr_bg = image_metrics.psnr(p, r, bg)
        ssim_bg = image_metrics.ssim(p, r, bg)
        psnr_nwt = image_metrics.psnr(p, r, nwt)
        ssim_nwt = image_metrics.ssim(p, r, nwt)

        def _first_finite(t: torch.Tensor) -> float:
            v = float(t[0].item()) if t.numel() else float("nan")
            return v if v == v else float("nan")  # NaN-passthrough

        return (
            _first_finite(psnr_wt),
            _first_finite(ssim_wt),
            _first_finite(psnr_bg),
            _first_finite(ssim_bg),
            _first_finite(psnr_nwt),
            _first_finite(ssim_nwt),
        )

    @staticmethod
    def _v3_per_region_metrics(
        img_pred: torch.Tensor,
        real_box: torch.Tensor,
        m_netc_img: torch.Tensor | None,
        m_ed_img: torch.Tensor | None,
        m_et_img: torch.Tensor | None,
        m_wt_img: torch.Tensor | None,
        m_brain_img: torch.Tensor | None,
        image_metrics: ImageMetrics,
    ) -> dict[str, float]:
        """Per-region PSNR/SSIM/MAE/MSE for the S1 v3 sub-regions.

        Returns a flat dict suitable for ``DictWriter.writerow``. Keys:

        * ``psnr_db_{et,netc,ed,bnwt}``
        * ``ssim_{et,netc,ed,bnwt}``
        * ``mae_{whole,wt,bg,bnwt,et,netc,ed}``
        * ``mse_{whole,wt,bg,bnwt,et,netc,ed}``
        * ``n_voxels_{brain,wt,bnwt,netc,ed,et}``

        Empty regions return NaN for PSNR/SSIM/MAE/MSE and 0 for the voxel
        count. ``brain`` (the "whole" region) is the precise brain mask
        when available, else the ``real_box > 0`` skull-strip-foreground
        inference, matching the convention in :meth:`_region_psnr_ssim`.
        """
        p = img_pred[None, None] if img_pred.ndim == 3 else img_pred
        r = real_box[None, None] if real_box.ndim == 3 else real_box

        # Whole-brain mask (alias "whole") and the disjoint brain_not_wt region.
        if m_brain_img is not None and m_brain_img.numel() > 0:
            brain = m_brain_img if m_brain_img.ndim == 5 else m_brain_img[None, None]
        else:
            brain = r > 0
        wt = (
            m_wt_img
            if m_wt_img is not None and m_wt_img.ndim == 5
            else (m_wt_img[None, None] if m_wt_img is not None else None)
        )
        bg = (~wt) if wt is not None else None
        bnwt = (brain & ~wt) if (wt is not None) else None

        def _scalar(t: torch.Tensor | None) -> float:
            if t is None or t.numel() == 0:
                return float("nan")
            v = float(t[0].item())
            return v if v == v else float("nan")

        out: dict[str, float] = {}

        def _emit(name: str, mask: torch.Tensor | None) -> None:
            if mask is None:
                out[f"psnr_db_{name}"] = float("nan")
                out[f"ssim_{name}"] = float("nan")
                out[f"mae_{name}"] = float("nan")
                out[f"mse_{name}"] = float("nan")
                out[f"n_voxels_{name}"] = 0
                return
            out[f"psnr_db_{name}"] = _scalar(image_metrics.psnr(p, r, mask))
            out[f"ssim_{name}"] = _scalar(image_metrics.ssim(p, r, mask))
            out[f"mae_{name}"] = _scalar(image_metrics.mae(p, r, mask))
            out[f"mse_{name}"] = _scalar(image_metrics.mse(p, r, mask))
            out[f"n_voxels_{name}"] = int(mask.sum().item())

        # ET / NETC / ED come from m_tumor; bnwt from brain & ~wt.
        _emit("et", m_et_img)
        _emit("netc", m_netc_img)
        _emit("ed", m_ed_img)
        _emit("bnwt", bnwt)
        # Whole-volume MAE/MSE (PSNR/SSIM already in the row from the
        # legacy code path) — emit only the MAE/MSE columns here.
        whole_mask = brain
        if whole_mask is not None:
            out["mae_whole"] = _scalar(image_metrics.mae(p, r, whole_mask))
            out["mse_whole"] = _scalar(image_metrics.mse(p, r, whole_mask))
            out["n_voxels_brain"] = int(whole_mask.sum().item())
            # B7 (2026-07-29) — brain-masked PSNR/SSIM to unblock ssim_brain monitor.
            out["psnr_db_brain"] = _scalar(image_metrics.psnr(p, r, whole_mask))
            out["ssim_brain"] = _scalar(image_metrics.ssim(p, r, whole_mask))
        else:
            out["mae_whole"] = float("nan")
            out["mse_whole"] = float("nan")
            out["n_voxels_brain"] = 0
            out["psnr_db_brain"] = float("nan")
            out["ssim_brain"] = float("nan")
        # B8 (2026-07-29) — intensity statistics for §18 contrast readout.
        et_bool = m_et_img.bool() if m_et_img is not None else None
        bnwt_bool = bnwt.bool() if bnwt is not None else None
        out["p995_pred_brain"] = (
            float(torch.quantile(p[whole_mask], 0.995).item())
            if whole_mask is not None and whole_mask.any()
            else float("nan")
        )
        out["p995_real_brain"] = (
            float(torch.quantile(r[whole_mask], 0.995).item())
            if whole_mask is not None and whole_mask.any()
            else float("nan")
        )
        out["mean_et_pred"] = (
            float(p[et_bool].mean().item())
            if et_bool is not None and et_bool.any()
            else float("nan")
        )
        out["mean_et_real"] = (
            float(r[et_bool].mean().item())
            if et_bool is not None and et_bool.any()
            else float("nan")
        )
        out["mean_bnwt_pred"] = (
            float(p[bnwt_bool].mean().item())
            if bnwt_bool is not None and bnwt_bool.any()
            else float("nan")
        )
        out["mean_bnwt_real"] = (
            float(r[bnwt_bool].mean().item())
            if bnwt_bool is not None and bnwt_bool.any()
            else float("nan")
        )
        # B13 (2026-07-29) — MS-SSIM (monai.metrics); NaN when WT bbox < 90 voxels.
        # ms_ssim_brain/wt_bbox expect 3-D numpy arrays (squeezed from the
        # (1,1,H,W,D) batched tensors used throughout this function).
        _p_np = p.squeeze(0).squeeze(0).float().cpu().numpy()
        _r_np = r.squeeze(0).squeeze(0).float().cpu().numpy()
        _brain_np = (
            whole_mask.squeeze(0).squeeze(0).cpu().numpy()
            if whole_mask is not None
            else (_r_np > 0)
        )
        out["ms_ssim_brain"] = _ms_ssim_brain(_p_np, _r_np, _brain_np)
        if wt is not None:
            _wt_np = wt.squeeze(0).squeeze(0).cpu().numpy()
            out["ms_ssim_wt_bbox"] = _ms_ssim_wt_bbox(_p_np, _r_np, _wt_np)
        else:
            out["ms_ssim_wt_bbox"] = float("nan")
        # WT and BG MAE/MSE (paired with the legacy PSNR/SSIM).
        if wt is not None:
            out["mae_wt"] = _scalar(image_metrics.mae(p, r, wt))
            out["mse_wt"] = _scalar(image_metrics.mse(p, r, wt))
            out["n_voxels_wt"] = int(wt.sum().item())
        else:
            out["mae_wt"] = float("nan")
            out["mse_wt"] = float("nan")
            out["n_voxels_wt"] = 0
        if bg is not None:
            out["mae_bg"] = _scalar(image_metrics.mae(p, r, bg))
            out["mse_bg"] = _scalar(image_metrics.mse(p, r, bg))
        else:
            out["mae_bg"] = float("nan")
            out["mse_bg"] = float("nan")
        return out

    # S1 v3 CSV column manifest. Order is the same on every row so a partially-
    # populated row (e.g. patient with no m_tumor) still writes blank cells in
    # the right places.
    _V3_EXTRA_COLS: tuple[str, ...] = (
        "psnr_db_et",
        "ssim_et",
        "psnr_db_netc",
        "ssim_netc",
        "psnr_db_ed",
        "ssim_ed",
        "psnr_db_bnwt",
        "ssim_bnwt",
        "mae_whole",
        "mae_wt",
        "mae_bg",
        "mae_bnwt",
        "mae_et",
        "mae_netc",
        "mae_ed",
        "mse_whole",
        "mse_wt",
        "mse_bg",
        "mse_bnwt",
        "mse_et",
        "mse_netc",
        "mse_ed",
        "n_voxels_brain",
        "n_voxels_wt",
        "n_voxels_bnwt",
        "n_voxels_netc",
        "n_voxels_ed",
        "n_voxels_et",
        # B7 (2026-07-29): brain-masked PSNR/SSIM
        "psnr_db_brain",
        "ssim_brain",
        # B8 (2026-07-29): intensity statistics
        "p995_pred_brain",
        "p995_real_brain",
        "mean_et_pred",
        "mean_et_real",
        "mean_bnwt_pred",
        "mean_bnwt_real",
        # B13 (2026-07-29): MS-SSIM
        "ms_ssim_brain",
        "ms_ssim_wt_bbox",
    )

    @classmethod
    def _write_metrics_csv(cls, path: Path, rows: list[dict[str, Any]]) -> None:
        cols = [
            "cohort",
            "role",  # B2 (2026-07-29): "cv" or "test_only"
            "epoch",
            "patient_id",
            "nfe",
            "psnr_db",
            "ssim",
            "psnr_db_wt",
            "ssim_wt",
            "psnr_db_bg",
            "ssim_bg",
            "psnr_db_nwt",
            "ssim_nwt",
            "latent_mse",
            "latent_l1",
            "latent_cosine",
            "gen_sec",
            "decode_sec",
            # 2026-06-09 overhaul: which mask supplied the brain region for the
            # `nwt` PSNR/SSIM cells in this row — `masks/brain_latent` (precise,
            # loaded from the latent H5) or `real_box>0` (skull-strip-foreground
            # inference, fallback for H5s pre-dating `vena-encode-brain-to-latent`).
            "brain_mask_source",
            *cls._V3_EXTRA_COLS,
        ]
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in cols})

    @staticmethod
    def _write_timing_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        import statistics

        by_nfe: dict[int, list[dict[str, Any]]] = {}
        for r in rows:
            by_nfe.setdefault(int(r["nfe"]), []).append(r)
        cols = [
            "nfe",
            "n_patients",
            "gen_sec_mean",
            "gen_sec_std",
            "decode_sec_mean",
            "decode_sec_std",
            "total_sec_mean",
        ]
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for nfe in sorted(by_nfe):
                rs = by_nfe[nfe]
                gen = [r["gen_sec"] for r in rs]
                dec = [r["decode_sec"] for r in rs]
                tot = [g + d for g, d in zip(gen, dec, strict=True)]
                w.writerow(
                    {
                        "nfe": nfe,
                        "n_patients": len(rs),
                        "gen_sec_mean": f"{statistics.fmean(gen):.6g}",
                        "gen_sec_std": f"{(statistics.stdev(gen) if len(gen) > 1 else 0.0):.6g}",
                        "decode_sec_mean": f"{statistics.fmean(dec):.6g}",
                        "decode_sec_std": f"{(statistics.stdev(dec) if len(dec) > 1 else 0.0):.6g}",
                        "total_sec_mean": f"{statistics.fmean(tot):.6g}",
                    }
                )

    @staticmethod
    def _write_aggregate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        """One row per (cohort, nfe, region) summarising the metrics.csv.

        For each region in
        ``("whole", "wt", "bg", "bnwt", "netc", "ed", "et")`` and each
        ``(cohort, nfe)`` group, emit mean/std (NaN-skipping) for
        ``psnr_db, ssim, mae, mse`` and the patient count.

        Per-region columns are read from the row by suffix. ``whole`` reads
        from ``psnr_db`` / ``ssim`` / ``mae_whole`` / ``mse_whole``; the
        other regions use ``psnr_db_<region>`` / ``ssim_<region>`` /
        ``mae_<region>`` / ``mse_<region>``. Missing or NaN cells are
        excluded from the mean (n_patients counts only the populated rows).
        """
        import math
        import statistics

        regions = ("whole", "wt", "bg", "bnwt", "netc", "ed", "et")

        def _metric_key(region: str, metric: str) -> str:
            if region == "whole" and metric in ("psnr_db", "ssim"):
                return metric  # legacy whole-volume cols carry no suffix
            return f"{metric}_{region}"

        def _finite_values(rs: list[dict[str, Any]], key: str) -> list[float]:
            values: list[float] = []
            for r in rs:
                v = r.get(key, "")
                if v in ("", None):
                    continue
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isnan(fv):
                    continue
                values.append(fv)
            return values

        # Group rows by (cohort, nfe).
        by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for r in rows:
            by_key.setdefault((str(r.get("cohort", "")), int(r["nfe"])), []).append(r)

        cols = [
            "cohort",
            "nfe",
            "region",
            "n_patients",
            "psnr_db_mean",
            "psnr_db_std",
            "ssim_mean",
            "ssim_std",
            "mae_mean",
            "mae_std",
            "mse_mean",
            "mse_std",
        ]
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for cohort, nfe in sorted(by_key):
                rs = by_key[(cohort, nfe)]
                for region in regions:
                    psnr_vals = _finite_values(rs, _metric_key(region, "psnr_db"))
                    ssim_vals = _finite_values(rs, _metric_key(region, "ssim"))
                    mae_vals = _finite_values(rs, _metric_key(region, "mae"))
                    mse_vals = _finite_values(rs, _metric_key(region, "mse"))

                    def _mean(v: list[float]) -> str:
                        return f"{statistics.fmean(v):.6g}" if v else ""

                    def _std(v: list[float]) -> str:
                        return f"{(statistics.stdev(v) if len(v) > 1 else 0.0):.6g}" if v else ""

                    w.writerow(
                        {
                            "cohort": cohort,
                            "nfe": nfe,
                            "region": region,
                            # n_patients = max populated count across the four
                            # metrics. Region with empty PSNR also tends to
                            # have empty MAE/MSE, so this is rarely > psnr_vals length.
                            "n_patients": max(
                                len(psnr_vals), len(ssim_vals), len(mae_vals), len(mse_vals)
                            ),
                            "psnr_db_mean": _mean(psnr_vals),
                            "psnr_db_std": _std(psnr_vals),
                            "ssim_mean": _mean(ssim_vals),
                            "ssim_std": _std(ssim_vals),
                            "mae_mean": _mean(mae_vals),
                            "mae_std": _std(mae_vals),
                            "mse_mean": _mean(mse_vals),
                            "mse_std": _std(mse_vals),
                        }
                    )

    @staticmethod
    def _write_aggregate_cv_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        """Patient-mean then cohort-mean aggregate, cv cohorts only (B1, 2026-07-29).

        Fixes the scan-weighted bias of :meth:`_write_aggregate_csv`: longitudinal
        cohorts (e.g. LUMIERE, ≈ 7 scans/patient) no longer dominate the aggregate.

        Algorithm
        ---------
        1. Filter ``rows`` to those with ``role == "cv"``.  Raise if any
           ``role == "test_only"`` row would have leaked into the aggregate.
        2. Group by ``(cohort, patient_id, nfe)`` → per-patient mean of each metric.
        3. Group patient means by ``(cohort, nfe, region)`` → cohort mean over
           patients (unweighted — every patient counts once regardless of scan count).
        4. Write one row per ``(cohort, nfe, region)``.

        Raises
        ------
        AssertionError
            If a ``test_only`` row's cohort is found in the cv subset
            (indicates a role-tagging bug upstream).
        """
        import math
        import statistics

        # Defensive: verify no test_only rows slipped through (should be impossible
        # because role is set from cv_names in _run_multi_cohort, but belt+braces).
        test_only_in_cv = [r for r in rows if r.get("role") == "test_only"]
        if test_only_in_cv:
            cohorts = {r.get("cohort") for r in test_only_in_cv}
            raise AssertionError(
                f"B2 guard: test_only cohorts {cohorts} reached aggregate_cv.csv. "
                "This leaks held-out test data into the training monitor. "
                "Check cohort role tagging in _run_multi_cohort."
            )

        cv_rows = [r for r in rows if r.get("role", "cv") == "cv"]
        if not cv_rows:
            # No cv cohorts in this epoch; write a valid empty file.
            with path.open("w", newline="") as f:
                pass
            return

        regions = ("whole", "wt", "bg", "bnwt", "netc", "ed", "et", "brain")
        metrics_4 = ("psnr_db", "ssim", "mae", "mse")

        def _metric_key(region: str, metric: str) -> str:
            if region == "whole" and metric in ("psnr_db", "ssim"):
                return metric
            if region == "brain" and metric == "psnr_db":
                return "psnr_db_brain"
            if region == "brain" and metric == "ssim":
                return "ssim_brain"
            if region == "brain" and metric == "mae":
                return "mae_whole"  # brain MAE stored as mae_whole
            if region == "brain" and metric == "mse":
                return "mse_whole"  # brain MSE stored as mse_whole
            return f"{metric}_{region}"

        def _fv(v: object) -> float | None:
            if v in ("", None):
                return None
            try:
                fv = float(v)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None
            return None if math.isnan(fv) else fv

        # Step 1: patient-level mean → group by (cohort, patient_id, nfe).
        pat_key_to_rows: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
        for r in cv_rows:
            key = (str(r.get("cohort", "")), str(r.get("patient_id", "")), int(r["nfe"]))
            pat_key_to_rows.setdefault(key, []).append(r)

        # Each patient-key collapses to a single averaged row.
        pat_rows: list[dict[str, Any]] = []
        for (cohort, _pid, nfe), scan_rows in pat_key_to_rows.items():
            pat_row: dict[str, Any] = {"cohort": cohort, "nfe": nfe, "n_scans": len(scan_rows)}
            for region in regions:
                for metric in metrics_4:
                    col = _metric_key(region, metric)
                    vals = [fv for r in scan_rows if (fv := _fv(r.get(col))) is not None]
                    pat_row[col] = statistics.fmean(vals) if vals else float("nan")
            pat_rows.append(pat_row)

        # Step 2: cohort mean over patients → group by (cohort, nfe, region).
        by_cohort_nfe: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for pr in pat_rows:
            by_cohort_nfe.setdefault((str(pr["cohort"]), int(pr["nfe"])), []).append(pr)

        cols = [
            "cohort",
            "nfe",
            "region",
            "n_patients",
            "n_scans_total",
            "psnr_db_mean",
            "psnr_db_std",
            "ssim_mean",
            "ssim_std",
            "mae_mean",
            "mae_std",
            "mse_mean",
            "mse_std",
        ]
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for (cohort, nfe), prs in sorted(by_cohort_nfe.items()):
                n_patients = len(prs)
                n_scans = sum(int(pr.get("n_scans", 1)) for pr in prs)
                for region in regions:
                    psnr_col = _metric_key(region, "psnr_db")
                    ssim_col = _metric_key(region, "ssim")
                    mae_col = _metric_key(region, "mae")
                    mse_col = _metric_key(region, "mse")
                    psnr_vals = [fv for pr in prs if (fv := _fv(pr.get(psnr_col))) is not None]
                    ssim_vals = [fv for pr in prs if (fv := _fv(pr.get(ssim_col))) is not None]
                    mae_vals = [fv for pr in prs if (fv := _fv(pr.get(mae_col))) is not None]
                    mse_vals = [fv for pr in prs if (fv := _fv(pr.get(mse_col))) is not None]

                    def _mean(v: list[float]) -> str:
                        return f"{statistics.fmean(v):.6g}" if v else ""

                    def _std(v: list[float]) -> str:
                        return f"{(statistics.stdev(v) if len(v) > 1 else 0.0):.6g}" if v else ""

                    w.writerow(
                        {
                            "cohort": cohort,
                            "nfe": nfe,
                            "region": region,
                            "n_patients": n_patients,
                            "n_scans_total": n_scans,
                            "psnr_db_mean": _mean(psnr_vals),
                            "psnr_db_std": _std(psnr_vals),
                            "ssim_mean": _mean(ssim_vals),
                            "ssim_std": _std(ssim_vals),
                            "mae_mean": _mean(mae_vals),
                            "mae_std": _std(mae_vals),
                            "mse_mean": _mean(mse_vals),
                            "mse_std": _std(mse_vals),
                        }
                    )
