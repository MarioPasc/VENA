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
from typing import Any

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

logger = logging.getLogger(__name__)


class ExhaustiveValConfigError(Exception):
    """Raised on a malformed exhaustive-validation job config."""


class _TrunkJobCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    checkpoint: Path
    arch_json: Path | None = None
    arch_overrides: dict[str, Any] = Field(default_factory=dict)
    class_token: int = 9
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0)
    trainable: bool = True


class _CNJobCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conditioning_inputs: list[str]
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

    ema_snapshot: Path
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

    @classmethod
    def from_yaml(cls, path: Path | str) -> ExhaustiveValJobConfig:
        with Path(path).open("r") as f:
            return cls.model_validate(yaml.safe_load(f))


class ExhaustiveValEngine:
    """Runs the full image-space validation for one snapshot/epoch."""

    def __init__(self, cfg: ExhaustiveValJobConfig) -> None:
        self.cfg = cfg
        self.device = self._resolve_device(cfg.device)

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
        )
        module = module.to(self.device)
        module.setup()
        self._load_ema_snapshot(module, cfg.ema_snapshot)
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

            val_patient_keys = _decode(f[f"splits/cv/fold_{self.cfg.fold}/val"])
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
        sampler = get_sampler(cfg.integrator)(scheduler=module.rflow.scheduler)

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

        self._write_metrics_csv(out_dir / "metrics.csv", metric_rows)
        self._write_timing_csv(out_dir / "timing.csv", metric_rows)
        write_latent_preds_h5(
            out_dir / "latent_preds.h5",
            latent_entries,
            epoch=cfg.epoch,
            run_id=cfg.run_id,
            extra_attrs={"nfe_levels_json": json.dumps(list(cfg.nfe_levels))},
        )
        self._render_best_worst(
            module, vae, ssim_by_pid, latents_by_pid_nfe, gen_decode_time, pid_to_image_h5, out_dir
        )
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
        dataset = LatentH5Dataset(cfg.latents_h5, patient_ids)
        ssim_by_pid.update({pid: [] for pid in patient_ids})
        pid_to_image_h5.update({pid: cfg.image_h5 for pid in patient_ids})

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
            except Exception as exc:
                logger.warning("exhaustive-val: patient '%s' failed (%s); skipping.", pid, exc)
                continue
            logger.info("  [%d/%d] %s done", i + 1, len(patient_ids), pid)

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
        budgets = self._split_n_patients(int(cfg.n_patients), len(all_cohorts))
        logger.info(
            "exhaustive-val: cohort budgets %s (total=%d, n_cohorts=%d)",
            {c.name: b for c, b in zip(all_cohorts, budgets, strict=True)},
            cfg.n_patients,
            len(all_cohorts),
        )

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
            dataset = LatentH5Dataset(cohort.latent_h5, patient_ids)
            ssim_by_pid.update({pid: [] for pid in patient_ids})
            pid_to_image_h5.update({pid: cohort.image_h5 for pid in patient_ids})

            # ``i`` indexes ``dataset`` (per-cohort, 0-based) so that
            # ``dataset[i]`` returns the row for ``pid``. The previous
            # implementation passed a global running counter here, which both
            # mislabelled the first cross-cohort patient and raised
            # ``IndexError`` on every subsequent one.
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
                    )
                except Exception as exc:
                    logger.warning(
                        "exhaustive-val: patient '%s' (cohort %s) failed (%s); skipping.",
                        pid,
                        cohort.name,
                        exc,
                    )
                    continue
                logger.info(
                    "  cohort=%s [%d/%d] %s done", cohort.name, i + 1, len(patient_ids), pid
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

        # WT mask in image space (NN-upsampled from the latent mask the dataset
        # already loaded). Used for region-masked PSNR/SSIM below — the decoded
        # volume is already on the GPU, so the extra metric calls are near-free.
        m_wt_img = self._wt_mask_in_image_space(batch, real_box.shape)

        for nfe in cfg.nfe_levels:
            z_pred, gen_t = self._sample(module, sampler, z_target, int(nfe))
            latents_by_pid_nfe[(pid, int(nfe))] = z_pred.detach().to("cpu")
            latent_entries.append((pid, int(nfe), z_pred[0].detach().cpu().numpy()))

            img_pred, dec_t = self._decode(vae, z_pred, crop_spec)
            psnr, ssim = full_volume_psnr_ssim(img_pred, real_box, image_metrics)
            psnr_wt, ssim_wt, psnr_bg, ssim_bg = self._region_psnr_ssim(
                img_pred, real_box, m_wt_img, image_metrics
            )

            mask = torch.ones_like(z_target, dtype=torch.bool)
            lat_mse = float(latent_metrics.mse(z_pred, z_target, mask)[0].item())
            lat_l1 = float(latent_metrics.l1(z_pred, z_target, mask)[0].item())
            lat_cos = float(latent_metrics.cosine(z_pred, z_target, mask)[0].item())

            ssim_by_pid[pid].append(ssim)
            gen_decode_time[(pid, int(nfe))] = gen_t + dec_t
            metric_rows.append(
                {
                    "epoch": int(cfg.epoch),
                    "cohort": cohort,
                    "patient_id": pid,
                    "nfe": int(nfe),
                    "psnr_db": psnr,
                    "ssim": ssim,
                    "psnr_db_wt": psnr_wt,
                    "ssim_wt": ssim_wt,
                    "psnr_db_bg": psnr_bg,
                    "ssim_bg": ssim_bg,
                    "latent_mse": lat_mse,
                    "latent_l1": lat_l1,
                    "latent_cosine": lat_cos,
                    "gen_sec": gen_t,
                    "decode_sec": dec_t,
                }
            )
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
            render_comparison_figure(
                real.cpu(),
                synth_by_nfe,
                time_by_nfe,
                slices,
                patient_id=pid,
                mean_ssim=mean_ssim[pid],
                title_tag=tag,
                out_path=out_dir / f"figure_{tag}.png",
            )
            logger.info("wrote figure_%s.png (%s, mean SSIM=%.4f)", tag, pid, mean_ssim[pid])

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
        m_wt = batch.get("m_wt")
        if m_wt is None:
            return None
        if not isinstance(m_wt, torch.Tensor):
            return None
        if m_wt.dim() == 4:
            # CSR datasets occasionally return (1,h,w,d); add the batch dim.
            m_wt = m_wt.unsqueeze(0)
        target_spatial = tuple(int(s) for s in image_shape[-3:])
        m_up = F.interpolate(m_wt.float(), size=target_spatial, mode="nearest")
        return m_up.bool()

    @staticmethod
    def _region_psnr_ssim(
        img_pred: torch.Tensor,
        real_box: torch.Tensor,
        m_wt_img: torch.Tensor | None,
        image_metrics: ImageMetrics,
    ) -> tuple[float, float, float, float]:
        """Region-masked PSNR/SSIM for WT and BG.

        ``decode_box`` returns a 3-D ``(H, W, D)`` tensor and the whole-volume
        helper adds two leading dims internally; the masked metric helpers
        expect ``(B, C, H, W, D)``, so we promote both volume and mask here.
        Empty regions or a missing mask return ``nan`` for that pair so the
        CSV cell renders blank (downstream tooling reads ``""`` as NaN).
        """
        if m_wt_img is None or m_wt_img.numel() == 0:
            return (float("nan"),) * 4
        p = img_pred[None, None] if img_pred.ndim == 3 else img_pred
        r = real_box[None, None] if real_box.ndim == 3 else real_box
        wt = m_wt_img if m_wt_img.ndim == 5 else m_wt_img[None, None]
        bg = ~wt
        psnr_wt = image_metrics.psnr(p, r, wt)
        ssim_wt = image_metrics.ssim(p, r, wt)
        psnr_bg = image_metrics.psnr(p, r, bg)
        ssim_bg = image_metrics.ssim(p, r, bg)

        def _first_finite(t: torch.Tensor) -> float:
            v = float(t[0].item()) if t.numel() else float("nan")
            return v if v == v else float("nan")  # NaN-passthrough

        return (
            _first_finite(psnr_wt),
            _first_finite(ssim_wt),
            _first_finite(psnr_bg),
            _first_finite(ssim_bg),
        )

    @staticmethod
    def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        cols = [
            "cohort",
            "epoch",
            "patient_id",
            "nfe",
            "psnr_db",
            "ssim",
            "psnr_db_wt",
            "ssim_wt",
            "psnr_db_bg",
            "ssim_bg",
            "latent_mse",
            "latent_l1",
            "latent_cosine",
            "gen_sec",
            "decode_sec",
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
