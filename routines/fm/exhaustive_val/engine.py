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
import yaml
from pydantic import BaseModel, ConfigDict, Field

from vena.model.autoencoder.maisi.decode.engine import MaisiDecoder
from vena.model.autoencoder.maisi.loader import load_autoencoder
from vena.model.autoencoder.maisi.preprocessing import DepthPad
from vena.model.fm.eval import (
    full_volume_psnr_ssim,
    load_real_t1c_normalised,
    render_comparison_figure,
    select_content_slices,
    write_latent_preds_h5,
)
from vena.model.fm.inference import EulerSampler
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


class _CNJobCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conditioning_inputs: list[str]
    arch_overrides: dict[str, Any] = Field(default_factory=dict)


class ExhaustiveValJobConfig(BaseModel):
    """Self-contained job spec written by the launcher (one YAML, one run)."""

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

    latents_h5: Path
    image_h5: Path
    fold: int = 0

    ema_snapshot: Path
    nfe_levels: list[int] = Field(default_factory=lambda: [1, 10, 50])
    n_patients: int = 20
    device: str = "cuda:1"
    output_dir: Path
    figure_n_slices: int = 10
    figure_slice_offset: int = 10

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
        module.eval()
        return module

    def _load_ema_snapshot(self, module: FMLightningModule, snapshot: Path) -> None:
        state = torch.load(snapshot, map_location=self.device, weights_only=True)
        # The launcher saves the EMA *shadow* model state_dict directly.
        module.ema.ema_model.load_state_dict(state)
        logger.info("loaded EMA snapshot %s", snapshot)

    def _val_patient_ids(self) -> list[str]:
        import h5py

        with h5py.File(self.cfg.latents_h5, "r") as f:
            ids = [
                b.decode() if isinstance(b, bytes) else str(b)
                for b in f[f"splits/cv/fold_{self.cfg.fold}/val"][:]
            ]
        return ids[: self.cfg.n_patients]

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
        sampler = EulerSampler(scheduler=module.rflow.scheduler)

        patient_ids = self._val_patient_ids()
        dataset = LatentH5Dataset(cfg.latents_h5, patient_ids)

        metric_rows: list[dict[str, Any]] = []
        latent_entries: list[tuple[str, int, Any]] = []
        latents_by_pid_nfe: dict[tuple[str, int], torch.Tensor] = {}
        ssim_by_pid: dict[str, list[float]] = {pid: [] for pid in patient_ids}
        gen_decode_time: dict[tuple[str, int], float] = {}

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
                )
            except Exception as exc:
                logger.warning("exhaustive-val: patient '%s' failed (%s); skipping.", pid, exc)
                continue
            logger.info("  [%d/%d] %s done", i + 1, len(patient_ids), pid)

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
            module, vae, ssim_by_pid, latents_by_pid_nfe, gen_decode_time, out_dir
        )
        logger.info("exhaustive-val epoch=%d complete -> %s", cfg.epoch, out_dir)
        return out_dir

    def _process_patient(
        self,
        i: int,
        pid: str,
        dataset: LatentH5Dataset,
        module: FMLightningModule,
        sampler: EulerSampler,
        vae: MaisiDecoder,
        latent_metrics: LatentMetrics,
        image_metrics: ImageMetrics,
        metric_rows: list[dict[str, Any]],
        latent_entries: list[tuple[str, int, Any]],
        latents_by_pid_nfe: dict[tuple[str, int], torch.Tensor],
        ssim_by_pid: dict[str, list[float]],
        gen_decode_time: dict[tuple[str, int], float],
    ) -> None:
        cfg = self.cfg
        item = dataset[i]
        batch = {
            k: (v.unsqueeze(0).to(self.device) if isinstance(v, torch.Tensor) else v)
            for k, v in item.items()
        }
        z_target = batch["z_t1c"]
        real_t1c = load_real_t1c_normalised(cfg.image_h5, pid).to(self.device)
        module._val_cond = module.conditioning(batch)

        for nfe in cfg.nfe_levels:
            z_pred, gen_t = self._sample(module, sampler, z_target, int(nfe))
            latents_by_pid_nfe[(pid, int(nfe))] = z_pred.detach().to("cpu")
            latent_entries.append((pid, int(nfe), z_pred[0].detach().cpu().numpy()))

            img_pred, dec_t = self._decode(vae, z_pred, real_t1c.shape[-1])
            psnr, ssim = full_volume_psnr_ssim(img_pred, real_t1c, image_metrics)

            mask = torch.ones_like(z_target, dtype=torch.bool)
            lat_mse = float(latent_metrics.mse(z_pred, z_target, mask)[0].item())
            lat_l1 = float(latent_metrics.l1(z_pred, z_target, mask)[0].item())
            lat_cos = float(latent_metrics.cosine(z_pred, z_target, mask)[0].item())

            ssim_by_pid[pid].append(ssim)
            gen_decode_time[(pid, int(nfe))] = gen_t + dec_t
            metric_rows.append(
                {
                    "patient_id": pid,
                    "nfe": int(nfe),
                    "psnr_db": psnr,
                    "ssim": ssim,
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
        sampler: EulerSampler,
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
        self, vae: MaisiDecoder, z_pred: torch.Tensor, original_depth: int
    ) -> tuple[torch.Tensor, float]:
        padded_depth = int(z_pred.shape[-1]) * 4
        after = padded_depth - int(original_depth)
        pad = DepthPad(
            before=0, after=after, original_depth=int(original_depth), padded_depth=padded_depth
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = vae.decode(z_pred, pad)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return out.image[0, 0].float().clamp(0.0, 1.0), time.perf_counter() - t0

    def _render_best_worst(
        self,
        module: FMLightningModule,
        vae: MaisiDecoder,
        ssim_by_pid: dict[str, list[float]],
        latents_by_pid_nfe: dict[tuple[str, int], torch.Tensor],
        gen_decode_time: dict[tuple[str, int], float],
        out_dir: Path,
    ) -> None:
        mean_ssim = {pid: (sum(v) / len(v)) for pid, v in ssim_by_pid.items() if v}
        if not mean_ssim:
            logger.warning("no SSIM scores; skipping qualitative figures.")
            return
        best = max(mean_ssim, key=mean_ssim.get)
        worst = min(mean_ssim, key=mean_ssim.get)
        for tag, pid in (("best", best), ("worst", worst)):
            real = load_real_t1c_normalised(self.cfg.image_h5, pid).to(self.device)
            synth_by_nfe: dict[int, torch.Tensor] = {}
            time_by_nfe: dict[int, float] = {}
            for nfe in self.cfg.nfe_levels:
                z = latents_by_pid_nfe[(pid, int(nfe))].to(self.device)
                img, _ = self._decode(vae, z, real.shape[-1])
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

    @staticmethod
    def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        cols = [
            "patient_id",
            "nfe",
            "psnr_db",
            "ssim",
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
