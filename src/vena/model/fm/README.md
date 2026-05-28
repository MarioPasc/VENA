# `vena.model.fm` — Flow-Matching Generator

_Last updated: 2026-05-28_

Conditional **latent flow-matching** generator for gadolinium-free synthesis of
T1 post-contrast brain MRI ($\widehat{T_{1c}}$). A **ControlNet** branch built
around the **frozen MAISI-V2** flow-matching trunk injects the pre-contrast
modalities and the vessel/tumour mask priors; only the ControlNet is trained.
Sampling, metrics, and evaluation operate in the MAISI VAE latent space
(4-channel, 4× spatial compression), decoding to image space only for
image-domain metrics.

See `.claude/rules/model-coding-standards.md` for the conventions every file
here follows, and the proposal (`docs/proposal.md`) for the method and the
ablation plan.

## Layout

```
src/vena/model/fm/
├── controlnet/        ControlNet conditioning branch + losses
│   ├── conditioning.py    ConditioningAssembler: YAML-driven channel stack
│   │                      (latent:* / mask:* / prior:*), the single point of
│   │                      variation for ablations; supports mask perturbation.
│   ├── maisi_controlnet.py MaisiControlNet: trunk-cloned encoder/mid with a
│   │                      zero-init output projection (init_from_trunk).
│   ├── downsample/        Conditioning downsamplers (identity / nearest /
│   │                      trilinear / pooling) selected per spec.
│   ├── losses/            Composite loss: cfm (default S1), reconstruction,
│   │                      contrastive; builder assembles per training stage.
│   └── base.py            AbstractControlNet contract.
├── maisi/             Frozen MAISI-V2 FM trunk adapter
│   ├── trunk.py           load_trunk(): build + load the diff-UNet rflow trunk,
│   │                      log checkpoint SHA-256; returns a TrunkHandle.
│   ├── config.py          TrunkConfig (class token, spacing, arch overrides).
│   └── configs/           Arch JSONs for trunk and decode.
├── sampler/
│   └── rflow.py           RFlowEngine: thin wrapper over MONAI RFlowScheduler —
│                          sample_timesteps / add_noise / target_velocity (x1-x0).
├── inference/         Sampling + benchmarking
│   ├── euler.py           EulerSampler: Euler integration of the rflow schedule,
│   │                      NFE = num_inference_steps.
│   ├── timing.py          NFETimingProbe: CUDA-synced per-section timing
│   │                      (trunk / controlnet / decode), warm-up drop.
│   └── base.py            BaseSampler + get_sampler registry.
├── ema/
│   └── warmup_ema.py      WarmupEMA: Karras-style warm-up EMA (ema_pytorch),
│                          updated once per optimiser step.
├── metrics/           Region-masked metrics
│   ├── latent.py          Masked MSE / L1 / cosine in latent space.
│   ├── image.py           Whole-/masked-volume 3-D PSNR + SSIM (MONAI), [0,1].
│   └── regions.py         RegionResolver: per-batch masks (brain / wt /
│                          wt_dilated via GPU max_pool3d / bg / vessel).
├── eval/
│   └── exhaustive.py      Image-space exhaustive-validation helpers: real-T1c
│                          loader with encoder-matched percentile normalisation,
│                          whole-volume PSNR/SSIM, content-slice selection,
│                          best/worst comparison figure, latent_preds.h5 writer.
└── lightning/         Training orchestration
    ├── module.py          FMLightningModule: training step (ControlNet only),
    │                      EMA, grad-clip + grad-norm logging, the sampling
    │                      closure reused by evaluation.
    ├── data.py            LatentH5Dataset + LatentH5DataModule (UCSF-PDGM
    │                      latents H5, fold splits, WT mask from tumour latent).
    └── callbacks/
        ├── checkpointing.py     VENACheckpointCallback (rotated top-k +
        │                        last.ckpt) and BestCheckpointCallback (ema_best).
        ├── train_csv.py         train_step.csv + train_epoch.csv (clean, no
        │                        white cells; replaces Lightning's CSVLogger).
        ├── exhaustive_launcher.py  Async second-GPU exhaustive validation:
        │                        snapshots EMA, launches the subprocess on cuda:1,
        │                        skip-if-busy, joins at fit end, logs GPU usage.
        ├── sigterm.py           SIGTERM-safe final checkpoint.
        └── nfe_timing.py / qualitative.py / val_csv.py
                                 In-process validation writers — currently
                                 UNWIRED (validation is offloaded; see below).
```

## Training and validation flow

* **Training (primary GPU).** `routines/fm/train` builds `FMLightningModule`
  (trunk + ControlNet + EMA) and trains the ControlNet on the composite loss.
  Validation is **not** run in-process: the Trainer uses `limit_val_batches=0`,
  `logger=False`, and selects `ema_best` on the epoch-aggregated training loss.
  Clean per-step / per-epoch CSVs and `logs/train.log` make each run
  self-contained.
* **Exhaustive validation (second GPU, async).** On a slow cadence
  (`exhaustive_val.every_epochs`) `ExhaustiveValLauncher` snapshots the EMA
  shadow and launches `routines/fm/exhaustive_val` on `cuda:1`. That subprocess
  samples each validation patient at several NFE levels (default `{1,10,50}`),
  decodes to image space, compares against the **real T1c** (normalised exactly
  as the encoder input), and writes PSNR/SSIM, per-NFE timing, `latent_preds.h5`,
  and best/worst qualitative figures — while training continues uninterrupted.

## Key invariants

* Frozen MAISI VAE + trunk are immutable and never enter checkpoints; only the
  ControlNet (and its EMA shadow) are saved.
* Image metrics require intensity-space parity: decoded predictions are `[0,1]`
  (MAISI), so the real reference is `percentile_normalise(0, 99.5)`-d before
  comparison.
* EMA updates and grad-norm logging are gated to fire once per optimiser step
  (gradient-accumulation safe).
* 3-D throughout; region dilation and masking run on-device (no CPU/NumPy
  round-trips in the hot path).
