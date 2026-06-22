# VENA

> **V**essel **E**ncoded **N**eural **A**ugmentation — SWAN-conditioned latent flow matching for gadolinium-free synthesis of T1 post-contrast brain MRI.

## End goal

Replace the gadolinium-based contrast agent in brain-tumour follow-up MRI with a generative model that synthesises T1 post-contrast ($\widehat{T_{1c}}$) from a pre-contrast multimodal input plus a **vessel prior $M_v$ extracted from SWAN/SWI** and a tumour-shape prior $M_{\text{tum}}$. The model is a **conditional latent flow-matching** generator on top of the **MAISI-V2 VAE-GAN** latent space, with a **ControlNet-style** branch that injects the vessel and tumour masks. Internal validation on UCSF-PDGM ($N{=}501$, GE 3 T), external validation on a Málaga in-house cohort (multi-vendor, glioma + meningioma). Target deliverable: MICCAI 2026 or *MedIA*/*IEEE TMI* journal submission.

The differentiator versus prior work (Kleesiek 2019, Preetha 2021, McCaD 2024, CFM 2025, T1C-RFlow 2025, TumorFlow 2026) is **explicit vessel-aware conditioning via SWAN**, in latent space, with vessel-resolved evaluation — not just whole-volume PSNR/SSIM.

## Documentation source-of-truth

| Asset | Path |
|---|---|
| Proposal (method, losses, evaluation, ablations, timeline) | `/media/mpascual/Sandisk2TB/research/vena/docs/proposal.md` |
| Literature review (CNN/GAN → diffusion → flow-matching → vessel methods) | `/media/mpascual/Sandisk2TB/research/vena/docs/literature.md` |
| External code, checkpoints, datasets — canonical paths | `src/external/LINKS.md` |
| Project rules (enforced) | `.claude/rules/` |

Whenever this file drifts from the proposal, **the proposal wins**.

## Project rules

Read these before non-trivial work:

- [`.claude/rules/coding-standards.md`](.claude/rules/coding-standards.md) — Python conventions, env, testing, library-first policy, exception-handling, docstring-drift hygiene.
- [`.claude/rules/preflight-pattern.md`](.claude/rules/preflight-pattern.md) — Routine layout (`routines/<bucket>/<name>/`), thin engine over `src/vena/` library code, `decision.json` v0.3.0 contract, hard pre-flight gate enforcement.
- [`.claude/rules/extensibility.md`](.claude/rules/extensibility.md) — `vena.common` adapter surface (MAISI re-exports, shared decode helpers), `vena.data.cohort` protocol + registry for adding new pathologies, `MultiCohortLatentDataModule`-only data path.
- [`.claude/rules/h5-design-principles.md`](.claude/rules/h5-design-principles.md) — Schema-versioned, self-describing HDF5 artifacts (applies to the UCSF-PDGM cache at `UCSFPDGM_image.h5`).
- [`.claude/rules/training-stages.md`](.claude/rules/training-stages.md) — Six-phase timeline (data → pipeline → training → internal val → external val → writing) and the canonical routine names that map onto it.
- [`.claude/rules/external-deps.md`](.claude/rules/external-deps.md) — How to consume frozen MAISI-V2 weights and UCSF-PDGM data; what `src/external/` is and is not.
- [`.claude/rules/model-coding-standards.md`](.claude/rules/model-coding-standards.md) — FM-generator conventions (`src/vena/model/fm/`): training-only module + offloaded validation, metric-CSV logging, EMA/grad-accum, intensity-space parity, async second-GPU exhaustive validation, `vena.common` import discipline.

## Quick context

| Property | Value |
|---|---|
| Modality (input core) | $T_{1\text{pre}}$, T2, FLAIR, SWAN |
| Modality (target) | $T_{1c}$ (post-contrast) |
| Conditioning priors | Vessel mask $M_v$ (Frangi on SWAN, soft); tumour mask $M_{\text{tum}}$ (BraTS-style segmenter) |
| Native shape (UCSF-PDGM, isotropic 1 mm) | ~`(240, 240, 155)` after skull-strip / co-registration |
| Latent (MAISI-V2 VAE) | 4× spatial compression, 4 channels |
| Generator | Latent FM (rectified-flow / linear interpolant), DiT or U-Net trunk |
| Conditioning route | ControlNet branch with **scale-ramped zero-init** (`MaisiControlNet.output_scale` non-persistent buffer multiplied into the down-block + mid-block residuals; sigmoid 0 → 1 over `output_scale_ramp.ramp_steps` via `OutputScaleRampCallback`). Removes the cold-start dead-time the literal-zero-init ControlNet otherwise pays. |
| Loss (default) | **L1 velocity loss** ($\mathcal{L}_{\text{CFM}}$ with `loss.cfm.norm: l1`) — median-seeking, preserves sharp enhancing-rim boundaries the L2 (retired) objective smeared into a halo. Decoder-feature LPL ($\mathcal{L}_{\text{dec}}$) added in stage S3 (see `.claude/notes/changes/decoder_perceptual_loss_s3*.md`). |
| Inference | Rectified-flow sampling, 1–10 Heun/Euler steps; <10 s/volume on A100 |
| Train cohort | UCSF-PDGM, 400 train / 50 val / 50 test (patient-level split) |
| External cohort | Hospital U. Regional de Málaga (glioma + meningioma, multi-vendor) |
| Conda env | `vena` (Python ≥3.11) |

## Layout

```
VENA/
├── CLAUDE.md                       # this file
├── pyproject.toml                  # deps, ruff, pytest, mypy, console scripts (vena-<bucket>-<name>)
├── src/
│   ├── external/LINKS.md           # canonical external paths (MAISI, UCSF-PDGM, Picasso mirror) — never edit other files here
│   └── vena/                       # library code: importable, unit-testable
│       ├── common/                 # canonical adapter surface; re-exports MAISI primitives + shared decode helpers (vena.common.decode.{decode_box, decode_depth_identity}). All cross-module MAISI usage routes through here.
│       ├── data/
│       │   ├── cohort/             # CohortProtocol + decorator-based CohortRegistry. Add a new pathology by writing a niigz reader + @register_cohort. See HOWTO.md.
│       │   ├── niigz/              # NIfTI-source per-cohort readers (UCSF-PDGM, BraTS-GLI). Each registers via @register_cohort at import time.
│       │   ├── h5/                 # NIfTI → image-domain H5 + image → latent H5 converters. Shared validator/writer/manifest under shared/.
│       │   ├── registry/           # CorpusRegistry JSON loader; the cohort catalogue consumed by the multi-cohort data path.
│       │   └── augment/            # 3D latent-space transforms (flip, translate, gamma, rotate) + AugmentationPipeline + AugmentationTracker callback.
│       ├── preflight/              # Library implementations of gating pre-flights (vessel_mask, latent_aug_equivariance, priors_validation, venous_atlas_build).
│       ├── prior_maps/             # Vessel / perfusion / cellularity / susceptibility prior-map computation (consumed by the corresponding routines).
│       └── model/
│           ├── autoencoder/maisi/  # Frozen MAISI-V2 VAE adapter (loader, encoder, decoder, preprocessing). Reach in only when extending the adapter; everywhere else use vena.common.
│           └── fm/                 # FM generator: controlnet/ (MAISI-style branch + assembler + losses + downsamplers), ema/, inference/ (samplers + timing probe), maisi/ (trunk loader + grad-safe patch), metrics/ (latent + image + regions), sampler/ (RFlow), eval/ (exhaustive-val helpers), lightning/ (the only LightningModule + MultiCohortLatentDataModule + callbacks).
├── routines/                       # CLI entrypoints — one YAML arg, thin engine (per .claude/rules/preflight-pattern.md)
│   ├── preflights/                 # Gating checks (vessel_mask, latent_aug_equivariance, priors_validation, venous_atlas_build). Each writes decision.json consumed downstream.
│   ├── h5_datasets/                # Phase-1 H5 converters per cohort (ucsf_pdgm, brats_gli).
│   ├── encode/                     # MAISI VAE encoding of image H5 → latent H5.
│   ├── prior_maps/                 # Per-prior runners: vessel_priors, perfusion_priors, cellularity_priors, susceptibility_priors.
│   └── fm/                         # FM trainer (train/) + async exhaustive validation (exhaustive_val/). train/exceptions.py defines PreflightGateError.
├── artifacts/<routine>/<UTC>/      # Per-routine outputs: report.md, figures/, tables/, decision.json. <routine>/LATEST symlink points at the most recent.
├── experiments/<run_id>/           # FM training runs: checkpoints/, logs/train.log, metrics/{train_step,train_epoch,augmentations_per_epoch}.csv, exhaustive_val/epoch_NNN/, decision.json (schema 0.3.0).
├── tests/                          # pytest; marker rule per pyproject.toml. tests/data/cohort/ tests/data/registry/ tests/common/ tests/routines/fm/ tests/model/fm/ are the primary directories.
└── .claude/
    ├── rules/                      # Project-wide rules (above). Agent-directed; treat as enforceable.
    ├── skills/                     # /explore /test /dl-scientist /refactor /server3 (launch on icai-server).
    └── hooks/                      # Session hooks (compact-context.sh).
```

## Conda env and quick commands

```bash
# Activate
source ~/.conda/envs/vena/bin/activate           # or: conda activate vena

# Install (CUDA wheel selected at install time — see pyproject.toml notes)
~/.conda/envs/vena/bin/pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cu124  # RTX 4060
~/.conda/envs/vena/bin/pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cu121  # RTX 3060
# Picasso A100 inside NGC Singularity: torch already in image, use --no-deps then install non-torch deps.

# Test
~/.conda/envs/vena/bin/python -m pytest -m "not slow and not gpu" -v --tb=short
~/.conda/envs/vena/bin/python -m pytest -m "fm and gpu" -v

# Lint / format
~/.conda/envs/vena/bin/python -m ruff check src/ routines/ tests/
~/.conda/envs/vena/bin/python -m ruff format src/ routines/ tests/
```

## Hardware

- Local A: RTX 4060 8 GB (Debian 12, CUDA 12.4). For development, smoke runs, light visualisation.
- Local B: RTX 3060 12 GB (CUDA 12.2). Larger smoke batches.
- Picasso (UMA HPC): SLURM, **Singularity only** (no Docker), 8× A100 40 GB per node (effective ~39 GB). Full training target: ~5 days on 4× A100 (per proposal §5).

See `src/external/LINKS.md` for the Picasso mirror of MAISI weights and the UCSF-PDGM H5 cache.

## Rectified-flow timestep convention (load-bearing)

The MAISI trunk + MONAI `RFlowScheduler` use **integer timestep codes**
in `[0, num_train_timesteps)`. In this codebase `num_train_timesteps = 1000`
and `use_discrete_timesteps = true`. Two conventions coexist; mixing them
silently is the 2026-06-19 S3 bug (3.5h of training discarded because
`x̂_1 = x_t + (1 - 998)·v` was being fed to the LPL decoder).

| Symbol | Domain | Meaning | Where it appears |
|---|---|---|---|
| `timesteps` | int, `[0, 1000)` | raw discrete code | `RFlowEngine.sample_timesteps` output; first arg to the trunk |
| `α` | float, `[0, 1]` | noise fraction | `α = timesteps.float() / T` |
| `t_dn` | float, `[0, 1]` | data fraction (design note "t") | `t_dn = 1 - α`; **1 = data, 0 = noise** |

Forward and target velocity (`src/vena/model/fm/sampler/rflow.py`):

```
x_t = (1 - α) · x_clean + α · x_noise              # MONAI add_noise
u   = x_clean - x_noise = x_1 - x_0                # RFlowEngine.target_velocity
v   ≈ u                                            # network output
```

One-step clean estimate (used by S3 LPL):

```
x̂_1 = x_t + α · v                                  # MONAI / codebase form
    = x_t + (1 - t_dn) · v                          # design-note form (.claude/notes/changes/decoder_perceptual_loss_s3.md)
```

LPL high-SNR gate (Berrada 2025): `t_dn > t_min` (e.g. `t_min = 0.4`
from `decoder_lpl_profile/LATEST/decision.json`). Equivalent to
`α < 1 - t_min`. **Never compare raw `timesteps` to `t_min`** — the
gate would pass on every `timesteps ≥ 1`.

The canonical implementation lives in `FMLightningModule.training_step`
S3 branch — copy it verbatim when a new module needs `x̂_1`. The
`decoder_lpl_profile` preflight (`phase2_separation.py`) uses
design-note `t_b ∈ [0, 1]` directly (it samples `t_b` from a sweep
list, not via the trunk's integer codes), so its `(1 - t_b)·v` form
is the SAME formula `α · v` — internally consistent.

Pitfall in adapters: if you pass a float `t ∈ [0, 1]` to the MAISI
trunk's `forward(x_t, timesteps, ...)`, the sinusoidal embedding will
treat `0.4` as 400× smaller frequency than the trunk was trained on.
Always scale to integer code form (`(t * T).long()`) before the trunk
call.

## S1 v2 baseline recipe (2026-06-20 — load-bearing)

Production S1 (`routines/fm/train/configs/runs/picasso_s1_1000ep_fft.yaml`) carries four recipe deltas over the retired 2026-06-12 S1 (L2 + hard zero-init, 1500 ep, plateaued at 26.5 dB whole-vol PSNR / 18.3 dB WT-PSNR with the curve flat past epoch 475). All four are pinned in YAML and exercised by the smoke at `routines/fm/train/configs/smoke/loginexa_s1_4ep_fft.yaml`:

1. **`loss.cfm.norm: l1`** — median-seeking; sharp enhancing-rim boundaries. T1C-RFlow (Eidex 2025) and most 3D medical latent-FM baselines use L1; the retired L2 produced a blurry tumour-region manifold (analysis §4 / E1 in `decoder_perceptual_loss_s3_analysis_2026-06-20.md`).
2. **`model.controlnet.output_scale_ramp: {enabled: true, ramp_steps: 5000, steepness: 10.0}`** — `MaisiControlNet.output_scale` is a non-persistent buffer multiplied into every down-block + mid-block residual *before return* in `forward`. `OutputScaleRampCallback` (`vena.model.fm.lightning.callbacks.output_scale_ramp`) fills it with `sigmoid(steepness * (step/ramp_steps - 0.5))` on `on_train_batch_start`, capped at 1.0 after `ramp_steps`. Removes the cold-start dead-time vs T1C-RFlow's warm channel-concat. Sits upstream of the `vena.model.fm.maisi.grad_safe` out-of-place residual-add patch — no interaction with grad-checkpointing. The buffer is `persistent=False`: on resume the formula is recomputed from `global_step`.
3. **`rflow.use_timestep_transform: true` + `rflow.base_img_size_numel: 129024`** — SD3-style resolution-aware timestep weighting (Esser et al. 2024, arXiv:2403.03206); concentrates sampled timesteps at α ∈ [0.3, 0.7] where structure forms. **Complementary to the LPL `t_min=0.4` gate, not antagonistic** (analysis §4b). `base_img_size_numel` matches VENA's brain-box latent (48×56×48). **Load-bearing pitfall**: when `use_timestep_transform=True`, the inference `EulerSampler` must receive `input_img_size_numel` or MONAI's `timestep_transform` divides `None / int` and every per-patient exhaustive-val pass fails with a silent WARNING. The patch at `routines/fm/exhaustive_val/engine.py:332` plumbs `cfg.rflow.base_img_size_numel` through; fallback `48*56*48`.
4. **`model.controlnet.conditioning_inputs[3]: mask:wt:zero_out`** — S1 does not condition on the WT mask (no region-contrastive loss in S1), but the channel slot is preserved via the new `ZeroOutDownsampler` (returns `torch.zeros_like(x)`, `out_channels` inherits the kind-based default). Warm-start S1 → S2 / S3 is byte-identical in channel layout; S2/S3 YAMLs swap to `mask:wt:identity` (current) or `mask:wt:lift_to_4ch` (reserved infrastructure: learned `Conv3d(1, 4, kernel_size=1)`; `LiftTo4ChDownsampler.out_channels == 4`; lands now but used only when S2/S3 retrain with mask conditioning).

**EarlyStopping monitor for S3 only**: `train/total_epoch` was patched on 2026-06-20 to project the LPL contribution onto `lambda_max` (steady-state) instead of the live `lam_active` (warmup ramp), so the monitor no longer grows monotonically across S3 warmup and EarlyStopping no longer fires at exactly `warmup_epochs + patience`. See `feedback_s3_monitor_pitfall.md` and `decoder_perceptual_loss_s3_analysis_2026-06-20.md`.

## Conventions in one line each

- **Library code in `src/vena/`, routines in `routines/<bucket>/<name>/` are thin engines** — see `.claude/rules/preflight-pattern.md`.
- **All MAISI primitives import from `vena.common`** (re-export layer); never reach into `vena.model.autoencoder.maisi.*` from a sibling module — see `.claude/rules/extensibility.md`.
- **`MultiCohortLatentDataModule` is the only training data path**; `data.corpus_registry` is required and `data.latents_h5` is rejected at config validation. A single-cohort run uses a single-cohort registry JSON.
- **Adding a new pathology cohort** = write `vena/data/niigz/<name>.py` with `@register_cohort` + a `routines/h5_datasets/<name>/` converter + a registry-JSON entry. See `src/vena/data/cohort/HOWTO.md`.
- **Frozen pretrained models are immutable** — never edit `src/external/*` (other than `LINKS.md`), never write to checkpoint paths.
- **Pre-flights are gating** — `_assert_preflight_gates(cfg)` runs at the top of `Engine.run()` and raises `PreflightGateError` before any side effect.
- **`decision.json` v0.9.0** in every training run carries trunk + VAE SHA-256, cohorts used, augmentation gate path, LPL coupling fields — see `.claude/rules/preflight-pattern.md`. Bumped from 0.8.0 by the 2026-06-09 CFG-dropout + the 2026-06-20 LPL coupling changes.
- **Conditioning assembler is downsampler-aware** — `ConditioningAssembler.channels_per_spec` consults `downsampler.out_channels` and only falls back to the kind-based default when the operator returns `None`. Channel-lifting downsamplers (`lift_to_4ch`) override the default; stateless operators (`identity`, `nearest`, `zero_out`, `avg_pool`, `trilinear`) preserve it.
- **Exhaustive-val comparison figure** (2026-06-20 global overhaul, `vena.model.fm.eval.exhaustive.render_comparison_figure`) — black background, NFE rows sorted by SSIM descending, per-NFE PSNR/SSIM annotation in the row ylabel, per-slice intensity-matched display window (synth anchored to the real slice's per-slice `(min, max)`). Suptitle is bare `"<TAG> — <patient>"`; aggregate SSIM removed. The caller `_render_best_worst` builds a `psnr_ssim_by_pid_nfe` map from `metric_rows` and passes the per-patient slice.
- **3D throughout** — no 2D ops in the core pipeline; 2.5D only in clearly-labelled evaluation utilities.
- **YAML-driven** — every hyperparameter through OmegaConf/Pydantic; round-trip into the produced artifact.
- **No bare `except Exception`** in library code; either narrow the type or re-raise after logging — see `coding-standards.md` rule 15.
- **No external private-attribute writes** (`module._foo = ...`); add a public method on the owning module — see `coding-standards.md` rule 18.
- **Conventional commits**, no force-push, no co-author trailer.

## Open questions tracked in `DECISIONS.md`

This file does not exist yet; it will be created the first time a non-trivial architectural decision is made (vesselness method choice, residual-vs-direct target parameterisation, ControlNet skip-connection scheme, etc.). One entry per decision: date, options considered, choice, rationale, reversibility.
