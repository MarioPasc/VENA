# VENA augmentation — refined proposal (v2)

**Status.** Refinement of `unrefined_proposal.md` with empirical measurements from
server 3 (RTX 4090, MAISI-V2 frozen VAE-GAN, real `UCSFPDGM_image.h5`), grounded
in the existing VENA augmentation infrastructure and the latent-equivariance
preflight already on disk.

**Audience.** The implementer agent. Read this file *and*
`unrefined_proposal.md` in full before coding — the unrefined proposal carries
the literature review (SynthSeg, Augment-to-Augment, nnU-Net, Boydev, MAISI
constraints) that is not duplicated here.

**Empirical anchors used throughout.** Measured 2026-06-03 on `icai-server`,
`CUDA_VISIBLE_DEVICES=0` (RTX 4090, 24 GB), via
`scripts/bench_encode_aug.py`:

| Quantity | Value |
|---|---|
| Encoder | MAISI-V2 VAE (`autoencoder_v2.pt`), frozen, `precision_mode=autocast`, `norm_float16=True` |
| Input box | `(240, 240, 160)` after centred crop-pad, fp32 input → autocast |
| Latent shape | `(B=1, C=4, H=60, W=60, D=40)` |
| Per-modality encode time | **0.89–0.94 s** (median 0.92 s; std ≤ 0.03 s) |
| Per-scan (4 modalities) | **3.65 s median** |
| Per-modality latent footprint | 1.125 MB fp16 |
| Per-scan latent on disk (BraTS-GLI, 4 mods + mask) | **7.20 MB** |
| Disk free `/media/hddb` | 1.1 TB |
| CV-eligible scan count (server3 corpus) | **2 988 scans** across UCSF-PDGM, BraTS-GLI, UPENN-GBM, IvyGAP, LUMIERE |

The 2 988 figure differs from the unrefined proposal's "~2 800 sessions"
because LUMIERE longitudinal sessions (≈597, inferred from 4.3 GB / 7.2 MB)
and UPENN-GBM (611) push the count up; deduplication of the UCSF-PDGM ⊂
BraTS-GLI overlap (~293 scans) has already been applied at the
`cohort_dedup` preflight (see `project_cohort_dedup` memory entry). For per-
patient budgeting the CV pool is still ≈1 700 unique IDs; for *augmentation
budgeting* what matters is the per-scan count, since augmentation is
session-keyed.

---

## TL;DR (decisions)

1. **Offline pre-encoded bank, K = 4 variants, all CV scans, single one-time
   pass.** Online tier remains `flip_lr` (p=0.5) + `translate` ≤ 8 voxels
   (p=0.5) on the cached latents, untouched.
2. **Library: TorchIO for the bank-builder, with MONAI's
   `RandHistogramShiftd` for the monotonic intensity remap (the
   Augment-to-Augment fix that has no TorchIO equivalent).**
3. **Variant menu locked to four nuisance axes:** `v1` field/scanner
   (bias + Bloch-proxy gamma), `v2` contrast-shape (histogram shift +
   brightness/contrast), `v3` SNR/resolution (noise + anisotropy +
   low-resolution simulation + low-p motion), `v4` anatomy (light elastic +
   small-angle affine, joint over inputs + target + mask).
4. **Routine to ship:** `routines/offline_aug/maisi/` — engine, CLI, configs
   per cohort, SLURM script for Picasso. Output: a `*_latents_aug.h5` per
   cohort, schema-versioned, paired with `decision.json` for the FM trainer
   to consume.
5. **Time/storage envelopes (RTX 4090, K=4, all 2 988 CV scans):
   ≈ 10 h on one GPU, ≈ 5 h on two GPUs; +55 GB of latents added on top of
   the existing ~24 GB v0 bank.**

The architecture switch (FFT → LoRA(r=8) on the trunk + ControlNet) **does
not change any of these decisions**: see §1.

---

## Q1 — Offline pre-encode vs. on-the-fly encode

**Answer: pre-encode, decisively. Offline is ≈ 20× cheaper in wall-clock per
training run; the LoRA switch does not move this break-even.**

### Per-step accounting

The on-the-fly route runs the MAISI encoder inside the training loop. Cost
is dominated by the encoder forward pass on the `(B, 1, 240, 240, 160)`
input:

| Setting | Time |
|---|---|
| One modality, autocast + fp16 GroupNorm (best case) | 0.92 s |
| Four modalities, batch = 1 | 3.7 s |
| Four modalities, batch = 2 | **7.4 s** |
| Four modalities, batch = 4 | 14.7 s |

A LoRA(r=8) trunk + ControlNet *training step* on the same GPU is empirically
~0.6–1.0 s (one optimiser update at `bf16` with grad-accum=1). The LoRA
switch shrinks optimizer states and gradient compute on the trunk, but the
trunk **forward pass is unchanged**, which is what dominates step time at
this model scale. Net per-step accounting at `batch=2` (the current setup):

| Path | Time / optimiser step |
|---|---|
| Offline pre-encode (latent streamed, no VAE) | ~0.8 s |
| On-the-fly encode (4 mods × bsz 2) | ~0.8 + 7.4 = ~8.2 s |

→ **~10× slowdown per step**, sustained over every epoch of every training
run. A full 100-epoch run at 1 929 scans / batch 2 = 96 450 batches:
+ 7.4 s × 96 450 = +198 h ≈ +8.3 days on a single A100 / RTX 4090.

### One-time cost of the offline path

| K | Encode wall-clock (1 × RTX 4090) | Bank storage |
|---|---|---|
| 2 | 5.3 h | 32 GB |
| **4** | **9.9 h** | **55 GB** |
| 6 | 14.5 h | 78 GB |
| 8 | 19.1 h | 100 GB |

Splitting across the two server-3 GPUs halves the wall-clock (the converter
is embarrassingly parallel by cohort). Picasso A100 nodes encode ~1.5× faster
than RTX 4090 on this workload (memory-bound GroupNorm; data not measured
locally but anchored to MAISI's reported throughput).

### Sensitivity to the LoRA switch

LoRA(r=8) on the trunk affects only the *trainable* parameter count and
optimizer states. It leaves untouched:

- VAE encoder forward time (the encoder is always frozen);
- VAE encoder peak VRAM (~8–10 GB on the 240³ box at autocast);
- the cost of decoding once per validation step (separate, already
  offloaded to the second GPU via `routines/fm/exhaustive_val/`).

A naive "we have spare VRAM headroom thanks to LoRA, let's encode on the
fly" argument fails on two grounds:

1. **The encoder takes 8–10 GB peak** alongside the trunk (~6 GB)
   + ControlNet (~3 GB) + activations. On 24 GB RTX 4090 the joint
   peak is uncomfortably tight; on 40 GB A100 (effective ~39 GB) it is
   feasible but still slower per step.
2. **The constraint that matters is *time*, not VRAM.** Even with infinite
   VRAM the on-the-fly path stays ≈10× slower per step because the encoder
   forward has to happen serially before the trunk forward starts.

### Diversity argument (the only real downside of offline)

Offline pre-encodes `K` fixed realisations per scan; the model sees the same
`K` nuisance patterns every epoch (mod online flip/translate). On-the-fly
draws a fresh nuisance per step. For VENA this matters less than it would
elsewhere because:

- **Online tier still draws fresh per step** (flip and translate, latent-
  safe per the preflight). For a transform group of effective cardinality
  ~16 (2 flips × 8 translate magnitudes), the joint diversity is
  `K × 16 ≈ 64` per scan with `K=4`, comparable to the unique-augmentation
  surface area you get from `K=64` epochs of online sampling at `p=0.3`.
- **The LoRA adapter capacity is bounded** (r=8 gives ~10⁶ trainable
  parameters); ID-overfitting pressure is low so a finite bank does not
  meaningfully degrade OOD coverage of *smooth* nuisances (bias field,
  gamma, blur) which saturate quickly in K.

**Falsifiable check.** Train at `K ∈ {0, 2, 4, 6}` and plot test PSNR /
SSIM on BraTS-Africa-Glioma (external OOD proxy). The plateau onset is the
operational `K`. Budget for this ablation: 4 × 100 epochs × ~24 h = ~4 days
on one A100 (or 4 days × 2 if running the LoRA+FFT axes jointly).

### Verdict for Q1

| Criterion | Offline | On-the-fly |
|---|---|---|
| Per-run wall-clock (100 epochs) | + 0 h | +~200 h |
| One-time setup | 5–10 h | 0 h |
| Amortisation over N runs | divides by N | does not amortise |
| VRAM headroom on RTX 4090 | comfortable | tight |
| VRAM headroom on A100 40 GB | comfortable | comfortable |
| Diversity per epoch (per scan) | K × 16 | unbounded |
| Diversity-vs-OOD-cost return | high at K ≥ 4 (LoRA argument) | overshoots |

→ **Pre-encode K = 4 variants per CV scan.** Re-evaluate K only if the
plateau check above does *not* flatten by K = 4 on BraTS-Africa-Glioma.

---

## Q2 — Number and type of augmentations

### How many: K = 4, augment all CV scans

The unrefined proposal lands on K = 4 from a stratified-sampling argument
(four distinct nuisance axes, no draw wasted on a near-no-op). The empirical
constraints reinforce that landing:

- K = 4 fits in **5 h on the dual-GPU server** and **55 GB on disk** — both
  trivial.
- K = 6 buys denser coverage of the dominant axis (a second independent
  bias-field draw + a noise/blur combo) but doubles the wall-clock and
  storage and gives diminishing returns under LoRA's low capacity.
- K = 2 saves nothing meaningful (5.3 h, 32 GB) and undercovers the four
  nuisance axes.

**No subset sampling — augment all 2 988 CV scans.** The per-scan benefit is
roughly uniform; subsetting buys nothing and complicates split bookkeeping.

### Variant menu (locked)

Mapping to the existing preflight gate (`flip_lr`, `translate` are
latent-safe; `gamma` is image-domain-only; `rotate_yaw`, `rotate_roll` are
rejected from the latent tier and live exclusively in the offline tier):

| Variant | Transform (library) | Probability of *each* transform inside the variant | Applies to | Why |
|---|---|---|---|---|
| `v0` clean | — | — | — | Already on disk in `*_latents.h5`. Keeps the unaugmented distribution present at training-time (calibration). |
| `v1` field/scanner | TorchIO `RandomBiasField(order=3, coefficients=(-0.5, 0.5))` + TorchIO `RandomGamma(log_gamma=(-0.3, 0.3))` | bias `p=1.0`; gamma `p=0.5` | **inputs only** (t1pre, t2, flair) | Dominant 3 T → 1.5 T low-frequency nuisance. SynthSeg & nnU-Net evidence; covers what the foreground-percentile normalisation does **not** absorb. |
| `v2` contrast-shape | MONAI `RandHistogramShiftd(num_control_points=(8, 12), prob=1.0)` + TorchIO `RandomBrightnessContrast` (small) | histogram shift `p=1.0`; brightness/contrast `p=0.4` | **inputs only** | Field-dependent T1/T2 contrast shape; the Augment-to-Augment monotonic intensity remap. *This is the only transform that perturbs the **shape** of the intensity transfer function*; pure gamma does not. Critical because foreground-percentile normalisation re-anchors the endpoints but cannot undo a piecewise-linear remap. |
| `v3` SNR/resolution | TorchIO `RandomNoise(std=(0, 0.05))` + TorchIO `RandomAnisotropy(downsampling=(1.5, 4.0))` + TorchIO `RandomBlur(std=(0, 1.5))` + TorchIO `RandomMotion(num_transforms=(1, 3))` (low p) | noise `p=1.0`; anisotropy `p=0.7`; blur `p=0.4`; motion `p=0.1` | **inputs only** | Low-field SNR drop + thick-slice 1.5 T acquisitions + plausible patient-motion artefact. The motion term is the cheapest clinical-realism transform from TorchIO. |
| `v4` anatomy | TorchIO `RandomElasticDeformation(num_control_points=7, max_displacement=4.0)` + TorchIO `RandomAffine(scales=(0.9,1.1), degrees=10, translation=8)` | elastic `p=0.7`; affine `p=0.7` | **inputs AND target AND WT mask** (joint, via TorchIO `Subject`) | The one geometric augmentation that is *not* latent-equivariant (the preflight rejected rotation in the latent tier). Mask uses `LabelMap` so it is nearest-neighbour-warped together with the images. |

The preflight gate (`vena.data.augment.config.build_pipeline_from_yaml`)
already supplies the latent-safe allowlist; the **offline bank-builder runs
in image space and is not gated by that allowlist** — it is, by
construction, the place where image-domain-only and rejected-from-latent
transforms live. State this contract explicitly in the engine's module
docstring so a future reader does not re-enforce the gate at the wrong
layer.

### Train-time sampling

At sample-time the trainer's `MultiCohortLatentDataset` picks one of
`{v0, v1, v2, v3, v4}` for each scan with weights `[0.2, 0.2, 0.2, 0.2, 0.2]`
(uniform over the bank; the clean-fraction lever is **20 %** of seen samples
to keep ID calibration intact — the unrefined proposal's 30 % is overly
conservative now that K = 4 distributes nuisance more evenly). Then the
existing online latent pipeline (`flip_lr` p=0.5, `translate` ≤ 8 vox
p=0.5) runs on top of whichever variant was sampled — **unchanged from
today's training**.

Bench the clean fraction in {0.0, 0.1, 0.2, 0.3} on a 4-epoch smoke; expect
the OOD–ID gap to be smallest at 0.1–0.2 with LoRA capacity bounded.

### Two blind spots from the unrefined proposal, sharpened

1. **Percentile-foreground normalisation partially undoes global
   photometric perturbation.** The encoder does `foreground_only=True`,
   `(lower=0, upper=99.5)` percentile rescaling. A *spatial* nuisance
   (bias field, blur, anisotropy) **survives** because it deforms the
   intensity field non-uniformly. A *global* nuisance (constant brightness
   shift, gamma applied to the whole volume) is partially renormalised
   away because the 99.5-percentile re-anchors the top of the dynamic
   range. **This is why `v2`'s histogram shift matters more than `v1`'s
   gamma**: the histogram shift perturbs the *shape* of the transfer
   function (relative ordering of percentiles 0–25–50–75–99.5 is broken),
   and that perturbation survives renormalisation.

   **Acceptance check for the implementer.** Encode one volume clean and
   then under a global `γ=0.7` gamma; measure the latent-space L2 distance.
   If it is < 5 % of the inter-patient L2 distance, drop gamma from `v1`
   (keep bias only) and rely on `v2` for non-spatial contrast variation.
   Cost: one notebook cell.

2. **Bias field ≠ Bloch-equation contrast change.** The unrefined proposal
   flags this. The pragmatic decision is to *not* implement a Bloch-based
   re-synthesis for v1 (out of scope, expensive), and instead rely on the
   compositional coverage of `v1 (spatial bias) + v2 (transfer-function
   shape) + v3 (SNR + resolution)`. Acknowledge the gap in the manuscript
   §Limitations.

### What we are *not* adding

- **`RandomGhosting`, `RandomSpike`** — secondary clinical artefacts. They
  add real coverage but at the cost of bank-build time and storage; the
  field-strength axis dominates VENA's OOD spectrum, not artefact
  prevalence. Hold for v2 of the bank if the K-ablation suggests the
  artefact axis matters.
- **Inpainting/masking augmentations** (random erasing, cutout). Not
  field-relevant; would distort the conditioning–target spatial coherence.
- **Rician noise.** TorchIO's Gaussian on magnitude is adequate at our
  target field strengths (≥ 1.5 T); the divergence from Rician is < 1 %
  at SNR > 5 (fact, Gudbjartsson & Patz 1995, *Magn. Reson. Med.* 34:910).

---

## Q3 — Libraries

**Agree with the unrefined proposal: TorchIO is the primary library;
MONAI's `RandHistogramShiftd` fills the one TorchIO gap.**

### TorchIO (Pérez-García et al. 2021, DOI 10.1016/j.cmpb.2021.106236)

- Native `Subject` abstraction. Register `t1pre`, `t2`, `flair` as
  `ScalarImage` (with `include=`), `t1c` as `ScalarImage` (target), and
  the WT mask as `LabelMap`. Spatial transforms apply consistently across
  all members; the mask is nearest-neighbour-warped. **This eliminates the
  whole class of mask-desync bugs by construction.**
- Physically motivated `RandomBiasField` (polynomial low-frequency, Sled et
  al. 1998 model), `RandomMotion` (k-space rejection sampling, Shaw et al.
  2019), `RandomAnisotropy` (downsample + back-up sample, matching the
  thick-slice acquisition story).
- Mature, BSD-licensed, ≈1.5 k citations as of 2026.

### MONAI's `RandHistogramShiftd` (already in our stack)

- Implements the piecewise-linear monotonic intensity remap from the
  Augment-to-Augment paper; TorchIO has no equivalent. Specifically:
  draws `n_control_points` uniformly in `[0, 1]`, sorts them, and
  remaps the intensity histogram through that monotonic transfer.
- Lives in `monai.transforms.intensity.dictionary`; integrating it
  alongside TorchIO is a 10-line adapter that wraps the dict transform
  in a `Subject`-compatible callable.

### Dependencies to add

```toml
# pyproject.toml [project.dependencies]
torchio = ">=0.19.6"
```

MONAI is already pinned. TorchIO is BSD-3-Clause, no GPL contamination.
Per `.claude/rules/coding-standards.md` rule 6, declare it in
`pyproject.toml` in the same change and document the rationale in the
commit body.

### What we do *not* use

- **`batchgenerators`** (the nnU-Net augmentation engine). Strong but
  numpy-first, no MRI-specific physical models, no `Subject`-style
  joint-spatial handling, and adds a third aug library. Skip.
- **Hand-coded TorchVision-style transforms.** Forbidden under
  `coding-standards.md` rule 6 (library-first).
- **Custom monotonic remap.** Use MONAI's `RandHistogramShiftd`;
  reimplementing it would re-derive a published primitive for no gain.

---

## Q4 — Wall-clock comparison, end-to-end

Same numbers as Q1, rendered as the two pipelines:

```
[Pipeline A: offline pre-encode]                  one-time   per-run
  augment-image  ─▶ encode ─▶ append to bank H5   10 h       0
  train (latent streamed, no VAE in loop)         0          ~0.8 s/step
  ─────────────────────────────────────────────────────────────────
  Net cost over 100-epoch run on RTX 4090         10 h       +21 h

[Pipeline B: on-the-fly encode]                   one-time   per-run
  augment-image (per batch)  ─▶ encode (per modality) ─▶ trunk  ~8 s/step
  ─────────────────────────────────────────────────────────────────
  Net cost over 100-epoch run on RTX 4090         0          +220 h
```

Pipeline A reaches **its own break-even after the first training run**
(10 h amortised over +0 h/run vs. +200 h/run for B). VENA's actual run
budget includes (at minimum) the LoRA-vs-FFT ablation × the
λ_v/λ_t/SWAN-encoding axes from proposal §7, comfortably ≥ 4 full runs —
so the offline path saves ~800 GPU-hours over the project lifetime.

### Picasso scaling

A100 40 GB per the proposal §5 budget is ~5 days for a full 4× A100 run.
Per-modality encode on A100 is ~1.5× faster than RTX 4090 (data
extrapolated from MAISI's release notes; verify in a 30-min smoke before
the production bank-build). So the K=4 bank-build on **one A100** is
~7 h, **four A100s parallelised by cohort** ~2 h. This fits in a single
overnight Picasso slot; submit via the `picasso-sbatch` skill.

### Risk: bank gets stale

The bank is keyed by:

- the VAE checkpoint SHA-256 (frozen for the project),
- the augmentation YAML SHA-256 (per-variant `p` and per-transform
  hyperparameters),
- the source image H5 path + `crop_box` + percentile-normalisation params
  (already recorded in each cohort's image-H5 root attrs).

Persist all three into the bank H5's `decision.json` v0.1.0; the FM
trainer asserts they match its config at startup. Re-building the bank is
~10 h, no big deal — but the gate prevents silent training on a
mis-paired bank.

---

## Implementation blueprint

A new routine, `routines/offline_aug/maisi/`, following
`.claude/rules/preflight-pattern.md`:

```
routines/offline_aug/maisi/
├── cli.py
├── configs/
│   ├── ucsf_pdgm.yaml
│   ├── brats_gli.yaml
│   ├── upenn_gbm.yaml
│   ├── ivy_gap.yaml
│   └── lumiere.yaml
├── slurm/
│   ├── launcher_offline_aug.sh
│   └── worker_offline_aug.sh
└── engine/
    ├── __init__.py
    └── offline_aug_engine.py
```

Library code under `src/vena/data/augment/offline/`:

```
src/vena/data/augment/offline/
├── __init__.py
├── bank_builder.py        # main loop: subject build → TorchIO Compose → encode → H5 append
├── variants.py            # the four `make_variant_X(p_overrides) -> tio.Compose` builders
├── torchio_adapters.py    # MONAI RandHistogramShiftd wrapped as a TorchIO Transform
└── h5_writer.py           # latents_aug.h5 append-mode writer (CSR offsets per (scan, variant))
```

### `latents_aug.h5` schema (v0.1.0)

Same principles as `h5-design-principles.md`:

- Root attrs: `schema_version="0.1.0"`, `created_at`, `producer`,
  `config_json` (the augmentation YAML), `git_sha`,
  `source_latents_h5_sha256`, `vae_checkpoint_sha256`,
  `image_h5_sha256`.
- Groups:
  - `latents/{v1,v2,v3,v4}/{t1pre,t1c,t2,flair}` shape
    `(N_scans, 4, 60, 60, 40)`, fp16, chunked `(1, 4, 60, 60, 40)`,
    gzip-4 (token compression vs. raw — gzip buys ~5 % on near-Gaussian
    latents but lets us scan the whole file with `ptdump`). `t1c` is
    written **only for `v4`**; the loader resolves `v1/v2/v3` t1c lookups
    against the clean `*_latents.h5` (per the input-only contract).
  - `masks/m_wt/{v0,v4}` shape `(N_scans, 60, 60, 40)`, int8.
    Per-class avg-pool downsampling of the warped tumour mask
    (the existing `mask_downsampler` used by the encode routine).
  - `ids` vlen str, length `N_scans`, identical order to the source
    `*_latents.h5`.
- `decision.json` (alongside the H5) declares the per-variant transform
  list, hyperparameters, and the sampling weights for the trainer.

### `MultiCohortLatentDataset` integration

The change is small and additive:

1. Each cohort entry in `corpus_*.json` gains an optional
   `latents_aug_h5` path; when present, the per-cohort dataset is
   wrapped in an `AugBankSampler` that draws variant ∈ {v0..v4} per
   `__getitem__` with the configured weights, then reads the per-variant
   latent slabs.
2. `data.augmentation_config_path` and `data.preflight_decision_path`
   (existing) continue to gate the **online latent pipeline** —
   unchanged. They do not gate the bank because the bank's transforms
   live in image space.
3. The `decision.json` v0.3.0 emitted by `routines.fm.train` gains
   `latents_aug_h5_path` and `latents_aug_h5_sha256` keys (bump
   `schema_version` to 0.4.0 — this is a forward-compatible add but
   keeps round-trippability honest).

### CSV logging hooks

The existing `AugmentationTracker` callback writes
`metrics/augmentations_per_epoch.csv`. Extend its column set to record
which bank variant (`v0..v4`) was drawn per sample; the per-epoch
aggregate gives a free check that the sampling weights match
expectation (and surfaces under-sampling of any single variant due to
a worker-seed bug).

---

## Test plan

- **Unit** (`tests/data/augment/offline/`, marker `unit`):
  - Each variant builder accepts a probability override and returns a
    `tio.Compose` with the right number of `RandomXxx` transforms.
  - `torchio_adapters.MonaiHistogramShift` round-trips a `Subject` and
    leaves the `LabelMap` unmodified.
  - `H5Writer` writes a deterministic schema and refuses to write a
    second time without `overwrite=True`.
- **GPU smoke** (marker `gpu`, on icai-server):
  - Build the bank for 4 patients of UCSF-PDGM, K=4, end-to-end.
  - Confirm `latents_aug.h5` validates against the schema.
  - Confirm the trainer with `augmentation_config_path` + `latents_aug_h5`
    loads the bank, samples from it, and the per-epoch CSV records
    `{v0,v1,v2,v3,v4}` non-zero counts.
- **Diagnostic notebook** (one-off, not committed):
  - Encode one volume clean and under `γ=0.5/0.7/1.5`; report latent L2.
    Confirms or refutes the "percentile-foreground normalisation absorbs
    global gamma" hypothesis. Drop gamma from v1 if confirmed.

---

## What is unchanged

- The latent-equivariance preflight at
  `/media/hddb/mario/artifacts/latent_aug_equivariance/LATEST/decision.json`
  remains the source of truth for what the *online* latent pipeline may
  do; the bank's image-domain transforms are out of its scope.
- `MultiCohortLatentDataModule` is still the only training data path
  (`extensibility.md`).
- The FM trainer keeps the trunk-trainable path (LoRA can be added as a
  separate change; the bank is agnostic to whether the trunk is
  full-rank or LoRA-adapted).

---

## Open questions for the implementer

1. **LUMIERE longitudinal handling.** Each session is its own row in the
   latent H5 (per `add-dataset` playbook). Confirm that bank-building
   keys on the (patient, session) compound, not on patient alone, so
   the bank does not collapse two visits to the same realisation.
2. **WT mask provenance for `v4`.** The mask warped under elastic+affine
   must come from the **image-space** WT (the `masks/tumor` group in
   the cohort image H5), not from the latent-space `m_wt` (which is the
   downsampled int8). Confirm by checking that the v4 mask round-trips
   to the same WT-Dice as v0 within ε on a 4-patient smoke.
3. **Cohort-specific variant overrides?** UCSF-PDGM and BraTS-GLI are
   3 T-dominant; LUMIERE is 1.5/3 T mixed; IvyGAP has a known acquisition
   heterogeneity. A first-cut single-config-fits-all is fine; consider
   per-cohort overrides only if the K-ablation suggests one cohort is
   over-augmented.

---

## References (additions over `unrefined_proposal.md`)

- Gudbjartsson H., Patz S. (1995). The Rician distribution of noisy MRI
  data. *Magnetic Resonance in Medicine* 34(6):910–914.
  DOI: 10.1002/mrm.1910340618. (Justifies Gaussian-on-magnitude as
  adequate at ≥ 1.5 T SNR.)
- Sled J.G., Zijdenbos A.P., Evans A.C. (1998). A nonparametric method
  for automatic correction of intensity nonuniformity in MRI data.
  *IEEE Trans. Med. Imaging* 17(1):87–97. DOI: 10.1109/42.668698.
  (Underlying model TorchIO's `RandomBiasField` simulates.)
- Shaw R. et al. (2019). MRI k-space motion artefact augmentation:
  Model robustness and task-specific uncertainty.
  *MIDL 2019*. (Motion simulation in TorchIO `RandomMotion`.)
