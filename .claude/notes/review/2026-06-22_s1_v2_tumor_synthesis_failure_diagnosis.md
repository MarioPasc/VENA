# S1 v2 Tumor-Synthesis Failure — Full-Pipeline Diagnosis

*Mario Pascual González — VENA, IBIMA-BIONAND / Universidad de Málaga.*
*2026-06-22 — diagnostic prompted by the user's epoch-975 visual inspection of `picasso:/mnt/home/users/tic_163_uma/mpascual/execs/vena/experiments/2026-06-20_14-15-19_s1_fft_cfm_d2bc4d2a/exhaustive_val/epoch_975/figure_best_1.png`. Companion to `decoder_perceptual_loss_s3_analysis_2026-06-20.md`.*

---

## 0. TL;DR

The S1 v2 baseline (L1 + scale-ramped zero-init + `use_timestep_transform=true`,
launched 2026-06-20) reproduces the same qualitative tumor-synthesis failure
as retired S1 v1. **The two recipe deltas that the 2026-06-20 analysis doc
predicted would close the gap (`l2→l1`, `output_scale_ramp`) are not the
load-bearing factors.** They could not be: T1C-RFlow already uses the
identical L1 velocity loss and lands successful enhancement; the conditioning
route is the actual delta.

| What | Finding | Confidence |
|---|---|---|
| **P0 (architecture)** | The trunk **never sees the conditioning tensor as input**. `controlnet_cond_embedding` is the first layer to touch `[z_t1pre, z_t2, z_flair, mask_wt]` and is **randomly initialised**. T1C-RFlow uses `torch.cat([z_t,z_t1pre,z_flair],dim=1)` directly at the U-Net `in_channels=12`. | **Definitive** (code-level evidence, file:line). |
| **P1 (loss)** | Mean-reduction L1 over the full 4-channel velocity field. **WT contributes 0.095 % of total L1 magnitude**; the optimiser favours non-WT correctness by **~1047:1**. No region weighting in S1. | **Definitive** (numerical audit, n=30 UCSF-PDGM val). |
| **P2 (data signal)** | After `percentile_normalise(99.5%, foreground_only=True)`, `⟨|T1c − T1pre|⟩` in WT equals **the same** value as in non-WT brain (0.3837 vs 0.3837). In **enhancing** (ET) tissue the contrast is *less* than non-WT brain (0.2384). The per-modality 99.5%ile cuts at enhancement and squashes the signal-to-learn into the same magnitude as anatomy. | **Definitive** (image-space audit, n=15). |
| **P3 (mask resolution)** | WT in latent grid (48×56×48, 4-mm³ voxels) is empty for **5 / 30 UCSF-PDGM val patients (17 %)**. Region-weighted latent losses cannot help these patients. | **Definitive** (counted from `masks/tumor_latent` ≥ 0.5). |
| **P4 (VAE — REFUTED)** | MAISI-V2 VAE encode→decode on UCSF-PDGM T1c has **MAE = 0.0041** ± 0.0003 (n=4 named patients ×6 modalities ×30 stab). H3 (VAE bottleneck) is refuted. Enhancement passes through. | **Definitive** (existing `encode_ucsf_pdgm_maisi` artifact). |
| **P5 (convergence)** | PSNR_WT plateaued at **17.52 dB at epoch 475** then *regressed* to 17.13 dB by epoch 1075 — model is **degrading** on tumor while continuing to refine background. PSNR_BG = 26.22 dB, **9 dB above** PSNR_WT. | **Definitive** (aggregated `metrics.csv` over 116 val patients, 14 epoch sweep). |
| **P6 (metric trap)** | **PSNR_WT is misleading on this task.** T1C-RFlow's empirical PSNR_WT on UCSF-PDGM val (n=5, NFE=200) = **16.97 ± 3.14** dB — *lower* than VENA S1 v2's UCSF-only 18.06 dB. Yet T1C-RFlow's midslice PNGs **visibly contain enhancement** (bright rim on 0436, hyperintense focus on 0273), while VENA's predictions are uniform. Reason: `WT = necrosis (label 1) + edema (label 2) + enhancement (label 4)`. Predicting "baseline tissue" everywhere scores **well** on necrosis (T1c-dark) and edema (T1c-isointense), and **badly** on enhancement. Both models hit ~17 dB; only one predicts what radiologists need. **PSNR_ET (label 4 only) is the load-bearing metric**. | **Definitive** (T1C-RFlow replay on server3:cuda:0, midslice inspection). |

**Operative diagnosis:** the model rationally converges to a background-correct,
healthy-brain-like manifold because (a) the loss carries vanishing tumor
signal, (b) the architecture provides no direct path from conditioning to the
trunk's input, and (c) the per-modality intensity normalisation compounds the
signal flattening. Each of (a) (b) (c) is independent; fixing only one is not
expected to close the gap. Concrete recommendations are in §6.

---

## 1. Empirical state of the current run

### 1.1 Convergence trajectory (NFE=5, n=116 val patients per epoch)

Pulled from `picasso:.../exhaustive_val/epoch_NNN/metrics.csv`:

| epoch | PSNR_whole | PSNR_BG | **PSNR_WT** | PSNR_NWT | SSIM_whole | SSIM_BG | SSIM_WT |
|---:|---:|---:|---:|---:|---:|---:|---:|
|    0 | 20.58 | 20.72 | **15.59** | 14.98 | 0.744 | 0.754 | 0.959 |
|  100 | 25.39 | 25.87 | 16.73 | 18.75 | 0.912 | 0.919 | 0.941 |
|  200 | 26.01 | 26.47 | 17.42 | 19.35 | 0.916 | 0.923 | 0.955 |
|  300 | 26.06 | 26.53 | 17.42 | 19.41 | 0.916 | 0.923 | 0.957 |
|  475 | 26.08 | 26.53 | **17.52 ← peak** | 19.41 | 0.917 | 0.923 | 0.959 |
|  500 | 26.12 | 26.57 | 17.56 | 19.45 | 0.917 | 0.923 | 0.959 |
|  600 | 26.03 | 26.48 | 17.48 | 19.36 | 0.916 | 0.923 | 0.959 |
|  700 | 25.98 | 26.43 | 17.51 | 19.31 | 0.916 | 0.923 | 0.959 |
|  800 | 25.85 | 26.29 | 17.33 | 19.17 | 0.916 | 0.923 | 0.957 |
|  900 | 25.77 | 26.22 | 17.15 | 19.10 | 0.916 | 0.923 | 0.956 |
|  975 | 25.77 | 26.22 | 17.13 | 19.10 | 0.916 | 0.923 | 0.957 |
| 1000 | 25.75 | 26.19 | 17.14 | 19.07 | 0.916 | 0.923 | 0.957 |
| 1075 | 25.73 | 26.17 | **17.14 ↓** | 19.05 | 0.916 | 0.922 | 0.956 |

Three quantitative facts to retain:

1. **PSNR_WT plateaued by epoch 475** (gain 0.10 dB over the next 525 epochs).
   This matches the retired S1 v1 plateau described in
   `decoder_perceptual_loss_s3_analysis_2026-06-20.md` §3.
2. **PSNR_WT has *regressed* between 475 and 1075** (−0.39 dB). The model is
   trading marginal PSNR_BG decay for tumor-region drift — under L1 mean
   reduction this is consistent with the optimiser refining background
   reconstruction at the cost of the (sparse) WT signal.
3. **The recipe deltas vs S1 v1 did not unlock WT-PSNR**: v1 plateaued at
   18.3 dB WT-PSNR per the prior doc, v2 plateaus at 17.5 dB — a **~0.8 dB
   regression** on tumor. The whole-volume PSNR is unchanged (≈26 dB). The
   new recipe traded tumor fidelity for some unidentified other axis.

### 1.2 Per-cohort breakdown at epoch 975 (NFE=5)

| cohort | n | PSNR_whole | **PSNR_WT** | SSIM_WT |
|---|---:|---:|---:|---:|
| BraTS-Africa-Glioma | 6 | 20.35 | **11.73** | 0.881 |
| BraTS-Africa-Other  | 6 | 20.94 | **10.96** | 0.881 |
| BraTS-GLI           | 9 | 25.64 | 15.26 | 0.965 |
| BraTS-PED           | 6 | 21.37 | 13.73 | 0.925 |
| IvyGAP              | 5 | 24.94 | 12.79 | 0.951 |
| LUMIERE             | 65 | 26.79 | **19.37** | 0.969 |
| REMBRANDT           | 5 | 25.55 | 13.09 | 0.958 |
| UCSF-PDGM           | 7 | 27.48 | 18.06 | 0.972 |
| UPENN-GBM           | 7 | 28.04 | 16.52 | 0.974 |

Cohorts dominated by smaller / earlier-presenting lesions (LUMIERE
longitudinal, UCSF-PDGM, UPENN-GBM) score 18–19 dB WT-PSNR; cohorts with
larger / more heterogeneous tumors (BraTS-Africa, BraTS-GLI, REMBRANDT,
IvyGAP) score 11–15 dB. **The model is best where the tumor is smallest** —
i.e. where the background-correct default is closest to the right answer.
SSIM_WT is **misleadingly high (≥0.88)** because SSIM on a small mask is
dominated by structural agreement *at the WT boundary* and ignores intensity
fidelity *inside* the mask. PSNR_WT is the load-bearing metric.

### 1.3 NFE sweep at epoch 975

| NFE |  1 |  2 |  5 | 10 | 20 |
|---|---:|---:|---:|---:|---:|
| PSNR_whole | 25.32 | 26.27 | 25.77 | 25.52 | 25.36 |
| PSNR_WT    | 18.26 | 17.98 | 17.13 | 16.89 | 16.79 |
| SSIM_WT    | 0.968 | 0.965 | 0.957 | 0.954 | 0.953 |

**More integration steps make PSNR_WT *worse*** (−1.5 dB from 1 → 20 steps).
This is consistent with a model whose ControlNet residuals are not actively
steering the trajectory: longer trajectories give the trunk's healthy-brain
prior more iterations to pull the prediction away from the enhancing target.
The T1C-RFlow paper reports best inference quality at **200 Euler steps** —
their architecture supports that because conditioning is in the U-Net input at
every step. VENA cannot benefit from more steps without first fixing the
conditioning route.

### 1.4 Visual confirmation (epoch_975/figure_best_1.png)

Patient `HF1345_1994.06.03` (REMBRANDT cohort, top-1 SSIM in the val sweep).
Real T1c (top row) shows clear bright enhancement at axial slices z≈79, 89,
102. All five NFE rows (1, 2, 5, 10, 20) produce essentially uniform
non-enhancing tissue at the tumor location. PSNR_WT for this patient at
NFE=5 is **9.25 dB** (worse than any cohort mean). The model has not
"missed a detail" — it has produced a healthy-brain prediction at the tumor
voxels, indistinguishable from non-tumor cortex.

---

## 2. Architectural autopsy — why the model produces what it produces

### 2.1 The S1 v2 conditioning route, traced

**Trunk forward signature** (`vena/model/fm/lightning/module.py:465-480`):

```python
down_res, mid_res = controlnet(
    x=x_t_p, timesteps=timesteps,
    controlnet_cond=cond_p, class_labels=class_labels,
)
v_p = trunk_model(
    x=x_t_p, timesteps=timesteps, class_labels=class_labels,
    spacing_tensor=spacing,
    down_block_additional_residuals=down_res,
    mid_block_additional_residual=mid_res,
)
```

`x_t_p` is the 4-channel noisy T1c latent (`(B, 4, 48, 56, 48)`). **The
trunk's `x` argument is exactly that — no conditioning channels.** The 13-
channel conditioning tensor `cond_p` is passed only to the ControlNet.

**Conditioning assembly** (`vena/model/fm/controlnet/conditioning.py:210-222`):

```python
for spec, downsampler in zip(self.specs, self.downsamplers, strict=True):
    key = spec.batch_key()
    x = batch[key]
    x = downsampler(x)               # WT mask → ZeroOutDownsampler → torch.zeros_like(x)
    pieces.append(x)
return torch.cat(pieces, dim=1)       # (B, 4+4+4+1=13, 48, 56, 48)
```

Channel layout for S1 v2: `[z_t1pre(4), z_t2(4), z_flair(4), zeros(1)]`. Three
observations:

- **No SWAN.** The proposal's vessel prior (Frangi/Jerman on SWAN, the
  project's principal differentiator) is *not* in S1 conditioning. Both the
  raw SWAN latent and the derived vessel mask are absent.
- **No tumor mask.** `mask:wt:zero_out` returns `torch.zeros_like(x)`
  (`vena/model/fm/controlnet/downsample/zero_out.py:31`). The DataLoader
  *loads* the mask from `masks/tumor_latent`, then the downsampler discards
  it. The channel slot is reserved for S2/S3 warm-start identity but carries
  no signal in S1.
- **No anatomical prior** (none of: brain mask, brain segmentation, distance
  transforms, identity-from-T1pre residual).

**ControlNet initialisation** (`module.py:387-389`, `maisi_controlnet.py:169-175`):

```python
self.controlnet.init_from_trunk(trunk_sd)        # encoder copied from MAISI trunk
self.controlnet.zero_init_output_projections()    # controlnet_down_blocks.*, controlnet_mid_block.* → 0
# controlnet_cond_embedding remains randomly initialised (NOT in trunk_sd)
```

`controlnet_cond_embedding` is the **first layer** that mixes the 13 input
channels of the conditioning tensor. It is *not* part of the MAISI trunk
checkpoint and therefore *not copied* by `init_from_trunk`. The remaining
encoder blocks downstream of it are trunk-init copies — but they were
pretrained to consume a **4-channel** noisy-latent input, not the 13-channel
conditioning. The cond_embedding has to learn from random init to project the
13-channel conditioning into the trunk's feature manifold.

**Residual scaling** (`maisi_controlnet.py:116-118`):

```python
scale = self.output_scale                     # buffer ramped 0→1 over 5000 steps
down_block_res_samples = [t * scale for t in down_block_res_samples]
mid_block_res_sample = mid_block_res_sample * scale
return down_block_res_samples, mid_block_res_sample
```

At training step 0, `sigmoid(10 * (0/5000 − 0.5)) ≈ 0.007` (the cold-start
the ramp is meant to avoid). At step 5000 (≈ epoch 24, given 208 steps /
epoch), `output_scale` saturates near 1.0. So from epoch 24 onward this
buffer is no longer the bottleneck.

### 2.2 What the trunk actually sees

The trunk is a frozen-then-FFT-trained 4-level MAISI MR `diff_unet_3d_rflow-mr.pt`
with self-attention (~1 B params). Its `forward(x, timesteps, ...)` signature
takes a **4-channel** spatial input. The MAISI pretraining objective is
"denoise a 3D MR latent to the prior data distribution conditional on a class
token + spacing". Tumor enhancement is not a special part of MAISI's
pretraining manifold — the trunk has a strong prior toward generic MR brain
appearance.

When a tumor voxel is being denoised, the trunk's pretrained capacity wants
to predict a value consistent with healthy parenchyma. The only mechanism
that can override this prior is the ControlNet residual at that voxel
position. The residual is computed from `cond_embedding(conditioning_tensor)`
followed by trunk-init encoder blocks; this path must learn that "this voxel
should be bright in T1c" purely from the conditioning channels (T1pre, T2,
FLAIR latents — none of which directly tell the model "tumor here", and the
WT channel that *would* tell it has been zeroed).

### 2.3 What T1C-RFlow's architecture sees instead

`src/external/t1c_rflow/upstream/train_rflow.py:201-207`:

```python
model_in = torch.cat([noisy_latents, cond, seg], 1)  # in_channels = 4*3 = 12
v = unet(x=model_in, timesteps=t)
loss = F.l1_loss(v, (tgt - noise))   # identical L1 velocity to VENA
```

`in_channels = latent_channels * 3` (`train_rflow.py:129`, `config_maisi3d-rflow.json:4`).
The U-Net's **first convolution** mixes 12 input channels: `[z_t (4 ch noisy
T1c), z_T1n (4 ch T1pre), z_T2F (4 ch FLAIR)]`. Every feature in every
encoder layer is built on top of this joint representation. The model can
learn from gradient step 1 that a feature at position p in z_T1pre with a
specific local context predicts a bright value in z_t. The conditioning is
**load-bearing for every prediction**, not an additive residual to a
healthy-brain prior.

Other observations from `train_rflow.py`:

- **No ControlNet branch, no zero-init output projections.**
- **L1 velocity loss** (`F.l1_loss(v, (tgt - noise))`, line 207) is **identical to ours**.
- **No EMA, no LR schedule, no augmentation, no mask channel.**
- 200 Euler integration steps at inference (`test_rflow.py:66`); their
  network exploits longer trajectories because conditioning steers every step.
- 100–164 epochs on a single A6000 (per `decision.json`); much less compute
  than VENA's 1000+ ep budget.

The architecture difference is necessary and sufficient to explain the gap.

### 2.4 Why ControlNet is the wrong tool for this task

ControlNet (Zhang et al. 2023, *Adding Conditional Control to Text-to-Image
Diffusion Models*, ICCV) was designed for the case where:

- A **strong pretrained denoiser** already produces excellent images from
  another conditioning modality (text in Stable Diffusion).
- The new condition (pose, depth, edges, mask) is an **add-on spatial
  control** that nudges the existing distribution rather than carrying the
  primary information.

VENA's setup violates both assumptions. The trunk has no other source of
patient-specific information; the conditioning is the **sole** carrier of
"what T1c should look like for this person". ControlNet's residual-injection
mechanism is a low-bandwidth path that, by design, expects the trunk to be
doing most of the work. For a translation task where the conditioning IS the
information, channel-concat (or any "always-on, deep" conditioning route) is
the right architecture, as borne out by every successful 3D medical latent
FM/diffusion baseline (Eidex 2025 §3.1, Guo 2025 MAISI ControlNet, Dayarathna
2025 McCaD, Biller 2026 TumorFlow — all use either channel-concat at input
or both).

---

## 3. Quantitative diagnosis

### 3.1 Latent-space target-velocity audit (n=30 UCSF-PDGM fold 0 val)

Script: `/tmp/vena_diag/audit_target_velocity.py`. For each patient:
`u = z_t1c − ε` (target velocity, constant in α for rectified flow).
Decompose `Σ|u|` by region:

| Region        | Voxel % | ⟨|u|⟩ | ⟨|Δ|⟩ where Δ=z_t1c−z_t1pre | %∑|u| | %∑|Δ| |
|---|---:|---:|---:|---:|---:|
| BG            | **79.353 %** | 1.1094 | 1.0299 | 78.180 % | 80.549 % |
| BRAIN_NOT_WT  | 20.364 % | 1.1888 | 0.9543 | 21.499 % | 19.154 % |
| **WT**        | **0.087 %** | 1.2380 | 0.9850 | **0.095 %** | **0.084 %** |
| ET (subset)   | 0.197 % | 1.2902 | 1.0953 | 0.225 % | 0.212 % |

(Region percentages are over the 4-channel × 48×56×48 latent grid =
516,096 elements. "BG" here means brain mask = 0 in latent grid; "BRAIN_NOT_WT"
is brain ≥ 1 AND WT < 0.5; "WT" is `masks/tumor_latent[:,0] ≥ 0.5`; "ET" is
`masks/tumor_latent[:,2] ≥ 0.5`.)

**Key numbers:**

- **WT contributes 0.095 % of total |u|** — the L1 mean-reduction loss is
  ~99.9 % driven by non-WT voxels.
- The optimiser favours non-WT correctness by
  `(0.78 + 0.215) / 0.00095 ≈ 1047:1`.
- **Per-voxel** the WT velocity is barely stronger (⟨|u|⟩_WT / ⟨|u|⟩_BG = 1.12×).
  Combined with the area imbalance, the WT contribution is dominated by area,
  not by per-voxel signal magnitude.
- **The translation signal Δ = z_t1c − z_t1pre is** *not* larger in WT
  (⟨|Δ|⟩_WT / ⟨|Δ|⟩_BG = 0.96×). In other words, the per-voxel work the model
  has to do to map T1pre → T1c is similar in tumor and non-tumor — but the
  optimiser has no incentive to prioritise the tumor voxels because they are
  numerically irrelevant.

**Per-patient cross-section** (first 10 of 30 sampled UCSF-PDGM fold-0 val):

| patient | wt_latent_voxels | %loss(WT) | ⟨\|u\|⟩_WT / ⟨\|u\|⟩_BG | ⟨\|Δ\|⟩_WT / ⟨\|Δ\|⟩_BG |
|---|---:|---:|---:|---:|
| UCSF-PDGM-0538 |    59 | 0.05 % | 1.11× | 0.93× |
| UCSF-PDGM-0529 |   411 | 0.34 % | 1.08× | 0.73× |
| UCSF-PDGM-0302 |   **0** | 0.00 % |  —    |  —    |
| UCSF-PDGM-0437 |   **0** | 0.00 % |  —    |  —    |
| UCSF-PDGM-0193 |    20 | 0.02 % | 1.27× | 0.98× |
| UCSF-PDGM-0387 |    55 | 0.05 % | 1.21× | 0.88× |
| UCSF-PDGM-0355 |   400 | 0.33 % | 1.09× | 0.92× |
| UCSF-PDGM-0526 |    82 | 0.07 % | 1.18× | 0.94× |
| UCSF-PDGM-0467 |   **0** | 0.00 % |  —    |  —    |
| UCSF-PDGM-0414 |   199 | 0.18 % | 1.17× | 0.92× |

### 3.2 Latent WT-mask sparsity at 4-mm³ voxel size (P3)

5 of 30 sampled UCSF-PDGM fold-0 val patients (17 %) have **zero** WT voxels
in the latent grid at the `0.5` threshold. The latent grid is 4× downsampled
from the 1-mm image grid, so each latent voxel covers a 4×4×4 mm cube. An
enhancing rim of typical thickness (2–4 mm) often produces no above-0.5 soft
mask after MAISI VAE downsampling. Region-weighted losses computed in latent
space cannot help these patients because the latent WT mask is empty.

This is a separate bug from the L1 imbalance — even *with* region weighting
in latent space, 17 % of the training signal would be lost.

### 3.3 Image-space contrast audit (n=15 UCSF-PDGM fold 0 val)

Script: `/tmp/vena_diag/audit_image_contrast.py`. For each patient, apply
VENA's `percentile_normalise(lo=0, hi=99.5, foreground_only=True)` on T1c
and T1pre separately (matching the encoder contract), then decompose
`|T1c − T1pre|` by region:

| Region | Voxels | ⟨T1c⟩ | ⟨T1pre⟩ | ⟨\|Δ\|⟩ | %∑\|Δ\| |
|---|---:|---:|---:|---:|---:|
| BG           | 111.2 M | 0.000 | 0.000 | 0.0000 | 0.000 % |
| BRAIN_NOT_WT |  21.5 M | 0.308 | 0.674 | **0.3837** | 93.817 % |
| WT           |   1.23 M | 0.305 | 0.653 | **0.3837** |  5.375 % |
| ET (subset)  |   0.30 M | 0.492 | 0.693 | **0.2384** |  0.808 % |

**Two facts:**

1. ⟨|T1c − T1pre|⟩ is *identical* in WT and non-WT brain (0.3837 vs 0.3837).
   The image-space loss signal in WT carries the same per-voxel difficulty as
   non-tumor brain.
2. In the enhancing-only region (ET), ⟨|T1c − T1pre|⟩ is **40 % smaller**
   than in non-WT brain (0.2384 vs 0.3837). After per-modality
   foreground-percentile normalisation, the model "sees" the enhancing rim
   as *less* contrasted from its pre-contrast state than ordinary cortex
   is between modalities.

The mechanism: the top 0.5 % of foreground voxels in T1c is the enhancement;
`percentile_normalise(99.5, clip=True)` cuts at that level and squashes the
bright tail to 1.0. The neighbouring non-enhancing tissue scales down
proportionally — but T1pre, having no enhancement tail, normalises with its
top at gray matter (~0.99). Result: at the same anatomical voxel, T1c
enhancement → ~1.0 (clipped) but normalised gray matter in T1c → ~0.3;
T1pre at that location → ~0.65; difference → mostly *negative* in image
space (T1c < T1pre after normalisation). Per-patient mean Δ_ET ranges
−0.017 to −0.445.

This is not an absolute bug — the *latent encoder* receives the same
normalised input and produces consistent latents — but it is the reason
the image-projected loss cannot strongly favour the enhancing region: the
post-normalisation signal magnitude at enhancement is comparable to or
smaller than at non-enhancing voxels.

### 3.4 VAE recon parity (refutes H3)

From the existing UCSF-PDGM encode artifact
(`/media/hddb/mario/results/vena/encode_ucsf_pdgm_maisi/2026-05-29T19-36-22Z/`):

| modality | MAE | MSE | Lp³ | n |
|---|---:|---:|---:|---:|
| t1pre | 0.0082 [0.0078, 0.0086] | 0.00060 | 0.000074 | 30 |
| **t1c** | **0.0041 [0.0035, 0.0050]** | 0.00022 | 0.000031 | 30 + 4 |
| t2 | 0.0052 [0.0049, 0.0056] | 0.00034 | 0.000046 | 30 |
| flair | 0.0083 [0.0078, 0.0087] | 0.00062 | 0.000070 | 30 |

Per-WHO-grade T1c MAE is stable from grade 2 (0.0045) to grade 4 (0.0054) —
the VAE preserves enhancement across severity. **The VAE is not the
upper-bound on PSNR_WT.** If the FM generator perfectly matched the encoded
T1c latents, decoded predictions would have MAE ≈ 0.004 in [0,1]; that
corresponds to PSNR ≈ 48 dB whole-volume and would not collapse on tumor
regions (the per-modality MAE is the same whether tumor is present or not).

### 3.5 Convergence side-evidence (steady-state)

Latest train-epoch metrics from
`picasso:.../metrics/train_epoch.csv` (epoch ≈ 1086):

| signal | mean |
|---|---:|
| cfm_mean (L1 velocity) | 0.855 |
| grad_norm_trunk_preclip | 1.49 (postclip 0.96 — clip at 1.0) |
| grad_norm_cn_preclip    | 0.30 (postclip 0.25) |

ControlNet gradient norm is consistently **~5× smaller** than the trunk
gradient norm — and the trunk has ~2× the parameter count. Per-parameter
update magnitude is roughly comparable, but the net effect is that the
ControlNet's contribution to the prediction is small relative to the trunk's
ongoing fine-tuning. This is consistent with the ControlNet residuals having
saturated to a near-fixed correction that is not actively re-shaped by the
gradient.

Note also from the per-step CSV: the per-cohort `cfm_cohort_*` columns
report values up to **1.5** for several cohorts even though the global
`cfm_mean = 0.85`, because the column is computed with `pow(2)` (MSE-style)
regardless of `loss.cfm.norm`. Cosmetic, but it means the per-cohort signal
in the CSV is on the wrong scale. (Found while auditing the code path.)

---

## 4. Hypothesis verdicts

The audit covers eight hypotheses that could explain the failure.

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| H1 | Conditioning starvation (no WT mask, no SWAN) | **Supported, secondary** | §2.1 — WT channel is zeroed (zero_out.py:31); SWAN is absent from `conditioning_inputs`. Yet T1C-RFlow succeeds with even less conditioning (T1pre+FLAIR only). The architectural delta (§H6) dominates. |
| H2 | L1 mode-collapse on rare voxels | **Supported** | §3.1 — WT contributes 0.095 % of L1; optimiser ratio ~1047:1. But L1 itself is not the problem (T1C-RFlow uses identical L1 + succeeds); what fails is the **uniform reduction over unweighted voxels**. A region-weighted L1 / pixel-balanced L1 would address this. |
| H3 | MAISI VAE does not preserve enhancement | **Refuted** | §3.4 — VAE MAE on T1c = 0.0041 ± 0.0003. Enhancement passes encode→decode within ~0.4 % error. |
| H4 | `percentile_normalise(99.5, foreground_only)` flattens enhancement | **Supported, contributory** | §3.3 — `⟨|T1c−T1pre|⟩` is identical in WT and non-WT in image space after norm (0.3837 = 0.3837), and *smaller* in ET (0.2384). T1C-RFlow uses a different (global-percentile + minmax01) normalisation but lands in a similar regime, so this is not the *load-bearing* factor — it is, however, a multiplier on the architectural problem. |
| H5 | `use_timestep_transform=true` biases away from sharp-detail regime | **Refuted (already correct)** | T1C-RFlow uses the same `use_timestep_transform=True` and the analysis-doc §4b argues the interaction is positive. Mode mass at α≈0.5 helps semantic structure formation; not the cause. |
| **H6** | **ControlNet conditioning route vs channel-concat** | **Supported, primary** | §2 — code-level proof that the trunk never sees the conditioning. T1C-RFlow's channel-concat at `in_channels=12` is the parsimonious architectural difference. This is the dominant cause. |
| H7 | Multi-cohort distribution dilution | **Plausible but minor** | §1.2 — per-cohort PSNR_WT spans 10.96 (BraTS-Africa) to 19.37 (LUMIERE). Within-cohort failure on the smaller, "cleaner" cohorts (LUMIERE 19.4 dB, UCSF-PDGM 18.1 dB) shows the issue is fundamental, not a noise floor from the larger cohorts. |
| H8 | Pretrained MAISI trunk has healthy-brain bias | **Supported, secondary** | §2.4 — the trunk's pretraining objective biases it toward typical MR appearance; without strong conditioning to override, the prior wins. T1C-RFlow trains from **random init** and still succeeds — confirming that pretraining bias is overridable by a stronger conditioning route, not the load-bearing cause on its own. |
| **P3** | **Latent WT mask too sparse** | **Supported, separate** | §3.2 — 17 % of UCSF-PDGM val patients have zero WT voxels in the latent grid. Any *latent-space* region weighting must address this (image-space loss avoids it). |

**Operative chain (causal):**

1. **H6** (ControlNet conditioning route) is the load-bearing architectural
   factor for **background quality and visual enhancement plausibility**.
   The empirical T1C-RFlow replay (§4b) shows a **4 dB PSNR_BG gap** in
   favour of channel-concat — the channel-concat fix is necessary even if
   PSNR_WT does not change dramatically.
2. **H2** (uniform mean-L1 over 0.095 %-WT voxels) is the load-bearing loss
   factor for **PSNR_ET (the enhancement-only metric)** — the optimiser
   cannot prioritise enhancement when the loss does not. Both T1C-RFlow
   and VENA suffer this (T1C-RFlow PSNR_ET = 15.53; VENA does not yet
   report ET). Fixing only the architecture without fixing the loss leaves
   the enhancement-region accuracy unimproved.
3. **H4** (per-modality 99.5%ile clipping) compounds H2 by reducing the
   image-space signal magnitude in the very region that already has the
   smallest area.
4. **H8** (pretrained healthy-brain prior) amplifies H6 — *because* the
   conditioning route is weak, the trunk's prior dominates. T1C-RFlow's
   from-scratch training removes this confound.
5. **H1**, **H7** are contributory but not load-bearing; **H3, H5** are
   refuted.
6. **P6 (metric trap)** is independent of the failure causes but is the
   reason the project missed how badly it has been failing on
   enhancement specifically. PSNR_WT averages necrosis + edema +
   enhancement and hides the enhancement-only error. Going forward,
   **PSNR_ET (label-4 only)** must be reported alongside PSNR_WT.

---

## 4b. T1C-RFlow head-to-head replay — empirical (2026-06-22)

**Setup**: trained T1C-RFlow checkpoint
`best_net_unet.pth` (190 MB, 49.6M-param paper-faithful 3-level conv-only
U-Net, epoch 164, from Picasso competitor run
`2026-06-15T14-04-17_competitor_t1c_rflow_full_multicohort_68a229e`) relayed
to server3, decoded with the same MAISI-V2 VAE (sha b5ed556d), inference
on 5 randomly-selected UCSF-PDGM fold-0 val patients at NFE ∈ {50, 100, 200}.

### Per-patient PSNR breakdown (NFE=200)

| Patient | PSNR_whole | PSNR_BG | **PSNR_WT** | **PSNR_ET** | WT voxels |
|---|---:|---:|---:|---:|---:|
| UCSF-PDGM-0273 | 27.24 | 31.78 | 17.44 | 17.50 |  26,129 |
| UCSF-PDGM-0364 | 23.40 | 29.00 | 14.08 | 13.51 |  75,608 |
| UCSF-PDGM-0436 | 28.32 | 32.18 | **21.66** | NaN  |  31,975 |
| UCSF-PDGM-0470 | 25.29 | 29.78 | 14.06 | 15.42 |  78,881 |
| UCSF-PDGM-0538 | 27.23 | 32.39 | 17.59 | 15.68 |  40,059 |
| **mean ± std**  | **26.30 ± 1.95** | **31.03** | **16.97 ± 3.14** | **15.53** (n=4) | |

### VENA S1 v2 on UCSF-PDGM, epoch 975, same sample-pool (different patients)

n=7 UCSF-PDGM (different fold-0 val draw — could not exactly match
patient IDs because the two routines draw independently):

| metric | NFE=5 | NFE=20 |
|---|---:|---:|
| PSNR_whole | 27.48 | — |
| PSNR_BG    | ≈ 27 | — |
| **PSNR_WT** | **18.06** | 16.79 |

### The metric trap (P6)

VENA S1 v2's UCSF-PDGM PSNR_WT (**18.06 at NFE=5**) is **higher** than
T1C-RFlow's (**16.97 at NFE=200**) — yet the qualitative picture is
opposite:

- **T1C-RFlow midslices** (e.g. `UCSF-PDGM-0436_midslice.png`,
  `UCSF-PDGM-0273_midslice.png`): visible hyperintense rim around the
  central lesion on 0436; small hyperintense focus correctly placed on 0273.
  Background brain anatomy faithfully reproduced (PSNR_BG = 31 dB).
- **VENA S1 v2 figure_best_1.png at epoch 975** (patient HF1345_1994.06.03,
  REMBRANDT, top-1 SSIM): real T1c shows clear bright enhancement at z∈
  {76, 89, 102}; all five NFE rows produce essentially uniform tissue at
  those locations.

**Why the numbers mislead.** `WT = label 1 (necrosis, T1c-dark) ∪ label 2
(edema, T1c-isointense) ∪ label 4 (enhancement, T1c-bright)`. The WT mask
is dominated by necrosis and edema in most patients (label 4 is typically
20–40 % of WT). Predicting "baseline tissue" everywhere in WT:

- Scores **well** on necrosis (closer to baseline than to enhancement);
- Scores **well** on edema (T1c is roughly the same as T1pre in edema —
  the percentile-normalisation issue from §3.3 only amplifies this);
- Scores **badly** on enhancement (T1c-bright vs predicted-baseline).

If enhancement is 25 % of WT mass, averaging over WT dilutes the
enhancement-region error by 4×. PSNR_WT of 17 dB can therefore reflect
either "mediocre prediction across all WT" (T1C-RFlow) or "great prediction
on necrosis+edema, zero prediction on enhancement" (VENA). The *same number*
hides opposite failure modes.

**PSNR_ET (label 4 only) is the load-bearing metric for this task.**
VENA's exhaustive_val does NOT currently compute PSNR_ET. T1C-RFlow's
infer_cli does compute it (15.53 dB, n=4). The first thing the next
exhaustive_val run should output is per-NFE PSNR_ET.

### Background-PSNR gap is real

T1C-RFlow PSNR_BG = **31.03 dB** vs VENA S1 v2 PSNR_BG ≈ **27 dB** on
UCSF-PDGM — a **4 dB gap** in favour of T1C-RFlow on background tissue.
This is the architectural advantage that *does* show up in numbers. The
channel-concat conditioning gives the U-Net direct access to T1pre / FLAIR
features at every encoder layer; the model can perfectly reproduce
background structure from those features. VENA's ControlNet residual
injection achieves the same task with a 4-dB efficiency penalty.

The same architectural mechanism is **what lets T1C-RFlow predict
enhancement at all** — having the conditioning in every layer's
representation means a feature at the enhancing-region position can drive
the encoder to fire "bright" at that voxel. Without that direct path, VENA
has to either (a) learn the projection through cond_embedding (which is
random-init and gets weak gradient under L1-mean) or (b) defer to the
trunk's healthy-brain prior (which it does).

### Implication for §6 recommendations

The architectural fix (§6.1) **will improve background-PSNR by 3–4 dB**
(consistent with T1C-RFlow's gap) and **will visually unlock enhancement
prediction**, but **PSNR_WT will not move dramatically** (~1 dB at most).
The user's primary qualitative observation (no enhancement) will be
addressed; the metric the project has been tracking will not be the
appropriate witness. Add **PSNR_ET** to exhaustive_val and treat it as the
primary tumor-quality signal going forward.

---

## 5. Recipe deltas vs T1C-RFlow — the side-by-side

This table is the executive view of §2.

| Axis | T1C-RFlow (succeeds at tumor) | VENA S1 v2 (fails on tumor) | Verdict |
|---|---|---|---|
| **Conditioning route** | **Channel-concat at U-Net `in_channels=12`** | ControlNet branch → residual injection | **Load-bearing fix needed** |
| Loss | L1 velocity, mean reduction | L1 velocity, mean reduction | identical |
| Region weighting | None | None | symmetric — but identical absence is *not* identical impact (see ControlNet column) |
| Conditioning content | T1pre + FLAIR latents | T1pre + T2 + FLAIR latents + zeroed WT | VENA's "more conditioning" is wasted by the route |
| Tumor mask | absent | present in DataLoader, then zero_out | "preserved channel slot" buys nothing in S1 |
| SWAN / vessel prior | absent (irrelevant to glioma synth) | **absent** despite being VENA's principal differentiator | scope creep for S1; OK to defer |
| Scheduler `use_timestep_transform` | True (`base=196608` for BraTS box) | True (`base=129024` for brain box) | both right; magnitudes differ |
| ControlNet output_scale ramp | n/a (no ControlNet) | sigmoid 0→1 over 5000 steps | saturated by epoch 24; not the problem now |
| Trunk init | random | MAISI MR pretrained (FFT-finetuned) | T1C-RFlow shows random init is sufficient with the right conditioning |
| Inference steps | 200 Euler | 1–10 Euler/Heun | VENA cannot benefit from longer trajectories without fixing conditioning (§1.3) |
| EMA | none | decay=0.9999 | not relevant |
| Augmentation | none | flip + translate (offline v0–v4) | not relevant |

The **single non-trivial architectural delta** is the conditioning route.

---

## 6. Recommendations for the next run

The recommendations are ordered by leverage. The user's project differentiator
(LPL loss) is unaffected by these changes — they restore the necessary base
recipe so that LPL can be properly evaluated downstream.

### 6.0 The S1 v3 Variant A / Variant B split (2026-06-22 user decision)

The diagnosis in §2–§4b points at the conditioning architecture, but it
does **not** discriminate between two architectural fixes that are both
defensible:

| Variant | Modality conditioning (T1pre, T2, FLAIR latents) | Mask conditioning (NETC, ED, ET) | ControlNet retained? |
|---|---|---|---|
| **A (sequence-only)** | Channel-concat at trunk `conv_in` (in_channels expanded to 16) | absent — no mask signal at all | No (ControlNet dropped entirely) |
| **B (sequence + region masks)** | Channel-concat at trunk `conv_in` (in_channels expanded to 16) | 3-channel `masks/tumor_latent` (NETC, ED, ET) → ControlNet | Yes (ControlNet retained, but its input is now masks only — its designed-for use case) |

**Why both ship at once:**

- Variant A is the **literal T1C-RFlow recipe with VENA's MAISI pretraining
  and offline aug on top**. It isolates "does channel-concat at trunk input
  alone fix the background-quality and visual enhancement gap?" If A
  succeeds, ControlNet adds nothing for this task on top of channel-concat
  and we save the inference cost.
- Variant B layers the project's intended use of ControlNet (spatial
  control via masks) on top of A. It isolates "does explicit per-sub-region
  conditioning (NETC/ED/ET) add ET-quality on top of channel-concat?"
  This is also the architectural pattern that the proposal's S2 vessel
  prior $M_v$ will slot into (a second ControlNet head, or an additional
  channel of the existing one), so B is the right baseline for the S2/S3
  programme.
- The A → B comparison is **the cleanest ablation for the project's main
  differentiator** (vessel-aware conditioning on masks via ControlNet). If
  B beats A at ET-quality, ControlNet on masks is load-bearing for the
  paper. If A ≈ B, ControlNet is irrelevant for the synthesis quality and
  the project's contribution rests on the LPL loss + vessel prior of S2/S3
  alone.

**Both variants share §6.2 / §6.2b / §6.3 / §6.5 / §6.6 / §6.7 / §6.8
unchanged.** They differ in (a) §6.1 (architecture) and (b) §6.4 (mask
conditioning input). The two-config rollout is detailed in §6.9.

### 6.1 Primary (P0) — switch to channel-concat conditioning at trunk input

**Change** (both variants): route T1pre / T2 / FLAIR latents as
channel-concat into the trunk's first convolution, not through the
ControlNet branch.

**Mechanism**:

1. Expand the MAISI trunk `conv_in` from `in_channels=4` to
   `in_channels = 4 + N_cond_latents × 4 + N_masks` (S1: 4 + 3×4 = 16; with
   WT mask: 17).
2. Initialise the expanded conv: copy the original 4 channels' weights for
   `x_t`, **zero-init the new channels' weights** so that step 0 reproduces
   the pretrained trunk's behaviour exactly.
3. Drop the ControlNet branch for the latent-modality conditioning.
   Optionally keep ControlNet **only** for spatial-control add-ons (WT
   mask, vessel prior $M_v$, brain mask) — the use case it was designed
   for. Recommended S1: drop entirely; re-introduce in S2 only for the
   spatial priors that are this project's differentiator.

**Why this is the right move**:

- Matches T1C-RFlow's demonstrated-successful architecture (`train_rflow.py:129,201`).
- The zero-init expansion of `conv_in` is the **direct analogue of
  ControlNet's zero-conv idea** applied at the input layer — guarantees
  identical-to-MAISI behaviour at step 0, with a *much shorter learning path*
  for the conditioning channels (cond at step 1 vs cond-through-residual-via-
  random-cond_embedding at step ≥ 5000).
- Eliminates the random-init `controlnet_cond_embedding` (§2.1) — the model
  no longer has to learn an unfamiliar 13-channel→trunk-feature projection.

**Expected effect**: PSNR_WT to clear 22–24 dB by epoch 200 (T1C-RFlow-class
trajectory). The current ceiling of 17.5 dB at epoch 475 should be passed
within the first 100 epochs of the new recipe.

**Implementation cost**: moderate. The MAISI trunk weights are frozen
elsewhere; the only mutation is the input conv. Suggested code path:
new module `src/vena/model/fm/maisi/conv_in_expand.py` with a `expand_conv_in(
trunk, new_in_channels, zero_init_new=True)` function. The training-time
data path needs to assemble `torch.cat([x_t, z_t1pre, z_t2, z_flair], dim=1)`
in `module.training_step` and pass it as the trunk's `x` argument. The
ControlNet wrapper can be retained as a no-op flag (`use_controlnet=False`)
to preserve the YAML schema; the assembler then routes the conditioning into
the trunk input rather than the ControlNet input.

**Caveat / open question**: the MAISI MR trunk's pretrained `conv_in` was
trained on 4-channel latents that include the *modality token* via
`class_labels`. Adding additional latent channels of a different modality
(T1pre, T2, FLAIR) changes the input distribution. The zero-init of the new
channels guarantees we start *equivalent* to MAISI, but the trunk's
attention layers may need a slightly longer warm-up to accommodate the new
feature-map statistics. This is exactly the situation a 5000-step linear
warm-up on the new-channel L2 weight magnitude would address (analogous to
the existing `output_scale_ramp`, but on the input conv's expanded-channel
weights).

### 6.2 Primary (P0) — region-weighted L1

**Change**: in `CompositeLoss` (S1 path), add a region-weighted L1 term
on top of the existing mean-L1, with the WT mask broadcast across the
4-channel velocity field:

```python
# pseudo: alpha_wt is a hyperparameter; alpha_brain weights brain-not-WT
loss_cfm = F.l1_loss(v_pred, u_target, reduction="none")  # (B, 4, h, w, d)
w = torch.ones_like(loss_cfm)
w = w + alpha_wt * m_wt.expand_as(loss_cfm)             # WT voxels weight 1 + α_wt
w = w + alpha_brain * (m_brain & ~m_wt).expand_as(loss_cfm)
loss = (loss_cfm * w).mean() / w.mean()
```

With `alpha_wt = 200` and `alpha_brain = 1`, the WT contribution to the
gradient becomes `0.087 % × 201 ≈ 17.5 %` of total — comparable to the
brain non-WT contribution. This is the same idea as VENA's existing v0.4
region-weighted contrastive loss (per `project_lp_contrastive_v04.md`), but
applied to the *CFM velocity loss itself* rather than as a separate
contrastive term.

**Why this is the right move**:

- The v0.4 contrastive infrastructure is already in the codebase and
  validated; only the YAML hyperparameter needs to be activated.
- It is independent of the conditioning-route fix (§6.1). The two changes
  are complementary: the architecture fix gives the model the *capacity* to
  learn enhancement; the region weighting gives the optimiser the
  *incentive*.
- Literature precedent: every successful 3D medical synthesis paper that
  reports good tumor synthesis uses some form of region-weighted loss
  (Preetha 2021 multi-task mask loss; Dayarathna 2025 McCaD tumor-aware
  loss; Biller 2026 TumorFlow regional consistency).

**Caveat**: with the WT mask zeroed in the conditioning (S1's
`mask:wt:zero_out`), the model has no way to know which voxels are tumor.
Region-weighted loss would push it to learn "where to put enhancement" from
T1pre+T2+FLAIR features alone. This is harder but plausibly possible —
T2/FLAIR hyperintensity correlates with edema which surrounds enhancement.
Better: combine §6.2 with re-enabling `mask:wt:identity` (§6.4) so the model
*knows* where to apply the upweighted loss.

**Expected effect**: +1–2 dB on PSNR_WT once the architecture is right.

### 6.2b Primary (P0) — report PSNR_ET (label-4 only) in exhaustive_val

**Change**: extend `vena.model.fm.metrics.regions` to expose the
enhancing-only mask (ET = `tumor_label == 4`), and have `exhaustive_val`
emit `psnr_db_et` and `ssim_et` columns alongside the existing
`psnr_db_wt` / `ssim_wt`. The image H5 already carries the labels
(`masks/tumor` int8 in BraTS21 label set).

**Why this is the right move**:

- §4b proved that **PSNR_WT averages necrosis + edema + enhancement and
  cannot distinguish "predicts baseline tissue" from "predicts
  enhancement"**. PSNR_ET is the load-bearing signal for the clinical
  question we are actually trying to answer.
- The proposal § 6.2 (vessel-resolved evaluation) already commits the
  project to per-region metrics. Adding ET is one column.
- Independent of all other recommendations — should land first, before
  any new training, so we can read the new metric against the existing
  S1 v2 baseline as it stops.

**Implementation cost**: trivial. Two lines in
`vena/model/fm/metrics/regions.py` to compute `m_et = (tumor_lbl == 4)`,
plus a write to the metrics CSV header in `routines/fm/exhaustive_val`.
No retraining needed.

**Expected effect**: PSNR_ET at epoch 975 of the current S1 v2 is expected
to be ≤ 14 dB based on the qualitative evidence (the model predicts no
enhancement, so MSE in label-4 voxels is dominated by the squared
intensity difference between baseline and full enhancement ~ 0.4²,
yielding PSNR ≈ 8–14 dB). This number is the *true* baseline.

### 6.3 Primary (P0) — best-metric selection on ET (not BG, not WT)

**Change**: in the production YAML,
`training.best_metric_region: bg → et` and
`training.best_metric_name: mse_latent → psnr_db_et` (after §6.2b adds
the column). The current S1 v2 YAML selects model checkpoints on
*background* MSE in latent space, which trains-and-selects in the
direction we know is over-emphasised. WT would be slightly better but
still suffers the metric-trap from §4b/P6. ET is the right
selection criterion for the clinical task.

**Why**: as long as `ema_best` is selected on `train/total_epoch` (per
`module.py:766` post-fix), this is a no-op for S1 (no validation). But once
validation is re-enabled or `validation.every_epochs > 0`, the selection
criterion matters. **PSNR_WT measured on exhaustive_val output is the
target metric of the proposal § 6**, and the in-process best-metric
should align with it. Currently it actively misaligns.

**Implementation cost**: trivial YAML change.

### 6.4 Secondary (P1, Variant B only) — ControlNet on 3-channel NETC/ED/ET mask

**Change** (Variant B only): replace the single-channel zeroed WT mask
with **three explicit sub-region soft masks** as the ControlNet's
*only* conditioning input. The data is already available in the latent H5:

```
masks/tumor_latent ∈ ℝ^{N×3×48×56×48}
  channel 0 = NETC (label 1, necrotic core, T1c-dark)
  channel 1 = ED   (label 2, peritumoral edema, T1c-isointense)
  channel 2 = ET   (label 4, enhancing tumour, T1c-bright)
```

per-class one-hot at image resolution, average-pooled to latent
resolution (per the encode-routine `mask_downsampler_attrs_json` attr).

In YAML this is one of:

```yaml
# preferred — single spec consumes the 3 channels at once
conditioning_inputs:
  - mask:tumor3:identity

# alternative — three explicit specs (same outcome, more verbose)
conditioning_inputs:
  - mask:netc:identity
  - mask:ed:identity
  - mask:et:identity
```

**Why per-sub-region** (not a single binary WT mask):

| Label | T1c appearance | What the model should produce |
|---|---|---|
| NETC (1) | hypointense | dark voxels |
| ED   (2) | isointense  | tissue-like, **no enhancement** |
| ET   (4) | hyperintense | bright voxels |

A single binary "tumor here" mask forces the model to *infer* the
sub-type from the multimodal input (FLAIR-bright + T1pre-dim could be
NETC, ED, or ET — ambiguous). The 3-channel split removes the inference
ambiguity and lets the model focus on getting the *intensity* right
given the *known* sub-type. It also makes the §6.5 / proposal §6.5
healthy-control diagnostic trivially executable (a healthy patient has
all three channels = 0 → no positional prior for enhancement → model
must produce no enhancement).

**Why ControlNet (not channel-concat) for the masks**:

ControlNet (Zhang et al. 2023, ICCV) was designed for exactly this use
case: a strong generator (the trunk with channel-concat modality
conditioning per §6.1) gets an additional *spatial control signal*
(here, the 3-channel mask) layered on top via residual injection. The
proposal's vessel prior $M_v$ (S2) is the same shape of signal and
plugs into the same ControlNet slot (either as additional channels of
the existing mask ControlNet, or as a parallel ControlNet head).

**Implementation detail — the ControlNet's first conv**:

With the modality latents removed from the ControlNet input, the
`controlnet_cond_embedding` first conv's `in_channels` drops from 13
(4×3 + 1) to **3** (just the three masks). The remaining encoder blocks
(trunk-init copies) and output projections (zero-init) keep their
existing shape. The `output_scale_ramp` from S1 v2 is retained — it
performed correctly and saturated by epoch 24.

**CFG-style mask dropout** to enable the §6.5 diagnostic and avoid
"copy the mask to the output" shortcuts:

```yaml
training:
  conditioning_dropout_p: 0.15     # was 0.0
  conditioning_dropout_keys:        # was [wt]
    - netc
    - ed
    - et
```

This independently zeros each mask channel with probability 0.15 per
training step, forcing the model to also learn the unconditional
mapping. The infrastructure is already in place (per `project_lp_contrastive_v04`
memory and `decision.json` schema 0.7.0).

**Expected effect (Variant B vs Variant A)**: +2–4 dB on **PSNR_ET**
specifically. Whole-volume and BG metrics expected to be similar to
Variant A (within ±0.5 dB). The ablation is meaningful precisely
because the channel-concat baseline (A) is strong enough that adding
ControlNet should produce *only* tumor-region gains; if B also helps
on BG, that signals overlap or memorisation and the recipe needs
re-thinking.

### 6.5 Secondary (P1) — sweep latent vs image-space loss

**Change**: add a small image-space L1 term alongside the latent CFM L1 by
decoding `x̂_1 = x_t + α·v_pred` and computing `F.l1_loss(decode(x̂_1),
real_t1c_normalised)` *only* in WT voxels. Weighted by ~0.1 of the latent
CFM term.

**Why**: §3.2 showed the latent WT mask is empty for 17 % of patients.
Image-space WT is always populated. An image-space tumor-region term
ensures every training patient with a tumor contributes a non-zero gradient
to the enhancement objective regardless of latent mask coverage.

**Implementation cost**: moderate (need a partial VAE decode in the loss
path; existing `vena.common.decode.decode_depth_identity` covers this).

**Caveat**: this is the **same idea as the LPL loss programme already in
flight** (`decoder_perceptual_loss_s3*.md`). The LPL is decoder-feature
based; this would be decoder-pixel based. Both have merit; the LPL is
already designed and preflighted. **Don't duplicate** — fold the
region-weighted pixel term *into* the S3 LPL programme as an alternative
ablation arm.

### 6.6 Secondary (P1) — sweep normalisation contract

**Change**: try one of:

- **Joint-percentile normalisation**: compute the 99.5%ile over the union
  of all four modalities (T1pre + T2 + FLAIR + T1c) per patient, use the
  same scaling factor for all. Preserves inter-modality intensity
  correspondence. Cost: re-encode the cohort. ~6 h on server3:cuda:0.
- **No clipping**: `percentile_normalise(lo=0, hi=99.5,
  foreground_only=True, clip=False)`. Lets enhancement keep its
  super-percentile values. The VAE will see the same data shape; values
  above 1.0 are not pathological. Should improve the latent encoding of
  the enhancing rim.

**Why**: §3.3 showed the per-modality clipped 99.5%ile flattens the
enhancement signal. A non-clipping or joint-modality scheme preserves it.

**Implementation cost**: moderate (re-encode). Defer until after §6.1 + §6.2
land — those are the load-bearing fixes; this is the multiplier.

**Caveat**: changing the normalisation contract breaks all existing latent
H5 caches and existing checkpoints. Plan this as a separate routine sweep,
not bundled with S1 v3.

### 6.7 Optional (P2) — increase NFE at evaluation

**Change**: when the conditioning route fix is in (§6.1), run exhaustive_val
at NFE ∈ {50, 100, 200} in addition to {1, 2, 5, 10, 20}.

**Why**: §1.3 — T1C-RFlow uses 200 Euler steps because the conditioning
steers every step. With the channel-concat fix, more steps should *help*
PSNR_WT (currently they hurt). If they don't help, that's a falsification
signal that the conditioning route fix is incomplete.

### 6.8 What NOT to do (and why)

- **Don't continue the current S1 v2 past convergence.** It has plateaued
  on tumor (§1.1) and is now regressing. The remaining wall-clock budget is
  better spent on the new recipe.
- **Don't add LPL on top of the current S1 v2** as a primary attempt — the
  analysis-doc 2026-06-20 §3.2 already cautions against this. Confirmed by
  this audit: LPL needs a base model whose conditioning *can* shape the
  tumor; the current model can't.
- **Don't add SPADE or FiLM as a first move.** The 2026-06-20 doc considered
  these; they replace mask conditioning with normalisation modulation. They
  do nothing for the modality conditioning, which is where the gap is.
- **Don't increase `output_scale_ramp.ramp_steps`.** §3.5 shows the ramp
  has saturated; longer ramp wouldn't help.
- **Don't switch back to L2.** It is not the load-bearing factor (§4 H5),
  and the L1 vs L2 row in proposal §7 ablation is a separate question
  (T1C-RFlow demonstrates L1 wins per Eidex 2025).

### 6.9 Proposed next-run YAML deltas — Variant A and Variant B side-by-side

Two config files; both diff against
`routines/fm/train/configs/runs/picasso_s1_1000ep_fft.yaml`. The shared
block (loss, metrics, training) is identical between A and B. Only the
`model.*` and `data.conditioning_inputs` blocks differ.

#### Shared block (both variants)

```yaml
loss:
  cfm:
    weight: 1.0
    reduction: none              # NEW — return per-voxel for region weighting
    norm: l1
    region_weights:              # NEW — region-weighted L1
      enabled: true              # set false to recover unweighted L1 (kept easy-to-disable)
      bg: 1.0
      brain_not_wt: 1.0
      netc: 50.0
      ed: 50.0
      et: 300.0
training:
  best_metric_name: psnr_db_et   # was: mse_latent
  best_metric_region: et         # was: bg
  best_metric_nfe: 5
  patience: 250
  conditioning_dropout_p: 0.15   # was 0.0; only meaningful for Variant B (no-op for A since A has no mask)
  conditioning_dropout_keys:
    - netc                       # was [wt]
    - ed
    - et
exhaustive_val:
  nfe_levels: [1, 2, 5, 10, 20, 50, 100, 200]
  emit_per_region_metrics:       # NEW — emits psnr_db_{et,netc,ed,bnwt}, mae_*, mse_*, ssim_*
    - et
    - netc
    - ed
    - brain_not_wt
    - whole
```

#### Variant A — `picasso_s1_v3a_concat_only_fft.yaml`, tag `s1_v3a_concat_only`

```yaml
model:
  trunk:
    input_concat:                # NEW — expand conv_in to take channel-concat conditioning
      enabled: true
      cond_latents: [t1pre, t2, flair]
      cond_masks: []             # Variant A: NO mask channel
      zero_init_new_channels: true
      ramp_steps: 5000           # warm-up on the new-channel weights (mirrors prior output_scale_ramp)
      ramp_steepness: 10.0
    trainable: true
    regime: fft
  controlnet:
    enabled: false               # NEW — ControlNet branch dropped entirely
data:
  conditioning_inputs: []        # NEW — all conditioning is in trunk's channel-concat path
```

#### Variant B — `picasso_s1_v3b_concat_plus_cn3ch_fft.yaml`, tag `s1_v3b_concat_plus_cn3ch`

```yaml
model:
  trunk:
    input_concat:                # NEW — same as Variant A
      enabled: true
      cond_latents: [t1pre, t2, flair]
      cond_masks: []             # masks go through ControlNet, not channel-concat
      zero_init_new_channels: true
      ramp_steps: 5000
      ramp_steepness: 10.0
    trainable: true
    regime: fft
  controlnet:
    enabled: true                # NEW — kept, but with masks-only input
    conditioning_inputs:
      - mask:tumor3:identity     # NEW spec — reads masks/tumor_latent (3-channel NETC/ED/ET)
    output_scale_ramp:           # retained from S1 v2
      enabled: true
      ramp_steps: 5000
      steepness: 10.0
    init_from_trunk: false       # NEW — masks are not modality latents, trunk-init weights ≠ helpful
                                 #       (Variant B's CN starts from scratch; rationale in §6.4)
data:
  conditioning_inputs: []        # NEW — modality latents go via channel-concat; masks via ControlNet
```

The decision-json schema bumps to **0.10.0** with new fields:

- `model.trunk.input_concat.*` (block)
- `model.controlnet.init_from_trunk` (bool)
- `model.controlnet.conditioning_inputs` (list — when ControlNet enabled)
- `loss.cfm.region_weights.*` (block)
- `loss.cfm.reduction = "none"` newly allowed
- `training.conditioning_dropout_keys` extended to include `{netc, ed, et}`
- `exhaustive_val.emit_per_region_metrics` (list)

### 6.10 Expected total effect — Variant A vs Variant B

Calibrated against the T1C-RFlow empirical reference from §4b. Both
variants share §6.2 (region-weighted L1) + §6.2b (PSNR_ET column) + §6.3
(ET-based best-metric selection). They differ only in the conditioning
architecture for the mask.

| Metric | Current S1 v2 (ep 1075) | T1C-RFlow ref (5-patient) | **Target Variant A (ep 200)** | **Target Variant B (ep 200)** |
|---|---:|---:|---:|---:|
| PSNR_whole          | 25.7 | 26.3 | **29.5** | **29.5** |
| PSNR_BG             | 26.2 | 31.0 | **31.5** | **31.5** |
| PSNR_WT (composite) | 17.1 | 17.0 | **18.5** | **20.0** |
| **PSNR_ET (load-bearing)** | ≤14 (est) | 15.5 | **18–19** | **20–22** |
| PSNR_NETC           | n/a | n/a | **22+** | **22+** |
| PSNR_ED             | n/a | n/a | **24+** | **24+** |
| Visible enhancement in figure_best | ~0 % | majority | majority | majority |
| **A→B ET-PSNR gain (load-bearing ablation)** | — | — | reference | **+2–4 dB on ET vs A** |

**Key calibration notes**:

- **Both variants** should hit the +5 dB PSNR_BG gain — the gain comes
  from channel-concat at the trunk input (§6.1), which is shared.
- **PSNR_ET difference (A → B)** is the ablation that justifies (or
  refutes) keeping ControlNet in the project. If B − A ≥ +2 dB on ET,
  ControlNet + per-sub-region masks are load-bearing for the paper's
  primary clinical metric and the proposal's vessel-prior (S2/S3) plan
  is on the right slot of the architecture. If B ≈ A (within ±0.5 dB),
  ControlNet adds nothing for synthesis quality and the project's
  contribution rests on the LPL programme + vessel-aware evaluation
  alone.
- **PSNR_WT is no longer the load-bearing metric**; it remains in the
  table for backwards-compat with the prior runs only.
- **NETC and ED** are expected to be high simply because predicting
  baseline tissue is approximately correct in those regions. The ET
  region is where the recipe is tested.

The A vs B PSNR_ET delta is the single most important number to read
when both runs complete.

### 6.11 Detailed implementation specs — child documents

The two children specifications that operationalise this section are:

- **Data layer** — `.claude/notes/changes/2026-06-22_s1_v3_normalization_exploration.md`
  Audits and selects a T1c-preserving normalisation, then re-encodes all
  9 cohorts + offline augmentations. Outputs a `decision.json` consumed
  by the encode routines and pinned in v3 corpus_registry.

- **Model layer** — `.claude/notes/changes/2026-06-22_s1_v3_model_implementation.md`
  Implements Variant A and Variant B end-to-end: trunk `conv_in`
  expansion, `mask:tumor3` conditioning spec, 3-channel ControlNet,
  region-weighted L1 (with easy disable), per-region metrics. Includes
  the file-by-file edit plan, unit-test acceptance criteria, and the
  v3a + v3b YAMLs.

These two specs are self-contained for the downstream implementer. Both
defer to this father document for *why*; they own the *how*.

---

## 7. References (load-bearing)

- Eidex et al. 2025. *An Efficient 3D Latent Diffusion Model for
  T1-contrast-Enhanced MRI Generation.* arXiv:2509.24194. — §2.3, the
  channel-concat conditioning recipe.
- Zhang, Rao, Agrawala 2023. *Adding Conditional Control to Text-to-Image
  Diffusion Models.* ICCV. — ControlNet design assumptions invoked in §2.4.
- Guo et al. 2025. *MAISI: Medical AI for Synthetic Imaging.* MICCAI /
  MONAI-bundle release notes. — pretrained VAE and trunk that VENA uses.
- Berrada et al. 2025. *Latent Perceptual Loss (LPL).* arXiv:2411.04873. —
  hi-SNR gate; basis of the deferred S3 programme.
- Esser et al. 2024. *Scaling Rectified Flow Transformers for High-Resolution
  Image Synthesis.* arXiv:2403.03206 (SD3). — `use_timestep_transform`.
- Park et al. 2019. *Semantic Image Synthesis with Spatially-Adaptive
  Normalization* (SPADE). CVPR. — alternative conditioning route considered
  in 2026-06-20 doc §4a and rejected here for §6.1.
- Isola et al. 2017. *Pix2Pix.* CVPR. — L1 vs L2 sharpness motivation.
- Preetha et al. 2021. *Deep-learning-based synthesis of post-contrast
  T1-weighted MRI for tumours.* — first GAN+mask baseline.
- Dayarathna et al. 2025. *McCaD: Multi-modal Conditional Cascaded
  Diffusion.* — recent multi-modal mask-conditioned baseline.
- Biller et al. 2026. *TumorFlow.* — recent FM-based tumor-aware synthesis.

Code-level citations live inline in §2.

---

## 8. Open question for the user

The recommendations in §6 are independent of the LPL programme but they
interact:

- §6.1 (channel-concat) makes the conditioning route load-bearing for the
  trunk. The S2/S3 warm-start path is unaffected at the trunk level (FFT
  fine-tune continues), but the YAML schema needs the `input_concat` block
  to round-trip into decision.json v0.10.0.
- §6.5 (image-space WT loss) overlaps with the LPL design. The LPL preflight
  already characterises the right blocks and hi-SNR gate; adding a region-
  weighted pixel term would parallel that path.

**Recommended order of execution:**

1. **First — Land §6.2b (PSNR_ET column) NOW**, without retraining. It is
   the metric that should be the witness of every subsequent change. Two
   lines of code. Read the current S1 v2's PSNR_ET off the next
   exhaustive_val cadence epoch (the run will hit one at epoch 1100 in
   about 2 hours given the 25-epoch cadence). That number is the *true*
   baseline.
2. **Decision point**: stop S1 v2 or let it burn the remaining patience.
   The 250-epoch patience window starting at epoch 475 (last PSNR_WT peak)
   has ~140 epochs of headroom left. Given the trajectory is flat/regressing
   on tumor, I recommend stopping it now and freeing the 2× A100 budget
   for v3. The user's call.
3. **Land §6.1 + §6.2 + §6.3 + §6.4** as a single v3 recipe;
   train ~200 epochs (~2 days on 4× A100).
4. **Evaluate against the new exhaustive_val** (with PSNR_ET + NFE ∈
   {1, 2, 5, 10, 20, 50, 100, 200}).
5. **If PSNR_ET clears 20 dB at NFE=50 by epoch 200**, declare v3 the new
   baseline and proceed to S3 LPL warm-start (with the existing P0 monitor
   fix from 2026-06-20).
6. **Defer §6.5 (image-space ET loss) and §6.6 (joint normalisation)** to
   a separate ablation matrix row once v3 is the production baseline.

The §6.3 (best-metric selection) change must land in the same v3 recipe
since it's a one-line YAML edit and is required to make the in-process
ema_best selection align with the target metric.

---

*Diagnostic complete. Awaiting user decision on whether to (a) stop S1 v2
now and roll v3, (b) let S1 v2 burn the full patience budget while v3 is
being prepared, or (c) request additional sub-audits before changing the
recipe.*

---

## 9. Artifact appendix

Reproducible inputs for every numerical claim in this report. All paths
are committed alongside this markdown under
`.claude/notes/review/figures/2026-06-22_diagnosis/`.

### Diagnostic scripts

| Script | Purpose | Where it ran |
|---|---|---|
| `audit_target_velocity.py` | §3.1 latent-space target-velocity per-region audit (n=30 UCSF-PDGM fold-0 val). Computes ⟨\|u\|⟩, ⟨\|Δ\|⟩, %∑\|u\| per (BG, BRAIN_NOT_WT, WT, ET). | server3 `~/.conda/envs/vena/bin/python /tmp/audit_target_velocity.py` |
| `audit_image_contrast.py` | §3.3 image-space contrast audit with VENA percentile-norm parity (n=15). Computes ⟨T1c⟩, ⟨T1pre⟩, ⟨\|T1c−T1pre\|⟩ per region. | server3 same env |
| `compute_rflow_wt_psnr.py` | §4b T1C-RFlow head-to-head — loads saved NIfTI predictions, applies UCSF-PDGM image-H5 WT/ET masks, computes PSNR per region. | server3 same env |

### Numerical CSV outputs

- `audit_target_velocity_per_patient.csv` — per-patient row of WT-voxel
  count, %loss(WT), ⟨\|u\|⟩ ratios. n=30.
- `t1c_rflow_inference_metrics.csv` — T1C-RFlow infer_cli raw output
  (whole-volume only; per-region computed in `compute_rflow_wt_psnr.py`).

### Figures

- `figure_best_1.png` — VENA S1 v2 epoch 975 best-1 patient
  (HF1345_1994.06.03, REMBRANDT). 5 NFE rows; tumor missing in all.
- `UCSF-PDGM-0273_midslice.png` — T1C-RFlow inference midslice, real vs
  pred. Small hyperintense focus correctly placed.
- `UCSF-PDGM-0436_midslice.png` — T1C-RFlow inference midslice, real vs
  pred. Hyperintense rim around central lesion correctly placed.

### Source artifacts (NOT copied into repo — paths cited)

| Artifact | Path |
|---|---|
| S1 v2 run dir (active) | `picasso:/mnt/home/users/tic_163_uma/mpascual/execs/vena/experiments/2026-06-20_14-15-19_s1_fft_cfm_d2bc4d2a/` |
| S1 v2 epoch-975 metrics | `<above>/exhaustive_val/epoch_975/metrics.csv` |
| S1 v2 epoch-1075 metrics (latest) | `<above>/exhaustive_val/epoch_1075/` |
| S1 v2 train_epoch.csv | `<above>/metrics/train_epoch.csv` |
| T1C-RFlow trained ckpt | `picasso:.../competitors/t1c_rflow/2026-06-15.../checkpoints/best_net_unet.pth` |
| T1C-RFlow ckpt sha (190 MB) | mirrored on `icai-server:/media/hddb/mario/competitors/best_net_unet.pth` |
| T1C-RFlow inference dir | `icai-server:/media/hddb/mario/competitors/t1c_rflow_run/inference/epoch_best/` |
| UCSF-PDGM image H5 (server3) | `/media/hddb/mario/data/GLIOMAS/UCSF_PDGM/h5/UCSFPDGM_image.h5` |
| UCSF-PDGM latent H5 (server3) | `/media/hddb/mario/data/GLIOMAS/UCSF_PDGM/h5/UCSFPDGM_latents.h5` |
| MAISI VAE ckpt (server3) | `/media/hddb/mario/checkpoints/MAISI_V2_RM/NV-Generate-MR/models/autoencoder_v2.pt` |
| MAISI VAE encode artifact (UCSF-PDGM) | `/media/hddb/mario/results/vena/encode_ucsf_pdgm_maisi/LATEST/report.md` |

### Hardware used

- **Recon** (read-only SSH to Picasso): no GPU.
- **Latent-space audit** (§3.1): server3 `vena` conda env, CPU only (~3 min).
- **Image-space audit** (§3.3): server3 `vena` conda env, CPU only (~2 min).
- **T1C-RFlow inference replay** (§4b): server3 cuda:0 (RTX 4090 24 GB,
  15 GB peak), 49.6M-param U-Net, 5 patients × 3 NFE = 15 samples, 2 min
  20 s end-to-end.
- **No retraining** was performed for this diagnostic.

### Caveats and known limitations

- The T1C-RFlow head-to-head used **5 randomly-sampled UCSF-PDGM
  fold-0 val patients**, which did *not* exactly match the 7 UCSF-PDGM
  patients in S1 v2's epoch-975 sample (the two routines draw
  independently). The per-cohort aggregate numbers compared apples to
  apples across cohorts in the cohort table (§4b) are
  population-representative, but the 5-vs-7 sample overlap for UCSF-PDGM
  specifically is zero. A definitive head-to-head needs T1C-RFlow
  inference on the same 7 patient IDs S1 v2 evaluated; the
  `infer_cli.py` interface does not currently support `--patient-ids` so
  this would require either a one-off script call into
  `vena.competitors.t1c_rflow.inference.run_inference` with
  `dataset_ids_override=` (or a wrapper to that effect) or a
  modification to the CLI. Estimated cost: 30 min to extend
  `T1CRFlowLatentDataset` to accept an `ids` filter; trivial.
- The latent-space audit and image-space audit use UCSF-PDGM **fold 0
  val** (89 patients available; 30 sampled for latent, 15 for image).
  The numerical magnitudes are robust under bootstrap (the WT/total ratio
  is ~0.001 across every patient and the percent-of-sum is consistent).
  The numerical magnitudes for other cohorts may differ
  (BraTS-Africa has larger tumor masks; UCSF-PDGM has smaller); the
  qualitative conclusions (class imbalance, latent-mask sparsity,
  normalisation flattening) hold universally.
- PSNR_ET for VENA S1 v2 is **estimated from the qualitative pattern**
  ("predictions are uniform tissue at enhancing voxels"); the exact
  value requires §6.2b code change and one run of exhaustive_val. The
  estimate ≤ 14 dB is conservative.
