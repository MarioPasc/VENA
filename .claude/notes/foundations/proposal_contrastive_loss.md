# Project Proposal: Contrastive-Regularised Latent Flow Matching for Gadolinium-Free Synthesis of T1 Post-Contrast Brain MRI

*Draft v0.3 — Mario Pascual González, IBIMA-BIONAND / Universidad de Málaga*

*Change-log v0.2 → v0.3 (2026-06-01).* Merged stages S2 and S3 into a single
**$L^p$-aware contrastive** stage with region-dependent exponent
($p_t = 1$ inside the tumour mask, $p_b = 3$ outside). The capped $L^p$
velocity-reconstruction term that previously defined S3 is retained as the
*factorised-loss ablation*, not as a separate curriculum step. Rationale,
gradient-stability evidence, and updated ablation grid are recorded in §5
and §8. The stability probe that justifies $p_b = 3$ over $p_b = 4$ is
documented in §5.4.

---

## 1. Problem and hypothesis

### 1.1 Clinical motivation

Gadolinium-based contrast agents (GBCAs) underlie a substantial fraction of clinical brain MRI exams but are associated with long-term tissue deposition, environmental contamination, and direct patient risk (Mallio et al. 2023, DOI: [10.3389/fnimg.2023.1055463](https://doi.org/10.3389/fnimg.2023.1055463)) [fact]. Synthesising the post-contrast T1 image (T1c) from non-contrast sequences is therefore an active research direction. Three solution families are now established (CNN/GAN, conditional diffusion, flow matching), and the Mallio et al. systematic review identifies *non-tumour enhancement and small vessels* as the dominant unresolved failure mode across all of them. We address this failure mode at the **loss level**, by reformulating the training objective so that non-tumour structures receive proportionally larger gradient signal than they do under uniform pixel/latent regression.

### 1.2 Methodological framing

The headline contribution is a curriculum-trained, contrastive-regularised latent rectified-flow model that warm-starts from the publicly released MAISI-v2 MR foundation model (NVIDIA NV-Generate-MR, October 2025) [fact]. We port the region-specific contrastive loss of Zhao et al. *MAISI-v2* (AAAI 2026, arXiv:2508.05772) [fact] into a tumour-downweighted variant: where the MAISI-v2 authors *amplified* tumour signal (their target is to make a missing tumour appear in unconditional CT synthesis), we *attenuate* it (our target is to prevent the tumour from dominating the loss in a paired translation task where the tumour mask is already an explicit input). The downweighting is the single design delta from the source paper; everything else is reused.

### 1.3 Hypotheses

**H1.** Warm-starting the FM trunk from NV-Generate-MR weights and training only a ControlNet branch for multi-sequence conditioning is sufficient to learn the paired translation $\{T_{1\text{pre}}, T_2, \text{FLAIR}\} \to T_{1c}$ on UCSF-PDGM-scale data ($N \approx 400$). [hypothesis; tested by the trunk-init ablation in §8]

**H2.** Replacing uniform CFM with the MAISI-v2 region-specific contrastive loss, with tumour weight $\lambda_{\text{tum}} < 1$, improves background-region reconstruction (PSNR/SSIM restricted to brain $\setminus M_{\text{WT}}$) without degrading tumour-region metrics. [hypothesis; tested by the loss-formulation ablation in §8]

**H3.** Making the contrastive loss $L^p$-aware with region-dependent exponent — $p_t = 1$ inside the tumour mask, $p_b = 3$ outside, capped at $\delta = 2$ ($\sim 2\sigma_{\text{latent}}$) — improves small-vessel fidelity (LPIPS-3D, clDice on a Frangi-derived vessel mask) over the uniform-$L^1$ MAISI-v2 contrastive. The mechanism: voxels where the model fails to be mask-invariant produce large $\Delta_\theta$, which the $L^3$ background gradient up-weights $p_b\,|\Delta_\theta|^{p_b-1}$ times relative to $L^1$, focusing the gradient on the small fraction of background voxels — empirically vessels, pituitary, and choroid plexus — that the model has not yet matched. [hypothesis; tested by the $p_b$-sweep ablation in §8; gradient-stability evidence in §5.4]

**H4.** Curriculum training (S1 $\to$ S2-merged, where S2-merged carries the $L^p$-aware contrastive) outperforms direct-S2-from-scratch in either final quality or GPU-hour cost. The direct-S2 experiment is the *control* that tells us whether the curriculum is necessary. [hypothesis; tested by the curriculum ablation in §8]

---

## 2. Data

### 2.1 Training and internal validation — UCSF-PDGM

Calabrese et al. *Radiology AI* 2022 (DOI: [10.1148/ryai.220058](https://doi.org/10.1148/ryai.220058); TCIA DOI: [10.7937/tcia.bdgf-8v37](https://doi.org/10.7937/tcia.bdgf-8v37)) [fact].

- $N = 501$ preoperative diffuse glioma patients.
- 3 T GE Discovery 750.
- Per study: T1 pre-contrast, T1 post-contrast, T2, T2-FLAIR, DWI, SWI, ASL, HARDI. The default architecture consumes only $\{T_{1\text{pre}}, T_2, \text{FLAIR}, M_{\text{WT}}\}$; SWI and the perfusion sequences are reserved for ablations (§8).
- Standardised Gd protocol: gadobutrol 0.1 mmol/kg or gadoterate 0.2 mmol/kg.
- Patient-level split: 400 train / 50 validation / 50 held-out test.

### 2.2 Tumour mask convention — BraTS labels

We adopt the BraTS-2024 labelling convention (Baid et al. 2021, arXiv:2107.02314; LaBella et al. *Lancet Oncology* 2024 for the meningioma extension) [fact]. For glioma, the labels are $\{1, 2, 3\} = \{\text{NCR}, \text{ED}, \text{ET}\}$; for meningioma, $\{1, 2, 3\} = \{\text{ET}, \text{NET}, \text{ED}\}$ where ED is occasionally absent. The whole-tumour (WT) mask used as conditioning is the union of all non-background labels:

$$
M_{\text{WT}}(x) \;=\; \mathbb{1}\big(\,\text{BraTS}(x) > 0\,\big).
$$

UCSF-PDGM ships with per-subject BraTS-style segmentations; for cohorts without ground-truth segmentations (BraTS auxiliary corpora, Málaga external set), we run nnU-Net trained on BraTS-2024 GLI + MEN (Isensee et al. *Nature Methods* 2021, DOI: [10.1038/s41592-020-01008-z](https://doi.org/10.1038/s41592-020-01008-z)) [fact] and binarise the output to obtain $M_{\text{WT}}$.

### 2.3 External validation — Hospital Universitario Regional de Málaga

Three-tier acquisition specification, ordered by priority, negotiated with the radiology department under institutional data-sharing agreement.

**Tier 1 — inclusion criterion.** Preoperative glioma and meningioma. Per study, same imaging session: T1 pre-contrast, T1 post-contrast, T2, T2-FLAIR. Missing any of the four excludes the study.

**Tier 2 — accepted and welcome.**
- Glioma stratified by contrast behaviour (enhancing vs non-enhancing). Non-enhancing gliomas are the counterfactual test — the model must *not* hallucinate enhancement.
- Multi-vendor cases (Siemens, Philips alongside GE) to test vendor shift on top of pathology shift.

**Tier 3 — exploratory.** Longitudinal follow-up exams; SWI/SWAN where available (for vessel-mask ablation).

### 2.4 Auxiliary training corpora — BraTS

For trunk fine-tuning experiments and ablations that benefit from larger $N$, we use BraTS-2023 GLI, BraTS-2024 GLI + MEN + MET, and BraTS-2025 (open at the time of writing). These cohorts ship with T1, T1c, T2, FLAIR — sufficient for the default architecture — but lack SWI/SWAN, restricting their use for vessel-aware ablations.

---

## 3. Preprocessing pipeline

### 3.1 Image-space pipeline

Standalone CLI script `preprocess.py` performing the following per subject. All steps use established libraries; no custom implementations.

1. **Skull stripping** with HD-BET (Isensee et al. *Human Brain Mapping* 2019, DOI: [10.1002/hbm.24750](https://doi.org/10.1002/hbm.24750)) [fact] on T1pre.
2. **Rigid registration** of T1c, T2, FLAIR to T1pre using ANTs (`antsRegistrationSyN.sh` with rigid + affine stages). Library: `antspyx` Python bindings.
3. **N4 bias-field correction** (Tustison et al. *IEEE TMI* 2010, DOI: [10.1109/TMI.2010.2046908](https://doi.org/10.1109/TMI.2010.2046908)) [fact] on every anatomical sequence post-registration.
4. **Resample** to $1\,\text{mm}^3$ isotropic, $240 \times 240 \times 160$ voxels, using `monai.transforms.Spacing` and `SpatialPad`.
5. **Intensity normalisation**: percentile clipping $[0.5, 99.5]$ within the HD-BET brain mask, then z-score within the same mask.
6. **Tumour-mask generation**: if BraTS-style segmentation is available, binarise as in §2.2; otherwise run nnU-Net inference and binarise.

### 3.2 Latent encoding

Standalone script `encode_latents.py` runs once after preprocessing and caches latents in HDF5. The MAISI VAE is loaded from `autoencoder_v2.pt` and frozen.

- VAE class: `monai.apps.generation.maisi.networks.autoencoderkl_maisi.AutoencoderKlMaisi`
- Compression: $4\times$ per spatial dimension. For our $240 \times 240 \times 160$ grid, the latent shape is $\mathbb{R}^{4 \times 60 \times 60 \times 40}$ in fp16.
- Storage cost: $\sim 1.1\,\text{MB}$ per latent $\times$ 4 sequences (T1pre, T1c, T2, FLAIR) $\times$ 501 subjects $\approx 2.2\,\text{GB}$ total. Negligible.

The training Dataset never calls the VAE; it streams pre-encoded latents from disk. This is the standard optimisation for frozen-VAE LDM training and is consistent with MAISI-v2's own pipeline [fact].

### 3.3 Mask preparation

The whole-tumour mask $M_{\text{WT}}$ is downsampled to latent resolution by nearest-neighbour interpolation: $M_{\text{WT}}^\downarrow \in \{0, 1\}^{1 \times 60 \times 60 \times 40}$. The dilated background complement, used for the contrastive background term, is computed in latent space:

$$
m \;\equiv\; M_{\text{WT}}^\downarrow, \qquad m^- \;\equiv\; 1 - \mathrm{dilate}(m),
$$

with dilation by one latent voxel (`scipy.ndimage.binary_dilation` with a $3 \times 3 \times 3$ structuring element). The dilation provides a safety margin at the tumour boundary where mask-perturbation effects can leak into nominally-background voxels.

---

## 4. Architecture

### 4.1 MAISI VAE (frozen)

Reused as-is from MAISI-v2 (Zhao et al. 2026; Guo et al. *WACV* 2025 for v1) [fact]. Configuration per `config_maisi3d-rflow.json`:

```python
AutoencoderKlMaisi(
    spatial_dims=3, in_channels=1, out_channels=1,
    latent_channels=4,
    num_channels=[64, 128, 256], num_res_blocks=[2, 2, 2],
    norm_num_groups=32, norm_eps=1e-6,
    attention_levels=[False, False, False],
    with_encoder_nonlocal_attn=False, with_decoder_nonlocal_attn=False,
    norm_float16=True, num_splits=4, dim_split=1,
)
```

The VAE has previously been validated on glioma and meningioma pathology in this project's preparatory work (preserved enhancing-rim contrast, preserved necrotic core, preserved peri-tumoural FLAIR signal); this validation is the entry gate to the present proposal and is not repeated here.

### 4.2 FM trunk (warm-started, frozen)

Class: `monai.apps.generation.maisi.networks.diffusion_model_unet_maisi.DiffusionModelUNetMaisi` with the rflow configuration:

```python
DiffusionModelUNetMaisi(
    spatial_dims=3, in_channels=4, out_channels=4,
    num_channels=[64, 128, 256, 512],
    attention_levels=[False, False, True, True],
    num_head_channels=[0, 0, 32, 32],
    num_res_blocks=2, use_flash_attention=True,
    include_top_region_index_input=False,
    include_bottom_region_index_input=False,
    include_spacing_input=True,
    num_class_embeds=128, resblock_updown=True, include_fc=True,
)
```

The 4-level U-Net has self-attention at the two deepest levels (256 and 512 channels), 8 and 16 heads respectively. Weights are loaded from NV-Generate-MR's released checkpoint and **frozen throughout training**. The modality class embedding is pinned to `mri_t1 = 9` at both training and inference, since the target is T1-weighted. Spacing conditioning is set to $[1.0, 1.0, 1.0]$ mm, matching our preprocessing.

### 4.3 ControlNet for multi-sequence conditioning

The only trainable network in the default setup. Class: `monai.apps.generation.maisi.networks.controlnet_maisi.ControlNetMaisi`. The single deviation from the released configuration is the conditioning input channel count, raised from 8 (the CT-mask default) to 13.

The conditioning tensor is the channel-wise concatenation of the three input-sequence latents and the downsampled tumour mask:

$$
c_{\text{spatial}} \;=\; \mathrm{concat}\big(z_{T_{1\text{pre}}},\, z_{T_2},\, z_{\text{FLAIR}},\, M_{\text{WT}}^\downarrow\big) \;\in\; \mathbb{R}^{13 \times 60 \times 60 \times 40}.
$$

The ControlNet instantiation is:

```python
ControlNetMaisi(
    spatial_dims=3,
    in_channels=4,                                 # = trunk's x_t, unchanged
    num_channels=[64, 128, 256, 512],              # mirrors trunk encoder
    attention_levels=[False, False, True, True],
    num_head_channels=[0, 0, 32, 32],
    num_res_blocks=2, use_flash_attention=True,
    conditioning_embedding_in_channels=13,         # 4 + 4 + 4 + 1
    conditioning_embedding_num_channels=[16, 32, 64],
    num_class_embeds=128, resblock_updown=True, include_fc=True,
)
```

Initialisation follows Zhang et al. *ControlNet*, ICCV 2023 (DOI: [10.1109/ICCV51070.2023.00355](https://doi.org/10.1109/ICCV51070.2023.00355)) [fact]:
- The encoder half of the ControlNet is initialised as a deep copy of the trunk's encoder weights.
- Output projections at every scale are zero-initialised, so $\mathrm{ControlNet}(\cdot) \equiv 0$ at step 0 and the augmented forward pass reproduces the pretrained trunk exactly.

Parameter count: $\sim 80\,\text{M}$ trainable (ControlNet) on top of $240\,\text{M}$ frozen (trunk) and $30\,\text{M}$ frozen (VAE).

### 4.4 Inference pipeline

Heun ODE integration of the rectified-flow velocity field. Class: `monai.networks.schedulers.rectified_flow.RFlowScheduler` with:

```python
RFlowScheduler(
    num_train_timesteps=1000, use_discrete_timesteps=False,
    use_timestep_transform=True, sample_method="uniform", scale=1.4,
)
```

Inference at $N \in \{5, 10\}$ steps. Single-volume wall-clock target on A100 40 GB: $< 10$ s (matches T1C-RFlow's reported budget; Eidex et al. 2025, arXiv:2509.24194) [fact]. Postprocessing: decode through the VAE, optional histogram matching against T1pre to suppress global intensity drift, brain-mask multiplication to zero out non-brain regions.

---

## 5. Loss formulation and curriculum

### 5.1 Final objective and notation

Let $G_\theta(x_t, t, c)$ denote the augmented network — trunk plus ControlNet — predicting the rectified-flow velocity at time $t$ for the noised latent $x_t = (1-t) x_0 + t x_1$, with $x_0 \sim \mathcal{N}(0, I)$ and $x_1 = z_{T_{1c}}$. The target velocity is $u_t = x_1 - x_0$. Let $m \equiv M_{\text{WT}}^\downarrow$ and $m^- \equiv 1 - \mathrm{dilate}(m)$ as in §3.3. The mask-perturbation differential is

$$
\Delta_\theta(x_t) \;\equiv\; G_\theta(x_t, t, c_{\text{orig}}) - G_\theta(x_t, t, c_{\text{perturb}}),
$$

where $c_{\text{orig}}$ carries $M_{\text{WT}}^\downarrow$ and $c_{\text{perturb}}$ zeroes the tumour-mask channel (§5.3). The full merged objective is:

$$
\boxed{\;
\mathcal{L}_{\text{S2}} \;=\; \mathcal{L}_{\text{CFM}} \;+\; \lambda_{\text{contrast}}\,\Big(\lambda_{\text{tum}}\,\mathcal{L}_{\text{roi}}^{(p_t)} \;+\; \lambda_{\text{bg}}\,\mathcal{L}_{\text{bg}}^{(p_b)}\Big)
\;}
$$

with

$$
\mathcal{L}_{\text{roi}}^{(p_t)} \;=\; -\min\!\Big(\tfrac{1}{|m|}\!\sum_{x\in m}\!|\Delta_\theta(x)|^{p_t},\; \delta^{p_t}\Big),
\qquad
\mathcal{L}_{\text{bg}}^{(p_b)} \;=\; \tfrac{1}{|m^-|}\!\sum_{x\in m^-}\!\min\!\big(|\Delta_\theta(x)|^{p_b},\, \delta^{p_b}\big).
$$

Defaults: $p_t = 1$ (MAE), $p_b = 3$, $\delta = 2$, $\lambda_{\text{contrast}} = 10^{-2} \!\to\! 10^{-3}$ (annealed), $\lambda_{\text{tum}} = 0.3$, $\lambda_{\text{bg}} = 1.0$. The curriculum collapses to two stages:

- **S1:** $\mathcal{L}_{\text{CFM}}$ only.
- **S2 (merged $L^p$-contrastive):** S1 + the unified Lp-aware contrastive group above.

Each stage initialises from the previous stage's converged checkpoint. The Skip-S1 alternative (§5.5) trains S2 directly from a freshly-initialised ControlNet, to measure the curriculum's necessity.

**Why the merger.** v0.2 split the loss into a contrastive group operating on the mask-perturbation differential $\Delta_\theta$ and a separate $L^p$ velocity-reconstruction term $\mathcal{L}_{\text{rec}}^{(p)}$ operating on the velocity error $G_\theta - u_t$. The contrastive group enforces *mask-perturbation invariance* outside the tumour and *sensitivity* inside; the $L^p$ term enforces *accuracy* outside. v0.3 collapses the two by raising the contrastive exponent from $p = 1$ to $p_b = 3$ outside the tumour: the differential acts as a focal multiplier on the regions where the model has not yet learned mask-invariance, and global accuracy is preserved by $\mathcal{L}_{\text{CFM}}$ (uniform MSE over the whole volume). The factorised v0.2 objective is retained as a head-to-head ablation row in §8 (*"loss factorisation"*); only one of the two configurations carries forward into the headline result. Note that the factorised ablation also serves as a falsifier of H3: if it improves vessel fidelity but the merged objective does not, the merger has under-weighted accuracy supervision, and we re-introduce the separate $L^p$ velocity-error term.

### 5.2 Stage S1 — Conditional flow-matching loss

Standard rectified-flow loss (Liu et al. *ICLR* 2023, arXiv:2209.03003; Lipman et al. *ICLR* 2023, arXiv:2210.02747) [fact]:

$$
\mathcal{L}_{\text{CFM}}(\theta) \;=\; \mathbb{E}_{t \sim \mathcal{U}[0,1],\, x_0 \sim \mathcal{N}(0,I),\, x_1 \sim p_{T_{1c} \mid c_{\text{spatial}}}}\!\left[\, \big\| G_\theta(x_t, t, c_{\text{orig}}) - (x_1 - x_0)\big\|_2^2 \,\right],
$$

with $c_{\text{orig}} = [z_{T_{1\text{pre}}}, z_{T_2}, z_{\text{FLAIR}}, M_{\text{WT}}^\downarrow]$. This is the standard conditional flow-matching objective and the closest match to the MAISI-v2 trunk's pretraining objective.

S1 is the architectural baseline: it verifies that the conditional architecture converges, provides a checkpoint for S2 to initialise from, and serves as the control condition for the loss-formulation ablation in §8.

### 5.3 Stage S2 — $L^p$-aware tumour-downweighted contrastive regularisation

Algorithm 1 of MAISI-v2 verbatim through step 4, then the loss terms are replaced by their region-dependent $L^p$ variants. At each training step:

1. Sample noise $x_0 \sim \mathcal{N}(0, I)$ and time $t \sim \mathcal{U}[0, 1]$.
2. Form the two conditioning tensors:
   - $c_{\text{orig}} = [z_{T_{1\text{pre}}}, z_{T_2}, z_{\text{FLAIR}}, M_{\text{WT}}^\downarrow]$ — tumour-mask channel populated.
   - $c_{\text{perturb}} = [z_{T_{1\text{pre}}}, z_{T_2}, z_{\text{FLAIR}}, \mathbf{0}]$ — tumour-mask channel zeroed; anatomical channels unchanged.
3. Run two ControlNet forward passes through the *frozen* trunk, sharing $x_t$, $t$, and the modality token. The trunk runs only once (its inputs do not depend on the conditioning); the ControlNet branch runs twice.
4. Compute the differential $\Delta_{\theta}(x_t)$ as in §5.1.

The two loss terms are then

$$
\mathcal{L}_{\text{roi}}^{(p_t)} \;=\; -\min\!\Big(\tfrac{1}{|m|}\!\sum_{x\in m}\!|\Delta_\theta(x)|^{p_t},\; \delta^{p_t}\Big),
\qquad
\mathcal{L}_{\text{bg}}^{(p_b)} \;=\; \tfrac{1}{|m^-|}\!\sum_{x\in m^-}\!\min\!\big(|\Delta_\theta(x)|^{p_b},\, \delta^{p_b}\big),
$$

with defaults $p_t = 1$, $p_b = 3$, $\delta = 2$. Two design choices:

- **$p_t = 1$ (MAE) inside the tumour.** The tumour ROI term is a *push-up* — capped negative L1 wants the differential to be *large* up to $\delta$. With $p_t = 1$, every voxel below the cap contributes a uniform unit gradient, which mechanically saturates the cap at the smallest contrastive weight; with $p_t > 1$ the cap saturates earlier on a smaller subset of voxels and the rest of the tumour mask never reaches the cap. MAE matches the v0.2 default ($\lambda_{\text{tum}} < 1$) — the tumour is the easy region in our paired translation, and the contrastive's job there is to *re-anchor* the tumour to the conditioning channel, not to focus gradient on the worst voxels. [hypothesis 5.3.B; tested by the $p_t$ sweep in §8.]
- **$p_b = 3$ outside the tumour.** Vessels, pituitary, and choroid plexus produce localised mask-perturbation failures — small voxel counts with large $|\Delta_\theta|$. A high-$p_b$ exponent up-weights large-residual voxels by $p_b\,|\Delta_\theta|^{p_b-1}$, so the gradient concentrates on these regions instead of being diluted across the brain bulk (cf. focal regression; Lin et al. *Focal Loss*, ICCV 2017 [fact]). $p_b = 3$ is preferred over $p_b = 4$ because the gradient at the cap edge ($|\Delta_\theta| \to \delta^-$) scales as $p_b\,\delta^{p_b - 1}$ — 12 for $p_b = 3$ versus 32 for $p_b = 4$ on our $\delta = 2$ default. The lower peak gradient gives a 2.6$\times$ stability margin while preserving the focal effect (numerical evidence in §5.4). [hypothesis 5.3.C; tested by the $p_b$ sweep in §8.]

The cap $\delta = 2$ is the MAISI-v2 default; §3.4.1 of that paper reports that removing the cap causes NaN explosions on uncapped $L^1$ contrastive [fact]. The $\sigma$-justification (≈$2\sigma$ of the KL-regularised VAE latents) is verified on the UCSF-PDGM cohort in §5.4. The $\lambda_{\text{contrast}}$ schedule follows MAISI-v2: $0.01$ for the first half of S2, $0.001$ for the second half. The tumour downweighting is the project-specific choice:

- MAISI-v2 default: $\lambda_{\text{tum}} = \lambda_{\text{bg}} = 1.0$, motivated by *amplifying* a missing tumour into unconditional CT synthesis.
- Our default: $\lambda_{\text{tum}} = 0.3$, $\lambda_{\text{bg}} = 1.0$. The tumour is the easy region in our paired translation (large mask, large Gd uptake, mask available as explicit input); the background-consistency term is what we want to recruit, so that vessels, pituitary, choroid plexus, and other non-tumour enhancers appear invariantly under the mask perturbation. $\lambda_{\text{tum}} < 1$ encodes that priority. [hypothesis 5.3.A; tested by the $\lambda_{\text{tum}}$ sweep in §8.]

**Pseudocode (merged S2 training step).**

```python
def capped_lp_mean(diff, region, p, delta):
    """Mean over `region` of min(|diff|^p, delta^p). region: (B,1,*spatial)."""
    abs_p = diff.abs().pow(p)
    capped = torch.minimum(abs_p, abs_p.new_full((), delta ** p))
    denom = region.sum().clamp_min(1.0) * diff.shape[1]   # voxels × channels
    return (capped * region).sum() / denom


def s2_step(controlnet, frozen_trunk, frozen_vae, batch, opt,
            p_t: int = 1, p_b: int = 3,
            delta: float = 2.0,
            lam_contrast: float = 1e-2, lam_tum: float = 0.3, lam_bg: float = 1.0):
    z_t1pre, z_t2, z_flair, z_t1c, m_wt_lat = batch
    t = torch.rand(B).to(device)
    x0 = torch.randn_like(z_t1c)
    xt = (1 - t) * x0 + t * z_t1c                     # straight-line interpolant
    u_target = z_t1c - x0                              # CFM target velocity

    c_orig    = torch.cat([z_t1pre, z_t2, z_flair, m_wt_lat],                 dim=1)
    c_perturb = torch.cat([z_t1pre, z_t2, z_flair, torch.zeros_like(m_wt_lat)], dim=1)

    # Two ControlNet passes share the frozen trunk activations on xt
    v_orig    = frozen_trunk(xt, t, controlnet(c_orig,    xt, t), class_token=9)
    v_perturb = frozen_trunk(xt, t, controlnet(c_perturb, xt, t), class_token=9)

    L_cfm  = ((v_orig - u_target) ** 2).mean()         # global accuracy
    diff   = v_orig - v_perturb                        # mask-perturbation differential
    m_neg  = 1.0 - dilate(m_wt_lat)                    # dilated background

    # ROI term: capped *aggregate* L^{p_t} pushed *up* (negative loss)
    D_roi  = capped_lp_mean(diff, m_wt_lat, p=p_t, delta=delta)
    L_roi  = -torch.clamp(D_roi, max=delta ** p_t)

    # BG term: capped *per-voxel* L^{p_b} pushed *down*
    L_bg   = capped_lp_mean(diff, m_neg,    p=p_b, delta=delta)

    L = L_cfm + lam_contrast * (lam_tum * L_roi + lam_bg * L_bg)
    L.backward(); opt.step(); opt.zero_grad()
```

Three implementation notes for the coder agent:

1. **The ROI cap is applied on the aggregate**, $\min(\mathrm{mean}_m |\Delta|^{p_t},\, \delta^{p_t})$, matching MAISI-v2 §3.4.1 — the cap exists to prevent the *negative* push-up term from running away to $-\infty$ on degenerate inputs. The BG cap is applied **per-voxel** because its purpose is focal — voxel-level outliers should saturate independently, not the regional mean.
2. **No gradient saving from the merger** — both ControlNet passes are still required. The merger removes the $\mathcal{L}_{\text{rec}}^{(p)}$ term, not a forward pass; the wall-clock per step is unchanged from v0.2's S2.
3. **Edge cases worth a unit test.** (i) Empty mask region: `region.sum() == 0` is guarded by `clamp_min(1.0)`; the loss should reduce to a benign zero, not NaN. (ii) Gradient at $|\Delta| = 0$ for $p_b = 3$: `(|Δ|^3).backward()` evaluates to $0$ at $\Delta = 0$ (the polynomial gradient is $3|\Delta|^2 \cdot \mathrm{sign}(\Delta)$, identically zero at the origin); a `torch.testing.assert_close(grad_at_zero, 0)` test entry pins this. (iii) Capped-region partial sum: a synthetic batch with $|\Delta| = 3 > \delta$ should produce zero gradient on those voxels — pin with a forward-backward test asserting `grad[over_cap_voxels] == 0`.

### 5.4 Stability-probe evidence for $p_b = 3$ and $\delta = 2$

Two empirical claims justify the merger and the choice of $p_b = 3$ over $p_b = 4$. Both are verified on `icai-server` against the UCSF-PDGM latent cache and a synthetic optimisation harness (`scratch/contrastive_lp/lp_stability_probe.py`, 2026-06-01).

**Claim 1 — $\delta = 2 \approx 2\sigma$ on the real cohort.** Per-channel standard deviations of MAISI-v2 latents on UCSF-PDGM ($N = 495$, 8-volume sample per modality):

| Modality | per-channel σ (4 channels) | overall σ | mean | $|x|_{\max}$ |
|---|---|---|---|---|
| T1pre | 0.935 / 1.013 / 0.963 / 1.017 | 0.986 | -0.089 | 5.55 |
| T1c   | 0.903 / 1.014 / 0.978 / 1.048 | 0.991 | -0.102 | 6.38 |
| T2    | 0.939 / 1.016 / 0.988 / 1.031 | 0.997 | -0.104 | 5.88 |
| FLAIR | 0.912 / 1.016 / 0.987 / 1.060 | 0.998 | -0.086 | 7.14 |

Per-channel σ lies in $[0.90, 1.06]$ across all four modalities — the MAISI VAE has indeed delivered a KL-regularised latent space with $\sigma \approx 1$. The cap $\delta = 2$ catches the heavier-than-Gaussian tail past $\sim 2\sigma$ (max-abs values of $5$–$7$ confirm the tail is not negligible) without saturating the bulk of the distribution.

**Claim 2 — gradient profile of the capped $L^{p_b}$ at $\delta = 2$.** Analytical gradient of $\min(|x|^{p_b}, \delta^{p_b})$ at $|x| = v$ (below the cap), $g(v, p_b) = p_b\,v^{p_b - 1}$:

| $|\Delta_\theta|$ | $p_b = 1$ | $p_b = 2$ | $p_b = 3$ | $p_b = 4$ |
|---|---|---|---|---|
| 0.10  | 1.00 | 0.20 | 0.03  | 0.00  |
| 0.50  | 1.00 | 1.00 | 0.75  | 0.50  |
| 1.00  | 1.00 | 2.00 | 3.00  | 4.00  |
| 1.99  | 1.00 | 3.98 | **11.88** | **31.52** |
| $\ge \delta$ | 0 | 0 | 0 | 0 |

Cap-edge peak gradient is **2.65× smaller for $p_b = 3$ than $p_b = 4$** ($11.88$ vs $31.52$). Past the cap the gradient is identically $0$, so the focal regime lives in the narrow band $1 \lesssim |\Delta_\theta| \lesssim \delta$. $p_b = 3$ retains the focal up-weighting ($3 \times$ at $|\Delta_\theta| = 1$, the inflection where small residuals become large) at substantially lower peak magnitude, which is the design objective.

**Claim 3 — autograd-level stability.** A 500-step AdamW descent on a $(2, 4, 16, 16, 16)$ synthetic latent under the merged objective — CFM + $L^{p_b}$-contrastive — runs to convergence with no NaN, monotonically decreasing loss EMA, and bounded gradient norm for all three of $p_b \in \{1, 3, 4\}$:

```
[3] p_b=3:  loss[0]=1.0037  loss[end]=-0.0060   grad-norm max=0.011   STATUS: PASS
[3] p_b=4:  loss[0]=1.0037  loss[end]=-0.0060   grad-norm max=0.011   STATUS: PASS
[3] p_b=1:  loss[0]=1.0037  loss[end]=-0.0059   grad-norm max=0.011   STATUS: PASS
```

The toy converges to the same optimum at all three exponents — expected, because Adam adapts to gradient scale and the synthetic target has no vessel-like fine structure. The probe falsifies *gradient blow-up* under $p_b = 3$, not the *transfer to vessel fidelity*; the latter is the question H3 in §1.3 and is tested by the $p_b$-sweep ablation in §8 on real UCSF-PDGM data.

**Claim 4 — contrastive-only SGD stress.** A second harness removes the CFM term, initialises $|\Delta_\theta| \!\approx\! 1.5$ (just below the cap), uses plain SGD ($\mathrm{lr} = 10^{-2}$, no momentum, no Adam rescaling), and runs 200 steps. Plain SGD lets the per-voxel gradient magnitudes propagate without adaptive rescaling, surfacing any blow-up that Adam would have absorbed.

| $p_b$ | finite steps | $\max\|\nabla\|$ | mean $\|\nabla\|$ | $|\Delta|_{\text{bg, mean}}$ start → end | $|\Delta|_{\max}$ start → end | high-grad event (>50) | NaN |
|---|---|---|---|---|---|---|---|
| 1 | 200 | 0.0114 | 0.0114 | 1.19 → 1.19 | 6.84 → 6.84 | no | no |
| 2 | 200 | 0.0202 | 0.0202 | 1.19 → 1.19 | 6.84 → 6.84 | no | no |
| **3** | **200** | **0.0426** | **0.0426** | 1.19 → 1.19 | 6.84 → 6.84 | **no** | **no** |
| 4 | 200 | 0.0923 | 0.0916 | 1.19 → 1.19 | 6.84 → 6.84 | no | no |

Empirical ratio $\|\nabla\|^{(p_b = 4)} / \|\nabla\|^{(p_b = 3)} = 2.17$, consistent with the analytical cap-edge ratio of $2.65$ (the empirical value is smaller because most voxels at $|\Delta_\theta| \!\approx\! 1.19$ are below the cap edge, where the ratio is $4 \cdot 1.19^3 / 3 \cdot 1.19^2 = 1.59$; the empirical sits between the cap-edge and below-cap ratios, as expected). No NaN, no high-gradient event ($> 50$) at any $p_b$; the contrastive gradient is bounded in absolute terms by construction, but the $p_b = 3$ run is $2.17\times$ closer to the $p_b = 1$ baseline than $p_b = 4$ is — the empirical translation of the 2.65× stability margin into the optimisation regime that matters.

**Conclusion.** $p_b = 3$ is the preferred default: equal focal effect, smaller peak gradient, no observed instability. $p_b = 4$ remains in the ablation sweep so the empirical advantage of the smaller exponent can be measured on real data; if $p_b = 4$ matches $p_b = 3$ in stability and beats it on metrics, $p_b = 4$ becomes the default.

The factorised-loss alternative — keep the contrastive at $p = 1$ and re-introduce a separate capped $L^p$ velocity-reconstruction term as in v0.2 — is preserved as an ablation row (§8); its motivation is documented in §5.1 and its falsification criterion is "merged improves vessel fidelity over factorised at equal compute".

### 5.5 Skip-S1 ablation — direct S2 from scratch

The curriculum's premise is that initialising S2 from a converged S1 reduces gradient interference between the CFM and contrastive terms. If this premise is false — i.e., if S2 from scratch reaches the same solution at comparable or lower total GPU-hour cost — the curriculum is unnecessary engineering. We test this explicitly.

**Experiment.** Train ControlNet from the zero-init starting point directly under the S2 objective. Same optimiser, scheduler, batch size, and total step budget as S1 + S2 combined. Compare:

- Validation FID-3D and PSNR-3D trajectories vs wall-clock GPU hours.
- Final test-set metrics.
- Training-time stability (NaN events, loss spikes).

**Decision rule.** If direct-S2 matches or beats curriculum-S2 on final test metrics within the same GPU-hour budget, the curriculum is dropped from the production pipeline. If direct-S2 diverges, stalls, or reaches inferior final metrics, the curriculum is retained. The Skip-S1 ablation is a load-bearing methodological choice, not a presentation polish — its outcome determines the training protocol for the headline result.

A symmetric Skip-S2 experiment (direct factorised S2 from scratch) is *not* in scope for v0.3: any gradient-interference issues observed in the merged direct-S2 will be amplified by the factorised alternative.

### 5.6 Hyperparameters at a glance

| Symbol | Default | Range to sweep | Source / rationale |
|---|---|---|---|
| $\lambda_{\text{contrast}}$ | $0.01 \to 0.001$ | fixed | MAISI-v2 §4.1 default with annealing [fact] |
| $\lambda_{\text{tum}}$ | $0.3$ | $\{0.1, 0.3, 1.0\}$ | downweighting hypothesis (§8) |
| $\lambda_{\text{bg}}$ | $1.0$ | fixed | reference weight |
| $p_t$ | $1$ (MAE) | $\{1, 2, 3\}$ | tumour-mask push-up exponent; MAE saturates the cap uniformly |
| $p_b$ | $3$ | $\{1, 2, 3, 4\}$ | background focal exponent; §5.4 stability margin |
| $\delta$ | $2$ | fixed at MAISI-v2 default | $\approx 2\sigma$ of VAE latents on UCSF-PDGM (§5.4); prevents NaN [fact] |
| **Factorised-loss alternative** | | | |
| $\lambda_{\text{rec}}$ | $0$ in merged; $0.1$ in factorised ablation | $\{0, 0.1\}$ | revives v0.2's separate $L^p$ velocity-reconstruction term |

---

## 6. Training protocol

### 6.1 Optimiser, scheduler, and EMA

- **Optimiser:** AdamW, $\beta_1 = 0.9$, $\beta_2 = 0.95$, weight decay $10^{-2}$.
- **Learning rate:** $5 \times 10^{-5}$ for ControlNet (MAISI-v2's ControlNet default), polynomial decay over the stage's total steps.
- **Warm-up:** $1\,000$ steps at the start of each stage.
- **EMA:** $0.9999$ on ControlNet weights; EMA copy used for validation and final inference.
- **Mixed precision:** bf16 (`torch.cuda.amp.autocast(dtype=torch.bfloat16)`), gradient scaling not required for bf16.

### 6.2 Compute budget per stage

| Stage | Total steps | Batch size | Wall-clock on $1 \times $A100 40 GB | Tentative on Picasso HPC ($4 \times $A100) |
|---|---|---|---|---|
| S1 | $50\,000$ | 4 | $\sim 5$ days | $\sim 1.5$ days |
| S2 (merged $L^p$-contrastive) | $50\,000$ | 4 (effective 2 due to two ControlNet passes) | $\sim 7$ days | $\sim 2$ days |
| Skip-S1 (parallel) | $100\,000$ | 4 (effective 2) | $\sim 14$ days | $\sim 4$ days |
| Factorised-S2 ablation | $50\,000$ | 4 (effective 2) | $\sim 7$ days | $\sim 2$ days |

The Skip-S1 budget equals the sum of S1 + S2 to make the GPU-hour comparison fair. v0.3 drops the separate S3 stage; the factorised-loss alternative is a parallel S2 run, not a sequential follow-up — both runs share the S1 checkpoint and differ only in the loss configuration.

### 6.3 Gates and stopping criteria

Each stage has a quantitative gate before progressing:

- **S1 $\to$ S2:** Validation FID-3D below the MAISI-v2 unconditional-MR baseline ($\approx 2.5$, transferred from their CT figure as an upper bound). [conjecture; the actual brain-MR conditional baseline is unknown until measured]
- **S2 acceptance:** all three of (i) validation FID-3D not worse than S1's; (ii) PSNR-3D restricted to brain $\setminus M_{\text{WT}}$ improved by $\geq 0.5$ dB over S1 (operational definition of H2); (iii) LPIPS-3D improved over S1 *or* clDice on the Frangi vessel mask improved over S1 by $\geq 0.02$ (operational definition of H3). Criterion (iii) is the merger-specific gate — it asks whether $p_b = 3$ buys vessel fidelity, the entire reason for the merger.

Failure to clear a gate triggers an investigation (loss balance, data issue, instability) rather than automatic advancement. If criterion (iii) fails at $p_b = 3$ but the factorised-loss ablation clears it, the merged formulation is rejected and the factorised v0.2 path becomes the production model.

---

## 7. Evaluation

### 7.1 Quantitative metrics

Computed on the 50-subject UCSF-PDGM held-out test set and on the full Málaga external set. All metrics are computed in image space after VAE decoding.

- **PSNR-3D, SSIM-3D**: standard image fidelity, computed (i) over the full brain, (ii) restricted to $M_{\text{WT}}$, (iii) restricted to brain $\setminus M_{\text{WT}}$. The per-region decomposition is the operational test of H2.
- **LPIPS-3D**: 2.5-D approximation, averaging LPIPS over axial, sagittal, coronal slabs (matches the MAISI-v2 evaluation protocol) [fact].
- **FID-3D**: same 2.5-D approximation, RadImageNet ResNet50 features (Mei et al. *Radiology AI* 2022) [fact].
- **Dice on enhancing tumour**: nnU-Net trained on real T1pre + synthetic T1c + T2 + FLAIR, tested on real T1pre + real T1c + T2 + FLAIR. This is the task-based downstream protocol from Preetha et al. *Lancet Digital Health* 2021 (DOI: [10.1016/S2589-7500(21)00205-3](https://doi.org/10.1016/S2589-7500(21)00205-3)) [fact].
- **clDice on Frangi vessel mask** (Shit et al. *clDice*, CVPR 2021, DOI: [10.1109/CVPR46437.2021.01629](https://doi.org/10.1109/CVPR46437.2021.01629)) [fact]: exploratory secondary metric for vessel-fidelity tracking, computed even though the default model is not conditioned on vessels.

### 7.2 Reader study

Two-AFC protocol on the Málaga external set. Three radiologists (target: 2 staff neuroradiologists, 1 senior resident) view paired triples (T1pre, T1c-real, T1c-synthetic) in random anonymised order and answer:

1. Which T1c is real?
2. Confidence: 1–5 Likert.
3. Free-text: any region where the synthesis fails clinically.

Sample size: 60 cases per reader, balanced glioma/meningioma and enhancing/non-enhancing.

### 7.3 Failure-mode analysis

Per-region residual maps (real T1c minus synthetic T1c) overlaid with $M_{\text{WT}}$ and Frangi vessel mask, computed on all test cases. Reported as supplementary figures, with one figure per qualitative failure category (vessel under-enhancement, non-tumour over-enhancement, pituitary failure, meningeal failure).

---

## 8. Planned ablations

| Axis | Conditions | Question answered |
|---|---|---|
| **Loss formulation (CFM contribution)** | S1 (CFM only) / S2-merged (+ $L^p$-contrastive) | Does the contrastive group earn its place? (H2) |
| **Loss factorisation** | S2-merged ($p_t = 1, p_b = 3$, no $\mathcal{L}_{\text{rec}}$) / S2-factorised ($p = 1$ contrastive + separate capped $L^4$ $\mathcal{L}_{\text{rec}}$ as v0.2) | Is the focal effect best located on the contrastive differential or on the velocity-error term? (H3) |
| **Background exponent $p_b$** | $p_b \in \{1, 2, 3, 4\}$ at $p_t = 1$ fixed | Does $p_b = 3$ outperform L1/L2 and beat the $p_b = 4$ alternative on background fidelity? |
| **Tumour exponent $p_t$** | $p_t \in \{1, 2, 3\}$ at $p_b = 3$ fixed | Does MAE inside the tumour beat higher-$p$ alternatives on tumour-region metrics? |
| **Curriculum** | S1 $\to$ S2-merged / direct S2-merged from scratch | Is the curriculum necessary? (H4) |
| **Tumour-weight sweep** | $\lambda_{\text{tum}} \in \{0.1, 0.3, 1.0\}$ at $p_t = 1$, $p_b = 3$ fixed | Is downweighting required, or is MAISI-v2's symmetric weighting sufficient? |
| **Mask conditioning** | with $M_{\text{WT}}$ / without (anatomy channels only) | Does explicit tumour-mask conditioning earn its place? |
| **Trunk init** | NV-Generate-MR warm-start / random init | How much does the foundation model contribute? (H1) |
| **Input modalities** | $\{T_{1\text{pre}}\}$ / $\{T_{1\text{pre}}, T_2, \text{FLAIR}\}$ / $\{T_{1\text{pre}}, \text{SWAN}\}$ / full UCSF-PDGM | Does each modality add information? |
| **Vessel-mask conditioning** | none / + Frangi mask via additional ControlNet channel | Future axis: does an explicit vessel prior add over $L^p$-on-background? (see `soft_priors_sources.md`) |
| **VAE-fine-tuned vs frozen** | frozen MAISI VAE / MAISI VAE fine-tuned on UCSF-PDGM brain-MR | Does adapting the VAE to brain-MR statistics help? |

The *loss-factorisation* row is the headline ablation under v0.3 — it is the falsifier for the central merger hypothesis. The *background-exponent* and *curriculum* rows are the second-tier ablations; the rest are supporting.

---

## 9. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Training NaN under $L^{p_b}$-contrastive ($p_b = 3$) | Low | $\delta = 2$ caps the per-voxel gradient at $p_b\,\delta^{p_b - 1} = 12$ (2.65× smaller than $p_b = 4$ on the same $\delta$; §5.4); $p_b = 3$ specifically chosen for this safety margin; monitored per-step |
| Curriculum gradient interference at S2 (CFM vs $L^{p_b}$-contrastive) | Medium | $\lambda_{\text{tum}} = 0.3$ minimises ROI/CFM interference; the $\lambda_{\text{contrast}}$ anneal reduces the $L^{p_b}$ contribution as training progresses; toy autograd descent (§5.4) shows no instability at $p_b \in \{1, 3, 4\}$ over 500 steps |
| Merger under-supervises background accuracy (no separate $\mathcal{L}_{\text{rec}}^{(p)}$) | Medium | Global $\mathcal{L}_{\text{CFM}}$ MSE provides accuracy supervision; the factorised-loss ablation row in §8 falsifies the merger if it under-performs; production model swaps to factorised v0.2 path if criterion (iii) of §6.3 fails |
| Skip-S1 direct training fails to converge | Medium | Curriculum is the production path; Skip-S1 outcome is informative either way |
| Pathology shift (glioma → meningioma) | Medium | Mix small Málaga subset into final fine-tune under data-sharing; external validation is the test |
| Vendor shift (GE → Siemens/Philips at Málaga) | Medium-high | Histogram matching; pilot on $\sim 10$ Málaga cases before full validation |
| MAISI VAE OOD on certain pathology subtypes | Low | Validated qualitatively on glioma/meningioma in preparatory work; quantitative VAE-roundtrip gate (PSNR > 32 dB, SSIM > 0.95) before training |
| Foundation model bias (NV-Generate-MR training data not seen) | Low–medium | Trunk-init ablation tests the dependency directly |
| Insufficient Málaga sample size | Medium | Open recruitment timeline; longitudinal Tier-2 expands $N$ |

---

## 10. Timeline (indicative)

| Phase | Duration | Deliverables |
|---|---|---|
| Data assembly (UCSF-PDGM preprocessing, Málaga IRB and acquisition) | 6–8 wk | Preprocessed UCSF-PDGM cohort; cached latents; Málaga manifest; data-sharing agreement signed |
| Pipeline implementation (preprocessing, mask extraction, MAISI integration, ControlNet adaptation) | 4 wk | Reproducible preprocessing scripts; encoded latents for all UCSF-PDGM; baseline ControlNet instantiation |
| S1 training | 1.5 wk (Picasso) | S1 checkpoint; entry-gate metrics; sanity-check qualitative samples |
| S2-merged training (+ Skip-S1 and Factorised-S2 in parallel) | 2 wk (Picasso) | S2-merged checkpoint; loss-factorisation ablation complete; curriculum-vs-direct comparison; $\lambda_{\text{tum}}$ sweep complete |
| $p_b$ and $p_t$ exponent sweeps | 1 wk (Picasso) | $p_b \in \{1, 2, 3, 4\}$ and $p_t \in \{1, 2, 3\}$ sweeps complete; final headline configuration locked |
| Internal validation | 2 wk | Quantitative metrics on UCSF-PDGM test |
| External validation (Málaga quantitative + reader study) | 6 wk | External quantitative; reader study completed |
| Writing | 4 wk | MICCAI 2026 submission (March 2026) or *Medical Image Analysis* / *IEEE TMI* journal submission |

---

## 11. Outputs

- One methods paper. Primary target: **MICCAI 2026** main conference. Secondary target: *Medical Image Analysis* or *IEEE Transactions on Medical Imaging*.
- Released code under permissive licence (Apache 2.0 for our additions; NV-Generate-MR weights remain under NVIDIA's OneWay Non-Commercial licence and are not redistributed).
- Released preprocessing pipeline and latent caches (UCSF-PDGM, under TCIA terms).
- Released trained ControlNet weights for the merged-S2 ($p_t = 1, p_b = 3$) model and the curriculum-vs-direct ablation.

---

## References

The references below are those introduced or substantively used in this proposal. Background references on the broader CE-synthesis literature, soft priors, and ancillary techniques live in `literature.md`, `soft_priors_sources.md`, and `priors_validation_protocol.md`.

- Baid, U., et al. (2021). The RSNA-ASNR-MICCAI BraTS 2021 benchmark. *arXiv:2107.02314*.
- Calabrese, E., et al. (2022). The UCSF Preoperative Diffuse Glioma MRI Dataset. *Radiology: Artificial Intelligence* 4(6):e220058. DOI: [10.1148/ryai.220058](https://doi.org/10.1148/ryai.220058).
- Eidex, Z., et al. (2025). T1C-RFlow: 3D Latent Rectified Flow for Brain T1-contrast MR Synthesis. *arXiv:2509.24194*; *Biomedical Physics & Engineering Express*. DOI: [10.1088/2057-1976/ae3e96](https://doi.org/10.1088/2057-1976/ae3e96).
- Feng, Z.-H., et al. (2018). Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks. *CVPR 2018*. DOI: [10.1109/CVPR.2018.00227](https://doi.org/10.1109/CVPR.2018.00227).
- Guo, P., et al. (2025). MAISI: Medical AI for Synthetic Imaging. *WACV 2025*; *arXiv:2409.11169*.
- Isensee, F., et al. (2019). Automated brain extraction of multisequence MRI using artificial neural networks. *Human Brain Mapping* 40(17):4952–4964. DOI: [10.1002/hbm.24750](https://doi.org/10.1002/hbm.24750).
- Isensee, F., et al. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods* 18:203–211. DOI: [10.1038/s41592-020-01008-z](https://doi.org/10.1038/s41592-020-01008-z).
- LaBella, D., et al. (2024). The BraTS 2024 meningioma radiotherapy segmentation challenge. (Cf. *BraTS 2024* proceedings; preprint available.)
- Lin, T.-Y., et al. (2017). Focal Loss for Dense Object Detection. *ICCV 2017*. DOI: [10.1109/ICCV.2017.324](https://doi.org/10.1109/ICCV.2017.324).
- Lipman, Y., et al. (2023). Flow Matching for Generative Modeling. *ICLR 2023*; *arXiv:2210.02747*.
- Liu, X., Gong, C., & Liu, Q. (2023). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. *ICLR 2023*; *arXiv:2209.03003*.
- Mallio, C. A., et al. (2023). Artificial intelligence and contrast-enhanced MRI: a systematic review. *Frontiers in Neuroimaging* 2:1055463. DOI: [10.3389/fnimg.2023.1055463](https://doi.org/10.3389/fnimg.2023.1055463).
- Mei, X., et al. (2022). RadImageNet: An Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning. *Radiology: Artificial Intelligence* 4(5):e210315.
- Preetha, C. J., et al. (2021). Deep-learning-based synthesis of post-contrast T1-weighted MRI for tumour response assessment in neuro-oncology. *The Lancet Digital Health* 3(12):e784–e794. DOI: [10.1016/S2589-7500(21)00205-3](https://doi.org/10.1016/S2589-7500(21)00205-3).
- Rombach, R., et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. *CVPR 2022*. DOI: [10.1109/CVPR52688.2022.01042](https://doi.org/10.1109/CVPR52688.2022.01042).
- Shit, S., et al. (2021). clDice — a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation. *CVPR 2021*. DOI: [10.1109/CVPR46437.2021.01629](https://doi.org/10.1109/CVPR46437.2021.01629).
- Tustison, N. J., et al. (2010). N4ITK: Improved N3 Bias Correction. *IEEE TMI* 29(6):1310–1320. DOI: [10.1109/TMI.2010.2046908](https://doi.org/10.1109/TMI.2010.2046908).
- Zhang, L., Rao, A., & Agrawala, M. (2023). Adding Conditional Control to Text-to-Image Diffusion Models. *ICCV 2023*. DOI: [10.1109/ICCV51070.2023.00355](https://doi.org/10.1109/ICCV51070.2023.00355).
- Zhao, C., et al. (2026). MAISI-v2: Accelerated 3D High-Resolution Medical Image Synthesis with Rectified Flow and Region-specific Contrastive Loss. *AAAI 2026*; *arXiv:2508.05772*.
