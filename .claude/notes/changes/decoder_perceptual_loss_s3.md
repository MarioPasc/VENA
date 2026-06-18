# Decoder-Feature Perceptual Supervision for Latent T1c Synthesis (Stage S3)

*Design note — Mario Pascual González, IBIMA-BIONAND / Universidad de Málaga.*
*Companion to `proposal_contrastive_loss.md` and `training_routine.md`. Self-contained.*

---

## 0. Scope and notation

VENA is a ControlNet branch over a **frozen MAISI-v2 VAE trunk**, trained with a rectified-flow / conditional-flow-matching (CFM) objective to synthesise the contrast-enhanced T1 latent $z_{T_{1c}}$ from non-contrast conditioning. Let:

- $D$ — the frozen MAISI VAE **decoder**; $E$ — its encoder; $\phi_\ell(D;\cdot)$ — the activation tensor at decoder block $\ell$.
- $x_1 = z_{T_{1c}}$ — target latent; $x_0 \sim \mathcal N(0,I)$ — noise.
- $x_t = (1-t)x_0 + t x_1$ — straight-line interpolant ($t=1$ is data, $t=0$ is noise).
- $u_t = x_1 - x_0$ — target velocity; $G_\theta(x_t,t,c)$ — network velocity prediction.

This note defines **Stage S3**, a *decoder-feature perceptual* stage that augments the base CFM objective with an image-aware term computed on intermediate decoder activations. S3 is studied in two initialisations: **warm-started from the converged S1 (CFM-only) checkpoint**, and **trained from scratch**. The region-specific contrastive stage (S2 in the main proposal) is orthogonal to this note and is not required for the S1→S3 study.

---

## 1. Motivation

**The pixel-space wall.** A 3D generative model that operates directly on voxels must run attention over $N = H\cdot W\cdot D$ tokens; self-attention is $\mathcal O(N^2)$ in memory and compute, and for full-resolution brain volumes this is prohibitive on a single A100 40 GB. The ambient pixel manifold is also higher-dimensional and demands far more paired data to cover. Latent diffusion (Rombach et al. 2022, DOI: [10.1109/CVPR52688.2022.01042](https://doi.org/10.1109/CVPR52688.2022.01042)) resolves both problems by compressing to a small latent grid, which is why the MAISI latent trunk is the de-facto standard for 3D medical synthesis (Guo et al. 2025, *WACV*, arXiv:2409.11169; Eidex et al. 2026, *Biomed. Phys. Eng. Express* 12:015075, DOI: [10.1088/2057-1976/ae3e96](https://doi.org/10.1088/2057-1976/ae3e96)) [fact].

**The disconnect this creates.** Training supervises a quantity in latent space (the velocity, or equivalently a reweighted latent reconstruction — see §2.2), but the object of clinical judgement is the **decoded image**. The VAE allocates latent capacity according to *reconstruction* importance, not *clinical* importance; a small-amplitude, spatially localised signal such as gadolinium enhancement (rim, vessels, choroid plexus, pituitary) can therefore receive disproportionately little gradient under a uniform latent metric. This latent–image disconnect is precisely the gap that decoder-side perceptual supervision was introduced to close for latent diffusion (Berrada et al. 2025, *Boosting Latent Diffusion with Perceptual Objectives*, arXiv:2411.04873) [fact].

**The design tension.** We want an image-aware learning signal without paying the pixel-space VRAM cost we deliberately avoided. The resolution is to read the signal from **intermediate decoder features** rather than full RGB output, and to fire it only where it is cheap and reliable. This note specifies how.

---

## 2. The idea and its mathematical background

### 2.1 The one-step image-aware target

The network outputs a velocity, not an image. We never apply an image loss to the velocity directly; we apply it to the **one-step clean estimate** implied by the interpolant:

$$
\hat x_1(x_t,t) \;=\; x_t + (1-t)\,G_\theta(x_t,t,c).
$$

If $G_\theta = u_t$ exactly, then $\hat x_1 = x_t + (1-t)(x_1-x_0) = t x_1 + (1-t) x_1 = x_1$ [fact]. $\hat x_1$ is an affine, fully differentiable function of the network output, so any loss placed on $D(\hat x_1)$ or $\phi_\ell(D;\hat x_1)$ backpropagates cleanly to $\theta$. There is **no regime incompatibility**: the velocity field is supervised by a metric *pulled back through the decoder*.

### 2.2 Why a latent reconstruction term adds nothing, but a decoded term does

A pure-latent reconstruction loss on $\hat x_1$ is algebraically a reweighted velocity loss. From

$$
\hat x_1 - x_1 \;=\; (1-t)\bigl(G_\theta - u_t\bigr)
\quad\Longrightarrow\quad
\lVert \hat x_1 - x_1 \rVert_2^2 \;=\; (1-t)^2\,\lVert G_\theta - u_t \rVert_2^2,
$$

it carries no information beyond a time weight (the standard $v$/$x$/$\epsilon$ parameterisation equivalence; Salimans & Ho 2022, arXiv:2202.00512; Kingma & Gao 2023, *NeurIPS*, arXiv:2303.00848) [fact]. The **decoded** loss is genuinely different: $D$ is nonlinear and **not** an isometry, so $\lVert \phi_\ell(D;\hat x_1) - \phi_\ell(D;\cdot)\rVert$ is not any reweighting of a latent norm. It measures error on a perceptual manifold rather than on the entangled latent coordinates. That metric change is the entire point [fact].

### 2.3 The gating consequence (load-bearing)

The gradient of any decoder-side loss carries the prefactor $\partial \hat x_1/\partial G = (1-t)\,I$:

$$
\frac{\partial \mathcal L_{\text{dec}}}{\partial \theta}
= (1-t)\; J_{\phi_\ell}(\hat x_1)^{\!\top}\,\partial_{\phi}\mathcal L \;\frac{\partial G_\theta}{\partial \theta}.
$$

Hence near data ($t\to1$) the estimate $\hat x_1\to x_1$ is reliable but the gradient is suppressed by $(1-t)\to0$; near noise ($t\to0$) the one-step estimate is a wild extrapolation (and off-manifold for the decoder) yet the prefactor is $\approx1$. **The naïve decoded loss applies its strongest gradient exactly where its target is least trustworthy.** This is the time-dependent singular weighting analysed for the $x$-prediction-with-velocity-loss pairing (Hong et al. 2026, arXiv:2602.10420) [fact], and the off-manifold regime is the documented divergence mode for image-space supervision on a frozen decoder (saturated codes, "broken pixels"; *One Small Step in Latent…* 2025, arXiv:2511.10629) [fact].

**Mitigation — hard high-SNR gate.** Restrict the term to a high-SNR window $t > t_{\min}$. This is exactly what LPL does: the loss is applied only for high signal-to-noise ratios via a hard threshold (Berrada et al. 2025) [fact]. SD3's logit-normal timestep bias toward $t\approx0.5$ is the general-domain instantiation of the same concern (Esser et al. 2024, arXiv:2403.03206) [fact]. A smooth weight $w(t)\propto\min(1/(1-t),c)$ is an alternative but deliberately reintroduces the singularity; the hard gate is preferred and empirically supported [fact].

This gating is a property of $\hat x_1$ (the decoder *input*) and is therefore **depth-independent**: feeding an unreliable $\hat x_1$ corrupts activations at every block. Feature standardisation (§2.4) absorbs some of the magnitude blow-up but not the directional error, so the gate is retained regardless of readout depth [hypothesis: standardised intermediate features tolerate a marginally wider window than raw RGB — checkable in §4].

**Caveat on the LUA citation (arXiv:2511.10629).** The saturated-code pathology was documented in a *pixel-only* training regime (latent super-resolution adapter, no latent anchor) — the curriculum mitigation it proposes is needed precisely because nothing else keeps the latent on-manifold. S3 retains $\mathcal L_{\text{CFM}}$ throughout, which acts as a continuous latent anchor at every step. The pathology is real but the **severity** is plausibly milder in our regime; the gate plus the surviving CFM term should jointly contain it. The §4.4 off-manifold sentinel verifies this empirically.

### 2.4 The decoder-feature perceptual loss

For a chosen set $\mathcal A$ of decoder blocks, define the loss as a depth-weighted, outlier-masked, standardised feature distance between the one-step estimate and a **target reconstruction** $r$ (see §3.3 for the choice of $r$):

$$
\mathcal L_{\text{dec}}
= \mathbb 1[t>t_{\min}]\;
\sum_{\ell\in\mathcal A} \frac{w_\ell}{C_\ell\,\lvert\Omega_\ell\rvert}
\sum_{c=1}^{C_\ell}\sum_{x\in\Omega_\ell}
\rho_{\ell,c}(x)\,\bigl(\phi'_{\ell,c}(D;\hat x_1)[x] - \phi'_{\ell,c}(D;r)[x]\bigr)^2,
$$

where $\phi'$ is the standardised feature (shared statistics, §3.3), $\rho_{\ell,c}\in\{0,1\}$ is the per-channel outlier mask, $\Omega_\ell$ the spatial support, $C_\ell$ the channel count, and $w_\ell$ the depth weight. LPL sets $w_\ell$ to the **inverse of the block's upscaling factor**, because the loss amplitude was found to grow by $\sim2\times$ per $2\times$ resolution increase (Berrada et al. 2025) [fact]. This mirrors classical perceptual losses (Johnson et al. 2016, *ECCV*, DOI: [10.1007/978-3-319-46475-6_43](https://doi.org/10.1007/978-3-319-46475-6_43); Zhang et al. 2018 LPIPS, *CVPR*, DOI: [10.1109/CVPR.2018.00068](https://doi.org/10.1109/CVPR.2018.00068)) but reads from the *autoencoder's own decoder* rather than an external network [fact].

### 2.5 Landscape

| Family | Representative work | Where supervision lives | Relevance |
|---|---|---|---|
| Two-stage LDM | Rombach et al. 2022 | Latent only; VAE pretrained separately with perceptual+adversarial | The baseline disconnect we address |
| Decoder-feature perceptual | **Berrada et al. 2025 (LPL, ICLR'25)** | Decoder intermediate features, high-SNR gated; validated on DDPM, v-pred, and CFM | Direct precedent; the recipe we port |
| Self-perceptual diffusion | Lin & Yang 2024 (ICLR'24, arXiv:2401.00110) | The diffusion model itself as perceptual network; MSE pretrain → self-perceptual finetune curriculum | **Primary** precedent for the S1→S3 warm-start (closer than SRGAN) |
| Multi-step decode | Sargent et al. 2025 (FlowMo, *ICCV*) | Perceptual loss on $n$-step ODE sample, backprop through chain | Higher-cost alternative to one-step $\hat x_1$ |
| End-to-end VAE+DM | Leng et al. 2025 (REPA-E, *ICCV*) | Representation-alignment unlocks joint tuning; raw diffusion loss collapses the latent | Why we keep the VAE frozen |
| Pixel diffusion | DeCo / PixelGen / JiT (2025–26) | Pixel space; avoids VAE, costly | Confirms the pixel-space wall |
| Medical contrast synthesis | T1C-RFlow; MAISI-v2; TuLaBM (Rege 2026, arXiv:2603.19386); TumorFlow (Biller 2026, arXiv:2603.04058) | Latent only; decode at evaluation | The branch with an open gap; TuLaBM and TumorFlow target the same tumour-region failure mode via attention/biophysics rather than perceptual loss — complementary, not competing |
| Adversarial-feature medical synthesis | McCaD (Dayarathna 2025, *WACV*) | Image-space feature loss via patch-GAN discriminator, *not* VAE decoder features | Closest prior with any image-space supervision; uses discriminator instead of the autoencoder's own features |
| SR curriculum | Ledig et al. 2017 (SRGAN) | MSE pretrain → perceptual+adversarial finetune | Older analogue of the S1→S3 warm-start |

**Positioning.** Decoder-feature perceptual supervision is established for natural-image LDMs and demonstrated under flow matching (Berrada et al. 2025, ICLR; Lin & Yang 2024, ICLR) [fact], but the medical contrast-synthesis branch supervises purely in latent space and decodes only at evaluation (Eidex et al. 2026; Zhao et al. 2026, *AAAI*, arXiv:2508.05772; Rege et al. 2026, arXiv:2603.19386; Biller et al. 2026, arXiv:2603.04058) [fact]. A decoder-feature term targeted at the *enhancement-sensitive* layers, introduced as a cheap warm-started refinement, occupies that gap. **No 3D or medical extension of LPL exists in the literature as of June 2026** (verified). The two novel deltas against the LPL baseline are (i) **task-specific layer selection** by enhancement sensitivity rather than generic perceptual depth, and (ii) the **warm-start-vs-scratch ablation**, which the literature flags as underexplored even for natural images (Lin & Yang 2024 has the curriculum but no equal-budget ablation) [conjecture: this is a publishable methodological contribution; falsified if the ablation shows no metric or compute advantage either way]. (iii) the **region-weighted variant** introduced in §2.6, which is targeted at the user-reported deficit (under-synthesised enhancing tumour and missed vessels in S2 outputs).

### 2.6 Region-weighted variant (clinically targeted)

The base $\mathcal L_{\text{dec}}$ in §2.4 weights every voxel of $\Omega_\ell$ equally. The user-reported failure modes — *missed enhancement inside the WT mask*, *missed arteries/veins outside WT*, and *hyper-intense whole-volume outputs* — are **region-localised**, so a uniform feature MSE will spend most of its gradient on bulk parenchyma where the prior model is already adequate, and will dilute the sparse high-magnitude features (vessels, choroid plexus) that the user explicitly cares about. VENA already carries `m_wt` and `m_brain` masks in every training batch; with $\overline{\text{WT}} := \text{brain} \setminus \text{WT}$ the region set is $\mathcal R = \{\text{WT}, \overline{\text{WT}}\}$.

$$
\mathcal L_{\text{dec}}^{\text{region}}
= \mathbb 1[t>t_{\min}]\;
\sum_{\ell\in\mathcal A} \frac{w_\ell}{C_\ell}
\sum_{r\in\mathcal R} \alpha_r \,
\frac{1}{\max(|\Omega_\ell^{(r)}|,\,1)}
\sum_{c=1}^{C_\ell}\sum_{x\in\Omega_\ell^{(r)}}
\rho_{\ell,c}(x)\,\bigl|\phi'_{\ell,c}(D;\hat x_1)[x] - \phi'_{\ell,c}(D;\tilde z)[x]\bigr|^{p_r},
$$

with $\tilde z = D(z_{T_{1c}})$ the VAE-decoded target (per §3.3), $\Omega_\ell^{(r)}$ the region mask resampled to the spatial grid of block $\ell$, and $p_r \in \{1, 2, 3\}$ the per-region exponent (the generalisation of the squared error to an $L^{p_r}$-style term, justified below by the empirical per-region statistics).

#### Empirical anchor (loginexa probe on UCSF-PDGM-0004)

A single patient's $(z_{T_{1\text{pre}}}, z_{T_{1c}})$ pair was decoded through the frozen MAISI VAE, intermediate features $\phi_\ell$ captured at $\ell \in \{0, 2, 5\}$, and $|\phi_\ell(D; z_{T_{1c}}) - \phi_\ell(D; z_{T_{1\text{pre}}})|$ (channel-mean) computed per voxel as the enhancement-signal proxy. Per-region statistics:

| Block | $|\Omega^{(\text{WT})}|$ | $|\Omega^{(\overline{\text{WT}})}|$ | $\overline{|\Delta\phi|}_{\text{WT}}$ | $\overline{|\Delta\phi|}_{\overline{\text{WT}}}$ | ratio mean | $\max_{\text{WT}}$ | $\max_{\overline{\text{WT}}}$ |
|---|---|---|---|---|---|---|---|
| 0 (latent, 256ch) | 644 | 24,422 | 0.766 | 0.637 | **1.20×** | 1.05 | 1.28 |
| 2 (latent, 256ch) | 644 | 24,422 | 1.098 | 0.893 | **1.23×** | 1.85 | 2.30 |
| 5 (2× latent, 128ch) | 5,152 | 195,376 | 1.026 | 0.811 | **1.27×** | 2.40 | **3.19** |

Three findings load-bear on the design:

1. **The mean WT/$\overline{\text{WT}}$ ratio is small (1.20–1.27×).** The WT region's enhancement signal is only marginally stronger than parenchyma in the *mean*. A uniform feature MSE thus does **not** automatically focus on tumour.
2. **The maximum $|\Delta\phi|$ in $\overline{\text{WT}}$ exceeds that in WT** at every block, peaking at $3.19$ vs $2.40$ at block 5. These outlier voxels in $\overline{\text{WT}}$ are consistent with **vessels and small enhancing structures** — sparse, focal, high-magnitude. The mean dilutes them across $\sim 195{,}000$ voxels.
3. **Block 2 features are spatially structured** (see `tools/probes/out/UCSF-PDGM-0004/block_2_axial.png`): the WT contour aligns with a bright spot at $z\in\{24,31\}$, AND there is clear non-WT structure (a cortical/vessel-like ring at $z=16$). Block 5 is more diffuse at the same channel-mean projection but operates at 2× resolution. **The signal lives at level-1 readouts** as much as at level-0 — confirming that mid-depth readouts add information.

The visualisation also reveals a confound: **the channel-mean projection mixes channels that respond to enhancement with channels that do not**, so individual-channel signal is likely sharper than the panel suggests. Per-channel standardisation (§2.4 $\phi'$) plus outlier masking ($\rho_{\ell,c}$) should recover this.

#### Defaults (revised in light of the data)

- **Region weights** $\alpha_r$. The user's revised proposal — $\alpha_{\text{WT}} = 2$, $\alpha_{\overline{\text{WT}}} = 3$ — is **defensible by the data**: with per-region normalisation, $\alpha$ controls each region's share of the loss budget, not per-voxel intensity. The 2 : 3 split (40 % WT, 60 % $\overline{\text{WT}}$) reflects that (a) means are similar (1.27× ratio is small), (b) the highest-magnitude enhancement features live in $\overline{\text{WT}}$, and (c) the user-reported failure mode names *both* WT and $\overline{\text{WT}}$ deficits. Sweep in $\{(1,1), (2,1), (2,3), (3,2)\}$ to bracket; **production default $\alpha_{\text{WT}} = 2, \alpha_{\overline{\text{WT}}} = 3$**.
- **Per-region exponent $p_r$** (Lp variant). The user's proposal — $p_{\text{WT}} = 1$, $p_{\overline{\text{WT}}} = 3$ — directly amplifies the sparse high-magnitude $\overline{\text{WT}}$ tail (vessels) that the mean dilutes. Justified by the table: $\overline{\text{WT}}$'s `max` is the largest single signal in the volume. **Caveats:**
  - $|x|^3$ has gradient $3 |x|^2 \text{sign}(x)$, unbounded for large $|x|$. **Outlier masking $\rho$ is load-bearing**: with $\rho$ thresholded at $k\cdot\text{MAD}$, the per-voxel gradient is bounded by $3 (k\cdot\text{MAD})^2$. Recommend $k=5$ per Berrada 2025.
  - Global gradient clipping (already in production) provides a second safety net.
  - $p_{\text{WT}} = 1$ inside is *robust*-to-outlier; combined with $p_{\overline{\text{WT}}} = 3$ outside, the loss reads "many small/medium errors inside WT (L1 averages well), rare large errors outside (L3 amplifies tails)". This is internally coherent with the empirical picture.
  - **Treat $p_r$ as a sweep axis**, not a default. Production starting point is $p_{\text{WT}} = p_{\overline{\text{WT}}} = 2$ (vanilla L2 in both); the Lp variant runs as $p_{\text{WT}} = 1, p_{\overline{\text{WT}}} = 3$ on a confirmed-stable run only.
- **Soft-region alternative.** Instead of a hard threshold on the 3-channel `tumor_latent` soft probabilities, use the soft probability $p_{\text{tumor}}(x) \in [0,1]$ directly as a continuous weight:
  $$w(x) = \alpha_{\text{WT}}\, p_{\text{tumor}}(x) + \alpha_{\overline{\text{WT}}}\, (1 - p_{\text{tumor}}(x)),$$
  applied per voxel before the $L^p$ accumulation. This avoids the binary-mask boundary discontinuity at zero engineering cost — `tumor_latent` is already 3-channel soft in every cohort's latent H5 (confirmed by `vena.data.h5.latent_domain.manifest`). Worth a sweep entry.
- **Resampling convention.** $\mathcal A = \{2\}$ uses `m_wt` and `m_brain` at native latent resolution (no resampling). $\mathcal A = \{2, 5\}$ requires NN-upsample by 2 at block 5 (`F.interpolate(mode='nearest', scale_factor=2)`). Image-domain masks live in the image H5 and are *not* loaded by the latent dataloader; if K extends to 8, add an image-mask loader path in `LatentH5Dataset` rather than NN-upsampling by 4 from the latent (boundary stair-step would be visible).
- **Empty-region guard.** Already in the formula via $\max(|\Omega|,1)$. Empirically, the BraTS-style cohorts may have a ≤1 % rate of $|m_{\text{wt}}| = 0$ patients (early treatment effect, leptomeningeal disease); §4.7 quantifies this exactly.

**Relation to McCaD (Dayarathna 2025, *WACV*).** McCaD applies a spatial-attentive feature loss via a patch-GAN discriminator. The region-weighted LPL achieves a similar attentional effect through *known anatomy masks* (no adversarial training), reading from the VAE's *own* features. This avoids the GAN-stability cost and matches VENA's existing region-resolved evaluation axes.

The region-weighted variant is treated as the **first-class S3 objective** when the user-reported failure modes are the optimisation target; the uniform §2.4 version is retained as the ablation control. Reference figure: `tools/probes/out/UCSF-PDGM-0004/block_{0,2,5}_axial.png` (per-block $|\Delta\phi|$ with WT overlay, axial slices) — pulled to local from the loginexa probe; commit alongside this doc as supplementary evidence.

---

## 3. Fit within the methodology and the MAISI encoder–decoder

### 3.1 Pipeline placement

VAE encoding is **offline**: latents are cached per subject in HDF5 (the augmented-latent-bank design). S3 reintroduces the decoder into the loop **only** at high-SNR steps and **only** to the depth $\max\mathcal A$. **The full RGB head is not an optional ablation — it is infeasible** on A100-40GB without tile-based decode (cf. §3.5: K=10 OOMs even on a 32 GB V100 in isolation). The base CFM forward already produces $G_\theta$; S3 adds a partial decode of $\hat x_1$ and a partial decode of the cached target reconstruction $\tilde z = D(z_{T_{1c}})$.

### 3.2 Favourable property of the MAISI decoder

The MAISI VAE was trained with a **perceptual objective** alongside reconstruction and KL, with MR intensities normalised so the 0th–99.5th percentile maps to $[0,1]$ (Guo et al. 2025, arXiv:2409.11169) [fact]. Its decoder features are therefore *already* a learned perceptual space — and a **medical** one, plausibly more appropriate than an ImageNet-LPIPS network for radiological structure [conjecture; testable: decoder-LPL vs RadImageNet-feature loss on the enhancing-region metric]. This is the principled justification for reading from $D$ rather than an external perceptual network.

### 3.3 Two MAISI-specific design choices

- **Target the VAE's own reconstruction, not the raw image.** Use $r = z_{T_{1c}}$ decoded through the same $D$, i.e. compare $\phi_\ell(D;\hat x_1)$ against $\phi_\ell(D; z_{T_{1c}})$. Comparing against the raw image would add an irreducible floor equal to the VAE reconstruction error and ask the flow model to correct distortion it cannot represent through the frozen decoder [fact]. Since both arguments pass through $D$, the target features are cached-latent-derived and require no raw-image I/O at train time.
- **Shared, prediction-derived standardisation.** LPL found that standardising *both* target and prediction features with the **statistics of the prediction** markedly outperforms separate normalisation (FID 3.79 vs 4.79) (Berrada et al. 2025) [fact]. With VENA's small effective batch (two ControlNet passes), per-batch statistics are noisy — maintain an **EMA of feature statistics** instead [hypothesis: EMA stats reduce variance without bias; check gradient-norm stability in §5].

### 3.4 Memory reality

Peak GPU memory in the MAISI pipeline occurs during **VAE decoding**, controlled by `autoencoder_tp_num_splits` and sliding-window inference size (MONAI MAISI docs) [fact]. A partial-depth decode-with-backprop on A100 40 GB must budget for this: plan tensor-splitting and/or gradient checkpointing of the decoder *before* assuming the partial decode is free. §3.5 below replaces this paragraph's qualitative warning with the actual probe numbers.

### 3.5 Realised constraints from the MAISI VAE-GAN decoder (probed)

The MAISI VAE was instantiated and probed on a Picasso loginexa **V100-DGXS-32GB** node in fp16 autocast (matching `norm_float16=True` in `configs/autoencoder_v2.json`). **Two probe shapes were measured**: an initial $(B{=}2, C{=}4, 60, 60, 40)$ pass, and a corrected production-shape pass at the actual UCSF-PDGM latent of $(B{=}2, C{=}4, 48, 56, 48)$ — the production crop box `(192, 224, 192)` ÷ 4 (canonical `LATENT_SPATIAL` in `vena.data.h5.latent_domain.manifest`). All subsequent design uses the corrected numbers. The decoder is a flat 11-block `ModuleList` with no skip connections and no attention. Per-block geometry at the production latent:

| Block | Type | Output shape (production latent) | Level | Note |
|---|---|---|---|---|
| 0 | MaisiConvolution | $(2, 256, 48, 56, 48)$ | 0 (latent) | entry conv, 4→256 |
| 1, 2 | MaisiResBlock | $(2, 256, 48, 56, 48)$ | 0 (latent) | 3.54 M params/each |
| 3 | MaisiUpsample (×2) | $(2, 256, 96, 112, 96)$ | 0→1 | |
| 4, 5 | MaisiResBlock | $(2, 128, 96, 112, 96)$ | 1 (2× latent) | 1.36 M / 0.89 M |
| 6 | MaisiUpsample (×2) | $(2, 128, 192, 224, 192)$ | 1→2 | |
| 7, 8 | MaisiResBlock | $(2, 64, 192, 224, 192)$ | 2 (full image res) | |
| 9 | GroupNorm | $(2, 64, 192, 224, 192)$ | 2 | |
| 10 | Conv head | $(2, 1, 192, 224, 192)$ | 2 | RGB-equivalent |

Total decoder parameters: **12.13 M**. Measured cost of one S3 step (target no-grad + prediction with-grad + backward) at depth $\max\mathcal A = K$, at the production latent shape on V100-DGXS-32GB:

| $K$ | Output spatial × channels | **Peak VRAM (V100, fp16)** | Time (target + fwd + back) [s] | Picasso A100-40GB verdict |
|---|---|---|---|---|
| 2 | $48 \times 56 \times 48 \times 256$ | **4.59 GB** | $0.47 + 0.19 + 0.37 = 1.02$ | comfortable; ≈35 GB free for trunk+CN+CFM |
| 5 | $96 \times 112 \times 96 \times 128$ | **26.07 GB** | $0.66 + 0.67 + 2.34 = 3.67$ | fits the decoder alone, but only **~14 GB** left for the rest — joint training needs gradient checkpointing |
| 8 | $192 \times 224 \times 192 \times 64$ | **OOM (>32 GB on V100)** | — | extrapolation ~50–70 GB → over A100-40GB too; **tile-based decode + grad-checkpoint** required |
| 10 | $192 \times 224 \times 192 \times 1$ (RGB) | **OOM** | — | impossible without tiling |

The numbers are ~4× the naive activation-memory extrapolation: autograd saves *all* intermediates inside each `MaisiResBlock` (conv1, norm1, conv2, norm2), not just block outputs. **This forces a redesign of the depth question** — the cost is not a small adjustment to add on top of an S1 step, it is the dominant memory term. The earlier $(60,60,40)$ probe at K=5 reported 29.04 GB; the corrected production-shape probe at $(48,56,48)$ reports 26.07 GB (~10 % reduction, matching the voxel-count ratio).

#### Design ceiling — three regimes (calibrated against Picasso A100 40 GB)

Production S1+FFT on A100 uses approximately 10–14 GB (trunk ~80 M FFT + ControlNet + CFM forward/backward at B=2). The decoder graph adds on top:

1. **K=2 (latent-grid readout only).** $\mathcal A \subseteq \{0, 1, 2\}$. 4.6 GB decoder + ~12 GB trunk/CN = ~17 GB on A100-40GB; ~23 GB margin. **Default for Variant B (cross-device on cuda:1)** where the decoder lives on a separate GPU. The §2.6 empirical anchor shows the WT-vs-$\overline{\text{WT}}$ separation is *equally* strong at block 2 as at block 5 (1.23 vs 1.27) — so K=2 is not a weak-signal cop-out, it is a defensible production target.
2. **K=5 (single readout at block 5, or $\mathcal A=\{2,5\}$).** 26 GB decoder + ~12 GB trunk/CN = ~38 GB — at the A100-40GB ceiling, no headroom for spikes (Adam state transients, attention kernel scratch). **Requires gradient checkpointing on the decoder** (`torch.utils.checkpoint_sequential`, expected ~6–8 GB) **or** cross-device placement on a fresh A100 cuda:1. With grad-checkpointing the budget becomes 8 + 12 = 20 GB, comfortable. Adding block 2 to $\mathcal A$ alongside block 5 incurs **zero extra activation memory** (the forward pass to block 5 already computes block 2's output).
3. **K=8/10 (full-res / RGB readout).** Infeasible without tile-based partial decode (chunked along the depth axis) *and* gradient checkpointing. Defer unless K=5 results justify the engineering cost.

#### Cost estimate per gated step (Picasso A100)

A partial decode at K=5 with gradient checkpointing is expected at ~6–8 GB peak VRAM and ~1.5× the no-checkpoint backward time. On A100, the K=5 step time should be ~1.7× faster than V100 (Volta→Ampere tensor-core uplift) — so $3.67/1.7 \approx 2.2$ s/step without checkpointing, **~3.3 s/step with checkpointing**. With `hi_frac ≈ 0.4` (40 % of micro-batches enter the gated branch), the amortised per-step overhead is **~+30–50 %** over an S1+FFT step (which is ~2.5 s/step on A100 from current production logs). **A 100-epoch warm-started S3 finetune is therefore ~1.3–1.5× the wall-clock of the same-length S1 run** — on Picasso A100, that is ~0.5–1 day extra on top of S1's 7-day budget. **Feasible.**

At K=2 the overhead is much smaller (decoder step ~0.4 s on A100; amortised +~10–15 %), making K=2 the cheapest first experiment.

### 3.6 Compute placement: single-GPU vs device-parallel

Two architectural variants for placing the partial-decode workload:

**Variant A — in-process on the training GPU (cuda:0).** S3 step builds the autograd graph for the partial decode on the same GPU as the trunk+ControlNet. *Pro*: simple — no cross-device gradient transport. *Con*: trunk + ControlNet + partial-decode-with-grad all compete for the same A100-40GB budget; **requires gradient checkpointing for K≥5**. Exhaustive validation continues to run on cuda:1 asynchronously via the existing `ExhaustiveValLauncher`, undisturbed.

**Variant B — device-parallel on cuda:1 (user-proposed).** Decoder is loaded on cuda:1; the prediction tensor $\hat x_1$ is moved cuda:0→cuda:1, the partial decode runs on cuda:1, the scalar loss is moved cuda:1→cuda:0, `loss.backward()` propagates gradients through the cross-device boundary, the `.grad` lands back on cuda:0's $\hat x_1$ feeding the trunk graph. PyTorch autograd handles this natively (cf. `torch.nn.parallel.DistributedDataParallel` internal `to()` ops); cross-device gradient tensor transfer is ~23 MB at PCIe 4.0 (~25 GB/s) → **<1 ms overhead per step**.

*Pro*:
- Frees cuda:0 of the partial-decode VRAM footprint entirely. Trunk+ControlNet+CFM keep the full 40 GB.
- The cuda:1 budget at K=2 is ~5 GB (idle headroom on the 40 GB device), at K=5 is ~29 GB — still over the budget without grad-checkpointing on the cuda:1 side, but the **cuda:0 side is no longer co-stressed**.
- **The probe confirmed gradient transport works** end-to-end (z_pred.grad is finite on cuda:0 after backward through cuda:1, validated in the cross-device branch of the probe script).

*Con*:
- **Conflicts with the existing async `ExhaustiveValLauncher`** which also targets cuda:1. Resolution: serialise — pause training during exhaustive-val cadence epochs (user's proposal). Lower exhaustive-val cadence from every-50-epochs to every-100-epochs, reduce `n_patients` to 25, accept ~10-min pauses ~10 times across a 1000-epoch run = ~100 min of paused training in total. Tolerable.
- **Probe finding**: K=5 *also OOMs* on cuda:1 in isolation (29 GB > 32 GB V100; on A100-40GB it would fit). So Variant B does not unlock K≥5 *by itself*; it must be combined with gradient checkpointing or tile decode. Variant B's true value is at K=2: it frees cuda:0 of the 5 GB footprint and lets exhaustive-val share cuda:1 by serialisation.

*Recommendation*. **K=2 → Variant B** (cleanest separation, paused-val protocol). **K=5 → Variant A with `torch.utils.checkpoint` on blocks 0–5** (single-GPU footprint compressed; the cross-device transport isn't worth it when grad-checkpointing already fits the budget on one GPU). **K≥8 → tile decode + grad-checkpoint + Variant B** (deferred).

---

## 4. Pre-S3 checklist (run before either initialisation)

All items below operate on the **converged S1 checkpoint** and a held-out set of **real paired** $(T_{1\text{pre}}, T_{1c})$ volumes, using cached latents.

### PR-1 status (2026-06-18 — first decoder-LPL PR)

The first PR (this branch) ships the **library primitives** and the
**preflight routine** that closes §4.1, §4.2, §4.3, §4.4, and §4.7b on a
sweep of $N{=}18$ patients × 5 augmentation variants. The S3 training-step
integration and the S1→S3 production YAML are deferred to PR-2 once the
preflight emits its `decision.json` v1.0.

### Post-fix preflight run (2026-06-18T20:18Z) — definitive recipe

After the 2026-06-19 data audit v2 (v4 brain-mask synth-ones patched via
TorchIO seed-replay; aug schemas bumped to 0.2.0; per-cohort intersection
in the patient sampler), the preflight was re-run. **Full 90/90 coverage**
(up from 66/90 in the pre-fix run). Artefact at
`artifacts/preflights/decoder_lpl_profile/2026-06-18T20-18-44Z`.

Decision:

```json
A_recommended    = [2, 3]
w_l              = {2: 0.663, 3: 1.337}     # 99.6% identity vs pre-fix
t_min            = 0.4                       # unchanged
outlier_k        = {2: 5.0, 3: 5.0}          # unchanged
region_recipe    = α=(2,3) + per-cohort overrides
                   (LUMIERE 0.897 ↔ IvyGAP 1.828 → spread 2.04× > 1.5)
allowed_variants = [v0, v1, v2, v3, v4]      # v4 NOW PASSES
v4_brain_mask    = ok
```

**v4 verdict (post-fix data):** v4 pass-rate **98.1%** (106/108 patient-block
pairs) — better than v1, v2, v3 (96.3%, 95.4%, 96.3%). Median drift 0.027
vs the 0.20 gate. Pre-fix v4 was rejected at 43.1% pass-rate; the brain-mask
fix is fully responsible for v4's recovery. Block-5 W/nW v4-vs-v0 ratio is
within 1.0×–1.05× across every cohort (BraTS-GLI 0.88×). The
2026-06-18 audit's signature 2.2-3.4× v4 inflation is **completely gone**.

The recipe is stable under the data fix — `w_l` shifts by only 0.4%, and
A, t_min, outlier_k are byte-identical. The per-cohort α overrides are
larger in the post-fix run (LUMIERE 3.87 vs 2.61, IvyGAP 1.90 vs 3.01)
because the wider inter-cohort spread (2.04× vs 1.96×) accentuates the
mid-point anchor.

### Pre-fix preflight run (2026-06-18T16:40Z) — historical (superseded)

The full sweep ran in 4-way shard on loginexa V100-DGXS-32GB ×4 in ~7
minutes wallclock (~14.5 s per cell × 66 cells). Decision payload at
`/home/mpascual/research/code/VENA/artifacts/preflights/decoder_lpl_profile/2026-06-18T16-40-43Z/decision.json`.

| Knob | Measured | N=4 pilot prediction | Notes |
|---|---|---|---|
| `A_recommended` | `[2, 3]` | `{2, 5}` | **Block 3 (the level-0→level-1 upsample boundary) wins by error-concentration.** The pilot didn't profile block 3 individually; the level-1 readout {5} got the placeholder credit. |
| `w_l` | `{2: 0.666, 3: 1.334}` | inverted Berrada (`w_5 ≈ 2`) | **Berrada's inverse-upscale rule is wrong-signed on MAISI — confirmed at N=18.** The deeper block (3) carries ~2× block 2's depth-weight share. |
| `t_min` | `0.40` | Berrada default `0.70` | **Much earlier knee than SD-VAE.** MAISI's $\hat x_1$ becomes feature-reliable far below Berrada's gate; LPL sees gradient on a wider slice of the trajectory. Verify in S3 ablation whether 0.4 is stable or 0.5–0.6 is a safer production target. |
| `outlier_k` | `{2: 5.0, 3: 5.0}` | Berrada default `5` | Per-channel `p99/MAD` median below the heavy-tail trigger; no widening. |
| `region_recipe.alpha_*` | `(2.0, 3.0)` + per-cohort overrides | sweep `{(1,1), (2,1), (2,3), (3,2)}` | Per-cohort α_notwt spread `2.39 (BraTS-GLI) ↔ 3.63 (REMBRANDT)` — REMBRANDT's sub-unity W/nW ratio (0.96) pushes notWT weight up. |
| `allowed_variants` | `[v0, v1, v2, v3]` (v4 dropped) | drift-gate hypothesis | v1/v2/v3 pass at 94 % of patient-block pairs; **v4 fails at 57 %** (median drift 0.27 vs gate 0.20). |
| `v4_brain_mask_status` | `ok` | possible `broken_drop_v4` | v4 rejection is purely drift-driven; the 3× ratio inflation pattern from the 2026-06-18 data audit was NOT detected at this patient count (could surface with more patients). |

**Inter-cohort W/nW ratios at block 5 v0** (medians, N=3 per cohort):
`BraTS-GLI=1.465`, `LUMIERE=1.337`, `UPENN-GBM=1.169`, `IvyGAP=1.162`,
`UCSF-PDGM=1.143`, `REMBRANDT=0.962`. Spread = 1.465/0.962 = **1.52×**,
just over the 1.5 threshold → per-cohort overrides emitted.

**Per-block magnitude curve** (mean of per-cell mean-norm):
block 0 = 5.90, block 1 = 9.98, **block 2 = 9.17, block 3 = 18.36**,
block 4 = 9.02, block 5 = 8.34. The level-0→1 upsample boundary spikes
magnitude — likely because that block produces 2× spatial features
with the same channel count, so per-voxel L2 norms aggregate.

**Coverage gap (real finding):** only 66 of 90 expected cells ran.
The patient sampler picks 3 per cohort from the **clean** latent H5, but
the augmented H5 covers only a subset of patients in UCSF-PDGM (1/3) and
UPENN-GBM (1/3); BraTS-GLI and IvyGAP at 2/3; LUMIERE and REMBRANDT at
3/3. Missing-variant cells are silently skipped (correct behaviour, no
crash). The §4.7b drift gate sees 72 variant-cells per variant; if more
power is needed, constrain `patient_sampler.eligible_ids` to the
aug-covered subset.

**Closed by PR-1 (code-only, run-independent):**
- §4.5 R6 trunk-EMA resume path → `TrunkEMASnapshotCallback` mirrors the
  shadow into `<ckpt_dir>/trunk_ema_snapshot.pt` on every save;
  `FMLightningModule.setup` reloads it during WARM_START via the
  `_WarmStartCallback.setup` hook + public
  `set_pending_trunk_ema_snapshot` setter. Unit test at
  `tests/model/fm/test_trunk_ema_resume_snapshot.py` (7 sub-tests).
- §4.6 Decoder enumeration + peak-VRAM probe was already done in §3.5.
- §4.6 Gradient-checkpoint prototype → `partial_decode(..., grad_checkpoint=True)`
  is implemented + unit-tested on synthetic inputs (production K=5+grad-ckpt
  A100 memory measurement deferred to PR-2 — needs real trunk co-residency).

**Closes via preflight run (`routines/preflights/decoder_lpl_profile/`):**
- §4.1 magnitude curve, outlier_k, per-channel concentration.
- §4.2 pre/post separation, error-concentration, A selection.
- §4.3 normalisation convention + percentile-invariance (the engine
  asserts foreground_only=True at every decode; the §4.2 separation
  curve, measured in standardised feature space, IS the
  percentile-invariance check).
- §4.4 $\hat x_1$ reliability vs $t$, $t_{\min}$ knee.
- §4.7b coverage, drift gate, allowed-variant set, v4 brain-mask hard
  gate, per-cohort W/nW ratio, empty-region rate.
- §4.7b region-mask resampling sanity check is unit-tested
  (`tests/model/fm/lpl/test_region.py::test_nn_upsample_preserves_corner_position`).

**Deferred to PR-2 (S3 training integration):**
- §4.5 equal-budget warm-start vs scratch experiment.
- §4.5 EMA decay knob (`trunk_ema_decay`) — needs a YAML schema bump.
- §4.5 Keep-CFM-active wiring — `S3` stage in `CompositeLoss`.
- §4.5 Decision rule (the actual S3 ablation experiment).
- §4.6 cross-device end-to-end mini-step (Variant B).
- §4.6 exhaustive-val cadence rebalance (Variant B YAML knob).
- `routines.fm.train.engine._assert_preflight_gates` extension to
  require the `decoder_lpl_profile/LATEST/decision.json` when
  `cfg.run.stage == "s3"`.
- `decision.json` v0.8.0 → v0.9.0 schema bump with the LPL fields.
- `routines/fm/train/configs/runs/picasso_s3_lpl_fft.yaml`.

**Deferred (visual or one-off):**
- §4.1 adversarial-artifact inspection (visual check of mid-depth blocks).
- §4.2 cross-check vs clinical PSNR/SSIM-in-WT (exhaustive-val territory).
- §4.4 off-manifold sentinel (RGB visual sanity below $t_{\min}$).
- §4.4 VAE floor (the §3.5 probe already established this).
- §4.7b empty-region rate threshold tuning — the engine reports it; the
  rule comes after a real loginexa run shows the actual rate.

The checkboxes below stay `[ ]` until the loginexa preflight run lands
and the measured values are written to `decision.json`; the R6 fix is
the single inline tick.

### 4.1 MAISI VAE / decoder profiling

- [x] **Enumerate the decoder.** *Closed by PR-1.* 11-block flat `nn.ModuleList`, confirmed at runtime via `decoder_block_geometry`. Geometry table pinned in §3.5; the live load matched.
- [x] **Per-layer feature magnitude curve.** *Closed by 2026-06-18 loginexa run.* See the PR-1 results block above — `block 0 = 5.90`, `block 1 = 9.98`, `block 2 = 9.17`, **`block 3 = 18.36`** (upsample boundary), `block 4 = 9.02`, `block 5 = 8.34`. Production `w_l = {2: 0.666, 3: 1.334}`. The N=4 pilot's "inverted Berrada" thesis holds (block 5 < block 2: 8.34 < 9.17) but only by 10%, not the 40-60% the pilot suggested.
- [x] **Per-layer outlier fraction.** *Closed.* Per-channel `p99/MAD` ratios below the heavy-tail trigger (10) on every measured block; production `outlier_k = 5.0` everywhere (Berrada default).
- [ ] **Adversarial-artifact inspection.** Confirm from the released MAISI config whether the VAE objective included a patch discriminator (the LDM autoencoder it builds on typically does). If so, the deepest blocks may carry high-frequency hallucinated texture; visually inspect per-block feature maps for checkerboard patterns and bias $\mathcal A$ toward mid-depth blocks accordingly [hypothesis; resolved by inspection].

### 4.2 Enhancement-sensitivity-by-depth (task-specific)

- [x] **Pre/post separation per layer.** *Closed.* See `tables/pre_post_separation.csv`. Available for every block × {WT, notWT, global}.
- [x] **Error-concentration per layer.** *Closed.* `tables/error_concentration.csv`. WT residual per block: `b0=5.98, b1=7.43, b2=10.76, b3=10.50, b4=6.08, b5=7.39`. Aggregator picks `A_recommended = [2, 3]` — the WT-residual peak. WT > notWT at *every* block, consistent with the user-reported "S1 mostly misses gadolinium-uptake regions".
- [ ] **Cross-check against the metric you report.** Correlate the per-layer feature loss with image-space error restricted to the enhancing region. Keep blocks whose feature loss tracks the clinical metric; drop blocks that do not [falsifiable layer-selection criterion].

### 4.3 Normalisation and percentile-invariance

- [ ] **Confirm the normalisation convention.** The MAISI MR preprocessing scales the 0th–99.5th percentile to $[0,1]$ [fact]; ensure S3 targets and predictions are decoded under the identical convention so feature statistics are comparable.
- [ ] **Percentile-invariance check.** Enhancement has a global-intensity component; percentile-foreground normalisation (the same concern flagged for the augmentation bank) may partially cancel it, and the decoder's intensity-augmented training may render features partly invariant to it. Verify the enhancement signal **survives in the standardised feature space** before committing $\lambda_{\text{img}}$ — if §4.2 separation is weak after standardisation, the term will not help [hypothesis; this is the dominant failure risk].

### 4.4 Gating calibration and reconstruction floor

- [x] **$\hat x_1$ reliability vs $t$.** *Closed.* `tables/x1_reliability_vs_t.csv` + `figures/t_min_knee.png`. Mean feature distance: `t=0.30:11.01, 0.40:8.77, 0.50:6.62, 0.60:4.71, 0.70:3.11, 0.80:1.86, 0.85:1.32, 0.90:0.84, 0.95:0.40`. Aggregator's curvature-detection puts the knee at **`t_min = 0.40`** — much earlier than Berrada's 0.70. Curve is monotonically smooth so the knee is soft; PR-2 ablation should sweep `t_min ∈ {0.4, 0.5, 0.6, 0.7}` to confirm 0.4 is not over-aggressive.
- [ ] **Off-manifold sentinel.** Decode a handful of low-SNR $\hat x_1$ to RGB and confirm the saturated-code/"broken-pixel" failure (arXiv:2511.10629) [fact] appears below $t_{\min}$ — this validates that the gate is excluding the divergence regime.
- [ ] **VAE floor.** Record $\lVert D(z_{T_{1c}}) - I_{T_{1c}}\rVert$ on the held-out set as the irreducible reconstruction floor, to interpret S3 metrics and confirm the decision to target $r=D(z_{T_{1c}})$ rather than $I_{T_{1c}}$.

### 4.5 Warm-start-vs-scratch experimental setup

- [ ] **Equal-budget design.** Allocate identical GPU-hours to (A) S3 warm-started from the S1 EMA checkpoint and (B) S3 from a fresh ControlNet init. The warm-start branch's budget is the *additional* hours on top of S1; the from-scratch branch's budget equals S1 + that increment, matching total cost (the framing used for the Skip-S1 ablation).
- [ ] **EMA decay for the short stage — both ControlNet AND trunk EMA.** The production $0.9999$ EMA has a $\sim10^4$-step time constant; a short refinement will not reach the evaluated EMA weights and will appear as a null result. Lower to $\sim0.999$ (or reset) for S3 *for both `self.ema` (ControlNet) and `self.trunk_ema` (the unfrozen-trunk shadow)*. The current LM has only `ema_decay` exposed — extend the config to carry `trunk_ema_decay` if not already symmetric.
- [x] **Trunk-EMA resume path (load-bearing for warm-start).** *Closed by PR-1.* `TrunkEMASnapshotCallback` (in `vena.model.fm.lightning.callbacks.checkpointing`) mirrors `trunk_ema.ema_model.state_dict()` to `<ckpt_dir>/trunk_ema_snapshot.pt` on every save. `FMLightningModule.setup` reloads it via the public `set_pending_trunk_ema_snapshot(path)` setter that `_WarmStartCallback.setup` populates with `Path(ckpt).parent / "trunk_ema_snapshot.pt"` before `module.setup()` runs. Pre-R6 S1 checkpoints lack the snapshot file — a non-fatal warning logs, the trunk_ema starts fresh, and the S3 run still launches (with the documented caveat that the warm-start-vs-scratch comparison loses interpretability). Test coverage at `tests/model/fm/test_trunk_ema_resume_snapshot.py` (7 sub-tests).
- [ ] **Keep CFM active.** Both branches optimise $\mathcal L_{\text{CFM}} + \lambda_{\text{img}}\mathcal L_{\text{dec}}$. The decoder term constrains only high-SNR one-step predictions; CFM anchors the rest of the trajectory and preserves few-step sampling consistency.
- [ ] **Decision rule.** Within equal budget, compare final per-region image metrics (PSNR/SSIM on enhancing tumour and on brain $\setminus M_{\text{WT}}$, LPIPS-3D) and stability (NaN/spike counts, hyper-intense-volume rate). If warm-start matches or beats scratch at lower or equal cost, adopt warm-start; if scratch is strictly better, the enhancement deficit is *not* correctable post hoc and must be trained in — a substantive scientific finding, not a tuning detail [the experiment that the ablation actually answers].

### 4.6 Engineering feasibility (probed; required before launching)

- [x] **Decoder enumeration + peak-VRAM probe** (logged in §3.5). The probe ran on Picasso loginexa V100-DGXS-32GB with fp16 autocast and B=2, latent (60,60,40). Results: K=2 → 5.09 GB, K=5 → 29.04 GB, K=8/10 → OOM. Cross-device autograd was verified to transport `.grad` through cuda:1→cuda:0 correctly.
- [~] **Gradient-checkpointing prototype.** *Code-side closed by PR-1; production memory measurement deferred.* `vena.common.partial_decode(..., grad_checkpoint=True)` wraps the requested slice in `torch.utils.checkpoint.checkpoint_sequential(segments=2, use_reentrant=False)`. Unit test `tests/model/fm/lpl/test_partial_decode.py::test_grad_checkpoint_runs_and_backprops` confirms finite forward + autograd-connected backward on the synthetic decoder. The actual A100-40GB K=5 peak-VRAM measurement under real trunk co-residency is deferred to PR-2 (needs the S3 training-step branch).
- [ ] **Cross-device end-to-end mini-step.** Build a synthetic single-step path: trunk-on-cuda:0 forward → predict $\hat x_1$ → cross to cuda:1 → partial-decode forward → loss → cross-back-and-backward. Confirm gradients land on trunk parameters on cuda:0, peak VRAM on cuda:0 and cuda:1 separately. This is the §3.6 Variant B contract; verifying it on a real trunk forward (not just a dummy latent) catches autograd-quirks unique to MONAI's MAISI trunk.
- [ ] **Exhaustive-val cadence rebalance** (Variant B only). If S3 adopts Variant B, the existing `ExhaustiveValLauncher` cannot share cuda:1 concurrently with the gated-step decoder. Lower `exhaustive_val.cadence_every_epochs` (e.g. 100 instead of 50), reduce `n_patients` (e.g. 25 instead of 50), and add a `Trainer.fit` pause/resume around each exhaustive-val pass. Quantify the wall-clock cost: expected ~10 pauses × ~10 min = ~100 min for a 1000-epoch run.

### 4.7 Per-region enhancement separation (region-weighted variant)

#### 4.7a Mask format reference (the data S3 actually sees)

Pinning this here so the implementer never has to guess. Confirmed via direct inspection of the production latent H5 (`UCSFPDGM_latents.h5`, schema v2.x) and `vena.data.h5.latent_domain.manifest`:

| Field | dtype | shape (per scan) | semantics | source dataset |
|---|---|---|---|---|
| `masks/tumor_latent` | **float32** | `(3, 48, 56, 48)` | **3-channel soft** NETC / ED / ET probabilities in $[0,1]$. Avg-pool-downsampled from int8 BraTS labels via `PerClassAvgPoolDownsampler` (kernel 4, one-hot per class). **Not a binary WT mask.** | Latent H5 (every cohort) |
| `masks/brain_latent` | int8 (loaded as float32) | `(1, 48, 56, 48)` | **Hard binary** brain foreground at latent res. Derived from the int8 image-domain `masks/brain` by avg-pool + threshold. | Latent H5 (v2.0.0+ schema; absent on legacy v1.x) |
| Image-domain `masks/tumor` | int8 | `(192, 224, 192)` | **Raw BraTS labels** at image resolution. BraTS-2021: `{0,1,2,4}`; BraTS-2023: `{0,1,2,3}` (4→3 remap applied at write time). **Not loaded by the latent dataloader** — image H5 only. | Image H5 |
| Image-domain `masks/brain` | int8 | `(192, 224, 192)` | Hard binary brain at image res. | Image H5 |

**Derivation of the binary WT used at training time** (`vena.model.fm.lightning.data.LatentH5Dataset.__getitem__`):

```python
tumor_lat = h5["masks/tumor_latent"][row]                        # (3, 48, 56, 48) soft
soft_union = np.clip(tumor_lat.sum(axis=0, keepdims=True), 0.0, 1.0)
m_wt = (soft_union >= self.wt_threshold).astype(np.float32)      # threshold default 0.5
```

So `m_wt` at training time is a **hard binary** mask at latent resolution, derived from soft 3-channel probabilities.

**Implications for the region-weighted loss variants:**

- **Hard-binary variant (§2.6 default):** $\Omega_\ell^{(\text{WT})} = \{x : m_{\text{wt}}(x) \geq 0.5\}$, $\Omega_\ell^{(\overline{\text{WT}})} = \{x : m_{\text{brain}}(x) = 1 \land m_{\text{wt}}(x) < 0.5\}$. No background-of-brain voxels participate in either region (outside-brain is dropped). Per-block resampling: `F.interpolate(mode="nearest")`.
- **Soft variant:** use the latent-resolution `soft_union = clip(sum(tumor_lat, axis=0), 0, 1) ∈ [0,1]` directly as a per-voxel WT-membership weight, **multiplied by** $m_{\text{brain}}$ to keep outside-brain out. Per-block resampling: `F.interpolate(mode="trilinear", align_corners=False)` (smooths boundaries appropriately for soft probabilities).
- **3-region variant (deferred — sweep extension):** keep the 3 channels separate as $\{\text{NETC}, \text{ED}, \text{ET}\}$ and assign distinct $\alpha$'s. The user-reported failure mode names "the contrast in the tumor", which is the ET (enhancing tumour) channel — a 3-region variant lets ET get higher $\alpha$ than NETC and ED. Not in the first production run; revisit if the 2-region variant is insufficient.

The image-domain masks are **not consumed by the LPL loss** in any current variant — every readout in $\mathcal A \subseteq \{0, \ldots, 5\}$ operates at $\leq 2\times$ latent resolution, where NN-upsampling from the latent mask is exact (it's a recoverable coarsening, not a lossy downsample).

#### 4.7b Coverage requirements for the §4.7 / §4.2 separation curves

The §2.6 empirical anchor used **one** UCSF-PDGM patient (UCSF-PDGM-0004) on $v_0$. That is a sanity probe, not a calibration. Before pinning the production recipe, the separation curve must be measured across:

- [ ] **≥3 patients per training cohort** (UCSF-PDGM, BraTS-GLI, UPENN-GBM, IvyGAP, LUMIERE, REMBRANDT — the 6 cv cohorts; BraTS-Africa-Glioma and BraTS-Africa-Other if cheap to include). Pick patients to span: (i) different WT volumes (small / median / large), (ii) different anatomical locations (frontal / temporal / posterior fossa), (iii) different enhancement intensities (homogeneous / rim / necrotic). The mean-and-spread of the WT-vs-$\overline{\text{WT}}$ ratio across patients must be reported, not just the single-patient point estimate. **Target N = 18 patients minimum** (3 × 6 cv cohorts).
- [ ] **All five augmentation variants per patient**: $v_0$ (clean) plus $v_1, v_2, v_3, v_4$ from the augmented latent H5 (`*_aug.h5`). Variant semantics per the offline-augmentation pipeline:
  - $v_1$: random bias field + gamma (intensity, **inputs only**)
  - $v_2$: histogram shift + gamma (intensity, **inputs only**)
  - $v_3$: noise + anisotropy + blur + motion (degradation, **inputs only**)
  - $v_4$: elastic deformation + affine (rot ±10°, trans ≤ 8 vox, scale 0.9–1.1) — **applied to inputs, target, AND the WT mask** (the only geometric augmentation; most "extreme")
  Per-patient that is 5 runs; per-cohort 3 × 5 = 15 runs; full coverage 18 × 5 = **90 separation-curve evaluations**.
- [x] **Augmentation drift gate.** *Closed at N=18.* Pass-rates per variant (4/72 fail allowed at the 25 % threshold): `v1=94.4%, v2=94.4%, v3=94.4%`, **`v4=43.1%`** (rejected). v4 median drift 0.273 vs gate 0.20. Drift formula:
  $$\text{drift}_{v_k} = \left| \frac{(\overline{|\Delta\phi|}_{\text{WT}} / \overline{|\Delta\phi|}_{\overline{\text{WT}}})_{v_k} - (\overline{|\Delta\phi|}_{\text{WT}} / \overline{|\Delta\phi|}_{\overline{\text{WT}}})_{v_0}}{(\overline{|\Delta\phi|}_{\text{WT}} / \overline{|\Delta\phi|}_{\overline{\text{WT}}})_{v_0}} \right|.$$
  Required: drift < 0.20 (i.e. the ratio of region-mean $|\Delta\phi|$ moves by less than 20 %) for **every** variant on **every** patient at $\ell \in \mathcal A = \{2, 5\}$. The intensity variants $v_1, v_2$ should clear comfortably; the degradation $v_3$ is the marginal one; **$v_4$ is the load-bearing test** because it warps both the latent AND the WT mask — any non-equivariance of the partial decode under affine+elastic shows up here.
- [x] **Failure criterion.** *Closed.* `decision.json::allowed_variants = [v0, v1, v2, v3]`, `v4` dropped. PR-2's S3 train YAML must mask `v4` from its `variant_weights` block (or rebuild the offline aug bank with `v4` excluded). `v4_brain_mask_status` stayed `"ok"` because the 2026-06-18 audit's 3× ratio-inflation pattern wasn't triggered at this patient count — v4 fails the *generic* drift gate, not the brain-mask hard gate.
- [x] **Region-mask resampling sanity check.** *Closed by PR-1.* `tests/model/fm/lpl/test_region.py::test_nn_upsample_preserves_corner_position` and `::test_nn_upsample_preserves_opposite_corner` assert the exact §4.7b contract — a 1-voxel WT at latent corner `(0,0,0)` upsamples to the 2×2×2 corner of the 2× target; a WT at the opposite corner mirrors. Trilinear soft-variant boundary smoothness covered separately.
- [x] **Empty-region rate.** *Closed.* `tables/empty_wt_rate.csv` — every sampled cohort reports `fraction = 0.0` (the patient_sampler picks WT-volume tertiles, so empty-WT patients never get sampled at the 3-per-cohort budget). Re-measure with a wider sampler when relevant. The §2.6 `max(|Ω|, 1)` guard is retained for production safety.

#### 4.7c Provisional findings from the N=4 patient pilot (to be confirmed on the full sweep)

Four cohort probes have run so far on the S1+FFT EMA-best checkpoint with NFE=10 Euler sampling:

| cohort | patient | block 2 L_dec_global v0 | block 5 L_dec_global v0 | block-5 W/nW ratio v0 |
|---|---|---|---|---|
| UPENN-GBM | UPENN-GBM-00571_11 | 1.358 | 0.552 | 1.43 |
| REMBRANDT | 900-00-5381_2005.07.14 | 1.524 | 0.929 | 1.07 |
| BraTS-GLI | BraTS-GLI-01533-000 | 1.369 | 0.615 | **0.82** |
| LUMIERE | Patient-065__week-079 | 1.496 | 0.738 | 1.61 |

These produce **provisional** observations on quantities the design currently leaves open — every item below must be re-verified on the full N=18-patient sweep before being treated as a load-bearing assumption.

- [ ] **PROVISIONAL (N=4, all 4 cohorts × 5 variants): per-channel L_dec at block 2 is essentially flat — strongly confirmed.** Pilot N=4 × 5 variants = 20 observations: max/median per-channel L_dec ratio at block 2 stays in **1.21 – 1.28** for *every* observation; at block 5 it ranges **1.48 – 2.80**. Top-channels-for-50%-loss = 113–116 / 256 (44–45 %) at block 2 vs 48–52 / 128 (38–40 %) at block 5 — also tight. **Implication if confirmed at N=18**: block 2's loss is the integral of many small, broadly-distributed errors — *not* the targeting of a few sharp ones. The "block 2 looks like noise" intuition (raised by the user) is then correct in the LPL-relevant sense: the channel-mean visualisation IS representative, and the perceptual content of block 2 is diffuse rather than selective. **Verification gate**: if the full-sweep block-2 max/median stays < 1.5 on ≥ 80 % of patient-cohort pairs (current pilot: 20/20 = 100 %), downweight block 2 in $\mathcal A = \{2, 5\}$ relative to block 5 (see next item) or drop it from $\mathcal A$ entirely and run K=5 alone.

- [ ] **PROVISIONAL (N=4, 4/4 cohorts): the Berrada inverse-upscale rule is wrong-signed for the MAISI decoder — strongly confirmed.** Berrada 2025 sets $w_\ell \propto 1/\text{upscale}_\ell$ because their SD-VAE shows loss amplitude growing with resolution. Pilot N=4 on the MAISI decoder shows the **opposite** sign on every cohort: block-5 / block-2 L_dec ratio = **0.41 – 0.61** (BraTS-GLI 0.45, LUMIERE 0.49, REMBRANDT 0.61, UPENN-GBM 0.41) — block 5 carries ~half the loss energy of block 2, consistently. **Implication if confirmed at N=18**: applying Berrada's $w_5 = 1/2$ would *compound* this (block 5 then contributes ~1/4 of block 2) — exactly the wrong adjustment. The correct production rule for MAISI is $w_\ell \propto \text{measured magnitude}_\ell$ (UPWEIGHT block 5); a pragmatic default is $w_2 = 1, w_5 = 2$ to normalise the two block contributions to roughly equal LPL energy. **Verification gate**: on the full N=18 sweep, report $\overline{\mathcal L_{\text{dec}}^{(\ell)}}$ per block; if block 5 < block 2 on ≥ 80 % of patient-cohort pairs (current pilot: 20/20 = 100 %), invert the Berrada sign and use the measured ratio.

- [ ] **PROVISIONAL (N=4, inter-cohort spread = 1.96×): per-region L_dec ratio is patient-dependent — global $\alpha$ may misfire on the cohort with the lowest ratio.** Pilot N=4 at block 5 v0, per-cohort:
  - **BraTS-GLI: 0.82** — LPL fires *more outside* WT than inside (sub-unity ratio).
  - REMBRANDT: 1.07 — balanced.
  - UPENN-GBM: 1.43 — moderate WT focus.
  - LUMIERE: 1.61 — strong WT focus.

  Max/min cohort spread = 1.61 / 0.82 = **1.96×** — exceeds the 1.5× threshold I set for "consider per-cohort $\alpha$" in §4.7b. The user's revised default $\alpha_{\text{WT}} = 2, \alpha_{\overline{\text{WT}}} = 3$ (a per-region budget 40 % WT / 60 % notWT) over-corrects further on BraTS-GLI where LPL *already* favours notWT (3.6:1 effective once $\alpha$ scales the existing 0.82 ratio). **Implication if confirmed at N=18**: a single global $(\alpha_{\text{WT}}, \alpha_{\overline{\text{WT}}})$ recipe likely helps some cohorts and hurts others. **Verification gate**: on the full sweep, compute per-cohort median ratio; if inter-cohort spread > 1.5× (current pilot: 1.96×), consider (a) **per-cohort $\alpha$** in the training data-mix, or (b) fall back to the soft-region variant (§2.6) which adapts to per-patient WT probability without committing to a global region budget, or (c) drop region weighting entirely and rely on the uniform §2.4 LPL.

- [ ] **PROVISIONAL (N=4, all 4 cohorts): v4 brain-mask inflation confirmed across cohorts.** Block-5 W/nW ratio v0 → v4: BraTS-GLI 0.82 → **2.77** (3.4×), LUMIERE 1.61 → **5.17** (3.2×), REMBRANDT 1.07 → **2.34** (2.2×), UPENN-GBM 1.43 → **4.28** (3.0×). The factor-of-3 inflation is uniform across cohorts and is the signature of the synth-ones brain-mask fallback documented in the 2026-06-18 data audit. **Action**: do not interpret v4 drift as LPL-relevant until the v4 brain-mask is fixed at the cohort-pipeline level; the §4.7b drift gate must be evaluated *with the corrected v4 masks* once the fix lands. Until then, drop v4 from the LPL augmentation gate.

- [ ] **PROVISIONAL (N=4, REMBRANDT v4 outlier): one cohort × variant combination shows ~50 % LPL energy drop under v4.** REMBRANDT v4 has L_dec_global at block 5 = **0.36** vs v0 = 0.93 — a 60 % drop unique to this patient-cohort pair (other 3 cohorts show v4 L_dec within 10 % of v0). This is consistent with v4 elastic+affine warping a small-WT patient into out-of-distribution geometry the S1+FFT model declines to track. **Implication**: v4-safe augmentation may need a per-cohort gate even after the brain-mask fix. Re-verify on N=18.

### 4.8 NFE / sampling-consistency check (cheap, run before launching)

- [ ] **Cold-vs-warm NFE behaviour on S1.** Before S3 starts, run the {1, 2, 5, 10, 50}-NFE sweep on the converged S1 checkpoint and record per-NFE PSNR/SSIM on the enhancing region. S3's anti-goal is *degrading* few-step sampling consistency; without this baseline, you cannot tell whether an S3 NFE-50 win came at the cost of an NFE-1/2 loss.

---

## 5. S3 integration and monitoring

### 5.1 Objective

$$
\boxed{\;
\mathcal L_{\text{S3}} \;=\; \mathcal L_{\text{CFM}} \;+\; \lambda_{\text{img}}\,\mathcal L_{\text{dec}}^{(\star)}
\;}
$$

where $\mathcal L_{\text{dec}}^{(\star)}$ is **the region-weighted variant from §2.6** (production default; targets the user-reported failure mode directly), with the uniform §2.4 version retained as the ablation control. In both cases: high-SNR gated ($t>t_{\min}$), depth-weighted ($w_\ell$ from §4.1), outlier-masked ($\rho$ from §4.1), shared-EMA-standardised (§3.3), targeting $\tilde z = D(z_{T_{1c}})$ to the depth $\max\mathcal A$ (§3.3). Depth ceiling pinned by §3.5: $\max\mathcal A = 2$ without engineering, $\max\mathcal A = 5$ with gradient checkpointing, $\max\mathcal A \geq 8$ deferred behind tile-based decode.

### 5.2 Training-step sketch

```python
def s3_step(controlnet, frozen_trunk, frozen_vae, feat_stats, batch, opt, cfg):
    """One S3 step: CFM loss + gated decoder-feature perceptual loss.

    feat_stats: EMA accumulator of per-layer feature mean/std (shared norm).
    cfg: t_min, layer set A, depth weights w_l, lambda_img, outlier thresholds.
    """
    z_t1pre, z_t2, z_flair, z_t1c, m_wt = batch
    t  = torch.rand(z_t1c.shape[0], device=z_t1c.device)
    x0 = torch.randn_like(z_t1c)
    xt = (1 - t) * x0 + t * z_t1c
    u  = z_t1c - x0

    c = torch.cat([z_t1pre, z_t2, z_flair, m_wt], dim=1)
    v = frozen_trunk(xt, t, controlnet(c, xt, t), class_token=9)

    loss_cfm = ((v - u) ** 2).mean()

    # Decoder-feature term only on the high-SNR subset (gated).
    hi = t > cfg.t_min
    loss_dec = xt.new_zeros(())
    if hi.any():
        x1_hat = xt[hi] + (1 - t[hi]).view(-1, 1, 1, 1, 1) * v[hi]
        # Partial decode to depth max(A); features returned per block in A.
        feat_pred = frozen_vae.decode_features(x1_hat,  layers=cfg.A)   # grad on
        with torch.no_grad():
            feat_tgt = frozen_vae.decode_features(z_t1c[hi], layers=cfg.A)
        loss_dec = lpl_distance(
            feat_pred, feat_tgt, stats=feat_stats,         # shared EMA standardisation
            outlier_thr=cfg.outlier_thr, depth_w=cfg.w_l,  # masking + depth weights
        )

    loss = loss_cfm + cfg.lambda_img * loss_dec
    loss.backward(); opt.step(); opt.zero_grad()
    feat_stats.update(feat_pred)                            # EMA of prediction stats
    return {"loss_cfm": loss_cfm, "loss_dec": loss_dec, "hi_frac": hi.float().mean()}
```

### 5.3 Hyperparameters

| Symbol | Default | Sweep | Rationale |
|---|---|---|---|
| $t_{\min}$ | 0.7 (from §4.4 knee) | $\{0.6,0.7,0.8\}$ | high-SNR gate (Berrada et al. 2025) [fact] |
| $\mathcal A$ | **$\{2, 5\}$** (one readout per scale; canonical) — *but see §4.7c PROVISIONAL: if block 2's per-channel L_dec stays flat on the full sweep, drop block 2 and use K=5 alone* | $\{2\}$ • $\{5\}$ • $\{0,2,5\}$ • $\{1,2,4,5\}$ | §2.6 empirical anchor: WT/notWT ratio similar at blocks 2 and 5 → both look informative on the channel-mean; but §4.7c N=2 pilot finds block 2 channel concentration ≈ flat (max/median 1.25), so block 2 may be redundant with the latent CFM metric |
| $w_\ell$ | **measured magnitude curve (§4.1)** — *§4.7c PROVISIONAL: pilot N=2 shows block 5 < block 2 in total L_dec, so the Berrada inverse-upscale rule is wrong-signed for MAISI. Production default should UPWEIGHT block 5, not downweight it. Pin from the full sweep.* | $w_5 / w_2 \in \{1, 2, 4\}$ | balance depth contributions; the Berrada rule is SD-VAE-specific and the sign must be inverted for MAISI (§4.7c) |
| $\lambda_{\text{img}}$ | small, **annealed in** | $\{0.01,0.05,0.1\}$ | the term is global; cold introduction perturbs the warm start |
| $\alpha_{\text{WT}}, \alpha_{\overline{\text{WT}}}$ (region-weighted) | **$(2, 3)$** | $\{(1,1), (2,1), (2,3), (3,2), (1,3)\}$ | §2.6 empirical: $\overline{\text{WT}}$ holds the highest-magnitude features (vessels, max 3.19 vs WT max 2.40 at block 5); user-reported deficit names both regions |
| $p_{\text{WT}}, p_{\overline{\text{WT}}}$ (Lp exponents) | **$(2, 2)$** (L2 in both, default) | $(1, 3)$ as ablation (sparse-tail amplifier outside WT) | $p=3$ amplifies sparse high-magnitude tails (vessels) but needs outlier masking + grad clip; $p=1$ inside is robust |
| outlier $k$ (per-channel MAD threshold) | 5.0 | $\{3, 5, 7\}$ | per Berrada 2025; load-bearing for the $p=3$ sweep |
| Soft-region weighting | off (binary WT) | on (continuous `tumor_latent` sum) | smoother boundaries; uses the already-3-channel-soft latent mask |
| EMA decay (S3, ControlNet **and** trunk) | $0.999$ | — | short-stage EMA must track (§4.5) |
| grad-checkpoint segments (if K≥5) | 2 (segments) | $\{1, 2, 4\}$ | trade extra backward time for VRAM (§4.6) |
| compute placement | Variant B at K=2; Variant A + grad-checkpoint at K=5 | swap as ablation | §3.6 |

### 5.4 What to monitor

- **The two terms separately.** Log $\mathcal L_{\text{CFM}}$ and $\mathcal L_{\text{dec}}$ independently in `train_step.csv`; a falling $\mathcal L_{\text{dec}}$ with a rising $\mathcal L_{\text{CFM}}$ signals the perceptual term dragging the model off the velocity field — reduce $\lambda_{\text{img}}$.
- **Decoder-term gradient norm.** Track $\lVert\partial\mathcal L_{\text{dec}}/\partial\theta\rVert$; spikes indicate the gate is admitting unreliable $\hat x_1$ (raise $t_{\min}$) or outlier masking is insufficient.
- **High-SNR fraction.** Log `hi_frac`; if too few steps fire the term, the gate is starving S3 of signal — widen the window or bias $t$ sampling toward high SNR for the gated term only.
- **Feature-statistic drift.** Monitor the EMA feature stats; large drift means the standardisation reference is unstable (small-batch noise) — increase the EMA window.
- **Per-region image metrics (the actual target).** At validation, decode and report PSNR/SSIM on the WT region and on brain $\setminus M_{\text{WT}}$, LPIPS-3D — S3's justification is improvement *here*, not in aggregate latent loss. VENA does not carry a vessel mask, so vessel performance is read indirectly through whole-volume LPIPS-3D and visual inspection.
- **Hyper-intense-volume rate.** Per validation patient, flag predictions where the 99.5th percentile of the decoded volume exceeds the 99.5th percentile of $D(z_{T_{1c}})$ by more than, say, 30 %. The user-reported failure mode is "we produce hyper-intense volumes" — a region-weighted $\mathcal L_{\text{dec}}^{(\star)}$ with $\alpha_{\overline{\text{WT}}}>0$ should suppress this rate over training. Track the rate as a first-class metric in `train_epoch.csv`.
- **Off-manifold sentinel.** Periodically decode a few low-SNR $\hat x_1$ to RGB and watch for saturated-code/broken-pixel onset (arXiv:2511.10629) [fact]; early appearance is the leading indicator of the cold-start divergence S3's warm start is meant to avoid. Caveat per §2.3: CFM remains active here, so severity should be milder than in the LUA setting.
- **NFE sweep + sampling consistency.** Keep the $\{1,2,5,10,50\}$-NFE validation sweep; confirm S3 does not degrade few-step sampling (the risk if CFM anchoring weakens).
- **Cross-device wall-clock (Variant B only).** Log the cuda:0↔cuda:1 transfer latency per gated step; if it exceeds ~5 ms (vs. predicted <1 ms), PCIe contention from exhaustive-val artifacts is in play and Variant B's serialisation policy needs tightening.

### 5.5 Sweep extensions (secondary axes the doc surfaces explicitly)

These are real-but-secondary design axes that did not exist in the original §2.4 formulation; each is a one-flag change in the LM and is worth a sweep row before declaring the production recipe.

| Axis | Default | Variant | Why bother |
|---|---|---|---|
| **Feature distance per region** | $L_2$ in both regions ($p=2$) | **$p_{\text{WT}}=1, p_{\overline{\text{WT}}}=3$** (per §2.6 empirical anchor) | The §2.6 probe shows the highest-magnitude $\|\Delta\phi\|$ voxels live in $\overline{\text{WT}}$ (vessels). $p=3$ outside amplifies these sparse tails; $p=1$ inside is robust to outliers. Requires outlier mask $k=5$ and grad-clip on. |
| **Region weights** | $\alpha_{\text{WT}}=2, \alpha_{\overline{\text{WT}}}=3$ | $(1,1), (3,2), (1,3)$ | Bracket the user's intuition (notWT-favoured, 2:3 split) against WT-favoured and uniform. |
| **Soft-region weighting** | off (binary WT threshold) | on (continuous `tumor_latent` sum as per-voxel weight) | Smoother boundary at zero engineering cost; `tumor_latent` is 3-channel soft in every cohort H5. |
| **Layer set $\mathcal A$** | $\{2, 5\}$ | $\{2\}$ • $\{5\}$ • $\{0,2,5\}$ • $\{1,2,4,5\}$ | Verifies that one-readout-per-scale is the right granularity (no intra-scale redundancy). |
| **Gated-branch $t$-sampling** | uniform | biased toward $t > t_{\min}$ for the gated term only | With B=2 and `hi_frac=0.4`, the dec term fires on $\sim 0.8$ samples/step — barely above noise. Biased sampling raises the effective dec gradient SNR without changing the marginal $t$ distribution of CFM. |
| **CFG-dropout interaction** | gated branch fires on dropped-conditioning samples | skip dec term for samples where the WT condition was dropped | Open question whether the unconditional branch should be pulled toward the perceptual target. A clean ablation isolates conditional vs unconditional S3 gradient. |
| **Compute placement** | per §3.6 (Variant A + grad-checkpoint at K=5; Variant B at K=2) | swap A↔B | Quantify the cross-device transport cost in a real training step. |

Add one row per sweep to the `ablations.yaml` registry; do not multiply sweeps combinatorially — each row varies one axis at a time from the recipe pinned in §5.3.

---

## 6. Risk summary (compact, for the design-review reader)

| ID | Risk | Mitigation in this doc |
|---|---|---|
| R1 | Standardised feature space cancels enhancement signal (§4.3) | §4.2 + §4.7 separation curves; falsify before launching |
| R2 | Off-manifold $\hat x_1$ at low $t$ corrupts decoder features (§2.3) | hard high-SNR gate (§5.3); §4.4 sentinel; CFM kept active to anchor the latent |
| R3 | Decoder VRAM cost at meaningful depth (§3.5) | Variant A at K=2 (5 GB) or K=5 with grad-checkpoint (~8 GB); Variant B for K=2 |
| R4 | `hi_frac × B=2` starves the dec term of gradient signal | Sweep biased $t$-sampling (§5.5); or accumulate dec loss across grad-accum cycles |
| R5 | MAISI VAE's patch-discriminator training may inject high-freq texture into late blocks (§4.1) | Memory cap already biases $\mathcal A$ toward mid-depth; visually inspect §4.1 |
| R6 | Trunk-EMA shadow re-initialised on resume → warm-start ablation invalid (§4.5) | Either fix `setup()` to load `trunk_ema_snapshot.pt` or document the reset |
| R7 | Exhaustive-val ↔ Variant B contention on cuda:1 | Lower cadence + n_patients + pause-resume (§4.6) |
| R8 | CFG dropout interaction with the gated branch (§5.5) | Ablate; default = include dropped samples in dec term |
| R9 | Berrada's inverse-upscale $w_\ell$ rule is 2D-SD-VAE-specific (§2.4) | Measure 3D MAISI magnitude curve in §4.1 and use the measured weights |
| R10 | NFE-1/2 degradation from a strong dec gradient (§5.4) | §4.8 baseline + §5.4 NFE sweep monitor + reduce $\lambda_{\text{img}}$ if NFE-1 drops |

---

## 7. References

- Berrada, T., Astolfi, P., Hall, M., Havasi, M., Benchetrit, Y., Romero-Soriano, A., Alahari, K., Drozdzal, M., Verbeek, J. (2025). Boosting Latent Diffusion with Perceptual Objectives. *ICLR 2025*; *arXiv:2411.04873*.
- Biller, V., Bubeck, N., et al. (2026). TumorFlow: Physics-Guided Longitudinal MRI Synthesis of Glioblastoma Growth. *arXiv:2603.04058*.
- Chang, H., et al. (2025). Controllable Flow Matching for 3D Contrast-Enhanced Brain MRI Synthesis from Non-contrast Scans. *MICCAI 2025*; DOI: 10.1007/978-3-032-05325-1_12.
- Dayarathna, S., et al. (2025). McCaD: Multi-Contrast MRI Conditioned, Adaptive Adversarial Diffusion Model for High-Fidelity MRI Synthesis. *WACV 2025*; *arXiv:2409.00585*.
- Eidex, Z., et al. (2026). An Efficient 3D Latent Diffusion Model for T1-contrast Enhanced MRI Generation. *Biomed. Phys. Eng. Express* 12:015075. DOI: [10.1088/2057-1976/ae3e96](https://doi.org/10.1088/2057-1976/ae3e96); *arXiv:2509.24194*.
- Esser, P., et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis. *arXiv:2403.03206*.
- Guo, P., et al. (2025). MAISI: Medical AI for Synthetic Imaging. *WACV 2025*; *arXiv:2409.11169*.
- Hong, J., et al. (2026). Binary Flow Matching: Prediction-Loss Space Alignment for Robust Learning. *arXiv:2602.10420*. **Caveat**: binary/discrete-FM analogy only; for the continuous-FM $(1-t)^2$ singular-weighting argument cite Salimans & Ho 2022 + Kingma & Gao 2023 directly.
- Johnson, J., Alahi, A., Fei-Fei, L. (2016). Perceptual Losses for Real-Time Style Transfer and Super-Resolution. *ECCV 2016*. DOI: [10.1007/978-3-319-46475-6_43](https://doi.org/10.1007/978-3-319-46475-6_43).
- Kingma, D. P., Gao, R. (2023). Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation. *NeurIPS 2023*; *arXiv:2303.00848*.
- Ledig, C., et al. (2017). Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network. *CVPR 2017*. DOI: [10.1109/CVPR.2017.19](https://doi.org/10.1109/CVPR.2017.19).
- Leng, X., Singh, J., Hou, Y., Xing, Z., Xie, S., Zheng, L. (2025). REPA-E: Unlocking VAE for End-to-End Tuning with Latent Diffusion Transformers. *ICCV 2025*; *arXiv:2504.10483*.
- Lin, S., Yang, X. (2024). Diffusion Model with Perceptual Loss. *ICLR 2024*; *arXiv:2401.00110*. **Primary precedent** for the MSE-warm-start → perceptual-finetune curriculum; uses the diffusion model itself as the perceptual network rather than the VAE decoder.
- Lipman, Y., et al. (2023). Flow Matching for Generative Modeling. *ICLR 2023*; *arXiv:2210.02747*.
- Liu, X., Gong, C., Liu, Q. (2023). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. *ICLR 2023*; *arXiv:2209.03003*.
- Preetha, C.J., Meredig, H., Brugnara, G., et al. (2021). Deep-learning-based synthesis of post-contrast T1-weighted MRI for tumour response assessment in neuro-oncology. *Lancet Digital Health* 3(12):e784–e794. DOI: 10.1016/S2589-7500(21)00205-3.
- Razin, A., Kazantsev, D., Makarov, I. (2025). One Small Step in Latent, One Giant Leap for Pixels: Fast Latent Upscale Adapter for Your Diffusion Models. *arXiv:2511.10629*. **Caveat**: pixel-only training without a latent anchor; S3's CFM anchor softens the cited pathology.
- Rege, A., Dukre, A.M., Balci, N., Mahapatra, D., Razzak, I. (2026). TuLaBM: Tumor-Biased Latent Bridge Matching for Contrast-Enhanced MRI Synthesis. *arXiv:2603.19386*.
- Rombach, R., et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. *CVPR 2022*. DOI: [10.1109/CVPR52688.2022.01042](https://doi.org/10.1109/CVPR52688.2022.01042).
- Salimans, T., Ho, J. (2022). Progressive Distillation for Fast Sampling of Diffusion Models. *arXiv:2202.00512*.
- Sargent, K., Hsu, K., Johnson, J., Fei-Fei, L., Wu, J. (2025). Flow to the Mode: Mode-Seeking Diffusion Autoencoders for State-of-the-Art Image Tokenization. *ICCV 2025*; *arXiv:2503.11056*.
- Zhang, L., Rao, A., Agrawala, M. (2023). Adding Conditional Control to Text-to-Image Diffusion Models. *ICCV 2023*. DOI: [10.1109/ICCV51070.2023.00355](https://doi.org/10.1109/ICCV51070.2023.00355).
- Zhang, R., et al. (2018). The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS). *CVPR 2018*. DOI: [10.1109/CVPR.2018.00068](https://doi.org/10.1109/CVPR.2018.00068).
- Zhao, C., Guo, P., Yang, D., Tang, Y., He, Y., Simon, B., Belue, M., Harmon, S., Turkbey, B., Xu, D. (2026). MAISI-v2: Accelerated 3D High-Resolution Medical Image Synthesis with Rectified Flow and Region-specific Contrastive Loss. *AAAI 2026*; *arXiv:2508.05772*. DOI: 10.1609/aaai.v39i22.38309.
