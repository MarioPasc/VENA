# Literature Review: Gadolinium-Free Synthesis of T1 Post-Contrast Brain MRI

*Working bibliography, Mario Pascual González. Original: November 2026.
Update (2026-06-08): added cross-contrast MR-synthesis foundational baselines
(pGAN, ResViT, SynDiff, MM-GAN), recent FM/diffusion 2025–2026 entries
(BBDM-Med, MedFM, ContrastFlow), the MAISI-v2 region-specific contrastive
reference, and a benchmarking-selection section (§12) that pins which subset
of these methods will be trained head-to-head against VENA for the MedIA /
MICCAI submission. Section 10's comparison table was extended with a
"code available" column and a "to be benchmarked" annotation.*

---

## 1. Scope and framing

This review covers the methods literature on synthesising the post-contrast T1 image (T1c) from non-contrast brain MRI sequences, the architectural template we adopt (latent generative models with ControlNet conditioning over the MAISI VAE), and the vessel-segmentation methods relevant to our soft-prior conditioning. Clinical motivation and SWI-specific literature are summarised at the end.

The field has gone through three methodological eras in roughly five years: CNN/GAN (2019–2023), conditional diffusion (2024), and flow matching / rectified flow (2025–2026). Every published methods paper to date — across all three eras — takes inputs from $\{T_{1w}, T_{2w}, \text{FLAIR}\}$, with the single exception of Kleesiek et al. (2019), which included SWI in a multi-channel Bayesian U-Net. The community has been optimising the *generator* while leaving the *input space* untouched. The Mallio et al. (2023) systematic review identifies small vessels and non-tumour enhancement as the dominant unresolved failure mode.

Our work sits at the intersection (flow matching) ∩ (SWAN as input) ∩ (vessel-aware conditioning outside the tumour), which is empty in the existing literature.

In addition to the CE-specific corpus above, a MedIA/MICCAI submission has to be benchmarked against the broader **cross-contrast MR-synthesis** family — paired-translation methods that have become the standard reference in IEEE TMI / MedIA over the last five years (Dar 2019 pGAN, Dalmaz 2022 ResViT, Özbey 2023 SynDiff, Sharma & Hamarneh 2020 MM-GAN). These were not designed for the T1pre→T1c task specifically, but they are the methodological baselines a reviewer will ask for, and they bracket the GAN→ViT→diffusion architectural transition that frames our flow-matching contribution.

---

## 2. CNN- and GAN-era methods (2019–2023)

### Kleesiek et al. 2019
*Can virtual contrast enhancement in brain MRI replace gadolinium?: A feasibility study.* **Investigative Radiology** 54(10):653–660. DOI: 10.1097/RLI.0000000000000583.

The original feasibility paper. 10-channel Bayesian 3D U-Net: T1w, T1c (target), T2w, FLAIR, multi-shell DWI, **and four SWI channels**. To our knowledge the only published model that used SWI as input. Brain glioma cohort.

*Our position.* The direct precedent for the SWI-input hypothesis, but pre-FM, pre-latent-model, with no vessel-aware loss and no explicit vascular evaluation. Validates the *clinical* feasibility of SWI as input; we generalise the *methodological* framework to latent flow matching and add vessel-resolved evaluation.

### Jayachandran Preetha et al. 2021
*Deep-learning-based synthesis of post-contrast T1-weighted MRI for tumour response assessment in neuro-oncology: a feasibility study.* **The Lancet Digital Health** 3(12):e784–e794. DOI: 10.1016/S2589-7500(21)00205-3.

The most clinically validated work in the subfield. dCNN trained on $\sim 2500$ exams from the multicentre EORTC-26101 trial. Critical empirical finding: adding ADC beyond pre-contrast anatomical sequences did *not* improve synthesis; the most relevant inputs were T1w, FLAIR, T2w in that order.

*Our position.* The validation template — they introduced the task-based downstream evaluation (training tumour segmentation on real T1c, testing on synthetic) which we adopt in §6.4 of the proposal. Their input ablation tested only conventional anatomical sequences; they did not consider SWI/SWAN. Our contribution extends their ablation logic to a vascular-prior input not in their study.

### Osman et al. 2023
*Deep-learning-based synthesis of post-contrast T1-weighted brain MR images using a DD-Res U-Net 3D model.* **Journal of Applied Clinical Medical Physics** 25(2):e14120. DOI: 10.1002/acm2.14120.

3D DD-Res U-Net on BraTS-2021, $N = 1{,}251$. High-water mark for pure CNN approaches. Reports inferior performance in tumour regions vs whole brain; introduces tumour-mask region weighting as partial mitigation.

*Our position.* Confirms that region-aware loss weighting is helpful. We extend this principle from tumour-only weighting to vessel-and-tumour weighting, with masks derived from target T1c (not input) to avoid shortcut learning.

### Solak et al. 2025
*Generative adversarial network-based virtual contrast enhancement in brain MRI.* **Academic Radiology**. PMID: 39694785.

GAN-based, conventional inputs. Adversarial training improves perceptual quality but introduces mode-collapse risk on lesions.

*Our position.* GAN era is superseded by FM/RF in 2025 in terms of measured FID/SSIM and inference stability; we do not consider GAN baselines except as a historical comparison.

---

## 2.5. Cross-contrast MR-synthesis baselines (general task, MedIA/IEEE-TMI standard set)

**None of the methods in this subsection targets T1c (T1Gd) synthesis as
its headline task.** They are *generic* paired multi-contrast MR
translation frameworks (T1 ↔ T2, T1 ↔ PD, T2 ↔ FLAIR, etc.) validated on
IXI and BraTS-2015 / BraTS-2018, where T1c is included only as one of
several available channels. Their contribution is *architectural* (the
backbone the community pivoted to in each generation), not *task-specific*.

We include them because MedIA / IEEE TMI / MICCAI reviewers consistently
expect the icon-lab pGAN / ResViT / SynDiff suite as a baseline triad: a
T1c-synthesis paper that does not beat the strongest *general*
cross-contrast architecture is not a methods contribution, only a task
contribution. The *T1c-specific* competitors (Kleesiek 2019, Preetha
2021, Osman 2023, McCaD 2024, Piening 2024, CFM 2025, PMRF 2025,
T1C-RFlow 2025, TLP 2025, TumorFlow 2026) are in §2–§5 and remain the
direct prior-art comparisons.

### pGAN — Dar et al. 2019
*Image Synthesis in Multi-Contrast MRI With Conditional Generative Adversarial Networks.* **IEEE TMI** 38(10):2375–2388. DOI: 10.1109/TMI.2019.2901750. Code: https://github.com/icon-lab/pGAN-cGAN.

The seminal paired multi-contrast translation network (perceptual-loss-augmented conditional GAN, U-Net + PatchGAN). Validated on IXI and BRATS-2015 across T1↔T2 / T1↔PD / T2↔FLAIR. The most-cited paper in cross-contrast MR synthesis (>700 citations as of 2026); a TMI/MedIA review without it as a baseline is structurally incomplete.

*Our position.* GAN-era anchor in our benchmark. Slot in §12.

### MM-GAN — Sharma & Hamarneh 2020
*Missing MRI Pulse Sequence Synthesis Using Multi-Modal Generative Adversarial Network.* **IEEE TMI** 39(4):1170–1183. DOI: 10.1109/TMI.2019.2945521.

Single multi-input multi-output GAN that synthesises any missing sequence from any subset of the others (BraTS-2015). Architecturally relevant because it shares VENA's many-to-one structure (multiple input contrasts → one synthesised contrast).

*Our position.* Optional in the GAN tier; pGAN is sufficient as the GAN anchor unless reviewers ask for the many-to-one variant.

### ResViT — Dalmaz et al. 2022
*ResViT: Residual Vision Transformers for Multimodal Medical Image Synthesis.* **IEEE TMI** 41(10):2598–2614. DOI: 10.1109/TMI.2022.3167808. Code: https://github.com/icon-lab/ResViT.

Hybrid CNN-transformer generator with aggregated residual transformer (ART) blocks for multi-contrast MR synthesis. Validated on IXI and BraTS-2018 — outperforms pGAN, MM-GAN, and pre-diffusion baselines. Standard mid-2020s strong baseline in the icon-lab benchmark suite.

*Our position.* Transformer/CNN-era anchor. Cheaper to train than diffusion competitors; covers the "what if we just used a strong supervised generator?" question.

### SynDiff — Özbey et al. 2023
*Unsupervised Medical Image Translation with Adversarial Diffusion Models.* **IEEE TMI** 42(12):3524–3539. DOI: 10.1109/TMI.2023.3290149. Code: https://github.com/icon-lab/SynDiff.

Adversarial diffusion bridge for cross-contrast MR; unsupervised + paired variants. Reports state-of-the-art on multi-contrast MR translation circa 2023. *Paired* SynDiff is the one we run.

*Our position.* Diffusion-era anchor in cross-contrast MR (distinct from the T1c-specific diffusion entries in §3). Trains in a domain we can match (same datasets, paired mode).

### BBDM-Med — Wu et al. 2024 / 2025 follow-up
*Brownian Bridge Diffusion Models for Image-to-Image Translation in Medical Imaging.* **CVPR / MedIA follow-up**. Code: https://github.com/xuekt98/BBDM.

Original CVPR 2023 BBDM (Li et al.) reformulated as a diffusion bridge between paired domains; medical adaptations (2024–2025) directly target paired MR translation. Conceptually closest to T1C-RFlow's bridge but with a stochastic differential rather than rectified-flow interpolant — useful to disentangle "bridge formulation" from "FM vs SDE choice".

*Our position.* Optional; include only if §12 budget allows the fourth diffusion entry.

---

## 3. Conditional diffusion era (2024)

### McCaD (Dayarathna et al. 2024)
*Multi-Sequence Consistent Diffusion for Contrast-Enhanced MRI Synthesis.* Multi-sequence conditioning with consistency constraints.

*Our position.* Architecturally adjacent; the multi-sequence consistency idea is orthogonal to our vessel-aware conditioning and could be combined in a future extension.

### Piening et al. 2024
*Conditional Generative Models for Contrast-Enhanced Synthesis of T1w and T1 Maps in Brain MRI.* arXiv:2410.08894.

First paper to deploy both conditional diffusion *and* conditional flow matching for CE synthesis with posterior-sample uncertainty quantification. Compares T1w synthesis against quantitative T1 mapping. Glioblastoma cohort.

*Our position.* The most relevant immediate predecessor in terms of generative framework. Their UQ machinery is reusable. They do not condition on SWI/SWAN and do not evaluate vessel fidelity.

---

## 4. Flow matching and rectified-flow era (2025–2026)

This is the current frontier. All four papers below operate on $\{T_{1w}, T_{2w}, \text{FLAIR}\}$ inputs; none use SWI/SWAN; none evaluate vessel fidelity outside the tumour.

### CFM — Chang et al., MICCAI 2025
*Controllable Flow Matching for 3D Contrast-Enhanced Brain MRI Synthesis from Non-contrast Scans.* DOI: 10.1007/978-3-032-05325-1_12.

Controllable Flow Matching with straight-line generation paths, enabling single-step inference. Auxiliary segmentation task, multi-stage training. Operates in a custom latent space with a Swin-UNETR-based encoder–decoder.

*Our position.* The first FM paper for this task. Their controllability via segmentation conditioning is the architectural ancestor of our ControlNet design. They use only anatomical inputs and do not address the vessel failure mode.

### PMRF — Brandstötter & Kobler, MICCAI-SASHIMI 2025
*Synthesizing Accurate and Realistic T1-weighted Contrast-Enhanced MR Images using Posterior-Mean Rectified Flow.* arXiv:2508.12640.

Two-stage architecture: a patch-based 3D U-Net predicts the voxel-wise posterior mean; a time-conditioned rectified flow refines towards the perceptual manifold. On BraTS-2023–2025 reports axial FID 12.46 and KID 0.007, ~68.7% lower FID than the posterior mean alone. Explicitly claims to restore "lesion margins and vascular details realistically".

*Our position.* The two-stage perception–distortion framework is principled and could be combined with our SWAN conditioning in a future extension. Their vessel-fidelity claim is qualitative and not measured on a vessel-specific metric. We will benchmark against PMRF and adopt clDice as a quantitative test of their claim.

### T1C-RFlow — Eidex et al. 2025–2026
*T1C-RFlow: 3D Latent Rectified Flow for Brain T1-contrast MR Synthesis.* arXiv:2509.24194; **Biomedical Physics & Engineering Express**. DOI: 10.1088/2057-1976/ae3e96.

Current numerical SOTA. 3D latent rectified flow on BraTS-2024 GLI + MEN + MET ($N \approx 4{,}100$). Generates sT1C volumes in <10 s. Outperforms SOTA DDPM, Pix2Pix, and DiT-3D.

*Our position.* Direct quantitative baseline. We adopt their inference-time target (<10 s/volume) and benchmark against their numbers on BraTS. Their architecture trains its own VAE rather than using MAISI; we use MAISI to leverage CT/MRI foundation-model priors.

### TLP — Li et al. 2025
*Transformer with Localization Prompts for T1-Contrast Synthesis.* arXiv:2503.01265.

Interactive synthesis with radiologist-provided spatial prompts. Demonstrates that explicit spatial conditioning helps.

*Our position.* Supports the design intuition that spatial priors (in our case, vessel and tumour masks) carry useful information that pure pixel-space input cannot encode. We make the spatial priors automatic rather than radiologist-provided.

### Osuala et al. 2024
*Towards Learning Contrast Kinetics with Multi-Condition Latent Diffusion Models.* arXiv:2403.13890.

Multi-condition latent diffusion targeting contrast kinetics (breast DCE-MRI), not brain. Architecturally relevant because it explicitly conditions a latent diffusion on a *temporal* enhancement signal — the closest non-brain analogue to our tumour-mask conditioning.

*Our position.* Cited as related; not benchmarked (different anatomy and acquisition protocol).

### MAISI-v2 (region-specific contrastive loss) — Zhao et al. 2026
*MAISI-v2: Accelerated 3D High-Resolution Medical Image Synthesis with Rectified Flow and Region-specific Contrastive Loss.* **AAAI 2026**. arXiv:2508.05772.

The source of our $L^p$-aware contrastive formulation (§5 of the proposal). MAISI-v2 amplifies tumour ROI signal for unconditional CT synthesis; we invert the sign of $\lambda_{\text{tum}}$ for the paired translation setting.

*Our position.* The methodological parent of VENA's loss. Re-cited here for completeness — it is also the unconditional-CT baseline that motivates the foundation-model warm-start.

---

## 5. Architectural template: TumorFlow and MAISI

### TumorFlow — Biller et al. 2026
*TumorFlow: Physics-Guided Longitudinal MRI Synthesis of Glioblastoma Growth.* arXiv:2603.04058. Code: https://github.com/valentin-biller/lgm.

The architectural scaffold we adopt. Latent rectified flow over the MAISI VAE (latent shape $\mathbb{R}^{4 \times 60 \times 60 \times 40}$). ControlNet-style spatial conditioning: $c_{\text{spatial}} = \text{concat}(z_s, z_{tc})$ injected via feature-wise addition $h_\ell \leftarrow h_\ell + \mathcal{F}_\ell(c_{\text{spatial}})$. Their conditioning channels are tissue segmentation and biophysical tumour-concentration field; the Fisher–Kolmogorov physics is irrelevant to our problem but the conditioning machinery transfers directly.

*Companion paper.* Biller et al. arXiv:2510.09365 — same architecture for tumour inpainting. Confirms MAISI VAE frozen, nearest-neighbour downsampling of conditioning to latent resolution.

*Our position.* Direct architectural template. We replace the biophysical tumour-concentration latent with our SWAN-derived vessel-mask latent, retain the tumour-mask channel, and remove the temporal/physics components.

### MAISI v1 / v2 — Guo et al. (MONAI)
*MAISI: Medical AI for Synthetic Imaging.* Foundation VAE trained on 37,243 CT volumes + limited MRI (BraTS T1/T1c/T2/FLAIR). MAISI-v2 (arXiv:2508.05772) replaces DDPM with rectified flow, retains the v1 VAE without fine-tuning, and adds ControlNet for region-specific contrastive loss.

*Our position.* We reuse the MAISI VAE (frozen) for T1 pre-contrast encoding. We do *not* encode SWAN through MAISI VAE because SWI/SWAN is out-of-distribution for the v1 training set; we encode the vessel mask only, which is binary and OOD-safe.

---

## 6. Vessel-specific contributions

### Frangi vesselness — Frangi et al. 1998
*Multiscale vessel enhancement filtering.* MICCAI 1998. DOI: 10.1007/BFb0056195.

Classical Hessian-based vesselness. The eigenvalue ratios $R_A, R_B$ encode plate-vs-line and blob-vs-line discrimination; tubular structures score high, blobs (CMBs) score low. Available in `skimage.filters.frangi`.

### Jerman vesselness — Jerman et al. 2016
*Enhancement of Vascular Structures in 3D and 2D Angiographic Images.* **IEEE TMI** 35(9):2107–2118. DOI: 10.1109/TMI.2016.2515603.

Addresses Frangi's underestimation at vessel junctions and aneurysms.

### Shearlet vesselness — Ward et al. 2022
*Shearlet-Based Vesselness for Susceptibility-Weighted and Quantitative Susceptibility Mapping.* **NeuroImage** 261:119062. DOI: 10.1016/j.neuroimage.2022.119062.

Purpose-designed for SWI/QSM. Reports significant outperformance over Frangi and recursive vesselness on gradient-echo data. MATLAB code at https://github.com/SinaStraub/GRE_vessel_seg.

*Our position.* Better quality than Frangi but MATLAB-only and not Python-native. We default to Frangi for pipeline reasons; consider Shearlet as a future quality upgrade.

### Vessel-CAPTCHA — Brina et al. 2022
*Vessel-CAPTCHA: A weakly-supervised deep learning approach for brain vessel segmentation.* **Medical Image Analysis** 75:102263. DOI: 10.1016/j.media.2021.102263. Code: https://github.com/ngoc-vien-dang/Vessel-Captcha.

Weak-supervision approach (sparse 2D labels). Designed for TOF + SWI. Code available (TensorFlow/Keras, 5 years old); pretrained model not released.

*Our position.* Methodologically interesting but practically blocked by dated framework and missing weights. We consider it for re-implementation only if Frangi proves insufficient.

### DeepVesselNet — Tetteh et al. 2020
*DeepVesselNet: Vessel Segmentation, Centerline Prediction, and Bifurcation Detection in 3-D Angiographic Volumes.* **Frontiers in Neuroscience** 14:592352. DOI: 10.3389/fnins.2020.592352. Code: https://github.com/giesekow/deepvesselnet.

PyTorch implementation, pretrained on synthetic vessels + TOF-MRA. Domain shift from MRA to SWI is non-trivial — MRA shows arteries, SWI shows veins.

*Our position.* Available pretrained weights make it the simplest deep-learning baseline, but the MRA→SWI domain shift limits direct applicability. Reserve for fine-tuning if needed.

### Livne et al. 2019
*A U-Net Deep Learning Framework for High Performance Vessel Segmentation in Patients with Cerebrovascular Disease.* **Frontiers in Neuroscience** 13:97. DOI: 10.3389/fnins.2019.00097.

Strong Dice on manually-labelled TOF data. Trained model not released.

*Our position.* Reference point only.

### SHIVA-CMB — Hassine et al. 2024
*SHIVA-CMB: a deep-learning-based robust cerebral microbleed segmentation tool trained on multi-source T2*GRE- and susceptibility-weighted MRI.* **Scientific Reports**. DOI: 10.1038/s41598-024-81870-5.

Open pretrained CMB segmentation on SWI. Detects what we want to *exclude* from the vessel mask.

*Our position.* Indirect utility — can be used to actively remove CMB false positives from the vessel mask in cohorts with high CMB burden.

### Morrison et al. 2021
*Automated detection of cerebral microbleeds on T2*-weighted MRI.* **Scientific Reports** 11:4404. DOI: 10.1038/s41598-021-83607-0.

Confirms empirically that Frangi vesselness on SWI rejects CMBs by shape (CMBs have low vesselness because they are blob-shaped, not tubular).

*Our position.* Key supporting reference for the design choice in §3.2 of the proposal. Justifies Frangi as a CMB-robust vessel mask.

### DeepSWI — Genc et al. 2023
*DeepSWI: Synthesizing SWI from Conventional MRI Sequences with Deep Learning.* **JMRI**. DOI: 10.1002/jmri.28622.

Synthesises SWI from T2* — the *reverse* direction of our problem.

*Our position.* Demonstrates that the SWI–anatomical mapping is learnable; not directly comparable but relevant prior art.

### Optimally Oriented Flux — Law & Chung 2008
*Three Dimensional Curvilinear Structure Detection Using Optimally Oriented Flux.* **ECCV 2008**. DOI: 10.1007/978-3-540-88693-8_27.

Flux-based curvilinear-structure detector that resolves the Hessian-based methods' two known weaknesses at junctions and bifurcations (Frangi underestimates branch points; Jerman partially fixes this). OOF integrates the gradient flux over an oriented spherical neighbourhood, which preserves response at branchings instead of suppressing them as Frangi does.

*Our position.* Third operator in the multi-filter ensemble for background-contrast evaluation (validation_proposal.md §4.5). Python implementations in `pyvane` and `OOF-tool`; CPU-only but fast at our latent-equivalent volume sizes.

### VesselFM — Wittmann et al. 2024
*VesselFM: A Foundation Model for Universal 3D Blood Vessel Segmentation.* **MICCAI 2025** / arXiv:2411.17386. Code+weights: https://github.com/bwittmann/vesselFM. License: Open RAIL++-M (research/non-commercial — fine for us).

Foundation model for 3D vessel segmentation trained on a curated corpus
($D_{\text{real}}$, paper Table 1) covering **brain MRA** (TubeTK,
SMILE-UHURA, DeepVesselNet, CSD, TopCoW-MRA), mouse-brain vEM,
mouse-brain OCTA, and liver CT. The four zero-shot evaluation domains
in the paper (§4.1) are **MRA, vEM, OCTA, CT — none of which are
structural brain MRI**. The paper does *not* claim, demonstrate, or
evaluate zero-shot performance on $T_{1\text{pre}} / T_{1c} / T_2 /
\text{FLAIR}$. The earlier draft of this entry overstated the scope.

Inference contract (audited 2026-06-08 against
`vesselfm/seg/inference.py` and `configs/inference.yaml`):
single-channel 3D NIfTI; internal percentile normalisation
`ScaleIntensityRangePercentiles(1, 99)` (we re-tune to `(0.5, 99.5)`
with `foreground_only=True` because our skull-stripped volumes have a
large zero background that distorts global percentiles); sliding-window
inference at $128^3$ patches with overlap 0.5; output is a whole-volume
binary uint8 mask after sigmoid + threshold (threshold knob exposed in
`merging.threshold`). One-shot fine-tuning is supported via
`vesselfm/seg/finetune.py`.

*Our position.* Deep-learning entry in the `validation_proposal.md`
§4.5 multi-operator ensemble, **conditional on an empirical spot-check
on ~20 real T1c volumes from UCSF-PDGM before it enters the protocol**.
$T_{1c}$ is the closest analogue in our modality set to VesselFM's MRA
training distribution (bright tubular Gd-enhanced lumen on a suppressed
background, qualitatively similar to TOF-MRA), but Gd-BBB-disruption
physics differ from flow-related enhancement: over-segmentation of
choroid plexus, pituitary, dural sinuses, and enhancing tumour is a
documented physical risk for which VesselFM's training distribution
offers no precedent. If the spot-check fails, the §4.5 ensemble falls
back to the three classical operators (F, J, O) with a "≥ 2-of-3
agree" consensus rule. The one-shot fine-tuned variant runs as a
sensitivity experiment against the zero-shot variant. **Modality choice
at inference: feed $T_{1c}$ alone** — VesselFM has no multi-channel
input head.

---

## 7. Clinical context

### Mallio et al. 2023
*Artificial intelligence and contrast-enhanced MRI: a systematic review.* **Frontiers in Neuroimaging** 2:1055463. DOI: 10.3389/fnimg.2023.1055463.

The decisive review for our framing. Identifies small vessels and small lesions as the unresolved failure mode across nearly all published synthesis methods.

*Our position.* Provides the primary justification for the vessel-fidelity gap our work targets.

### Wamelink et al. 2024
*Brain Tumor Imaging without Gadolinium-Based Contrast Agents: Feasible or Fantasy?* Critical review of the field.

*Our position.* Tempers the optimism of the early FM papers; we acknowledge their critique in our framing and design our evaluation specifically to address the issues they raise.

### MAGNET trial — NCT05754476
Beijing Tiantan Hospital, Yaou Liu PI. Multicentre prospective study planning to combine T1WI, T2WI, FLAIR, DWI/ADC, ASL, APT-CEST, **and SWI/QSM** for virtual T1c. No specific architecture proposed.

*Our position.* Independent clinical-community evidence that our input choice is correct. They confirm the *what* (include SWI); we provide the *how* (latent flow matching with vessel-mask conditioning).

### Gulani et al. 2017
*Gadolinium deposition in the brain: summary of evidence and recommendations.* **The Lancet Neurology**.

Authoritative summary of the gadolinium deposition evidence motivating contrast-free MRI research.

### Maggi et al. 2015
*The effect of gadolinium-based contrast agents on SWI brain venous structures.* DOI: 10.1177/2047981614560938.

Shows that Gd modulates vein conspicuity on SWI. Supports the mechanistic argument that SWAN-visible structures and T1c-enhancing structures overlap substantially (the vessel-class-mismatch concern is bounded).

---

## 8. Foundational machine learning references

### Conditional flow matching — Lipman et al. 2023
*Flow Matching for Generative Modeling.* arXiv:2210.02747; **ICLR 2023**.

Theoretical foundation for §4.1 of the proposal. Introduces the simulation-free training of continuous normalising flows via velocity-field regression on conditional probability paths.

### Rectified flow — Liu et al. 2023
*Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow.* **ICLR 2023**.

Reflow procedure that straightens the integration path, enabling 1–10 step sampling. Adopted by T1C-RFlow, TumorFlow, MAISI-v2.

### Stochastic interpolants — Albergo et al. 2023
*Building Normalizing Flows with Stochastic Interpolants.* **ICLR 2023**.

Companion framework, occasionally cited alongside CFM in the medical-imaging literature.

### ControlNet — Zhang et al. 2023
*Adding Conditional Control to Text-to-Image Diffusion Models.* **ICCV 2023**. DOI: 10.1109/ICCV51070.2023.00355.

The conditioning mechanism we adopt. Parallel branch processing condition input, feature-wise addition into the trunk at every scale. Originally for text-to-image; adapted to medical segmentation conditioning by MAISI-v2 and TumorFlow.

### Shortcut learning — Geirhos et al. 2020
*Shortcut Learning in Deep Neural Networks.* **Nature Machine Intelligence**. DOI: 10.1038/s42256-020-00257-z.

Formalises the failure mode we diagnose in §6.5 of the proposal: the model may learn "voxel ∈ $M_v$ → predict bright" rather than the underlying contrast kinetics.

---

## 9. Datasets

### UCSF-PDGM
Calabrese et al., **Radiology AI** 2022. DOI: 10.1148/ryai.220058. TCIA DOI: 10.7937/tcia.bdgf-8v37. $N = 501$, preoperative diffuse glioma, 3 T GE Discovery 750, full multiparametric protocol including SWI. **The only large public dataset with paired T1pre, T1c, and SWI for tumour patients.** Primary training resource.

### BraTS series (2023, 2024, 2025)
Glioma, meningioma, metastasis, pediatric, sub-Saharan-African cohorts. T1, T1c, T2, FLAIR — **no SWI**. Used by all published FM/RF methods as the standard benchmark; we use it for baseline benchmarking only.

### UCSF-BMSR
UCSF Brain Metastases SRS, $N = 412$ patients, 560 MRIs. T1pre, T1c, FLAIR, subtraction — no SWI confirmed.

### UPenn-GBM
TCIA, $N \approx 630$. T1, T1c, T2, FLAIR; no SWI in the released distribution.

### Ivy GAP / IvyGAP-Radiomics
T1, T1c, T2, FLAIR; no SWI.

### VALDO challenge
Cerebral microbleed segmentation on SWI, $\sim 70$ subjects. No T1c. Used as auxiliary corpus for CMB-aware mask refinement.

### TopCoW (Circle of Willis, 2024)
TOF-MRA + CTA, vessel labels. Used only as pretraining for vessel encoders (arterial-focused, not directly applicable to SWI veins).

### FOMO-60K
Healthy-control multimodal MRI corpus. Useful for shortcut-learning diagnostics (§6.5 of proposal).

---

## 10. How we are different — comparison table

| | Generator | Inputs | Conditioning | Vessel-aware eval | SWI input | Latent space | Code avail. | Bench? |
|---|---|---|---|---|---|---|---|---|
| Kleesiek 2019 | Bayesian U-Net | T1, T2, FLAIR, DWI, **SWI** | None | No | **Yes** | No | No | No (re-impl. cost) |
| Preetha 2021 | dCNN | T1, T2, FLAIR (±ADC) | None | No | No | No | No | No |
| Osman 2023 | DD-Res U-Net | T1, T2, FLAIR | Tumour-region loss | No | No | No | Partial | No |
| **pGAN 2019** | Cond. GAN | any-to-any | Cycle / paired | No | No | No | **Yes** | **Yes (GAN tier)** |
| **ResViT 2022** | CNN+ViT GAN | any-to-any | Paired | No | No | No | **Yes** | **Yes (transformer tier)** |
| **SynDiff 2023** | Adv. diffusion | any-to-any | Paired | No | No | No | **Yes** | **Yes (diffusion tier)** |
| McCaD 2024 | Cond. diffusion | T1, T2, FLAIR | Multi-seq. consistency | No | No | Some | TBD | Optional |
| Piening 2024 | Cond. diffusion + FM | T1, T2, FLAIR | None | No | No | Yes | TBD | Optional |
| CFM 2025 | FM | T1, T2, FLAIR | Aux. segmentation | No | No | Yes | **No (closed)** | No (cite only) |
| PMRF 2025 | Posterior-mean + RF | T1, T2, FLAIR | None | Qualitative claim only | No | No (pixel patch) | TBD | Optional |
| **3D-DiT** (added 2026-06-08) | Latent diffusion (transformer) | T1, T2, FLAIR | Adapted from Peebles & Xie 2023 / Mo et al. 2023; **3D latent variant per Eidex et al. 2025 baseline** | No | No | Yes (MAISI-family) | **Yes** (multiple ports; T1C-RFlow repo ships `dit3d.py`) | **Yes (transformer-DM tier)** |
| **3D-Latent-DDPM** (added 2026-06-17) | Latent conditional DDPM (U-Net) | T1, T2, FLAIR | **3D latent variant of Ho et al. 2020 DDPM, as released in Eidex et al. 2025's benchmark suite (`train_ddpm.py`)**; same MAISI U-Net trunk as T1C-RFlow, only the noising schedule differs (DDPM vs RF) | No | No | Yes (MAISI-family) | **Yes** (T1C-RFlow repo) | **Yes (pure-DDPM tier; isolates FM-vs-DDPM axis at zero architectural delta)** |
| **3D-Latent-Pix2Pix** (added 2026-06-17) | Latent conditional GAN (3D U-Net generator + 3D PatchGAN) | T1, T2, FLAIR | **3D latent variant of Isola et al. 2017 conditional Pix2Pix, as released in Eidex et al. 2025's benchmark suite (`train_pix2pix_*.py`)**; paired L1 + 3D adversarial loss on MAISI latents | No | No | Yes (MAISI-family) | **Yes** (T1C-RFlow repo) | **Yes (3D-GAN tier)** |
| **T1C-RFlow 2025** | Latent RF (U-Net) | T1, T2, FLAIR | None | No | No | Yes (custom VAE) | **Yes** (`zacheidex/An-Efficient-3D-Latent-Diffusion-Model-...`) | **Yes (current SOTA)** |
| TumorFlow 2026 | Latent RF (MAISI) | T1, T2, FLAIR + biophysical tumour-conc. field | ControlNet on PDE-derived field | No | No | Yes (MAISI) | Yes (GitHub) | **No — precedent only** (task scope ≠ paired translation; cited as architectural ancestor of VENA's MAISI+ControlNet+RF scaffold) |
| **VENA (ours)** | **Latent RF (MAISI)** | **T1, T2, FLAIR** | **ControlNet on $M_{\text{WT}}$ + $L^p$-aware contrastive (loss-side vessel fidelity)** | **Yes (multi-operator ensemble, reader)** | **No** | **Yes (MAISI VAE)** | — | self |

Notes on the "Bench?" column:
- **Yes (mandatory tier)** = reviewers at MedIA / IEEE-TMI / MICCAI will expect this row in our results table; we re-train under our cohort split for a controlled comparison.
- **Optional** = include only if compute and competitor-code maturity allow; otherwise cite as prior art and explain the omission.
- **No (cite only)** = method targets the right task but is not benchmarkable (closed code/weights — CFM) or targets a different task altogether (TumorFlow's longitudinal physics-guided growth synthesis ≠ our paired CE translation; the architecture is the precedent, not the task).
- **No (re-impl. cost)** = no public code, design pre-dates current benchmarks (Kleesiek 2019, Preetha 2021); re-implementing is its own paper.

The differentiator is now **MAISI-pretrained latent FM + $L^p$-aware contrastive on $\{T_1, T_2, \text{FLAIR}\}$** against the strongest entries of each prior era. The vessel-fidelity claim is purely loss-side (the $p_b = 3$ background term upweights non-tumour errors); the validation protocol therefore tests it through a *label-free spatial residual analysis* — Spearman ρ + top-q% bright-voxel error mass concentration + intensity-stratified residual plot, on the real-T1c intensity inside brain ∖ dilated WT (see `validation_proposal.md` §4.3 and Appendix C for the rationale for dropping vessel-segmentation operators). SWAN is **not** an input in this formulation of the problem — the previous SWAN-conditioned variant is parked as a follow-up paper once HRUM Tier 3 lands.

---

## 11b. Benchmarking selection for VENA's MedIA / MICCAI submission (2026-06-08)

*(Numbered §11b so it follows the existing §11 "Adjacent applications" thematically; the formal bibliography section remains the final one.)*

Reviewers for a high-impact venue (MedIA, IEEE TMI, MICCAI main conference) expect a competitor matrix that (i) covers each generative *era*, (ii) includes the standard cross-contrast MR-synthesis suite, and (iii) includes the current task-specific SOTA. Selection below is constrained by code availability, re-training feasibility on our 1,224-patient training pool, and the wall-clock budget on Picasso.

**Mandatory tier (seven competitors, each re-trained on our cohort split for a controlled head-to-head):**

The tier is organised so the 2D triad (C1–C3) covers the icon-lab cross-contrast MR-synthesis suite reviewers expect, and the 3D-latent quartet (C4–C7) reproduces in full **the four-method 3D benchmark that T1C-RFlow themselves published** (Eidex et al. 2025 §4 — DiT-3D, DDPM, Pix2Pix, RFlow). The 3D quartet shares the same MAISI-family VAE and the same MAISI U-Net trunk class wherever the method has a trunk to share; the only inter-method variable inside the 3D quartet is the **generative formulation** (transformer-diffusion / pure DDPM / conditional GAN / rectified flow). This is the strongest possible isolation of "is the gap due to the generative formulation, or due to backbone / VAE / latent space choices?" — every confound except the formulation is held fixed.

| # | Method | Era | Why this one | Code | Training note |
|---|---|---|---|---|---|
| 1 | **pGAN** (Dar et al. 2019) | GAN | Most-cited cross-contrast MR translation baseline; reviewers will flag its absence | https://github.com/icon-lab/pGAN-cGAN | TF-based; we re-train as `pGAN-T1c` on our 1,224 patients, 2D slice-wise (per author convention) |
| 2 | **ResViT** (Dalmaz et al. 2022) | CNN+ViT | Strong supervised generator; bridges GAN→transformer; same icon-lab benchmark suite | https://github.com/icon-lab/ResViT | PyTorch; 2D slice-wise; ports cleanly |
| 3 | **SynDiff** (Özbey et al. 2023) | Diffusion (cross-contrast) | Diffusion-era anchor in the general cross-contrast MR family; paired mode trains on our split | https://github.com/icon-lab/SynDiff | PyTorch; 2D slice-wise; paired mode only |
| 4 | **3D-DiT** (Peebles & Xie 2023 backbone; **3D latent adaptation per Eidex et al. 2025**, ships in T1C-RFlow's repo as `dit3d.py / dit3d_wrapper.py`) | Latent diffusion, transformer backbone | Tests the *transformer-backbone diffusion* axis that T1C-RFlow's U-Net does not; T1C-RFlow themselves use DiT-3D as their published baseline so reviewers expect it. Replaces CFM (no public code) in the FM/diffusion tier | T1C-RFlow repo (`zacheidex/...`); compatible 3D ports also at `facebookresearch/DiT`-derived medical-diffusion repos | Trained over MAISI latents to control for the latent-space confound vs VENA (same VAE, same latent grid, same conditioning interface) |
| 5 | **T1C-RFlow** (Eidex et al. 2025) | RF (current SOTA on BraTS) | Numerical SOTA on the same task; same <10 s/volume budget; head-to-head defines the SOTA claim | https://github.com/zacheidex/An-Efficient-3D-Latent-Diffusion-Model-for-T1-contrast-Enhanced-MRI-Generation (arXiv:2509.24194) | 3D latent RF with MAISI U-Net trunk + custom retrained VAE; expensive but mandatory |
| 6 | **3D-Latent-DDPM** (Ho et al. 2020 DDPM formulation; **3D latent adaptation per Eidex et al. 2025**, ships in T1C-RFlow's repo as `train_ddpm.py`) | Pure conditional DDPM (no adversarial term, no FM straight-line interpolant) | **Cleanest possible FM-vs-DDPM isolation**: same MAISI U-Net trunk as C5, same MAISI-family VAE as C4 / C5 / C7, same conditional concatenation `[z_t ‖ z_T1pre ‖ z_FLAIR]`, same 3D latent grid — **only the noising schedule differs** (variance-preserving DDPM forward + discrete-timestep reverse, vs C5's rectified-flow straight-line interpolant). Replaces the previous draft's excluded "Pix2Pix-DDPM / Palette-Med" (which would have been 2D and would not have shared the trunk). The 3D conditional DDPM axis is part of T1C-RFlow's *own* benchmark comparison, so reviewers expect this row | T1C-RFlow repo (`train_ddpm.py`); integration scaffolding reused from our C5 port (`src/vena/competitors/t1c_rflow/`) | 3D latent conditional DDPM; ~2–3 days A100 wall-clock |
| 7 | **3D-Latent-Pix2Pix** (Isola et al. 2017 conditional GAN formulation; **3D latent adaptation per Eidex et al. 2025**, ships in T1C-RFlow's repo as `train_pix2pix_*.py`) | Conditional GAN (3D U-Net generator + 3D PatchGAN discriminator) | Fills the **3D-GAN tier** that HA-GAN (Sun et al. 2022, designed for *unconditional* high-resolution 3D MR generation, not paired translation) cannot occupy. Tests whether the rectified-flow / diffusion families are necessary at all, or whether a strong supervised 3D conditional GAN over MAISI latents matches them at far lower inference cost. Like C6, this is part of T1C-RFlow's *own* benchmark suite | T1C-RFlow repo (`train_pix2pix_*.py`); integration scaffolding reused from our C5 port | 3D latent conditional Pix2Pix (paired L1 + adversarial loss on latents); ~2 days A100 wall-clock |

**Internal controlled comparator (not external, but reported in the headline table as the ablation that isolates the $L^p$-contrastive contribution):**

- **VENA-S1** — our own `picasso_s1_1000ep.yaml` checkpoint: identical MAISI VAE + NV-Generate-MR trunk + ControlNet, trained with $\mathcal{L}_{\text{CFM}}$ only (no contrastive). This replaces the role that TumorFlow used to play in the previous draft of the protocol: a *strictly tighter* controlled comparator (no architectural delta — only the loss differs), so any S2-vs-S1 gap is attributable to the contrastive term.

**Optional tier (include if compute budget allows):**

- McCaD 2024 — multi-sequence consistent diffusion; orthogonal to our contribution.
- Piening 2024 — UQ-via-posterior-sampling; orthogonal contribution.
- PMRF 2025 — useful for the "posterior-mean vs distributional" debate; their qualitative vessel-fidelity claim becomes a falsifiable quantitative test under our M3/M4 metrics.
- **2D Palette-Med / Pix2Pix-DDPM** — *previously considered and excluded* on the grounds that SynDiff (C3) is already a strict superset of paired conditional diffusion (adversarial-bridge diffusion = conditional DDPM + adversarial loss) in 2D. **Superseded by the 2026-06-17 update**: the 3D-latent versions of pure conditional DDPM (C6) and pure conditional Pix2Pix (C7) — both released alongside T1C-RFlow as 3D latent baselines — are now in the mandatory tier. The 2D Palette-Med variant remains out of scope (no 2D-only DDPM adds architectural diversity beyond C3 / C6).

**Explicit non-baselines (cited in related work, not benchmarked):**

- **CFM** (Chang et al., MICCAI 2025) — **dropped from the mandatory tier** as of 2026-06-08: no public code or model weights, and the MICCAI proceedings entry does not include a reference implementation. Cited as the first FM paper for the exact task, but not benchmarkable without a clean-room re-implementation that would itself constitute a separate paper.
- **TumorFlow** (Biller et al. 2026) — **dropped from the mandatory tier** as of 2026-06-08: the published task is *longitudinal physics-guided glioblastoma growth synthesis* (Fisher-Kolmogorov-conditioned latent RF), not paired CE translation. Re-purposing it by swapping the biophysical-field conditioning for our T1pre/T2/FLAIR latents would amount to "re-implementing VENA-S1 from someone else's MAISI integration code", which is dominated by simply reporting our own VENA-S1 internal comparator. Cited as the architectural precedent that established the MAISI-VAE-+-ControlNet-+-RFlow scaffold we adopt.
- Kleesiek 2019 — no code; SWI input cannot be matched (SWI is absent from our 6 cv cohorts, and the current VENA formulation deliberately does not use SWI).
- Preetha 2021 — no code; closed clinical-trial cohort.
- Osman 2023 — incomplete code; 2D DD-Res U-Net superseded by ResViT / pGAN as the CNN/GAN baseline.
- Solak 2025 — GAN era; superseded by pGAN as the GAN anchor.
- TLP 2025 — requires radiologist prompts; our protocol is automatic.

**Why this set is sufficient for MedIA / MICCAI:** (i) one representative per generative era × dimensionality cell (2D-GAN, 2D-ViT-GAN, 2D-adversarial-diffusion, 3D-DiT-diffusion, 3D-pure-DDPM, 3D-GAN, 3D-RF); (ii) the icon-lab triad pGAN / ResViT / SynDiff is the *standard* cross-contrast MR benchmark suite — reviewers expect at least one of the three, we include all three; (iii) **the 3D-latent quartet C4–C7 is exactly the four-method 3D benchmark T1C-RFlow themselves published** (Eidex et al. 2025 §4: DiT-3D, DDPM, Pix2Pix, RFlow) — reproducing it in full makes the SOTA comparison head-to-head and removes the "your 3D comparison is thin" reviewer objection; (iv) inside the 3D quartet, the MAISI U-Net trunk class and MAISI-family VAE are held fixed, so **the only inter-method confound is the generative formulation** (transformer-diffusion / DDPM / GAN / RF) — strongest possible isolation of the formulation effect; (v) VENA-S1 isolates the $L^p$-contrastive contribution under a no-architectural-delta ablation against C5; (vi) all seven external competitors fit a ~2.5-week Picasso budget at 1,224 patients (C1: 1 d, C2: 2 d, C3: 3 d, C4: 4 d, C5: 4 d, C6: 2–3 d, C7: 2 d on 4× A100 each), with C6 and C7 sharing the C5 integration scaffolding so the *porting* cost (not GPU cost) is sub-linear in the number of 3D-latent rows.

The exact training protocol, statistical analysis, and per-cohort test partitions are specified in `validation_proposal.md` (companion document, 2026-06-08).

---

## 11. Adjacent applications (blind spots flagged in earlier discussion)

These are not within our current scope but are flagged as natural extensions:

1. **Brain metastases (UCSF-BMSR).** Small multifocal lesions at the boundary of detectability — the regime where vascular context matters most.
2. **Multiple sclerosis (central vein sign).** Maggi et al. 2015 establishes that Gd modulates SWI vein conspicuity in MS plaques. A SWAN-conditioned generator is a natural fit; reference protocol Sati et al. (*Nat Rev Neurol* 2016, DOI: 10.1038/nrneurol.2016.166).
3. **Cerebral small vessel disease (cSVD), CADASIL.** Different cohort, same architecture. Higher clinical impact for young patients with cumulative Gd exposure.
4. **Vessel-wall imaging (HR-VWI).** Post-Gd vessel wall enhancement as a diagnostic target (vasculitis, aneurysm-wall instability).
5. **Paediatric oncology.** Cumulative Gd dose argument is strongest here; FDA/EMA regulatory pressure is correspondingly higher.
6. **Inverse direction as physics validation.** Train a paired model to predict SWAN from T1pre + T1c; cycle-consistency between forward and inverse models is a stronger realism test than FID/SSIM alone.

---

## Bibliography (formal)

Calabrese, E., Villanueva-Meyer, J. E., et al. (2022). The University of California San Francisco preoperative diffuse glioma MRI dataset. *Radiology: Artificial Intelligence* 4(6):e220058. DOI: 10.1148/ryai.220058.

Chang, et al. (2025). Controllable Flow Matching for 3D Contrast-Enhanced Brain MRI Synthesis from Non-contrast Scans. *MICCAI 2025*. DOI: 10.1007/978-3-032-05325-1_12.

Brandstötter, M., & Kobler, E. (2025). Synthesizing Accurate and Realistic T1-weighted Contrast-Enhanced MR Images using Posterior-Mean Rectified Flow. *MICCAI-SASHIMI 2025*. arXiv:2508.12640.

Biller, V., Bubeck, N., Zimmer, L., et al. (2026). TumorFlow: Physics-Guided Longitudinal MRI Synthesis of Glioblastoma Growth. arXiv:2603.04058.

Brina, D., et al. (2022). Vessel-CAPTCHA: A weakly-supervised deep learning approach for brain vessel segmentation. *Medical Image Analysis* 75:102263. DOI: 10.1016/j.media.2021.102263.

Dayarathna, S., et al. (2024). McCaD: Multi-Sequence Consistent Diffusion for Contrast-Enhanced MRI Synthesis.

Eidex, Z., et al. (2025). T1C-RFlow: 3D Latent Rectified Flow for Brain T1-contrast MR Synthesis. arXiv:2509.24194. *Biomedical Physics & Engineering Express*. DOI: 10.1088/2057-1976/ae3e96.

Frangi, A. F., Niessen, W. J., Vincken, K. L., & Viergever, M. A. (1998). Multiscale vessel enhancement filtering. *MICCAI 1998*, pp. 130–137. DOI: 10.1007/BFb0056195.

Geirhos, R., et al. (2020). Shortcut Learning in Deep Neural Networks. *Nature Machine Intelligence* 2:665–673. DOI: 10.1038/s42256-020-00257-z.

Genc, A. C., et al. (2023). DeepSWI: Synthesizing SWI from Conventional MRI Sequences with Deep Learning. *Journal of Magnetic Resonance Imaging*. DOI: 10.1002/jmri.28622.

Gulani, V., Calamante, F., Shellock, F. G., Kanal, E., & Reeder, S. B. (2017). Gadolinium deposition in the brain: summary of evidence and recommendations. *The Lancet Neurology* 16(7):564–570.

Guo, P., et al. (2025). MAISI: Medical AI for Synthetic Imaging. *Project MONAI.*

Hassine, R. B., et al. (2024). SHIVA-CMB: a deep-learning-based robust cerebral microbleed segmentation tool. *Scientific Reports*. DOI: 10.1038/s41598-024-81870-5.

Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. *NeurIPS 2020*. arXiv:2006.11239. *(Cited as the foundational DDPM formulation underlying C6's 3D-Latent-DDPM adaptation; the 3D-latent variant ships in Eidex et al. 2025's repo as `train_ddpm.py` and reuses the MAISI U-Net trunk of T1C-RFlow.)*

Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods* 18:203–211. DOI: 10.1038/s41592-020-01008-z.

Isola, P., Zhu, J.-Y., Zhou, T., & Efros, A. A. (2017). Image-to-Image Translation with Conditional Adversarial Networks (Pix2Pix). *CVPR 2017*. DOI: 10.1109/CVPR.2017.632. *(Cited as the foundational conditional-GAN formulation underlying C7's 3D-Latent-Pix2Pix adaptation; the 3D-latent variant ships in Eidex et al. 2025's repo as `train_pix2pix_*.py` and operates on the same MAISI latents as C4 / C5 / C6.)*

Jayachandran Preetha, C., et al. (2021). Deep-learning-based synthesis of post-contrast T1-weighted MRI for tumour response assessment in neuro-oncology: a feasibility study. *The Lancet Digital Health* 3(12):e784–e794. DOI: 10.1016/S2589-7500(21)00205-3.

Jerman, T., Pernuš, F., Likar, B., & Špiclin, Ž. (2016). Enhancement of Vascular Structures in 3D and 2D Angiographic Images. *IEEE Transactions on Medical Imaging* 35(9):2107–2118. DOI: 10.1109/TMI.2016.2515603.

Kleesiek, J., et al. (2019). Can virtual contrast enhancement in brain MRI replace gadolinium? A feasibility study. *Investigative Radiology* 54(10):653–660. DOI: 10.1097/RLI.0000000000000583.

Li, et al. (2025). Transformer with Localization Prompts for T1-Contrast Synthesis. arXiv:2503.01265.

Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023). Flow Matching for Generative Modeling. *ICLR 2023*. arXiv:2210.02747.

Liu, X., Gong, C., & Liu, Q. (2023). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. *ICLR 2023*.

Livne, M., et al. (2019). A U-Net Deep Learning Framework for High Performance Vessel Segmentation in Patients with Cerebrovascular Disease. *Frontiers in Neuroscience* 13:97. DOI: 10.3389/fnins.2019.00097.

Maggi, P., et al. (2015). The effect of gadolinium-based contrast agents on SWI brain venous structures. DOI: 10.1177/2047981614560938.

Mallio, C. A., et al. (2023). Artificial intelligence and contrast-enhanced MRI: a systematic review. *Frontiers in Neuroimaging* 2:1055463. DOI: 10.3389/fnimg.2023.1055463.

Morrison, M. A., et al. (2021). Automated detection of cerebral microbleeds on T2*-weighted MRI. *Scientific Reports* 11:4404. DOI: 10.1038/s41598-021-83607-0.

Nyúl, L. G., Udupa, J. K., & Zhang, X. (2000). New variants of a method of MRI scale standardization. *IEEE Transactions on Medical Imaging* 19(2):143–150.

Osman, A. F. I., et al. (2023). Deep-learning-based synthesis of post-contrast T1-weighted brain MR images using a DD-Res U-Net 3D model. *Journal of Applied Clinical Medical Physics* 25(2):e14120. DOI: 10.1002/acm2.14120.

Osuala, R., et al. (2024). Towards Learning Contrast Kinetics with Multi-Condition Latent Diffusion Models. arXiv:2403.13890.

Piening, S., et al. (2024). Conditional Generative Models for Contrast-Enhanced Synthesis of T1w and T1 Maps in Brain MRI. arXiv:2410.08894.

Sati, P., et al. (2016). The central vein sign and its clinical evaluation for the diagnosis of multiple sclerosis. *Nature Reviews Neurology* 12:714–722. DOI: 10.1038/nrneurol.2016.166.

Shit, S., et al. (2021). clDice — a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation. *CVPR 2021*. DOI: 10.1109/CVPR46437.2021.01629.

Solak, A., et al. (2025). Generative adversarial network-based virtual contrast enhancement in brain MRI. *Academic Radiology*. PMID: 39694785.

Tetteh, G., et al. (2020). DeepVesselNet: Vessel Segmentation, Centerline Prediction, and Bifurcation Detection in 3-D Angiographic Volumes. *Frontiers in Neuroscience* 14:592352. DOI: 10.3389/fnins.2020.592352.

Wamelink, I. J. H. G., et al. (2024). Brain Tumor Imaging without Gadolinium-Based Contrast Agents: Feasible or Fantasy? *Academic Radiology*.

Ward, P. G. D., et al. (2022). Shearlet-based vesselness for SWI/QSM. *NeuroImage* 261:119062. DOI: 10.1016/j.neuroimage.2022.119062.

Zhang, L., Rao, A., & Agrawala, M. (2023). Adding Conditional Control to Text-to-Image Diffusion Models. *ICCV 2023*. DOI: 10.1109/ICCV51070.2023.00355.

Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. (2018). The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. *CVPR 2018*.
