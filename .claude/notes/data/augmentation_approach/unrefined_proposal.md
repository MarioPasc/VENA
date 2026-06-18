Let me ground this in the project's own conventions first, then search the literature.Now searching the literature on augmentation for 3D MRI synthesis and cross-scanner/field-strength generalization.## Restating the question concretely

You are training **VENA** — a ControlNet over a frozen MAISI VAE, optimized with rectified flow — to synthesize contrast-enhanced T1 (`t1c`) from three non-contrast inputs (`t1pre`, `t2`, `flair`) plus a whole-tumor mask, on **2,093 deduplicated patients** (of which the CV-eligible training pool is ~1,687 patients / up to ~2,800 sessions once the longitudinal BraTS-GLI and LUMIERE scans are counted; the `test_only` BraTS-Africa and BraTS-PED cohorts are held out). Latents are cached offline at `(C=4, H=48, W=56, D=48)` fp16 (~1.0 MB/modality), and the training Dataset streams latents without ever calling the VAE.

You will **not** do online VAE encoding. Your plan is a two-tier augmentation scheme:

1. **Online, in latent space (free):** left–right flip and integer-voxel translation, applied directly to cached latents each training step.
2. **Offline, pre-encoded bank:** image-space augmentations applied once, then encoded to latents and cached, so train-time encode cost stays zero.

You want me to (a) find the literature on which augmentations most help **3D MRI synthesis**, biased toward your dominant 3 T→1.5 T OOD shift; (b) name a library to apply them rather than hand-code; (c) recommend **how many offline variants `K` per subject** and **the per-transform probabilities**; all under the constraint that the architecture is a LoRA (r=8/16) adaptation, not a full fine-tune.

I will separate facts (cited) from conjectures and falsifiable hypotheses.

---

## 1. What the literature says actually helps

### The governing paradigm: domain randomization, and "strong is better"

The single most load-bearing result for your OOD goal comes from the SynthSeg line. SynthSeg adopts a domain-randomization strategy in which all generative parameters — orientation, contrast, resolution, artefacts — are fully randomized, exposing the network to vastly different examples each minibatch and forcing it to learn domain-independent features [fact, Billot et al. 2023, *Medical Image Analysis* 86:102789, DOI 10.1016/j.media.2023.102789]. Critically, the SynthSeg authors note that the strongest augmentation strategies often go beyond realistic shapes and intensity distributions, agreeing with the broader empirical finding that very strong augmentation is beneficial [fact]. The concrete intensity pipeline they randomize is instructive: a random bias field, noise injection, intensity rescaling, and a random voxel-wise exponentiation (gamma), with low-resolution and partial-volume effects modeled by Gaussian blurring and random subsampling [fact]. This is the canonical "which intensity operations matter for cross-scanner / cross-field generalization" answer: **bias field, gamma/exponentiation, noise, and resolution (blur+subsample)** — exactly the operations that are *not* latent-valid, which is why they must live in your offline bank.

### The directly analogous study: field-strength translation under data scarcity

The closest published analog to your task — a **3D, multi-contrast, field-strength image-to-image translation under a small paired corpus** — is the ULF-EnC challenge entry "Augment to Augment" (Zimmermann 2025, arXiv:2511.09366). The setting is 64 mT→3 T translation from T1/T2/FLAIR inputs, 50 paired volumes. Two findings transfer almost verbatim to VENA:

- The augmentation **split mirrors your input-only / joint-geometric requirement exactly**: geometric augmentations (affine — rotations, shifts, shearing — left-right flips, and moderate 3D non-rigid transforms) are applied consistently across the input contrasts and the target volume, while intensity and degradation augmentations are applied to the input volumes only [fact]. This independently corroborates the constraint from your previous turn: geometry is matched input↔target; photometric perturbation touches inputs only, leaving the `t1c` supervision clean.
- The specific photometric operations and their **probabilities**: a random monotonic intensity mapping is applied to each input volume in 20% of training samples, and in a further 20% the inputs are degraded by independent Gaussian noise (random small std) plus anisotropic blurring with randomized per-dimension kernel strength [fact]. Their ablation showed the diverse augmentation strategy gave a substantial fidelity gain, with "only affine" augmentation an intermediate and "no augmentation" the worst [fact].

Note one operation here that is *not* in the standard TorchIO/MONAI menu and that you should add deliberately: the **random monotonic intensity remapping** (piecewise-linear transfer function through randomly sampled support points). I flag why below — it is a partial fix for a blind spot.

### Corroborating evidence from MR-to-CT and segmentation

A multi-center 3D cGAN MR-to-CT study spanning 3 T and 0.35 T scanners reports that a preprocessing pipeline combined with elastic-transformation-based augmentation reduced input variability and improved both qualitative and quantitative cross-center performance [fact, Boydev et al. 2025, *Pattern Recognition Letters*, DOI 10.1016/j.patrec.2025.01.002]. And nnU-Net — the strongest "what reliably works" prior in 3D medical imaging — fixes its augmentation set to random rotations (±30°), scaling (0.7–1.4), elastic deformations, gamma/intensity perturbations, axis-wise mirroring, brightness/contrast shifts, and additive Gaussian noise applied on-the-fly [fact, Isensee et al. 2021, *Nature Methods* 18:203–211, DOI 10.1038/s41592-020-01008-z], plus simulation of low resolution [fact, index 34-1].

**Synthesis of the evidence.** The augmentations with literature support for *your specific OOD axis* are, in priority order: **bias field** (the dominant low-frequency scanner/field nuisance), **gamma + brightness/contrast and a monotonic intensity remap** (field-dependent contrast change), **Gaussian noise** (SNR drop at lower field), **blur/anisotropy + low-resolution simulation** (resolution loss), and **elastic + small-angle affine** (the high-value anatomical augmentation). The dihedral set (flip, 90° rotation, translation) is well-supported but low-value for OOD — which is precisely why it belongs in the cheap online tier.

---

## 2. Library: TorchIO (primary), MONAI (already-in-stack fallback)

Do not hand-code these. Two mature options:

**TorchIO** (Pérez-García et al. 2021, *Computer Methods and Programs in Biomedicine* 208:106236, DOI 10.1016/j.cmpb.2021.106236; arXiv:2003.04696) is purpose-built for MRI and is my primary recommendation for the offline bank. It provides domain-specific transforms including simulation of magnetic-field-inhomogeneity (bias field) and k-space motion artifacts, alongside random affine, elastic deformation, flip, noise, and blur [fact]. The decisive feature for you is the `Subject` abstraction: register `t1pre`/`t2`/`flair`/`t1c` as `ScalarImage`s and the WT mask as a `LabelMap`, and **every spatial transform is applied consistently to all members of the Subject** (with nearest-neighbor for the LabelMap), while intensity transforms are restricted to the input images via `include=`/`exclude=`. This makes the input-only / joint-geometric split a configuration detail rather than custom bookkeeping. The relevant classes: `RandomBiasField`, `RandomGamma`, `RandomNoise`, `RandomBlur`, `RandomAnisotropy`, `RandomMotion`, `RandomGhosting`, `RandomSpike`, `RandomAffine`, `RandomElasticDeformation`, composed with `Compose`/`OneOf`.

**MONAI** is the equivalent dictionary-transform API (`RandBiasFieldd`, `RandAdjustContrastd` = gamma, `RandGaussianNoised`, `RandGaussianSmoothd`, `RandGibbsNoised`, `RandKSpaceSpikeNoised`, `RandHistogramShiftd` ≈ monotonic remap, `Rand3DElasticd`, `RandAffined`). It is already in your dependency tree (the VAE and ControlNet are MONAI), and dictionary keys give the same input-only/spatial-consistent control.

**Recommendation:** use TorchIO in the offline bank-builder (a standalone script analogous to `encode_latents.py`), because its bias-field and motion models are physically grounded and the Subject-level spatial consistency removes a whole class of mask-desync bugs [conjecture — both libraries are correct; TorchIO simply makes the joint-geometric constraint harder to get wrong]. `RandHistogramShiftd` from MONAI is the cleanest way to get the Augment-to-Augment monotonic remap if you prefer not to implement the 4-point piecewise map yourself.

---

## 3. How many variants `K`, and the probabilities

### Conceptual point: an offline bank is *finite* domain randomization

Online augmentation draws a fresh nuisance per step (effectively infinite variety). An offline bank stores `K` **fixed** realizations, so the model can in principle memorize those `K` nuisance patterns. Three things make a small `K` nonetheless sufficient here:

1. **Your online latent tier multiplies the diversity.** Each bank variant is further perturbed online by flip (`p=0.5` → ×~2) and translation, so effective diversity ≈ `K` × (online geometric set), not `K`.
2. **LoRA's small effective dimension limits ID overfitting.** At r=8/16 the adaptation capacity is ~10⁶–10⁷ params; the ID-overfitting pressure that augmentation usually fights is already low, so augmentation's job here is almost purely OOD coverage — and OOD coverage of a smooth nuisance (bias field, gamma) saturates quickly in `K` [conjecture, grounded in the LoRA/d_eff argument from your prior turn].
3. **Variance argument.** For a continuous transform with bounded parameter range, a stratified finite sample of `K≈4–8` captures most of the parameter-space variance; the marginal benefit of variant `K+1` decays roughly as `1/K` [conjecture — *check by* training at `K∈{0,2,4,6}` and plotting OOD SSIM/PSNR on the BraTS-Africa/RHUH proxy; the curve should plateau by `K≈4–6`].

### Recommended scheme

Augment **all** CV-training subjects (not a subset — the per-subject benefit is uniform). I recommend a **stratified `K=4` bank** (each variant exercises a distinct nuisance, so no draw is wasted on a near-no-op, which is the failure mode of low-probability probabilistic composition at small `K`), retaining the clean original as `v0`. Push to `K=6` only if the OOD plateau check above has not flattened.

| Variant | Transforms (TorchIO) | Applies to | Per-transform `p` *within the variant* | Rationale |
|---|---|---|---|---|
| `v0` clean | — | — | — | already cached; keeps true distribution present |
| `v1` field/scanner | `RandomBiasField(coeff order 3)` **+** `RandomGamma(log_gamma∈[−0.3,0.3])` (p=0.4) | inputs only | bias `p=1.0`, gamma `p=0.4` | dominant 3 T→1.5 T low-frequency nuisance |
| `v2` contrast | `RandHistogramShift`/monotonic remap **+** brightness/contrast | inputs only | remap `p=1.0`, bright/contr `p=0.5` | field-dependent contrast change (the blind spot below) |
| `v3` SNR/resolution | `RandomNoise(std∈[0,0.05])` **+** `RandomAnisotropy`/`RandomBlur` | inputs only | noise `p=1.0`, blur `p=0.7` | low-field SNR + resolution loss |
| `v4` anatomy | `RandomElasticDeformation(light)` **+** `RandomAffine(rot ±10°, scale 0.9–1.1, shear)` | inputs **and** target (+ mask) | elastic `p=0.7`, affine `p=0.7` | only high-value geometric augmentation; non-latent-valid |
| (`v5`,`v6` if `K=6`) | bias+noise combo; second independent bias draw | inputs only | — | denser coverage of the dominant axis |

Optional clinical-realism transforms — `RandomMotion`, `RandomGhosting`, `RandomSpike` — at low probability (`p≈0.1` each, folded into `v3`) add robustness to acquisition artifacts but are secondary to the field-strength axis.

**Train-time sampling policy** (the "probability of augmentation" the user asked about, at the *sample* level): draw `v0` with probability ≈ **0.3** and one of the `K` augmented variants uniformly with probability ≈ **0.7**, then always apply the online latent flip (`p=0.5`) and translation (`p=0.5`) on top. Keeping `v0` at 0.3 matters because inference inputs are clean preprocessed volumes — the model must remain calibrated on the unaugmented distribution, not only the randomized one [conjecture, checkable by comparing ID test PSNR at clean-fraction ∈ {0.0, 0.3, 0.5}].

**Online latent tier (free, per step):** L-R flip `p=0.5` and integer latent-voxel translation `p=0.5` (a 1-voxel latent shift = 4 image voxels at 4× compression). These survive because a convolutional VAE is approximately equivariant to the dihedral/translation group [fact]; verify the decode-equivariance residual once as you noted previously.

### Storage

Per-modality latent ≈ 1.0 MB (cohorts.md shape) to 1.15 MB (proposal shape). Exploit that **input-only variants do not re-store the target** (reuse `v0`'s `t1c` latent): `v1`–`v3` store 3 latents each; `v4` stores 4.

$$
S_{\text{extra}} \approx \big[\underbrace{3\times 3}_{v_1,v_2,v_3} + \underbrace{4}_{v_4}\big]\times 1.05\,\text{MB} \approx 13.6\,\text{MB/subject}.
$$

At ~1,700 CV subjects → **~23 GB**; at ~2,800 sessions → **~38 GB**. `K=6` → ~34/56 GB. lzf gives little on near-Gaussian latents, so treat as raw [fact]. Trivially feasible on research storage and consistent with your earlier ~100 GB ceiling.

---

## 4. Positioning and blind spots

**Where this sits in the field.** Domain randomization (SynthSeg) is the dominant paradigm for MRI OOD robustness, and the most recent field-strength-translation challenge work (Augment-to-Augment, ULF-EnC) shows that **strong, diverse, input-only photometric augmentation + light joint geometry** is the winning recipe under data scarcity — the same regime as VENA. Combined with your LoRA-not-FFT decision, the narrative is coherent and publishable as a deliberate "bet on OOD via augmentation + minimal adaptation" design, with the earning ablation being **{no-aug / dihedral-only / +photometric bank}** evaluated on your four shift axes. This is an internal design component of the MICCAI 2026 / *Medical Image Analysis* submission, not a standalone paper.

**Blind spots** (where the idea has gaps you may not have flagged):

1. **Bias field ≠ field-dependent contrast change.** T1/T2 relaxation times are field-dependent, so 1.5 T alters GM/WM/lesion *contrast relationships*, not just adds a smooth bias and noise. A multiplicative bias field cannot reproduce this. The **monotonic intensity remap** (`v2`) is the partial fix — it perturbs the intensity transfer function — but it is not a physical 1.5 T contrast model [fact/conjecture]. If you later want a principled version, a Bloch-equation re-synthesis from estimated tissue maps would be the rigorous route (out of scope for v1).

2. **Percentile-foreground normalization can partially undo global photometric augmentation.** Your encode-time normalization is per-modality `[0, 99.95]` foreground percentile. A *global* brightness/gamma shift is largely renormalized away, whereas a *spatial* bias field is not. This is a reason to **weight bias field over global brightness** [conjecture — *check by* encoding a globally-gamma'd input and measuring the latent-space distance to the clean encoding; if it is near zero, global gamma is being normalized out and is wasted].

3. **Mask desync.** The WT mask is part of the conditioning. Any geometric transform (`v4`) must warp the mask too, then re-downsample to latent resolution by nearest-neighbor. TorchIO's `LabelMap` handles this; a hand-rolled pipeline silently breaks it [fact].

4. **Photometric augmentation cannot close anatomy/biology gaps.** Your pediatric (BraTS-PED) and partly your geographic axes are *anatomical/biological* shifts. No amount of bias field/gamma manufactures pediatric anatomy from adult data; elastic deformation gives only mild morphological variety. Be explicit in the paper that the bank targets the **acquisition** axis (field strength, scanner), not the **biology** axis [fact].

5. **Rician vs Gaussian noise.** TorchIO's `RandomNoise` is Gaussian on the magnitude image. At 1.5 T SNR this is an acceptable approximation; it diverges from true Rician statistics only at very low field (<0.5 T) [fact]. Fine for your stated target.

If useful, I can write the offline bank-builder in your OOP/dataclass/structured-logging project style: a `LatentBankBuilder` (TorchIO `Subject` assembly → stratified variant `Compose`s → MAISI encode → per-subject HDF5 append), with the input-only/joint-geometric split enforced by `include=`, a `DecodeEquivarianceQC` check for the online latent transforms, and custom exceptions for normalization/crop-box drift against the cohort invariants.