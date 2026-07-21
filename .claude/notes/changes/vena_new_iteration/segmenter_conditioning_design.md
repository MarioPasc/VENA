# Mask injection into v3a — second-stage refinement design

> **Purpose (this iteration = iter 6).** The decision is locked: **second-stage refinement of the
> v3a checkpoint by injecting a soft `[WT, NETC]` map via ControlNet**, to add tumour enhancement
> **without losing v3a's (already-competitive) whole-brain fidelity.** This iteration answers ONE
> question deeply:
>
> > *Given a **perfect** soft `[WT,NETC]` probability map, what is the best way to inject it into the
> > pretrained v3a checkpoint to allow tumour enhancement without losing whole-brain fidelity?*
>
> Assuming a **perfect (oracle GT `[WT,NETC]`) mask** deliberately isolates the *injection mechanism*
> from *segmenter quality* — it is the **T-13 fair-oracle** vehicle. **The segmenter itself (backbone,
> training regime, calibration, data) is DEFERRED to the next iteration** (Part B keeps the locked
> facts). Companion: `model_redesign_2026-07-21.md`. Status: design only, nothing launched.
>
> **Navigation.** See **§0 — Document map** (immediately below) for the agent-facing structure guide:
> section map, locked/open decisions, task-graph anchors, and invariants.

---

## §0 — Document map (read this first — agent-facing)

**What this is.** The design spec for VENA's tumour-enhancement conditioning: how a soft `[WT,NETC]` mask
is injected into the pretrained **v3a** latent-flow checkpoint (Part A), and how the segmenter that produces
that mask is built (Part B). Authority on *scope*: `model_redesign_2026-07-21.md` (same dir) — its §14
verdict, §15 task graph (T-01…T-14), §16 segmenter pivot, §17 T-01 result. Frozen evidence: `../../article/`.

**The bet (locked).** Second-stage refinement: warm-start **v3a** (`…_s1_v3a_concat_only_fft_ef000c9f`) →
inject a soft `[WT,NETC]` prior via **ControlNet** (zero-init + scale-ramp) → add enhancement **without
losing whole-brain fidelity**. NOT: a stochastic bridge, refinement-loss-only, adversarial-primary, or
SPADE-primary. The *why* is Part C (the iter-5 reviews).

**Section structure.**
| Section | Contains | Read if you are… |
|---|---|---|
| §0 (here) | this map | orienting |
| **PART A** | injection mechanism — P1–P4 properties; Q1 *which levels*; Q2 *ControlNet vs SPADE vs FiLM vs CA*; concrete recipe (A.5); no-regression gate (A.6) | building/validating the T-06 / T-13 run |
| **B.a** | segmenter model + loss (BSF-SwinUNETR; **DML+CE, not soft-Dice**) | choosing the model/loss |
| **B.b** | training data + **K-fold OOF** splits | building the mask pipeline |
| **B.c** | soft-map derivation (4-step: SDT→sigmoid, DML, temp-scale, avg-pool; + free ensemble) | producing the conditioning map |
| **B.d** | preprocessing, augmentation, nesting, label harmonisation | implementing the segmenter |
| **B.e** | verified references | citing |
| **PART C** | why-the-mask evidence + leakage taxonomy | defending the design |
| **Change log** | iteration trail (iters 1–7) = the reasoning history | understanding *why* a decision was made |

**Locked decisions.** bet = ControlNet mask-injection refinement of v3a · fusion = **ControlNet primary /
SPADE = T-07 ablation** (FiLM & CA rejected) · mask = soft `[WT,NETC]` (WT-only = fallback) · backbone =
**BSF-SwinUNETR** (Arm A BraTS-SSL / Arm B UKB-SSL leak-free / Arm C SegResNet baseline) · loss = **DML+CE
on SDT-soft targets** (Dice = eval gate only) · splits = **K-fold OOF** (the K fold-models double as a free
calibration ensemble) · generator training masks = OOF **predicted** · intensity norm = **99.95** canonical.

**Open decisions.** **D-d:** add BraTS-MEN? (gated on the Málaga pathology mix — glioma-only vs
+meningioma). **Soft-target path:** SDT+DML from the start vs free-only (ensemble+temp+avg-pool) then
upgrade. **Latent mask shape** (48,56,48) vs (60,60,40) — `[verify]` against the generator grid.

**Task-graph anchors** (defined in `model_redesign` §15/§16): **T-06** = deployable arm (predicted mask);
**T-13** = fair oracle (GT `[WT,NETC]`) = *this iteration's Part-A validation vehicle*; **T-14** =
false-enhancement safety; gates **G-NORM / G-SEG / G-SHORTCUT**.

**Invariants to honour.** routine pattern (one YAML arg, frozen Pydantic config, `Engine.run()->Path`, no
import-time side effects) · bump `decision.json` `schema_version` on new fields · matched **99.95** norm in
every metric path · **out-of-fold** masks (leakage) · the whole-brain **no-regression gate** (A.6) · trust
exhaustive-val PSNR_ET, not train loss.

---

# PART A — THE INJECTION MECHANISM (this iteration)

## A.0 Locked decision & the vehicle
- **Recipe skeleton.** Warm-start from **v3a `ema_best.ckpt`** (`2026-06-24_16-00-46_s1_v3a_concat_only_fft_ef000c9f`)
  → add a **fresh ControlNet** branch that takes the soft `[WT,NETC]` map → **zero-init output +
  scale-ramp** → train to add enhancement. Objective stays **L1 velocity CFM** (this iteration is about
  *where/how to inject*, not the loss).
- **Why warm-start v3a, not v3b.** v3a is the better *whole-brain* checkpoint (best of the latent tier)
  and carries **no mask machinery** — so a fresh `[WT,NETC]` ControlNet is a clean add (v3b's ControlNet
  expects the wrong `[NETC,ED,ET]` semantics). It also gives clean attribution: `v3a → v3a+mask` isolates
  the mask's contribution.
- **This iteration's mask = GT `[WT,NETC]` (oracle).** Removes the segmenter confound; = **T-13**.
  Deployable-predicted-mask swap is a later step (T-06).

## A.1 The four properties any injection must satisfy
| # | Property | Why it matters for refining v3a |
|---|---|---|
| **P1** | **Identity at init** — step 0 == v3a exactly | never damage v3a's proven whole-brain fidelity while learning the mask |
| **P2** | **Spatial locality** — influence concentrated in/near WT | enhancement is <0.1 % of voxels; the other 99.9 % must stay v3a |
| **P3** | **Multi-scale reach** — coarse + fine | coarse = "tumour here, it enhances"; fine = "enhance at this rim voxel" |
| **P4** | **Minimal risk to pretrained trunk weights** | the trunk holds the whole-brain competence — don't destabilise it |

## A.2 Q1 — At WHICH levels to inject
**Answer: all down-block resolutions + the mid-block (the standard ControlNet), residuals propagating
into the decoder via the skip connections.**

- The MAISI FM trunk is a 3D U-Net: `conv_in → down-blocks (↓res) → mid-block (min res) → up-blocks
  (↑res, skip from down) → conv_out`. A ControlNet (Zhang 2023, arXiv:2302.05543) is a trainable copy of
  the **down-blocks + mid-block**; it emits per-level feature residuals **added** to the trunk's
  down-block outputs + mid-block output, which then reach the **decoder through the skips**.
- **Why all levels (P3).** Enhancement placement needs both the coarse semantic decision (low-res
  down-blocks + mid: "this region enhances") and the fine rim localisation (high-res down-blocks → skips
  → high-res up-blocks: "these boundary voxels"). Restricting to the bottleneck alone loses the rim.
- **The level selection is partly *learned*.** Each level's ControlNet output is a **zero-init**
  projection → starts at 0 → training discovers which levels carry useful mask signal; unused levels stay
  ≈0. So we inject at all levels and let the data weight them, rather than hand-picking.

## A.3 Q2 — HOW to inject: zero-init ControlNet residual-add vs SPADE vs FiLM vs cross-attention
**Answer: ControlNet residual-addition with zero-init + scale-ramp is the PRIMARY. SPADE/adaLN-zero is
the ablation. FiLM and cross-attention are rejected for this task.**

| Mechanism | Spatial? (P2/P3) | Touches trunk weights? (P4) | Identity-at-init? (P1) | New params | Verdict |
|---|---|---|---|---|---|
| **ControlNet residual-add + zero-init + ramp** | **yes** — full feature-map residual | **no** — separate branch (trunk untouched / freezable) | **yes** — zero-conv + ramp | encoder-copy (adapter) | **PRIMARY** |
| **SPADE / adaLN-zero** | yes — spatial `γ(m),β(m)` | **yes** — modulates trunk norm in-place | yes *if* affine folded (B7) | small | **ablation (T-07)** — TA-ViT's; riskier P4 |
| **FiLM** | **no** — per-channel scalar, spatially uniform | yes — modulates norm | yes if zero-init | tiny | **REJECT** — cannot localise a spatial mask |
| **Cross-attention (CA)** | indirect | adds new CA layers | needs zero-init on CA out | medium–large | **REJECT** — see below |

**Reasoning (from the four properties):**
- **FiLM is disqualified.** FiLM (Perez 2018, arXiv:1709.07871) produces per-channel, **spatially
  uniform** `(γ,β)` — it modulates *what* not *where*. A `[WT,NETC]` mask is a spatial signal; FiLM would
  apply the same modulation everywhere and cannot say "enhance here, not there." (SPADE is literally
  "spatially-varying FiLM" — which is why SPADE, not FiLM, is the spatial generalisation.)
- **Cross-attention is the wrong tool here.** CA earns its cost when the conditioning is a *set of tokens
  whose spatial correspondence must be learned* (text→image). Our mask is **dense and already voxel-aligned
  with the latent** — voxel `(i,j,k)` of the mask *is* voxel `(i,j,k)` of the latent, so there is no
  correspondence to learn. CA adds new parameters (bad for a warm-start refinement), adds compute, spreads
  influence non-locally (hurts P2), and the MAISI trunk has no CA slots (only self-attention at low res).
- **SPADE / adaLN-zero is viable — it is TA-ViT's mechanism — but secondary.** SPADE (Park 2019,
  arXiv:1903.07291) / adaLN-zero (DiT, Peebles & Xie 2023, arXiv:2212.09748) modulate the trunk's own
  **normalisation** layers with spatial `γ(m),β(m)`, zero-init for identity. Parameter-efficient and
  spatial. But three strikes vs ControlNet for *this* task: (1) it modulates the trunk **in-place** →
  higher risk to the pretrained whole-brain features (P4); (2) the **§14.8-B7 landmine** — the trunk's
  GroupNorm has pretrained affine `(γ_pre,β_pre)≠(1,0)`; naive adaLN-zero discards it unless folded into
  the SPADE base or added post-norm; (3) it is **exactly TA-ViT** (Eidex 2025) → less differentiated.
  Keep it as the **T-07 ablation** with the affine-fold fix.
- **ControlNet residual-add + zero-init + scale-ramp is the primary — it is the only option that
  structurally guarantees P1+P4.** It is a **separate branch**: at init the trunk is untouched, and (see
  A.4) can be *frozen*, so v3a's whole-brain competence is preserved by construction. The output convs are
  **zero-init** → step-0 residual = 0 → refined model == v3a exactly (P1). It is **spatial + multi-scale**
  (P2/P3). And the mask input is spatially sparse (≈0 outside WT) → the residual is concentrated near the
  tumour → whole-brain preserved (P2). Proven: v3b (ControlNet-mask) beat v3a by **~+4 dB PSNR_ET** while
  leaving whole-brain essentially unchanged (PSNR_brain 18.35 vs 18.5). More differentiated from TA-ViT.

**On the "ramp".** Yes — the **scale-ramp** (`output_scale = sigmoid(steepness·(step/ramp_steps−0.5))`,
0→1 over ~5000 steps, the `OutputScaleRampCallback`) **is** the right tool for P1. Zero-init alone gives a
step-0 identity but a discontinuous first gradient; the ramp introduces the branch **gradually**, avoiding
the cold-start shock that would perturb v3a. Zero-init **and** ramp are a pair: zero-init sets the starting
point, the ramp controls the approach.

## A.4 Design levers that maximise "without losing whole-brain fidelity"
1. **Trunk-update policy (the biggest lever).**
   - **Primary — joint, low-LR, trunk-EMA** (v3b-proven): fine-tune trunk + ControlNet together at a low
     LR. v3b's evidence shows joint training *adds* enhancement while whole-brain stays put (PSNR_brain
     unchanged). Zero-init+ramp means we depart from v3a gently; short transport (v3a→v3a+enh) is a small
     change. Trunk-EMA + sampling from the EMA shadow (per model-coding-standards §2).
   - **Conservative ablation — freeze the trunk, train ControlNet only** (pure adapter). *Guarantees*
     the whole-brain generator is byte-unchanged; the ControlNet only adds a tumour-local correction.
     Stronger P1/P4 guarantee, possibly lower enhancement ceiling. Run it to bound the "trunk must adapt?"
     question.
2. **ControlNet input scope.** Feed the ControlNet the **mask only** (most localised residual, best P2)
   vs mask + light anatomy context (more capacity, risk of brain-wide residual). Ablation; default
   mask-only for locality.
3. **Optional locality gate.** Multiply the ControlNet residual by a **soft dilated-WT gate** →
   hard-bounds the correction to the tumour neighbourhood (a strong P2 guarantee). Use a *soft/dilated*
   gate so mask-boundary enhancement isn't clipped. Ablation — may be unnecessary if the sparse-mask
   input already yields local residuals.
4. **Whole-brain-fidelity guard in the loss (optional).** A small **anchor/consistency penalty** keeping
   the refined output ≈ v3a *outside* dilated-WT (e.g. L1 on `brain∖dilate(WT)` vs the v3a prediction)
   makes "don't touch whole-brain" an explicit objective, not just an inductive bias. Cheap insurance.
5. **Mask channel layout.** 2-ch `[WT,NETC]`, or 3-ch `[WT,NETC,zero_out]` (via `ZeroOutDownsampler`) to
   keep a byte-compatible slot for a future warm-start. Independent sigmoids, `NETC ⊆ WT`.

## A.5 Concrete recipe
```
resume_from      : v3a ema_best.ckpt (pin the Picasso path)
mask input       : GT [WT,NETC] soft, latent grid (oracle → this iter = T-13)
fusion           : MaisiControlNet, init_from_trunk (copy v3a down+mid), inject at all down+mid levels
output proj      : zero-init (zero-conv)         # P1
output_scale     : sigmoid ramp 0→1 over 5000 steps (OutputScaleRampCallback)   # P1
trunk update     : PRIMARY joint low-LR + trunk-EMA;  ABLATION freeze-trunk (adapter-only)
loss             : L1 velocity CFM  (+ optional whole-brain anchor A.4.4;  T-09 rim loss is a later stage)
rflow            : use_timestep_transform=true, base_img_size_numel=129024, plumb input_img_size_numel to sampler
EMA              : 0.9999 ; EarlyStopping patience 250
run family       : picasso_ref_v1_v3a+cn[WT,NETC]_fft.yaml  (loginexa smoke first)
decision.json    : add fusion=controlnet, mask_repr=[WT,NETC], mask_source=oracle(GT), warm_start_source=v3a,
                   trunk_update={joint_lowlr|frozen}, output_scale_ramp_steps
```

## A.6 How we know it worked (validation gate for this iteration)
On the **oracle `[WT,NETC]`** mask (T-13), at matched 99.95 norm:
- **PSNR_ET ↑ vs v3a** (target: recover most of the v3b−v3a ET gap), **AND**
- **whole-brain not-worse-than v3a** — the explicit "no-regression" gate: `MS-SSIM_brain ≥ v3a − MCID`
  and `MAE_brain ≤ v3a + MCID`. This gate is the operationalisation of "without losing whole-brain
  fidelity"; a refinement that improves ET but regresses whole-brain **fails**.
- **FP-safety**: on GT-ET≈0 cases, false-enhancement volume ≈ v3a (mask-gating must not *add* FP).
- Report per-NFE. The freeze-vs-joint and mask-only-vs-context ablations are decided on this gate.

## A.7 Q3 — Is training the segmenter on VENA's glioma training data defensible? (brief; deferred)
**Yes — with out-of-fold discipline, and it is the correct design, not leakage.** Train the segmenter on
VENA's **train-split patients only**; never let it see val/test (which you said you will respect).
Predicted masks for val/test are then **out-of-fold → clean**. For the *training* patients, generate their
masks **out-of-fold too** (k-fold *within* the train set) so this refinement sees realistic (test-quality)
masks during training and does not learn to over-trust an optimistic in-fold mask. This is the §C.2 L2
discipline; **full segmenter treatment is deferred to the next iteration.** (Note: this iteration uses the
GT oracle mask, so segmenter leakage does not even arise yet.)

---

# PART B — THE SEGMENTER (detailed: model+loss / data / soft-derivation / rest)

> The segmenter, detailed into the four axes the user scoped. It is **decoupled** from the injection
> mechanism (Part A), which is validated first on the **oracle** mask (T-13). Design only.
> `[WT,NETC]` soft, 2-ch (fallback 1-ch `[WT]` only if NETC fails G-SEG). ET is undefined pre-contrast;
> the generator resolves the enhancing subset of `WT∖NETC`.

## B.a — Model + loss (and the soft-Dice correction)
- **Model (D-a locked): SwinUNETR (feature_size=48) init from BrainSegFounder** (MONAI, Apache-2.0).
  Arm A = **BraTS-SSL** encoder (multimodal+tumour-aware; drop the T1ce slice, feed `[FLAIR,T1pre,T2]`,
  `strict=False`) — primary; Arm B = **UKB-SSL** (T1-only, leak-free) — headline if L3 purity demanded;
  Arm C = **SegResNet-from-scratch** — baseline that quantifies the pretraining gain. Bench alternatives:
  MedNeXt (stronger CNN), nnU-Net (augmentation reference).
- **Loss principle — overlap in the loss, softness in the derivation (B.c).** We do **not** optimise hard
  voxel accuracy — **Dice is the eval gate only** (G-SEG). The loss just has to place the region right,
  tolerant of boundary voxels (a Dice-family loss). But **standard soft-Dice pushes the output to
  overconfident 0/1** (Bertels 2019/2021, arXiv:1911.02278) → a near-*hard* map, the opposite of what the
  conditioner wants. So softness is produced in **B.c**, not read off the raw sigmoid.
- **⚠ Correction to "soft-Dice + …".** *If* we train against **soft targets** (recommended, B.c-1),
  **standard soft-Dice is NOT proper on soft labels** (asymmetric/ill-defined, pathological gradients —
  Wang et al., **Dice Semimetric Losses**, MICCAI 2023, arXiv:2303.16296). Use **DML (Dice Semimetric
  Loss) + CE** — DML equals soft-Dice on hard labels, well-defined on soft ones. So: **DML+CE on soft
  targets** (primary) *or* soft-Dice+CE on hard targets + post-hoc softening (simpler alt).
- **FN-weighting:** NETC is small and FN-costly → **Tversky / focal-Tversky** as the Dice term. Deep
  supervision on. **Keep the probabilistic range end-to-end** (SoftSeg, Gros MedIA 2021, arXiv:2011.09041):
  never binarise a soft mask in preprocessing/augmentation/pooling.

## B.b — Training data & splits (K-fold OOF locked)
- **Reuse the FM splits — no independent segmenter partition** (leakage). The segmenter must be OOF w.r.t.
  the FM's val/test.
- **K-fold out-of-fold (user decision, max rigor):** K fold-models (each on K−1 folds of FM-train) predict
  their held-out fold → every FM-train mask is OOF; one **all-FM-train** model predicts FM-val/test (OOF for
  them). Keep both single-model OOF so train/test mask quality is matched. **Free dividend:** the K fold-
  models **are a deep ensemble** → the calibrated ensemble mean B.c wants comes free. Cost ≈ K+1 trainings
  (~6 at K=5), cheap with BSF. Internal **val slice** per fold → early-stop + temperature calibration.
- **Cohorts:** train on the **pooled cv cohorts** (UCSF-PDGM 202, BraTS-GLI 1133, IvyGAP 34, LUMIERE 91,
  REMBRANDT 63, UPENN-GBM 164). **test_only** cohorts (Africa-Glioma/-Other, BraTS-PED) = **Ring B OOD** —
  never trained on; evaluated for the OOD gate.
- **Segmenter's own reported result:** per-cohort WT/NETC Dice + AHD + ECE/Brier on FM-val/test (OOF) +
  Ring B. **G-SEG:** WT ≥0.80, NETC ≥0.50 per cohort incl. Ring B; healthy → ~empty.

## B.c — Soft-probability derivation (verified 4-step recipe; the "free" mass distribution)
The soft `[WT,NETC]` map is **derived**, not read off the overconfident sigmoid. Four steps, each
independently beneficial, **all cheap/free**, and they **compound**:
1. **(target) SDT → sigmoid soft targets.** From each hard label compute the normalised **signed distance
   transform**, map `y_soft = sigmoid(SDT/σ)` (σ≈3 vox): 0.5 at the boundary, ~0.95 at 3σ inside, ~0.05 at
   3σ outside (Ma MIDL 2020; Kervadec boundary loss arXiv:1812.07032; **SVLS** = the Gaussian special case,
   Islam & Glocker IPMI 2021). Distributes mass by distance-to-boundary *by construction*.
2. **(loss) DML + CE on the soft targets** (B.a) — proper soft-label Dice (Wang 2023), **not** standard soft-Dice.
3. **(post-hoc) temperature scaling** — one scalar T fit on val by NLL (typ. 1.5–3.0); free, argmax-preserving,
   moves overconfident mass toward the simplex interior (Guo ICML 2017).
4. **(pool) average-pool to the latent grid** — `AvgPool3d(stride=4)` = **partial-volume integration**: a
   boundary latent voxel gets 0.25–0.75 by enclosed-lesion fraction. **Free, differentiable — the single most
   important "free" spatial-mass mechanism** ("no additional mechanism required").
- **Plus the free K-fold ensemble mean** (B.b) — further calibrates/smooths at no extra training cost.
- **Minimal free-only path** (if simplicity wins): hard-target soft-Dice+CE → **ensemble mean → temp →
  avg-pool**. Add Steps 1–2 (SDT+DML) only if the map is still too binary.
- **Tuning caveat:** Step-1 σ and Step-4 avg-pool both soften — tune σ so the *post-pool* latent map is
  graded, not mush (over-softening loses localisation).
- **Why soft > hard for the conditioner:** a continuous spatial prior carries strictly more information than
  a binary mask; a continuous tumour-density map conditions a latent-diffusion synthesiser *more precisely*
  than a binary ROI (biophysically-conditioned synthesis, **MICCAI 2025**, arXiv:2510.09365); SPADE-style
  modulation is designed for continuous signals (hard boundaries alias); seg-guided diffusion (Konz MICCAI
  2024, arXiv:2402.05210) confirms mask conditioning lifts fidelity/controllability.

## B.d — Preprocessing, augmentation, and the rest
- **Preprocessing:** `{t1pre,t2,flair}` (skull-stripped); **z-score-on-brain** (nonzero, channel-wise — the
  `downstream_seg` convention; independent of the VAE's 99.95); crop to the working resolution; segment at
  **image res** → per-class **avg-pool** to the latent grid (B.c-4). Never binarise (SoftSeg).
- **Augmentation (~80 % of robustness):** heavy intensity/bias/contrast (`RandBiasField`, `RandAdjustContrast`,
  `RandHistogramShift`, `RandGamma`, `RandGaussianNoise`) + spatial (flip, affine, mild elastic) +
  **modality-dropout** (randomly zero t2 *or* flair; t1c always absent). Pooled multi-cohort (FeTS +23–33 %).
  Add **BraTS-MEN 2023** iff Málaga has meningioma (D-d, open).
- **Nesting:** enforce/encourage `NETC ⊆ WT` (two independent sigmoids, region-based).
- **Inference mechanism:** the **K-fold ensemble mean** (free from B.b) is the pick; MC-dropout / TTA are
  cheaper alternates otherwise.
- **Label harmonisation:** `WT=(label>0)`, `NETC=(label==1)` — code-agnostic across BraTS-2021/2023.
- **Feasibility (de-risked):** WT Dice ~0.90 without T1c (Ruffle 2023); no-T1c TC drop is **ET-driven, not
  NETC** (necrosis is T1pre-dark) → NETC≥0.50 realistic; non-contrast enhancement predictable at 91.5 % sens
  (arXiv:2508.16650).
- **Output → T-04 cache** `masks/tumor_latent_pred (2,·,·,·)`; **`[verify]`** latent shape (48,56,48) vs
  (60,60,40) — match the generator grid.

## B.e — Verified references (parts c/d)
Soft-from-hard: Xue AAAI 2020 (SDM) · Ma MIDL 2020 · Kervadec boundary loss (1812.07032) · **SVLS**
(2104.05788). Soft-label Dice: **DML** (2303.16296). Overconfidence/calibration: Bertels (1911.02278 /
MedIA 2021) · DSC++ (2111.00528) · temp scaling (Guo 2017) · SSN (2006.06015) · MC-dropout (Gal 2016) ·
evidential (Sensoy 2018). Soft pipeline: **SoftSeg** (2011.09041). Soft-conditioning-helps-generator:
**density-conditioned synthesis** (2510.09365) · **SPADE** (1903.07291) · **seg-guided diffusion** (2402.05210).

---

# PART C — WHY THE MASK ROUTE (evidence from the iter-5 reviews) + leakage

## C.1 The refinement-vs-mask verdict (why we are here)
Two verified literature sweeps + the frozen v3a/v3b metrics settled it:
- **v3a's deficit is tumour-only** — whole-brain is already best-of-latent-tier (MAE_brain 0.095,
  MS-SSIM 0.919); it fails on MAE_wt (0.128 vs 0.095), ΔDice_ET (0.435 vs 0.073*), PSNR_ET (~12 vs ~16).
- **Refinement losses SHARPEN, don't LOCALISE** (15 papers unanimous; LPL/FFL/cascade/ReFlow). Sharpening a
  mean-brain that lacks enhancement adds no enhancement.
- **Adversarial places content only by HALLUCINATION** → unsafe for a Gd replacement (Kofler 2022 reader
  study: sens 56 %, "hallucinated findings"). Reserve arm behind a §6.5 gate.
- **The one positive-evidence segmenter-free route = subtraction target** (Osuala 2025) — but VENA's §14.2
  shows the *latent* residual is weak (VAE entangles enhancement); needs image-space/joint-norm.
- **The mask / predicted-mask route is the best-evidenced and = published SOTA (TA-ViT, Eidex Med Phys
  2025).** → build it. Optional cheap pre-flight before GPU-days: joint-normed **decoded** `loc_ET` test
  (does subtraction concentrate in image space?); if not, the mask is confirmed.
- **Competitor flags:** TA-ViT (must-cite baseline; = our approach), TuLaBM (2603.19386), Osuala subtraction
  (2508.13776), Kofler 2022 (safety citation).  *(\*the 0.073 is the leaky GT-`[NETC,ED,ET]` oracle — hence
  the T-13 fair `[WT,NETC]` oracle used this iteration.)*

## C.2 Leakage taxonomy (no cohort retirement, no re-run)
| # | Vector | Real? | Fix |
|---|---|---|---|
| L1 | BSF SSL saw masks/labels | **No** — unsupervised | premise false |
| L2 | segmenter trained on VENA-eval patients | **Yes** (the real one) | **out-of-fold** prediction (A.7) — not a corpus change |
| L3 | BSF-BraTS encoder saw BraTS images incl. T1ce (unsupervised) | mild | UKB-SSL init (leak-free) OR disclose + non-BraTS test headline |

VENA's training corpus is **untouched**; the segmenter is an add-on that emits masks. The 3-arm backbone
(B) measures the L3 trade-off empirically.

---

## Change log
- 2026-07-21 (iters 1–2) — segmenter Q1–Q3 reasoned; 3 verified lit sweeps + repo facts folded in.
  WT Dice ~0.90 no-T1c; NETC achievable (ET-driven drop); calibration recipe; soft `[WT,NETC]`;
  WT-only=fallback; no meningioma cohort.
- 2026-07-21 (iter 3) — BrainSegFounder probe: SwinUNETR fs=48, two encoder-only SSL ckpts; **UKB-SSL is
  T1-only** (corrected "T1+T2"); BraTS-SSL 4-ch domain-adapted encoder; neither is a segmenter. D-a locked.
- 2026-07-21 (iter 4) — leakage analysis (L1 false / L2 out-of-fold / L3 UKB-SSL); **no retire, no re-run**;
  SegResNet demoted to baseline.
- 2026-07-21 (iter 5) — both refinement reviews: refinement SHARPENS-not-LOCALISES; adversarial=hallucination
  (Kofler); subtraction weak-in-latent; **mask=SOTA=TA-ViT**. Verdict: build the mask route.
- 2026-07-21 (iter 6) — **document reorganised around the INJECTION MECHANISM** (user decision: bet on
  second-stage ControlNet mask-injection refinement of v3a). Part A (injection, this iteration): P1–P4
  properties; Q1 = inject at all down+mid levels (skips → decoder), levels learned via zero-init; Q2 =
  **ControlNet residual-add + zero-init + ramp PRIMARY** (only scheme guaranteeing P1+P4 via a separate/
  freezable branch), **SPADE/adaLN-zero = ablation** (TA-ViT's; B7 affine-fold), **FiLM rejected** (spatially
  uniform), **CA rejected** (dense pre-aligned mask needs no attention); trunk-update joint-low-LR primary /
  freeze-trunk conservative ablation; whole-brain no-regression gate. Q3 = segmenter-on-train-split
  defensible with out-of-fold. **Segmenter design DEFERRED (Part B keeps locked facts).**
- 2026-07-21 (iter 7) — **Part B detailed into B.a–B.e** (user scope: model+loss / data / soft-derivation /
  rest). Verified (c) sweep → 4-step soft-map recipe. **Correction: standard soft-Dice is IMPROPER on soft
  targets → use DML (Dice Semimetric Loss, Wang MICCAI 2023) + CE.** Loss principle = overlap in the loss,
  softness in the derivation. B.b = **K-fold OOF locked** (user) → the K fold-models double as a **free deep
  ensemble** for calibration. B.c "free" mass mechanisms compound: SDT→sigmoid soft targets + DML+CE + temp
  scaling + **avg-pool partial-volume** (the key free one) + free K-fold ensemble. B.d preprocessing/aug/
  nesting/harmonisation. Soft>hard for conditioning: density-conditioned synthesis (2510.09365), SPADE,
  seg-guided diffusion (2402.05210).
