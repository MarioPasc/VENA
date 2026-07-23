# Mask injection into v3a — second-stage refinement design

> **🔴 ERRATUM (2026-07-22): channel 0 = TC (tumour core = NETC+ET), NOT WT.** Verified during S1 that ~81% of WT is
> non-enhancing edema; conditioning on WT told the model to enhance mostly-edema. Throughout this doc, read every
> `[WT, NETC]` as `[TC, NETC]` with channel 0 = tumour core (edema excluded); `TC − NETC = ET` = the enhancing region.
> Config-driven (`TargetConfig.tumor_region="tc"`, wt kept for the S7 ablation); Phase-2 segmenter target = TC.
> See `[[project_channel0_tumor_core_not_wt]]` and `DEVELOPMENT/01_SHARED_CONTRACTS.md`.

> **🔴 ERRATUM (2026-07-23): the served latent grid is `(48,56,48)`, NOT `(60,60,40)`.** Verified against the
> producer (`data/h5/latent_domain/manifest.py`: `LATENT_SPATIAL=(48,56,48)`, `LATENT_CROP_BOX=(192,224,192)`,
> avg-pool stride 4), the v3a config (`base_img_size_numel=129024=48×56×48`), and Picasso disk. **The
> Locked-decisions line, the A.5 recipe, and A.8-§6 below have it BACKWARDS** — they call `(48,56,48)` "stale"
> and `129024` a "mismatch"; both are FALSE. `129024` is correct and needs NO reconciliation (`(60,60,40)`
> assumed the *uncropped* `240×240×155` volume and ignored the `(192,224,192)` crop). Every mask/cache is
> `(2,48,56,48)`. `01_SHARED_CONTRACTS.md` + `SESSIONS.md` were corrected in commit `9eed6c1`; the corrected
> spots below are annotated inline. See `[[reference_latent_grid_48_56_48]]`.

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

**Locked decisions.** bet = ControlNet mask-injection refinement of **v3a** (warm-start
`…_s1_v3a_concat_only_fft_ef000c9f`, add a **fresh** `[WT,NETC]` ControlNet — **confirmed iter-8;
supersedes `model_redesign` T-06/T-13's v3b source, which becomes the T-10 warm-start-source ablation**) ·
fusion = **ControlNet primary / SPADE = T-07 ablation** (FiLM & CA rejected) · mask = soft `[WT,NETC]`
(WT-only = fallback) · **generator grid = `(48,56,48)`** (the *served* latent grid; the iter-8 `(60,60,40)` was a transcription
error — see the grid erratum above and A.8-§6; `129024=48×56×48` is correct) · generator loss = **region-weighted CFM with
`{Brain, WT}` weights EQUAL initially (numerically ≡ unweighted L1); WT up-weight = deferred ablation**
(mechanism coded now, cf. Ibarra MICCAI 2025 — see A.8-§2) · segmenter backbone = **BSF-SwinUNETR**
(Arm A BraTS-SSL / Arm B UKB-SSL leak-free / Arm C SegResNet baseline, forkable from
`src/vena/validation/downstream_seg.py` — **supersedes `model_redesign` §16.6's SegResNet-primary**) ·
segmenter loss = **DML+CE on SDT-soft targets** (Dice = eval gate only) · splits = **K-fold OOF** (fold-models
double as a calibration ensemble — but **k-fold ≠ true deep ensemble for uncertainty**, arXiv:2605.18329: the
*mean* calibrates/smooths, the *variance* is confounded, see B.f-§3) · calibration = **per-class temperature
`T_WT`, `T_NETC`** (not one global T, see B.f-§2) · generator training masks = **clean GT for T-13 (oracle
ceiling)** / OOF **predicted + mask-perturbation aug for T-06 (deployable)** (see A.8-§7) · intensity norm =
**99.95** canonical.

**Open decisions.** **D-d:** add BraTS-MEN? (gated on the Málaga pathology mix — glioma-only vs
+meningioma). **Soft-target path:** SDT+DML from the start vs free-only (ensemble+temp+avg-pool) then
upgrade. **NETC SDT operator:** per-connected-component Euclidean vs signed-normalised geodesic (SiNGR,
arXiv:2405.16813) — NETC is often multifocal (see B.f-§4). **Deferred-ablation levers (coded-but-off this
iteration):** CFG-at-inference (FP-gated), noise-level-dependent `output_scale`, ensemble-variance
conditioning channel (see A.8-§8, B.f-§5).

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

## A.5 Concrete recipe  (updated iter-8 — A.8 holds the validated code fact behind each annotated line)
```
resume_from      : v3a ema_best.ckpt  (run 2026-06-24_16-00-46_s1_v3a_concat_only_fft_ef000c9f; pin Picasso path)
                   run.resume_from: <v3a run_id>  → WARM_START (weights-only; optimiser/EMA/RNG fresh)
mask input       : GT [WT,NETC] SOFT, latent grid (48,56,48) (oracle → this iter = T-13)
                   served as batch["m_wt_soft"] (NEW — pre-threshold union) + batch["m_netc"] (already served)
fusion           : MaisiControlNet, init_from_trunk (copies v3a down+mid into the CN encoder). Conditioning
                   enters the SEPARATE controlnet_cond_embedding hint net ([64] → zero spatial downsampling,
                   mask already at latent grid) and is ADDED to the CN conv_in output — NOT concatenated (A.8-§1)
cond specs       : TWO specs  mask:wt:identity + mask:netc:identity  (NOT one 2-ch key — the assembler
                   under-counts channels silently, A.8-§4).  MASK-ONLY controlnet_cond (no anatomy latents →
                   homogeneous [0,1] hint-net input; mixing latents needs per-channel norm, A.8-§3)
output proj      : zero-init (zero-conv)         # P1
output_scale     : sigmoid ramp 0→1 over 5000 steps (OutputScaleRampCallback; buffer persistent=False,
                   recomputed from global_step on resume; applied to every down+mid residual)   # P1
trunk update     : PRIMARY joint low-LR + trunk-EMA;  ABLATION freeze-trunk (adapter-only).
                   ⚠ trainable-trunk warm-start is single-shot / not resume-safe — trunk_ema is built in
                   setup() after the ckpt load, so it needs v3a's trunk_ema_snapshot.pt (A.8-§5).
                   freeze-trunk sidesteps this entirely and gives the strongest P1/P4 guarantee.
loss             : region-weighted CFM, regions {Brain = NOT-BG ∩ NOT-WT, WT}, EQUAL weights initially
                   ({brain:1.0, wt:1.0} ≡ unweighted L1 velocity — mechanism coded, WT up-weight is a
                   deferred ablation axis, A.8-§2) (+ optional whole-brain anchor A.4.4; T-09 rim loss later)
rflow            : use_timestep_transform=true.  base_img_size_numel=129024=(48×56×48) MATCHES the served grid
                   (48,56,48) — no reconciliation needed (the (60,60,40)=144000 "mismatch" was the grid error).
                   Plumb input_img_size_numel to the EulerSampler or every per-patient val silently fails
                   (feedback_euler_sampler_timestep_transform).
EMA              : 0.9999 ; EarlyStopping patience 250 ; monitor train/total_epoch
run family       : picasso_ref_v1_v3a+cn[WT,NETC]_fft.yaml  (loginexa smoke first)
decision.json    : schema 0.10.0. Set controlnet_enabled=true,
                   controlnet_conditioning_inputs=[mask:wt:identity, mask:netc:identity], mask_repr=[WT,NETC],
                   mask_source=oracle(GT), warm_start_source=v3a, trunk_update={joint_lowlr|frozen},
                   output_scale_ramp_steps, region_weights={brain:1.0, wt:1.0}
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

## A.8 — Iter-8 refinements: validated injection-code facts, normalization, loss, mask-perturbation

> Iter-8 audited the actual ControlNet code (`src/vena/model/fm/controlnet/`, `…/maisi/`, `…/lightning/`)
> and the v3a run config against Part A's claims, then closed the four Q1–Q4 decisions of 2026-07-22. The
> injection **levels** are confirmed correct; the fixes below are precision, normalization, plumbing, and
> the loss/robustness/oracle decisions. Each `A.8-§n` is referenced from A.5.

**§1 — Injection levels & mask-input path (CONFIRMED, with one precision fix).** The MAISI trunk is a 3D
U-Net and the ControlNet is a trainable copy of its **down-blocks + mid-block**. Residuals are emitted and
added back at **`conv_in` + every down-block resblock output (channels 64→128→256→512) + the mid-block**;
they reach the decoder **only through the residual-augmented skip connections** (no separate up-block
injection). This matches A.2 exactly. When the trunk is trainable, the two residual adds are rebound
**out-of-place** by `maisi/grad_safe.py` (numerics unchanged). **Precision fix to A.2:** the mask does **not**
concatenate to the noisy latent at `conv_in`. It enters a **separate `controlnet_cond_embedding` hint
network** (configured `[64]` → **zero** spatial downsampling, because the mask already lives on the 4×
latent grid) and is **added** to the ControlNet's `conv_in` output at full latent resolution. The trunk's
own `conv_in` is untouched by the mask.

**§2 — Generator loss (Q_B resolved: region-weighted CFM, equal weights first).** Do **not** run pure L1 vs
WT-weighted as a two-arm sweep yet — that would open an untuned paper axis prematurely. Instead **code the
region-weighted CFM mechanism** with regions `{Brain = NOT-BG ∩ NOT-WT, WT}` and run it at **equal weights
`{brain:1.0, wt:1.0}`, which is numerically identical to the current unweighted L1 velocity loss** (assert
this equivalence in a test — A.5). The `region_weights` infrastructure already exists (the retired v3b_rw
arm; `decision.json` 0.10.0 carries `region_weights`). **WT up-weighting (`{5,10,20}`) is then a single,
deliberate, later ablation axis** — motivated by the <0.1 %-voxel ET imbalance and by ROI-weighted CE-MRI
synthesis (Ibarra et al., *Comparing Conditional Diffusion Models for CE Breast MRI*, MICCAI 2025,
arXiv:2508.13776, which shows ROI-loss and mask-conditioning are **complementary**), and gated on **PSNR_ET
AND** the §6.5 false-positive-enhancement rate (over-weighting hallucinates enhancement into any mask-overlap).
Rationale for min-SNR-style imbalance concern: Hang et al., ICCV 2023, arXiv:2303.09556.

**§3 — Mask normalization (the "correctly normalized" audit answer).** Masks are fed at their **native range**
(soft `[0,1]`; the legacy binary WT is `{0,1}`) with **no `latent_scale`** applied, through a hint net that is
**always freshly initialised** (`controlnet_cond_embedding` is CN-only and never copied by `init_from_trunk`)
→ **no warm-start normalization mismatch** for the mask path. This is clean **iff the conditioning is
mask-only**. If anatomy latents are mixed into `controlnet_cond` (the A.4.2 "mask + context" option), the hint
net sees `[0,1]` mask channels alongside `~±several-unit` latent channels — a real scale disparity. **Decision:
default to MASK-ONLY** `controlnet_cond` (also the strongest P2 locality); if context is ever added, apply
explicit per-channel normalisation before assembly. (The trunk-input concat path — v3a's `[t1pre,t2,flair]`
latents into the trunk's 16-ch `conv_in` — is unrelated and unchanged.)

**§4 — Two-channel `[WT,NETC]` plumbing (DEFECT to avoid).** `ConditioningAssembler.channels_per_spec` reads
the `mask_channels` **constructor default (=1)**, not the runtime tensor shape. A single `mask:wt:identity`
spec fed a `(B,2,…)` tensor silently **under-counts** `total_channels`, builds the hint net's first conv with
the wrong `in_channels`, and only errors at the first forward. **Fix: declare TWO specs**
`mask:wt:identity` + `mask:netc:identity`, each a 1-ch batch key (`m_wt_soft`, `m_netc`). (`lift_to_4ch`
similarly needs `in_channels=2` if ever used on a 2-ch mask.) The wrapper never passes MONAI's
`conditioning_scale` (stays 1.0) and relies on `output_scale` alone — do not set both.

**§5 — Warm-start mechanics (v3a) & the trunk-EMA landmine.** `run.resume_from: <v3a_run_id>` classifies as
**WARM_START**: a `_WarmStartCallback` loads v3a weights only (optimiser / EMA / RNG start fresh); the fresh
ControlNet's down+mid blocks are seeded from v3a's trunk via `init_from_trunk`; the hint net + zero-init
output convs are fresh (correct by design). **Landmine:** with a **trainable trunk**, `self.trunk_ema` is
built in `setup()` *after* Lightning's checkpoint restore, so trainable-trunk warm-start is **single-shot,
not resume-safe** and depends on v3a having emitted a `trunk_ema_snapshot.pt` sibling for
`_maybe_load_trunk_ema_snapshot()`. **The freeze-trunk ablation (adapter-only) sidesteps this entirely** and
gives the strongest P1/P4 guarantee — run it first to de-risk, then the joint-low-LR primary.

**§6 — Grid `(48,56,48)` [CORRECTED 2026-07-23 — the iter-8 `(60,60,40)` claim in this section was WRONG].** The
DataModule serves latents and masks at **`(4|1, 48, 56, 48)`** — the MAISI 4× latent of the **`(192,224,192)` crop**
of ~`(240,240,155)`, NOT the uncropped full volume (the `240/4 × 240/4 × ⌈155/4⌉ = (60,60,40)` reasoning ignored the
crop). Verified from `data/h5/latent_domain/manifest.py` (`LATENT_SPATIAL=(48,56,48)`, `LATENT_CROP_BOX=(192,224,192)`)
and Picasso disk. Every mask target, avg-pool output, and `masks/tumor_latent_pred` cache is `(2, 48, 56, 48)` —
matching `model_redesign` T-04. `rflow.base_img_size_numel = 129024 = 48×56×48` **matches** the grid: there is NO
mismatch to reconcile (the `≠144000` claim was the error). **Also (still valid):** expose the soft tumour-core union
as `batch["m_tc_soft"]` (channel 0 of the cached `masks/tumor_latent_soft`; per the TC erratum) — the oracle
`[TC,NETC]` must be **soft**, not the 0.5-thresholded binary `m_wt`.

**§7 — Oracle → predicted gap & mask-perturbation (Q_C resolved: clean T-13, perturb T-06).** **T-13 uses the
clean GT `[WT,NETC]`** (no perturbation) so it measures the true injection **ceiling**. The deployable **T-06**
then adds **mask-perturbation augmentation** to close the train-on-GT → deploy-on-predicted distribution gap,
which is otherwise real (Ko et al., *Stochastic Conditional Diffusion Models*, ICML 2024, arXiv:2402.16506;
Ho et al. *Noise-Conditioning-Augmentation*, arXiv:2106.15282; Imagen, arXiv:2205.11487). Perturbation recipe
(latent grid): random **dilation/erosion ±3 vox**, **additive soft-prob Gaussian noise σ ∼ U[0, 0.15]**, and
**whole-map dropout p ≈ 0.15** (dual-purpose: enables optional CFG, §8). Preserve `NETC ⊆ WT` after
perturbation. **Report the oracle-vs-predicted PSNR_ET gap as a first-class table column** — TA-ViT (Eidex
2025, arXiv:2409.01622) does not report it, so it is a clean VENA contribution. This also updates A.6: the
no-regression gate is evaluated on the clean oracle (T-13), and the same gate + the T-14 FP study on the
predicted mask (T-06).

> **🔴 ITER-9 ADDITION (2026-07-23) — the oracle→predicted gap is a FIRST-CLASS OOD CONTRIBUTION, reported PER
> RING.** The deployable arm's OOD ceiling is the *segmenter*, which fails under the same shifts §4.4 measures —
> a coupled segmenter+generator OOD failure that is a **confirmed open problem** (no prior work; arXiv:2508.16650,
> N=11,089 across 10 cohorts, quantifies OOD segmenter degradation; Ko's condition-robustness is in-distribution
> only, with **no structured-error regime**). Actions: **(a)** report `PSNR_ET(oracle T-13) − PSNR_ET(predicted
> T-06)` **stratified by ring** (Ring A vs Ring B), never as one pooled number — see
> `../../article/03_generalization_ood.md` T3.6; **(b)** measure the segmenter's **Ring-B TC/NETC error
> distribution** (Ring-B cohorts carry GT) so the *structured* OOD failure modes are characterised — the
> dilation/erosion+Gaussian perturbation aug above does **not** reproduce them; **(c)** frame the honest
> **decomposition = localisation (segmenter) + intensity (generator)**: T-13's oracle mask *leaks post-contrast*
> (GT ET is defined by the real T1c), so it is a legitimate ceiling but NEVER the deployable method — the
> oracle→predicted gap *is* the price of localising enhancement without contrast.

**§8 — Deferred-ablation levers (coded-but-OFF this iteration).** Two levers are documented for later, gated
behind flags, not run in the T-13/T-06 headline: **(a) CFG-at-inference** (currently *not implemented* — only
training-time conditioning dropout exists). Guidance would amplify enhancement but is **FP-risk-double-edged**
for a Gd replacement; if trialed, restrict guidance to a **middle noise interval** (Kynkäänniemi et al.,
NeurIPS 2024, arXiv:2404.07724) and suppress saturation with **APG** (Sadat et al., arXiv:2410.02416), or use
**autoguidance** (Karras et al., NeurIPS 2024, arXiv:2406.02507); sweep `guidance_scale ∈ {1.0,1.5,2.0}`
against PSNR_ET **and** §6.5 FP rate; reject if `1.0` already maximises PSNR_ET. **(b) Noise-level-dependent
`output_scale`** — gate the ramp scalar at inference by a `sigmoid` window peaking at α ∈ [0.3,0.7]
(complementary to the already-on `use_timestep_transform`, SD3 Esser et al. arXiv:2403.03206); a ~3-line
change to `MaisiControlNet.forward`. Include only if it beats constant `output_scale=1.0`.

## A.9 — Normal-enhancement evaluation (iter-9, Q3 = evaluate-only)

> Gd enhances **normal** structures too — intracranial vessels, dural venous sinuses, choroid plexus,
> pituitary/infundibulum — but the tumour-core ControlNet conditions **none** of them, and (Q3 decision) **no
> conditioning channel is reserved**. The generator must reproduce normal enhancement from v3a's learned prior
> alone; this evaluation checks whether it does. **Evaluation only — the 2-ch `[TC,NETC]` layout is unchanged.**

- **Why (open gap + contribution + safety).** Normal-enhancement fidelity is the field's **named #1 open gap**
  (Moya-Sáez et al., *Front. Neuroimaging* 2023, DOI:10.3389/fnimg.2023.1055463); the SOTA baseline TA-ViT
  (Eidex 2025, arXiv:2409.01622) *structurally ignores* non-tumour enhancement; the only work that evaluates it
  used a **mask-free** model (choroid-plexus SynCE, *Radiology Advances* 2025, DOI:10.1093/radadv/umaf042) —
  evidence that ROI-only conditioning risks *losing* what whole-image models retain. Reporting it is a clean
  VENA contribution and a **safety signal** feeding the §6.5 false-enhancement study.
- **Regions.** Reuse existing machinery — the Frangi vessel prior (`src/vena/prior_maps/`) and the
  `venous_atlas_build` preflight — for normal-vessel / dural-sinus masks; add choroid-plexus + pituitary ROIs
  where an atlas/segmenter is available. Evaluate on **healthy-appearing** brain (exclude the tumour
  neighbourhood) so tumour enhancement doesn't confound.
- **Metric.** In-ROI fidelity of the *synthetic* T1c vs real T1c inside each normal-enhancing ROI (matched
  99.95 norm): MAE / PSNR in-ROI + a **normal-enhancement recall** (does the synth reproduce the real
  enhancement present in that ROI?). **Report per ring** (does normal enhancement degrade OOD?).
- **If the generator systematically misses normal enhancement**, that is the evidence-based motivation for a
  deferred vessel/SWAN conditioning channel (future iteration) — the decision waits on this measurement, per Q3.

---

# PART B — THE SEGMENTER (detailed: model+loss / data / soft-derivation / rest)

> The segmenter, detailed into the four axes the user scoped. It is **decoupled** from the injection
> mechanism (Part A), which is validated first on the **oracle** mask (T-13). Design only.
> `[WT,NETC]` soft, 2-ch (fallback 1-ch `[WT]` only if NETC fails G-SEG). ET is undefined pre-contrast;
> the generator resolves the enhancing subset of `WT∖NETC`.

## B.a — Model + loss (and the soft-Dice correction)
- **Model (D-a locked; arm priority updated iter-9 2026-07-23): SwinUNETR (feature_size=48) init from
  BrainSegFounder** (MONAI, Apache-2.0). Checkpoints LOCATED + pinned in `src/external/LINKS.md`.
  **Arm B = UKB-SSL** (`64-gpu-model_bestValRMSE.pt`; healthy UK Biobank, **no BraTS patients, no T1ce**) — **now the
  leak-free PRIMARY/HEADLINE** (L3 purity demanded: BraTS-SSL's patient-overlap + T1ce leak is **unfixable by OOF**
  because it happens at the SSL stage). Likely T1-focused → the 3-ch input stem may not transfer (`strict=False`, stem
  in `skipped`); deep blocks do. **Arm A = BraTS-SSL** (`model_bestValRMSE-fold*.pt`, multimodal+tumour-aware; drop the
  T1ce slice, feed `[FLAIR,T1pre,T2]`, `strict=False`) — **now the domain-matched COMPARATOR** (higher tumour match but
  leaks; run it to quantify the leakage↔Dice trade-off). Arm C = **SegResNet-from-scratch** — the no-pretraining floor.
  **NEVER the finetuned BraTS ckpt** (supervised = L1+L2+L3 leak). Bench alternatives: MedNeXt, nnU-Net.
  - **Arch / load-coverage measured (S5, 2026-07-23):** the two SSL ckpts are the **same Swin-UNETR family at
    different scales** — UKB is the MONAI default `depths=(2,2,2,2)` (loads **125/142 = 0.880**; 1-ch stem + 16 SSL
    heads skipped), BraTS is the Swin-T `depths=(2,2,6,2)` (6 stage-3 blocks; loads **182/198 = 0.919** once the
    depth is matched — only the 16 SSL pretext heads `rotation/contrastive/conv.*` drop, **0 encoder keys**). Per-arm
    `depths`/`num_heads` are pinned as module constants (`_BSF_BRATS_SWIN_KW` / `_BSF_UKB_SWIN_KW`) in
    `bsf_swinunetr.py`; the initial 0.636 was a construction-config mismatch (we defaulted to `(2,2,2,2)`), **not** a
    deeper-than-standard architecture. Stem-adaptation options in `bsf_stem_t1ce_removal.md` (drop-slice `[0,1,3]` chosen).
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
The soft `[WT,NETC]` map is **derived**, not read off the overconfident sigmoid.

> **🔴 ITER-9 DECISION (2026-07-23) — σ-only; temperature DROPPED.** An audit reframed the "4 compounding
> softeners": avg-pool (the **mandatory** image→latent bridge — kernel fixed by the 4× compression), DML+CE
> (the soft-target **loss**, not a softener), and the K-fold mean (**mandatory** for leak-free OOF masks) are
> infrastructure with **no tunable knob**. **SDT-σ is the ONE discretionary knob**, and it is load-bearing
> because it makes the oracle and predicted maps *match* (a segmenter trained toward SDT-σ targets emits
> SDT-σ-shaped softness ≈ the oracle's SDT-σ→pool). **Temperature (old step 3) is DROPPED from the pipeline** —
> soft-target+DML+CE already curbs overconfidence (label-smoothing→calibration, Müller et al. NeurIPS 2019,
> arXiv:1906.02629); post-hoc TS only adds a *segmentation*-ECE gain (Buddenkotte), and whether that helps
> *conditioning* (PSNR_ET) is the unproven Q3 open problem — not a knob to carry in the headline. **Calibration
> is MEASURED (report ECE/Brier per G-SEG), NOT corrected.** The pipeline is now **SDT-σ → DML+CE → K-fold OOF
> mean → avg-pool** (one knob). `temperature.py` stays as an unused utility (coding-agent hygiene: delete when
> no live caller remains). See `SESSIONS.md` planning-decision Q5.

The surviving mechanisms (each cheap/free) and they **compound**:
1. **(target) SDT → sigmoid soft targets.** From each hard label compute the normalised **signed distance
   transform**, map `y_soft = sigmoid(SDT/σ)` (σ≈3 vox): 0.5 at the boundary, ~0.95 at 3σ inside, ~0.05 at
   3σ outside (Ma MIDL 2020; Kervadec boundary loss arXiv:1812.07032; **SVLS** = the Gaussian special case,
   Islam & Glocker IPMI 2021). Distributes mass by distance-to-boundary *by construction*.
2. **(loss) DML + CE on the soft targets** (B.a) — proper soft-label Dice (Wang 2023), **not** standard soft-Dice.
3. ~~**(post-hoc) temperature scaling**~~ — **DROPPED (iter-9, see banner above).** Calibration is *measured*
   (ECE/Brier), not corrected; no `T_TC`/`T_NETC` in the pipeline. (Guo ICML 2017 retained as the reference for
   the *measured* ECE only.)
4. **(pool) average-pool to the latent grid** — `AvgPool3d(stride=4)` = **partial-volume integration**: a
   boundary latent voxel gets 0.25–0.75 by enclosed-lesion fraction. **Free, differentiable — the single most
   important "free" spatial-mass mechanism** ("no additional mechanism required").
- **Plus the free K-fold ensemble mean** (B.b) — further calibrates/smooths at no extra training cost.
- **Minimal free-only path** (superseded by the iter-9 σ-only decision): there is now one headline path —
  SDT-σ + DML+CE + K-fold mean + avg-pool, **no temperature**.
- **Tuning caveat (now the ONLY knob):** σ is the single free parameter. Tune it *once* by a post-pool grading
  criterion — e.g. the boundary-transition width in latent voxels, or the fraction of soft-mass inside hard-TC
  landing in a target band — and **share the same σ between the oracle target and the segmenter target** so the
  two maps match by construction. avg-pool has no knob (kernel fixed by the 4× compression); over-softening is
  controlled by σ alone.
- **ET = TC−NETC is a reported diagnostic, not a gate (iter-9).** Report ET-Dice + mean soft-value in the ET
  shell alongside TC/NETC so NETC miscalibration that corrupts the *enhancing* region is visible — ET is the
  load-bearing derived quantity the generator must enhance. (Gating stays on TC/NETC per G-SEG; ET is
  reported-for-visibility.)
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
- **Output → T-04 cache** `masks/tumor_latent_pred (2,48,56,48)` — matches the generator grid (resolved: the
  served grid is (48,56,48), not (60,60,40); see the grid erratum at the top).

## B.e — Verified references (parts c/d)
Soft-from-hard: Xue AAAI 2020 (SDM) · Ma MIDL 2020 · Kervadec boundary loss (1812.07032) · **SVLS**
(2104.05788). Soft-label Dice: **DML** (2303.16296). Overconfidence/calibration: Bertels (1911.02278 /
MedIA 2021) · DSC++ (2111.00528) · temp scaling (Guo 2017) · SSN (2006.06015) · MC-dropout (Gal 2016) ·
evidential (Sensoy 2018). Soft pipeline: **SoftSeg** (2011.09041). Soft-conditioning-helps-generator:
**density-conditioned synthesis** (2510.09365) · **SPADE** (1903.07291) · **seg-guided diffusion** (2402.05210).

**Iter-8 additions (all verified 2026-07-22).** Injection/robustness (Part A / A.8): oracle→predicted &
condition-noise-aug = **SCDM/label-diffusion** (Ko, ICML 2024, 2402.16506) · **NCA** (Ho cascaded, 2106.15282;
Imagen, 2205.11487) · **TA-ViT/T1C-RFlow** unreported oracle gap (Eidex, Med. Phys. 2025, 2409.01622);
ROI-weighted CE-MRI synthesis = **Ibarra** (MICCAI 2025, 2508.13776) · min-SNR imbalance (Hang, ICCV 2023,
2303.09556) · lesion-weighted 3D synth (2606.15457, *preprint*); guidance = **guidance-interval**
(Kynkäänniemi, NeurIPS 2024, 2404.07724) · **APG** (Sadat, 2410.02416, *preprint*) · **autoguidance**
(Karras, NeurIPS 2024, 2406.02507) · **SD3 timestep-transform** (Esser, 2403.03206). Segmenter (Part B /
B.f): per-class/Dirichlet calibration = **Kull** (NeurIPS 2019) · ensemble-still-needs-TS = **Buddenkotte**
(CBM 2023, 2209.09563) · **LTS** (Ding, ICCV 2021, 2008.05105) · **focal-calibration** (Mukhoti, NeurIPS 2020,
2002.09437); ensemble uncertainty = **deep ensembles** (Lakshminarayanan, NeurIPS 2017, 1612.01474) ·
**under-shift** (Ovadia, NeurIPS 2019, 1906.02530) · **k-fold≠deep-ensemble** (2605.18329, *preprint*) ·
**DGRNet** (2603.21086, *preprint*); SDT = **SiNGR** (Juanola, MICCAI 2024, 2405.16813) · **Ma DT study**
(MIDL 2020) · **GeoLS** (Vasudeva, MIDL 2024) · **skeleton-aware DT** (2310.05262); TTA = **Wang**
(Neurocomputing 2019, 10.1016/j.neucom.2019.01.103).

## B.f — Iter-8 segmenter refinements (calibration, ensemble, SDT, selection)

> A verified segmenter/soft-map literature sweep (2026-07-22) added five refinements to Part B. None change
> the locked backbone/loss/splits; they correct calibration and SDT details and add honest caveats.

**§1 — Backbone confirmed; SegResNet demoted to a *forkable* baseline.** **BSF-SwinUNETR (fs=48) is the
primary** (Arm A BraTS-SSL, Arm B UKB-SSL). **Arm C = SegResNet-from-scratch** is the pretraining-gain
baseline and should be **forked from the existing 4-input `src/vena/validation/downstream_seg.py`** (drop the
T1c input → 3-input `{T1pre,T2,FLAIR}`). This **supersedes `model_redesign` §16.6**, which named SegResNet as
*primary* — that predated the iter-3 BrainSegFounder probe.

**§2 — Calibration.** **🔴 SUPERSEDED by the iter-9 σ-only decision (2026-07-23): temperature is DROPPED —
calibration is MEASURED (ECE/Brier), NOT corrected.** The reasoning below (per-class TS, TS-after-ensemble) is
retained for provenance and for the *measurement* references, but no `T_TC`/`T_NETC` are fitted in the
pipeline; whether post-hoc calibration would improve *conditioning* (not just segmentation-ECE) is the Q3 open
problem, deliberately left unbet in the headline. Original reasoning: Fit **two independent
temperatures `T_WT`, `T_NETC`** on the OOF calibration split, not one global scalar — a global T is strictly
suboptimal for multi-region outputs (Kull et al., *Dirichlet calibration*, NeurIPS 2019; global TS is its
equal-off-diagonal special case). Crucially, **ensemble averaging does NOT make temperature scaling
redundant** for medical segmentation — residual miscalibration remains in the mean and TS on top gives a
significant further ECE reduction (Buddenkotte et al., *Calibrating ensembles…*, Comput. Biol. Med. 2023,
arXiv:2209.09563). Spatial **Local Temperature Scaling** (Ding et al., ICCV 2021, arXiv:2008.05105) is the
ceiling but is **not worth it here** — its boundary-calibration gain is washed out by the subsequent 4×
avg-pool to the latent grid. Optional training-time alternative to post-hoc TS: **focal-CE** produces
inherently better-calibrated models (Mukhoti et al., NeurIPS 2020, arXiv:2002.09437) — ablate against DML+CE.

**§3 — "Free deep ensemble" caveat: k-fold ≠ deep ensemble for uncertainty.** The K fold-models still give a
useful **ensemble mean** (calibration/smoothing) and are needed anyway for leak-free OOF masks. But k-fold
members train on **different data subsets**, so their per-voxel disagreement **conflates data-exposure with
seed variability** and over-estimates boundary uncertainty — it is *not* pure epistemic uncertainty (*Lost in
the Folds*, 2025, arXiv:2605.18329). Under distribution shift (the Málaga OOD cohort) true random-init deep
ensembles are the only method that stays calibrated (Ovadia et al., NeurIPS 2019, arXiv:1906.02530).
**Actionable:** keep k-fold OOF for the masks; use the mean as the soft map; **do not sell k-fold variance as
epistemic uncertainty** in the paper. If a clean uncertainty signal is later required, add a small
**random-init deep ensemble on the all-FM-train model** (same data, different seeds).

**§4 — SDT correctness for multifocal NETC + pooling order.** NETC is frequently **disconnected** in GBM
(satellite nodules / necrotic components, ~20–30 % of cases). **Naive Euclidean SDT mishandles the
inter-component gap** (assigns positive "interior" scores between lobes; He et al., MICCAI 2023,
arXiv:2310.05262). **Fix for the NETC channel:** per-connected-component Euclidean SDT (union) **or** a
signed **normalised geodesic** transform routed through image intensity (SiNGR, Juanola et al., MICCAI 2024,
arXiv:2405.16813; GeoLS, Vasudeva et al., MIDL 2024) — the segmenter sees `{FLAIR,T1pre,T2}` so geodesic is
available. Follow the DT best-practices: **per-class**, **normalised**, **clipped** (Ma et al., MIDL 2020).
**Pooling order is `SDT → sigmoid → avg-pool`** (the doc's B.c order is already correct); never avg-pool raw
signed SDT (small lesions get a confusing negative mean at the latent scale).

**§5 — Ensemble-variance conditioning channel = ablation-only.** The per-voxel STD across fold-models can be
added as an **optional 3rd conditioning channel** (precedent: DGRNet uses disagreement→spatial-gating for
BraTS, 2026, arXiv:2603.21086). Gate it behind an **ablation flag**; do not put it in the headline, and label
it "k-fold disagreement", not epistemic uncertainty (§3).

**§6 — Inference: k-fold mean primary; TTA optional on Málaga; no MC-dropout.** The K-fold ensemble mean is
the pick. Add **test-time augmentation** (flip + small rotation over the fold-models) **only on the external
Málaga cohort**, where it captures a complementary aleatoric axis (Wang et al., Neurocomputing 2019,
DOI:10.1016/j.neucom.2019.01.103). **Skip MC-dropout** — dominated by both TTA and ensembles on every
calibration metric under shift (Ovadia 2019).

**§7 — Segmenter selection: dual DSC + calibration, because the generator eats soft probs.** Dice measures
*where the boundary is*; the generator consumes the **real-valued probability**, so its **calibration**
(Brier / classwise-ECE) is at least as load-bearing — a sharp-but-overconfident mask can be a *worse*
conditioner than a slightly blurrier well-calibrated one, and mask quality propagates into synthesis quality
(Konz et al., *segmentation-guided diffusion*, MICCAI 2024, arXiv:2402.05210). No study directly ranks
Dice-vs-calibration for a downstream *generator* (an open gap → a small VENA methodological note). **Actionable:**
report **both DSC and Brier/classwise-ECE** per fold-model and per cohort; when within ~1 % DSC, prefer the
better-Brier model; run a 2-point ablation (Dice-best vs Brier-best ensemble → generated-image PSNR_ET).

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
| L3 | BSF-BraTS encoder saw BraTS-GLI (a VENA CV cohort) images incl. T1ce (unsupervised) — **not "mild": OOF cannot fix an SSL-stage representation leak** | real | **RESOLVED iter-9: UKB-SSL is the headline** (healthy UKB, no BraTS patients, no T1ce → leak-free); BraTS-SSL kept only as a leaky comparator that quantifies the trade-off |

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
- 2026-07-22 (iter 8) — **code-validated + four Q1–Q4 decisions locked + literature-hardened**, ahead of the
  v3a-resume implementation. Audited `controlnet/`, `maisi/grad_safe`, `lightning/{module,data}`, and the v3a
  run config against Part A. **Injection levels CONFIRMED correct** (conv_in + all down resblocks + mid,
  decoder via skips); added **A.8** with the precision fixes: mask enters a **separate `controlnet_cond_embedding`
  hint net (added, not concatenated)**; **mask-only normalisation** is clean, mask+context needs per-channel
  norm; **2-ch `[WT,NETC]` = TWO specs** (assembler under-counts a single 2-ch key); `output_scale` correct;
  **trunk-EMA warm-start is single-shot/not-resume-safe** (freeze-trunk sidesteps). **Grid resolved to
  `(60,60,40)`** [❌ CORRECTED iter-9 — BACKWARDS; served grid is `(48,56,48)`, `129024=48×56×48` is correct,
  see grid erratum] (served) — every `(48,56,48)`/`129024` reference is stale; expose **soft-WT** `m_wt_soft`.
  **Q_A → warm-start v3a + fresh ControlNet** (v3b → T-10 ablation; supersedes redesign T-06/T-13 source).
  **Q_B → region-weighted CFM coded but run at EQUAL `{brain,wt}` weights first** (≡ L1; WT up-weight =
  deferred ablation, Ibarra 2508.13776). **Q_C → clean-oracle T-13 ceiling; mask-perturbation aug only on
  T-06** (Ko 2402.16506, Ho NCA) + **report oracle-vs-predicted gap** (unreported in TA-ViT). **Q_D → seg
  module derives soft masks; a thin routine writes `masks/tumor_latent_pred (2,60,60,40)` to the latent H5.**
  CFG-at-inference (not implemented) + noise-level `output_scale` = **deferred FP-gated levers** (Kynkäänniemi
  2404.07724, Sadat 2410.02416). **B.f** segmenter refinements: **per-class `T_WT`/`T_NETC`** + TS-still-needed
  post-ensemble (Buddenkotte 2209.09563); **k-fold ≠ deep ensemble for uncertainty** (2605.18329) — mean OK,
  variance confounded; **geodesic/per-component SDT for multifocal NETC** (SiNGR 2405.16813); variance channel
  = ablation-only; optional TTA on Málaga, no MC-dropout; **dual DSC+Brier selection**. SegResNet demoted to
  Arm-C baseline (fork `downstream_seg`) — supersedes redesign §16.6. Implementation task-graph:
  `.claude/notes/changes/vena_new_iteration/DEVELOPMENT/`.
- 2026-07-23 (iter 9) — **scientist audit + 3 locked decisions + grid-doc fix.** Audited the injection mechanism
  (ENDORSED: ControlNet + zero-init + ramp is right; FiLM/CA rejections correct; DML+CE / oracle-split /
  channel-0=TC are sound; code review of `src/vena/segmentation/` = all invariants satisfied) and surfaced three
  blind spots, each a *confirmed open problem* (verified lit sweep). **(Q1) Soft-map = σ-only, temperature
  DROPPED** — most "softeners" are mandatory infrastructure (avg-pool / DML+CE / K-fold-mean); σ is the one knob
  (and makes oracle≈predicted match); calibration is MEASURED not corrected; ET=TC−NETC is a reported diagnostic
  (B.c banner; B.f-§2 superseded; `SESSIONS.md` Q5). **(Q2) Coupled segmenter+generator OOD failure = first-class
  contribution** — oracle→predicted gap reported PER RING + Ring-B segmenter-error distribution + honest
  localisation/intensity decomposition (A.8-§7 addition; `03_generalization_ood.md` T3.6; `SESSIONS.md` Q6).
  **(Q3) Normal-enhancement = evaluate-only** — new **A.9**: evaluate vessel/sinus/choroid/pituitary enhancement
  fidelity of the synth T1c (reuse Frangi / `venous_atlas_build`), **NO reserved channel** (Moya-Sáez 2023 open
  gap; TA-ViT ignores it; mask-free SynCE evidence; `SESSIONS.md` Q7). **Grid corrected:** this doc's
  `(60,60,40)` / `129024`-"mismatch" claims were BACKWARDS — served grid is `(48,56,48)`, `129024=48×56×48` is
  correct (grid erratum + fixes to Locked-decisions, A.5, A.8-§6, B.d). **Coding-agent notes:** `pool.py:141`
  bare-`assert` (rule-15) → raise `SegDerivationError`; `config.py:228` `avg_pool_stride` docstring still carries
  the grid bug (says 240→60…155→~40; should be 192→48, 224→56, 192→48 after crop); WT→TC docstring drift across
  `derivation/` (rule-17); the injection-sanity panel (task 40) must measure **image-space** residual locality
  *after decode* (latent-locality ≠ image-locality — VAE decoder spread + stochastic encoder weaken P2); verify
  the segmenter K-fold OOF respects the FM **cross-cohort dedup** (else a held-out FM-test patient leaks via its
  duplicate in a segmenter training cohort).
- 2026-07-23 (S4 built + pushed) — `vena.segmentation` library shipped (models/loss/data/metrics; 1313→1453 fast
  tests; commits `eab69b8` + `6f281db`). **BSF-SwinUNETR architecture measured + resolved:** the two SSL ckpts are
  the same Swin-UNETR family at different scales — **UKB = `depths=(2,2,2,2)`** (MONAI default; 125/142 = 0.880),
  **BraTS = `depths=(2,2,6,2)`** (Swin-T, 6 stage-3 blocks; **182/198 = 0.919** once matched — only the 16 SSL
  pretext heads drop). Initial 0.636 was a config mismatch, not a deeper-than-standard arch; per-arm `depths`/
  `num_heads` pinned as constants in `bsf_swinunetr.py` (no `ModelConfig` field). T1ce-stem-removal strategies
  documented (`bsf_stem_t1ce_removal.md`; **drop-slice `[0,1,3]` chosen**). Cross-cohort dedup leakage guard
  implemented (direct + transitive). See [[project_s4_segmenter_library]] + `SESSIONS.md` §S4/§S5.
