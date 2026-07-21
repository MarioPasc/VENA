# Model redesign — Stochastic Latent Bridge for T1c synthesis

> **Purpose of this document.** A self-contained design spec for the next-generation
> model, centered on a **stochastic latent bridge** `z_t1pre → z_t1c`. It is written
> to be handed to an adversarial reviewer (Fable 5) to find **blindspots, conflicting
> gradients, ill-posed steps, and missing/contradicting literature**. It therefore
> states mechanisms precisely, accounts honestly for what each component does *and
> does not* buy, and gates every training commit behind a cheap pre-flight.
>
> **Status:** design only. **Nothing is launched.** Rollout is deliberately staged
> (crawl → walk → run) to isolate gradients and fail cheap.
>
> Companion evidence (numbers are re-derived, frozen): `.claude/notes/article/CONTRIBUTION.md`,
> `00_HUB.md`, `05_vessel_spatial_residual.md`. Prior design iterations preserved in
> the change log (§12). When this drifts from the frozen sweep artifacts, the
> artifacts win.
>
> **⚠️ REVIEWER VERDICT (Fable 5, 2026-07-21) — read [§14](#14-adversarial-review-fable-5-2026-07-21--findings-evidence-required-patches) first.**
> The stochastic latent bridge should **not** be the spine (measured: the residual
> does not localise to enhancement — `loc_ET≈1.17` even for the clean mean latent;
> theory says bridges buy nothing for near-deterministic paired maps; the residual
> target and predicted-mask conditioning are both prior art). But the redesign is
> **not** dead: the VAE floor is *excellent* (`ρ_S≈0.00`, `PSNR_ET≈24 dB` on a clean
> round-trip — the "compression tax" is refuted, ~7 dB of generator headroom), so
> **stay in latent space, predict `z_t1c` directly, and make predicted-mask
> conditioning the primary lever.** Highest-priority caveat: the headline metric
> `ρ_S` swings its entire method-ranking spread (0.00→0.66) from a 99.5-vs-99.95
> normalisation mismatch — **audit that before redesigning to win it.** Full
> evidence, five loginexa experiments, and a ranked patch plan in §14.
>
> **🛠️ BUILD PLAN — [§15](#15-implementation--orchestration-plan--source-of-truth-for-task-generation) is the machine-facing source of truth** for task
> generation: 12 tasks (T-01…T-12) with resume-from checkpoints, hard gates
> (G-NORM/G-VAE/G-SEG/G-SHORTCUT), a DAG of execution waves, and a per-task file
> template. An orchestrating agent should consume §15, materialise one
> `.claude/notes/tasks/T-NN_*.md` per task, and iterate. **Start at T-01** (the ρ_S
> normalisation audit) — it can re-scope the target before any GPU-day is spent.

---

## 1. Thesis

Replace the noise→data generative formulation with a **stochastic bridge between the
pre-contrast and post-contrast latents**, so the model's entire learning target is
the **enhancement residual** `Δ = z_t1c − z_t1pre`. Then add, one at a time and only
if the previous stage holds: (2) a **latent wavelet velocity loss** to bias capacity
toward the high-frequency subbands where enhancement lives, and (3) **zero-init SPADE
conditioning on a *predicted* [WT, NETC] mask** to localize enhancement without a
ground-truth oracle.

The bridge is the load-bearing bet. Its justification is mechanical, not aesthetic:
it removes the **gradient-dilution** that makes v3a converge to a mean brain, and it
does so while advancing the three axes we actually care about — **generalization,
fidelity, few-step speed** — simultaneously rather than trading between them.

---

## 2. Why (compressed — full version in CONTRIBUTION.md)

- The current best model `v3a` (`picasso_s1_v3a_concat_only_fft.yaml`) is, per its own
  header, **"the literal T1C-RFlow architecture"**: channel-concat {T1pre,T2,FLAIR}
  into the trunk conv_in, plain **L1 velocity** CFM, MAISI-MR-pretrained trunk, no
  mask, no SWAN. It wins **nothing fair** — pixel tier (ResViT/pGAN) leads whole-brain
  fidelity; pixel **and** a latent-GAN (C7) lead the vessel/contrast axis ρ_S.
- **Mean-brain diagnosis.** v3a flows **noise → z_t1c** and must re-synthesise the
  whole brain; enhancement is **~0.09 % of the target voxels** → ~0 % of the gradient
  → averaged away. Evidence: v3a PSNR_ET peaks ~epoch 25 then *degrades*; MAE_wt 0.128
  is *worse* than T1C-RFlow (0.123) and the pixel tier (~0.11). The failure is
  objective-level, not optimization-level.
- **Encode pipeline is NOT the culprit (checked 2026-07-21).** Source H5
  `crop_box=[192,224,192]` at native 1 mm → `LATENT_SPATIAL=(48,56,48)` (exact 4×), no
  pre-VAE downsample (sliding-window `[80,80,32]` is OOM-only, shape-exact). Roundtrip
  MAE ~1.3–1.6 % (t1c 0.016). The latent is correct; leverage is in the **generator**.
- **Locked scope (from prior iterations):** deployable-*latent* goal + finding-led
  story; inputs **{T1pre,T2,FLAIR}** corpus-wide (SWAN deferred to a later fine-tune);
  **"VENA" name retired** (no vessel-encoding mechanism survives — new name TBD).

---

## 3. The core idea — the stochastic latent bridge

### 3.1 Formulation and convention alignment

Codebase convention (load-bearing, `.claude/rules`/`reference_timestep_convention`):
`α = timesteps/T ∈ [0,1]` is the **source fraction**, `t_dn = 1−α` the **data
fraction** (`α=0`⇒data, `α=1`⇒source). Noise→data rectified flow uses
`x_α = (1−α)·x_data + α·x_noise`, target velocity `u = x_data − x_noise`.

**Bridge = swap the source `x_noise` for `z_t1pre`.** Define `x0 := z_t1pre` (source),
`x1 := z_t1c` (data). Both are MAISI latents, shape `(4,48,56,48)`.

Deterministic (linear) interpolant:
```
x_α = (1−α)·z_t1c + α·z_t1pre
u   = ẋ  = z_t1c − z_t1pre  = Δ_enh          ← target velocity IS the residual
```
`α=1 ⇒ x=z_t1pre` (start of sampling); `α=0 ⇒ x=z_t1c` (end). Integration direction
and timestep semantics stay identical to the codebase (still integrate α:1→0).

**Stochastic bridge** — add an interpolant-noise term that vanishes at both endpoints:
```
x_α = (1−α)·z_t1c + α·z_t1pre + g(α)·ε ,   ε~N(0,I),   g(0)=g(1)=0
```
canonical choice `g(α)=σ·√(α(1−α))` (Brownian-bridge variance). The exact
training/sampling scheme is an **open implementation choice** to pin in Stage 1:
- **I²SB** (Liu 2023, arXiv:2302.05872) — Schrödinger-bridge posterior; principled,
  proven for I2I.
- **Stochastic Interpolants** (Albergo & Vanden-Eijnden, arXiv:2303.08797) — the
  general drift+score framework the above specialises.
- **BBDM** (Li 2023, arXiv:2205.07680) — Brownian-bridge diffusion in *latent* space
  for I2I; closest precedent to exactly what we propose.

Which one we adopt should follow the **path-geometry pre-flight (PF1)**, not taste.

### 3.2 What the bridge buys — mechanical accounting (no over-claiming)

| Benefit | Mechanism | Confidence |
|---|---|---|
| **Kills gradient dilution** | target = residual Δ, so enhancement gets ~100 % of the gradient instead of ~0.09 % | **high** — this is arithmetic, the dominant win |
| **Data-efficiency / generalization** | anatomy is *given* (start = z_t1pre), not learned from noise → less capacity spent → better OOD behaviour | medium-high (I²SB/BBDM evidence on translation) |
| **Few-step / speed** | z_t1pre ≈ z_t1c ⇒ short transport ⇒ fewer NFE | medium (must verify NFE curve) |
| **Anti-mean-brain by construction** | the mean brain *is* the start point; loss rewards only the transport onto enhancement | high |
| **Aleatoric coverage** | stochastic term + SDE sampler | **LOW — see §3.3** |

### 3.3 The honest limit (a blindspot to state loudly)

**A deterministic bridge still regresses the aleatoric part to the conditional mean.**
We have **one paired `z_t1c` per patient**. For two patients with near-identical
`z_t1pre` but different enhancement, a deterministic map must average → the truly
unpredictable BBB-leakage pattern is still suppressed. The bridge recovers the
**predictable-but-diluted** enhancement (the large chunk currently lost to gradient
dilution), *not* per-patient aleatoric novelty.

The **stochastic term is not a free fix for this.** Its rigorous role (stochastic
interpolants) is to make the learned marginals correct and to enable an SDE sampler to
express the model's *uncertainty*; it does **not** synthesise multi-sample supervision
we do not have. So we should **claim only**: the bridge removes gradient dilution and
recovers predictable enhancement + provides a principled uncertainty channel; we should
**not claim** it captures aleatoric enhancement. (Reviewer: is even this too strong?
See PF2 — if pre/post displacement is dominated by a global normalization offset, the
"residual = enhancement" premise is itself compromised.)

### 3.4 Relation to the pretrained trunk (the primary risk)

The MAISI trunk was pretrained as a **noise→data** rectified flow. In the bridge, at
high α the input is a **structured** latent (`z_t1pre`), not high-entropy noise — so
the trunk's high-α regime is fed off-distribution inputs, and its timestep embedding's
learned "α≈1 ⇒ noisy" semantics no longer hold. FFT fine-tuning must **remap** this.
This is the biggest reason to **start with the bridge alone** (Stage 1) and to run a
**warm-start smoke** before any long run. If the trunk cannot adapt cheaply, the
whole spine is in question and we fall back to the conservative loss-only route.

---

## 4. Supporting mechanisms (added only after the bridge holds)

### 4.1 Latent wavelet velocity loss (Stage 2)
Operationalises "enhancement is high-frequency" **in the latent, without decoding**.
3D Haar-DWT the target and predicted velocity; L1 per subband with weights
`w_LL < w_{LH,HL,HH,...}`. Cheap, faithful, no VAE decode. Caveat (blindspot):
**latent frequency ≠ image frequency** — the VAE mixes bands, so a latent-HH subband
is a *proxy* for image high-freq. If PF4/Study-7 shows image high-freq is largely
absent from the latent, this loss has a low ceiling and we escalate to a **gated
decoded** frequency loss (FFL, Jiang 2021 arXiv:2012.12821; or gradient-difference
loss) on `x̂_1` at high-SNR timesteps. Ramp the weight from 0 to avoid a Stage-1→2
gradient shock (§6).

### 4.2 SPADE on a *predicted* [WT, NETC] mask (Stage 3)
Zero-init spatially-adaptive modulation `h ← γ(m)⊙ĥ + β(m)` of the trunk norm layers,
`γ=1+Δγ, β=Δβ`, Δ-projections zeroed ⇒ step-0 identity ⇒ preserves the bridge
checkpoint byte-identically at introduction. `m` = soft **predicted** [WT, NETC]
probability map from {T1pre,T2,FLAIR} (PF3). Rationale: localization makes the
*predictable* enhancement deterministic ("enhance in the WT∖NETC shell"), so the
median within the mask is enhancement, not zero. **Not** the GT mask (oracle) — that
is a separate upper-bound arm only. Compared against the existing ControlNet-residual
(v3b) as the fusion baseline.

*(The retired region-weighted `-rw` loss is NOT reused — it was net-negative and
needs the GT mask. SPADE conditioning ≠ loss reweighting.)*

---

## 5. Staged rollout (crawl → walk → run)

Each stage warm-starts (`resume_mode: continue`) from the previous stage's checkpoint,
so contributions are attributable and gradients are introduced one at a time.

| Stage | Change (only this) | Hypothesis | Success criterion | Warm-start |
|---|---|---|---|---|
| **0** | **Pre-flights, no training commit** (§7) | the bridge premise is sound & the mask is learnable | PF1–PF4 decision rules met | — |
| **1** | **Bridge only** (CFM velocity on Δ, deterministic first, then add g(α)) | removing gradient dilution beats v3a on ET/ρ_S at equal fidelity | ≥ v3a MS-SSIM_brain **and** PSNR_ET improves; NFE-curve ≤ v3a | from MAISI trunk (baseline) |
| **2** | **+ latent wavelet loss** (ramped) | high-freq subband weighting sharpens enhancement/vessels | ρ_S ↓ and/or PSNR_ET ↑ vs Stage 1, no MS-SSIM_brain regression > MCID | from Stage 1 |
| **3** | **+ SPADE predicted [WT,NETC]** (zero-init, ramped) | localization recovers the predictable enhancement w/o oracle | MAE_wt between v3a (0.128) and v3b (0.096); ρ_S not worse | from Stage 2 |
| **R** | *(reserve)* LADD-style latent-adversarial on `x̂_1` + §6.5 gate | closes residual distributional gap if 1–3 under-deliver | ρ_S ↓ toward pixel tier **without** §6.5 false-enhancement rise | from Stage 3 |
| **U** | *(parallel)* GT-mask upper-bound arm | quantify the oracle ceiling for the paper | reported as bound beside Stage 3 | from Stage 2 |

**Start-slow protocol at every stage:** low LR / long warmup on the newly-unfrozen or
newly-added parameters; ramp any new loss weight from 0; monitor the conflicting-gradient
diagnostics (§6) from step 0; kill early on divergence rather than "training through it".

---

## 6. Conflicting gradients — diagnostics & mitigations

The user's central concern. Sources of conflict: (a) bridge-reformulation vs the trunk's
noise→data prior; (b) CFM velocity vs wavelet high-freq loss; (c) SPADE pathway vs the
trunk backbone.

**Diagnostics (log from step 0):**
- Per-term **gradient norms** on the shared trunk params (CFM vs wavelet vs SPADE-path),
  logged in `configure_gradient_clipping` (grad-accum-safe, per model-coding-standards §7).
- **Task-gradient cosine similarity** between the CFM and wavelet terms (PCGrad-style
  diagnostic, Yu 2020 arXiv:2001.06782). Persistent negative cosine ⇒ genuine conflict.
- Bridge health: does the **near-t1pre (high-α) velocity loss** descend, or does the
  trunk fight the structured input there? (Stratify the loss by α-bin.)

**Mitigations, in order of preference:**
1. **Staging + ramps + zero-init** (already the plan) — introduce one gradient at a time;
   new losses ramp from 0; SPADE starts at identity.
2. **Loss reweighting** — down-weight the wavelet term until cosine ≥ 0.
3. **Gradient surgery (PCGrad)** — project conflicting components out — only if 1–2 fail;
   adds cost and complexity.
4. **Separate the high-freq work into a branch** (a small conditioned detail-head that
   owns the wavelet loss) so the trunk isn't torn — a fallback architecture if the shared
   trunk cannot serve both bulk transport and high-freq sharpening.

---

## 7. Pre-flight experiments (Stage 0) — must pass before the run

All are **cheap** (existing paired latents + the frozen VAE decoder; hours, not GPU-days)
and each carries a **decision rule** that changes the plan.

### PF1 — Is the `z_t1pre → z_t1c` path realizable in the MAISI latent? (path geometry)
**Question:** does the straight-line interpolant stay on the VAE's decodable manifold, or
does it cut through off-manifold garbage (⇒ ill-conditioned velocity field, wrong loss)?
**Method:** for N paired patients, decode `x_α=(1−α)z_t1c+αz_t1pre` at α∈{0,.25,.5,.75,1};
(a) visual morph sanity; (b) **encode–decode consistency** `‖enc(dec(x_α))−x_α‖` vs α
(manifold proximity); (c) a curvature proxy — compare the linear-path length to the
geodesic estimated by short VAE-consistent steps. **Decision:** on-manifold & low
curvature ⇒ linear rectified-flow bridge + L2/L1 velocity is fine. Off-manifold/curved ⇒
prefer a **learned/curved interpolant** (I²SB posterior) or a **reflow** straightening
pass, and reconsider the velocity metric (§9).

### PF2 — Is the residual actually *enhancement*, or a normalization confound? (**critical**)
**Question:** `z_t1c−z_t1pre` = enhancement **only if** T1pre and T1c share an intensity
reference. Phase-1 normalised **per-modality** (`percentile_normalise` per volume) — so a
global pre/post intensity/contrast difference enters the residual as a non-enhancement
offset the model would learn to reproduce. **Method:** displacement statistics of
`Δ=z_t1c−z_t1pre`: spatial map of ‖Δ‖, fraction of ‖Δ‖ mass inside GT-WT vs outside,
per-channel histograms, correlation of the *global* Δ component with the pre/post
percentile-scale ratio. **Decision:** if Δ-mass is **localized to tumour/vascular
regions** ⇒ premise holds. If Δ is **diffuse/global** ⇒ the residual is confounded ⇒
adopt **joint pre/post normalization** before encode (cf. the V4 joint-modality audit,
`project_v3_normalization_audit_2026_06_22`) and/or model the global offset separately.
**This gates the entire bridge — run it first.**

### PF3 — Can [WT, NETC] be predicted from {T1pre, T2, FLAIR}? (predicted-mask feasibility)
**Question:** the user's — is a pre-contrast segmenter good enough to feed SPADE? (ET is
*undefined* pre-contrast; WT from FLAIR/T2 and NETC/necrosis from T1pre-hypo/T2 are
plausible.) **Method:** train/eval a small 3-input segmenter (or adapt downstream_seg
SegResNet to drop T1c) → Dice/AHD for WT and NETC vs GT, per cohort incl. Ring B.
**Decision:** WT Dice ≳ 0.8 and NETC Dice ≳ 0.5 ⇒ Stage 3 predicted-mask viable; below ⇒
Stage 3 uses a coarser single "active-tumour" prior, and the GT-mask arm (U) becomes the
only strong-conditioning result. **Note the safety upside to test:** healthy control ⇒
empty predicted mask ⇒ no enhancement prior ⇒ should pass §6.5.

### PF4 — VAE high-freq floor (Study 7, if not already run)
**Question:** is vessel/enhancement high-freq even *in* the latent, or destroyed at 4×
encode? **Method:** encode→decode **real T1c**; measure a **high-freq/vessel** recon
metric (Frangi conspicuity synth-vs-real, ρ_S of the recon residual), not just MAE.
**Decision:** high floor ⇒ latent wavelet loss (4.1) has headroom, no VAE change. Low
floor ⇒ the latent wavelet loss is capped, escalate to decoded frequency loss and
reconsider the (separately-gated) VAE-resolution branch. Prior: the ρ_S "bright
structures" (sinuses, choroid, dural) are mostly >4 mm ⇒ likely partly preserved ⇒
suspect the FM, not the VAE — but **measure**.

---

## 8. Risk / blindspot register (for the adversarial reviewer)

| # | Risk / blindspot | Detect via | Mitigation / fallback |
|---|---|---|---|
| B1 | **Normalization confound** — residual ≠ enhancement (per-modality norm) | **PF2** | joint pre/post norm; model global offset separately |
| B2 | **Trunk repurposing** — noise→data prior fights the bridge's structured high-α input | Stage-1 α-stratified loss; warm-start smoke | conservative loss-only route (keep noise→post) |
| B3 | **Single-sample aleatoric limit** — deterministic bridge averages unpredictable enhancement | held-out variance vs GT; §6.5 | claim only predictable-enhancement recovery; stochastic term + SDE for uncertainty, not novelty |
| B4 | **Stochastic term over-sold** — g(α)·ε may add noise without diversity | ablate σ; sample multiple draws per patient, measure spread | present as uncertainty channel, not multi-modal generator |
| B5 | **latent-freq ≠ image-freq** — wavelet-latent loss a weak proxy | PF4 | gated decoded frequency loss |
| B6 | **Path off-manifold** — linear interpolant decodes to garbage midpoints | **PF1** | curved/learned interpolant; reflow; metric change |
| B7 | **SPADE surgery on pretrained GroupNorm** — modifying norm affine may destabilise a pretrained trunk | Stage-3 smoke; zero-init check | keep ControlNet-residual fallback (v3b path exists) |
| B8 | **Predicted-mask quality** — poor NETC Dice ⇒ mis-localized enhancement | **PF3** | coarser single prior; GT-mask arm as the strong result |
| B9 | **Few-step + EMA still averages** — sampler-level mode collapse independent of objective | NFE curve; EMA vs non-EMA sampling | SDE/stochastic sampler; report per-NFE |
| B10 | **Conflicting gradients** — wavelet/SPADE tear the shared trunk | §6 cosine/grad-norm diagnostics | ramps, reweight, PCGrad, detail-branch |
| B11 | **Bridge helps fidelity but not ρ_S** — sharper mean is still a mean | ρ_S per stage vs pixel tier | escalate to reserve adversarial (R) |
| B12 | **Generalization claim unverified** — bridge may over-fit near-distribution | Ring-B (OOD) eval each stage | keep OOD in the loop, not just at the end |

---

## 9. Open loss-type question (tied to PF1/PF2)

The residual Δ is (expected) **sparse and high-magnitude in tumour/vessels, near-zero
elsewhere**. The velocity metric should follow the geometry PF1/PF2 reveal:
- **L1** (median) — robust, but suppresses the sparse high-magnitude signal we want.
- **L2** (mean) — preserves magnitude (good now that Δ *is* the signal), but blurs; also
  most sensitive to the B1 normalization offset.
- **Huber / magnitude-weighted** — compromise; up-weight where ‖Δ‖ is large.
- **Manifold-aware** — if PF1 shows curvature, an L2-in-raw-latent target is wrong; a
  metric respecting the VAE geometry (or a decoded-space term) may be needed.
Decide empirically in Stage 1 against PF1/PF2; do **not** default to v3a's L1.

---

## 10. Explicitly out of scope / deferred (so the reviewer doesn't chase them)

- **SWAN input / vessel prior** — deferred to a later fine-tune from the trained
  checkpoint on the SWAN-bearing subset. Not in the main model.
- **VAE-resolution swap** — separately gated on PF4/Study-7; a new VAE would forfeit
  MAISI trunk pretraining (new latent space ⇒ generator from scratch). Prefer an
  image-space residual refiner if ever justified.
- **Adversarial term** — reserve escalation (Stage R) behind the non-adversarial stages
  and a §6.5 false-enhancement gate. Not a Stage-1 lever.
- **Region-weighted `-rw` loss** — retired (net-negative, needs GT mask).
- **AGFN / GFlowNets** — off-target (discrete reward-proportional sampling; not 3D image
  latents). Not pursued.
- **Name** — "VENA" retired; new name TBD.

---

## 11. Literature anchors (verify status marked — Fable 5 to confirm/deny)

Bridge / interpolant (**core**):
- **I²SB: Image-to-Image Schrödinger Bridge**, Liu et al., ICML 2023 — arXiv:2302.05872. *[confident]*
- **BBDM: Brownian Bridge Diffusion for I2I** (latent), Li et al., CVPR 2023 — arXiv:2205.07680. *[confident]* — closest precedent.
- **Stochastic Interpolants**, Albergo, Boffi, Vanden-Eijnden 2023 — arXiv:2303.08797. *[confident]*
- **Rectified Flow**, Liu, Gong, Liu 2022 — arXiv:2209.03003. *[confident]*
- **Flow Matching**, Lipman et al. 2023 — arXiv:2210.02747. *[confident]*

High-frequency / wavelet:
- **Focal Frequency Loss**, Jiang et al., ICCV 2021 — arXiv:2012.12821. *[confident]*
- **WaveDiff: Wavelet Diffusion Models**, Phung et al., CVPR 2023 — arXiv:2211.16152. *[confident]*
- **3D Wavelet Flow Matching (WFM)** — arXiv:2604.21146. *[from web search, VERIFY]*

Conditioning:
- **SPADE**, Park et al., CVPR 2019 — arXiv:1903.07291. *[confident]*
- **FiLM**, Perez et al., AAAI 2018 — arXiv:1709.07871. *[confident]*

Multi-task gradients:
- **PCGrad (Gradient Surgery)**, Yu et al., NeurIPS 2020 — arXiv:2001.06782. *[confident]*

Medical baseline / reserve:
- **T1C-RFlow**, Eidex et al. 2025 — arXiv:2509.24194 (same MAISI VAE; the model we improve). *[from web search, VERIFY]*
- **Adversarial Flow Models / CAFM / LADD** — arXiv:2511.22475 / 2604.11521 / 2403.12015 (reserve escalation). *[from web search, VERIFY the future-dated ones]*

*Reviewer task:* confirm the confident set; verify or replace the web-search set; and
critically — **find bridge/latent-translation medical-imaging papers we missed** that
either support or refute the "residual-target bridge beats noise→post CFM for CE-MRI"
claim.

---

## 12. Reviewer questions (explicit asks for Fable 5)

1. Is the **stochastic-term claim (§3.3)** correctly bounded, or still over-stated given
   single-sample supervision?
2. Does **PF2 (normalization confound)** kill or merely complicate the bridge? Is joint
   pre/post normalization the right fix, or does it break the frozen encode contract?
3. Is the **staging order** (bridge → wavelet → SPADE) the right gradient-isolation order,
   or should SPADE precede wavelet (localize before sharpen)?
4. Which **stochastic bridge scheme** (I²SB vs stochastic-interpolant SDE vs BBDM) best
   fits a pretrained *noise→data* trunk with minimal repurposing (B2)?
5. Are there **conflicting-gradient** pairs we haven't listed, e.g. between the bridge's
   near-t1pre regime and the wavelet loss's high-α behaviour?
6. Is the **loss-type** (§9) decision framed correctly, or is a decoded-space term
   unavoidable from the start?
7. **Missing literature** that supports or refutes any load-bearing claim.

---

## 13. Change log
- 2026-07-21 (iter 1) — decisions D1–D7 locked; adversarial-vs-frequency literature;
  VAE-resolution gated on Study 7.
- 2026-07-21 (iter 2) — encode cleared (not under-feeding); high-freq-residual framework;
  bridge proposed as spine.
- 2026-07-21 (iter 3) — **document restructured around the stochastic latent bridge**;
  crawl-walk-run staging (§5); pre-flights PF1–PF4 (§7) incl. the normalization-confound
  gate; conflicting-gradient diagnostics (§6); risk/blindspot register (§8); literature
  anchors with verify-status (§11); reviewer questions (§12). Written for adversarial
  review by Fable 5.
- 2026-07-21 (iter 4, **Fable 5 adversarial review**) — §14 added. Ran PF2/PF4-style
  checks on loginexa (five scripts). Findings: bridge residual does not localise to
  enhancement (loc_ET≈1.17, mean latent); VAE floor is excellent (ρ_S≈0, PSNR_ET≈24 —
  refutes the compression-tax root cause); MAISI encoder is stochastic (cache stores
  posterior samples, `z_mu` discarded); ρ_S swings 0.00→0.66 from a 99.5/99.95 norm
  mismatch (metric-validity flag); residual-target + predicted-mask conditioning are
  prior art (Moya-Sáez 2025, Eidex TA-ViT 2025); aleatoric ceiling 8–20 %. Verdict:
  drop the stochastic-bridge spine, predict z_t1c directly, promote predicted-mask
  conditioning to primary, audit ρ_S normalisation first. All §11 citations verified
  real.
- 2026-07-21 (iter 5) — **§15 orchestration plan added** (source of truth for task
  generation). Decomposed the §14 model into 12 tasks (T-01…T-12) across 5 waves,
  with a module map, normative checkpoint lineage (T-06 warm-starts from v3b
  `ema_best.ckpt`), a hard-gate registry (G-NORM/G-VAE/G-SEG/G-SHORTCUT), a
  per-task-file template, and naming/decision.json conventions. Consuming agent
  materialises one `.claude/notes/tasks/T-NN_*.md` per task; T-01 (ρ_S norm audit)
  runs first.
- 2026-07-21 (iter 6) — **§16 added: segmenter-centred `Soft[WT,NETC]` pivot.**
  Session decisions with the user (chat, not adversarial review): (a) the
  deployable conditioning signal is a **soft 2-channel `[WT, NETC]`** prior (ET is
  undefined pre-contrast — see §16.1), replacing §15's `[NETC,ED,ET]`; (b) **the
  segmenter is now the methodological centre** — establish `Soft[WT,NETC]`, design/
  train/eval it, visualise, then run the same regime with GT masks as the ceiling;
  (c) **ControlNet stays primary, SPADE is the T-07 ablation** (§16.2 rationale);
  (d) the current v3b uses **GT `[NETC,ED,ET]` via ControlNet** — NOT a matched
  ceiling for a `[WT,NETC]` deployable arm, so the GT ceiling must be **re-run with
  GT-`[WT,NETC]`** (new **T-13**, §16.3); (e) the **`-rw` region-loss arm is dropped
  from the headline** — "VENA (ours)" in the fidelity table = the rw variant, retired;
  (f) new **false-enhancement (FP) safety study** across all 16 methods to test
  whether the ρ_S-winning latent-GAN C7 hallucinates enhancement (new **T-14**,
  §16.5). T-01 launched (Opus, Picasso) at the same time — it gates everything.
- 2026-07-21 (iter 7) — **§17 added: T-01 audit RESULT.** The ρ_S vessel headline is
  **largely a 99.5/99.95 normalisation artifact**; matched-norm collapses the
  latent-diffusion ranking and REFUTES "latent worse than identity / VENA = fix"
  (CONTRIBUTION suspended pending T-05). Canonical percentile = **99.95**. The audit
  boolean `latent_worse_than_identity_survives` is a bug. G-NORM fallback fired:
  the model retraining is now justified on **fidelity** (PSNR_ET / MAE_wt), not the
  dead vessel-fix story. Memory: `project_rho_s_norm_audit_2026_07_21`.

---

# 14. Adversarial review (Fable 5, 2026-07-21) — findings, evidence, required patches

> Requested in §12. Backed by (a) a verified literature sweep (§14.10), (b) the
> frozen VENA evidence (`CONTRIBUTION.md`, `05_vessel`, the 2026-06-22
> tumour-failure diagnosis, `2026-06-22_v3_normalization_decision.md`), and
> (c) **five experiments run on loginexa** (UCSF-PDGM test split, frozen MAISI VAE;
> scripts `pf_bridge.py` / `pf_mean.py` / `pf_decode_check.py` / `pf_norm_check.py`
> in the session scratchpad). Numbers are quoted inline. Where this supersedes the
> body, it is because it *measures* what §7 only proposed to measure. **Two of my own
> initial hypotheses were falsified by measurement (the VAE is NOT the bottleneck;
> sampling noise is NOT ~95 %); the corrected findings below are what the data
> supports.**

## 14.0 VERDICT

**Do not launch the stochastic latent bridge as the spine. The redesign's
*instinct* (predict enhancement) is right and the *layer* (MAISI latent) is
vindicated — but the *mechanism* (residual-target Schrödinger/Brownian bridge) is
wrong for this task, and the *lever* it neglects (mask conditioning) is the only
one with evidence.** In one line: **the VAE is not the problem, the generator is,
and the bridge does not fix the generator.**

The five pillars (each tagged with how it is supported):

1. **[MEASURED — the good news] The VAE floor is excellent; the "compression tax"
   root cause is refuted.** A clean encode→decode of real T1c gives
   **ρ_S ≈ 0.00** (pixel-tier level; cf. v3a 0.31, C4/C5/C6 0.46–0.73),
   **PSNR_ET ≈ 24 dB**, PSNR_brain ≈ 27 dB (≈ 29.5 at matched 99.95). The MAISI VAE
   reconstructs the bright vascular/enhancing structure faithfully — errors are
   *uncorrelated* with brightness. So the vessel-misplacement of the latent
   generators (ρ_S 0.46–0.73) is a **generator** failure, not a VAE-compression one.
   There is **~7 dB PSNR_ET and ~0.3 ρ_S of headroom** the current generators are
   leaving on the table. *This refutes `CONTRIBUTION.md` §1's stated root cause and
   answers the never-run Study 7.* → **stay in latent space.**
2. **[MEASURED — the bad news for the bridge] The residual does not concentrate on
   enhancement, even with all confounds removed.** For the *deterministic mean*
   residual `Δμ = μ_t1c − μ_t1pre`: `loc_ET = ⟨‖Δμ‖⟩_ET/bg = 1.17`, `loc_WT = 1.09`
   — the enhancement region is only ~17 % above background; the residual is a large,
   *diffuse* whole-brain displacement (‖Δμ‖_bg = 1.78). Joint-normalised and sampled
   variants land in the same band (loc_ET 1.12–1.23). **The §3.2 claim ("target = Δ
   ⇒ ~100 % of the gradient on enhancement, arithmetic, high-confidence") is
   empirically false.** The bridge's central mechanical benefit does not exist here.
3. **[THEORY] Bridges provably buy nothing in this regime.** *Demystifying
   Transition Matching* (AISTATS 2026, 2510.17991): the bridge>flow-matching
   advantage requires non-negligible target-conditional variance; for
   near-deterministic paired maps it vanishes. *Regression is All You Need* (IEEE
   JSTSP 2025, 2505.02048): multi-step stochastic sampling gives no PSNR/SSIM gain
   over one-step regression in paired medical translation. Warm-starting a
   noise→data trunk into a bridge is literature-confirmed unstable (2505.24406,
   2508.18095, 2412.20506) — B2 is real, not a formality.
4. **[NOVELTY] Both ideas the redesign leans on are prior art on this exact task.**
   Residual/subtraction target: Moya-Sáez *QIMS* 2025, Chen *npj Digit. Med.* 2022.
   Predicted-mask adaLN-zero conditioning (≡ Stage-3 zero-init SPADE): Eidex
   *Med. Phys.* 2025 (TA-ViT) **on UCSF-PDGM**, absent from VENA's competitor set.
5. **[MEASURED — audit the motivation] The headline metric ρ_S is confounded by
   intensity normalisation.** A 99.5-vs-99.95 percentile mismatch alone swings the
   VAE-round-trip ρ_S from **0.001 to 0.665** — the full width of the 16-method
   ranking. The encoder wrote latents at **99.95**; the metric path defaults to
   **99.5**. Before redesigning a model to "win" ρ_S, re-derive the ranking with
   synth and real forced to one percentile (§14.3); the "latent-worse-than-identity"
   premise may partly evaporate.

**Recommended pivot (detail in §14.9):** predict `z_t1c` directly (not the
residual), with **predicted-[WT,NETC]-mask conditioning as the primary lever**
(proven: v3b > v3a by ~4 dB PSNR_ET; TA-ViT; `CONTRIBUTION` §6 = "the one shot at a
fair model-win"), benchmarked against **v3b**. Keep at most the *deterministic*
informed-prior start (source = z_t1pre, WFM/LBM style) as a minor speed arm; drop
the stochastic term, the Schrödinger machinery, and the residual-target framing.

---

## 14.1 The VAE floor — measured, and it is not the bottleneck (§14.0-1)

`ρ_S(|recon−real|, real T1c)` over C-noT = brain∖dilate(WT,5) — the *identical*
statistic used to rank models in `05_vessel`. Clean fresh encode→decode, 20 UCSF
test patients (`pf_mean.py`, self-consistent 99.5):

| decode of real-T1c latent | ρ_S | PSNR_brain | PSNR_ET |
|---|---|---|---|
| **mean μ_t1c** | **−0.061** | 27.24 | 23.99 |
| **sample μ+εσ** | **0.002** | 27.14 | 23.98 |
| ref: pixel tier | 0.03–0.09 | — | — |
| ref: v3a / identity / C4–C6 | 0.31 / 0.35 / 0.46–0.73 | — | — |

Two consequences: (i) the "4× MAISI compression costs contrast-structure fidelity"
claim (`CONTRIBUTION.md` §1, and Study 7 owed) is **false** — the VAE round-trip is
pixel-tier on ρ_S and reaches PSNR_ET 24 dB vs the best generator's 17.3. The
latent tier's vessel failure is fixable *within* the latent, by a better generator.
(ii) The sampling noise barely touches the decoded image (mean vs sample differ
<0.1 dB) — the decoder absorbs the σ=0.36 posterior noise. [Cached-latent decode
reconciliation: §14.3.]

## 14.2 Why the bridge's residual target is ill-posed here (§14.0-2)

`MaisiEncoder.encode → encode_stage_2_inputs → z = z_mu + ε·z_sigma` (verified in
MONAI source): the cache stores one **posterior sample** per modality; `z_mu` is
discarded, and the FM target `x1 = latents/t1c` (`data.py:201`) is that noisy draw.
Decomposition on the test split (`pf_mean.py`, N=20):

| quantity | value | reading |
|---|---|---|
| posterior ⟨σ⟩ | 0.364 | per-element sampling std |
| sampling floor ⟨‖s1−s2‖⟩ (two draws, same μ,σ) | 1.206 | the noise part of the cached residual |
| deterministic signal ⟨‖Δμ‖⟩ (bg) | 1.785 | mean-to-mean residual magnitude |
| **signal/noise (magnitude)** | **1.48** | signal survives, but noise ≈ 30 % of the *sampled*-residual energy |
| `loc_ET(Δμ)` mean residual | **1.168** | enhancement only 17 % above background |
| `loc_WT(Δμ)` | 1.093 | WT ≈ background |
| `loc_ET` cached-sample / joint-norm-sample | 1.118 / 1.178 | same weak band under every normalisation |
| `frac_DC(Δμ)` | 0.031 | offset is spatially-varying, not a pure global DC |

The residual is a **large diffuse whole-brain displacement** (‖Δμ‖≈1.78 in healthy
tissue — largely the per-modality-normalisation scale mismatch plus genuine global
T1 changes) with only a **~17 % enhancement bump**. So "target = Δ" does *not*
concentrate the loss on enhancement (contra §3.2); it concentrates it on the diffuse
difference. Reconciled with §14.1: the VAE faithfully carries enhancement in *both*
z_t1pre and z_t1c, but their *difference* is enhancement-poor. **Predicting z_t1c
directly keeps the enhancement salient in the target; predicting the residual buries
it.** This is the opposite of the redesign's premise.

Sampling-noise verdict: real but **secondary** — ~30 % of the latent-residual energy,
~0 dB of decoded-image impact. Mean-caching (§14.9-P1) cleans a latent-space loss
but is not a quality lever on its own. (Corrects my initial "~95 % noise" reading.)

## 14.3 The ρ_S metric is confounded by intensity normalisation (**high priority**)

Decoding the *cached* z_t1c and scoring it against real T1c at two percentiles
(`pf_norm_check.py`, 6 patients) isolates the cause of the §14.1 discrepancy:

| cached-latent decode scored vs | PSNR_brain | ρ_S |
|---|---|---|
| real @ **99.5** (metric-path default) | 20.9 | **+0.665** |
| real @ **99.95** (the encoder's actual percentile) | **29.5** | **+0.001** |

Two conclusions:

1. **The cached latents are NOT degraded** (I retract that alarm). Scored at the
   *same* percentile the encoder used, they decode to PSNR 29.5 / ρ_S 0.001 —
   pixel-tier. The earlier "ρ_S 0.66" was a normalisation mismatch in my own
   comparison, not a data defect. The encoder's `encode_runtime_attrs_json` shows
   `percentile_upper = 99.95` and all-`full` encode (no sliding).
2. **ρ_S is acutely sensitive to a global intensity-scale mismatch.** A 99.5-vs-
   99.95 percentile difference — a modest global rescale — swings ρ_S from 0.001
   to 0.665, i.e. **the entire spread of the article's 16-method ranking (pixel
   0.03 → C4 0.73).** Mechanistically: if synth ≈ real·s for a scalar s≠1, then
   |synth−real| ≈ |1−s|·real is monotone in intensity, so ρ_S(|r|, real) → ±1
   independent of any structural/vessel fidelity.

**Implication for the redesign's motivation (must audit before building on it).**
The encoder wrote latents at 99.95; the image-metric path (`fm/eval/exhaustive.py`
default `percentile_upper=99.5`, docstring *claims* "matches the encoder" — a
code/doc drift) and the frozen ρ_S sweep (`vena/validation/io.py`, harmonises on
brain-`p99.5`) reference **99.5**. So VENA's decoded synth (99.95 space) is being
compared against a 99.5 reference. The latent competitors (C4–C7, same MAISI cache)
plausibly share this; the pixel competitors (trained/emitting at 99.5) and the
identity baseline do not — which is *exactly* the shape needed to manufacture the
headline "latent methods misplace contrast, worse than identity." I cannot fully
disentangle it here (v3a's reported ρ_S 0.31 < the 0.66 full-mismatch ⇒ the sweep
harmonises *partially*, so the reported gap is *some* real fidelity + *some* scale
artifact), but the artifact's magnitude is large enough that **the ρ_S ranking must
be re-derived with synth and real forced to an identical percentile before any
model is redesigned to "win" it.** If the gap shrinks under matched normalisation,
the redesign's premise ("latent flow misplaces contrast onto vessels") weakens and
VENA may already be competitive. This is the single highest-leverage verification
in this review — cheaper than any training run.

*(Note: `05_vessel` §7's shuffle-null defends against "positive ρ_S is trivially
expected" but NOT against a per-method scale offset — the shuffle permutes
intensities *within* a fixed prediction, so it cannot see a synth-vs-real global
rescale. The normalisation confound is orthogonal to the shuffle-null control.)*

## 14.4 Normalisation confound (PF2/B1) — real, mostly moot once the residual is dropped

Current cache = **V0 per-modality** (T1pre/T1c scaled independently; V4 joint
recommended 2026-06-22, **never adopted**). Image-space effect is large (normal
tissue 0.674→0.308 across pre/post; ET |Δ|=0.238 < background 0.384 because the
99.5 %ile clip crushes the gadolinium tail — 2026-06-22 diagnosis). It is the main
driver of the diffuse latent residual (§14.2). **But if the residual/bridge framing
is dropped and z_t1c is predicted directly, the confound stops being load-bearing**
(the model just reproduces whatever normalisation the target has). Joint norm is
only mandatory *if* a residual/enhancement-map loss is kept; then validate it
localises Δμ before spending GPU-days. Cost: ~3 dB whole-vol, KL≈2 on T2/FLAIR,
25 h + 150 GB.

## 14.5 Why the bridge is unmotivated (theory + literature)

- **2510.17991 (AISTATS 2026)** — bridge/TM > FM iff target-conditional variance is
  non-negligible; near-deterministic paired maps get no gain. VENA is in the no-gain
  regime (§3.3). Formalises the proposal's own honesty.
- **2505.02048 (IEEE JSTSP 2025)** — paired MRI/MR-CT: multi-step diffusion gives no
  systematic PSNR/SSIM gain over 1-step regression. Consistent with pixel regressors
  (pGAN/ResViT) beating every latent generator in `CONTRIBUTION.md`.
- **2505.24406 / 2508.18095 / 2412.20506** — a pretrained noise→data backbone needs
  endpoint-distribution reparameterisation to become a bridge; B2's "FFT will remap"
  is optimistic.
- **What *does* transfer** — informed-prior *plain* conditional flow: WFM (MIDL 2026,
  2604.21146; 3D BraTS, 26.8 dB, within 1–2 dB of diffusion at 250–1000× speed) and
  LBM (ICCV 2025, 2503.07535). Keep the informed-prior *start*, drop the stochastic
  bridge.

## 14.6 Novelty collisions (cite / out-compete, do not re-derive)

- Residual/subtraction target — **Moya-Sáez, *QIMS* 15(1) 2025** (synthesise the
  T1c−T1 map, add back) and **Chen, *npj Digit. Med.* 5:149 2022**. §1's "entire
  target is the enhancement residual" is not new.
- Predicted-mask adaLN-zero conditioning — **Eidex, *Med. Phys.* 52(4) 2025
  (TA-ViT)**, on **UCSF-PDGM**, same group as C5-T1C-RFlow; architecturally the
  Stage-3 SPADE. **Add to the 16-method registry** before claiming the predicted-mask
  arm.
- The genuinely novel piece is the **vessel-resolved evaluation** — but that is an
  *evaluation* contribution and already the paper's headline; the redesign adds no
  new *model* novelty on top.

## 14.7 The aleatoric ceiling caps the enhancement objective (any model)

- Azizova, *J. Neuroimaging* 2024 — radiologists reach **86–92 %** enhancement-quality
  accuracy from pre-contrast ⇒ ~8–14 % irreducible.
- Zheng, *Nat. Commun.* 2026 — BBB-status AUC **81.3 %** from non-contrast ⇒ ~19 %
  aleatoric floor (binary; voxel-intensity harder).
- Kleesiek, *Invest. Radiol.* 2019 — Bayesian aleatoric uncertainty clusters *on the
  enhancing rim*, exactly where the model must work.
- Fayolle, MICCAI-W 2025 — single-session synthesis misses *new/changed* enhancement
  (mean-regression); inherent to single-session paired synthesis, not a v3a artifact.

**Framing rule:** claim only "recovery of *predictable* enhancement"; never
per-patient novelty. For uncertainty, emit an explicit predictive-uncertainty map
(Piening, ISBI 2025) rather than dressing up the stochastic-bridge term as one.

## 14.8 Mechanism / math defects in the current spec

- **Stochastic term (B4) is a net liability** — with single-sample supervision the
  Brownian drift adds a denoising task (remove `g(α)·ε`) with zero diversity payoff;
  it *reintroduces* the burden the bridge meant to remove. Go deterministic.
- **SPADE "byte-identical" claim (B7) is wrong** — replacing the trunk GroupNorm
  affine with `γ=1+Δγ, β=Δβ` (Δ=0) discards the pretrained `(γ,β)_pre ≠ (1,0)`. Fold
  the pretrained affine into the SPADE base, or insert SPADE post-norm; else the
  warm-start silently perturbs the checkpoint.
- **Latent-wavelet loss has a Nyquist ceiling (B5)** — 1 latent voxel = 4 image mm;
  finest latent subband ≈ 4–8 mm. Vessels (1–3 mm) are below latent Nyquist: no
  latent loss can add vessel-scale detail (this is *consistent* with the VAE floor —
  the VAE preserves the >4 mm bright structures that dominate ρ_S, which is why
  ρ_S≈0, but 1–3 mm vessels are genuinely gone). Use it only for the cm-scale rim.
- **ρ_S is a metric trap** — it penalises error on the bright voxels an enhancement
  model must modify; v3a's low ρ_S is a *smoothness* artifact (it does not try). Do
  **not** make "ρ_S ↓ AND PSNR_ET ↑" a joint gate; optimise PSNR_ET + Frangi
  conspicuity (T5.4), report ρ_S as companion. (See also §14.3 — ρ_S is scale-fragile.)
- **Staging is a single point of failure** — the most speculative component gates
  Stages 2–3, and Stages 1–2 benchmark only vs v3a (weak; beat v3b). Reorder.
- **CFG mismatch** — `conditioning_dropout_p` drops conditioning channels; a bridge's
  `z_t1pre` initial value is not a droppable channel.

## 14.9 Constructive patch plan (ranked)

- **P0 — do NOT pivot to image space.** §14.1 settles it: the VAE floor is excellent
  (ρ_S≈0, PSNR_ET≈24). The generator is the bottleneck, in latent space, with ~7 dB
  of headroom. (This flips the "image-space refiner" fallback the body floats.)
- **P1 — drop the residual/bridge spine; predict z_t1c directly with strong
  conditioning.** The residual buries enhancement (§14.2); the direct target keeps it
  salient. Make **predicted-[WT,NETC]-mask conditioning the primary lever (Stage 1)**
  — proven (v3b +4 dB PSNR_ET, TA-ViT, `CONTRIBUTION` §6). Fix SPADE init (§14.8) or
  reuse the v3b ControlNet residual path. Benchmark vs **v3b**.
- **P2 — optional deterministic informed-prior start** (source = z_t1pre, plain
  rectified flow, no `g(α)·ε`; WFM/LBM). A *speed* arm at most; keep a from-scratch
  fallback for B2. Not the load-bearing fix.
- **P3 — optional target-cleanup re-encode** *only if* a latent residual/enh-map loss
  is retained: cache `z_mu` (not a sample) + joint pre/post norm, validated on the
  §14.2 localisation test first. ~25 h / 150 GB. Removes the ~30 % noise + the diffuse
  offset from a residual target; irrelevant if predicting z_t1c directly.
- **P4 — evaluation hygiene.** Audit the ρ_S normalisation (§14.3) FIRST. Add TA-ViT
  to competitors; cite Moya-Sáez for the residual target; report PSNR_ET + Frangi
  conspicuity as enhancement metrics, ρ_S as companion; correct `CONTRIBUTION.md` §1
  (VAE floor is high, not the tax).
- **P5 — cap claims to the aleatoric ceiling** (§14.7); explicit uncertainty map if
  wanted.

## 14.10 Literature appendix (all citations verified to resolve; none hallucinated)

Bridges/interpolants: I²SB (2302.05872) · BBDM (2205.07680; *image-space* — §11's
"(latent)" tag is wrong) · Stochastic Interpolants (2303.08797) · DDBM (2309.16948;
scratch, FID) · LBM (2503.07535, ICCV 2025) · IRBridge (2505.24406, ICML 2025) ·
Diffusion-Bridge-or-FM (2509.24531) · **Demystifying Transition Matching
(2510.17991, AISTATS 2026)**. Medical bridges: SynDiff (2207.08208, TMI 2023 —
*unsupervised*) · SelfRDB (2405.06789) · MR-CT SB (2404.11741, Med. Phys. 2025).
Refutation for paired: **Regression-Is-All-You-Need (2505.02048, IEEE JSTSP 2025)**.
Frequency: FFL (2012.12821) · WaveDiff (2211.16152) · **WFM (2604.21146, MIDL 2026 —
informed prior)**. Conditioning: SPADE (1903.07291) · FiLM (1709.07871) · **Eidex
TA-ViT (Med. Phys. 2025, mp.17600 — predicted-mask on UCSF-PDGM)**. Multi-task:
PCGrad (2001.06782). Virtual-Gd ceiling: Kleesiek 2019 · Preetha 2021 · Ammari 2022 ·
**Azizova 2024 (86–92 %)** · Wamelink 2024 · **Zheng 2026 (BBB AUC 81 %)** · Fayolle
2025 · **Moya-Sáez 2025 (subtraction target)** · Chen 2022. T1C-RFlow = 2509.24194
(*Biomed. Phys. Eng. Express*, journal).

---

# 15. Implementation & orchestration plan — source of truth for task generation

> **Audience: the orchestrating agent.** This section is the machine-facing
> contract. §14 is *why*; §15 is *what to build, in what order, gated on what*.
> The model of §1–13 is superseded by §14's verdict: **no bridge. Predict `z_t1c`
> directly; the deployable-predicted-mask conditioning is the primary lever.**
> Everything below decomposes that into independently-buildable tasks.

## 15.0 Contract for the consuming agent (read first)

1. **One Task (`T-NN`) → one task file → one (or a few) subagents.** For each Task
   in §15.4, materialise `.claude/notes/tasks/T-NN_<slug>.md` using the **task-file
   template** at the end of §15.0, then implement it. Keep the task file in sync as
   you iterate (status, blockers, produced artifacts, actual checkpoint paths).
2. **Gates are hard (§15.3).** A Task marked `gate: G-XX` must not launch its long
   run until the gate's `decision.json` exists and its decision rule passes. If a
   gate fails, follow its *fallback*, do not train through it.
3. **Checkpoint lineage is normative (§15.2).** Warm-start only via
   `run.resume_from = <absolute Picasso path to ema_best.ckpt>` (the WARM_START code
   path in `engine.py::_classify_resume_from` — creates a fresh run dir, copies
   weights, never touches the source). Pin the exact path in the task file after
   resolving it with `ssh picasso ls -d`.
4. **Follow the project rules** (`.claude/rules/*`): routine pattern (one YAML arg,
   frozen Pydantic config, `Engine.run()->Path`, no import-time side effects),
   `decision.json` schema bumps, coding-standards, `vena.common` import discipline,
   `MultiCohortLatentDataModule`-only data path. New library code in `src/vena/`,
   thin engines in `routines/`. Tests co-located, markered.
5. **Verify every number.** Training runs report exhaustive-val PSNR_ET/ρ_S as the
   authoritative signal, not train loss. Do not trust a plausible artifact — decode
   a sample and eyeball it (the empty-CSV / silent-WARNING traps in the server3 /
   loginexa skills are real).
6. **Platform ladder:** iterate on **server3 / loginexa** (fast, no queue) → validate
   the smoke → submit the real run to **Picasso** A100 via `picasso-sbatch`. Never
   submit an unvalidated config to the A100 queue.
7. **Scope discipline.** Build Wave 0–2 first (the deployable model). Waves 3–4 are
   ablations/extensions — do not start them until T-06 has a first exhaustive-val
   curve.

**Task-file template** (each `T-NN_*.md` must contain):
```
# T-NN — <name>
status: todo|in-progress|blocked|done      owner-wave: <0-4>
depends-on: [T-..]   gate: G-XX|none        resume-from: <ckpt path|none>
## Purpose            (1-2 sentences: what this unblocks)
## Inputs             (data H5s, checkpoints, decision.json artifacts — exact paths)
## Proposed architecture / approach   (concrete: layers, channels, shapes, formulae)
## Files to create / modify           (repo paths, following the routine pattern)
## Config / YAML deltas               (keys vs the parent config it copies)
## Acceptance criteria                (numeric where possible)
## Gate decision rule                 (what must hold to unblock downstream; fallback)
## Produces                           (artifact paths + decision.json fields)
## Cost / platform                    (GPU-h, node type)
## Variants to try ("modules")        (enumerate the arms + how to pick the winner)
```

## 15.1 Module map (the model, decomposed)

| Task | Module | Type | Status | Depends | Gate | Resume-from |
|---|---|---|---|---|---|---|
| **T-01** | ρ_S normalisation audit + canonical percentile | preflight/eval | todo | — | **produces G-NORM** | — |
| **T-02** | VAE floor (Study 7) formalised | preflight/report | ~done¹ | — | produces G-VAE | — |
| **T-03** | Pre-contrast tumour segmenter (train) | library+routine | todo | — | **produces G-SEG (PF3)** | — |
| **T-04** | Predicted-mask cache (`tumor_latent_pred`) | pipeline | todo | T-03 | G-SEG | — |
| **T-05** | Corrected evaluation harness | eval | todo | T-01 | G-NORM | — |
| **T-06** | **Generator MAIN — predicted-mask, direct `z_t1c`** | fm/train | todo | T-04,T-05 | G-NORM,G-SEG | **v3b `ema_best.ckpt`** |
| **T-07** | Mask-fusion ablation: SPADE/adaLN-zero vs ControlNet | fm/train | todo | T-06 | — | v3b or T-06 |
| **T-08** | Informed-prior start (x1-prediction) — speed arm | fm/train | todo | T-06 | — | T-06 |
| **T-09** | Gated decoded rim-frequency loss (Stage 2) | fm/train | todo | T-06 | — | T-06 best |
| **T-10** | Warm-start-source ablation (v3b vs v3a vs base) | fm/train | todo | T-06 | — | {v3b,v3a,base} |
| **T-11** | Aleatoric uncertainty head (optional novelty) | fm/model | todo | T-06 | — | T-06 best |
| **T-12** | Re-encode (`z_mu` + canonical norm) — *conditional* | data | todo | T-01 | **only if G-NORM demands** | — |

¹ T-02: the measurement already exists (§14.1: clean round-trip ρ_S≈0.00, PSNR_ET≈24 dB,
`pf_mean.py`/`pf_norm_check.py`). T-02 is packaging it as a `routines/preflights/vae_floor`
report for the paper, not re-measuring.

## 15.2 Checkpoint & artifact lineage (normative)

```
MAISI FM base trunk (frozen pretrained)
  /mnt/.../fscratch/checkpoints/NV-Generate-MR/diff_unet_3d_rflow-mr.pt
        │  (S1 v3 runs, already trained — DO NOT retrain)
        ├─ v3a  concat-only, no mask        experiments/2026-06-2*_s1_v3a_concat_only_fft_*/
        ├─ v3b  concat + 3ch-CN oracle mask experiments/2026-06-2*_s1_v3b_concat_plus_cn3ch_fft_*/   ◄── T-06 resumes here
        └─ v3b_rw  + region loss (RETIRED, net-negative)  ...2026-06-22_15-20-57_s1_v3b_rw_..._320b5ddd/
                                                             (known path; do NOT use as T-06 base)
NEW:
  segmenter ckpt (T-03) ──► predicted-mask cache masks/tumor_latent_pred (T-04)
        │
        ▼
  T-06  proposed generator  = v3b architecture, resume_from v3b/ema_best.ckpt,
        data.mask_source = predicted   ──►  the deployable model
        ├─ T-07 SPADE arm, T-08 informed-prior arm, T-09 +rim loss, T-11 +uncertainty
```

- **T-06 resume-from:** the **v3b (non-rw)** `ema_best.ckpt`. Resolve the exact run
  dir with `ssh picasso ls -d /mnt/home/users/tic_163_uma/mpascual/execs/vena/experiments/2026-06-2*_s1_v3b_concat_plus_cn3ch_fft_*` and pin it in `T-06`. **Not** v3b_rw
  (region loss retired). Copy that run's `model.trunk` + `model.controlnet` blocks
  **verbatim** into the T-06 YAML (same pattern the S3-on-v3 configs used).
- **Caveat (carry from `s1_v3` note):** `ema_best.ckpt` was selected on
  `mse_latent_bg`, not PSNR_ET, and its `trunk_ema_snapshot.pt` is ~25–75 epochs
  ahead of the live trunk. Acceptable for a warm-start; note it in T-06.
- **Encode base:** all latents come from the frozen VAE `autoencoder_v2.pt`. Do not
  re-encode unless **T-12** is triggered by G-NORM.

## 15.3 Gate registry (hard gates per component)

| Gate | Owner | Decision rule (PASS) | Artifact | Unblocks / fallback |
|---|---|---|---|---|
| **G-NORM** | T-01 | one canonical `percentile_upper` chosen and applied in encode **and** every metric path; the ρ_S ranking **re-derived** with synth+real forced to that percentile, and the change vs the frozen ranking quantified | `artifacts/preflights/rho_s_norm_audit/LATEST/decision.json` | Unblocks T-05/T-06 eval trust. **Fallback:** if matched-norm collapses the latent-vs-identity gap, re-scope the paper claim *before* training (the model may already be competitive). |
| **G-VAE** | T-02 | VAE round-trip ρ_S ≤ ~0.1 and PSNR_ET ≥ ~22 dB at the canonical norm (already measured ≈0.00 / 24 dB) | `artifacts/preflights/vae_floor/LATEST/decision.json` | Confirms "stay latent"; sets the generator's aspirational ceiling. **Fallback:** if it had failed (it did not), pivot to image-space refiner. |
| **G-SEG** (PF3) | T-03 | `WT Dice ≥ 0.80` **and** `NETC Dice ≥ 0.50` on held-out, per cohort incl. Ring B; healthy-control → ~empty mask | `artifacts/preflights/predicted_mask/LATEST/decision.json` | Unblocks T-04/T-06 predicted-mask conditioning. **Fallback:** collapse to a single coarse "active-tumour" prior channel; keep the oracle-mask arm (v3b) only as a reported upper bound. |
| **G-SHORTCUT** | T-06 eval | healthy-control false-positive enhancement volume ≈ 0 (proposal §6.5) | in T-06 exhaustive-val report | Deployability claim. **Fallback:** flag shortcut, add §6.5 diagnostic before any adversarial arm. |

## 15.4 Task specifications

### T-01 · ρ_S normalisation audit + canonical percentile → **G-NORM** (Wave 0, do first)
- **Purpose.** Decide the one intensity percentile used *everywhere* and quantify how
  much of the frozen ρ_S ranking is a normalisation artifact (§14.3: a 99.5↔99.95
  mismatch swings ρ_S 0.00→0.66).
- **Approach.** (a) Confirm the encoder percentile from `encode_runtime_attrs_json`
  (measured: **99.95**). (b) For all 16 methods on Ring A, re-derive ρ_S with synth
  and real *both* normalised at the same percentile (test 99.5 and 99.95). (c) Compare
  the re-derived ranking to the frozen `spatial_residual` artifact; report per-method
  Δρ_S and whether the "latent-worse-than-identity" ordering survives.
- **Files.** `routines/preflights/rho_s_norm_audit/` (engine reuses
  `vena.validation.spatial_residual` with a forced-percentile normaliser); reconcile
  `fm/eval/exhaustive.py` default (99.5) and `validation/io.py` harmonisation with the
  chosen value.
- **Acceptance.** decision.json with `canonical_percentile_upper`, per-method Δρ_S
  table, and a boolean `latent_worse_than_identity_survives`.
- **Gate.** G-NORM as in §15.3.
- **Cost.** CPU-hours (no training). **Highest priority — cheaper than any run.**

### T-02 · VAE floor (Study 7) formalised → G-VAE (Wave 0)
- **Purpose.** Turn the §14.1 measurement into the paper's Study-7 artifact and the
  generator's ceiling.
- **Approach.** Package `pf_mean.py`/`pf_norm_check.py` logic as
  `routines/preflights/vae_floor/`: encode→decode real T1c at the canonical percentile
  over Ring A; report ρ_S, PSNR_{brain,ET}, MS-SSIM, Frangi conspicuity of the recon.
- **Acceptance.** decision.json: `recon_rho_s`, `recon_psnr_et`, `is_bottleneck: false`.
- **Cost.** ~1 GPU-h.

### T-03 · Pre-contrast tumour segmenter → **G-SEG / PF3** (Wave 1)
> **⚠ AMENDED by §16.1 (iter 6).** The output is now soft **`[WT, NETC]`** (2 channels),
> NOT `[NETC, ED, ET]`. ET is undefined pre-contrast; the enhancing region the
> generator paints is a learned subset of `WT ∖ NETC`. The three-way ET-channel hack
> below is superseded — read §16.1/§16.6 before building the segmenter.
- **Purpose.** Produce the *deployable* enhancement prior — soft `[NETC, ED, ET]` from
  pre-contrast only — that replaces v3b's oracle mask.
- **Proposed architecture.** MONAI **SegResNet**, **3-input** `{t1pre, t2, flair}`
  (fork `vena.validation.downstream_seg`'s 4-input SegResNet, drop the `t1c` channel),
  output 3 soft class channels matching the oracle `masks/tumor_latent` semantics
  (per-class, codes NETC=1/ED=2/ET=4). **ET is the aleatoric channel** (undefined
  pre-contrast) — build it three ways and pick per G-SEG: (i) predict a coarse
  "enhancement-likely shell" = `WT ∖ NETC ∖ ED`; (ii) zero the ET channel (generator
  infers enhancement); (iii) predict ET directly and report its (expected-low) Dice as
  the honest ceiling. Loss: Dice + CE, deep supervision. Train at image res, then
  per-class-avg-pool to latent `(3, 48, 56, 48)` for conditioning.
- **Files.** library `src/vena/model/seg/` (SegResNet wrapper + config); routine
  `routines/preflights/predicted_mask/` (train + eval + decision.json). Tests under
  `tests/model/seg/`.
- **Inputs.** cohort image H5s (`images/{t1pre,t2,flair}`, `masks/tumor`), GT splits.
- **Acceptance / Gate.** WT Dice ≥ 0.80, NETC Dice ≥ 0.50 (per cohort, incl. Ring B);
  healthy-control → empty mask. Emit `decision.json` with per-class Dice/AHD + the
  chosen ET-channel policy. **Fallback** per §15.3.
- **Cost.** ~1 GPU-day train, minutes/inference.

### T-04 · Predicted-mask cache (Wave 1)
- **Purpose.** Make the predicted mask a first-class, reproducible training input.
- **Approach.** Run T-03 inference over every cohort; write `masks/tumor_latent_pred`
  `(3,48,56,48)` into each latent H5 **alongside** `masks/tumor_latent` (do not
  overwrite the oracle); bump latent-H5 `schema_version` and add a
  `predicted_mask_seg_sha256` attr. Validate shape/label parity with the oracle mask.
- **Files.** `routines/pipeline/mask_predict/` (thin engine over T-03 + the existing
  `per_class_avg_pool` downsampler + H5 writer/validator).
- **Acceptance.** every cohort has `masks/tumor_latent_pred`; validator passes; a
  spot-decoded overlay looks sane.
- **Cost.** ~2–4 GPU-h across cohorts.

### T-05 · Corrected evaluation harness (Wave 1)
- **Purpose.** Score the new model *fairly* and add the missing competitor.
- **Approach.** (a) Apply the T-01 canonical percentile to exhaustive-val and the
  spatial_residual sweep. (b) Report **PSNR_ET** (primary enhancement metric) + a
  literal **Frangi vessel-conspicuity ratio** (Study 5 T5.4) alongside ρ_S (reported
  as companion, never optimised). (c) **Add TA-ViT** (Eidex, Med Phys 2025, mp.17600 —
  predicted-mask on UCSF-PDGM) to the competitor registry via the `integrate-competitor`
  skill — it is the baseline the deployable model must beat.
- **Files.** patch `fm/eval/exhaustive.py` + `validation/io.py` normalisation; new
  `vena.validation` Frangi-conspicuity metric; competitor entry for TA-ViT.
- **Acceptance.** re-derived Table 5 (ρ_S) + a PSNR_ET/conspicuity table incl. TA-ViT.
- **Gate.** consumes G-NORM.

### T-06 · **Generator MAIN — the proposed model** (Wave 2, core deliverable)
- **Purpose.** The deployable model: predict `z_t1c` from pre-contrast anatomy + a
  *predicted* enhancement prior, recovering v3b's oracle margin without the oracle.
- **Architecture (concrete).** Frozen MAISI VAE latents `(4,48,56,48)`. MAISI FM U-Net
  trunk, **FFT + trunk-EMA**. Objective: **noise→`z_t1c`**, v-prediction, **L1** CFM
  (`loss.cfm.norm: l1`), `rflow.use_timestep_transform: true`,
  `base_img_size_numel: 129024`. **Conditioning:** (a) **anatomy channel-concat**
  `[z_t1pre,z_t2,z_flair]` → 16-ch `conv_in` (verbatim from v3b); (b) **predicted mask
  via the v3b 3-ch ControlNet** (`init_from_trunk`, zero-init output, `output_scale`
  ramp 0→1 over 5000 steps) — the only change from v3b is `data.mask_source: predicted`
  (reads `masks/tumor_latent_pred`, T-04). EMA 0.9999. EarlyStopping/patience 250.
- **Resume-from.** v3b `ema_best.ckpt` (§15.2). Copy v3b `model.trunk` +
  `model.controlnet` blocks verbatim; set `run.resume_from` to the pinned Picasso path.
- **Files.** `routines/fm/train/configs/runs/picasso_pm_v1_predmask_cn3ch_fft.yaml`
  (+ loginexa smoke `loginexa_pm_v1_*.yaml`, + slurm launcher/worker). A DataModule
  switch `data.mask_source: predicted|oracle` (default oracle for back-compat).
- **Acceptance.** on Ring A, at matched norm: **PSNR_ET ≥ v3b** within noise (target:
  recover ≥ ~70 % of the v3b−v3a ET gap with the predicted mask) **and** ρ_S ≤ v3a;
  pass **G-SHORTCUT**. Report per-NFE cost curve.
- **Gates.** G-NORM, G-SEG (needs T-04); G-SHORTCUT on its own eval.
- **Cost.** ~2–4 GPU-days on 4×A100 (warm-start ⇒ far shorter than from-scratch).
- **Variants to try.** mask channels `[NETC,ED,ET-shell]` vs `[NETC,ED,ET-zero]`
  (from T-03); soft-prob vs hard mask input. Pick on PSNR_ET + G-SHORTCUT.

### T-07 · Mask-fusion ablation — SPADE/adaLN-zero vs ControlNet (Wave 3)
- **Purpose.** Test the TA-ViT-style fusion against the ControlNet primary.
- **Approach.** Replace the ControlNet branch with **spatially-adaptive GroupNorm
  modulation** of the trunk from the predicted mask. **Fix the init (§14.8-B7):** base
  affine = pretrained `(γ,β)`, add zero-init `(Δγ,Δβ)` predicted from the mask (or a
  post-norm zero-init residual) so step-0 is truly identity. Compare vs T-06.
- **Files.** `src/vena/model/fm/controlnet/` sibling `spade_fusion.py`; a config arm.
- **Acceptance.** ΔPSNR_ET vs T-06 with matched compute; report parameter/compute cost.
- **Resume-from.** v3b (fresh fusion) or T-06 (warm).

### T-08 · Informed-prior start — speed arm (Wave 3)
- **Purpose.** Fewer NFE via short transport (WFM/LBM), **without** re-introducing the
  diffuse-residual target that §14.2 killed.
- **Approach.** Source `x0 = z_t1pre (+ σ·ε, small σ)`; interpolant `x_α=(1−α)z_t1c+α x0`.
  **Use x1-prediction** (network outputs `ẑ_t1c`, velocity `=(ẑ_t1c−x_α)/α`, loss on
  `‖ẑ_t1c−z_t1c‖`) with SNR-weighting to tame `α→0` — so supervision stays on the
  enhancement-salient `z_t1c`, not on `Δ`. Keep everything else = T-06.
- **Acceptance.** equal PSNR_ET at **fewer NFE** than T-06 (report the NFE–quality
  curve); if fidelity drops, it is a speed-only arm.
- **Resume-from.** T-06 (re-parameterise the head) or from base if x1-pred needs it.

### T-09 · Gated decoded rim-frequency loss (Wave 3, Stage 2)
- **Purpose.** Sharpen the cm-scale enhancing rim (not vessels — Nyquist-capped, §14.8).
- **Approach.** Reuse the S3 LPL/decoder machinery: a **gated decoded** frequency /
  gradient-difference term on `x̂_1` at high-SNR timesteps (`t_dn>t_min≈0.4`), masked to
  the predicted `WT∖NETC` shell, **ramped from 0**. Monitor per-term grad-norm + cosine
  vs the CFM term (§6 diagnostics); down-weight on persistent conflict.
- **Acceptance.** ΔPSNR_ET / rim-SSIM ↑ vs T-06, no MS-SSIM_brain regression > MCID.
- **Resume-from.** T-06 best `ema_best.ckpt`.

### T-10 · Warm-start-source ablation (Wave 3)
- **Purpose.** Clean attribution of the predicted-mask gain.
- **Approach.** Train T-06 recipe from **v3b** (default), **v3a** (mask learned fresh),
  and the **MAISI base** (full from-scratch, B2 fallback). Compare convergence + final.
- **Acceptance.** a 3-row table (source → PSNR_ET, GPU-h to plateau).

### T-11 · Aleatoric uncertainty head (Wave 4, optional novelty)
- **Purpose.** Emit per-voxel enhancement-uncertainty (Piening 2024), the honest way to
  present a gadolinium-free tool given the 8–20 % aleatoric floor (§14.7).
- **Approach.** second output head predicting a per-voxel variance/β on `x̂_1`; train
  with a heteroscedastic (β-NLL) term on the enhancement region. Report calibration vs
  held-out error; overlay uncertainty on the enhancing rim.
- **Resume-from.** T-06 best. **Non-blocking** for the deployable model.

### T-12 · Re-encode (`z_mu` + canonical norm) — *conditional* (Wave 4)
- **Purpose.** Only if G-NORM/T-01 selects a percentile ≠ the cached 99.95, **or** a
  later latent-space residual loss needs the deterministic mean. Cache `z_mu`
  (deterministic, not a sample) + the canonical normalisation.
- **Trigger.** `G-NORM.canonical_percentile_upper != 99.95` OR a task adds a latent
  residual/enh-map loss.
- **Cost.** ~25 h + ~150 GB, all cohorts + aug siblings; schema bump. **Do not run
  speculatively** — it invalidates the v3b warm-start (new latent space).

## 15.5 Execution waves (DAG)

```
Wave 0 (gates, cheap, parallel):     T-01(G-NORM)   T-02(G-VAE)
Wave 1 (new components, parallel):   T-03(G-SEG) ─► T-04        T-05  ◄─ needs G-NORM
Wave 2 (core, gated):                T-06  ◄─ needs T-04, T-05, G-NORM, G-SEG
Wave 3 (ablations, after T-06 curve):T-07  T-08  T-09  T-10
Wave 4 (optional):                   T-11        T-12(conditional on G-NORM)
```
Rule: **do not enter Wave 2 until G-NORM + G-SEG pass.** T-01 first — if the ρ_S gap is
a normalisation artifact, the whole target metric (and possibly the paper) re-scopes
before any GPU-day is spent.

## 15.6 Conventions the generated tasks must honour

- **Run naming:** `picasso_pm_<ver>_<recipe>_fft.yaml` (`pm` = predicted-mask family),
  loginexa smoke `loginexa_pm_<ver>_*.yaml`. Run-id stays `<UTC>_<stage>_<tag>_<sha>`.
- **decision.json:** bump `schema_version`; add `mask_source: predicted|oracle`,
  `segmenter_checkpoint_sha256`, `predicted_mask_h5_key`, `canonical_percentile_upper`.
  Never repurpose an existing key.
- **Warm-start audit:** record `resume_mode: warm_start`, `resume_source_run_id` (the
  v3b run), and the `ema_best`-selected-on-`mse_latent_bg` caveat.
- **Every long run** carries the four S1-v2 recipe deltas verbatim (L1,
  scale-ramped zero-init, `use_timestep_transform`, mask slot) unless the Task says
  otherwise; plumb `input_img_size_numel` into the sampler (else silent val failure).
- **Task tracking:** the orchestrating agent keeps a top-of-`§15.1` status column
  current and appends outcomes to each `T-NN_*.md`; when a Task closes, record the
  produced checkpoint/artifact path back into §15.2.

---

# 16. Segmenter-centred `Soft[WT,NETC]` pivot (iter 6, 2026-07-21)

> **Status: decisions from a design chat, not a measured review.** They AMEND §15
> (they do not supersede §14's verdict — no bridge; predict `z_t1c` directly; mask
> conditioning is the primary lever). Where §16 and §15 disagree on the *mask
> representation, the ceiling arm, the fusion mechanism, or the FP study*, §16 wins.
> T-01 (ρ_S norm audit) still runs first and can re-scope all of this.

## 16.0 Why this iteration exists — reframing the target

The user's driving question was *"is mask conditioning enough to close the gap
between the VAE ceiling and the generated T1c?"* The honest answer is **no, and
the VAE ceiling is the wrong yardstick.** Decompose the gap
`PSNR_VAE_ceiling − PSNR_generator`:

- **(A) measurement artifact** — the ρ_S ranking may be partly a 99.5/99.95
  normalisation confound (§14.3). **Audit first (T-01).**
- **(B) irreducible aleatoric error** — enhancement is only 81–92 % predictable
  from pre-contrast (§14.7: Azizova 86–92 %, Zheng BBB-AUC 81 %). No model crosses
  this. On PSNR_ET specifically the aleatoric tax is *largest* (enhancement is the
  unpredictable part): a Bayes-optimal predictor's ceiling is
  `24 − 10·log10(1 + f·σ²_tot/σ²_vae)` dB ≈ **19–21 dB**, not the 24 dB VAE floor —
  so v3b-rw (≈17.3) sits ~2–4 dB under the *reachable* ceiling, not 7 dB under the
  VAE floor.
- **(C) reducible generator error** — the only bucket any model change touches. Mask
  conditioning addresses a *slice* of (C) (localisation). It is a **fair-single-axis
  win**, not a gap-closer.

**Success is therefore re-defined** from "maximise PSNR toward the VAE ceiling"
(unreachable) to **"trustworthy, deployable contrast placement without an oracle"** —
which is what the segmenter-centred story delivers and what CONTRIBUTION §6 (T6.5)
already identified as the one shot at a *fair* model-win.

## 16.1 The conditioning signal — soft `[WT, NETC]`, 2 channels

**Decision.** Replace v3b's GT `[NETC, ED, ET]` (3-ch) with **soft `[WT, NETC]`**.

- **Rationale.** A *pre-contrast* segmenter can predict WT (FLAIR/T2 hyperintensity,
  Dice ≳ 0.85) and NETC/necrosis (T1pre-hypo / T2, moderate Dice). It **cannot**
  predict ET — enhancement is undefined without contrast (that is the whole task).
  So `[WT, NETC]` is the *deployable-honest* prior; a 3-ch `[NETC,ED,ET]` predicted
  mask fabricates the one channel a segmenter cannot know.
- **Semantics the generator resolves.** `WT = NETC ∪ ED ∪ ET`, so `WT ∖ NETC = ED ∪ ET`
  — the mask localises the lesion and excludes necrosis, and the generator learns
  *within* `WT ∖ NETC` which voxels enhance (edema is T2-bright and separable from
  enhancing tumour). The mask says **where enhancement is possible**, not where it is.
- **Soft, not hard.** Feed the segmenter's sigmoid probability directly (per-class
  avg-pooled to latent `(2, 48, 56, 48)`). The soft value **is** the first-order
  uncertainty signal (≈0.5 at ambiguous boundaries) — see §16.6.

## 16.2 Fusion mechanism — ControlNet primary, SPADE as the T-07 ablation

**Decision.** Keep the **v3b ControlNet** branch as the primary fusion; SPADE/adaLN
is the ablation (T-07), not the primary. Four reasons:

1. **One-variable discipline.** This iteration already changes the mask
   *representation* (`[NETC,ED,ET]→[WT,NETC]`) and the mask *source*
   (`GT→predicted`). Changing the *mechanism* simultaneously destroys attribution
   (orchestrate skill §7: "two arms, two code paths → the arms drift").
2. **Proven.** v3b-ControlNet already bought ~+0.45 dB PSNR_brain, ~26 % MAE_wt, and
   ~halved ρ_S-excess-over-identity vs v3a. It works.
3. **Novelty.** Predicted-mask **adaLN-zero/SPADE** on UCSF-PDGM *is* Eidex TA-ViT
   (Med. Phys. 2025, §14.6). ControlNet fusion is **more differentiated** from the
   closest prior art. SPADE would put us architecturally on top of TA-ViT.
4. **Risk.** SPADE-on-pretrained-GroupNorm has the §14.8-B7 landmine (the
   "zero-init byte-identical" claim is false — it discards the pretrained affine
   `(γ,β)≠(1,0)` unless folded in).

**Do NOT drop ControlNet** before T-07 measures SPADE. Adopt SPADE only if T-07 shows
a clear PSNR_ET win at matched compute.

**Warm-start note.** Changing the ControlNet's `cond_embedding` first conv from
`in_channels=3` to `in_channels=2` breaks *only that conv's* warm-start (it was
Kaiming-init in v3b anyway, `init_from_trunk: false`); the expensive trunk warm-start
from v3b `ema_best.ckpt` is unaffected. Alternatively keep 3 slots as
`[WT, NETC, zero_out]` (via the existing `ZeroOutDownsampler`) to preserve byte-layout
for warm-start; decide in T-06.

## 16.3 The fair ceiling — re-run GT with `[WT,NETC]` (new T-13)

**The current v3b GT `[NETC,ED,ET]` mask is NOT the matched ceiling** for a
`[WT,NETC]`-predicted deployable arm — it conditions on a strictly more informative
signal (the ET channel no segmenter can provide). Reporting predicted-`[WT,NETC]`
against GT-`[NETC,ED,ET]` conflates *segmenter quality* with *the ET-oracle bonus*.

**Decision — a clean three-arm ceiling decomposition:**

| arm | mask | provides | reads as |
|---|---|---|---|
| **maximal oracle** (existing v3b) | GT `[NETC,ED,ET]` | incl. GT enhancement location | upper-upper bound (uses info no tool has) |
| **fair oracle** (new T-13) | GT `[WT,NETC]` | perfect segmenter, no ET oracle | **the ceiling matched to the deployable signal** |
| **deployable** (T-06) | predicted `[WT,NETC]` | pre-contrast segmenter | the shippable model |

- `deployable → fair-oracle` gap = **segmenter-quality cost** (the paper's honest
  "what does an imperfect mask cost").
- `fair-oracle → maximal-oracle` gap = **the price of not knowing enhancement
  location pre-contrast** (partly the aleatoric floor, §14.7).

**Headline.** *"Mask conditioning on a deployable pre-contrast `Soft[WT,NETC]` prior
significantly improves latent-space T1c contrast placement; the GT-`[WT,NETC]` oracle
bounds it."* Minimum viable = deployable + fair-oracle (2 runs). Report maximal-oracle
only if the existing v3b number is reused (free — it exists).

## 16.4 Drop the `-rw` arm from the headline

`v3b_rw` (region-weighted L1) is net-negative (Study 6, and CONTRIBUTION §3:
"region weighting helps ✗"). The fidelity table's "**VENA (ours)**" row **is** the
`cond + rw` variant — retire it from the headline. The headline model is the
mask-conditioned **direct-`z_t1c`** generator (T-06), with GT-`[WT,NETC]` (T-13) as
the ceiling. Keep v3a (no-mask) beside it as the ablation baseline that isolates the
mask contribution.

## 16.5 False-enhancement (FP) safety study (new T-14)

**Motivation.** C7-Latent-Pix2Pix *wins* ρ_S (−0.19, §CONTRIBUTION-2) — an adversarial
latent method that reaches VAE-floor vessel fidelity. To argue the mask-gated flow is
the *right* answer despite C7's ρ_S win, we need the axis where an unconstrained
GAN is expected to fail: **hallucinated (false-positive) enhancement.** This is a
patient-safety axis for a gadolinium *replacement* and a genuine **third contribution**
(fidelity + vessel-ρ_S + safety).

**Design (all 16 methods, no new cohort needed):**
- **Tumour-region FP** — stratify Ring A by GT ET volume; on cases with **GT ET ≈ 0**
  (non-enhancing tumour), measure synthetic false-enhancement volume inside WT.
- **Contralateral / healthy-tissue FP** — measure enhancement painted in the
  contralateral hemisphere / brain∖WT where none exists in real T1c.
- **(if a control cohort is ever sourced)** proposal §6.5 healthy-control protocol →
  G-SHORTCUT.

**Honest caveats (orchestrate skill: report what is computed, not what is tidy):**
1. ρ_S excludes the tumour (brain∖dilate(WT)); C7's ρ_S win says nothing about
   tumour-region FP. The FP study is **orthogonal** to ρ_S — it is a new measurement,
   not a re-slice.
2. The outcome is **not guaranteed** to demote C7. If C7 does not hallucinate, we
   cannot use safety to beat it — accept and report that.
3. VENA must be run through the **same** FP harness to show mask-gating *prevents* the
   FP (empty predicted mask on a non-enhancing case ⇒ no enhancement prior ⇒ no false
   enhancement). That symmetry is the contribution.

## 16.6 Segmenter methodology (for the parallel session that designs it)

Deliverables for the segmenter (the "methodological centre" the user asked to
establish), in order:
1. **Define the target** — soft `[WT, NETC]` from `{T1pre, T2, FLAIR}` (drop the T1c
   channel from `vena.validation.downstream_seg`'s 4-input SegResNet). Loss Dice+CE,
   deep supervision. Train at image res, per-class avg-pool to latent `(2,48,56,48)`.
2. **Train + evaluate** — per-cohort WT/NETC Dice + AHD vs GT, incl. Ring B (OOD).
   **Gate G-SEG:** WT Dice ≥ 0.80, NETC Dice ≥ 0.50; healthy control → ~empty mask.
3. **Visualise** — mask overlays on axial slices per cohort; the soft-probability map
   (not thresholded) is what the generator consumes.
4. **Uncertainty — honest scope.** The soft sigmoid probability **already is** an
   uncertainty signal (≈0.5 at ambiguous boundaries) and is free — use it directly as
   the conditioning input. A **separate explicit epistemic-variance channel**
   (MC-dropout / deep-ensemble variance) to "allocate spatial probability density" is
   a *mild stretch*: defensible but must earn its place via an ablation (does the extra
   channel improve PSNR_ET or FP over the soft-prob alone?). Do **not** headline it
   unless the ablation pays. Distinct from the *generator-output* uncertainty head
   (T-11) — that is output aleatoric uncertainty, a different mechanism.

## 16.7 Amendments to the §15 task graph

- **T-03** — segmenter output is soft `[WT, NETC]` (2-ch), not `[NETC,ED,ET]`. G-SEG
  now gates on WT Dice ≥ 0.80 **and** NETC Dice ≥ 0.50 only (no ET Dice).
- **T-04** — cache key `masks/tumor_latent_pred` becomes `(2,48,56,48)`.
- **T-06** — conditioning = predicted `Soft[WT,NETC]` via ControlNet (§16.2); mask slot
  is 2-ch (or `[WT,NETC,zero_out]` for warm-start byte-layout). Acceptance vs the
  **fair oracle T-13**, not vs GT-`[NETC,ED,ET]` v3b.
- **T-13 (new)** — *fair-oracle ceiling*: T-06 recipe with **GT `[WT,NETC]`** mask.
  Resume from v3b `ema_best.ckpt`. Produces the matched ceiling for §16.3.
- **T-14 (new)** — *false-enhancement study* (§16.5), all 16 methods, Ring A + Ring B.
  Reuses the prediction cache + GT ET labels; CPU-side like the spatial_residual sweep.
- **T-07** — unchanged in intent (SPADE ablation), now explicitly *secondary* to the
  ControlNet primary per §16.2.

---

# 17. T-01 ρ_S normalisation audit — RESULT (2026-07-21, iter 7)

> **Outcome: the confound is confirmed and large enough to break the headline.**
> Artifact `artifacts/preflights/rho_s_norm_audit/2026-07-21T14-09-50Z` (16 methods ×
> 247 Ring-A patients, git 135395a, clean run). Numbers independently re-derived from
> the per-patient CSVs — patient-collapse reproduces the article method (identity
> P99.5 0.349 ≈ CONTRIBUTION 0.351), so the collapse is not an aggregation fluke.

## 17.1 Matched-normalisation ρ_S (frozen → matched@99.95)

| method | frozen | matched@99.95 | Δ |
|---|--:|--:|--:|
| C0-Identity (ref) | 0.436 | 0.447 | ~0 |
| **C4-3D-DiT** | 0.741 | 0.027 | **−0.71** |
| **C5-T1C-RFlow** | 0.550 | −0.067 | **−0.62** |
| **C6-3D-LDDPM** | 0.507 | −0.040 | **−0.55** |
| VENA-v3a (fair) | 0.373 | −0.044 | −0.42 |
| VENA-v3b-rw (oracle) | 0.212 | −0.080 | −0.29 |
| C2-ResViT (pixel) | 0.054 | 0.009 | −0.05 |
| C1-pGAN-t1pre (pixel) | 0.004 | −0.084 | −0.09 |
| C7-Latent-Pix2Pix | −0.417 | −0.511 | −0.10 |

The collapse is **latent-specific**: the latent methods (decoded in the encoder's
99.95 space, mis-scored vs a 99.5 reference) crater; pixel/identity barely move — the
exact signature §14.3 predicted. Corrected order (worst→best): **identity 0.45** →
SynDiff-t1pre 0.27 → C4 0.03 ≈ ResViT 0.01 ≈ v3b −0.02 ≈ C6 −0.04 ≈ **v3a −0.04** ≈
C5 −0.07 ≈ pGAN −0.08 → pGAN-flair −0.18 → **C7 −0.51**.

## 17.2 What it refutes (CONTRIBUTION suspended, pending T-05)
- **"Latent diffusion/flow misplace contrast worse than identity"** — REFUTED. Under
  matched norm C4/C5/C6 (0.03/−0.07/−0.04) are all *better* than identity (0.45).
- **"VENA is the latent-tier fix"** — no failure left to reverse; VENA is mid-pack.
- Survives: C7 still leads robustly; identity worst (⇒ "any synthesis beats doing
  nothing" — generic/weak); and the normalisation-fragility is itself a methodological
  finding worth reporting.

## 17.3 Bug in the audit artifact
`decision.json`/`report.md` `latent_worse_than_identity_survives: true` is **wrong** —
contradicts its own table AND the independent patient-collapse (identity 0.372 worst;
no latent method > identity at either percentile). Numbers right, boolean inverted.
Fix in `routines/preflights/rho_s_norm_audit/` before the artifact is cited.

## 17.4 Canonical percentile + fix (feeds T-05)
Canonical `percentile_upper = 99.95` (the encoder's). Patch every
`percentile_normalise` in `fm/eval/exhaustive.py::load_real_t1c_normalised` and the
`validation/io.py` harmonisation to 99.95, then re-run `spatial_residual` + Holm-Wilcoxon
(T-05) for the corrected ranking WITH significance (this audit is point-estimates only).

## 17.5 Implications for the retraining (§15/§16)
- **Retraining is now justified on FIDELITY**, not the dead vessel story. v3b's
  mask-conditioning margin (MAE_wt 0.096 vs v3a 0.128; Dice_ET_synth 0.56 oracle) is on
  decoded intensity and is **not** normalisation-artifactual — so the segmenter-centred
  `Soft[WT,NETC]` model (§16, T-06/T-13) remains a valid contribution, reframed as
  *deployable mask-conditioned tumour-enhancement fidelity*.
- **The metric may be the wrong target.** ρ_S has now failed twice — normalisation-
  fragile (§17) AND uncorrelated with tumour-Dice (Spearman +0.04, §16.5). Rethink
  whether ρ_S is the headline axis at all before optimising a model to it; the honest
  clinical axis is enhancement-region fidelity (PSNR_ET) + false-enhancement safety.
- **G-NORM fired** (§15.3 fallback): matched-norm collapsed the latent-vs-identity gap
  ⇒ re-scope the paper claim before spending GPU-days.
