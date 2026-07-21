# Study 2 — Cost–quality Pareto (speed vs fidelity)

**Paper §4.6 · Priority: PRIMARY · Data status: ✅ BUILT + VERIFIED · Scope: Ring A**
**Routine:** `routines/validation/studies/cost_quality_study.py` → artifacts at
`results/article/cost_quality/LATEST/` (independently verified 2026-07-20, git_sha `cebc078`).

References `00_HUB.md` §2. This is the deployment-relevance argument: for a
gadolinium-free tool the question is *quality per second per GB*, not quality
alone.

> **Verified result (re-derived from the CSV, not transcribed).** On the
> cost–quality Pareto (median s/vol × MS-SSIM_brain, Ring A), the pre-registered
> **v3b-rw is NOT on the frontier — it is strictly dominated by its own no-mask
> ablation v3a** (1.762 s / 0.9191 beats 1.820 s / 0.9182 on *both* axes). The
> frontier is {C0-Identity, C1-pGAN-t2, **VENA-v3a**, C5-T1C-RFlow, **VENA-v3b**}.
> C6-3D-LDDPM (1000 NFE) is **28.5× slower** than VENA@5 for *lower* MS-SSIM —
> the few-step latent tier dominates the many-step diffusion baseline. A
> figure-caption headline, and a third independent strike against the
> pre-registered arm. Note C1-pGAN-t2 is a *supplementary* panel; mark it as such
> in the figure.

---

## 1. Question & claim

*At what inference cost does each method reach its fidelity?* The frontier is
built by plotting **every method at every NFE it was evaluated at** — so each
iterative method contributes a curve, and one-shot methods (GANs, Pix2Pix)
contribute a single point.

**Claim:** VENA (and the few-step latent-FM tier, C5-T1C-RFlow) sits on the
efficient frontier — near-best latent fidelity at **≤5 NFE / <10 s / volume**,
dominating C6-3D-LDDPM (1000 NFE) which needs ~200× the steps for no fidelity
gain, while the image tier trades away 3D-consistency (Study 1 `zgd`) for speed.

## 2. Axes & encoding

- **x-axis:** inference cost. Primary = **wall-clock seconds/volume** (median,
  A100-40GB, from `inference_seconds`). Secondary panel = **NFE** (the
  formulation-level cost, hardware-independent). Report both — reviewers ask for
  the hardware-independent one; clinicians care about seconds.
- **y-axis:** fidelity. Primary = **MS-SSIM_brain** (your pick; whole-volume,
  perceptually weighted, less noisy than region metrics). Secondary = MAE_brain
  (matches the primary endpoint). Produce both; lead with MS-SSIM_brain.
- **point size / colour:** peak VRAM (`peak_vram_mb`); tier by colour (image vs
  latent vs VENA).
- **markers:** one marker per (method, NFE). Connect same-method markers into a
  curve; annotate the `sel NFE` marker with a ring.

VENA NFEs on disk: **{1, 2, 5, 10, 20}**. C4/C5: 6 NFE levels each. C6: {…,1000}.
One-shot: C0, C1-pGAN, C2-ResViT, C7 (single point each). C3-SynDiff: 4-step.

## 3. Data — paths

**Source:** same `paired_fidelity/LATEST/per_scan/paired_fidelity_patient.csv`
(has `nfe`, `inference_seconds`, `peak_vram_mb`, all fidelity metrics) — **no
generation needed**. Convenience aggregate `tables/cost_table.csv` and
`figures/cost_quality_pareto.png` already exist; **rebuild from per_scan** so the
axis/metric choices above are ours, not the routine's defaults.

Filter Ring A; per (method, nfe) take median seconds + mean fidelity over
patients (report IQR on time — inference time is right-skewed). Mark
Pareto-dominant points (no other point better on *both* axes).

## 4. Sub-analyses / tables

**Table 2A — matched-NFE=5 head-to-head** (Hub §2.3): C4, C5, VENA-v3a,
VENA-v3b-rw at NFE=5 — the *only* apples-to-apples comparison of generative
formulation at equal step budget. This isolates "is VENA's recipe better than
T1C-RFlow's *at the same cost*" from selection-NFE confounds.

**Table 2B — cost ledger:** per method — sel NFE, median s/vol, peak VRAM,
params (trunk + controlnet for VENA; from decision.json). State what the timer
includes: **latent methods include VAE decode; image methods are a single forward
pass** (fairness note — do not hide it).

## 5. Fairness & caveats (state in caption)

- The NFE axis is only meaningful for iterative methods; GANs are structurally
  1-step — that *is* the story (one-shot speed vs iterative refinement), not an
  unfair comparison.
- **VENA sampling is unseeded** (§2.5⑤) → per-NFE draws differ; the curve is a
  mean over patients, so this is averaged out, but note it.
- Time measured on A100-40GB; state driver/torch versions. Peak VRAM is the
  deployability gate on an 8–12 GB clinical workstation.
- Decode time is shared across the latent tier (same VAE) — so latent-tier
  *differences* are pure generator cost.

## 6. Figures

**Pareto family — one figure per (metric, region)** (2026-07-20 spec).
`fig_pareto_{metric}_{region}.png` for metric ∈ {ms_ssim, ssim, psnr} × region ∈
{brain, wt, bg_undilated} (9 figures; +et/tc once Study 8 lands). Encoding:

- **white background**; x = median inference_s/vol (**log**); y = mean(metric,region),
  all higher-better; **dominance recomputed per (metric, region)**.
- **generation space → marker *shape*** (not colour): latent `o` · pixel `s` ·
  reference(C0) `D`.
- **one colour per method**; a method's NFE points share its colour and are joined
  by the **closest path** (nearest-neighbour in normalised plotted coords).
- **frontier points → white star inside the marker, alpha 1.0**; **dominated →
  alpha 0.6**; **VENA-* always alpha 1.0** (so a dominated VENA arm stays legible).
- per-point NFE label; method name at each line's endpoint; VRAM lives in table2b.

The horizontal VAE-floor line (Study 7) overlays the same axes once available.
The `ms_ssim_brain` panel is the one that shows **v3b-rw faded/star-less while
v3a is starred** — the Pareto-domination result.

## 7. Reviewer objections & pre-emptions

| Objection | Pre-emption |
|---|---|
| "Timing is hardware-specific" | Dual x-axis: NFE (hardware-independent) + seconds (A100, stated). |
| "Latent methods hide VAE decode time" | Table 2B states the timer scope explicitly; decode is counted. |
| "Why not just use the fastest GAN?" | Pareto shows GAN speed but Study-1 `zgd` + Study-5 residual show what one-shot 2D trades away. |
| "NFE=20 barely differs from NFE=5" | That's the finding — few-step suffices; show the flat tail and pick sel-NFE=5. |

## 8. Task checklist

- [ ] **T2.1** Rebuild Pareto from per_scan (Ring A), x∈{seconds, NFE}, y∈{MS-SSIM_brain, MAE_brain}; mark dominance.
- [ ] **T2.2** Table 2A matched-NFE=5 (C4/C5/v3a/v3b-rw) with Wilcoxon+δ.
- [ ] **T2.3** Table 2B cost ledger (NFE, s/vol IQR, VRAM, params, timer scope).
- [ ] **T2.4** Overlay VAE floor line once Study 7 lands.
