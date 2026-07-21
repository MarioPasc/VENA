# VENA — Headline & Contribution Decision

> **⚠️ 2026-07-21 — HEADLINE UNDER AUDIT, DO NOT SUBMIT AS-IS.** The T-01 ρ_S
> normalisation audit (`artifacts/preflights/rho_s_norm_audit/2026-07-21T14-09-50Z`,
> 16 methods × 247 Ring-A) shows the vessel-axis ranking below is **largely a
> 99.5/99.95 normalisation artifact**. Under matched normalisation the latent
> methods collapse (C4 0.74→0.03, C5 0.55→−0.07, C6 0.51→−0.04), **identity becomes
> the worst**, and **"latent worse than identity / VENA = the latent-tier fix"
> (§0, §2 claims 1–2, §4) is REFUTED** — VENA is mid-pack; C7 still leads. §0/§2/§4
> vessel-axis claims are **suspended** pending **T-05** (re-run `spatial_residual` +
> Holm-Wilcoxon at canonical **99.95**). The audit's boolean
> `latent_worse_than_identity_survives:true` is a **bug** (contradicts its own
> table). Detail: memory `project_rho_s_norm_audit_2026_07_21`, redesign §17.

> The strategic core of the paper: *what do we claim, and on what evidence?*
> Written 2026-07-20 after re-deriving every axis from the frozen sweep CSVs.
> Supersedes any earlier "VENA wins" framing. When this disagrees with a study
> doc, this doc's evidence tables win (they are the re-derived numbers).
> Companion contracts: `00_HUB.md` (§2 registry/stats/regions). Vessel detail:
> `05_vessel_spatial_residual.md`.

---

## 0. TL;DR — the recommended headline

**There is no metric on which a *fair* (no-oracle) VENA arm cleanly beats the
field.** Pixel-domain 2D methods (ResViT, pGAN) lead whole-brain fidelity; pixel
methods **and** a latent-GAN (C7) lead the vessel/contrast-residual axis (ρ_S).
So the paper must **not** headline "VENA wins metric X" — a reviewer re-deriving
the numbers will find the same thing.

**Recommended headline — finding-led, VENA as the fix within its class:**

> *Whole-volume PSNR/SSIM rates latent diffusion/flow models as competitive
> gadolinium-free $T_{1c}$ synthesisers — but a vessel-resolved residual metric
> ($\rho_S$) shows they systematically misplace synthetic contrast onto
> vascular/enhancing structure, **worse than doing nothing** (C4/C5/C6:
> $\rho_S$ 0.46–0.73 vs identity 0.35). VENA is the only latent-generative recipe
> that significantly reverses this (v3a $\rho_S$ 0.31, all $p\le10^{-15}$ vs
> C4/C5/C6, large effects), at few-step (<2 s) inference.*

- **Headline model:** `VENA-S1-v3a` (fair, deployable, 3-input) + `VENA-S1-v3b-rw`
  (oracle upper bound) shown together.
- **Headline metric:** $\rho_S$ (spatial-residual / vessel-contrast fidelity) —
  chosen because it *exposes a field-relevant failure invisible to standard
  metrics*, not because VENA tops it. VENA's honest role: best of the latent
  diffusion/flow tier + the fix.
- **Never omit:** 2D pixel methods and the latent-GAN C7 achieve lower $\rho_S$;
  VENA's *strong* $\rho_S$ leans on the oracle mask.

---

## 1. Where VENA ranks on every axis (Ring A, sel-NFE, patient-collapsed, N=247)

All values re-derived 2026-07-20 from
`analyses/{paired_fidelity,spatial_residual}/LATEST/per_scan/*.csv`.

| Axis (↓/↑ = better) | Field leader | VENA-v3a (fair) | VENA-v3b-rw (oracle) | Verdict |
|---|---|---|---|---|
| **MAE_brain ↓** | C2-ResViT **0.0778** (pixel) | 0.0953 — **best latent**, 6/16 | 0.0955, 7/16 | best latent, loses to pixel |
| **MS-SSIM_brain ↑** | v3b* 0.9334 / ResViT 0.9322 | 0.9191 — 2nd latent (C5 0.9249 wins) | 0.9182 | not even best latent |
| **PSNR_brain ↑** | ResViT ~19.8 (pixel) | ~18.5 (top of latent) | ~18.35 | best latent, loses to pixel |
| **ρ_S C-noT ↓** (vessel) | C7 **−0.19** / pGAN 0.03 (pixel) | 0.3088 ≈ identity 0.3506 | 0.1973 — best latent-*flow* | pixel **and** C7 beat VENA |
| **ZGD → 1** (3D consist.) | C6-LDDPM 0.934 | 0.763 | 0.771 | no VENA advantage |
| **Cost (few-step <2 s)** | frontier {v3a, v3b, C5, pGAN, C0} | on frontier | dominated by v3a | on frontier, not unique |

\* v3b = oracle 3-channel [NETC,ED,ET] mask. Full 16-method ρ_S ranking in §2.

**Root cause (one sentence):** the 4× MAISI-VAE compression costs both bulk
fidelity **and** contrast-structure fidelity, so 2D pixel regression beats every
latent method on the axes we care about; VENA *mitigates* the latent-tier vessel
failure but does not *eliminate* the compression tax.

---

## 2. The vessel axis (ρ_S, C-noT) in full — significance-tested

`ρ_S` = Spearman(|residual|, real-$T_{1c}$ intensity) over brain∖dilate(WT);
**lower = errors avoid the bright enhancing/vascular voxels = better.** In C-noT
those bright voxels are predominantly vessels/sinuses/choroid/dural enhancement
(Hub §2.6 / Study 5 §1). The tumour is excluded from the metric region, so the
oracle mask does **not** trivially inflate it — yet it still helps (see below).

**Full ranking (patient-collapsed, N=247):**

| rank | method | tier | ρ_S | | rank | method | tier | ρ_S |
|--:|---|---|--:|---|--:|---|---|--:|
| 1 | C7-Latent-Pix2Pix | latent-GAN | **−0.19** | | 9 | **VENA-S3-LPL** | VENA | 0.184 |
| 2 | C1-pGAN-flair | pixel | 0.03 | | 10 | **VENA-v3b-rw** | VENA(oracle) | **0.197** |
| 3 | C3-SynDiff-flair | pixel | 0.04 | | 11 | **VENA-v3b** | VENA(oracle) | 0.205 |
| 4 | C1-pGAN-t2 | pixel | 0.06 | | 12 | **VENA-v3a** | VENA(fair) | **0.309** |
| 5 | C1-pGAN-t1pre | pixel | 0.09 | | 13 | C0-Identity | ref | 0.351 |
| 6 | C2-ResViT | pixel | 0.09 | | 14 | C6-3D-LDDPM | latent | 0.459 |
| 7 | C3-SynDiff-t2 | pixel | 0.12 | | 15 | C5-T1C-RFlow | latent | 0.506 |
| 8 | C3-SynDiff-t1pre | pixel | 0.18 | | 16 | C4-3D-DiT | latent | 0.725 |

**Paired Wilcoxon + matched-pairs rank-biserial (rb), N=247:**

| comparison | median Δρ_S | p | rb | reading |
|---|--:|--:|--:|---|
| v3a − C0-Identity | −0.077 | 0.017 | −0.18 | **fair arm beats identity — but small** (57%) |
| v3a − C5-T1C-RFlow | −0.148 | ~0 | −0.73 | **fair arm ≫ C5 (large)** |
| v3a − C4-3D-DiT | −0.361 | ~0 | −0.99 | **≫ C4 (near-total)** |
| v3a − C6-3D-LDDPM | −0.116 | 1e-15 | −0.59 | **≫ C6 (large)** |
| v3a − C1-pGAN-t1pre | +0.159 | ~0 | +0.66 | **pGAN beats v3a (large)** |
| v3a − C7-Pix2Pix | +0.505 | ~0 | +0.88 | **C7 beats v3a (very large)** |
| v3b-rw − C0-Identity | −0.216 | 1e-9 | −0.45 | oracle mask ~doubles the margin vs identity |
| v3b-rw − C1-pGAN-t1pre | +0.076 | 7e-6 | +0.33 | pGAN still beats oracle VENA (smaller) |

**Defensible ρ_S claims (all significance-backed):**
1. Latent **diffusion/flow** baselines (C4/C5/C6) are **worse than identity** on
   vessel fidelity — novel, important negative result for the field.
2. VENA (even fair v3a) **significantly reverses** that, with large effects vs
   C4/C5/C6 and a small-but-significant edge over identity.
3. The tumour-mask conditioning roughly **doubles** VENA's margin over identity.
4. **Pixel-domain 2D methods and the latent-GAN C7 beat VENA** on this axis
   (large effects) — the VAE-compression cost VENA cannot fully pay off.

**Claims we must NOT make:** "VENA has the best vessel fidelity" (false — pixel/C7
win); "whole-volume metrics and ρ_S agree" (they don't — that's the point).

---

## 3. What does NOT survive as a headline (delete from the manuscript)

- ✗ "VENA beats SOTA on fidelity" — 7th/16 (v3b-rw), best fair arm loses to pixel.
- ✗ "VENA has the lowest ρ_S / best vessel fidelity" — pixel methods and C7 are lower.
- ✗ "Region weighting helps" — net-negative ablation (Study 6).
- ✗ "3D-native ⇒ better z-consistency" — ZGD does not support it (C6 wins; VENA over-smooths).
- ✗ Any tumour/MS-SSIM/ΔDice win headlined without the oracle-mask caveat + v3a beside it.

---

## 4. Recommended framing (finding-led) — the paper's three contributions

1. **Methodological:** a pre-registered, **vessel-resolved** evaluation of
   gadolinium-free $T_{1c}$ synthesis (16 methods, 9 cohorts, 653 patients) that
   surfaces a failure whole-volume metrics miss.
2. **Empirical finding:** latent diffusion/flow synthesisers **misplace synthetic
   contrast onto vascular structure, worse than identity** — a caution for the
   fast-growing latent-diffusion contrast-synthesis literature (T1C-RFlow, LDDPM,
   DiT).
3. **Model:** VENA is the latent-generative recipe that **significantly mitigates**
   this failure (best of the latent diffusion/flow tier on ρ_S *and* MAE), at
   few-step inference — with an honest, quantified analysis of what a tumour-shape
   prior buys (oracle bound v3b-rw vs deployable v3a; predicted-mask arm pending).

This turns "we don't win" into "we found something the field needs to know, and
we built the fix within the paradigm." That is a legitimate, citable MedIA paper;
a marginal SOTA claim is not, and would be refuted in review.

---

## 5. If forced to a single (model, metric) headline

- **Model:** `VENA-S1-v3a` (fair) — with `v3b-rw` as the oracle upper bound.
- **Metric:** `ρ_S` (C-noT), framed per §0/§4 (VENA = the latent-tier fix, not the
  overall winner).
- **Support:** best-latent MAE_brain (0.0953) + on the cost–quality frontier.

Do **not** pick a whole-volume fidelity metric as the headline — every one has
VENA losing to the pixel tier.

---

## 6. The one experiment that could upgrade this to a fair model-win — T6.5

Run the **predicted-mask ablation** (Study 6 A5 / T6.5): train/infer v3b-rw with a
tumour mask predicted from **pre-contrast** inputs. If the predicted mask recovers
most of v3b-rw's ρ_S margin (rb vs identity −0.45 → still large without the GT
mask), then *"deployable VENA — no gadolinium, no ground-truth mask — is the only
latent method with significantly trustworthy contrast placement"* becomes a
**fair** headline. Highest-leverage remaining experiment; prioritise over figure
polish. (Cost: 1 training + 1 inference sweep; needs a pre-contrast WT segmenter.)

---

## 7. Verifications still owed before submission

- [ ] Holm-correct the ρ_S Wilcoxon family (§2 p-values are raw; the frozen
  `spatial_residual` artifact already Holm-corrects the competitor-16 family —
  reconcile §2 against `wilcoxon_results.csv`).
- [ ] Confirm the ρ_S ranking on **Ring B (OOD)** — does the latent-diffusion
  vessel failure hold pediatric/African? (Study 3 machinery.)
- [ ] VAE-floor (Study 7) to nail the "compression tax" root-cause claim.
- [ ] ET/TC-region fidelity (Study 8) — the tumour axis we actually want to lead.

---

## 8. How to chat about this with an agent — the doc set to hand it

To discuss the headline/contribution with a fresh agent, give it **exactly these**
(small, self-contained, no raw dumps):

1. **`.claude/notes/article/CONTRIBUTION.md`** (this file) — the decision + evidence.
2. **`.claude/notes/article/00_HUB.md`** — the shared contracts (registry §2.1,
   cohorts §2.2, stats §2.3, region taxonomy §2.6, honest framing §0).
3. **`.claude/notes/article/05_vessel_spatial_residual.md`** — the ρ_S metric
   definition + Study-5 design (the headline axis).
4. **`.claude/notes/article/01_paired_fidelity.md`** + **`02_cost_quality_pareto.md`**
   — the fidelity/cost context (where VENA sits).

For an agent that should *re-derive* numbers rather than trust the tables, also
point it at the two source CSVs (read-only):
`analyses/paired_fidelity/LATEST/per_scan/paired_fidelity_patient.csv` and
`analyses/spatial_residual/LATEST/per_scan/spatial_residual.csv`, plus the
registry `src/vena/validation/registry.py` for method roles / SELECTION_NFE.

Do **not** hand it 03/04/06/07/08/09 for a headline chat — they are downstream
studies and will dilute the discussion.
