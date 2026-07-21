# Study 1 — Paired fidelity (headline)

**Paper §4.2 · Priority: PRIMARY · Data status: ✅ BUILT + VERIFIED (one recompute
task, T1.3) · Scope: Ring A, patient-collapsed (247 patients)**
**Routine:** `routines/validation/studies/paired_fidelity_study.py` → artifacts at
`results/article/paired_fidelity/LATEST/` (independently verified 2026-07-20:
v3b-rw MAE_brain 0.0955 rank **7/16**, MAE_wt 0.0948 vs v3a 0.1283 Holm-p<1e-4;
n=247; git_sha `cebc078`).

References Shared Contracts in `00_HUB.md` (§2.1 registry, §2.3 stats, §2.4
scoring, §2.5 limitations). Do not restate them here.

---

## 1. Question & claim

*How faithfully does each method reconstruct post-contrast $T_{1c}$ from
pre-contrast inputs, in-distribution?* This is the headline fidelity table.

**Honest claim (post-sweep):** VENA is the **best latent-space method** on
whole-brain fidelity but does **not** win the primary endpoint outright — the
image-domain tier (C2-ResViT, C1-pGAN) leads whole-brain MAE, a **VAE
compression tax** shared by every latent method (bounded in Study 7). Tumour-region
fidelity separates only with the **oracle mask** (`v3b-rw`), which is reported as
an *upper bound* beside the no-mask *lower bound* (`v3a`).

## 2. Falsifiable hypotheses

- **H1** (tier effect): whole-brain MAE(image tier) < MAE(latent tier), method-paired. *Supported in sweep: every image method beats every latent method on MAE_brain.*
- **H2** (VENA-best-latent): VENA-v3a ≤ every other latent competitor on MAE_brain. *Supported: VENA 0.0950 < C7 0.0994 < C5 0.1016 < C6 0.1112 < C4 0.1976.*
- **H3** (oracle effect): MAE_wt(v3b-rw) < MAE_wt(v3a), and the gap is **absent** whole-brain. *Supported: −0.0335 MAE_wt, +0.0001 MAE_brain.*
- **H0-neg** (region weighting): v3b-rw improves tumour vs v3b. *Rejected: n.s.*

## 3. Methods & grouping

All **16** rows (§2.1). **Group rows by tier** with a visual rule inside the table:

```
── Reference ──────────  C0-Identity
── Pixel / image ──────  C1-pGAN-t1pre · C2-ResViT · C3-SynDiff-t1pre
── Latent ─────────────  C4-3D-DiT · C5-T1C-RFlow · C6-3D-LDDPM · C7-Latent-Pix2Pix
── VENA (latent) ──────  v3a (no mask)  ·  v3b-rw (GT 3ch mask, headline)
   supplementary panels: pGAN-t2/flair, SynDiff-t2/flair  (grey, no test)
   ablation-only rows moved to Study 6: v3b, S3-LPL
```

Bold the best value per column *within each tier*; underline the global best.
Every row carries a modality-count badge {1,2,3} and a mask badge (GT⊕ 3-channel
[NETC,ED,ET] for v3b/v3b-rw).

**Tumour region (read with Hub §2.6):** Study 1's tumour column is **WT** (whole
tumour — *includes* edema), the only tumour region in the current data. The
design-aligned regions are **TC** (solid tumour, no edema — the model's actual
target) and **ET** (enhancement); both require the Study-8 re-run (T8), and once
available **TC leads** the tumour column (Study 8 §1). Do not describe WT as
"tumour minus edema" — that is TC.

## 4. Data — paths & tasks

**Source (read-only):**
`…/analyses/paired_fidelity/LATEST/per_scan/paired_fidelity_patient.csv`
Columns used: `method, cohort, ring, nfe, patient_id, mae_{brain,wt,bg_undilated},
rmse_*, psnr_*, ssim_*, ms_ssim_*, zgd, raw_p995`.

Filter: `ring == "A"`, reduce to `sel NFE` per method via
`_shared.filter_to_selection_nfe`, collapse scans→patients (mean within patient), then
patient-stratified bootstrap 10k for CIs and paired Wilcoxon + Cliff's δ vs
`v3b-rw` (and vs `v3a` as the no-oracle counterpart).

**Task T1.3 (must do before publication):** confirm whether `ssim_wt` / `ms_ssim_wt`
in the CSV are the **full-map-averaged-over-region** value or the retired
training-time **mean-fill proxy** (indefensible for WT < 1% of volume, §2.5⑦). If
proxy, recompute from the full SSIM map (MONAI 1.5.2 `SSIMMetric` with
`return_full_image=True`, average inside the region) or fall back to WT-bbox SSIM
and document in the caption. **`brain`/`bg` SSIM are unaffected** (large regions).

Everything else in this study is a table/figure render — **no data generation**.

## 5. Metrics & statistics

Per §2.3. Primary ranking metric = **MAE**. Report MAE, PSNR, SSIM, MS-SSIM for
each region {brain, wt, bg_undilated}; `zgd` as a 2D-artifact discriminator
(§6, sub-table). Cell format: **mean [95% CI]**, with a significance glyph vs
v3b-rw (Holm-corrected within the metric×region cell) and Cliff's δ. Mark
sub-MCID wins (|Δ| < 0.01) explicitly.

## 6. Tables

**Table 1 (main) — Ring-A fidelity, headline.** Rows = 16 methods (tiered).
Columns (grouped): `MAE_brain ↓ | PSNR_brain ↑ | SSIM_brain ↑ | MS-SSIM_brain ↑`
then the same block for `_wt`. Each cell `mean [lo, hi]`; append `†` (Holm-sig vs
v3b-rw), Cliff's δ subscript. Footnote: modality counts, oracle mask, MCID, n=247.

**Table 2 (supp S1) — under-saturation audit.** `raw_p995` per method (mean) +
scored-space flag (raw/harmonised). Shows why C0 is harmonised and C4 is
penalised (§2.4). Pre-empts the "unfair harmonisation" reviewer objection.

**Table 3 (supp S2) — z-gradient (`zgd`).** Per method; flags the 2D tier
(pGAN/SynDiff, ZGD > 1 → inter-slice discontinuity) vs 3D methods (ZGD ≈ 1). A
cheap, striking argument for 3D-native synthesis.

**Table 4 (supp) — bg_undilated** (brain∖wt) MAE/SSIM — confirms tumour effects
are tumour-local, not bulk-tissue. (v3b-rw vs v3a differ in wt, identical in bg.)

## 7. Figures

**Fig 1 — visual-inspection panel (design; render task F-visual).**
Proposed layout, reusing the exhaustive-val figure conventions
(`vena.model.fm.eval.exhaustive.render_comparison_figure`: black background,
per-slice intensity window anchored to the *real* slice's (min,max)):

- **Rows** = a curated set of methods in tier order: `real T1c` (reference),
  C0, C1-pGAN, C2-ResViT, C5-T1C-RFlow, **v3a**, **v3b-rw**. (7 rows keeps it legible.)
- **Columns** = 3 representative Ring-A patients (one clean glioma, one with a
  large enhancing rim, one multifocal) at a fixed axial slice through the tumour.
- **Overlays:** a zoomed inset on the enhancing rim; a difference map (|synth −
  real|) strip on the right of each panel with a shared colourbar.
- Caption states NFE per method and that v3b-rw used the GT mask.
- Pick patients by **median** MAE_brain (representative, not cherry-picked
  best/worst) + note the selection rule in the caption (anti-cherry-pick).

**Fig 2 — forest family, one per (metric, region)** (2026-07-20 spec).
`fig_forest_{metric}_{region}.png` for metric ∈ {mae, psnr, ms_ssim, ssim} ×
region ∈ {brain, wt, bg_undilated} (12 figures; MAE kept as the primary-endpoint
forest). Each: white background, point = mean + 95 % CI whiskers. Encoding:

- **rows grouped by space** {pixel, latent, reference(C0)}, and **within each
  group sorted by descending performance** for that metric (best at the top).
- **significance vs VENA drawn on the plot**: Holm-adjusted p of each method vs
  `v3b-rw` (per family) as `*** / ** / * / ns`; `v3b-rw` = `ref`; supplementary =
  `n/t`. VENA arms highlighted; reference line at C0 (and dashed at v3b-rw).
- caption carries n=247 and MCID=0.01; the VAE-floor line (Study 7) overlays the
  `mae_brain` and `ssim/ms_ssim_brain` panels once available.

One glance per panel shows the tier gap, where VENA sits, and which competitors
are (not) significantly separated from it.

## 8. Reviewer objections & pre-emptions

| Objection | Pre-emption |
|---|---|
| "VENA doesn't win — why publish?" | Reframe (Hub §0): rigorous benchmark + best-latent + vessel result + bounded mask analysis. Lead with honesty. |
| "v3b-rw's mask is unfair" | v3a beside it in every row; the mask effect is *quantified* as a bounded ablation, and Study 6 adds the predicted-mask arm. |
| "Harmonisation inflated some methods" | Table S1 (under-saturation audit) + the raw/harmonised scoring rule stated. |
| "SSIM on a <1% region is meaningless" | T1.3 recompute; report WT-bbox fallback; note MS-SSIM needs sufficient support (4-level). |
| "Single-source pGAN/SynDiff is a weak baseline" | t1pre chosen a priori (pre→post direction); all input panels reported (supp); picking on test = oracle selection, refused. |
| "Only n=50 truly in-distribution" | Headline is Ring A (247, adult glioma near-dist by design, per user); UCSF-only cut available as a supp row; Ring B is Study 3. |

## 9. Task checklist

- [ ] **T1.1** Build Table 1 from `per_scan` (Ring A, sel-NFE, patient-collapse, bootstrap, Wilcoxon+δ, Holm). Verify VENA rank (expect 7th on MAE_brain).
- [ ] **T1.2** Tables S1 (under-sat), S2 (zgd), S-bg.
- [ ] **T1.3** Resolve SSIM-region provenance; recompute if mean-fill proxy.
- [ ] **T1.4** Fig 2 forest plot (needs Study-7 floor for the reference line).
- [ ] **T1.5** Design + render Fig 1 visual panel (median-MAE patient selection).
- [ ] **T1.6** Cross-check every number against `paired_fidelity/LATEST/report.md` caveats (oracle gap, full ranking) — they derive from the same rows and must agree.
