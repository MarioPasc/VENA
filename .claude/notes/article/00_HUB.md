# VENA — Article Validation Plan (HUB)

> Single source of truth for the paper's experimental section. Every study doc
> (`01`–`09`) references the **Shared Contracts** below rather than restating
> them. Target venue: *Medical Image Analysis* (MedIA). When this hub drifts from
> the frozen Phase-2 artifacts, **the artifacts win** — re-derive, don't guess.

Companion (methodology, not article) docs still authoritative for *how the
numbers were produced*: `.claude/notes/prompts/01_SHARED_CONTRACTS.md` (wins over
everything), `03_paired_fidelity.md`, `04_spatial_residual.md`,
`05_downstream_seg.md`, and `prompts/HANDOFF.md` (the post-sweep state of play).

---

## 0. Contribution framing — the honest, post-sweep version

The full pre-registered sweep (405 prediction files, 32,715 scans, 653 patients,
`paired_fidelity/LATEST` git_sha `285f203`) **did not** confirm the pre-sweep
story. The paper's spine must be built on what *survived*, not on the retired
claims. This framing governs every study doc.

> **⇒ Headline decision: see [`CONTRIBUTION.md`](CONTRIBUTION.md).** Bottom line
> (all re-derived 2026-07-20): **no *fair* VENA arm wins any metric outright** —
> the 2D pixel tier (ResViT/pGAN) leads whole-brain fidelity, and pixel methods
> **plus** the latent-GAN C7 lead the vessel axis ($\rho_S$). The recommended
> headline is **finding-led** (latent diffusion/flow misplaces synthetic contrast;
> VENA is the fix *within the latent tier*), **not** "VENA wins metric X". The
> bullets below are the surviving building blocks, read through that lens.

**What the paper IS (defensible after the sweep):**

1. **A pre-registered, vessel-/contrast-resolved benchmark** of gadolinium-free
   $T_{1c}$ synthesis: **16 methods × 9 cohorts × 2 OOD rings, 653 patients**,
   with a frozen pre-registration, patient-level statistics, and honest negative
   results. *The benchmark itself is a first-class contribution* — MedIA rewards
   methodological rigor over SOTA-chasing.
2. **VENA is the best latent-space method on whole-brain MAE** (v3a 0.0953,
   beating C4/C5/C6/C7; it *loses* MS-SSIM_brain to C5). But the **entire latent
   tier pays a VAE compression tax** — every latent method loses whole-brain
   fidelity *and* vessel fidelity ($\rho_S$) to the 2D pixel tier (Study 1 +
   Study 7 bound this by the VAE reconstruction floor).
3. **Tumour-shape conditioning as a bounded analysis**: oracle GT mask
   (`v3b-rw`, *upper bound*) vs no mask (`v3a`, *lower bound*), with a
   **predicted-mask** arm planned (Study 6) to estimate the deployable number.
   This is a novel, honest treatment of "how much does a tumour prior buy you".
4. **The spatial-residual / vessel-fidelity finding (Study 5) — the headline
   axis.** Latent *diffusion/flow* baselines (C4/C5/C6) place error on
   contrast-enhancing structure **worse than identity** ($\rho_S$ 0.46–0.73 vs
   0.35); VENA **significantly reverses** it (fair `v3a` 0.31, large effects vs
   C4/C5/C6, $p\le10^{-15}$; oracle `v3b-rw` 0.20, ~double the margin vs identity).
   **Caveat that MUST be stated:** 2D pixel methods (pGAN/ResViT, $\rho_S$
   0.03–0.09) and the latent-GAN **C7 ($-0.19$) beat VENA** on this axis — VENA is
   the fix *within the latent diffusion/flow tier*, not the overall winner. Full
   ranking + significance in [`CONTRIBUTION.md`](CONTRIBUTION.md) §2.
5. **Honest negatives, reported as results**: region weighting (`-rw`) is a net
   negative vs its own ablation; decoder perceptual loss (LPL) is inert. MedIA
   values these.

**Claims that did NOT survive — delete from the manuscript and our mental model:**

- ✗ "VENA beats SOTA by 43%" — an artifact of a throwaway audit script; does not
  reproduce in the routine.
- ✗ "VENA wins the primary endpoint" — on pre-registered MAE$_\text{brain}$ it
  ranks **7th of 16**, behind C2-ResViT, all pGAN panels, and *its own ablations*.
- ✗ "Region weighting improves the tumour" — n.s. on MAE$_\text{wt}$ (p=0.997)
  and SSIM$_\text{wt}$ (p=0.281); costs whole-brain MAE/SSIM.
- ✗ "Explicit vessel conditioning is VENA's headline mechanism" — scope changed;
  SWAN is an input, and the vessel result (Study 5) is a *comparative diagnostic*
  across all methods, not a VENA-exclusive win.
- ✗ "Conc(5%) shows VENA's errors avoid bright voxels" — a harmonisation
  artifact; **$\rho_S$ is the discriminator, Conc is not** (does not separate
  VENA from C5, p=0.86).

**The load-bearing integrity rule (applies to Studies 1, 4, 6):** `v3b-rw`
receives a **ground-truth 3-channel tumour mask (NETC / ED / ET sub-regions)** as ControlNet conditioning that
**no competitor receives**. Its tumour and downstream-seg wins are quantifiably
an oracle artifact (20/61 patients where synthetic beats *real* $T_{1c}$ at its
own segmentation). **`v3a` (no mask) must appear beside `v3b-rw` in every table
where the mask can leak.** Do not "fix" the leak — it is a frozen property of the
Phase-1 predictions. Report it. The cost–quality Pareto (Study 2) confirms the
same story a third way: **v3b-rw is strictly Pareto-dominated by v3a** (faster
*and* higher whole-brain MS-SSIM), so the mask buys nothing outside the tumour it
is handed.

---

## 1. Study index

| # | Study | Paper § | Data status | Priority | Doc |
|---|---|---|---|---|---|
| 1 | Paired fidelity (headline, Ring A) | 4.2 | ✅ **built+verified** | **primary** | `01_paired_fidelity.md` |
| 2 | Cost–quality Pareto (speed vs fidelity, all NFE) | 4.6 | ✅ **built+verified** | **primary** | `02_cost_quality_pareto.md` |
| 3 | Generalization / OOD (per-cohort + Ring B) | 4.4 | ✅ ready | secondary | `03_generalization_ood.md` |
| 4 | Downstream segmentation | 4.5 | ✅ ready | secondary | `04_downstream_segmentation.md` |
| 5 | Vessel / spatial-residual fidelity | 4.3 | ✅ ready | **primary** | `05_vessel_spatial_residual.md` |
| 6 | VENA ablations (mask / rw / LPL / NFE + planned) | 4.7 | ⚠️ partial | secondary | `06_vena_ablations.md` |
| 7 | VAE reconstruction ceiling | 4.2 (floor) | 🔨 needs-gen (cheap) | support | `07_vae_reconstruction_ceiling.md` |
| 8 | Enhancing-region (ET) fidelity | 4.2 (ext) | 🔨 needs-gen (re-run) | support | `08_enhancing_region_fidelity.md` |
| 9 | Shortcut / healthy-control safety (§6.5) | 4.8 | ⛔ needs-cohort | optional | `09_shortcut_healthy_control.md` |

Legend: ✅ all numbers on disk · ⚠️ mix of ready + planned · 🔨 requires a bounded
generation step · ⛔ blocked on data not yet sourced.

---

## 2. Shared Contracts

### 2.1 Method registry (16 methods)

Tier = generation space (the paper's primary grouping axis). "sel NFE" = frozen
selection NFE at which the method is compared. Modalities = network inputs.

| ID | Model | Tier | Inputs | sel NFE | Role / family |
|---|---|---|---|---|---|
| **C0-Identity** | $T_{1pre}$ pass-through | image (null) | T1pre | 1 | competitor (floor) |
| **C1-pGAN-t1pre** | pGAN (Dar 2019), 2D GAN | pixel | T1pre | 1 | competitor |
| **C2-ResViT** | ResViT (Dalmaz 2022) | pixel | T1pre,T2,FLAIR | 1 | competitor |
| **C3-SynDiff-t1pre** | SynDiff (Özbey 2023), 2D diff. | pixel | T1pre | 4 | competitor |
| **C4-3D-DiT** | 3D DiT (Eidex 2025) | latent | T1pre,FLAIR | 5 | competitor |
| **C5-T1C-RFlow** | T1C-RFlow (Eidex 2025), SOTA | latent | T1pre,FLAIR | 5 | competitor |
| **C6-3D-LDDPM** | 3D-LDDPM (Eidex 2025) | latent | T1pre,FLAIR | 1000 | competitor |
| **C7-3D-Latent-Pix2Pix** | latent Pix2Pix | latent | T1pre,FLAIR | 1 | competitor |
| C1-pGAN-t2 / -flair | pGAN single-source panels | pixel | T2 / FLAIR | 1 | supplementary (no test) |
| C3-SynDiff-t2 / -flair | SynDiff single-source panels | pixel | T2 / FLAIR | 4 | supplementary (no test) |
| **VENA-S1-v3a** | VENA, concat-only, **no mask** | latent | T1pre,T2,FLAIR | 5 | VENA ablation (no-oracle) |
| **VENA-S1-v3b** | VENA, ControlNet, no region-wt | latent | +GT 3ch mask | 5 | VENA ablation |
| **VENA-S1-v3b-rw** | VENA, ControlNet + region-wt loss | latent | +GT 3ch mask | 5 | **VENA headline** (pre-registered) |
| VENA-S3-LPL-b2c | VENA + decoder perceptual loss | latent | +GT 3ch mask | 5 | VENA ablation (LPL null) |

**Test families (Holm-corrected separately):** competitor family = 8 (C0–C7
primary panels); VENA-ablation family = 3 (v3a, v3b, S3-LPL); supplementary
panels (4) reported but in no family. All 16 rows appear in every table.

**Fairness facts to disclose in captions** (see §2.5): modality count differs
(pGAN/SynDiff 1 · C4–C7 2 · VENA/ResViT 3); v3b/v3b-rw/LPL receive the GT
3-channel tumour sub-region mask [NETC,ED,ET] (§2.6); all latent methods (incl. C5)
decode through the *same* frozen
`autoencoder_v2.pt` (the proposal's "custom VAE for C5" claim was wrong — the
real disk state is fairer).

### 2.2 Cohort → ring → shift (authoritative, HANDOFF-corrected)

Ring A and Ring B are **separate endpoint families — never pooled**. Collapse
longitudinal scans to patients before every statistic (LUMIERE 72→11).

| Cohort | Ring | scans | patients | Shift from training distribution |
|---|---|---:|---:|---|
| **UCSF-PDGM** | A | 50 | 50 | **in-distribution held-out test** (GE 3T, single site) |
| BraTS-GLI | A | 127 | 114 | multi-site / multi-vendor |
| UPENN-GBM | A | 62 | 62 | different institution, GBM-only |
| IvyGAP | A | 5 | 5 | single site (small n) |
| LUMIERE | A | 72 | 11 | single site, **longitudinal** |
| REMBRANDT | A | 5 | 5 | older cohort (small n) |
| **Ring A** | — | **321** | **247** | adult glioma, near-distribution |
| BraTS-Africa-Glioma | B | 95 | 95 | **geographic OOD** (sub-Saharan Africa) |
| BraTS-Africa-Other | B | 51 | 51 | geographic **+ pathology** OOD (non-glioma) |
| BraTS-PED | B | 260 | 260 | **age OOD** (pediatric) |
| **Ring B** | — | **406** | **406** | out-of-distribution |
| **TOTAL** | — | **727** | **653** | — |

### 2.3 Statistical protocol (frozen in pre-registration)

- **Primary endpoint:** MAE on `brain`, **Ring A**, each method at its `sel NFE`.
- **Primary test:** paired two-sided **Wilcoxon signed-rank**, `v3b-rw` vs each
  competitor, α = 0.05. (`v3a` reported as the no-oracle counterpart everywhere.)
- **Effect size:** **Cliff's δ** per comparison. *(Your pre-registration mandates
  Cliff's δ — this supersedes the earlier "Cohen's d" ask. If a parametric
  secondary is wanted, add Cohen's $d_z$ as a supplement, never as primary.)*
- **CIs:** patient-stratified **bootstrap, 10,000 resamples**, seed pinned.
- **Multiple comparisons:** **Holm–Bonferroni within each (metric, region)
  cell**, families partitioned by role (competitor 8 / ablation 3).
- **MCID = 0.01** on the [0,1] scale — wins below this are "statistically but not
  clinically significant". State this threshold in every fidelity caption.
- **NFE:** every method at its own `sel NFE`; **plus a matched-NFE=5 sub-table**
  for the few-step latent tier (the only apples-to-apples generative comparison).
- **Any reduction of an arm to one value per patient goes through one shared
  helper** — `_shared.filter_to_selection_nfe` in the study layer (re-exports the
  sweep's `_filter_to_selection_nfe`); the sweep's own is `_series_at_selection_nfe`.
  The NFE-asymmetry bug (HANDOFF §6) came from two code paths; never re-implement
  with a bare `groupby("method")`.

### 2.4 Scoring-space rule (load-bearing; HANDOFF §4/§5)

Phase-1 pre-normalised every prediction to [0,1] via
`percentile_normalise(lower=0, upper=99.5, foreground_only=True)`. Phase-2 does
**not** re-normalise. Per (scan, method): if `raw_p995 ≤ 1.05 and raw[brain].min
≥ −0.05` → score `raw`; else score `harmonised`. In practice **C0-Identity is
the only method scored `harmonised`** (its raw p99.5 ≈ 1424–2466); all other 15
score `raw`. `data_range = 1.0` for PSNR/SSIM/MS-SSIM. Under-saturation
(e.g. C4-3D-DiT p99.5 = 0.38) is a **reportable finding (Table S1), not a bug**.

### 2.5 Global limitations to state (not fix)

| # | Limitation | Where it bites |
|---|---|---|
| ① | Modality-count asymmetry (1 / 2 / 3 inputs) | every fidelity comparison caption |
| ② | Oracle GT 3-channel tumour mask [NETC,ED,ET] for v3b/v3b-rw/LPL | Studies 1, 4, 6 — v3a mandatory beside |
| ③ | SegResNet (BraTS-pretrained) degrades on Ring B | Study 4 — only Δ valid on Ring B, not absolute Dice |
| ④ | Phase-1 double-harmonisation / under-saturation | Study 1 Table S1 |
| ⑤ | VENA sampling unseeded (`torch.randn_like`) → not bit-reproducible | Methods + Limitations |
| ⑥ | pGAN/SynDiff run as single-source (t1pre chosen a priori) | Study 1 caption |
| ⑦ | SSIM-region proxy (mean-fill) not publication-defensible | Study 1 task T1.3 |
| ⑧ | $\rho_S$ measures error–vs–bright-intensity, vessels dominant but not exclusive | Study 5 framing |

### 2.6 Tumour region taxonomy (BraTS) — corrected 2026-07-20

VENA's `derive_sub_labels` (`src/vena/validation/downstream_seg.py`) handles both
label systems (selected per-cohort by the H5 attr `label_system`):

| Region | Composition | BraTS2021 | BraTS2023 | Edema? |
|---|---|---|---|---|
| **WT** whole tumour | NCR + edema + ET | labels {1,2,4} (all >0) | {1,2,3} (all >0) | **includes edema** |
| **TC** tumour core | NCR + ET | {1,4} | {1,3} | **excludes edema** |
| **ET** enhancing tumour | enhancing only | {4} | {3} | n/a |

**Correction:** "everything except the edema" is **TC**, *not* WT — WT *includes*
edema. The region that is "enhancing + non-enhancing solid tumour, no edema" is
**TC**. Fidelity currently scores only `wt` (+ brain, bg); **TC and ET fidelity
require the Study-8 re-run (T8)** — see Study 8 for the lead-region recommendation.

**What v3b-rw actually conditions on / weights** (config
`routines/fm/train/configs/runs/picasso_s1_v3b_rw_concat_plus_cn3ch_fft.yaml`):

- **ControlNet conditioning** = a **3-channel soft mask [NETC, ED, ET]** (the full
  tumour decomposed into sub-regions) — *not* a single WT channel. v3a conditions
  on none (channel-concat of {t1pre,t2,flair} only); v3b/v3b-rw add the 3-channel
  ControlNet on top.
- **Region-weighted (`-rw`) loss** = multiplicative per-voxel weights on the L1
  velocity loss (`build_region_weight_tensor`, τ=0.5): **ET ×300, NETC ×50,
  ED ×50, background / brain-not-tumour ×1**. Raises WT from 0.095 % → ~17 % of
  loss mass. (Scalar weights, *not* exponents; the retired contrastive-v0.4
  "p=3/p=1" note does not apply here.)

---

## 3. Data locations

Archive root (local, byte-verified vs Picasso):
`/media/mpascual/Sandisk2TB/research/vena/results/fm/inference/analyses/`

| Routine | `LATEST` | Source-of-truth file |
|---|---|---|
| paired_fidelity | `paired_fidelity/LATEST/` (2026-07-17T11-54-55Z) | `per_scan/paired_fidelity_patient.csv` |
| spatial_residual | `spatial_residual/LATEST/` (2026-07-20T08-41-41Z, Holm-corr.) | `per_scan/spatial_residual.csv` |
| downstream_seg | `downstream_seg/LATEST/` (2026-07-18T15-58-58Z) | `per_scan/downstream_seg.csv` |
| preregister | `preregister/LATEST/` (2026-07-17T09-33-09Z) | `ring_partitions.json`, `decision.json` |

**Rule for every table/figure in the paper:** rebuild from the `per_scan` CSV
(source of truth). Aggregate `tables/*.csv` are convenience cuts and may lag a
correction (they did once — HANDOFF §6). Predictions H5 schema and the full file
inventory: `results/fm/inference/README.md` and `.../analyses/README.md`.

**Article-artifact output (the `routines/validation/studies/` layer).** The study
routines *consume* `inference/` + `inference/analyses/` and *emit* organised,
article-level tables + figures to:
`/media/mpascual/Sandisk2TB/research/vena/results/article/<study>/<UTC>/{tables,figures,decision.json}`
(+ a `LATEST` symlink per study). Each study reuses `vena.validation.registry` +
`vena.validation.stats` and the shared `routines/validation/studies/_shared.py`
(DOMAIN grouping, per-scan resolver, the one selection-NFE reducer). Studies never
re-run a sweep and never reimplement a statistic.

Predictions (Picasso): `picasso:~/execs/vena/inference/` (405 + 45 files, 9
cohorts). Run all *new* analysis from `fscratch/repos/VENA-validation` (a real
git repo — `git_sha` resolves).

---

## 4. Cross-study generation task tracker

Data that must be *created* (paths above are read-only until then). Each task is
detailed in its study doc; this is the roll-up.

| Task | For | Cost | Blocker |
|---|---|---|---|
| T1.3 Recompute region-SSIM from full SSIM map (drop mean-fill proxy) | 1,3 | CPU, small | verify what's in per_scan |
| T7 VAE recon floor: encode→decode real $T_{1c}$, score vs real | 7 | 1 GPU-h | none (images on disk) |
| T8 Re-run fidelity with **ET** (and TC) region masks on frozen preds | 8 | CPU sweep, ~re-run of §4.2 | none |
| T5.4 (opt.) True vessel-mask conspicuity (Frangi synth vs real, C-noT) | 5 | CPU | design metric |
| T6.5 (planned) Predicted-mask v3b-rw: segmenter on pre-contrast → retrain → re-infer | 6 | **new training + inference** | user go/no-go |
| T6.6 (candidate) Input-count ablations (T1pre / +T2 / +FLAIR / +SWAN) | 6 | **N new trainings** | user go/no-go |
| T9 Source healthy-control cohort; false-positive-enhancement protocol | 9 | data + inference | **cohort not on disk** |
| F-visual Visual-inspection panel (Study 1) — design + render | 1 | render | design (see 01 §7) |

---

## 5. Open decisions (parked for the user)

1. **Primary figure of the paper** — is it the Ring-A headline table (Study 1),
   the Pareto (Study 2), or the vessel-residual panel (Study 5)? Drives the abstract.
2. **Predicted-mask ablation (T6.5)** — go/no-go and *which* pre-contrast
   segmenter (BraTS SegResNet on {T1pre,T2,FLAIR}? a dedicated model?).
3. **Input-count ablations (T6.6)** — commit compute, or ship as future work?
4. **Reader study** — you deferred it; confirm it stays out of *this* submission
   (it materially raises acceptance odds but is Phase-5 effort).
5. **ET vs WT as the tumour headline region** — Study 8 argues ET is the
   money metric for a *contrast* paper; confirm we re-run and lead with ET.

---

*Last updated: 2026-07-20. Studies 1–9 authored from HANDOFF (post-sweep) +
Phase-2 design docs + on-disk `per_scan` schema. Numbers not yet independently
re-derived from CSVs where marked "spec"; re-derive at render time.*
