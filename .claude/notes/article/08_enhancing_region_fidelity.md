# Study 8 — Tumour sub-region fidelity (WT · TC · ET)

**Paper §4.2 (extension) · Priority: support · Data status: 🔨 needs-gen (fidelity
re-run on frozen preds) · Scope: Ring A (+ Ring B for supp)**

References `00_HUB.md` §2.6 (region taxonomy) and Study 1 (`01_paired_fidelity.md`).
Adds the tumour-core (TC) and enhancing (ET) fidelity that Study 1's whole-tumour
(WT) column cannot isolate — the clinical heart of a contrast-synthesis paper.

---

## 1. The three tumour regions and which one leads

Per the corrected BraTS taxonomy (Hub §2.6 — `derive_sub_labels`):

| Region | = | Excludes edema? | What it tests | In data now? |
|---|---|---|---|---|
| **WT** | NCR + edema + ET | **no** (includes edema) | whole abnormality; edema is ~T1c-invisible → **dilutes** | ✅ (Study 1) |
| **TC** | NCR + ET | **yes** | solid tumour: bright enhancing **and** dark non-enhancing core | 🔨 T8 |
| **ET** | enhancing only | — | the pure gadolinium signal / BBB breakdown | 🔨 T8 |

**Correcting the premise (2026-07-20):** "the whole tumour except the edema" is
**TC**, not WT — WT *includes* edema. The user's stated goal — *"replicate both
the enhancing and non-enhancing parts of the tumour, excluding edema"* — is
therefore **TC** exactly.

**Why TC is the right lead, and it aligns with the model's own design.** v3b-rw
conditions on the 3 sub-region channels [NETC, ED, ET] and its region-weighted
loss weights **ET ×300, NETC ×50, ED ×50** (Hub §2.6) — i.e. the model is trained
to get the enhancing rim *bright* **and** the necrotic core *dark* (both inside
TC), while edema is quasi-irrelevant on $T_{1c}$. Leading on TC rewards exactly
that behaviour and penalises a model that lazily brightens the whole tumour.

**Recommended tumour reporting (supersedes the earlier "lead with ET" note and
Hub §5.5):**
- **Lead the main table with TC** — matches the scientific goal and the model's
  emphasis; excludes the T1c-invisible edema that only adds noise to WT.
- **Report ET beside it** — the enhancement-specific number a reviewer checks
  first, and where the oracle-mask gain concentrates.
- **Keep WT** (with edema) as the broadest region for literature comparability
  (most synthesis papers report WT/TC/ET; we should too).

A single caveat for TC/ET: they are *smaller* regions than WT, so the SSIM-region
treatment (Study-1 T1.3) matters even more — likely bbox-SSIM + documented; drop
MS-SSIM where 4-level support is insufficient and say so.

## 2. Claim / hypothesis

- ET fidelity separates methods *more* than WT (the enhancing rim is the hard,
  high-frequency structure GANs blur and latent methods smear).
- The oracle-mask gain (v3b-rw vs v3a) is **larger on ET than WT** (the mask most
  helps exactly where enhancement concentrates) — quantify it; it sharpens the
  mask-conditioning bound (Studies 1, 6).

## 3. Generation task T8

Re-run the **fidelity scoring** with **TC** (tumour core) and **ET** (enhancing)
region masks, on the **frozen Phase-1 predictions** (no re-inference).
- Region source: corpus H5 `masks/tumor` multi-label → derive WT/TC/ET with
  **`vena.validation.downstream_seg.derive_sub_labels(tumor, label_system)`** —
  it already handles both label systems (BraTS2021 ET=label 4 vs BraTS2023 ET=label
  3), keyed off the per-cohort H5 `label_system` attr. Do NOT hard-code label 4.
- Add regions to the fidelity scorer's region list; the routine already computes
  per-region MAE/RMSE/PSNR/SSIM/MS-SSIM — this is adding two region masks, not new
  metrics.
- Apply the **same SSIM-region treatment** decided in Study-1 T1.3 (full-map
  average or bbox — ET is even smaller than WT, so mean-fill is *especially*
  indefensible here; likely ET-bbox SSIM + report the choice).
- Emit `paired_fidelity_patient.csv` **v2** with `*_et`, `*_tc` columns, or a
  sibling `paired_fidelity_et.csv`. Keep the frozen v1 intact.

**Cost:** a re-run of the CPU fidelity sweep (~195 CPU-h at the measured
21.5 CPU-s/vol × 32,715, but restrictable to Ring A + sel-NFE → far less; shard
per prediction file like the original). **No GPU, no re-inference.**

## 4. Placement

Extends Study 1: add `MAE_et | PSNR_et | SSIM_et | MS-SSIM_et` block to Table 1
(or a dedicated **Table 8**), and an ET row-pair (v3a/v3b-rw) with the mask-gain
delta. If you confirm Hub §5.5, ET becomes the *lead* tumour column and WT moves
to supp.

## 5. Reviewer objections & pre-emptions

| Objection | Pre-emption |
|---|---|
| "WT hides the enhancement result" | ET added; leads the tumour fidelity. |
| "ET SSIM on a tiny region" | ET-bbox SSIM + documented; MS-SSIM only where 4-level support exists, else drop MS-SSIM_et and say so. |
| "ET empty in some patients" | NaN, not 0; report n per cell (same rule as Study 4). |
| "Re-run could change v1 numbers" | v1 frozen; ET is additive columns; brain/wt/bg unchanged (verify by diffing shared columns). |

## 6. Task checklist

- [ ] **T8.1** Add ET, TC region masks to the fidelity scorer (BraTS label convention).
- [ ] **T8.2** Re-run fidelity on frozen preds (Ring A + sel-NFE first; extend if needed); emit et/tc columns without touching v1.
- [ ] **T8.3** Verify shared (brain/wt) columns are byte-identical to v1.
- [ ] **T8.4** ET table + mask-gain-on-ET delta; decide ET-vs-WT lead (Hub §5.5).
- [ ] **T8.5** Propagate ET to Study 7 (VAE floor on ET) and Study 6 (mask ablation on ET).
