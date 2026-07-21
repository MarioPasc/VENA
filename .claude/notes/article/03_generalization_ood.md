# Study 3 — Generalization & distribution shift

**Paper §4.4 · Priority: secondary · Data status: ✅ ready · Scope: per-cohort +
Ring B**

References `00_HUB.md` §2. This is Study 1's machinery re-cut **per cohort** and
**by ring**, to characterise how each method degrades under named shifts.

---

## 1. Question & claim

*How does synthesis fidelity degrade as the input distribution moves away from
the GE-3T adult-glioma training regime?* Same metrics as Study 1; the new axis is
the **named shift** (§2.2): multi-site → different institution → geographic
(African) → pathology (non-glioma) → **age (pediatric)**.

**Claim (hypotheses to confirm at render):** (i) fidelity degrades monotonically
Ring A → Ring B; (ii) **pediatric (BraTS-PED)** is the largest drop for every
method (adult-trained); (iii) the **tier gap persists OOD** (image tier still
leads whole-brain); (iv) *open sub-question*: does the oracle mask (`v3b-rw`)
help **more** OOD (anatomy less reliable) or **less** (mask still perfect)?
— report whichever the data shows.

## 2. Design

- **Primary unit:** per **ring** (A vs B) — statistically powered, and the
  pre-registered endpoint family split. Ring A and Ring B **never pooled** (§2.2).
- **Secondary unit:** per **cohort**, with n shown and **tiny cohorts (IvyGAP 5,
  REMBRANDT 5) flagged** — report point estimates + CI but do not run per-cohort
  significance on n=5 (underpowered; state this, don't hide it).
- **Degradation metric:** ΔMAE_brain(cohort) = MAE(cohort) − MAE(UCSF-PDGM
  in-dist), per method — a "shift penalty" that is directly comparable across
  methods.

Note: unlike Study 4, **absolute fidelity metrics remain valid on Ring B** here
(reconstruction error is well-defined regardless of scanner) — the segmenter-shift
caveat (§2.5③) is a Study-4 problem only.

## 3. Data — paths

**Source:** `paired_fidelity/LATEST/per_scan/paired_fidelity_patient.csv`
(`cohort`, `ring` columns already present). Convenience aggregates
`tables/method_cohort_agg.csv`, `tables/method_ring_agg.csv` exist — **rebuild
from per_scan** for consistent sel-NFE handling and patient collapse (esp.
LUMIERE 72→11). No generation needed. Task T1.3 (SSIM provenance) applies here too.

## 4. Tables

**Table 3A — by ring.** Rows = 16 methods (tiered). Columns = {MAE_brain,
MS-SSIM_brain} × {Ring A, Ring B} + the A→B degradation Δ. v3b-rw/v3a paired.

**Table 3B — by cohort (supp).** Rows = methods; columns = 9 cohorts (MAE_brain,
mean [CI], n badge). Heat-shade by degradation. This is the "where does each
method break" map.

**Table 3C — shift-penalty (supp).** ΔMAE vs UCSF-PDGM per method per cohort;
sort cohorts by mean penalty (expected: PED worst).

## 5. Figures

**Fig 3 — degradation ladder.** x = cohorts ordered by increasing shift severity
(UCSF → BraTS-GLI → UPENN → IvyGAP/REMBRANDT → Africa-Glioma → Africa-Other →
PED); y = MAE_brain; one line per method (tier-coloured), VENA-v3a/v3b-rw
highlighted. Shows the tier gap holding (or not) across the shift axis.

**Fig 3-supp — per-cohort violin** of patient MAE_brain for the 4 headline
methods (C2-ResViT, C5-T1C-RFlow, v3a, v3b-rw) — shows spread, not just means,
and where distributions overlap.

## 6. Reviewer objections & pre-emptions

| Objection | Pre-emption |
|---|---|
| "n=5 cohorts are noise" | Reported with n badges, no per-cohort tests; ring-level is the powered unit. |
| "Ring B mixes 3 different shifts" | Each cohort labelled by shift type (§2.2); PED/Africa reported separately, never averaged into one 'OOD' number. |
| "Longitudinal LUMIERE double-counts" | Scans collapsed to 11 patients before stats (§2.3). |
| "Why does method X improve OOD?" | If any method's MAE drops OOD, flag as a harmonisation/intensity-range artifact (Table S1 logic), not genuine gain. |

## 7. Task checklist

- [ ] **T3.1** Table 3A by-ring (rebuild from per_scan; Ring A/B separate).
- [ ] **T3.2** Table 3B per-cohort heat table + n badges.
- [ ] **T3.3** Table 3C shift-penalty; confirm PED is worst.
- [ ] **T3.4** Fig 3 degradation ladder + violin supp.
- [ ] **T3.5** Report the oracle-mask-OOD sub-question outcome (helps more/less).
