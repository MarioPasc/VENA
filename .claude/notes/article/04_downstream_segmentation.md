# Study 4 — Downstream tumour segmentation

**Paper §4.5 · Priority: secondary · Data status: ✅ ready · Scope: patients with
GT seg (≈242)**

References `00_HUB.md` §2. **This study leaks the oracle mask harder than any
other (§2.5②) — the integrity rule is non-negotiable here.**

---

## 1. Question & claim

*Is a synthetic $T_{1c}$ good enough to substitute for the real one in a
downstream tumour-segmentation pipeline?* Run a fixed BraTS segmenter on the
4-channel input with **real vs synthetic $T_{1c}$**, measure the Dice drop.

**Claim:** the deployable answer lies **between** `v3a` (no mask, honest
ΔDice_ET ≈ 0.435) and `v3b-rw` (oracle mask, ΔDice_ET ≈ 0.073) — the latter is an
**upper bound only**, because in **20/61 patients the synthetic image segments
the tumour *better than the real* $T_{1c}$**, which is impossible without the GT
mask leaking into the metric. The leak is **most direct for ET-Dice**: v3b-rw is
conditioned on the GT **ET sub-region channel itself** (the 3-channel [NETC,ED,ET]
mask, Hub §2.6), so the segmenter is handed the answer to the very region it is
scored on. The honest downstream cost of synthesis is the `v3a` number; `v3b-rw`
shows the ceiling *given a perfect tumour prior*.

## 2. Protocol (fixed, from Phase-2)

- **Segmenter:** MONAI Model-Zoo `brats_mri_segmentation` (SegResNet,
  BraTS-pretrained). **Not nnU-Net** (not installed — do not add).
- **Input:** 4-channel {FLAIR, T1ce, T1, T2}, all harmonised. **Only the T1ce
  channel changes** between arms (real vs synthetic); other 3 channels identical.
- **Metric:** $\Delta\text{Dice}(m, \ell) = \text{Dice}(\text{real}) −
  \text{Dice}(\text{synth})$ for label $\ell$. **Positive Δ = degradation.**
- **Primary label:** **ET-Dice** (enhancing tumour — the region gadolinium
  reveals). WT, TC secondary.
- **Paired subset:** patients with GT `masks/tumor` in the corpus H5 (≈242 in the
  full sweep; the earlier 61-patient paired set is superseded).

## 3. Data — paths

**Source:** `downstream_seg/LATEST/per_scan/downstream_seg.csv`
Columns: `method, cohort, ring, nfe, scan_id, patient_id, pred_mode,
wt_join_dice, dice_{wt,tc,et}_{real,synth}, delta_{wt,tc,et}`.
Aggregates `tables/method_cohort_agg.csv`, `method_ring_agg.csv`,
`wt_join_dice_per_scan.csv`; figure `figures/delta_wt_per_method.png` exist.
**No generation needed** — render + stats only.

Collapse scans→patients; ET may be empty → **NaN, not 0** (a patient with no
enhancing tumour cannot contribute an ET-Dice). Report n per cell after NaN drop.

## 4. Tables

**Table 4 (main) — ΔDice by region, in-distribution (Ring A).** Rows = 16
methods (tiered, v3a & v3b-rw adjacent + labelled ⊕oracle). Columns = ΔDice_ET |
ΔDice_TC | ΔDice_WT, each `mean [CI]`, Wilcoxon vs v3b-rw, Holm per label-cell.
Extra column: **"# patients synth > real"** (the leak counter) — this single
column makes the oracle argument visually undeniable.

**Table 4B — per-shift (Ring B).** ΔDice_ET per Ring-B cohort (Africa-Glioma,
Africa-Other, PED). **Caption: absolute Dice on Ring B is uninterpretable
(segmenter domain shift, §2.5③) — only Δ is valid**, and even Δ is noisier there.

**Reference row:** always include `dice_*_real` medians (the ceiling: WT/TC ≈
0.92, ET ≈ 0.85) so ΔDice is anchored.

## 5. Figures

**Fig 4A — ΔDice_ET forest** (mean + CI), tiered, with the leak-counter annotated
per method; horizontal line at Δ=0 (perfect substitute) and a marker where the
count of "synth > real" is non-trivial.

**Fig 4B — qualitative panel (your spec): one row per dataset, one column per
competitor, showing the synthetic $T_{1c}$ only.**
- Rows = 9 cohorts (or a curated 6: UCSF, BraTS-GLI, UPENN, Africa-Glioma,
  Africa-Other, PED). Columns = a curated method set (real T1c | C2-ResViT |
  C5-T1C-RFlow | v3a | v3b-rw).
- Enhancement: overlay **the segmenter's ET contour** (synth-driven) vs the **GT
  ET contour** on each tile — turns "here is a T1c" into "here is what the
  downstream model *sees*". This directly visualises the leak (v3b-rw contours
  hug GT suspiciously) and the OOD failure (PED row).
- Fixed per-tile window anchored to the real slice (§2.4 convention).

## 6. Reviewer objections & pre-emptions

| Objection | Pre-emption |
|---|---|
| "v3b-rw beats real — impossible, bug?" | Not a bug: the GT mask conditions v3b-rw and the metric is scored against the same GT. Reported as the oracle leak, with the count. |
| "Which segmenter? nnU-Net is standard" | SegResNet BraTS Model-Zoo, fixed, pretrained; nnU-Net not installed. State it; it's a *relative* comparison (Δ), so absolute segmenter strength cancels. |
| "Absolute Dice low on Africa" | Segmenter domain shift; only Δ reported on Ring B, with the caveat. |
| "ET empty patients skew means" | NaN-dropped, n reported per cell. |

## 7. Task checklist

- [ ] **T4.1** Table 4 main (Ring A) with leak-counter column; verify v3a ΔDice_ET ≫ v3b-rw.
- [ ] **T4.2** Table 4B per-shift ΔDice_ET (Ring B) + absolute-invalid caveat.
- [ ] **T4.3** Fig 4A forest with leak annotation.
- [ ] **T4.4** Fig 4B qualitative grid (row=cohort, col=method, synth T1c + ET contours).
- [ ] **T4.5** Confirm paired-subset n and NaN handling match the sweep's report.md.
