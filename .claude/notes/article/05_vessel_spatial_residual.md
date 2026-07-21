# Study 5 — Vessel & contrast-structure fidelity (spatial residual)

**Paper §4.3 · Priority: PRIMARY · Data status: ✅ ready (+1 optional gen) ·
Scope: Ring A, condition C-noT**

References `00_HUB.md` §2. This is the mechanistic finding that discriminates
methods where whole-volume metrics saturate, and the honest reframing of VENA's
original vessel motivation.

---

## 1. Question, scope-change, and the naming rule

VENA's original goal was *explicit vessel reproduction via SWAN conditioning*.
**That scope changed** — SWAN is now an input, not a headline mechanism. This
study is therefore a **comparative diagnostic across all methods**: *do
gadolinium-free synthesisers reproduce the bright, contrast-enhancing non-tumour
structures — of which cerebral vessels are the dominant class — or do their
errors concentrate exactly there?*

**Naming rigor (§2.5⑧ — do not overclaim):** the statistic $\rho_S$ correlates
the residual $|r(x)|$ with **real $T_{1c}$ intensity**, *not* a vessel
segmentation. In the **C-noT** condition (brain minus dilated whole-tumour) the
bright voxels are "exclusively vessels, dural venous sinuses, choroid plexus,
pituitary, pineal, dural enhancement." So the defensible claim is *"errors track
bright contrast-uptake structure, predominantly vascular"* — **not** "vessel
Dice". If a reviewer wants a literal vessel metric, that is optional task T5.4.

## 2. Claim (corrected 2026-07-20 — re-derived + significance-tested; see `CONTRIBUTION.md` §2)

**The "latent vs pixel" question is now answered, and it is not flattering: the
image tier wins $\rho_S$.** State the claim precisely — over-claiming here is the
single easiest way to get the paper rejected.

- **C5.1 (corrected)** VENA is the **least structure-correlated of the latent
  *diffusion/flow* tier** ($\rho_S$: v3b-rw 0.197, v3b 0.205, v3a 0.309 — vs
  C4 0.725 / C5 0.506 / C6 0.459). It is **not** the least of the *latent* tier:
  the latent-GAN **C7-Latent-Pix2Pix ($-0.19$) is lower**, and it is **not** the
  least overall.
- **C5.2** Latent diffusion/flow baselines (C4/C5/C6) are **worse than identity**
  (0.35) on $\rho_S$ — a real, novel negative result. VENA (even fair `v3a`)
  **significantly reverses** it: v3a vs C4/C5/C6 rank-biserial −0.99/−0.73/−0.59,
  all $p\le10^{-15}$; v3a vs identity −0.18 ($p=0.017$, small but significant);
  the oracle mask ~doubles the identity margin (v3b-rw −0.45).
- **C5.3 (the caveat that MUST appear)** **2D pixel methods (pGAN/ResViT $\rho_S$
  0.03–0.09) and the latent-GAN C7 significantly beat VENA** on this axis
  (v3a vs pGAN-t1pre rb +0.66; v3a vs C7 rb +0.88). The VAE-compression cost hits
  vessel fidelity too; VENA mitigates the latent-*diffusion/flow* failure but does
  not overtake the pixel tier.
- **C5.4** **$\rho_S$ discriminates; Conc(5%) does not** (VENA vs C5 Conc p=0.86).
  Report $\rho_S$ as *the* statistic, Conc as a null-behaved companion (retires
  proposal §4.3.3/§4.3.5).

**Framing consequence:** the headline is *"latent diffusion/flow misplaces
contrast; VENA is the latent-tier fix"* — not *"VENA has the best vessel
fidelity."* The literal-vessel metric T5.4 (Frangi conspicuity) would only help if
it flips the pixel-vs-VENA ordering; do not assume it will.

## 3. Definitions (from Phase-2)

| Term | Meaning |
|---|---|
| $\rho_S$ | Spearman($|r(x)|$, $T_{1c}^{\text{real}}(x)$) over region R, per scan. Higher = errors concentrate on bright voxels = worse. |
| **C-noT** | R = brain $\setminus$ dilate($M_{WT}$, k=5) (`max_pool3d` k=5, radius 2). **The vessel-fidelity condition.** |
| C-WB | R = whole brain (incl. tumour). Global contrast-uptake fidelity. |
| Conc(5%) | error mass in the top-5% brightest real voxels, normalised so E=1 under independence. Companion, not discriminator. |
| shuffle-null | intensities permuted within brain; recompute $\rho_S$, Conc. Verified null $\rho \approx 0.000\ \forall$ method — defends against "positive $\rho_S$ is trivially expected". |
| deciles | mean $|r|$ per real-$T_{1c}$ decile → intensity-stratified residual curve. |

## 4. Data — paths

**Source:** `spatial_residual/LATEST/per_scan/spatial_residual.csv` (Holm-corrected
2026-07-20). Columns: `rho_s(+_lo/_hi/_p), conc_{01,05,10}, mi_ksg,
null_*_{mean,std}, delta_*, decile_01..10, condition, pred_mode`.
Aggregates `tables/patient_stats.csv`, `tables/wilcoxon_results.csv`; figures
`fig_rho_s_cnot.png`, `fig_conc05_cnot.png` exist. **No generation needed** for
the core study.

Filter `condition == "C-noT"`, Ring A, sel-NFE, patient-collapse. Stats: Wilcoxon
+ Cliff's δ vs the reference, Holm family = 2 stats × 8 competitors = 16 (§2.3).

**Optional T5.4 (strengthens the vessel claim, addresses the naming concern):**
compute a **literal vessel metric** — Frangi vesselness on synth vs real $T_{1c}$
within C-noT, then vessel-Dice / conspicuity-ratio. This is the direct descendant
of VENA's original $M_v$ pipeline and converts "predominantly vascular" from an
interpretation into a measurement. CPU cost; needs a metric spec (Frangi σ range,
threshold — reuse `routines/preflights/vessel_mask` settings).

## 5. Tables

**Table 5 (main) — C-noT structure fidelity, Ring A.** Rows = 16 methods
(tiered). Columns: `ρ_S (mean [CI])` | `ρ_S vs C0 Cliff's δ` | `Conc(5%)` |
`Δ-to-shuffle ρ_S`. Bold lowest ρ_S per tier; the story is VENA < C0 < {C4,C5,C6}.
Footnote: ρ_S discriminates, Conc does not (give the C5 p=0.86).

**Table 5B — C-WB (supp):** same, whole-brain (includes tumour) — shows the
effect is not just a tumour-exclusion artifact.

## 6. Figures

**Fig 5A — ρ_S per method** (the existing `fig_rho_s_cnot.png`, restyled):
tiered bar/forest, C0 line marked, VENA highlighted, shuffle-null band at ρ≈0.
One glance: latent generators above identity, VENA below.

**Fig 5B — intensity-stratified residual (deciles).** x = real-$T_{1c}$ decile,
y = mean $|r|$; one line per headline method (C0, C5, v3a, v3b-rw). Shows *where*
in the intensity range errors live — the diffusion tier's curve rises in the top
deciles (bright/vascular), VENA's stays flat. This is the mechanistic figure.

**Fig 5C — qualitative residual maps.** For 2 patients: real $T_{1c}$ (with
vessels visible) | residual $|r|$ for C5 | for v3a | for v3b-rw, windowed
identically, tumour region masked out (C-noT). Visually: C5's residual lights up
on vessels; VENA's is diffuse. If T5.4 is done, overlay the Frangi vessel mask.

## 7. Reviewer objections & pre-emptions

| Objection | Pre-emption |
|---|---|
| "$\rho_S$ isn't vessel-specific" | Naming rule (§1): claim "contrast-uptake structure, predominantly vascular"; C-noT isolates it; optional T5.4 gives a literal vessel metric. |
| "Positive $\rho_S$ is expected trivially" | Shuffle-null ρ≈0 verified; report Δ-to-shuffle. |
| "Conc says otherwise" | Conc is null-behaved (≈3.0, doesn't separate); we report it *as* a companion and explain why ρ_S is the discriminator. |
| "Latent vs pixel — unproven" | Resolve at render by placing image-tier ρ_S in Table 5 (§2 open question); state the conclusion the data supports, not a presupposed one. |
| "This contradicts your proposal §4.3" | Yes — Hub §5 flags the proposal revision; state it as a corrected finding (rigor, not weakness). |

## 8. Task checklist

- [ ] **T5.1** Table 5 (C-noT, Ring A) + Table 5B (C-WB); Wilcoxon+δ, Holm-16.
- [ ] **T5.2** Fill the image-tier ρ_S values → settle latent-vs-pixel conclusion.
- [ ] **T5.3** Fig 5A (restyle), 5B decile curve, 5C residual maps.
- [ ] **T5.4 (optional)** Literal Frangi vessel-conspicuity metric within C-noT.
- [ ] **T5.5** Draft the proposal-revision note (§4.3.3/§4.3.5 → ρ_S).
