# TASK V2 — `spatial_residual`: §4.3 bright-region error concentration

**Read `01_SHARED_CONTRACTS.md` first, completely.** Then read
`.claude/notes/validation/validation_proposal.md` **§4.3 in full** (it is the
densest section in the protocol — §4.3.1 through §4.3.5) plus §6 and Appendix C.

| | |
|---|---|
| **Model** | Opus 4.8, effort `max` |
| **Isolation** | **git worktree**, branch `task/validation-spatial-residual` |
| **Depends on** | T0 `validation-core` (merged into main before you start) |
| **Runs** | in parallel with V1 (`paired_fidelity`) and V3 (`downstream_seg`) |
| **Lane (you own)** | `src/vena/validation/spatial_residual.py`, `routines/validation/spatial_residual/**`, `tests/validation/test_spatial_residual.py`, `tests/validation/test_spatial_residual_engine.py` |
| **Do not touch** | T0's modules (read-only to you), `pyproject.toml`, the other agents' lanes, `CLAUDE.md`, `.claude/rules/**`, `src/external/**` |

Run the contracts §2 import-isolation self-check first and paste the output.

---

## 1. Why this exists — read this before writing any code

**This routine carries the paper's headline scientific claim.**

VENA's differentiator is vessel-aware fidelity. The 2026-06-17 revision
**deleted** the entire vessel-segmentation apparatus (Frangi/Jerman/OOF/
VesselFM) because no cohort ships ground-truth vessel labels and Hessian
operators are out-of-domain on structural T1c (Appendix C). The vessel claim is
now tested **label-free**, from the intensity residual alone — by this routine.

So: "does the model under-synthesise the bright contrast-uptake regions outside
the tumour?" is answered here or nowhere.

Because there is no segmenter to blame, **the statistics are the entire
apparatus**. A sloppy null makes the headline claim indefensible. Read §4.3.5
("Defence against the 'trivial ρ' reviewer attack") and treat it as the
specification, not as commentary.

---

## 2. Setup (§4.3.1)

Per scan, residual `r(x) = T1c(x) − T̂1c(x)` on the already-harmonised volumes.
**Recompute it on the fly — it is not stored, and you must not store it**
(134 G free locally; contracts §11 trap 10).

Two conditions, different meanings — report both:

- **C-WB** (whole brain): `R = brain`. Global contrast-uptake fidelity;
  includes the enhancing tumour.
- **C-noT** (background): `R = brain \ dilate(M_WT, k=5)`. Bright voxels here
  are *exclusively* vessels, dural venous sinuses, choroid plexus, pituitary,
  pineal, dural enhancement. **C-noT is the vessel-fidelity claim of the paper.**

Use `vena.validation.regions.region_masks(..., dilate_k=5)` → its `bg` key
(the **dilated** one; `bg_undilated` is §4.2's, do not confuse them —
contracts §11 trap 8). `k=5` ⇒ `max_pool3d(kernel_size=5, padding=2)` ⇒ radius 2.
YAML parameter, round-tripped into `decision.json`.

---

## 3. The statistics (§4.3.2)

### S1 — Spearman rank correlation `ρ_S(|r|, T1c)` over R, per scan

One scalar per (scan, method, condition). Large positive ρ ⇒ errors are
systematically larger in bright voxels of the real volume — the clinical
failure mode.

**Reuse `spearman_with_bootstrap_ci`** from
`vena.preflight.priors_validation.statistics.correlation` (re-exported by
`vena.validation.stats`). Do not write a second Spearman.

Justification for ρ over Pearson is already settled (Yin & Carroll 1990;
Bishara & Hittner 2012: under heavy-tailed data Pearson inflates Type-I to
~20% at α=0.05). Do not substitute Pearson "for speed".

Cost note: ~1e6 brain voxels/scan; `scipy.stats.spearmanr` is O(n log n),
~0.2 s. Fine at full resolution — **do not subsample S1.**

### S2 — Top-q% bright-voxel error mass concentration

For `q ∈ {1%, 5%, 10%}`:
```
B_q   = { x ∈ R : T1c(x) > Q^{1−q}_real(R) }     # intra-volume quantile over R
Conc(q) = Σ_{x∈B_q} |r(x)|  /  ( q · Σ_{x∈R} |r(x)| )
```
Under spatial independence of `|r|` and `T1c` over R, `E[Conc(q)] = 1`.
`>1` ⇒ error concentrates in bright voxels above chance. `<1` ⇒ avoids them.

`Conc(5%)` is the headline number: *"the top 5% of background bright voxels
carry X× their fair share of the model's error"* — dimensionless,
scale-invariant, falsifiable against 1.

**Watch the denominator**: `q` is the *nominal* fraction, but ties and the
discrete quantile mean `|B_q| / |R| ≠ q` exactly. Use the **realised** fraction
`|B_q|/|R|` in the denominator, not the nominal `q`, or `Conc` is biased by the
tie structure. Volumes are float32 with a large dynamic range so ties are rare —
but C0-Identity and heavily-quantised outputs will have them. Document the
choice; assert `E[Conc]=1` on a random-independent synthetic as your test.

### S3 — Intensity-stratified residual plot (Bland-Altman adaptation)

Partition R into **intensity deciles of T1c**; per decile, mean `|r|` + 95% CI
across Ring-A **patients** (not scans) per method. The discretised
Bland-Altman plot (Bland & Altman 1986/1999); the slope is the
proportional-bias diagnostic. One figure per ring, all methods overlaid.

Store per-scan decile means in the `per_scan` CSV (10 columns) so the figure is
reproducible from the CSV without re-reading 289 GB.

### Exploratory — KSG mutual information

`MI(|r|, T1c)` over R, Kraskov-Stögbauer-Grassberger, **k=5**.
**Use `sklearn.feature_selection.mutual_info_regression(..., n_neighbors=5)` —
that IS the KSG estimator.** Do not implement k-NN MI yourself.

MI catches U-shaped/plateau failure modes ρ misses. **Supplementary only; NOT
in any multiplicity family** (§4.3.3, §6.3).

**Cost trap:** KSG is k-NN in 2D — O(n log n) with a large constant, and
infeasible at n≈1e6 × 21,015 volumes. **Subsample voxels** (e.g. 20k–50k,
uniformly at random within R, seeded per scan). MI is a per-voxel-distribution
statistic, so uniform subsampling is unbiased for it. **Log the subsample size
in `report.md` and `decision.json`** (contracts §11 trap 11). Report the
subsample-induced variance from a small repeat experiment.

---

## 4. The null (§4.3.5) — the part that decides whether the claim survives

A reviewer will argue that positive `ρ_S` is *trivially* expected: the bright
tail of T1c is partly anatomy (myelin, fat partial-volume, calcium), so any
imperfect model correlates positively. Three defences, all mandatory:

### 4.1 C-noT itself
Restricting R to `brain \ dilate(WT, 5)` removes the dominant enhancement
source and most GM/WM partial-volume myelin contrast. Already in §2.

### 4.2 Per-scan intensity-shuffle null — the load-bearing one

Shuffle `T1c(x)` values **uniformly at random within the brain mask**,
preserving the marginal intensity distribution while destroying spatial
correspondence with `|r|`. Recompute S1 and S2 on the shuffled volume.
Report **delta-to-shuffle = observed − shuffle-mean as the primary effect
size.** Delta ≈ 0 ⇒ the correlation *was* trivial. Delta > 0 significantly ⇒
real signal. (Standard spatial-null construction: Alexander-Bloch 2018.)

**Subtlety you must get right:** shuffle within `brain`, but the statistic is
computed over `R`. For C-noT, shuffling over `brain` and then restricting to
`R` moves tumour-bright voxels into the background region — which is arguably
the *point* (it preserves the brain-wide marginal). But it is also arguable that
the null should shuffle **within R** to preserve R's own marginal. These give
different nulls. **The proposal says "within the brain mask".** Implement that
as the default, **also implement shuffle-within-R**, report both, and say which
is primary in `report.md`. This is exactly the kind of choice a reviewer probes,
and having both costs almost nothing.

**Compute:** the proposal's §11 wrapper says `--shuffle-null 1000`.
1000 × 21,015 volumes is infeasible and **unnecessary**. Reason it out and put
the reasoning in `report.md`:
- For S1, shuffling makes `E[ρ_S] = 0` **analytically** (correspondence is
  destroyed); the shuffle only estimates the null's *variance*.
- For S2, `E[Conc(q)] = 1` **analytically** — the proposal says so itself
  (§4.3.2). Again the shuffle estimates variance.
- With ~1e6 voxels the null is extremely tight around its analytic mean
  (SE ~ 1/√n). **~100 shuffles is ample**; make it a YAML parameter defaulting
  to 100, log it, and **empirically justify it**: run one scan at
  {10, 50, 100, 500} shuffles, show the null mean/SD has converged, put that
  convergence check in `report.md`. Then the reduction is a measured fact, not
  a shortcut.

### 4.3 C0-Identity as the upper bound
`T̂1c ≡ T1pre`, so `|r| = |T1c − T1pre|` — **maximally concentrated in bright
contrast-uptake regions by construction**. C0 is the ceiling on bright-region
error concentration. **`VENA-S1-v3b-rw` must beat C0 on S1 and S2.** If it
cannot, the model added no enhancement information beyond identity — a
stop-the-line finding. Report it plainly either way.

---

## 5. Statistics (§4.3.4)

Collapse scans → patients **first** (`stats.collapse_to_patient`; LUMIERE
72 → 11; contracts §11 trap 4).

- Paired Wilcoxon on per-patient `Conc(5%)` under **C-noT**, VENA vs each
  competitor. Two-sided, α=0.05.
- Paired Wilcoxon on per-patient `ρ_S` under **C-noT**, VENA vs each competitor.
- **Multiplicity**: Holm-Bonferroni over **2 stats × 8 competitors = 16 tests**
  — one family (§4.3.4). This is a *secondary* family; it does **not** inflate
  §4.2's primary family, but it carries its own correction.
- Bootstrap CI: 10,000 patient-stratified resamples (§4.3.3 Table 3).
- Ablation family (v3b, v3a, S3-LPL-b2c) — separate correction.
- Ring A / Ring B separate.
- Both S1 and S2 are **equal headline tests** — report side by side (§4.3.3).

---

## 6. Outputs (contracts §9)

`routines/validation/spatial_residual/` → `<output_root>/spatial_residual/<UTC>/`

- **`per_scan/spatial_residual.csv`** — one row per
  (method, cohort, nfe, scan_id, condition) with `patient_id`,
  `rho_s`, `conc_01`, `conc_05`, `conc_10`, `mi_ksg`, the shuffle-null mean/SD
  for each, the deltas-to-shuffle, and the 10 decile means. Frozen header.
- `tables/`
  - **Table 3** (main text): method rows × region columns (C-WB, C-noT) ×
    statistic blocks; each cell `Conc(5%) ± CI95` and `ρ_S ± CI95`; bootstrap
    10,000 patient-stratified; Holm-adjusted p vs VENA.
  - **Table S2** (supplementary): `Conc(1%)`, `Conc(10%)`, KSG MI.
  - Per-cohort × method, Ring A / Ring B separate.
- `figures/`
  - **Figure 3**: `ρ_S` heat-map on a **cohort × method** grid, colour = ρ_S
    under C-noT, one cell per Ring-A cohort.
  - **Figure 4**: the S3 intensity-stratified residual plot — mean `|r|` per
    T1c decile, all methods overlaid, one panel per ring, 95% CI bands.
  - `Conc(5%)` under C-noT per method: patient-level distribution,
    **significance brackets vs VENA (Holm, family of 16)**, a reference line at
    `Conc = 1` (the independence null), C0-Identity marked as the ceiling.
  - Delta-to-shuffle per method — the figure that actually defends the claim.
  - **Qualitative** (required): black background; the **residual heat-map**
    `|r|` for a given patient, one row per method, overlaid with the WT contour
    and the brain boundary, alongside the real T1c. Diverging colormap for
    signed `r`; anchor the colour scale **identically across methods** so panels
    are comparable (this is the analogue of the house per-slice `vmin/vmax`
    rule — here the comparison is across methods, so the scale must be shared).
    This is the §9 failure-mode figure: it must make
    "vessel/sinus/choroid-plexus under-enhancement" visible.
- `report.md`, `decision.json` — including the shuffle-count convergence
  evidence, the MI subsample size, and the shuffle-domain choice (§4.2 above).

---

## 7. Acceptance criteria

- [ ] Import-isolation self-check pasted.
- [ ] **`E[Conc(q)] = 1` on a synthetic where `|r|` ⟂ `T1c`** — the core
      correctness test. Tolerance from the analytic SE, not hand-tuned.
- [ ] **`E[ρ_S] = 0`** on the same synthetic.
- [ ] `Conc(q)` on a synthetic where error is *deliberately* concentrated in the
      bright tail returns a known value you compute by hand.
- [ ] **C0-Identity has the highest `Conc(5%)` and `ρ_S` under C-noT of any
      method.** This is the designed ceiling — if some real method exceeds C0,
      either that method is worse than identity at vessels (a genuine, major
      finding) or your statistic is wrong. **Investigate before reporting.**
- [ ] Shuffle-count convergence check run and included (§4.2).
- [ ] MI subsample size logged; subsample-induced variance quantified.
- [ ] LUMIERE collapses 72 → 11 before any test.
- [ ] Holm family = exactly 16 for the main family; assert it in code.
- [ ] Real-data smoke on the contracts §13 subset; artifact folder inspected;
      tree + 3 real numbers pasted.
- [ ] Ruff clean; tests marked `validation`; synthetic fixtures (reuse T0's
      `conftest.py`).
- [ ] Smoke wall-clock reported and extrapolated to the full sweep — **this
      routine is the compute-heaviest of the three** (Spearman + 100 shuffles ×
      2 conditions × 21,015 volumes). The orchestrator needs the number to size
      the SLURM job. If it extrapolates past ~12 h on 335 CPU nodes, say so and
      propose where to cut (fewer NFE points? shuffles only at selection_nfe?).

## 8. Notes

- Engine must be **shardable** by (method, cohort, nfe) for the Picasso fan-out;
  `per_scan` CSVs merge afterwards. Design for it from the start.
- The shuffle null is the compute hot spot. Consider: shuffle-null only at each
  method's `selection_nfe` (the NFE sweep's purpose is the cost-quality Pareto,
  not the vessel claim) — that cuts 45 (method,nfe) pairs to 16. **Propose it,
  justify it, log it** if you take it; do not do it silently.
- `torch.no_grad()`, `.detach()`, free between volumes. Device a YAML param
  defaulting to CPU (this must run on `cpu_partition`).
- Empty region → `NaN`, counted, reported. Never silently dropped.
