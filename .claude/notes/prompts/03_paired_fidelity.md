# TASK V1 — `paired_fidelity`: §4.2 voxel-wise fidelity + §4.5 cost + §4.7 ZGD

**Read `01_SHARED_CONTRACTS.md` first, completely.** Then read
`.claude/notes/validation/validation_proposal.md` §4.2, §4.5, §4.7, §6.

| | |
|---|---|
| **Model** | Opus 4.8, effort `max` |
| **Isolation** | **git worktree**, branch `task/validation-paired-fidelity` |
| **Depends on** | T0 `validation-core` (merged into main before you start) |
| **Runs** | in parallel with V2 (`spatial_residual`) and V3 (`downstream_seg`) |
| **Lane (you own)** | `src/vena/validation/metrics_paired.py`, `routines/validation/paired_fidelity/**`, `tests/validation/test_metrics_paired.py`, `tests/validation/test_paired_fidelity_engine.py` |
| **Do not touch** | `src/vena/validation/{io,registry,regions,stats,plotting,audit,artifacts}.py` (T0's, read-only to you), `pyproject.toml` (already registers your console script), the other agents' lanes, `CLAUDE.md`, `.claude/rules/**`, `src/external/**`, `render_comparison_figure` |

**Isolation is not boilerplate.** Run the §2 self-check from the contracts as
your first command and paste the output. Without `PYTHONPATH=<WORKTREE>/src`
you will silently test the main checkout's code.

---

## 1. Why this exists

This routine produces **the primary endpoint of the paper**: MAE on the brain
region of Ring A, `VENA-S1-v3b-rw` vs the 8-competitor family (proposal §6.1).
Everything else in the submission is secondary to this table. It also produces
the cost-quality Pareto that backs the few-step rectified-flow claim.

---

## 2. What to compute — per (method, cohort, nfe, scan)

All in **image space**, on the already-harmonised volumes, `data_range=1.0`,
**no re-normalisation** (contracts §7).

### 2.1 §4.2 — four metrics × three regions

Regions (from `vena.validation.regions.region_masks`):
- `brain` — the HD-BET/CBICA mask
- `wt` — whole tumour
- `bg_undilated` — `brain \ WT`, **undilated** (§4.2's background; note §4.3
  uses the *dilated* variant — do not conflate, contracts §11 trap 8)

Metrics:
- **MAE** — the primary ranking statistic (Reinke 2024 §3.1: MSE is dominated
  by the residual heavy tail; MAE is the robust choice).
- **RMSE** — reported alongside.
- **PSNR-3D** — never reported alone.
- **SSIM-3D** — see §3, this needs care.
- **MS-SSIM-3D** — `monai.metrics.MultiScaleSSIMMetric`, 4 levels, weights
  `[0.0448, 0.2856, 0.3001, 0.3633]` (Wang 2003). **monai 1.5.2 is installed —
  do not add `pytorch-msssim`.**

Reuse `vena.model.fm.metrics.ImageMetrics` for MAE/MSE/PSNR (contracts §8).
**Never** copy the private `_psnr`/`_ssim` helpers from
`src/vena/competitors/*/inference.py`.

### 2.2 §4.7 — z-gradient discontinuity ratio (ZGD)

```
mean|∂z I| = E_{x∈brain} |I(x,y,z+1) − I(x,y,z)|
ZGD        = mean|∂z T̂1c| / mean|∂z T1c|
```
Per volume, over the brain mask. Cheap — compute it in the same pass.

ZGD > 1 ⇒ more inter-slice variation than real (the 2D tier's slice-stacking
artefact). ≈1 ⇒ z-statistics match. <1 ⇒ over-smoothed in z. Expected to be
≈1 for the 3D-native tier and >1 for C1/C2/C3.

### 2.3 §4.5 — inference cost

Read straight from `metadata/inference_seconds` and `metadata/peak_vram_mb` —
**already measured in Phase 1, do not re-time anything.** Aggregate mean ± std
per (method, nfe). This is free: you are opening the files anyway.

---

## 3. The SSIM-region problem — decide it explicitly, do not paper over it

**SSIM is a local windowed statistic. "SSIM restricted to a region" is not
well-defined**, and this is a real methodological choice a reviewer will probe.

`vena.model.fm.metrics.ImageMetrics.ssim` fills out-of-region voxels with the
per-volume in-region mean, then calls MONAI. `model-coding-standards.md` rule 14
says outright that this "is a rough training-time proxy and degenerates on tiny
regions" — and the exhaustive-val job deliberately uses whole-volume SSIM instead.
A WT region is often <1% of the volume. The mean-fill proxy is **not** defensible
for a headline table.

**Required treatment — the principled option:** compute the **full SSIM map**
over the volume once, then average the map inside each region. That is the
natural meaning of "SSIM over region R" and it degrades gracefully.
Investigate `monai.metrics.regression.compute_ssim_and_cs` (and
`SSIMMetric(..., return_full_image=True)` if present in monai 1.5.2) to get the
map rather than the reduced scalar. Same for MS-SSIM where the API allows;
where it does not, say so.

If the map is genuinely unavailable from MONAI's API:
1. **Report the finding** rather than silently falling back.
2. Fall back to: whole-brain SSIM as the headline (both volumes are 0 outside
   the brain, so this is exact and needs no masking), plus **WT-bounding-box**
   SSIM for the tumour region (crop to the WT bbox + a small margin, compute
   SSIM there). Bounding-box cropping is the standard treatment in the synthesis
   literature and is defensible; mean-fill is not.
3. Document the choice in `report.md` **and** `decision.json`, with the reason.

Whatever you choose, **the same treatment applies to every method** — that is
what makes the comparison valid.

---

## 4. Statistics (§6) — within your families

Per contracts §4.1. **Collapse scans → patients first**
(`stats.collapse_to_patient`) — trap #4, LUMIERE 72 → 11.

- **Primary endpoint**: MAE on `brain`, Ring A, at each method's
  `selection_nfe`. Paired Wilcoxon `VENA-S1-v3b-rw` − competitor, two-sided,
  α=0.05, one test per competitor, **Holm-Bonferroni over the family of 8**.
- **Secondary**: each (metric, region) cell is its **own family** of 8 with its
  own Holm correction (§6.3). Do not pool them into one giant correction, and
  do not leave them uncorrected.
- **Effect size**: Cliff's δ per comparison. **CI**: patient-stratified
  bootstrap, 10,000 resamples, seed pinned.
- **MCID = 0.01** on the [0,1] scale: a win below it is "statistically but not
  clinically significant" — label it as such (`stats.MCID`).
- **Ablation family** (n=3: v3b, v3a, S3-LPL-b2c) — separate Holm correction.
- **Ring A and Ring B are separate endpoint families** (§6.4). Never pool.
- Per-cohort breakdowns are **exploratory** — label them so.

Reporting discipline (§6.3): secondary endpoints are *confirmatory* only if the
primary rejects H₀; otherwise they are **exploratory** and must say so in
`report.md`.

---

## 5. Outputs (contracts §9)

`routines/validation/paired_fidelity/` → `<output_root>/paired_fidelity/<UTC>/`

- **`per_scan/paired_fidelity.csv`** — the sample-wise tidy CSV. One row per
  (method, cohort, nfe, scan_id) with `patient_id`, every metric × region, ZGD,
  `inference_seconds`, `peak_vram_mb`. Frozen header, no white cells.
  **This is the input to the paper's statistical pass — it matters most.**
- `per_scan/paired_fidelity_patient.csv` — the patient-collapsed view.
- `tables/` — headline table (method × metric, Ring A, at selection_nfe, with
  Holm-adjusted p, Cliff's δ, bootstrap CI, MCID flag); matched-NFE=5
  sub-table; per-cohort × method tables, Ring A and Ring B separate;
  supplementary 2D-panel table (the C1/C3 t2/flair rows); §4.5 cost table;
  §4.7 ZGD table.
- `figures/`
  - **Primary**: MAE-on-brain per method, Ring A, patient-level distribution
    (box/violin + points), **significance brackets vs VENA, Holm-corrected**,
    C0-Identity marked as the null floor. State the family and the correction
    in the caption.
  - Region grid: metric × region small-multiples.
  - **Cost-quality Pareto**: MAE (or SSIM) vs `inference_seconds`, one point per
    (method, NFE), NFE-swept methods as connected curves. This figure carries
    the few-step-RF claim.
  - ZGD per method — the 2D tier should separate visibly from the 3D tier.
  - **Qualitative** (required): black background; one row per method (ordered by
    the quality metric, **descending**), real T1c row on top; a few axial slices;
    per-slice `vmin/vmax` anchored to the **real** slice; per-method
    PSNR/SSIM in the row ylabel; WT contour overlaid. Use
    `plotting.render_method_comparison_figure` from T0 and
    `select_content_slices`. Best/median/worst patients by primary-endpoint MAE.
- `report.md`, `decision.json` — per contracts §9. `decision.json` records the
  SSIM treatment chosen (§3), the family assignment, and every parameter.

---

## 6. Acceptance criteria

- [ ] Import-isolation self-check pasted.
- [ ] **C0-Identity is beaten by every real method inside the WT.** If not, your
      metric is wrong — this is the designed sanity anchor. Report the numbers.
- [ ] **C0-Identity's brain MAE is a plausible non-zero floor** and
      `VENA-S1-v3b-rw` beats it. A method that cannot beat identity has added
      no enhancement information.
- [ ] LUMIERE collapses 72 scans → 11 patients in the patient-level CSV;
      assert it in the smoke.
- [ ] `data_range=1.0` everywhere; grep your diff for any renormalisation and
      confirm there is none.
- [ ] SSIM treatment decided, implemented, justified in `report.md` **and**
      `decision.json`.
- [ ] Holm families are exactly the 8 / 3 of contracts §4.1 — assert the family
      size in code so a future method addition cannot silently change it.
- [ ] Ring A and Ring B never pooled.
- [ ] Real-data smoke on the contracts §13 subset, artifact folder inspected,
      tree + 3 real numbers pasted. Not "exit 0".
- [ ] Ruff clean; unit tests marked `validation`, synthetic fixtures only
      (reuse T0's `conftest.py` fixture).
- [ ] Report the wall-clock of the smoke and **extrapolate to the full sweep**
      (45 method-nfe × 467 scans = 21,015 volume-reads; reference measurement:
      0.076 s/vol read). The orchestrator needs this to size the SLURM job.

## 7. Notes

- The engine must be **shardable** for the Picasso sweep: a YAML filter over
  methods / cohorts / nfe, so the orchestrator can fan out across
  `cpu_partition` and merge the `per_scan` CSVs afterwards. Design for that
  from the start; retrofitting it is painful.
- GPU is optional and opportunistic (SSIM/MS-SSIM are conv-bound). Make the
  device a YAML param defaulting to CPU so the routine runs on `cpu_partition`.
  If you use CUDA, `torch.no_grad()` + explicit `.detach()` + free between
  volumes (`coding-standards.md` rule 13).
- Empty region (e.g. a scan with no WT voxels) → `NaN`, counted, reported in
  `report.md`. Never silently dropped. `ImageMetrics` already returns NaN on an
  empty mask — propagate it, and use `_finite_mean`-style aggregation.
