# VENA Validation — Phase-1 Fairness Audit and Refactor Log

*Mario Pascual González, 2026-07-06. Companion to
[`validation_proposal.md`](validation_proposal.md). Audits whether the
prediction-generation half of the benchmark ("Phase 1") is built
correctly and compares every method fairly, records the real test-set
sizes, and logs the code changes made to close two of the gaps.*

The benchmark is intended to run in two phases:

1. **Phase 1 — inference.** Every method (C0–C7 + A1-VENA-S1 + VENA)
   predicts $\widehat{T_{1c}}$ for every test scan of every cohort; the
   result is written to a validated per-`(method, cohort, NFE)`
   predictions H5 at the end of a Picasso job.
2. **Phase 2 — analysis.** Separate CPU jobs read those frozen H5s and
   compute the per-region / whole-volume metrics, spatial-residual
   statistics, and the statistical-analysis tables (§4–§6 of the
   proposal).

Phase 1 is built (`routines/fm/inference/`, `src/vena/inference/`).
Phase 2 does not exist yet. This document is only about Phase 1.

---

## 1. Verdict

Phase 1 is architecturally sound and most fairness controls are correct.
Two substantive fairness problems were found; both now have an agreed
resolution (one by planned ablation, one by a code change applied here).
Two smaller issues were fixed in code (NFE matching, longitudinal
provenance). The remaining items are Phase-2 obligations.

---

## 2. Phase-1 controls verified correct

Read from `routines/fm/inference/engine.py`, `src/vena/inference/base.py`,
`src/vena/inference/harmonisation.py`, and all nine adapters.

- **Unified driver.** `InferenceEngine` is a single
  `method × cohort × patient × NFE` sweep; one Picasso job produces
  `predictions/<method>/<cohort>/nfe_NNN.h5` and calls
  `assert_predictions_valid` on each.
- **Shared reference, byte-identical across methods.**
  `_build_reference_cache` computes the real $T_{1c}$ + T1pre/T2/FLAIR +
  brain/WT masks **once per `(cohort, patient)`**; every method reuses
  the same reference block.
- **Same test patients, same masks** across all methods (resolved once
  per cohort).
- **Intensity harmonisation is uniform.** Every adapter calls the same
  `apply_harmonisation` =
  `percentile_normalise(lower=0.0, upper=99.5, foreground_only=True)`
  with the cohort's HD-BET `masks/brain`, exterior zeroed; the real
  $T_{1c}$ reference is harmonised by the identical recipe; all outputs
  are mapped back to the reference's native grid. *(The `lower=0.0` vs
  the proposal's `0.5` is deliberate and documented in
  `harmonisation.py` — it matches the encoder contract, and being
  applied identically to every method **and** the reference it
  introduces no between-method bias.)*
- **Timing / VRAM per §5.2.** Warm-up passes, CUDA-synced wall-clock
  over the whole `predict()` body, `reset_peak_memory_stats` before each
  call.
- **VAE confound controlled.** The entire latent tier (C4, C5, C6, C7,
  A1, VENA) decodes through the *same* frozen `autoencoder_v2.pt` —
  including C5-T1C-RFlow, which the proposal said would use its own VAE.
  This deviation is *fairer* than the written proposal.
- **Split integrity (no train/test leakage).** `make_nested_cv_splits`
  receives **unique patient IDs** (`patient_ids = [pid for pid, _ in
  patient_groups]`), KFold runs at patient level, and
  `patients/{keys,offsets}` CSR-group each patient's scans contiguously,
  so no longitudinal patient's timepoints straddle train/val/test.
  Verified on real data: LUMIERE's 72 test scans belong to 11 patients,
  all held out together.

---

## 3. Fairness concerns (ranked) and resolutions

| # | Severity | Concern | Resolution |
|---|---|---|---|
| ① | Critical | **Input-modality mismatch.** VENA/A1 and ResViT condition on 3 modalities `{t1pre, t2, flair}`; C4–C7 on 2 `{t1pre, flair}` (no T2, hard-wired: adapters read `input_latents[0/1]`, `cond_latents=2`); pGAN/SynDiff on 1 (single-source panels). This confounds the proposal's "only the generative formulation differs" claim for C4–C7 vs VENA. | **Addressed by planned ablation.** A VENA input-ablation is being trained: `{T1pre+FLAIR}`, `{T1pre+T2}`, `{T2+FLAIR}`. The `{T1pre+FLAIR}` variant exactly matches the C4–C7 conditioning, so the C4–C7-vs-VENA comparison is made at matched inputs. (Retraining C4–C7 with a T2 latent remains an alternative but the ablation is the cleaner control.) |
| ② | Critical | **VENA-S2 receives the ground-truth WT tumour mask** (`mask:wt:identity`) as a conditioning channel; no competitor does. Extra oracle localisation. It also means the A1(S1)-vs-VENA(S2) comparison flips *both* the loss (add contrastive) *and* the mask signal (S1 uses `mask:wt:zero_out`), so it is not the zero-delta ablation the proposal claims. | **Addressed by a no-mask VENA run** (in progress): a VENA variant trained without the tumour mask gives a comparator with no oracle localisation, closer to the competitors and a cleaner isolation of the loss term. |
| ③ | Medium | **NFE not matched, and diverged from the pre-registered plan** (config had C4/C5 selection=100, C6=1000 vs proposal §5.1's 5/20; VENA=5, no common NFE). | **Fixed in code** (see §5). C4-DiT and C5-RFlow now share VENA's `{1,2,5,10,20}` grid with `selection_nfe=5`; C6-DDPM gets matched `{5,10,20}` points for the Pareto and keeps `1000` as its faithful best. |
| ④ | Medium | **Longitudinal pseudoreplication.** The engine expands each test patient to all scans (LUMIERE: 11 patients → 72 scans), and `/metadata/patient_id` stored the *scan* id, so Phase-2 could not collapse to patient level → non-independent observations in the paired Wilcoxon and an anti-conservative bootstrap. | **Fixed in code** (see §5). The predictions H5 now stores the true patient key and a distinct `scan_id`; Phase-2 must group by `patient_id`. |
| ⑤ | Low | VENA `_sample` uses unseeded `torch.randn_like` → non-reproducible, cross-NFE draws differ. | Open. Seed per patient before launch. |
| ⑥ | Low | pGAN/SynDiff run as 3 single-source rows (architecturally one-to-one) — under-conditioned vs 3-input methods. | Report best panel per method; disclose the one-to-one nature in the discussion. |
| ⑦ | Low | No `artifacts/validation/<UTC>/ring_partitions.json` pre-registration + hash yet (P3 integrity). | Open. Freeze the resolved test IDs (§4) before unblinding. |

---

## 4. Real test-set sizes

Read from the mounted H5 caches via `resolve_test_scan_patient_pairs`
(scan-level, CSR-expanded). Split policy: patient-level nested CV,
`seed=42`, 5 folds, per-cohort held-out `splits/test` (Ring A);
`role=test_only` cohorts contribute every patient (Ring B).

| Ring | Cohort | test patients | test scans |
|---|---|---:|---:|
| **A** | UCSF-PDGM | 50 | 50 |
| A | BraTS-GLI | 114 | 127 |
| A | UPENN-GBM | 62 | 62 |
| A | IvyGAP | 5 | 5 |
| A | LUMIERE | 11 | 72 |
| A | REMBRANDT | 5 | 5 |
| **A total** | | **247** | **321** |
| **B** | BraTS-Africa-Glioma | 95 | 95 |
| B | BraTS-Africa-Other | 51 | 51 |
| B | BraTS-PED | 260 | 260 |
| **B total** | | **406** | **406** |

Notes:

- The real Ring-A pool (247 patients / 321 scans) is larger than the
  proposal §3 plan (173 / 253) — the cohorts grew since the draft. The
  §3 per-cohort test numbers are stale and must be replaced by the table
  above.
- **Patient-level is load-bearing.** LUMIERE is 11/247 = 4.5% of Ring-A
  *patients* but 72/321 = 22% of *scans*. The primary endpoint must be
  one MAE per patient (proposal §6.4) so scan count cannot tilt the
  ranking. The provenance fix (§5) is what makes this enforceable.
- Ring C (HRUM/Málaga) is not yet on disk — no reader, no registry
  entry.
- **Action:** run the same resolver on Picasso and freeze the output to
  `artifacts/validation/<UTC>/ring_partitions.json` (closes ⑦).

Probe used: `scratchpad/testset_sizes.py` (read-only; point `image_h5`
at `corpus_picasso.json` on the cluster).

---

## 5. Changes applied in this pass

### 5.1 NFE matching (concern ③)

`routines/fm/inference/configs/models/benchmark_full.yaml`:

- **C4-3D-DiT** and **C5-T1C-RFlow**: `nfe_list: [1, 2, 5, 10, 20, 100]`,
  `selection_nfe: 5` — shares VENA/A1's `{1,2,5,10,20}` grid so the
  primary head-to-head is at a common **NFE = 5**; `100` kept as the
  native cost-quality anchor.
- **C6-3D-LDDPM**: `nfe_list: [5, 10, 20, 1000]`, `selection_nfe: 1000` —
  the `{5,10,20}` points give a matched-NFE Pareto row (the RFlow-vs-DDPM
  few-step advantage, Eidex 2025 §5); `1000` stays the DDPM-faithful
  best and headline point.
- Unchanged: C0/C1/C2/C7 at NFE = 1 (single forward; match VENA @ 1),
  C3-SynDiff at its native 4-step bridge, A1/VENA at `{1,2,5,10,20}`
  select 5.

Result: at NFE = 5 the whole few-step-capable latent tier (VENA, A1,
C4, C5) is directly matched; the GAN tier matches VENA @ 1. The
per-family selection and the cost-quality Pareto must be pre-registered
before unblinding, and the proposal §5.1 NFE table updated to match.

### 5.2 Longitudinal provenance (concern ④) — predictions-H5 schema 1.0 → 1.1

Problem: `/metadata/patient_id` held the *scan* id, so longitudinal
patients could not be collapsed for a valid paired test.

- `src/vena/inference/image_dataset.py`: added
  `resolve_test_scan_patient_pairs()` returning `(scan_id, patient_id)`
  pairs via the CSR expansion; `resolve_test_patient_ids()` now delegates
  to it (behaviour identical).
- `src/vena/inference/h5_writer.py`: `PerPatientRecord` gains a
  `scan_id` field; writer emits `/metadata/scan_id`; validator now
  enforces uniqueness on **`scan_id`** (one row per scan) and explicitly
  permits repeated `patient_id` (longitudinal). `SCHEMA_VERSION` bumped
  to `"1.1"`.
- `routines/fm/inference/engine.py`: builds a `(cohort, scan_id) →
  patient_id` map and writes each row's true `patient_id` + `scan_id`.

**Phase-2 requirement:** metric aggregation must group by `patient_id`
(mean per patient) before the paired Wilcoxon, and the bootstrap must be
patient-stratified (proposal §6.2/§6.4).

Verification: `tests/inference/test_h5_writer.py` +
`tests/routines/fm/inference/test_engine.py` — 11 passed. Real-data
check: LUMIERE resolves 72 scans → 11 patients; UCSF 50 → 50.

---

## 6. Remaining action items

- **Phase-2 analysis layer** does not exist (`benchmarks.metrics`,
  `benchmarks.spatial_residual`, statistical plan). Must aggregate
  scan → patient (§5.2) and read the per-family selection NFE (§5.1).
- **Pre-registration:** freeze the §4 test IDs to
  `artifacts/validation/<UTC>/ring_partitions.json` + hash into each
  `decision.json` (⑦).
- **Seed VENA sampling** per patient (⑤).
- **Proposal reconciliation:** update §3 (real sizes), §4.1 (`lower=0.0`),
  §5.1 (matched NFE), and note C5 uses the shared MAISI VAE.
- Optional: give C4–C7 the WT-mask channel and/or a strict-loss-only A1
  (`mask:wt:identity`, no contrastive) if a fully architecture-matched
  isolation is wanted beyond the planned input ablations.
