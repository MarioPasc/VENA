---
batch_id: 2
theme: lambda_calibration
launch_date: 2026-07-02
result_date: 2026-07-09
parent_baseline: s1_v3b_rw / 320b5ddd
prior_batch: 1
n_arms: 4
status: analysed
picasso_job_ids: [1428339, 1428340, 1428341, 1428342]
picasso_run_ids:
  - 2026-07-02_11-37-58_s3_b2_a_lambda030_standard_fft_ba03fff5
  - 2026-07-02_11-37-55_s3_b2_b_lambda010_standard_fft_609edee9
  - 2026-07-02_11-37-55_s3_b2_c_lambda012_region_fft_bea6d0ff
  - 2026-07-02_11-37-55_s3_b2_d_lambda030_preflight_a_fft_aef00e21
git_sha: 42b6494
git_sha_executed: a1dd749
schema_version: "1.0"
---

# Batch 2 — LPL coupling-weight calibration and feature-depth ablation

## 1. Motivation

Batch 1 (`2026-06-28_batch_1_default_recipe.md`) tested the default LPL recipe (λ_max=1.0, warmup=30, A=[2,5], w_l={2:1,5:2}, t_min=0.4, α∈{(1,1),(2,3)}) on top of the three S1 v3 baselines. The tournament confirmed that LPL cannot substitute for mask conditioning (S3-A on v3a stayed at 4.7 dB UCSF PSNR_ET) but returned a null / slightly-negative result on the target metric: PSNR_ET on the training-distribution cohorts (UCSF-PDGM, BraTS-GLI) moved by −0.06 to −1.16 dB across S3-B/C/D. LUMIERE OOD improved +0.3 to +1.5 dB but that gain is dominated by a single longitudinal patient (Patient-025). Two mechanistic factors were identified as the probable root cause of the null:

1. **λ_max=1.0 makes LPL 80–91 % of total loss on v3b_rw** (CFM_raw ≈ 0.58 vs LPL_raw ∈ [2.28, 5.79]). LPL was the driver, not the regulariser. The batch-1 design-doc §5.3 listed λ as a sweep axis but never executed the sweep; λ=1.0 was inherited as a Berrada-2025 port.
2. **The production feature-depth A=[2,5] deviates from the preflight recommendation A=[2,3]** with w_l={2:0.663, 3:1.337}. Block 5 (deep decoder) may leak image-space semantics that CFM already learns; block 3 (mid-decoder upsample boundary) may carry more informative structural signal despite its higher magnitude.

Batch 2 tests these two axes on the strongest S1 baseline (v3b_rw, `320b5ddd`) via a 4-arm grid.

## 2. Hypotheses

1. **H1** (LPL as regulariser, not driver): dropping λ_max from 1.0 → 0.30 (LPL share 80 % → 58 %) preserves the batch-1 LUMIERE gain (+0.5 dB) *and* recovers the training-distribution regression (UCSF-PDGM ≥ 0.0 dB, BraTS-GLI ≥ 0.0 dB) vs the S1 warm-start.
2. **H2** (λ dose-response): comparing B2-A (λ_max=0.30, share ≈ 58 %) vs B2-B (λ_max=0.10, share ≈ 31 %) reveals a monotone effect. If B2-B ≥ B2-A on PSNR_ET across cohorts, LPL is effective only as a light anchor; if B2-A > B2-B, the middle regime is best.
3. **H3** (region-α at matched strength): with LPL contribution matched to B2-A (~58 %), region-α = (2, 3) beats standard α = (1, 1) on PSNR_ET WT-region metrics without degrading whole-brain PSNR — the batch-1 comparison was confounded because region-α at λ_max=1.0 pushed LPL to 91 % of total.
4. **H4** (feature depth): using preflight-recommended A=[2, 3] with w_l={2:0.663, 3:1.337} — despite block-3's 18.36 magnitude spike at the upsample boundary — improves the LPL signal's independence from CFM. If B2-D ≥ B2-A on PSNR_ET, the block-5 supervision is redundant with CFM.

## 3. Experiment design

All four arms share: warm-start = v3b_rw `320b5ddd` ema_best.ckpt, warmup=30 (isolate λ), t_min=0.4 (preflight), α=(varies), seed=1337, max_epochs=250, patience=100, total_steps=52000 (~250 ep with 208 steps/ep), eff. batch=8, LR=5e-5 cosine over `total_steps`, EMA decay 0.999, `use_timestep_transform=true`, `base_img_size_numel=129024`. Exhaustive_val cadence every 25 epochs on NFE ∈ {1, 5, 20} at `n_patients=90` total (~10/cohort with 9 cohorts) — **9× the batch-1 sample size** per user request 2026-07-02, so ΔPSNR_ET is measured on a statistically meaningful sample rather than n=1-2 outliers. All arms warm-start from Picasso path `/mnt/home/users/tic_163_uma/mpascual/execs/vena/experiments/2026-06-22_15-20-57_s1_v3b_rw_concat_plus_cn3ch_fft_320b5ddd/checkpoints/ema_best.ckpt`.

| Arm | λ_max | α (wt, notwt) | A | w_l | LPL_raw (est) | LPL share | Purpose |
|-----|------:|--------------|:---:|-----|--------------:|----------:|---------|
| **B2-A** | 0.30 | (1, 1) | [2, 5] | {2:1.0, 5:2.0} | ~2.28 | ~58 % | λ dose (mid); direct S3-C comparison |
| **B2-B** | 0.10 | (1, 1) | [2, 5] | {2:1.0, 5:2.0} | ~2.28 | ~31 % | λ dose (light); regulariser regime probe |
| **B2-C** | 0.12 | (2, 3) | [2, 5] | {2:1.0, 5:2.0} | ~5.65 | ~58 % | region-α at strength matched to B2-A |
| **B2-D** | 0.30 | (1, 1) | **[2, 3]** | **{2:0.663, 3:1.337}** | TBM | ~58 % (target) | preflight-aligned feature depth |

`LPL_raw` in the table is the standard-recipe raw magnitude measured at the v3b_rw warm-start in batch 1 (ep 0). B2-D's magnitude is estimated to fall within 30 % of B2-A's after the preflight w_l normalisation, but this is unmeasured — a smoke will confirm before submission.

LPL share formula (at steady state, λ = λ_max):

    share = λ_max · LPL_raw / (CFM_raw + λ_max · LPL_raw)          with CFM_raw = 0.58 (v3b_rw)

Solved for equal share at B2-A and B2-C:

    0.30 · 2.28 / (0.58 + 0.30 · 2.28)  =  λ · 5.65 / (0.58 + λ · 5.65)
    → λ = 0.12

Configs and launchers:

| Arm | YAML | SLURM launcher |
|-----|------|----------------|
| B2-A | `routines/fm/train/configs/runs/picasso_s3lpl_b2_a_lambda030_standard_fft.yaml` | `launcher_picasso_s3lpl_b2_a_lambda030_standard.sh` |
| B2-B | `routines/fm/train/configs/runs/picasso_s3lpl_b2_b_lambda010_standard_fft.yaml` | `launcher_picasso_s3lpl_b2_b_lambda010_standard.sh` |
| B2-C | `routines/fm/train/configs/runs/picasso_s3lpl_b2_c_lambda012_region_fft.yaml` | `launcher_picasso_s3lpl_b2_c_lambda012_region.sh` |
| B2-D | `routines/fm/train/configs/runs/picasso_s3lpl_b2_d_lambda030_preflight_A_fft.yaml` | `launcher_picasso_s3lpl_b2_d_lambda030_preflight_A.sh` |

## 4. Compute budget

At the batch-1 measured throughput (~35 min/epoch on 2 A100 40GB) and target 250 max_epochs:
- Per arm: 250 × 35 min ≈ 146 h = ~6.1 A100·days.
- 4 arms × 6.1 = **~24.4 A100·days total** (each arm uses 2 A100s co-resident: cuda:0 training + cuda:1 async exhaustive_val).
- Requested SLURM walltime: `--time=7-00:00:00` per arm.

Compared to batch 1's 158 A100-hours (~6.6 A100-days), batch 2 is ~4× the compute. This is justified because batch 1 stopped at 38–93 epochs (SLURM early cut), not at convergence — batch 2 needs to actually reach convergence to distinguish signal from noise.

**Batch-1 termination — resolved 2026-07-02**: `sacct` on the four batch-1 jobs shows exit 0 with Timelimit 7d; the `.err` logs carry an explicit EarlyStopping trigger line. The termination was NOT SLURM wall-clock — it was `EarlyStopping(monitor="train/total_epoch", mode="min", patience=30)` firing on the projected `cfm + lambda_max·lpl`. With λ_max=1.0 the LPL raw magnitude dominates the monitor (LPL_raw ≈ 4× CFM_raw on v3b_rw), and the LPL noise floor keeps the monitor pinned to an early best. Batch 2 mitigates this with (i) lower λ_max so CFM dominates the monitor, and (ii) `patience=100` so temporary LPL fluctuations do not fire EarlyStopping. No pre-launch investigation is needed — the fix is baked into every batch-2 YAML.

## 5. Launch log

- `git rev-parse HEAD` at launch: `42b6494` (same as batch 1 — no code changes between batches).
- Preflight verifications (all PASSED before submission):
  - `LplConfig.from_decision` parses all 4 YAMLs cleanly on the local vena env (block, w_l, α, t_min, outlier_k round-trip as expected; see the `~/.conda/envs/vena/bin/python` in-repo dry-run).
  - `picasso:/mnt/home/users/tic_163_uma/mpascual/execs/vena/experiments/2026-06-22_15-20-57_s1_v3b_rw_concat_plus_cn3ch_fft_320b5ddd/checkpoints/ema_best.ckpt` present (4 040 846 961 bytes, mtime 2026-06-26 04:45).
  - `decoder_lpl_profile/LATEST/decision.json` symlink resolves to `2026-06-18T20-18-44Z`.
  - `corpus_picasso.json` present.
  - 4 launcher dry-runs on Picasso emit expected sbatch commands.
- Submission at 2026-07-02 ~13:35 CEST via `bash launcher_picasso_s3lpl_b2_<arm>.sh` on Picasso (`REPO_DIR=/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA`):

| Arm | Job ID | Node | Status at launch |
|-----|-------:|-----|------------------|
| B2-A (λ=0.30 standard) | 1428339 | exa03 | RUNNING immediately |
| B2-B (λ=0.10 standard) | 1428340 | exa04 | RUNNING immediately |
| B2-C (λ=0.12 region) | 1428341 | exa04 | RUNNING immediately |
| B2-D (λ=0.30 preflight A) | 1428342 | exa04 | RUNNING immediately |

All 4 jobs allocated 2× A100 40GB, 16 CPUs, 256 GB RAM, `--time=7-00:00:00`, `--constraint=dgx`. Three jobs co-reside on `exa04` (8× A100 node, 2 GPUs per job = 6 GPUs used, fits). No queue wait.

Monitoring commands:

```
ssh picasso 'squeue -j 1428339,1428340,1428341,1428342'
ssh picasso 'tail -20 /mnt/home/users/tic_163_uma/mpascual/execs/vena/logs/vena-s3lpl-b2-a-lambda030-standard_1428339.out'
ssh picasso 'ls /mnt/home/users/tic_163_uma/mpascual/execs/vena/experiments/ | grep b2_'
```

## 6. Results

All four arms ran to the full 250-epoch schedule (`epoch 249 done` → `FM-train completed`), `sacct` State=COMPLETED, ExitCode 0:0. EarlyStopping never fired (the `patience=100` mitigation from §4 worked). Local mirror: `/media/mpascual/Sandisk2TB/research/vena/results/fm/vena/lpl_module/<run_id>/`.

### 6.1 Run summary

| Arm | Job | Final ep | Wall-clock | measured LPL_raw (ep0) | **measured LPL share** | design target |
|-----|----:|---------:|-----------:|-----------------------:|-----------------------:|-------------:|
| B2-A (λ=0.30, std, A=[2,5]) | 1428339 | 249 | 5 d 22:44 | 2.6490 | **58.6 %** | ~58 % ✓ |
| B2-B (λ=0.10, std, A=[2,5]) | 1428340 | 249 | 6 d 14:32 | 2.6496 | **32.0 %** | ~31 % ✓ |
| B2-C (λ=0.12, region, A=[2,5]) | 1428341 | 249 | 6 d 14:43 | 6.9246 | **59.7 %** | ~58 % ✓ |
| B2-D (λ=0.30, std, **A=[2,3]**) | 1428342 | 249 | 4 d 08:13 | **1.2655** | **40.3 %** | ~58 % ✗ |

Share = `λ_max·LPL_raw / (CFM_raw + λ_max·LPL_raw)`, CFM_raw = 0.5619 (measured at ep 0, all arms).

**Design deviation (B2-D).** §3 estimated B2-D's `LPL_raw` "within 30 % of B2-A's". Measured value is **1.27, i.e. 2.1× *lower* than B2-A's 2.65**, so B2-D's steady-state share landed at 40.3 %, not the intended ~58 %. B2-D is therefore **not strength-matched to B2-A**, and H4 cannot be tested as pre-registered (see §7.4).

### 6.2 Training-loss trajectory

| Arm | CFM ep0 → ep249 | LPL ep0 → ep249 | ΔLPL (relative) |
|-----|-----------------|-----------------|----------------:|
| B2-A | 0.5618 → **0.5380** (−4.2 %) | 2.6490 → 2.4462 | **−7.66 %** |
| B2-B | 0.5619 → **0.5380** (−4.3 %) | 2.6496 → 2.4478 | **−7.62 %** |
| B2-C | 0.5619 → **0.5381** (−4.2 %) | 6.9246 → 6.3751 | **−7.94 %** |
| B2-D | 0.5619 → **0.5381** (−4.2 %) | 1.2655 → 1.1599 | **−8.34 %** |

Both endpoints are **invariant to λ across a 3× range**. This is the batch's central observation; see §7.1.

### 6.3 Validity check — LPL was actually applied

Before interpreting a null, the loss plumbing was verified. At ep 249, `total_mean` reproduces `cfm_mean + λ_max · lpl_mean` **exactly** for every arm:

| Arm | cfm + λ_max·lpl | logged `total_mean` |
|-----|-----------------|--------------------:|
| B2-A | 0.5380 + 0.30·2.4462 = 1.2719 | 1.2719 ✓ |
| B2-B | 0.5380 + 0.10·2.4478 = 0.7828 | 0.7828 ✓ |
| B2-C | 0.5381 + 0.12·6.3751 = 1.3031 | 1.3031 ✓ |
| B2-D | 0.5381 + 0.30·1.1599 = 0.8860 | 0.8860 ✓ |

`hi_frac` (fraction of each batch passing the `t_dn > t_min = 0.4` high-SNR gate) held at **0.60–0.72** throughout, so the gate was not starving LPL of samples. **The null below is a scientific result, not a plumbing bug.**

### 6.4 Image-space results — ΔPSNR_ET, ep 0 (S1 warm-start) → ep 225

Final exhaustive_val cadence is **`epoch_225`** for all arms (cadence = 25 ep, so no ep-250 snapshot). Full CSVs: `<run_id>/exhaustive_val/epoch_225/metrics.csv`.

**Sample size — stated precisely (corrected 2026-07-09).** The evaluation covers **81 independent subjects / 131 scans**, each evaluated at 3 NFE (393 rows per cadence). Per-cohort *subject* counts at NFE=1:

| Cohort | subjects | scans | note |
|---|---:|---:|---|
| BraTS-GLI | 11 | 11 | |
| UCSF-PDGM | 10 | 10 | training distribution |
| BraTS-Africa-Glioma / -Other / -PED / UPENN-GBM | 10 each | 10 each | |
| **LUMIERE** | **10** | **60** | longitudinal: 3–14 scans per subject |
| IvyGAP | 5 | 5 | |
| REMBRANDT | 5 | 5 | |

**LUMIERE is not better powered than the other cohorts.** Its 60 rows are longitudinal timepoints of just **10 subjects** (Patient-015: 14 scans; Patient-007: 10; **Patient-025: 8** — the same subject that batch 1 identified as dominating its spurious +0.5 dB gain). Within-subject scans are strongly correlated, so LUMIERE's effective independent n ≈ 10, identical to UCSF-PDGM. Any LUMIERE Δ must be read against the same ≈±0.5 dB noise floor as every other cohort (§7.3) — it does **not** get a √60 reduction.

Pooled over NFE ∈ {1, 5, 20}:

| Cohort | B2-A | B2-B | B2-C | B2-D |
|--------|-----:|-----:|-----:|-----:|
| **UCSF-PDGM** (train dist.) | **−0.01** | −0.11 | −0.05 | −0.11 |
| **BraTS-GLI** (train dist.) | **+0.04** | −0.14 | +0.03 | +0.08 |
| LUMIERE (OOD, longitudinal) | +0.11 | +0.13 | **+0.28** | +0.18 |
| REMBRANDT | **+0.72** | +0.61 | +0.18 | −0.24 |
| UPENN-GBM | +0.09 | +0.07 | +0.02 | +0.02 |
| IvyGAP | −0.05 | −0.01 | −0.01 | −0.01 |
| BraTS-Africa-Glioma | +0.10 | +0.21 | −0.08 | −0.02 |
| BraTS-Africa-Other | −0.05 | +0.21 | +0.21 | +0.10 |
| BraTS-PED (severe OOD) | −0.07 | −0.08 | −0.05 | −0.09 |

ΔSSIM_ET is ≤ ±0.0034 everywhere (4 decimals; largest is B2-A REMBRANDT +0.0034).

### 6.5 The pre-registered STOP criterion (§8, at NFE=1)

Bar: **ΔPSNR_ET_UCSF ≥ +0.50 dB AND ΔPSNR_ET_LUMIERE ≥ +0.80 dB.**

| Arm | ΔPSNR_ET UCSF-PDGM @NFE=1 | ΔPSNR_ET LUMIERE @NFE=1 | Meets bar? |
|-----|--------------------------:|------------------------:|:----------:|
| B2-A | +0.12 (10.73 → 10.85) | +0.21 (21.72 → 21.94) | ✗ |
| B2-B | −0.18 (10.79 → 10.62) | +0.23 (21.78 → 22.01) | ✗ |
| B2-C | +0.04 (10.76 → 10.80) | **+0.51** (21.65 → 22.16) | ✗ |
| B2-D | −0.17 (10.76 → 10.58) | +0.36 (21.74 → 22.10) | ✗ |

Best UCSF = **+0.12** (needed +0.50). Best LUMIERE = **+0.51** (needed +0.80), and that arm (B2-C) scores +0.04 on UCSF. **No arm meets either threshold. The STOP criterion fires.**

## 7. Analysis

### 7.1 The decisive finding: LPL's gradient does no measurable work

Across a **3× λ range** (0.10 → 0.30), **two α variants** ((1,1) and (2,3)), and **two feature depths** (A=[2,5] and A=[2,3]):

- CFM endpoints at ep 249 are **0.5380, 0.5380, 0.5381, 0.5381** — identical to 4 decimals.
- LPL declines by **−7.66 %, −7.62 %, −7.94 %, −8.34 %** — a spread of <0.8 pp.

The arms share seed 1337, warm-start, and dataloader, so identical trajectories are what you expect **if and only if the LPL gradient perturbs the optimisation negligibly**. Had LPL's gradient been doing work, the runs would have diverged — and critically, **tripling λ from 0.10 to 0.30 would have deepened the LPL decline. It did not** (−7.62 % vs −7.66 %; the λ=0.30 arm reduces its own loss term *no more* than the λ=0.10 arm).

The parsimonious reading: the ~8 % LPL decline is **incidental, not causal**. As CFM improves, the one-step estimate x̂₁ = x_t + α·v gets closer to x₁, so the frozen decoder's features on x̂₁ drift toward those on x₁ and LPL falls — *as a passive readout of CFM progress*. LPL is a **spectator on its own objective** at every strength tested.

This is consistent with §6.4: the 4.2 % CFM improvement bought ~0.0 dB on the training-distribution cohorts. Both the latent-space and image-space evidence say the same thing.

### 7.2 Hypothesis verdicts

**H1 (λ 1.0 → 0.30 preserves LUMIERE gain *and* recovers training-distribution regression) — REFUTED** (as a conjunction; its safety half holds).
- *Recovery clause — CONFIRMED.* Batch 1 lost −0.06 to −1.16 dB on UCSF/BraTS-GLI. Batch 2 B2-A: UCSF **−0.01**, BraTS-GLI **+0.04**. The λ=1.0 damage is gone; λ calibration works exactly as the mechanism in §1.1 predicted.
- *Preservation clause — REFUTED.* The batch-1 LUMIERE gain was ~+0.5 dB; B2-A delivers **+0.11** (+0.21 @NFE=1). Lowering λ did not preserve the gain — it removed it along with the damage.
- **Interpretation:** batch-1's LUMIERE "+0.5 dB" and its UCSF "−1.16 dB" were *the same phenomenon* — a large λ perturbing the model off the CFM optimum, which happened to help one OOD cohort (dominated by Patient-025, per §1) and hurt the training distribution. Remove the perturbation, and both effects vanish. There was never a regularisation benefit to recover.

**H2 (λ dose-response is monotone) — INCONCLUSIVE / no detectable effect.**
A (share 58.6 %) vs B (share 32.0 %): UCSF −0.01 vs −0.11; LUMIERE +0.11 vs +0.13; BraTS-GLI +0.04 vs −0.14. Signs disagree across cohorts and all magnitudes sit inside the ±0.2 dB cohort-to-cohort scatter. Neither "B ≥ A ⇒ light anchor" nor "A > B ⇒ mid regime" is supported. **A dose-response cannot be measured because there is no dose effect** (§7.1).

**H3 (region-α at matched strength beats standard) — INCONCLUSIVE, weakly positive on OOD only.**
The strength match *succeeded*: measured shares 59.7 % (B2-C) vs 58.6 % (B2-A), so this comparison is clean — the λ=0.12 solve in §3 was correct even though the estimated `LPL_raw` (5.65) undershot the measured value (6.92).
- B2-C wins on LUMIERE: **+0.28** pooled / **+0.51 @NFE=1**, the largest single gain in the batch, with no whole-brain degradation (ΔPSNR_whole +0.09).
- B2-C loses on REMBRANDT (+0.18 vs +0.72) and ties/loses on UCSF (−0.05 vs −0.01) and BraTS-GLI (+0.03 vs +0.04).
Not a consistent advantage. The one real signal — region-α helping the longitudinal OOD cohort — is the *only* result in this batch that approaches the pre-registered bar, and it still misses it (+0.51 vs +0.80).

**H4 (feature depth A=[2,3] ≥ A=[2,5] ⇒ block-5 redundant) — UNTESTABLE AS DESIGNED (confounded).**
B2-D's measured `LPL_raw` = 1.27 (vs the ~2.28 assumed in §3), putting its share at **40.3 %**, not 58 %. B2-D is thus mid-way between B2-B (32 %) and B2-A (59 %) in *strength*, while also differing in *depth* — the exact confound H3 was constructed to avoid. Comparing anyway (D vs A): UCSF −0.11 vs −0.01; LUMIERE +0.18 vs +0.11; REMBRANDT **−0.24 vs +0.72**; BraTS-GLI +0.08 vs +0.04. Mixed and small. **No conclusion about block-5 redundancy can be drawn.** Given §7.1, a strength-matched re-run would be uninformative anyway: if the gradient does no work, its feature depth cannot matter.

### 7.3 Cross-cohort behaviour

The training-distribution cohorts (UCSF-PDGM, BraTS-GLI) are flat to marginally negative (−0.14 … +0.08). OOD cohorts drift slightly positive (LUMIERE +0.11 … +0.28; BraTS-Africa-Other +0.10 … +0.21) — consistent with 250 further epochs of CFM training mildly improving generalisation, not with an LPL-specific effect (the drift is present at λ=0.10 as strongly as at λ=0.30).

**BraTS-PED remains catastrophically OOD** and untouched by LPL: PSNR_ET ≈ 3.4 dB, SSIM_ET ≈ 0.194, Δ ≈ −0.08 in every arm. Pediatric enhancement is a separate failure mode; no regulariser at this layer addresses it.

**REMBRANDT is the batch's noisiest cohort** (**n = 5 subjects**, tied smallest): B2-A +0.72 vs B2-D −0.24 — a **0.96 dB swing between arms whose CFM endpoints are identical to 4 decimals**. Since those two models are numerically indistinguishable on the training objective, that swing *is* noise, and it gives a direct empirical estimate of the **per-cohort noise floor at n≈5–10 subjects: σ ≈ 0.5 dB.**

It follows that **no per-cohort Δ below ≈0.5 dB in this batch is distinguishable from noise** — which is *every* number in §6.4. Critically, this includes the batch's best result: B2-C's LUMIERE **+0.51 dB @NFE=1 is ≈1σ**, because LUMIERE's effective n is 10 subjects, not 60 (§6.4). It is not evidence of a region-α effect.

### 7.4 NFE curve

Pooled over cohorts, PSNR_ET is highest at NFE=1 and decreases with NFE (B2-A ep225: 17.32 / 17.01 / 16.65 dB at NFE = 1 / 5 / 20), reproducing the known LPL-biases-toward-NFE=1 pattern — **but the identical ordering is present at ep 0 (17.20 / 16.91 / 16.62), before any LPL training.** The NFE ordering is a property of the S1 warm-start, not an LPL effect. Δ(NFE=1) − Δ(NFE=20) is ≤ +0.09 dB in every arm.

(Note: whole-volume `SSIM` at NFE=1 reads ≈0.22 vs ≈0.90 at NFE≥5 in the raw CSVs, across *all* arms and *both* epochs. This is a known artefact of single-step sampling, unchanged by this batch, and does not affect the ET-region metrics used above.)

### 7.5 Surprises

1. **The four CFM endpoints agreeing to 4 decimals.** Not anticipated; it is the cleanest possible demonstration that the LPL term is optimisation-inert here, and it is stronger evidence than any single Δ-metric.
2. **B2-D's `LPL_raw` being 2.1× lower than estimated.** A=[2,3] with the preflight w_l={2:0.663, 3:1.337} yields raw ≈1.27 despite block-3's documented "18.36 magnitude spike". The spike is evidently normalised away by the preflight weights. The §3 estimate should have been a measurement — a smoke run was promised ("a smoke will confirm before submission") and did not gate the launch.
3. **The batch-1 LUMIERE gain does not survive λ reduction**, which retro-explains batch 1 as perturbation rather than regularisation (§7.2/H1).

## 8. Next-batch recommendation — **STOP**

**Decision: STOP the LPL programme for VENA.** Do not launch batch 3.

### Justification against the pre-registered bar

The bar was set *before* results (§8, batch-2 launch): `ΔPSNR_ET_UCSF ≥ +0.50` **and** `ΔPSNR_ET_LUMIERE ≥ +0.80` at NFE=1. Outcome (§6.5): best UCSF **+0.12**, best LUMIERE **+0.51** (different arms). **Both axes missed, by every arm.**

This is not a "needs more tuning" null. Two independent lines of evidence say the mechanism itself is absent:

1. **Optimisation-inert (§7.1).** λ ∈ {0.10, 0.12, 0.30} and two feature depths give CFM endpoints identical to 4 decimals and LPL declines identical to <0.8 pp. Tripling λ does not deepen LPL's own descent. The gradient does no work.
2. **Adequately powered where it counts (§6.4).** 81 independent subjects / 131 scans × 3 NFE across 9 cohorts, versus batch 1's n=1–2 outliers. Every training-distribution Δ is ≤ |0.14| dB against a per-cohort noise floor of σ ≈ 0.5 dB (§7.3). Note the power is *per cohort* modest (5–11 subjects); the strength of the null rests on **λ-invariance (point 1), which is a within-batch control and does not depend on cohort n at all.**

Tuning axes that remain (t_min sweep, feature-standardisation P2 fix, w_l re-weighting) all modulate a gradient that has been shown to contribute nothing at 3× dynamic range in λ. Spending another ~24 A100·days to sweep them is not justified.

**Cumulative cost of the LPL programme:** batch 1 ≈ 6.6 A100·days + batch 2 ≈ 24 A100·days ≈ **31 A100·days**, best result **ΔPSNR_ET_LUMIERE +0.51 dB** on one OOD cohort at n=60, with **ΔPSNR_ET_UCSF ≈ 0.00** on the training distribution.

### Recommended pivot

Per the batch-2 launch contract, pivot to a regulariser that acts where VENA's error actually lives. Ranked:

1. **Vessel-conspicuity loss (proposal §6.2).** The first-principles choice: VENA's differentiator is SWAN-derived vessel conditioning, and no current loss term supervises vessel-region fidelity. Unlike LPL, it targets a region where the model is measurably weak and where the paper's claim lives.
2. **Image-domain LPIPS on decoded volumes.** Moves the perceptual term out of the frozen-decoder *latent feature* space (where §7.1 shows it is inert) into the pixel space the metrics are computed in. Costlier (requires decode-in-the-loop) but tests the perceptual hypothesis where it can actually bind.
3. **Do neither; bank the S1 v3b_rw baseline.** Defensible. §7.1 shows 250 further epochs of plain CFM bought −4.2 % training loss and ≈0.0 dB image-space — VENA may already be at this architecture's PSNR ceiling, and effort is better spent on the ablation/validation matrix than on regularisation.

### What to carry forward (positive results worth citing)

- **λ calibration works.** The batch-1 training-distribution regression (−0.06 … −1.16 dB) is *fully eliminated* at λ_max ≤ 0.30 (§7.2/H1). If any latent-space perceptual term is ever reintroduced, λ_max ≤ 0.30 is the safe operating point, and the share formula in §6.1 (with **measured**, never estimated, `LPL_raw`) is how to set it.
- **`patience=100` fixed the EarlyStopping pathology.** All four arms reached the full 250 epochs; batch 1 died at 38–93. Keep it.
- **The 9-cohort exhaustive_val protocol is the right evaluation instrument** and should become the default for every future VENA regulariser trial. It also yields a usable **per-cohort noise floor of σ ≈ 0.5 dB at n≈5–10 subjects**, which sets the minimum detectable effect: *any* future per-cohort claim below +0.5 dB is unfalsifiable at current n.
- **Two sample-size traps to avoid next time.** (i) `metrics.csv` rows are `subject × NFE`, so row counts overstate n by 3×. (ii) **LUMIERE's 60 rows are longitudinal scans of only 10 subjects** (Patient-015 ×14, Patient-007 ×10, Patient-025 ×8) — it looks 6× better powered than it is, and Patient-025 is precisely the subject that drove batch 1's spurious gain. **Aggregate LUMIERE per-subject before computing any Δ**, or the batch-1 artefact recurs structurally. If a genuinely better-powered OOD read is needed, increase *subjects*, not timepoints.

### Process lessons

- §3 estimated B2-D's `LPL_raw` and promised a confirming smoke; the smoke did not gate the launch, and the arm shipped at 40.3 % share instead of 58 %, forfeiting H4 (§7.4). **Measure raw loss magnitudes on the warm-start before assigning λ — never estimate.**
- The frontmatter `git_sha: 42b6494` records the *local* HEAD at launch; the Picasso `REPO_DIR` executed at **`a1dd749`** (per every run's `git_commit.txt`). Recorded as `git_sha_executed`. Future launchers should echo the remote SHA into the launch log.

## Cross-references

- Prior batch: `2026-06-28_batch_1_default_recipe.md`
- Parent baseline: `.claude/notes/changes/s1_v3/2026-06-28_s1_v3_results_and_s3_plan.md`
- LPL design doc (frozen): `.claude/notes/changes/decoder_perceptual_loss_s3.md`
- Preflight decision.json: `artifacts/preflights/decoder_lpl_profile/2026-06-18T20-18-44Z/decision.json`
- Berrada 2025: *Latent Perceptual Loss for Flow Matching Models*, arXiv:2506.16744
- LPL journal skill: `.claude/skills/lpl-journal/SKILL.md`
