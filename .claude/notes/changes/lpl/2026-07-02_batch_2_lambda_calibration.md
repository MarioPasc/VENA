---
batch_id: 2
theme: lambda_calibration
launch_date: 2026-07-02
result_date: null
parent_baseline: s1_v3b_rw / 320b5ddd
prior_batch: 1
n_arms: 4
status: launched
picasso_job_ids: [1428339, 1428340, 1428341, 1428342]
picasso_run_ids: []
git_sha: 42b6494
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

## 6. Results (populated on completion)

*(POST section — batch is in `designed` state)*

## 7. Analysis (populated on completion)

*(POST section — batch is in `designed` state)*

## 8. Next-batch recommendation (populated on completion)

*(POST section. If the best batch-2 arm delivers ΔPSNR_ET_UCSF ≥ +0.5 dB and ΔPSNR_ET_LUMIERE ≥ +0.8 dB at NFE=1, the LPL programme continues to batch 3 (t_min sweep + feature-standardisation P2 fix). Otherwise STOP the LPL programme and pivot to a different regulariser — image-domain LPIPS on decoded volumes or the vessel-conspicuity loss from proposal §6.2.)*

## Cross-references

- Prior batch: `2026-06-28_batch_1_default_recipe.md`
- Parent baseline: `.claude/notes/changes/s1_v3/2026-06-28_s1_v3_results_and_s3_plan.md`
- LPL design doc (frozen): `.claude/notes/changes/decoder_perceptual_loss_s3.md`
- Preflight decision.json: `artifacts/preflights/decoder_lpl_profile/2026-06-18T20-18-44Z/decision.json`
- Berrada 2025: *Latent Perceptual Loss for Flow Matching Models*, arXiv:2506.16744
- LPL journal skill: `.claude/skills/lpl-journal/SKILL.md`
