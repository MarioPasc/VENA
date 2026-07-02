---
batch_id: 1
theme: default_recipe
launch_date: 2026-06-28
result_date: 2026-07-02
parent_baseline: s1_v3b_rw / 320b5ddd (also s1_v3b / c698f45a, s1_v3a / ef000c9f)
prior_batch: null
n_arms: 4
status: analysed
picasso_job_ids: [1398776, 1398777, 1398778, 1398779]
picasso_run_ids:
  - 2026-06-28_11-16-50_s3_v3a_lpl_k5_standard_fft_s1warm_ab26190c
  - 2026-06-28_11-16-50_s3_v3b_lpl_k5_standard_fft_s1warm_cc4f38ed
  - 2026-06-28_11-16-50_s3_v3b_rw_lpl_k5_standard_fft_s1warm_5dd0fe5b
  - 2026-06-28_11-16-48_s3_v3b_rw_lpl_k5_region_fft_s1warm_6096c987
git_sha: 42b6494
schema_version: "1.0"
---

# Batch 1 — S3-on-v3 default-recipe tournament

## 1. Motivation

The three S1 v3 baselines (`v3a` concat-only, `v3b` concat+CN3ch, `v3b_rw` concat+CN3ch+region-weighted CFM) all hit the 400 000-step training budget at epoch ≈1923 and plateaued — slopes on PSNR_ET fell below 10⁻³ dB/epoch over the last 300 epochs (`.claude/notes/changes/s1_v3/2026-06-28_s1_v3_results_and_s3_plan.md`). The bottleneck is the latent-only CFM objective itself: its median-seeking L1 velocity target is agnostic to the *decoded* image manifold, so tumour-region sharpness saturates once the latent MSE plateaus. Berrada 2025 (*Latent Perceptual Loss for Flow Matching Models*, arXiv:2506.16744) proposed applying a decoder-mediated perceptual loss on the one-step clean estimate `x̂_1 = x_t + α·v`, gated to high-SNR timesteps to keep the decoder trustworthy. The MAISI decoder was profiled in `decoder_lpl_profile` (2026-06-18): the knee of the `x̂_1` feature-reliability curve puts `t_min = 0.40` (Berrada's SD-VAE default was 0.70) and readout blocks A=[2,3] carry the most informative signal per the preflight; production overrode to A=[2,5] per design-doc §5.3.

**The single question this batch asks**: does adding LPL on top of a saturated S1 v3 CFM checkpoint improve PSNR_ET / SSIM_ET, and does region-aware α on LPL stack constructively with region-weighted CFM?

## 2. Hypotheses

1. **H1** (LPL as ET refiner): warm-starting the strongest S1 (v3b_rw) with LPL improves PSNR_ET on training-distribution cohorts (UCSF-PDGM, BraTS-GLI) by ≥+0.5 dB.
2. **H2** (LPL can't substitute mask conditioning): S3-A on v3a (no ControlNet) closes ≤50% of the gap to S3-B/C on v3b/v3b_rw.
3. **H3** (region-α stacks with RW-CFM): S3-D (LPL α=(2,3)) ≥ S3-C (LPL α=(1,1)) on PSNR_ET WITHOUT degrading PSNR_whole.
4. **H4** (LUMIERE OOD holds up): domain-shift regression on the n=8 LUMIERE cohort stays within −0.5 dB PSNR_ET vs the S1 warm-start.

## 3. Experiment design

| Arm | Warm-start | λ_max | warmup | LPL α (wt, notwt) | A | w_l | t_min |
|-----|-----------|------:|-------:|-------------------|:---:|-----------------|------:|
| S3-A | v3a `ef000c9f` (concat only, no CN) | 1.0 | 30 | (1, 1) | [2, 5] | {2:1.0, 5:2.0} | 0.40 |
| S3-B | v3b `c698f45a` (concat+CN3ch, plain L1) | 1.0 | 30 | (1, 1) | [2, 5] | {2:1.0, 5:2.0} | 0.40 |
| S3-C | v3b_rw `320b5ddd` (concat+CN3ch+RW-CFM) | 1.0 | 30 | (1, 1) | [2, 5] | {2:1.0, 5:2.0} | 0.40 |
| S3-D | v3b_rw `320b5ddd` (same as S3-C) | 1.0 | 30 | **(2, 3)** | [2, 5] | {2:1.0, 5:2.0} | 0.40 |

Common: max_epochs=200, patience=30, total_steps=41600 (~200 ep), eff. batch = 2·4 = 8, LR=5e-5 cosine w/ 500-step warmup, EMA decay 0.999, bf16-mixed, seed 1337, `use_timestep_transform=true`, `base_img_size_numel=129024`.

Configs and launchers:

| Arm | YAML | SLURM launcher |
|-----|------|----------------|
| S3-A | `routines/fm/train/configs/runs/picasso_s3_v3a_k5_standard_fft.yaml` | `launcher_picasso_s3_v3a_k5_standard.sh` |
| S3-B | `routines/fm/train/configs/runs/picasso_s3_v3b_k5_standard_fft.yaml` | `launcher_picasso_s3_v3b_k5_standard.sh` |
| S3-C | `routines/fm/train/configs/runs/picasso_s3_v3b_rw_k5_standard_fft.yaml` | `launcher_picasso_s3_v3b_rw_k5_standard.sh` |
| S3-D | `routines/fm/train/configs/runs/picasso_s3_v3b_rw_k5_region_fft.yaml` | `launcher_picasso_s3_v3b_rw_k5_region.sh` |

## 4. Compute budget

Estimated at design time: 200 ep × 35 min ≈ 117 h per arm × 4 arms ≈ 20 GPU-days on 2 A100s per arm (co-resident training + async exhaustive_val). Requested SLURM walltime: 7 days per arm.

## 5. Launch log

Launched 2026-06-28 ~12:30Z from the prior session. `git rev-parse HEAD` at launch was `42b6494` ("change conditioning"). Warm-start ema_best.ckpt paths verified present on Picasso before submission. All 4 jobs came up clean per the prior session (`loaded=1978 missing=10 unexpected=0`; LPL ramp = ep/30 exact; zero tracebacks). Picasso job IDs 1398776–1398779.

## 6. Results (populated 2026-07-02)

**Termination — CORRECTED 2026-07-02 after direct log inspection**: all four runs terminated by **EarlyStopping**, not SLURM (sacct exit 0 for all 4; Timelimit was 7 days; the .err logs each carry a `Monitored metric train/total_epoch did not improve in the last 30 records. Signaling Trainer to stop.` line). The initial analysis-agent claim of "SLURM wall-clock" was wrong. The projected monitor `cfm + lambda_max·lpl` hit its minimum very early (S3-A best=2.868 at ep ≈ 6-7) and did not improve for the next 30 records — so at ep ~37 EarlyStopping fired. S3-C/D best occurred later (~ep 62), firing at ep ~92. This means **λ_max=1.0 combined with patience=30 kills S3-LPL training almost immediately**: the LPL raw magnitude drops fastest in the first ~10 epochs (as the model reduces the easy modes) and then oscillates in the 2.0–2.4 range with no sustained trend, keeping the projected total pinned to its early minimum. Batch 2 uses `patience=100` on the projected monitor and much lower λ_max so the CFM (stable) dominates the monitor rather than the LPL (noisy).

**Aggregate PSNR_ET, NFE=5, at the last exhaustive_val cadence per arm**:

| Arm | last ep | UCSF-PDGM | LUMIERE | BraTS-GLI | IvyGAP | REMBRANDT | BraTS-Africa-G | BraTS-Africa-O | UPENN-GBM |
|-----|--------:|----------:|--------:|----------:|-------:|----------:|---------------:|---------------:|----------:|
| S3-A | 25 | 4.73 | 19.99 | 10.96 | 11.78 | 7.33 | 10.20 | 9.04 | 9.18 |
| S3-B | 25 | 9.41 | 17.99 | 15.99 | 16.07 | 15.93 | 13.11 | 9.40 | 11.25 |
| S3-C | 75 | 10.70 | 19.55 | 18.19 | 16.44 | 14.74 | 13.18 | 8.77 | 11.23 |
| S3-D | 75 | 10.36 | 19.55 | 17.93 | 16.58 | 14.50 | 14.57 | 8.52 | 11.66 |

**ΔPSNR_ET vs the warm-start (ep 0 = S1 endpoint checkpoint), NFE=5**:

| Arm | UCSF | LUMIERE | BraTS-GLI | IvyGAP | REMBRANDT | UPENN-GBM |
|-----|-----:|--------:|----------:|-------:|----------:|----------:|
| S3-A | −0.01 | **+0.34** | −0.33 | +0.09 | +0.16 | −0.04 |
| S3-B | **−1.16** | +0.83 | −0.72 | +0.00 | +1.24 | −0.21 |
| S3-C | −0.06 | **+0.65** | −0.14 | +0.00 | +0.32 | +0.84 |
| S3-D | −0.54 | +0.74 | −0.21 | +0.12 | +0.20 | +0.23 |

NFE=1 shows larger LUMIERE gains: S3-D week-000 → week-070 mean +1.47 dB over warm-start at NFE=1. NFE=1 uniformly ≥ NFE=5 on aggregate PSNR_ET (see §7 for the mechanistic reason).

**Training-loss trajectories** (post-warmup epochs at λ=1.0):

- S3-C: cfm 0.58→0.58 (flat), lpl 2.65→2.30 (−13% over 63 epochs). Total 3.20→2.73.
- S3-D: cfm 0.58→0.58, lpl 6.71→5.65 (−16%). Total 7.28→6.21. The region variant carries the same warm-start CFM but ~2.5× the LPL magnitude.
- S3-A/B (7 post-warmup epochs): cfm essentially flat, lpl noisy, no discernible trend.

**LPL contribution to total loss at λ_max=1.0 steady state**:

| Arm | CFM_raw | LPL_raw | Total | LPL share |
|-----|-------:|-------:|------:|----------:|
| S3-A | 0.89 | 1.98 | 2.87 | **69 %** |
| S3-B | 0.85 | 1.82 | 2.67 | **68 %** |
| S3-C | 0.58 | 2.30 | 2.87 | **80 %** |
| S3-D | 0.58 | 5.65 | 6.23 | **91 %** |

The LPL term dominated the gradient in every arm; on the region variant it drove 91 % of the loss.

**S3-C vs S3-D identity check** (previous-session claim of "identical results"): REFUTED. Per-epoch LUMIERE NFE=5 |Δ_PSNR_ET| max = 7.92 dB at week-019 ep 50 (S3-C 30.72 vs S3-D 22.80). The runs are genuinely different models; the identical CFM values epoch-by-epoch trace to identical CFM formulas, not identical weights. The previous session's memo was recording the 2026-06-19 P0-killed batch, not this v3 tournament.

**Wall-clock**: S3-A 22h49m, S3-B 22h59m, S3-C 58h49m, S3-D 53h43m. Total ≈ 158 A100-hours.

## 7. Analysis

**H1 — LPL as ET refiner**: *REFUTED for training-distribution cohorts*. S3-C on the strongest baseline moves UCSF-PDGM PSNR_ET −0.06 dB and BraTS-GLI −0.14 dB — both within the noise floor but the sign is wrong. The refinement target is not being met.

**H2 — LPL can't substitute mask conditioning**: *CONFIRMED*. S3-A on UCSF-PDGM PSNR_ET stays at 4.73 dB — an architectural floor imposed by v3a's no-ControlNet channel-concat-only conditioning. LPL cannot recover from the missing mask signal.

**H3 — Region-α stacks with RW-CFM**: *INCONCLUSIVE, tilting REFUTED*. S3-D matches or slightly under-performs S3-C on training-distribution cohorts (UCSF −0.34, BraTS-GLI −0.26 vs S3-C) but matches on LUMIERE. Since the LPL term already contributed 91 % of total loss for S3-D (vs 80 % for S3-C), the region-α is fighting the region-weighted CFM (which pushes ET correctness at et=300) while LPL pushes notWT decoder agreement (α_notwt=3). The two are opposed at the gradient level, but the LPL side dominates. The test wasn't at fair regularisation strength.

**H4 — LUMIERE OOD holds up**: *CONFIRMED (better than target)*. LUMIERE PSNR_ET improved by +0.34 to +1.47 dB (NFE=1) across all four arms. This contradicts the previous session's memo which reported "1.6–3.6 dB regression" — that claim was likely a mis-read from the 2026-06-19 P0-killed batch (in which no LPL training happened; the "regression" was noise floor across the same warm-start weights).

**Cross-cohort observation**: LPL helps the small OOD cohort (LUMIERE, n=8) more than the training-distribution ones (UCSF n=2, BraTS-GLI n=1). This is counter-intuitive but likely reflects two effects: (i) the decoder features are more shift-invariant than the latent MSE, so LPL is a milder regulariser that pulls slightly-mismatched latents back to the manifold; (ii) LUMIERE is longitudinal (all patients from `Patient-025`), so any single-patient improvement multiplies across timepoints. This is not evidence that LPL generalises well — the ET refinement on the actual training distribution stayed flat or regressed.

**NFE=1 uniformly beats NFE=5**: this is a signature effect of LPL training. LPL supervises the **one-step clean estimate** `x̂_1 = x_t + α·v` at every timestep independently; it never sees multi-step trajectories. The model learns to produce accurate one-step reconstructions and does not internalise a smooth velocity field for chained Euler integration. Berrada 2025 §5.2 warns about this. This is a design characteristic of LPL, not a bug; it is however inconvenient for downstream inference where NFE=5-10 was expected to be the sweet spot.

**λ_max=1.0 was too aggressive**. LPL should function as a regulariser on a converged CFM baseline; at ≥80 % of total loss it is instead the *primary objective* and CFM becomes a soft anchor. The 2026-06-18 design doc §5.3 flagged λ_img as a sweep axis but never executed the sweep; batch 1 inherited λ=1.0 as a unit-scale Berrada-2025 port.

**Preflight-vs-production feature-depth deviation**: preflight (`decoder_lpl_profile/LATEST/decision.json`) emitted `A_recommended=[2, 3]` with `w_l = {2: 0.663, 3: 1.337}`; production overrode to `A=[2, 5]` with `w_l = {2:1.0, 5:2.0}` per design-doc §5.3 (canonical K=5 for cross-scale supervision). No sweep confirmed which is better on MAISI-v2. Block 5 is deeper into the decoder and may leak image-space information that CFM already learns, reducing the perceptual signal's independence.

**EarlyStopping constraint (was: mis-diagnosed as SLURM wall-clock)**: S3-A/B stopping at 7 post-warmup epochs is grossly insufficient to draw conclusions. The correct diagnosis is that with λ_max=1.0 the projected monitor `cfm + lambda_max·lpl` is dominated by the noisy LPL signal (LPL raw ≈ 2× CFM raw on v3b_rw), so the best score sits at an early epoch and EarlyStopping with patience=30 fires almost immediately. Batch 2 must (i) drop λ_max so CFM dominates the monitor and (ii) raise patience so noise does not kill training.

## 8. Next-batch recommendation: CONTINUE (batch 2 — λ calibration)

The evidence is not sufficient to kill the LPL programme. Two facts point to under-tuned rather than misapplied:

1. LPL raw magnitude was 2–7× CFM raw magnitude → λ_max=1.0 turned LPL into the driver, not the regulariser it should be.
2. S3-A/B only saw 7 post-warmup epochs — the effect of LPL on those arms is not measurable.

**Batch 2 axes** (single warm-start = v3b_rw, four arms):

1. **λ calibration (standard)**: λ_max ∈ {0.10, 0.30} → LPL contribution ∈ {~30 %, ~58 %}. Pins the mid-λ point where CFM stays dominant.
2. **Region-α at matched strength**: λ_max=0.12 with α=(2,3) → LPL contribution matched to λ_max=0.30 standard (~58 %). Isolates the region axis at equivalent regularisation strength (batch 1's S3-D was 91 % LPL, not comparable to S3-C at 80 %).
3. **Preflight-aligned feature depth**: λ_max=0.30 standard with A=[2, 3] and preflight w_l={2:0.663, 3:1.337}. Isolates the depth-of-readout axis at fixed λ.

**Batch 2 will NOT sweep**:
- `t_min` — requires adding `t_min_override` kwarg to `LplConfig.from_decision`; deferred to batch 3.
- Feature standardisation on hi-SNR-only (design-doc P2, unfixed) — deferred to batch 3 or 4.
- Warm-start from `last.ckpt` vs `ema_best.ckpt` — deferred; keep ema_best consistent for now.

**Compute budget for batch 2**: max_epochs=250 (buys +50 ep headroom vs batch 1), patience=100 (buys convergence room past early SLURM cut), total_steps=52000. Wall-clock request 7 days per arm; on the observed 35 min/ep this delivers ~285 ep even without SLURM early cut. 4 arms × ~4-5 A100-days = ~16-20 A100-days total. Comparable to batch 1's 158 A100-hours.

**Stopping criterion for batch 3 decision**: if batch 2's best arm shows ΔPSNR_ET ≤ +0.5 dB on UCSF-PDGM and ≤ +0.8 dB on LUMIERE at NFE=1 relative to the S1 warm-start, we STOP the LPL programme and pivot to a different regulariser (candidate: image-domain LPIPS on decoded volumes; vessel-conspicuity loss per proposal §6.2).

## Cross-references

- Parent baseline: `.claude/notes/changes/s1_v3/2026-06-28_s1_v3_results_and_s3_plan.md`
- LPL design doc (frozen): `.claude/notes/changes/decoder_perceptual_loss_s3.md`
- LPL 2026-06-20 post-mortem (frozen): `.claude/notes/changes/decoder_perceptual_loss_s3_analysis_2026-06-20.md`
- Preflight decision.json: `artifacts/preflights/decoder_lpl_profile/2026-06-18T20-18-44Z/decision.json`
- Local results mirror: `/media/mpascual/Sandisk2TB/research/vena/results/fm/vena/lpl_module/`
- Batch 2 successor: `.claude/notes/changes/lpl/2026-07-02_batch_2_lambda_calibration.md`
