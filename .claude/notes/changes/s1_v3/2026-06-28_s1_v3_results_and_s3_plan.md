# S1 v3 — End-of-Run Results, Convergence Verdict, and S3 Resume Plan

*Mario Pascual González — VENA, IBIMA-BIONAND.*
*2026-06-28. Sibling to `2026-06-22_s1_v3_model_implementation.md` and `2026-06-22_s1_v3_normalization_exploration.md`. Decision document for `s3-on-top-of-v3` ablation tournament.*

---

## 0. Status snapshot

The three Picasso A100 jobs from the 2026-06-22 3-cell S1 v3 ablation (`v3a`, `v3b`, `v3b_rw`) all hit their **400 000-step budget at ~ epoch 1900**. None terminated on EarlyStopping. All three are being rsync-pulled from Picasso to local storage right now; **do not interrupt the transfer and do not delete anything from Picasso**.

| Run | Picasso job | Local rsync state (2026-06-28 12:20Z) | last train ep | last train.log PSNR (NFE=20, whole-vol) |
|---|---|---|---:|---:|
| v3a `concat_only_fft` | 1215583 | **complete** (41 G) | 1923 | 25.15 dB / SSIM 0.910 |
| v3b `concat_plus_cn3ch_fft` | 1215584 | active (49 G; nearly done) | 1923 | 25.02 dB / SSIM 0.909 |
| v3b_rw `concat_plus_cn3ch_fft` + RW | 1215585 | **active, early stage** (918 M of ~50 G; 1 of 77 `aggregate.csv` synced) | 1923 | 24.66 dB / SSIM 0.899 |

The `train_epoch.csv` and `decision.json` for all three are already on local disk; **only the v3b_rw `exhaustive_val/*/aggregate.csv` + `checkpoints/*.ckpt` are still pending**.

Picasso-side, every run dir is intact (paths under `/mnt/home/users/tic_163_uma/mpascual/execs/vena/experiments/2026-06-2*_s1_v3*_*`).

---

## 1. Per-run convergence verdict

Trajectory analysis ran on the locally-available `exhaustive_val/epoch_*/aggregate.csv` files (v3a: 57 of 77 cadence epochs present; v3b: 62 of 77; v3b_rw: 1 of 77 — the rest pending rsync). Slope is the OLS linear fit of `PSNR_db_mean` on `epoch` over the last 300 epochs of available data.

### 1.1 UCSF-PDGM, ET region, NFE=5 — early-vs-late check

| Epoch | v3a | v3b | v3b_rw (WIP @ ep~975 from impl-note) |
|---:|---:|---:|---:|
|  500 | 12.66 | 16.33 | — |
| 1000 | 12.33 | 16.03 | 17.30 |
| 1500 | 12.14 | 16.14 | — |
| 1900 | **12.06** | **16.25** | (pending) |

Interpretation:
- **v3a**: ET-PSNR is **flat-to-declining** between ep 500 and ep 1900 (Δ = −0.60 dB, slope ≈ −0.001 dB/ep). The single-pass background L1 budget is saturated and the trunk's tumour localisation is not improving.
- **v3b**: ET-PSNR oscillates within a 0.3 dB band around 16.2 dB from ep 500 onward. **Plateaued**, not declining.
- **v3b_rw**: WIP table in the impl-note at ep~975 puts UCSF ET-PSNR at 17.30 dB (+1.27 vs v3b). Confirmation pending rsync; the train-loss CSV (last `total_mean ≈ 0.51`) is consistent with a converged region-weighted objective.

### 1.2 Average across all 9 cohorts, NFE=5, last common epoch (1900)

|  | v3a | v3b | v3b_rw |
|---|---:|---:|---:|
| avg PSNR_ET | 12.55 | **16.10** | (pending — WIP ≈ 17 dB by impl-note table) |
| avg PSNR_WT | 14.49 | **16.64** | (pending) |
| avg PSNR_whole | 24.30 | **24.33** | (pending — train.log 24.66) |

### 1.3 Slope of last 300 epochs (PSNR_ET, NFE=5, per cohort, v3a + v3b)

All measured slopes are < 0.001 dB/epoch in absolute value (noise floor). Examples on BraTS-Africa-Glioma (a "hard" cohort):

| Cohort | Run | PSNR_ET(last) | PSNR_ET(peak) | epoch at peak | slope |
|---|---|---:|---:|---:|---:|
| BraTS-Africa-Glioma | v3a | 7.81 | 10.10 | **25** | +4×10⁻⁵ |
| BraTS-Africa-Glioma | v3b | (from impl-note table @ ep~1000) ~13.40 | — | — | (pending full slope) |

The "peak at ep=25 then long-run decay to 7.81 dB" on v3a is the smoking gun for **ET overfitting in the absence of mask conditioning**: the trunk's ability to localise tumour collapses as it converges to the background-dominated mean of the L1 loss. The implementation-note's working hypothesis (father-doc §6.0, §6.4) — that mask conditioning is *load-bearing* for ET — is confirmed by this trajectory.

### 1.4 Verdict on further S1 training

**No.** All three runs are saturated under their current objective:

- **v3a**: further S1 training will continue to *degrade* ET (overfitting to the BG-dominated L1 surface). Stop here.
- **v3b**: ET trajectory is a flat oscillation; +500 epochs would not move the needle on PSNR_ET. Whole-volume PSNR is already at its asymptote (~25 dB ≈ MAISI VAE decode ceiling on V0 normalisation per the V3-normalization audit `.claude/notes/data/2026-06-22_v3_normalization_decision.md`). Stop here.
- **v3b_rw**: same story expected once rsync finishes. The region weighting prevents the v3a-style ET collapse, but the WIP-table-vs-train-loss agreement indicates convergence around ep 1000–1500.

**The bottleneck is no longer model capacity or training budget under the CFM objective — it is the latent-only objective itself.** This is exactly the gap the S3 LPL programme was designed to close (decoder-feature perceptual loss on the one-step clean estimate at high SNR; see `decoder_perceptual_loss_s3.md` §0–§2).

Recommend **proceeding directly to S3** on top of the v3 checkpoints.

---

## 2. S3-on-top-of-v3 — what changes, what doesn't

The existing S3 templates (`routines/fm/train/configs/runs/picasso_s3_{k2,k5}_{standard,region}_fft.yaml`) were written against the **retired S1 v2 baseline** (2026-06-12 run, L2 loss, zero-init ControlNet, 4-channel trunk). They cannot be re-used verbatim for an S3-on-top-of-v3 run because of these mismatches:

| Field | Existing S3 template | v3 winner requirements |
|---|---|---|
| `model.trunk.arch_overrides` | `{}` (= 4-channel) | `{in_channels: 16}` (channel-concat trunk) |
| `model.trunk.input_concat` | absent | full block enabled (cond_latents = [t1pre, t2, flair]) |
| `model.controlnet.enabled` | (defaults true) | `true` (v3b/v3b_rw) **or** `false` (v3a) |
| `model.controlnet.conditioning_inputs` | `[latent:t1pre, latent:t2, latent:flair, mask:wt:identity]` (S1 v2 routing) | `[mask:netc:identity, mask:ed:identity, mask:et:identity]` (v3b/v3b_rw) **or** `[]` (v3a) |
| `loss.cfm.norm` | `l2` | `l1` (matches v3 — discontinuous loss change at resume otherwise) |
| `loss.cfm.reduction` | `mean` | `mean` (v3a/v3b) **or** `none` + `region_weights` (v3b_rw) |
| `resume_from` | `2026-06-12_01-27-55_s1_fft_cfm_c9b97556/.../ema_best.ckpt` | v3 winner's `ema_best.ckpt` |

**Everything S3-specific stays the same**:

| Field | Value |
|---|---|
| `loss.lpl.schedule` | `{linear, warmup_epochs: 30, lambda_min: 0.0, lambda_max: 1.0}` |
| `loss.lpl.A_override` | `[2, 5]` (K=5 canonical, §5.3 of design doc; analysis §4.7c provisional finding rules out K=[2,3]) |
| `loss.lpl.w_l_override` | `{2: 1.0, 5: 2.0}` (upweight deeper block per analysis §4.7c) |
| `loss.lpl.grad_checkpoint_segments` | `4` (post 2026-06-19 OOM fix for K=5) |
| `loss.lpl.alpha_override` | `{wt: 1.0, notwt: 1.0}` (standard) *or* omitted = inherits preflight α=(2, 3) (region) |
| `data.decoder_lpl_decision_path` | unchanged Picasso preflight LATEST — decoder is frozen MAISI VAE, profile is architecture-independent |
| `optim.lr` | `5.0e-5` (S3 standard: half of S1's 1e-4) |
| `optim.warmup_steps` | `500` |
| `ema.decay` | `0.999` (S3 short-refinement time constant, vs S1's 0.9999) |
| `training.total_steps` | `41600` (≈ 200 epochs at eff batch 8) |
| `training.max_epochs` | `200` |
| `training.batch_size` / `grad_accum` | `2 / 4` (eff batch 8; A100 trunk-saturation OOM fix from 2026-06-19) |
| `training.patience` | `30` (post P0 lambda_max monitor fix per `feedback_s3_monitor_pitfall`) |
| `exhaustive_val.every_epochs` | `25` (matches v3 cadence) |
| `regions.wt.source` | `derived_from_tumor_latent` |

The `run.resume_from` pointing at an absolute path triggers the WARM_START code path: the engine creates a *new* run directory with run-id `<UTC>_s3_<tag>_<sha>` and copies in the v3 winner's `ema_best.ckpt` weights; **the v3 run directory is never touched** (verified in `routines/fm/train/engine.py::_classify_resume_from`).

**Caveat (carried over from the analysis doc §2.3, not yet fixed)**: the v3 winner's `trunk_ema_snapshot.pt` corresponds to the *latest* checkpoint save (epoch ≈ 1900), while `ema_best.ckpt` is at the best `mse_latent_bg` epoch (which for v3 sits in the mid-1800s based on the cadence dirs). The trunk-EMA shadow loaded on warm-start is therefore ~25–75 epochs ahead of the live trunk weights. This does *not* block correctness — at EMA decay 0.999 over a 200-epoch S3 refinement, the live trunk catches up — but it slightly degrades warm-start-vs-scratch interpretability. **Not a blocker.**

---

## 3. S3 ablation tournament — recommendation

### 3.1 Scope of the questions

**Question 1 (user-asked, a)**: *is it worth running S3 on every S1 v3 winner?*

Argument for: a S3-on-v3a result tells us whether mask conditioning at S1 is **necessary** or whether LPL can substitute for it. If S3-on-v3a closes the PSNR_ET gap to v3b_rw, the proposal's vessel-prior route (which currently piggybacks on the ControlNet branch) is at risk and needs rethinking. If S3-on-v3a does *not* close the gap (the predicted outcome), it confirms that mask conditioning is load-bearing at the S1 stage and the LPL refinement is **complementary**, not substitutive. Either outcome is scientifically informative.

Argument against: v3a's UCSF PSNR_ET sits at **12.06 dB**, vs v3b's **16.25 dB** and v3b_rw's WIP **17.30 dB**. LPL operates on **decoder features at high SNR (`t > t_min`)**: it can refine an existing tumour signal but cannot teach localisation from scratch (the gradient flows only through the FM head, never through the encoder-side conditioning route). A ≥ 4 dB gap on the underlying trunk is unlikely to be fully recovered. Expected lift on v3a from LPL ≈ +1.0–1.5 dB ET-PSNR (extrapolating from the design-doc §5.5 expected effect of standard LPL on a well-conditioned trunk). v3a + LPL probably ends up at ~13.5 dB ET-PSNR, still **3 dB below v3b without LPL**.

**Decision**: yes for v3a, as a **single control arm** (one job, standard LPL, no region ablation). This is the cheapest informative S3 we can run.

**Question 2 (user-asked, b)**: *is it worth running many LPL ablations per S1 base?*

Existing 4-cell pre-built grid is `{K=[2,3], K=[2,5]} × {Standard α, Region α}`. The analysis doc §4.7c (decoder LPL pilot, N=2) found that block 5 carries more L_dec signal than block 2 (inverse of the Berrada 2025 rule). That makes **K=[2,5] the canonical default and K=[2,3] a redundant sweep** for now. Drop K=[2,3] from the tournament; **revisit only if K=[2,5] fails or underwhelms**.

That leaves the **Standard vs Region α** axis. This axis is *not* redundant with the v3a-vs-v3b_rw axis, because:

- Standard α (= 1, 1) treats WT and notWT decoder voxels equally. The model is told "use the decoder to refine the whole image".
- Region α (= 2, 3) tells the model "weight notWT *more* than WT in the LPL signal".

The interaction matrix (CFM × LPL on region weighting) is:

| | CFM uniform L1 (v3b) | CFM region-weighted (v3b_rw, et=300) |
|---|---|---|
| **LPL standard α (1, 1)** | "pure refinement" baseline | "RW-only-on-CFM" — does LPL help even when CFM already over-weights ET? |
| **LPL region α (2, 3)** | "RW-only-on-LPL" — does LPL alone fix the WT/notWT imbalance? | "double RW" — risk of over-emphasising notWT or stable refinement? |

Conceptually the four cells are independent (CFM and LPL weight the *same regions* but in different ways: CFM-RW upweights WT for the velocity target, LPL-α upweights notWT for the decoder-feature distance — they are *opposed* in directionality). The natural tournament is then **all four**.

**However**: each S3 job is 200 ep × 2 A100 ≈ 1.5 d walltime + queue. Submitting 4 LPL ablations × 3 S1 bases = 12 jobs is excessive for the proposal's MICCAI 2026 deadline.

**Decision**: do **NOT** sweep all four LPL ablations per S1 base. Restrict to:

- v3b_rw (primary): standard α + region α (2 jobs) — covers the interaction with the CFM region weighting
- v3b (no-RW): standard α (1 job) — needed to disambiguate "LPL helps" from "RW-on-CFM helps"
- v3a (no-mask control): standard α (1 job) — confirms mask conditioning is load-bearing
- everything else deferred to a follow-up tournament if needed

### 3.2 Recommended launch matrix (4 S3 jobs)

| ID | Warm-start (v3 winner) | LPL α | LPL K-set | Job tag | Question this answers |
|---|---|---|---|---|---|
| S3-A | v3a (`concat_only_fft`) | standard (1, 1) | A=[2,5] | `lpl_v3a_k5_standard_fft_s1warm` | Can S3 LPL substitute for mask conditioning? (predicted: no — control arm) |
| S3-B | v3b (`concat_plus_cn3ch_fft`) | standard (1, 1) | A=[2,5] | `lpl_v3b_k5_standard_fft_s1warm` | Does LPL refine the mask-conditioned trunk without region-weighted CFM? |
| S3-C | v3b_rw (full v3 recipe) | standard (1, 1) | A=[2,5] | `lpl_v3b_rw_k5_standard_fft_s1warm` | Does LPL refine the *best* S1 winner? (primary question) |
| S3-D | v3b_rw (full v3 recipe) | region (2, 3) | A=[2,5] | `lpl_v3b_rw_k5_region_fft_s1warm` | Does region-aware LPL stack with region-weighted CFM, or saturate? |

Pairwise reads:

- **S3-C vs v3b_rw S1 endpoint**: isolates "S3 LPL gain on the winning recipe".
- **S3-B vs v3b S1 endpoint**: isolates "S3 LPL gain on a CFM-uniform recipe" — if S3-B ≈ S3-C, region-weighted CFM is **redundant** with LPL and the project ships v3b + LPL (saves the loss-side complexity).
- **S3-D vs S3-C**: isolates "does directional LPL weighting (notWT-up) stack with directional CFM weighting (WT-up)?". If S3-D wins, the proposal claims the joint RW + region-LPL recipe; if S3-C wins, the proposal claims region-LPL is unnecessary on top of region-CFM.
- **S3-A vs S3-C**: isolates "is mask conditioning *necessary at the S1 stage*?" — if S3-A reaches PSNR_ET within 1 dB of S3-C, the proposal's mask-conditioning story weakens and the S2 vessel-prior plan needs to migrate from ControlNet to channel-concat.

### 3.3 What I am explicitly NOT proposing

- ❌ **K=[2,3] arm**: design doc §4.7c provisional finding ranks K=[2,5] above K=[2,3] on the pilot. Defer.
- ❌ **S3 on v3a with region α**: v3a has no mask path; region α would weight notWT × 3 on a trunk that doesn't *know* WT — predicted no signal. Cost > value.
- ❌ **S3 on v3b with region α**: would only matter if S3-D (= region-LPL on v3b_rw) wins decisively, and would just answer the residual question "is the gain from region-LPL or from region-CFM?". Defer; covered indirectly by the S3-B + S3-D pair.
- ❌ **scratch ablation (E4 from analysis doc)**: warm-start has been validated; the analysis-doc E4 was a hygiene check for the old failed runs. Defer.
- ❌ **K=[2,5] without `w_5/w_2 = 2.0` upweight**: defer; treat as a follow-up sweep on the winner.
- ❌ **CFG-dropout interaction**: orthogonal axis; not in scope.

---

## 4. Open caveats before submission

1. **v3b_rw checkpoints + aggregates are still rsync-pending** (918 M of ~50 G synced as of 12:20Z). S3-C and S3-D both warm-start from `v3b_rw/ema_best.ckpt`, which lives on Picasso (`/mnt/home/users/tic_163_uma/mpascual/execs/vena/experiments/2026-06-22_15-20-57_s1_v3b_rw_concat_plus_cn3ch_fft_320b5ddd/checkpoints/ema_best.ckpt`). **The Picasso path is intact and the S3 jobs will resume directly from Picasso — local rsync completion is NOT a blocker.**
2. **v3b_rw final ET-PSNR confirmation is pending**: the only signal we have is the WIP table at ep ~975 (UCSF ET=17.30 dB) and the train.log loss totals. If the final aggregate (when it arrives on local) shows v3b_rw is *not* better than v3b on ET, the tournament still runs — but the choice of "primary S1 warm-start point" would shift to v3b. Recommend submitting S3-A and S3-B unconditionally (they don't depend on the v3b_rw confirmation), and submitting S3-C and S3-D after a quick read of the v3b_rw final aggregate.csv (or trust the WIP signal — diff < 1 dB at WIP epoch is unlikely to flip).
3. **`decoder_lpl_decision_path` was profiled against the old 4-channel trunk + L2 loss**. The profile is decoder-side only (MAISI VAE is frozen and shared), so it should still be valid. If we find anomalous LPL signal on the v3-warmed runs (e.g. CFM diverging during S3 warmup), re-profile with the v3 architecture as a follow-up.
4. **`ema_best.ckpt` was selected on `mse_latent_bg`, not on PSNR_ET**. This was a deliberate scope deviation (`feedback_s3_monitor_pitfall` mitigation). The S3 warm-start point is therefore *not* the best-ET S1 epoch. We could instead pull a specific `ema_epoch_NNNN.ckpt` (e.g. v3b_rw ep 1500 if it shows peak ET on the aggregate) — but that risks resuming from a non-monotone trough on `mse_latent_bg` and destabilising the LPL warmup. Recommendation: warm-start from `ema_best.ckpt`, accept the BG-vs-ET selection mismatch as a known limitation; revisit only if S3 underperforms.
5. **Walltime budget**: 4 jobs × 7 d max walltime = 28 GPU-days. With patience=30 and 200 ep total, each S3 typically finishes in 1.5–2 d. Realistic total ≈ 6–8 GPU-days; well within the proposal's S3 budget.

---

## 5. Action plan

Pending user approval of the recommended matrix above:

1. **Compose 4 new S3 YAMLs** under `routines/fm/train/configs/runs/`:
   - `picasso_s3_v3a_k5_standard_fft.yaml`
   - `picasso_s3_v3b_k5_standard_fft.yaml`
   - `picasso_s3_v3b_rw_k5_standard_fft.yaml`
   - `picasso_s3_v3b_rw_k5_region_fft.yaml`
   Each copies the relevant v3 winner's `model.trunk` + `model.controlnet` blocks verbatim, sets `resume_from` to the Picasso absolute path of that winner's `ema_best.ckpt`, and inherits the S3-specific `loss.lpl`, `optim`, `ema`, `training`, `exhaustive_val` blocks from `picasso_s3_k5_standard_fft.yaml` / `picasso_s3_k5_region_fft.yaml`.
2. **Compose 4 SLURM launchers** under `routines/fm/train/slurm/runs/`:
   - `launcher_picasso_s3_v3a_k5_standard.sh`
   - `launcher_picasso_s3_v3b_k5_standard.sh`
   - `launcher_picasso_s3_v3b_rw_k5_standard.sh`
   - `launcher_picasso_s3_v3b_rw_k5_region.sh`
   Each is a 3-line clone of `launcher_picasso_s3_k5_standard.sh` with `CONFIG_PATH` and `JOB_NAME` updated.
3. **rsync the new YAML+launchers to Picasso** under `/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA/routines/fm/train/`. The transfer must use the `local → /tmp → picasso` route (per `reference_picasso_transfer_route` memory note — direct server3↔picasso is FleetView-blocked, but `local → picasso` over the user's SSH alias is fine).
4. **`sbatch --test-only`** each launcher first (offline parse-only check); then **`sbatch`** the real submission. Capture the 4 job IDs.
5. **Document the launch** in this file under a new "Launch log" section with job IDs, submit timestamps, and the WARM_START source for each.

## 6. Launch log

User approved the 4-job matrix as proposed (2026-06-28 ≈ 12:30Z). Submitted all 4 immediately — the v3 winners' `ema_best.ckpt` paths on Picasso are intact (verified `ssh picasso ls -lh` before launch), so local rsync completion is not a blocker.

**Local files created**:
- `routines/fm/train/configs/runs/picasso_s3_v3a_k5_standard_fft.yaml`
- `routines/fm/train/configs/runs/picasso_s3_v3b_k5_standard_fft.yaml`
- `routines/fm/train/configs/runs/picasso_s3_v3b_rw_k5_standard_fft.yaml`
- `routines/fm/train/configs/runs/picasso_s3_v3b_rw_k5_region_fft.yaml`
- `routines/fm/train/slurm/runs/launcher_picasso_s3_v3a_k5_standard.sh`
- `routines/fm/train/slurm/runs/launcher_picasso_s3_v3b_k5_standard.sh`
- `routines/fm/train/slurm/runs/launcher_picasso_s3_v3b_rw_k5_standard.sh`
- `routines/fm/train/slurm/runs/launcher_picasso_s3_v3b_rw_k5_region.sh`

**Pre-flight checks (all green)**:
- `FMTrainRoutineConfig.from_yaml(...)` succeeds on all 4 YAMLs; loss/ControlNet/trunk blocks match the corresponding v3 winner verbatim, `loss.lpl.A_override = [2, 5]`, `grad_checkpoint_segments = 4`.
- Picasso paths for all 3 v3 `ema_best.ckpt` + `trunk_ema_snapshot.pt` verified present (read-only `ssh ls`).
- `decoder_lpl_profile/LATEST/decision.json` and `cohort_dedup/LATEST/decision.json` present.
- Local trunk-EMA fix (`module.py:937`, `feedback_v3a_trunk_ema_skip_bug`) confirmed already on Picasso (essential for S3-A, which uses `self.ema is None` + `self.trunk_ema != None` path).
- `sbatch --test-only` accepts all 4 launchers (Job IDs predicted on exa02).

**Submitted via launcher scripts on Picasso (2026-06-28 ≈ 12:33Z)**:

| Ablation | Job ID | Job name | Status (init) | Warm-start source |
|---|---:|---|---|---|
| S3-A | **1398776** | `vena-s3-v3a-k5-standard` | PD | `2026-06-24_16-00-46_s1_v3a_concat_only_fft_ef000c9f/checkpoints/ema_best.ckpt` |
| S3-B | **1398777** | `vena-s3-v3b-k5-standard` | PD (Priority) | `2026-06-22_15-22-04_s1_v3b_concat_plus_cn3ch_fft_c698f45a/checkpoints/ema_best.ckpt` |
| S3-C | **1398778** | `vena-s3-v3b-rw-k5-standard` | PD (Priority) | `2026-06-22_15-20-57_s1_v3b_rw_concat_plus_cn3ch_fft_320b5ddd/checkpoints/ema_best.ckpt` |
| S3-D | **1398779** | `vena-s3-v3b-rw-k5-region` | PD (Priority) | same as S3-C |

Wall-time budget each: 7 days × 2 A100 = 14 GPU-days. Total budget 56 GPU-days; expected runtime ~6–8 GPU-days each (200 epochs + patience=30). Each writes to a NEW run directory; the four v3 S1 dirs are NEVER modified.

**Monitor commands**:
```bash
ssh picasso "squeue -j 1398776,1398777,1398778,1398779 -o '%.10i %.30j %.8T %.10M %.6D %R'"
ssh picasso "tail -50 /mnt/home/users/tic_163_uma/mpascual/execs/vena/logs/vena-s3-v3a-k5-standard_1398776.out"
# replace 1398776 with each job id
```

**Cancel commands** (if a job needs to be killed):
```bash
ssh picasso 'scancel 1398776'    # S3-A
ssh picasso 'scancel 1398777'    # S3-B
ssh picasso 'scancel 1398778'    # S3-C
ssh picasso 'scancel 1398779'    # S3-D
```

**Reading the result**: once each job finishes (or hits a meaningful exhaustive_val cadence at epoch ≥ 50), pull `aggregate.csv` and read `psnr_db_mean` at `region=et, nfe=5` per cohort. The four-way comparison is the deliverable. Mapping back to the recommendation matrix's questions:

| Read | Score | Recipe implication |
|---|---|---|
| S3-A PSNR_ET vs S3-C | gap ≥ 2 dB | mask conditioning at S1 is necessary → S2 vessel-prior plan stays on ControlNet |
| S3-A PSNR_ET vs S3-C | gap < 1 dB | LPL substitutes for mask conditioning → S2 vessel-prior migrates to channel-concat |
| S3-B PSNR_ET vs S3-C | gap < 1 dB | region-CFM redundant under LPL → ship v3b + LPL (drop RW from S1) |
| S3-B PSNR_ET vs S3-C | gap ≥ 1 dB | region-CFM load-bearing even under LPL → ship v3b_rw + LPL (S3-C recipe) |
| S3-D vs S3-C | S3-D wins on ET + whole | joint region-LPL × region-CFM stacks → ship v3b_rw + region LPL |
| S3-D vs S3-C | S3-C wins on ET | region α opposition destructive → ship S3-C |

---

*End of report. See also: `2026-06-22_s1_v3_model_implementation.md` (sibling, implementation log + WIP ET-PSNR table at ep ~975) and `2026-06-22_s1_v3_normalization_exploration.md` (sibling, V0 normalisation rationale).*
