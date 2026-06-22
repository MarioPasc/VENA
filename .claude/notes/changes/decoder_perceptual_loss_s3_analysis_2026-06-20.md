# S3 LPL Null-Result Analysis (2026-06-20)

*Mario Pascual González — VENA, IBIMA-BIONAND / Universidad de Málaga.*
*Companion to `decoder_perceptual_loss_s3.md`. Read after the design note.*

---

## 0. TL;DR

The four S3 LPL runs launched on Picasso on 2026-06-19 did **not** stagnate
because LPL is a bad idea, nor because the LPL gradient never reaches the
trunk. They stagnated because **EarlyStopping was monitoring a quantity
that grows monotonically during LPL warmup**, killing training exactly at
the post-warmup epoch where the LPL signal would have started teaching.
The model checkpoints saved as `ema_best` are essentially the warm-start
point.

| What | Finding |
|---|---|
| **P0 (blocker)** | `EarlyStopping(monitor="train/total_epoch", mode="min", patience=30)` + LPL `warmup_epochs=30` schedule → monitor grows 1.39 → ~3.96 from epoch 0 to 30 *because of the schedule alone*, EarlyStopping interprets this as 30 consecutive no-improvement epochs and fires at epoch 30. `ema_best` is the warm-start. |
| **P1 (cosmetic)** | `train_step.csv::total = cfm` always, even when LPL is in the backward. `CompositeLoss` sets `per_term["total"]` *before* the LPL is added at `module.py:665`. Misled the first telemetry audit into the wrong root cause. |
| **P2 (deferred)** | Decoder-feature extraction + EMA-stats update run on **all** `B` samples, not just hi-SNR. Design note §5.2 sketch only extracts hi-SNR. Adds ~2× VRAM and pollutes feature_stats with off-manifold lo-SNR features. Not the cause of the null result. |
| **Architecture gap vs T1C-RFlow** | Same MAISI latent shape — the user's "different latent resolution" hypothesis is **falsified**. The two recipe deltas that most parsimoniously explain T1C-RFlow's qualitative tumor advantage are **L1 (not L2) velocity loss** and **channel-concat conditioning (not zero-init ControlNet)**. |
| **Tumor-region issue** | Independent of LPL. The S1 (1500-epoch CFM-only) checkpoint plateaued at whole-vol PSNR 26.5 dB / WT-PSNR 18.3 dB with the curve flat past epoch 475. The model has converged to a manifold that under-represents enhancement. LPL alone — even with P0 fixed — was unlikely to break out of this in 25 epochs without a complementary recipe change. |

**Fixes shipped this turn** (`src/vena/model/fm/lightning/module.py`):

1. P0: rewrite `train/total_epoch` to project the LPL contribution onto
   `lambda_max` instead of the schedule-driven `lam_active`, removing the
   spurious warmup ramp from the monitor. S1/S2 (no LPL) is byte-identical.
2. P1: refresh `per_term["total"]` after the LPL is added so the per-step
   CSV reflects the true backward objective.
3. Added `train/cfm_epoch` as a steady secondary signal for stability
   diagnostics.

P2 (hi-SNR pre-gate, region resampling polish, etc.) is **not** shipped
in this turn — it is a design-fidelity issue, not a correctness one.

---

## 1. Evidence chain

### 1.1 The four S3 LPL runs

```
2026-06-19_15-52-00_s3_lpl_k2_region_fft_s1warm_39e2b8e7     # A=[2,3] region
2026-06-19_15-52-00_s3_lpl_k2_standard_fft_s1warm_b08f3d7d   # A=[2,3] uniform
2026-06-19_15-53-09_s3_lpl_k5_region_fft_s1warm_710b7e47     # A=[2,5] region
2026-06-19_15-53-09_s3_lpl_k5_standard_fft_s1warm_4bd84fc7   # A=[2,5] uniform
```

All four:

- Warm-start from `s1_fft_cfm_c9b97556/checkpoints/ema_best.ckpt`.
- `trunk_trainable=true`, `regime=fft` (joint trunk + ControlNet fine-tune).
- `optim.lr=5e-5`, `cosine`, 500-step warmup; `ema_decay=0.999`.
- `lpl.schedule.warmup_epochs=30`, `lambda_min=0.0`, `lambda_max=1.0`.
- `training.patience=30`, `training.max_epochs=200`, `total_steps=41600`.

Outcome at scrape time (2026-06-20 ~08:10 UTC):

| Run | Last epoch | Why it stopped |
|---|---|---|
| `s3_lpl_k2_region_…39e2b8e7` | 30 | EarlyStopping fired |
| `s3_lpl_k2_standard_…b08f3d7d` | 30 | EarlyStopping fired (`FM-train completed` at 05:29 UTC) |
| `s3_lpl_k5_region_…710b7e47` | 12 (running) | not stopped yet — same trajectory toward epoch 30 |
| `s3_lpl_k5_standard_…4bd84fc7` | 12 (running) | same |

### 1.2 The monitor

```
$ ssh picasso "grep -m1 EarlyStopping <run>/logs/train.log"
2026-06-19 17:52:06,077 INFO routines.fm.train.engine
    EarlyStopping ENABLED: monitor=train/total_epoch mode=min patience=30 epochs
```

`engine.py:1295` hardcodes `ckpt_monitor = "train/total_epoch"`, then uses
the same key for the `ModelCheckpoint` (`save_top_k=1, mode="min"`) and
`EarlyStopping`.

### 1.3 What `train/total_epoch` is, before this fix

`module.py:766` (pre-fix):

```python
self.log("train/total_epoch", total, on_step=False, on_epoch=True, batch_size=B)
```

where `total` was, at the moment of the log call, the **live** quantity
fed to `loss.backward()`:

```python
# module.py:588
total, per_term = self.composite(...)                # total = cfm
...
# module.py:665   (only on S3 LPL-active steps)
total = total + lam_active * lpl_scalar              # total = cfm + λ(epoch)·lpl
...
return total                                          # → loss.backward()
```

with `lam_active = compute_lambda_img(self.lpl_config.schedule, epoch)`
ramping linearly `0 → lambda_max=1.0` over `warmup_epochs=30`.

### 1.4 The empirical monitor trajectory

From `s3_lpl_k2_standard_…b08f3d7d/metrics/train_epoch.csv`
(`lpl_skipped_mean=0` everywhere → the try block in `module.py:650-679`
never fails; LPL is computed every step):

| epoch | cfm_mean | lpl_mean | lambda_img_active_mean | live total_epoch (≈cfm + λ·lpl) |
|---|---|---|---|---|
| 0 | 1.3938 | 1.0535 | 0.000 | **1.394** |
| 5 | 1.398 | 1.029 | 0.167 | 1.570 |
| 9 | 1.4142 | 1.0639 | 0.300 | 1.733 |
| 18 | 1.396 | 1.075 | 0.600 | 2.041 |
| 24 | 1.390 | 0.983 | 0.800 | 2.177 |
| 30 | 1.4071 | 1.0421 | 1.000 | **2.449** |

For `s3_lpl_k2_region_…39e2b8e7` (region-weighted, scale ~2.4× higher):

| epoch | cfm_mean | lpl_mean | λ active | live total_epoch |
|---|---|---|---|---|
| 0 | 1.3938 | 2.5728 | 0.000 | **1.394** |
| 9 | 1.4144 | 5.4084 | 0.300 | 3.037 |
| 30 | 1.4075 | 2.5499 | 1.000 | **3.958** |

`EarlyStopping(mode="min")` saw a monotonically growing series, recorded
"no improvement" 30 consecutive times, and triggered at epoch 30. The
"best" `ema_best.ckpt` was saved at epoch 0 (warm-start point + a few
gradient steps). Independently, this is consistent with the user-observed
PSNR profile from the async exhaustive validation:

```
epoch_000: PSNR = 26.60 dB   SSIM = 0.929   (≈ warm-start ceiling)
epoch_025: PSNR = 26.70 dB   SSIM = 0.930   (NFE=5, n=17)
S1 best  : PSNR = 27.03 dB   SSIM = 0.924   (epoch 975)
```

S3 starts **below** S1 best (0.43 dB lower at NFE=5) because the fresh
controlnet EMA at `decay=0.999` and the new feature_stats EMA both need
some steps to stabilise; over 25 epochs of fine-tuning, only 0.10 dB
recovered before the run was killed. The image-space SSIM gain (0.924 →
0.930, +0.006) is consistent with marginal LPL benefit at λ ∈ [0, 0.8]
captured by the EMA — but truncated before reaching the steady-state
window.

### 1.5 Why the first audit misdiagnosed this as "LPL not in backward"

The first telemetry pass observed `train/total = train/cfm` at every step
in `train_step.csv`, including epoch-30 step 6448 where `lambda_img_active=1`
and `lpl=2.17`. With the per-step expected `total = cfm + 1.0·lpl ≈ 1.91`
but the CSV showing `total = 1.687`, the natural inference was "the LPL
tensor is never added to the loss returned by `training_step`".

The actual mechanism is shallower. `CompositeLoss.forward`
(`controlnet/losses/base.py:167`) does:

```python
per_term["total"] = total.detach()                 # total = cfm at this point
return total, per_term
```

The LightningModule then adds the LPL contribution to the **local** `total`
at line 665 but does not refresh `per_term["total"]`. The per-step CSV
column `total` therefore tracks composite-only (cfm) regardless of the
LPL state. The local `total` returned at line 787 — the actual backward
objective — does include the LPL term whenever the gate at lines 600-604
fires. This was verified by:

- `lpl_skipped_mean = 0` across all 30 epochs of both k2 runs → the
  `try` block at line 650-679 never catches; line 665 executes every step.
- `train/total_epoch` (logged at line 766 from the local `total`, not
  from `per_term["total"]`) ramps from 1.39 to 3.96 over warmup, which
  matches `cfm + lam_active * lpl` exactly.
- `lpl_b2_mean` and `lpl_b3_mean` are populated and non-NaN at every
  epoch in `train_epoch.csv` → the LPL forward path is alive.

P0 (the monitor bug) is therefore the operative pathology, not "LPL is
detached". P1 (the per-step CSV is misleading) is the reason the first
diagnosis pointed the wrong direction.

---

## 2. The patch

Two changes to `src/vena/model/fm/lightning/module.py`. Both are
backwards-compatible: S1/S2 runs are byte-identical because their
`per_term` has no `lpl` key.

### 2.1 P0 — steady-state monitor

`module.py:766` becomes:

```python
monitor_total = total
lpl_val = per_term.get("lpl")
cfm_val = per_term.get("cfm")
if (
    self.lpl_loss is not None
    and self.lpl_config is not None
    and isinstance(lpl_val, torch.Tensor)
    and torch.is_tensor(cfm_val)
    and torch.isfinite(lpl_val).all()
):
    schedule = getattr(self.lpl_config, "schedule", None)
    lam_max = (
        float(getattr(schedule, "lambda_max", lam_active))
        if schedule is not None
        else float(lam_active)
    )
    monitor_total = cfm_val + lam_max * lpl_val
self.log("train/total_epoch", monitor_total, on_step=False, on_epoch=True, batch_size=B)
if torch.is_tensor(cfm_val):
    self.log("train/cfm_epoch", cfm_val, on_step=False, on_epoch=True, batch_size=B)
```

What this does:

- `monitor_total = cfm + lambda_max · lpl` — a stationary projection that
  matches the live `total` only at warmup completion. The *gradient* of
  the model still uses `cfm + lam_active · lpl` (the schedule's ramping
  intent is preserved); only the *monitor* sees a stable target.
- `train/cfm_epoch` is exported as a stability check: while LPL teaches,
  CFM should stay flat ±~0.03 (per the design note's "CFM remains the
  latent anchor"). Sudden CFM rise would flag LPL destabilisation.

Why this is logically sound:

| ε ∈ ε_warmup | live total (pre-fix) | steady total (post-fix) | meaning |
|---|---|---|---|
| 0 | cfm + 0·lpl = 1.39 | cfm + 1·lpl ≈ 2.45 | At epoch 0 the model has S1-quality LPL; we monitor the steady-state objective. |
| 15 | cfm + 0.5·lpl ≈ 1.93 | cfm + 1·lpl ≈ 2.45 | During warmup the live objective is half-weighted; the monitor sees the full one. |
| 30 | cfm + 1·lpl ≈ 2.45 | cfm + 1·lpl ≈ 2.45 | At warmup completion the two agree. |

EarlyStopping now sees the actual quantity the run is trying to minimise,
not an artefact of the schedule.

### 2.2 P1 — CSV parity

After `module.py:666`:

```python
per_term["total"] = total.detach()
```

This refreshes the per-step CSV's `total` column so it reflects the
backward objective at that step. Cosmetic, but prevents the next person
debugging an LPL run from chasing the same wrong root cause.

### 2.3 What is *not* fixed in this turn

- **Hi-SNR pre-gate (E.1 from Agent A).** Lines 657-660 extract features
  for all `B` samples and update `feature_stats` from all of them, including
  lo-SNR samples whose `x̂_1` is off-manifold. Design note §5.2 only extracts
  hi-SNR. Fix is non-trivial (need to slice `x1_hat`, `m_wt`, `m_brain`,
  `t_dn` consistently, and decide what to do when zero samples pass the
  gate in a batch). The empirical `lpl_skipped_mean = 0` shows OOM is not
  in play here, so this is *correctness-of-design* but not the cause of
  the null result.

- **Trunk-EMA snapshot epoch consistency (E.3 from Agent A).**
  `TrunkEMASnapshotCallback` overwrites a single `trunk_ema_snapshot.pt`
  at every checkpoint save. On S1→S3 warm-start the loaded live weights
  come from `ema_best.ckpt` (epoch K) but the sibling snapshot is from the
  last save (epoch N ≥ K). If K ≈ N this is harmless. For the S1 1500-ep
  run, `ema_best` is at epoch ~975 (last PSNR maximum) and last is at
  epoch 1494; the trunk-EMA shadow on warm-start is ~520 epochs ahead of
  the live trunk. This is a separate engineering hardening, not the
  null-result cause.

- **`A=[2,5]` for k5 runs.** The `decoder_lpl_profile` preflight only
  characterised `A=[2,3]`; the manual `w_l_override = {2: 1.0, 5: 2.0}`
  used by the k5 YAMLs has no preflight-validated grounding. With P0
  fixed, the k5 runs will at least complete enough epochs to compare
  meaningfully against k2, at which point the `w_l` for block 5 should
  be re-derived from a profile sweep that includes it.

---

## 3. Why the tumor-region failure is a separate problem

The user's load-bearing observation is *qualitative*: "no best figure
includes a tumor; the model skips the tumor entirely or fails to produce
enhancement." This persists in T1C-RFlow vs VENA comparisons and in the
S1 baseline. Fixing the LPL monitor will (a) let LPL actually train past
30 epochs, but (b) is unlikely on its own to unlock tumor synthesis at the
qualitative level the user wants.

The diagnostic is the S1 PSNR trajectory itself:

```
epoch 100: 25.45 dB        epoch 475:  26.44 dB   (gain ~1 dB)
epoch 475: 26.44 dB        epoch 1450: 26.54 dB   (gain ~0.1 dB)
```

S1 plateaued at epoch 475 with WT-PSNR 18.3 dB. The remaining 975 epochs
of training added 0.1 dB. The S1 model has converged to a CFM-optimal
solution that **does not represent enhancement well**. LPL pushes
decoder-feature distance — it does not change the loss-of-information
that the CFM objective already locked in. With S1's converged manifold as
the warm-start, LPL has to overcome a strong attractor on a flat-gradient
loss landscape; 25-30 epochs of LR=5e-5 against a 1500-epoch attractor is
empirically insufficient.

### 3.1 The T1C-RFlow gap is mechanical, not infrastructural

The user's hypothesis "T1C-RFlow uses a different latent resolution" is
**false**. Both VENA and the T1C-RFlow competitor consume the same
MAISI-V2 latent H5 (`corpus_picasso.json`), same brain-box crop, same
spatial dimensions. T1C-RFlow does better at tumor synthesis despite:

- Training from scratch (49.6M params, random init) on a 3-level
  no-attention U-Net — much smaller than VENA's MAISI 4-level + attention
  trunk;
- Using only T1pre + FLAIR (no T2, no SWAN, no WT mask, no offline aug);
- Training for 164 epochs total (~1/10 of S1's wall-clock budget).

The two recipe deltas that most parsimoniously explain the qualitative
advantage, in priority order:

| # | T1C-RFlow choice | VENA choice | Mechanism |
|---|---|---|---|
| 1 | **L1 velocity loss** (`F.l1_loss`) | L2 / MSE (`loss.cfm.norm: l2`) | L1 is median-seeking and preserves sharp transitions; L2 is mean-seeking and smears enhancing-rim boundaries into a halo. Tumor enhancement appears as a thin bright ring with binary-like voxel decisions, exactly the regime where L2 fails visually and L1 wins. Implementation: 1-line YAML change. |
| 2 | **Channel-concat conditioning** at U-Net input (12 channels) | **ControlNet adapter branch with zero-init output projections** (so the residual is *literally zero* at epoch 0) | T1C-RFlow's conditioning is active in every residual block from the first gradient step. VENA's zero-init zero-conv has `∂L/∂W = 0` at the very first step; the conditioning signal cold-starts and propagates only as upstream activations grow. For a sparse high-contrast signal like the tumor's WT mask, the warm-up cost is steep. |

A third, smaller delta:

| 3 | `use_timestep_transform=True` (logit-normal + transform) | logit-normal, no transform | Biases the timestep sampler toward intermediate α (~0.3-0.7) where semantic structure (the enhancement pattern) is encoded. VENA samples uniformly. The gain is ~0.05-0.1 dB whole-vol PSNR but the visual effect on tumor sharpness is plausibly larger. |

Other VENA-vs-Eidex differences (EMA-stabilised sampling, multi-cohort
balanced sampler, offline K=4 augmentation, the MAISI pretrained trunk)
are *advantages* once the conditioning route is fixed, not the cause of
the gap.

### 3.2 What this means for the LPL programme

The LPL design note is mechanistically correct. Decoder-feature
perceptual supervision is a credible Q1-venue contribution and the
preflight (`decoder_lpl_profile`, A=[2,3], `t_min=0.4`, `w_l = {2: 0.66,
3: 1.34}`) is empirically grounded. But LPL on top of a base model whose
conditioning is cold-started and whose loss is mean-blurry will measure
"can LPL repair a blurry tumor outline?" — not "can LPL deliver sharp
clinically-correct enhancement?" The S1 → S3 ablation is therefore
confounded with the L2/zero-conv issue.

The cleanest experimental order is:

1. **Fix the L2 → L1 and (optionally) the zero-init ramp first**, retrain
   S1 to a new converged baseline.
2. **Then** add LPL on top.

This is also defensible scientifically: L2 → L1 is a 1-character change
that other latent-FM medical synthesis papers explicitly favour
(T1C-RFlow itself uses L1). The S1-with-L1 baseline ablation falls out
naturally and strengthens the ablation matrix in proposal §7.

---

## 4. Recommended experiments — revised after 2026-06-20 user discussion

The user's decision after reading §0–§3 is to **retire the current S1
checkpoint** (1500-ep, L2 + hard zero-init) as the internal "L2 + zero-init"
ablation reference and **train a new S1 from scratch with the
recipe-corrected baseline**. This refines E2–E4 from the earlier draft.

### E1 (primary) — new S1 baseline, L1 + scale-ramped zero-init

- `loss.cfm.norm: l1` — one-line YAML change; `CFMLoss` already accepts
  `norm` ∈ {`l1`, `l2`} (verified in
  `vena/model/fm/controlnet/losses/cfm.py`).
- **Replace hard zero-init with a 5000-step sigmoid ramp** on the
  ControlNet output projections (§4a). Removes the cold-start deadtime
  identified in the T1C-RFlow comparison without injecting random
  perturbations into the pretrained MAISI trunk at step 1.
- Optionally enable `use_timestep_transform=True` with
  `base_img_size_numel ≈ 129024` (matching the brain-box latent
  average). See §4b.
- Optionally lift the 1-channel WT mask to 4 channels via a learned
  1×1×1 conv in the conditioning assembler (so its energy matches the
  three latent modalities). See §4c.
- Budget: 1500 epochs, ~7 days on 4× A100. Same wall-clock as the
  retired S1.
- Decision rule: declare new S1 the production baseline if at epoch 475
  (the old S1's PSNR-plateau onset) the whole-vol PSNR ≥ 26.5 dB AND
  WT-SSIM ≥ 0.965 AND `figure_best.png` visibly shows enhancing tumor
  in ≥30 % of validation patients. The retired S1 trajectory becomes
  the "L2 + hard zero-init" ablation row.

### E2 (primary, after E1) — S3 LPL warm-start from new S1

- Warm-start from the new S1's `ema_best.ckpt`.
- Two configs in parallel:
  - `s3_lpl_k2_region_fft_s1warm_v2` — A=[2,3], region-weighted (α_WT=2,
    α_notWT=3 from preflight). The user's preferred LPL recipe.
  - `s3_lpl_k2_standard_fft_s1warm_v2` — A=[2,3], uniform. Ablation
    control.
- Module already patched (P0 monitor fix + P1 CSV parity, this turn).
  No YAML changes needed beyond pointing `resume_from` at the new S1.
- Keep `warmup_epochs=30`, but **raise patience to 100** so even a
  pessimistic LPL-quality flatline gives 70 post-warmup epochs to learn.
- Budget: 100-150 epochs per run, ~3-5 days each on 4× A100. Two runs in
  parallel if 8 GPUs available.
- Decision rule: declare LPL effective if WT-SSIM at epoch 100 exceeds
  new-S1 best by ≥0.005 OR `figure_best.png` shows visibly sharper
  enhancing-rim than new-S1 in ≥30 % of validation patients.

### E3 (secondary, conditional on E2 positive) — A=[2,5] arm

- Defer until E2 confirms LPL is teaching at A=[2,3]. Block 5 was NOT
  characterised by `decoder_lpl_profile`; the `w_l[5]=2.0` is a manual
  extrapolation. If E2 lands, re-run `decoder_lpl_profile` to include
  block 5 before launching k5 jobs (~7 min on loginexa V100 ×4 per the
  PR-1 cost model).

### E4 (ablation, after E2) — S3-from-scratch (§4.5 of design note)

- The user's proposed "S1+S3 from the start with slow ramp until
  epoch 500" is the design-note §4.5 "scratch" ablation arm, not a
  replacement for the warm-start. Lin & Yang 2024 (arXiv:2401.00110)
  showed in the natural-image setting that MSE pretrain → self-perceptual
  finetune (the warm-start curriculum) outperforms joint training at
  equal compute budget — this is the precedent the design note explicitly
  cites in §2.5 as "Primary precedent for the S1→S3 warm-start
  (closer than SRGAN)".
- Run E4 only if E2 confirms LPL is effective; the §4.5 ablation then
  asks the falsifiable question "is the warm-start curriculum
  necessary, or does joint S1+S3 reach the same point at equal
  budget?". Single run, A=[2,3] region, λ_max=1 reached at epoch 500
  via cosine schedule, total 1500 epochs.
- Budget: ~10 days on 4× A100 (longer than E1 because every step pays
  the LPL decoder cost).

### E5 (hygiene, parallel) — P2 hi-SNR pre-gate

- Once E2 starts, retrofit the design-note-compliant
  `hi_mask = t_dn > t_min` pre-gate to halve the VAE-decoder VRAM and
  prevent feature_stats EMA pollution. Bundle with the trunk-EMA
  per-checkpoint-named snapshot fix (E.3 from Agent A). Code change
  only; no compute cost.

### Why not the user's proposed 5-job matrix (S1 + 4× S3-from-start)?

The user initially proposed:
1× S1 (L1, no-zero-init) + 4× S1+S3-from-start over {A=[2,3], A=[2,5]} ×
{standard, region}.

This conflates three independent questions that the proposal §7 ablation
matrix wants answered separately:

| Q | What it asks | Cleanest experiment |
|---|---|---|
| Q_a | Does L1 + ramped zero-init improve over the retired L2 + hard zero-init S1? | E1 alone |
| Q_b | Does LPL help on top of the L1 baseline? | E1 → E2 (warm-start, the literature-precedent path) |
| Q_c | Does S3-from-scratch beat S1→S3 warm-start at equal budget? | E4 (single run, after E2 is positive) |
| Q_d | Region-weighted vs uniform LPL? | The two E2 arms answer this directly |
| Q_e | A=[2,3] vs A=[2,5]? | E3 — but only after Q_b is positive at A=[2,3] |

The 5-job matrix as proposed answers Q_a, Q_c, Q_d simultaneously but
none of them cleanly: every S3-from-start run conflates "L1 helps" with
"LPL helps" with "scratch beats warm-start". The 3-job plan E1+E2 (two
arms) answers Q_a, Q_b, Q_d cleanly at ~13 days of 4× A100 budget
(~70 % less compute than the 5-job plan's ~35 days). E3, E4 are queued
behind a positive Q_b result, which is the correct dependency order
(don't pay for A=[2,5] or from-scratch if A=[2,3] warm-start doesn't
teach).

---

## 4a. Conditioning route — Q1 from the 2026-06-20 user discussion

The user asked whether something "more recent and proven" than
channel-concat (FiLM, etc.) is appropriate, given that the project's main
contribution is the LPL loss, not the conditioning architecture.

**Recommendation: stay with channel-concat.** It is the *de-facto*
standard in 3D medical latent FM/diffusion (Eidex et al. 2025
arXiv:2509.24194; Guo et al. 2025 MAISI; Dayarathna et al. 2025 McCaD;
Biller et al. 2026 TumorFlow). Deviating to FiLM or cross-attention for
spatial map conditioning is principled only when the conditioning is the
wrong shape:

| Mechanism | Conditioning shape | Use case | Verdict for VENA |
|---|---|---|---|
| Channel-concat | Spatial maps `(C, h, w, d)` aligned with the latent | Latent modalities, masks | **Standard. Keep.** |
| Cross-attention (Stable Diffusion / DiT) | Sequence of tokens | Text prompts, class embeddings | Flattens spatial structure → wrong shape for masks. Could fit for cohort_id / scanner_id as a token, but VENA doesn't need that. |
| FiLM (Perez 2018) | Global vector → per-channel scale + shift | Global modifiers (e.g. dose level, sequence type) | Cannot localise enhancement in space. Wrong shape for the WT mask. |
| AdaLN / AdaGN (DiT, SD3) | Global vector → adaptive normalisation params | Time conditioning, class | Same as FiLM — global, not spatial. Useful for time/class injection (DiT already uses it for timestep) but cannot replace spatial-map conditioning. |
| AdaLN-Zero (DiT) | Same as AdaLN, zero-init | Time / class with cold-start guarantee | Has the same cold-start pathology we are escaping. |
| SPADE (Park 2019, CVPR) | Spatial maps → per-pixel scale + shift in normalisation | Semantic image synthesis from masks | **Plausible upgrade for the WT mask**: modulates feature statistics conditional on the mask in every normalisation layer. Adds ~5-10 M params and a non-trivial code surface. Defer — see below. |

For VENA's actual conditioning load (T1pre, T2, FLAIR as 4-channel
latents; WT as a 1-channel mask; brain mask), the right granularity is:

- **Latent modalities → channel-concat** (current behaviour; correct).
- **WT mask → channel-concat at 4 channels** (refinement). The current
  implementation concatenates the 1-channel mask alongside the
  3 × 4 = 12 latent channels (total 13), giving the mask 1/13 = 7.7 % of
  the input energy. The §1 conditioning embedding then projects via a
  3-conv stack [13 → 8 → 32 → 64 → latent_dim], where the 12 latent
  channels dominate the first conv's filter response. Lifting the mask
  via a learned 1×1×1 conv `Conv3d(1, 4, kernel_size=1)` before the
  assembler raises the mask to 4/16 = 25 % of input energy — equal
  share with each latent modality. One-block code change in
  `vena.model.fm.controlnet.assembler`; no parameter inflation outside
  the lifting conv (~16 weights).
- **SPADE as a future ablation, not the new baseline.** SPADE for the
  WT mask is the strongest spatially-localising alternative if and only
  if the channel-concat + lifted-mask refinement still leaves enhancing
  rim under-resolved at the qualitative-figure level. The compute and
  code cost is moderate (the user's primary contribution is LPL — moving
  to SPADE blurs that story). Treat as a §7 ablation row contingent on
  E2 not closing the gap.

The "ControlNet" branch itself remains the residual-injection mechanism;
only the *contents* of the conditioning tensor change. Combined with the
**scale-ramped zero-init** in E1 (replacing the hard zero), the
cold-start objection from the T1C-RFlow comparison (H1) is closed.

---

## 4b. `use_timestep_transform` × LPL gate interaction — Q2

The user asked whether enabling MONAI's `use_timestep_transform=True`
would interact negatively with the LPL high-SNR gate
(`t_dn > t_min = 0.4`, equivalently `α < 0.6`).

**The interaction is positive, not negative.** Both mechanisms push the
training mass toward the same intermediate-α regime where structure
(including tumor enhancement) is formed:

| Mechanism | Effect on training-time α distribution |
|---|---|
| Logit-normal sampler (default, m=0, s=1) | Centred at α = 0.5; smooth fall-off into both tails. |
| `use_timestep_transform=True` (Esser et al. 2024 SD3, `mode_scale`) | Re-weights the logit-normal toward α ≈ 0.5 by a factor proportional to `base_img_size_numel` — concentrates mass at α ∈ [0.3, 0.7], thins the tails. |
| LPL hard gate (Berrada et al. 2025) | Hard cutoff at α < 0.6 (`t_dn > 0.4`). Removes ~30-35 % of training mass at the noise end (α > 0.6). |

The intersection of "transform-concentrated mass" (α ∈ [0.3, 0.7]) and
"LPL-active mass" (α < 0.6) is α ∈ [0.3, 0.6] — the regime where:

- `x̂_1 = x_t + α v_orig` has enough signal that the decoder features
  are on-manifold (the LPL gate's whole rationale);
- The trunk velocity prediction `v_orig` is most informative about
  semantic structure (the transform's whole rationale; cf. Esser et al.
  2024 §3.2 on "structure forms in the middle of the trajectory").

Negative interaction: none. Mildly redundant where both push away from
α ≈ 1 (noise end), but the LPL gate is a hard cutoff and the transform
is a soft re-weighting, so they do not double-count.

**Quantitative expectation:** the empirically observed `hi_frac ≈ 0.65`
(fraction of sampled steps with α < 0.6) under default logit-normal
would rise to ~0.70-0.78 with the transform on. More LPL gradient per
epoch → faster convergence of the LPL-driven contribution.

**Recommendation:** enable `use_timestep_transform=True` together with
the L1 loss in the new S1. The two come from the same paper-family of
solutions to the latent-image disconnect (SD3 / LPL / FlowMo all use
some variant of timestep mass-shaping). Use
`base_img_size_numel = 129024` (≈ 48 × 56 × 48, matching the brain-box
latent average) rather than the upstream T1C-RFlow value of
`64 × 64 × 48 = 196,608`, which is calibrated to the BraTS crop and
slightly off for VENA's narrower brain-box.

Caveat: verify that `use_timestep_transform` is plumbed from the YAML
through `vena.model.fm.sampler.rflow.RFlowEngine` to MONAI's
`RFlowScheduler`. The flag exists upstream but VENA's wrapper may not
expose it; an `__init__` kwarg addition + decision.json schema bump
covers this.

---

## 4c. S1+S3 from the start — Q3

The user proposed a single-pass S1+S3 job with a smooth schedule that
ramps `λ_img` from 0 to 1 over ~500 epochs (λ active from very early,
but at low magnitude until mid-training). The argument: "shape the loss
landscape from the beginning so the model doesn't lock into an
S1-only attractor".

**This IS the design-note §4.5 "S3 from scratch" ablation arm — not the
primary recipe.** Three reasons to prefer the warm-start (E2) as the
primary and defer the from-start (E4) to a secondary ablation:

1. **Stronger literature precedent.** Lin & Yang 2024 ICLR
   (arXiv:2401.00110, *Diffusion Models as Perceptual Networks*) showed
   in the natural-image setting that the MSE-pretrain → self-perceptual-
   finetune curriculum **outperforms** joint training at equal compute
   budget. The design note's §2.5 explicitly tags this as the "primary
   precedent for the S1→S3 warm-start (closer than SRGAN)". The
   from-start arm has weaker precedent and is the one we are asking the
   data to validate.

2. **Decoupled experimental design.** The S1→S3 warm-start path lets us
   measure E1 (does L1 + ramp-zero-init improve over the retired S1?)
   and E2 (does LPL add anything on top?) as independent comparisons.
   A single-pass S1+S3 conflates the two: a positive WT-SSIM result
   does not tell us whether L1 alone got us there or whether LPL
   contributed.

3. **Compute economics.** Every step of an S1+S3 from-start pays the
   LPL decoder cost (partial decode + autograd through K=2 blocks,
   ~3.3 s/step with grad-checkpoint vs ~2.5 s/step for S1 alone).
   At 1500 epochs that's a +1.3× wall-clock multiplier — call it
   ~9-10 days on 4× A100 per run, vs 7 days for E1 alone. If the
   primary contribution turns out to be L1 rather than LPL (which the
   T1C-RFlow comparison suggests is plausible), the from-start arm
   spent that extra compute redundantly.

The cleanest experimental order is therefore:

1. **E1** (S1, L1, ramped zero-init) — establishes the new baseline.
2. **E2** (S1→S3 warm-start, both region and standard) — measures
   marginal LPL contribution on top.
3. **E4** (S1+S3 from-start, single run, A=[2,3] region) — falsifies or
   confirms "the warm-start curriculum is necessary" (§4.5 ablation).

If E2 is negative ("LPL adds nothing on top of L1"), E4 is the salvage
experiment: maybe LPL needs to shape the manifold from the start. If
E2 is positive, E4 strengthens the ablation matrix as a §7 row but is
not load-bearing for the main claim.

---

---

## 5. Open questions

| # | Question | Where to resolve |
|---|---|---|
| Q1 | Is the LPL gradient *magnitude* large enough relative to CFM that 5e-5 LR moves the trunk in 100 epochs? | Inspect `grad_norm_cn_preclip` and `grad_norm_trunk_preclip` from `train_epoch.csv` post-fix; the design-note §3.5 budget assumed `~+30-50%` per-step overhead, which implies similar gradient magnitude order. |
| Q2 | Does the per-channel feature standardisation EMA stabilise within ~5 epochs (design-note §3.3 hypothesis)? | Add a per-epoch log of `feature_stats.var.mean()` per block in a follow-up PR; not load-bearing for the immediate fix. |
| Q3 | Should the k5 runs be stopped now, or allowed to run to natural EarlyStopping? | Stop now. They are spending 40 min/epoch producing the same monitor pathology as k2. Re-launch with the fix when E1 is ready. |
| Q4 | Should `decoder_lpl_profile` be re-run to include block 5 before any k5 LPL experiments? | Yes. The preflight sweep needs an extension to A ⊆ {0,...,5} so `w_l[5]` and `outlier_k[5]` have empirical backing. ~7 min on loginexa V100 ×4. |
| Q5 | Is L2→L1 publishable as a finding even outside the LPL story? | Yes, in the proposal §7 ablation table: one row "loss norm: l2 → l1, all else equal". Independent of LPL. |

---

## 6. Appendix — raw evidence per claim

| Claim | Source |
|---|---|
| EarlyStopping monitor = `train/total_epoch` mode=min patience=30 | `routines/fm/train/engine.py:1295`, `engine.py:1339-1352`; `<run>/logs/train.log` line "EarlyStopping ENABLED: monitor=train/total_epoch …" |
| Per-step CSV `total` column always equals `cfm` | `controlnet/losses/base.py:167` sets `per_term["total"] = total.detach()` before LPL added; verified empirically in `train_epoch.csv` (`total_mean == cfm_mean` to 5 decimals every epoch) |
| LPL forward is alive (gate fires every step) | `train_epoch.csv::lpl_skipped_mean = 0` across all 30 epochs of both k2 runs; `lpl_b2_mean`, `lpl_b3_mean`, `lpl_wt_mean`, `lpl_notwt_mean` all populated and non-NaN |
| LPL is in the backward | `module.py:665` `total = total + lam_active * lpl_scalar`; `module.py:787` `return total`. `lpl_scalar` is built from `partial_decode(x1_hat)` where `x1_hat = x_t + α·v_orig` — `v_orig` carries gradient through the trunk. `decoder_feature_extractor` (`fm/lpl/hooks.py:33`) calls `partial_decode` without `torch.no_grad()`. |
| S3 PSNR at epoch 25 = 26.70 < S1 best 27.03 | `<run>/logs/train.log` "epoch 25 validation … NFE=5 PSNR=26.70±3.21 dB" |
| S3 PSNR at epoch 0 = 26.60 (warm-start ceiling) | `<run>/exhaustive_val/epoch_000/metrics.csv` aggregated NFE=5 |
| S1 PSNR plateau at epoch 475 | exhaustive_val table in `/tmp/vena_t1c_rflow_comparison.md` §4 |
| T1C-RFlow uses L1 | `decision.json` of `2026-06-15T14-04-17_competitor_t1c_rflow_full_multicohort_68a229e` plus upstream source (`F.l1_loss(noise_pred, x_clean - x_noise)`) |
| T1C-RFlow uses channel-concat conditioning | `decision.json` "model_in = cat([noisy_t1c, t1pre, flair], dim=1)" — 12 in-channels into single U-Net |
| Both VENA and T1C-RFlow use the same MAISI-V2 latent | both `corpus_picasso.json`; same VAE checkpoint sha b5ed556dc648… |
| Berrada 2025 LPL uses hard high-SNR gate | arXiv:2411.04873 §3.2 (the gating rule the design note ports verbatim) |
| L1 vs L2 sharpness: median- vs mean-seeking | Lehmann & Casella 1998, *Theory of Point Estimation*, 2nd ed., Springer (Theorem 4.1.2); empirical confirmation in Isola et al. 2017 *Pix2Pix*, CVPR (the L1 vs L2 figure motivating their L1 + adversarial choice) |

---

## 7. Closing note

The LPL programme is **not** abandoned by this analysis. The mechanism
(image-aware supervision via decoder features, hi-SNR-gated) is sound,
the preflight is rigorous, and the integration is functionally correct
(modulo the monitor bug fixed here and the deferred P2 hi-SNR pre-gate).
The four runs of 2026-06-19 failed because of a single misconfigured
monitor key, not because of a deeper methodological problem.

That said, the *tumor synthesis* gap versus T1C-RFlow is a *separate*
issue that LPL is unlikely to close on its own. The L2 → L1 ablation
(experiment E2) is the highest-leverage single change available right
now; it is one-character at the YAML level and should be run in parallel
with the LPL re-launch (experiment E1) on a separate GPU. If E2
demonstrates qualitative tumor improvement, the proposal §7 ablation
matrix gains a strong new row independent of the LPL contribution, and
the LPL story becomes "we ALSO show that decoder-feature perceptual
supervision adds X dB / Y SSIM on top of the L1 baseline" rather than
"LPL alone failed to overcome an L2-blurry baseline".

— end —
