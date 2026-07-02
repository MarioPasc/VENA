---
name: lpl-journal
description: Maintain the LPL (Latent Perceptual Loss, Berrada 2025) experiment journal at `.claude/notes/changes/lpl/`. Each batch of LPL tuning jobs launched to Picasso becomes one dated .md entry, updated in place when the jobs return. Preserves the "scientific story" from the S1 v3 CFM baseline through iterative LPL tuning toward MICCAI 2026 / MedIA submission. Invoke when the user says "add an LPL batch entry", "record the LPL results", "start the next LPL iteration", "which LPL batch is this", or when about to launch or analyse S3-LPL jobs. Also cite this skill when writing SLURM launchers or YAMLs under the `picasso_s3lpl_b<N>_*` naming convention.
---

# LPL journal — how to maintain `.claude/notes/changes/lpl/`

## Purpose

VENA is iteratively tuning the Berrada 2025 Latent Perceptual Loss (LPL) module on top of the S1 v3 CFM baselines (`v3a`, `v3b`, `v3b_rw`). The paper-facing narrative requires a *scientific story*: what we tried, why, what happened, what we learned. This journal is that story, one .md entry per batch of Picasso submissions.

The journal serves three readers:
1. **Future Claude sessions** picking up the LPL programme after `/clear`.
2. **The user** reviewing the reasoning trail during paper-writing.
3. **The reviewer / co-author** verifying that hyperparameter choices were principled, not tuned on the test set.

## Where things live

| Asset | Path |
|---|---|
| Journal folder | `/home/mpascual/research/code/VENA/.claude/notes/changes/lpl/` |
| Parent baseline plan | `.claude/notes/changes/s1_v3/2026-06-28_s1_v3_results_and_s3_plan.md` |
| LPL design doc (frozen) | `.claude/notes/changes/decoder_perceptual_loss_s3.md` |
| LPL 2026-06-20 post-mortem (frozen) | `.claude/notes/changes/decoder_perceptual_loss_s3_analysis_2026-06-20.md` |
| Preflight decision.json | `artifacts/preflights/decoder_lpl_profile/LATEST/decision.json` |
| Picasso mirror of preflight | `/mnt/home/users/tic_163_uma/mpascual/execs/vena/artifacts/preflights/decoder_lpl_profile/LATEST/decision.json` |
| Warm-start checkpoints (Picasso) | `/mnt/home/users/tic_163_uma/mpascual/execs/vena/experiments/2026-06-{22,24}_..._s1_v3*_ef000c9f/checkpoints/ema_best.ckpt` |
| Warm-start checkpoints (local mirror) | `/media/mpascual/Sandisk2TB/research/vena/results/fm/vena/2026-06-{22,24}_..._s1_v3*_*/checkpoints/ema_best.ckpt` |
| Local S3 result mirror | `/media/mpascual/Sandisk2TB/research/vena/results/fm/vena/lpl_module/` |
| Batch-N YAMLs | `routines/fm/train/configs/runs/picasso_s3lpl_b<N>_<arm>_fft.yaml` |
| Batch-N launchers | `routines/fm/train/slurm/runs/launcher_picasso_s3lpl_b<N>_<arm>.sh` |

## File naming

`.claude/notes/changes/lpl/<YYYY-MM-DD>_batch_<N>_<theme>.md`

- `<YYYY-MM-DD>` is the **launch** date, never bumped on later edits.
- `<N>` is a monotonically-increasing integer starting at 1. Skip no numbers; a killed / withdrawn batch keeps its number and is marked `status: withdrawn` in the frontmatter.
- `<theme>` is 2–4 words in `snake_case`, chosen to describe the batch's single dominant hypothesis (e.g. `default_recipe`, `lambda_calibration`, `feature_depth_sweep`, `perceptual_ablation`, `stop_lpl`).

Examples:
- `2026-06-28_batch_1_default_recipe.md` — the first S3-LPL tournament
- `2026-07-02_batch_2_lambda_calibration.md` — the follow-up calibrating λ_max
- `2026-07-15_batch_3_feature_depth_sweep.md` — the depth-sweep follow-up

If a batch's design is decoupled from its result analysis by many weeks, the entry stays one file (living doc). Do NOT create a separate `_results` file.

## Frontmatter — required

```yaml
---
batch_id: <int>                           # monotonic
theme: <snake_case>                       # matches filename
launch_date: <YYYY-MM-DD>                 # ISO
result_date: <YYYY-MM-DD | null>          # null until batch completes
parent_baseline: <S1 run_id or ref>       # e.g. "s1_v3b_rw / 320b5ddd"
prior_batch: <int | null>                 # e.g. 1 (for batch 2); null for batch 1
n_arms: <int>
status: designed | launched | in_progress | analysed | superseded | withdrawn
picasso_job_ids: [<int>, ...]             # populated at submission
picasso_run_ids: [<str>, ...]             # populated when the run dirs appear
git_sha: <short-sha>                      # `git rev-parse --short HEAD` at launch
schema_version: "1.0"
---
```

## Body sections — required order

Every entry must contain these sections in this order. Sections marked (LIVE) are populated at launch; sections marked (POST) are populated when the batch returns.

### 1. Motivation (LIVE)

One short paragraph:
- What the prior batch (or the parent baseline) found.
- The one **scientific question** this batch asks.
- The one **theoretical anchor** (cite Berrada 2025 arXiv:2506.16744 or the design doc).

If this is batch 1, the "prior" is the parent S1 baseline; cite its plan doc.

### 2. Hypotheses (LIVE)

Enumerated hypotheses, each with an *observable prediction*:
```
H1. Reducing λ_max from 1.0 → 0.3 preserves LUMIERE +Δ but reduces BraTS-GLI ΔPSNR_ET regression.
H2. …
```
A hypothesis without an observable prediction is not a hypothesis.

### 3. Experiment design (LIVE)

A table `Arm × (warm_start, λ_max, warmup, α, A, w_l, t_min, other)` with one row per arm. Include the target LPL:CFM contribution ratio computed from the previous batch's raw magnitudes (LPL_raw / (CFM_raw + λ_max·LPL_raw)). Never lower than 4 columns; never more than 8.

Include the config path and launcher path per arm.

### 4. Compute budget (LIVE)

Wall-clock estimate per arm (based on ~35 min/epoch at the current config) × N arms. State the SLURM walltime requested and the total GPU-hours.

### 5. Launch log (LIVE)

- `git rev-parse HEAD` at launch time
- `sbatch` submissions: job IDs and paths
- Any pre-flight verification (dry-run of launcher, YAML parse test, ema_best.ckpt path check)

### 6. Results (POST)

Per-arm summary table:
```
| Arm | Final ep | Wall-clock | UCSF PSNR_ET | LUMIERE PSNR_ET | BraTS-GLI PSNR_ET | Notes |
```
Include ΔPSNR_ET vs the parent S1 endpoint (which is `ep 0` of the S3 exhaustive_val — the warm-start state). Cite epoch-level `train/lpl_epoch` and `train/cfm_epoch` trajectory in words (e.g. "LPL 2.65 → 2.30 over 63 post-warmup epochs, CFM 0.58 → 0.58").

### 7. Analysis (POST)

Answer each Hypothesis (H1, H2, …) with **CONFIRMED / REFUTED / INCONCLUSIVE** and the evidence (table row or number). Then discuss:
- Cross-cohort behaviour (training-distribution vs OOD).
- NFE curve (NFE=1 vs NFE=5 — LPL biases toward NFE=1).
- Any surprises.

Do NOT conflate CONFIRMED with "the recipe works". A CONFIRMED hypothesis that only moves the metric 0.1 dB is still a null result.

### 8. Next-batch recommendation (POST)

Choose ONE of:
- **CONTINUE**: propose the next batch (arms, design axis). Justify each arm.
- **STOP**: LPL is not helping VENA further. Justify by pointing to the evidence bar we set (e.g. "3 batches of tuning, best ΔPSNR_ET_LUMIERE +0.7, cost of BraTS-GLI −0.5"). Recommend alternative regularisation (adversarial, vessel-conspicuity, etc.).
- **PIVOT**: change a fundamental knob outside the current design axis (e.g. different feature extractor, different decoder, replace LPL with LPIPS on decoded volumes).

## Rules of engagement

- **Journal is append-only for entries**. Do not delete an old entry. Mark it `status: superseded` and link the successor.
- **Design vs analysis are ONE file per batch**. Add the POST sections to the same file when results arrive; do not create a `_results.md` sibling.
- **Sanity-check the previous session's claims against actual data before citing them.** The 2026-06-28 session summary claimed "LUMIERE regressed 1.6–3.6 dB"; the actual `metrics.csv` shows +0.3 to +1.5 dB gains. Always verify.
- **Cite numbers to 2 decimals** for dB metrics, 4 decimals for SSIM.
- **Every "improved" / "regressed" statement carries a signed Δ**.
- **Every hyperparameter choice cites its source**: preflight `decision.json`, prior batch analysis, design doc §, or Berrada 2025 §.
- **Never skip section 8** (Next-batch recommendation). That's what the next session reads first.
- **Never propose more than 4 arms per batch** without an explicit budget justification in section 4. Picasso queue is scarce; 4 arms × ~6 GPU-days = 24 GPU-days is the ambient cap.

## LPL numerical calibration (as of 2026-07-02, load-bearing)

The batch-1 tournament measured raw loss magnitudes on the S1 warm-start (v3b_rw parent):

| Term | Raw magnitude at warm-start (ep 0) |
|---|---|
| CFM (v3b_rw, region-weighted L1) | ≈ 0.58 |
| CFM (v3b, plain L1) | ≈ 0.88 |
| LPL standard, α=(1,1), A=[2,5], w_l={2:1,5:2} | ≈ 2.28 |
| LPL region,   α=(2,3), A=[2,5], w_l={2:1,5:2} | ≈ 5.79 |

The LPL contribution fraction at steady state is `λ_max · LPL_raw / (CFM_raw + λ_max · LPL_raw)`. For v3b_rw:

| Regime | λ_max standard | λ_max region | Notes |
|---|---:|---:|---|
| LPL dominates (80%+) | ≥ 1.0 | ≥ 0.30 | batch-1 defaults; degrades OOD |
| Balanced (~50%) | ≈ 0.25 | ≈ 0.10 | probable sweet spot |
| Regulariser (~30%) | ≈ 0.11 | ≈ 0.045 | soft anchor |

Use these numbers to justify λ_max choices. Bump the table when a new baseline (v4?) is measured.

## Related VENA rules and conventions

- `.claude/rules/model-coding-standards.md` — FM training conventions (module.py, S3 branch, EMA rules).
- `.claude/rules/preflight-pattern.md` — routine layout, decision.json v0.9.0 contract, PreflightGateError.
- `CLAUDE.md > Rectified-flow timestep convention` — α = timesteps/1000, t_dn = 1−α, x̂_1 = x_t + α·v. The LPL gate uses `t_dn > t_min`.
- `CLAUDE.md > S1 v2 baseline recipe` — describes L1 CFM + scale-ramp + timestep transform (all inherited by S1 v3 and every LPL batch).

## Anti-patterns (do not do)

- Do NOT edit `decoder_perceptual_loss_s3.md` or `decoder_perceptual_loss_s3_analysis_2026-06-20.md` — those are frozen historical records of the original LPL landing.
- Do NOT rename an entry's file after launch. The batch_id and launch_date are the citation key.
- Do NOT copy raw CSVs into the entry body — cite the local mirror path (`/media/mpascual/Sandisk2TB/research/vena/results/fm/vena/lpl_module/<run_id>/`) or the Picasso path.
- Do NOT declare a batch "won" based on training loss alone — the exhaustive_val PSNR_ET / SSIM_ET across cohorts is authoritative.
- Do NOT invent new naming schemes. `picasso_s3lpl_b<N>_<arm>_fft.yaml` is the batch-2+ convention; batch 1 used the ad-hoc `picasso_s3_v3<X>_k5_{standard,region}_fft.yaml`.
