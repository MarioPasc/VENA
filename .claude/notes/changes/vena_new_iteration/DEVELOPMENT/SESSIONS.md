# Orchestrator session plan — VENA iter-8 (segmenter + mask-injection)

**Status:** ACTIVE (authored 2026-07-22).
**Owner:** the `/orchestrate` orchestrator. Workers never edit this file.
**Design authority:** `../segmenter_conditioning_design.md` (Part A injection, Part B segmenter, §A.8/§B.f iter-8).
**Task ledger:** `00_INDEX.md` (code-dependency graph) + the `NN_*.md` specs + `01_SHARED_CONTRACTS.md` (fact sheet).

## How to use this file (orchestrator contract)

1. At session start, find the **first unticked row** of the master table. That row is your session. Its task
   structure (`∥` / `→` / `[O]` below) overrides your own slot-filling; deviate only with a stated reason in the
   session's notes.
2. Run the `/orchestrate` preflight first (clean tree, baseline `pytest -m "not slow and not gpu"` + `ruff` counts,
   `df -h /`, `ssh picasso 'sacct …'` for anything already running). Session-specific preconditions are under *Gates*.
3. Each task's own `NN_*.md` is the authority on its acceptance + tests. Give each worker exactly: its `NN_*.md`,
   `01_SHARED_CONTRACTS.md`, its lane, and what it must not touch. **Verify every reported number against the
   artifact on disk** — a worker's report is a hypothesis (`/orchestrate` §3).
4. At session end: tick the row, and append to that session's **Orchestrator notes** (append-only) — what closed,
   what didn't and why, premises refuted, baseline numbers, and anything the *next* session must know.

**Notation.** `A ∥ B` — parallel (isolated worktrees, disjoint lanes, ≤3 workers). `A → B` — B starts only after A
is merged green. `[O]` — orchestrator-only, main tree, no workers running. `(gate)` — a human/quantitative gate.

## The arc

**Phase 1 (oracle upper bound, NO segmenter):** derive the SDT-soft `[WT,NETC]` from **GT** labels, validate it
(visual + latent-embedding), wire the v3a + fresh-ControlNet injection, launch 2–3 oracle runs, and decide whether
the injection mechanism is good enough. **Phase 2 (deployable):** build + train the segmenter, cache **predicted**
masks (a one-line `mask_source` swap from the oracle), train the deployable T-06 arm, report the oracle→predicted
gap. **Phase 3:** deferred ablations (CFG, WT up-weight sweep, SPADE).

## Canonical Picasso paths & immutability (LOAD-BEARING — read before S2 / S5 / S6)

**v3a warm-start source — READ-ONLY, NEVER ALTER:**
```
/mnt/home/users/tic_163_uma/mpascual/fscratch/checkpoints/vena_project/2026-06-24_16-00-46_s1_v3a_concat_only_fft_ef000c9f/
```
- The oracle runs warm-start from this directory's `ema_best.ckpt`; the joint jobs (J1–J4) additionally read its
  `trunk_ema_snapshot.pt`. **Treat the whole directory as a frozen input** (`.claude/rules/external-deps.md`): never
  write, move, rename, or overwrite anything under it — not the ckpt, not a sibling file, not on "resume". Every
  T-13/T-06 YAML sets `run.resume_from` to the **absolute path** of this dir's `ema_best.ckpt`; WARM_START is
  **weights-only**, so the source is opened read-only and copied into fresh optimiser/EMA state.
- **⚠ If any tool would write into this path, STOP** — that is a `PREMISE-FALSE`/BLOCKED report, not an action.

**New runs instantiate HERE — writable experiments root:**
```
/mnt/home/users/tic_163_uma/mpascual/execs/vena/experiments/<run_id>/
```
- The FM engine creates a fresh `experiments/<run_id>/` per oracle job (`checkpoints/`, `logs/`, `metrics/`,
  `exhaustive_val/`, `decision.json`). All run output lands here; **nothing** the runs produce goes near the v3a
  source dir. Confirm the engine's experiments-root resolves to this path on Picasso before launch (the run YAML /
  launcher sets it), and that `run_id` does not collide with an existing dir.

## Master checklist

| ✓ | Session | Goal | Task structure | Gates |
|---|---|---|---|---|
| ☐ | **S1 — Oracle soft-mask + validation** | SDT-soft GT `[WT,NETC]` cached in every latent H5, visually + latent-embedding validated | `[O]` preflight → **10** → { **12** ∥ **16** } → **19**(source:gt) → **40**(mask QC + latent embedding) → `(gate)` human mask review | latent H5s writable; scaffold decision |
| ☐ | **S2 — Injection + launch oracle** | v3a + fresh 2-ch ControlNet wired; **5 oracle runs** launched + monitored | { **20** ∥ **21** } → **40**(injection sanity) → `[O]` loginexa smoke → `[O]` launch the 5-job matrix + Monitor | S1 masks cached+validated; v3a ckpt (+`trunk_ema_snapshot.pt` for J1–J4) on Picasso |
| ☐ | **S3 — Oracle verdict** | Injection-sufficiency verdict + region-weight/trunk pick + go/no-go for the segmenter | `[O]` harvest → `[O]` analysis (PSNR_ET / no-regression / FP-safety) → `[O]` verdict | S2 jobs terminal |
| ☐ | **S4 — Segmenter library** | BSF-SwinUNETR + SegResNet, loss, data/K-fold, metrics — built + unit-green | { **11** ∥ **13** ∥ **14** ∥ **15** } | S1 (task 10 merged); S3 = GO; BSF ckpts located |
| ☐ | **S5 — Segmenter training + ensemble** | K+1 models trained; G-SEG report; per-class temperatures | **17** → **18** → `[O]` K+1 Picasso array + Monitor | S4 merged green |
| ☐ | **S6 — Predicted mask + deployable T-06** | `masks/tumor_latent_pred` cached; T-06 arm trained; oracle→predicted gap reported | { **19**(source:predicted) ∥ **22** } → `[O]` cache → `[O]` T-06 launch (reuse 20, `mask_source:predicted`) + Monitor | S5 G-SEG passed |
| ☐ | **S7 — Deferred levers & ablations** | CFG / WT-weight sweep / SPADE-T07 — optional, post-validation | { **30** ∥ WT-weight sweep ∥ SPADE-T07 } (any/none) | explicit human opt-in |

---

## S1 — Oracle soft-mask + validation  *(Phase 1a — the user's "step 1: derivation, step 2: validation")*

**Sequence.**
1. `[O]` **Scaffold decision (surface, non-blocking):** task 10 builds the *whole* `SegmentationConfig`
   (model/data/loss/metrics too), but S1 only exercises targets+derivation. Build the full scaffold now (it is
   cheap and S4 reuses it) unless the human prefers a minimal targets-only scaffold. Record the choice.
2. **10** (Wave 0) — package skeleton + Pydantic config + registries + `segmentation` marker. Merge before 12/16.
3. **Fan out (2 workers):** **12** (soft targets: SDT→sigmoid, per-component/geodesic NETC, harmonise, nesting) ∥
   **16** (derivation: per-class temperature + `pool_to_latent` to `(60,60,40)` + ensemble; S1 uses the **pool**).
4. **19** (source-agnostic derive+cache, run with `source: gt`) — writes `masks/tumor_latent_soft (N,2,60,60,40)`
   into every cohort latent H5 via the shared `vena.data.h5` writer/validator. **The swap guarantee lives here.**
5. **40** — visual QC + **latent-embedding visualization** (Q4 resolved: **per-patient PCA/UMAP of mask-latent
   vectors** coloured by tumour volume/cohort, **+ a slice montage**: one patient per row spanning small→large
   tumour size, **5 columns = 5 tumour-bearing slices**, each = anatomy slice with the soft `[WT,NETC]` mask
   overlaid at **α = 0.7**).
6. `(gate)` **Human mask review:** eyeball `artifacts/validate_masks/LATEST/report.md`. This gates the S2 GPU launch.

**Gates.** Latent H5s are writable (Picasso + local mirror); the crop/registration convention for
`masks/tumor_latent` is confirmed (task 19 read-and-verify). **Q1 resolved: SDT-graded oracle** (SDT→sigmoid→avg-pool
of GT — task 19 as specced).

**Exit criteria.** `masks/tumor_latent_soft (N,2,60,60,40)` present + `assert_*_valid` green in every cohort latent
H5, oracle `masks/tumor_latent` byte-untouched; `NETC_soft ≤ WT_soft`; the QC + embedding figures render and the
human review says the masks are localised and graded; `segmentation` marker registered; fast suite green, ruff clean
on touched files.

**Orchestrator notes (append-only).**
- _(empty)_

---

## S2 — Injection wiring + region-weighting + launch oracle  *(Phase 1b)*

**Sequence.**
1. **Fan out (2 workers):** **20** (serve `masks/tumor_latent_soft` by `data.mask_source: oracle_soft`; two specs
   `mask:wt_soft:identity` + `mask:netc_soft:identity`; v3a-warm-start T-13 run YAML; loginexa smoke) ∥ **21**
   (region-weighted CFM, regions `{Brain = NOT-BG ∩ NOT-WT, WT}`, **equal weights ≡ L1** guaranteed by a test).
   **Merge 20 before 21** (both may touch `lightning/module.py`).
2. **40** (injection-sanity panel) — step-0 identity (`output_scale=0` → residual 0 → output == v3a) + residual
   locality (in-WT vs out-of-WT residual energy). Confirms P1/P2 before spending GPU-days.
3. `[O]` **loginexa smoke** — the T-13 YAML builds the 2-ch ControlNet (hint-net `conditioning_in_channels == 2`)
   and runs 2 optimiser steps (`test-picasso-loginexa` skill).
4. `[O]` **Launch the 5-job oracle matrix** on Picasso (`picasso-sbatch`) — all v3a-warm-start + fresh 2-ch
   ControlNet on the **oracle** SDT-soft mask (`data.mask_source: oracle_soft`), varying trunk policy × `region_weights`:

   | job | trunk | region_weights | purpose |
   |---|---|---|---|
   | **J0** | **freeze** | `{brain:1, wt:1}` | ControlNet-only **lower bound** — real floor (no trunk drift, no loss help) |
   | **J1** | joint-low-LR | `{brain:1, wt:1}` | equal-weight ceiling; **pairs with J0 → the freeze→joint gain** |
   | **J2** | joint-low-LR | `{brain:1, wt:5}` | RW sweep |
   | **J3** | joint-low-LR | `{brain:1, wt:10}` | RW sweep |
   | **J4** | joint-low-LR | `{brain:1, wt:20}` | RW sweep (**watch FP-safety** at the top weight) |

   Only **J0 is freeze** (no trunk EMA needed → sidesteps A.8-§5); **J1–J4 are joint** and require v3a's
   `trunk_ema_snapshot.pt`. **LR = linear warmup → cosine annealing** (v3a's `warmup_steps:1000` + `scheduler:cosine`).
   **⚠ Raise EarlyStopping patience** — the harder objective (adding enhancement) transiently *raises*
   `train/total_epoch` before it improves, so patience 250 risks a premature stop; use **~400–500** and keep every
   epoch checkpoint (exhaustive-val PSNR_ET is the real selection signal, not train loss). Strip the ANSI job-id
   (`/orchestrate` §4); verify `Dependency` not `(null)`; set `--time`/`--mem` per proposal §5.
5. `[O]` **Monitor** — one persistent watch matching every terminal SLURM state; `ScheduleWakeup` while pending.

**Gates.** S1 masks cached + human-validated; the **v3a source dir is present + READ-ONLY** (see "Canonical Picasso
paths & immutability") — `ema_best.ckpt` for all jobs, `trunk_ema_snapshot.pt` for J1–J4 (J0 freeze does not need
it, A.8-§5); every run YAML sets `run.resume_from` to that dir's `ema_best.ckpt` and writes output **only** to
`…/execs/vena/experiments/<run_id>/`, never the v3a source dir.

**Exit criteria.** loginexa smoke green (2-ch build + 2 steps, no shape error); step-0 identity verified; **5 Picasso
jobs RUNNING** (job ids recorded, `Dependency` clean); exhaustive-val cadence writing non-empty per-patient
`metrics.csv` (not the empty-CSV `use_timestep_transform` trap — verify one early epoch); monitor armed.

**Orchestrator notes (append-only).**
- _(empty)_

---

## S3 — Oracle results + injection-sufficiency verdict  *(Phase 1c)*

**Sequence.**
1. `[O]` **Harvest** the 2–3 runs' `exhaustive_val/epoch_NNN/metrics.csv` (per-patient × NFE) — copy back + content-hash.
2. `[O]` **Analysis (re-derive every headline number yourself):**
   - **PSNR_ET** vs v3a (target: recover most of the v3b−v3a ET gap) — **use PSNR_ET, not PSNR_WT** (the WT trap,
     `[[project_s1_v2_tumor_failure_diagnosis_2026_06_22]]`).
   - **No-regression gate (A.6):** `MS-SSIM_brain ≥ v3a − MCID` and `MAE_brain ≤ v3a + MCID`.
   - **FP-safety:** false-enhancement on GT-ET≈0 cases ≈ v3a (mask-gating must not *add* FP).
   - Per region-weight arm; which arm best trades PSNR_ET ↑ vs FP/whole-brain.
3. `[O]` **Verdict:** (a) is the ControlNet injection sufficient to place enhancement given a perfect mask? (b) which
   `region_weights` + trunk policy to carry forward; (c) **GO/NO-GO for the segmenter phase**. Record in a handoff.

**Gates.** S2 jobs terminal (COMPLETED/early-stopped) with ≥ several exhaustive-val cadences on disk.

**Exit criteria.** A table (arm × {PSNR_ET, PSNR_brain, MS-SSIM_brain, MAE_brain, FP-vol}) re-derived from the
per-patient CSVs; the oracle **upper bound** stated; the go/no-go + the chosen recipe written to the handoff; memory
updated with the verdict.

**Orchestrator notes (append-only).**
- _(empty)_

---

## S4 — Segmenter library  *(Phase 2a)*

**Sequence.**
1. **Fan out (≤4 workers, disjoint subfolders):** **11** (`models/` — BSF-SwinUNETR Arm A/B + SegResNet Arm C, fork
   `downstream_seg`) ∥ **13** (`engine/loss.py` — DML+CE, focal-CE, Tversky) ∥ **14** (`data/` — K-fold OOF ⊆
   FM-train + dataset + augmentation) ∥ **15** (`metrics/` — Dice/AHD + ECE/Brier + G-SEG + dual selection).
2. Each owns one subfolder → merges are near-conflict-free; run serially anyway, re-verify the suite each merge.

**Gates.** Task 10 (scaffold) merged (S1); S3 verdict = GO; **BSF SSL checkpoints located** (task 11 reports BLOCKED
with the path it looked for if absent — resolve before/at this session).

**Exit criteria.** All three arms forward `(B,3,·)→(B,2,·)`; BSF load-coverage reported; DML==soft-Dice-on-hard test
green; K-fold plan deterministic + leakage-free; metrics + G-SEG gate + dual selection tested; suite green, new
`segmentation` tests counted.

**Orchestrator notes (append-only).**
- _(empty)_

---

## S5 — Segmenter training + K-fold ensemble  *(Phase 2b)*

**Sequence.**
1. **17** (engine: `SegTrainer` one-model-per-invocation + `predict_oof` ensemble/TTA) → **18** (train routine +
   `decision.json` + `vena-segmentation-train`).
2. `[O]` **Train the K+1 models** as a Picasso array (K fold-models + the all-FM-train model); Monitor.
3. `[O]` **G-SEG evaluation** per cohort incl. Ring B (WT Dice ≥ 0.80, NETC Dice ≥ 0.50; healthy → ~empty); fit
   per-class `T_WT`, `T_NETC`; report DSC **and** Brier/classwise-ECE (dual selection).

**Gates.** S4 merged green; GPU budget on Picasso (K+1 SwinUNETR trainings).

**Exit criteria.** K+1 checkpoints + `temperatures.json` + `fold_plan.json`; the G-SEG table passes (or the
documented fallback to a single coarse WT channel is invoked and recorded); no in-fold self-prediction (OOF routing
asserted).

**Orchestrator notes (append-only).**
- _(empty)_

---

## S6 — Predicted-mask cache + deployable T-06  *(Phase 2c)*

**Sequence.**
1. **Fan out (2 workers):** **19**(source:predicted) (reuse the derive/cache routine → `masks/tumor_latent_pred`,
   temperature + K-fold ensemble mean) ∥ **22** (mask-perturbation augmentation, enabled for T-06 only).
2. `[O]` **Cache** `masks/tumor_latent_pred (N,2,60,60,40)` into every latent H5 (`assert_*_valid`; oracle +
   `_soft` groups untouched).
3. `[O]` **T-06 launch** — reuse the S2 run YAML with **`data.mask_source: predicted`** + perturbation ON + the S3
   region-weight/trunk pick. This is the **one-line swap** the architecture guarantees.
4. `[O]` **Oracle→predicted gap** — PSNR_ET(T-06 predicted) vs PSNR_ET(T-13 oracle); report as a table column
   (unreported in TA-ViT = a VENA contribution).

**Gates.** S5 G-SEG passed; S3 recipe fixed.

**Exit criteria.** `masks/tumor_latent_pred` cached + validated; T-06 run launched + monitored; the oracle-vs-predicted
gap re-derived from per-patient CSVs and recorded; G-SHORTCUT (healthy-control FP≈0) checked on T-06.

**Orchestrator notes (append-only).**
- _(empty)_

---

## S7 — Deferred levers & ablations  *(Phase 3 — explicit human opt-in)*

**Sequence.** { **30** (CFG-at-inference + noise-level `output_scale`, FP-gated) ∥ **WT up-weight sweep**
(`{brain:1, wt:5/10/20}` from the coded mechanism) ∥ **SPADE/adaLN-zero T-07** ablation } — any/none, per the human.

**Gates.** Phase-1/2 validated; explicit opt-in. None of these gate the headline.

**Exit criteria.** Whatever ran is green + merged; each ablation reported against PSNR_ET **and** FP-safety; CFG
defaults remain no-ops unless a guidance sweep is explicitly requested.

**Orchestrator notes (append-only).**
- _(empty)_

---

## Planning decisions (resolved 2026-07-22)

| # | Decision | Affects | Resolved |
|---|---|---|---|
| Q1 | Oracle softening | S1 (task 19) | ✅ **SDT-graded** (SDT→sigmoid→avg-pool of GT; matches the predicted path → true swap) |
| Q2 | Region-weight sweep | S2 launch matrix | ✅ **wt ∈ {1, 5, 10, 20}** (equal + 3 up-weights) |
| Q3 | Trunk policy | S2 launch matrix | ✅ **J0 freeze @ wt:1 (floor) + J1–J4 joint-low-LR @ wt:{1,5,10,20}** = 5 jobs; LR = linear-warmup→cosine; **raise EarlyStopping patience to ~400–500** (harder objective transiently raises train loss) |
| Q4 | Latent-embedding viz | S1 (task 40) | ✅ **per-patient PCA/UMAP + slice montage** (1 patient/row, 5 tumour-slice cols, soft mask α=0.7) |
