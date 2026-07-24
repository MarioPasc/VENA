# S4 orchestrator launch prompt — Segmenter library

> Paste this to launch the **S4** orchestrator session. It embodies the refined `/orchestrate` skill
> (minimal self-contained briefs, ALWAYS-verify, watch-live-and-steer). Authored 2026-07-23 (iter-9).

---

You are the **orchestrator** for VENA session **S4 — Segmenter library**. **Invoke the `/orchestrate` skill and
follow it as written** — the verification discipline (§3, ALWAYS, every agent), worktree isolation + the split-brain
import trap (§2), minimal self-contained briefs, and watch-live-and-steer. You hold the plan; subagents write the
code; **you verify everything**.

**Authorities (you read these; workers get only their spec + the fact sheet + the brief below):**
- Design: `.claude/notes/changes/vena_new_iteration/segmenter_conditioning_design.md` Part B (+ §B.f, §A.9).
- Session plan: `.claude/notes/changes/vena_new_iteration/DEVELOPMENT/SESSIONS.md` §S4.
- Fact sheet: `.claude/notes/changes/vena_new_iteration/DEVELOPMENT/01_SHARED_CONTRACTS.md`.

## Goal

Build the segmenter **library** — tasks **11 (models) ∥ 13 (loss) ∥ 14 (data/K-fold) ∥ 15 (metrics)** — unit-green,
ruff-clean on touched files, fast-suite test count **strictly up** from baseline. **NO GPU** (library + unit tests
only; S5 trains). **S4 runs in PARALLEL with the oracle track (S1/S2/S3)** — the SEG and INJECT tracks share no code;
S4 is **FULLY UNBLOCKED** (the old "S3 verdict = GO" gate is removed, iter-9).

## Shared brief — give to EVERY worker, inline in the prompt (do NOT make them hunt for it)

- **Grid = `(48,56,48)`** (served MAISI latent). The **segmenter works at IMAGE resolution** with **z-score-on-brain**
  (nonzero, per-channel — the `downstream_seg` convention, NOT the VAE 99.95). Pooling to the latent grid is task 16's
  job (already merged), NOT S4's.
- **Predict `[TC, NETC]`**: channel 0 = **TC = tumour core = NETC+ET = `(label>0)&(label!=2)`, EDEMA EXCLUDED** (NOT
  whole-tumour). `TargetConfig.tumor_region` defaults `"tc"`. `TC−NETC = ET`.
- **🔴 TEMPERATURE IS DROPPED (planning-decision Q5).** No `T_TC`/`T_NETC` fitting, no `temperatures.json`.
  Calibration is **MEASURED** (ECE/Brier, task 15) but **NOT corrected**. Specs 16/17/18 still prescribe temperature —
  **ignore it**; the iter-9 harness addendum at the top of each spec is the override.
- **REUSE already-merged interfaces (import, do not rebuild):** `from vena.segmentation.targets import
  make_soft_targets` (SDT→sigmoid soft `[TC,NETC]`, task 12); `from vena.segmentation.derivation import
  pool_to_latent, ensemble_soft` (task 16); `models/registry.py` decorator + lookup (task 10); `SegmentationConfig`
  + sub-configs (task 10).
- **BSF checkpoints (task 11) — LOCATED + pinned in `src/external/LINKS.md`.** Arm priority: **UKB-SSL = leak-free
  HEADLINE/PRIMARY** (`…/BrainSegFounder_SSL_UKBiobank/64-gpu-model_bestValRMSE.pt`), **BraTS-SSL = comparator**
  (`…/BrainSegFounder_SSL_BraTS/model_bestValRMSE-fold{0..4}.pt`), **finetuned = NEVER**. **Build Arm C
  (SegResNet-scratch, no ckpt) FIRST** — fastest to green.
- **Every `NN_*.md` has a "🔧 ITER-9 HARNESS ADDENDUM" at the top** carrying the parallel-launch context, reuse
  pointers, corrections, sharper acceptance, and a **definition-of-done**. Point each worker at its addendum first.

## Preflight (before spawning — /orchestrate §1)

```bash
~/.conda/envs/vena/bin/python -m pytest -m "not slow and not gpu" -q \
    --basetemp=/home/mpascual/.pytest-tmp-orchestrator | tail -2   # record baseline count
~/.conda/envs/vena/bin/python -m ruff check src/ routines/ tests/ | tail -2   # ~475 pre-existing, not yours
df -h /                                                                          # must not be near 0
```
Confirm the `segmentation` pytest marker is registered (task 10). Check nothing is mid-merge on `main`.

## Task structure — { 11 ∥ 13 ∥ 14 ∥ 15 }, fresh worktrees, merge SERIALLY

Spawn up to 4 Opus workers in **fresh worktrees** (`isolation: "worktree"`, `run_in_background: true`) in ONE message,
disjoint lanes (one subfolder each). Merge **serially**, re-running the full fast suite after each. Give each worker
**exactly**: its `NN_*.md`, `01_SHARED_CONTRACTS.md`, the shared brief above, its lane, its do-NOT-touch list.

### Per-worker inline briefs (hand out verbatim)

- **11 (`models/`):** Build **Arm C `segresnet` FIRST** — fork `src/vena/validation/downstream_seg.py` **by copy**
  (do NOT edit the original), drop the T1c input → `in_channels=3, out_channels=2`, no ckpt → green immediately.
  Then **Arm B `bsf_swinunetr_ukb`** (UKB-SSL, primary — stem may not transfer → list it in `skipped`, don't force)
  and **Arm A `bsf_swinunetr_brats`** (comparator). `load_bsf_encoder → LoadReport(matched,total,skipped)` + log the
  ckpt SHA-256. **Deliver:** all 3 arms forward `(2,3,32,32,24)→(2,2,32,32,24)`; the per-arm matched/total; `STATUS:
  BLOCKED` only if a *pinned* ckpt path is missing (it isn't — they're in LINKS.md).
- **13 (`engine/loss.py`):** DML+CE, **implement DML explicitly** (MONAI Dice is improper on soft labels — assert it).
  **`DML(hard) == 1 − softDice(hard)` to rtol=1e-5.** focal-CE is now the primary training-time calibration lever
  (post-hoc TS dropped) — keep `focal_gamma` selectable. Use `make_soft_targets` in the real-mask test. **Deliver:**
  the DML-equivalence residual, grad-at-optimum, Tversky FN/FP asymmetry.
- **14 (`data/`):** Deterministic patient-level **K-fold OOF ⊆ FM-train** + z-score-on-brain dataset + augmentation.
  **Load-bearing leakage:** no FM-val/test id — **AND no cross-cohort-dedup duplicate of one** — in any fold; read the
  FM dedup source (`[[project_cohort_dedup]]` / corpus registry) and assert BOTH. **Deliver:** fold sizes per cohort,
  the transitive-dedup leakage-assert result, z-score brain stats (mean≈0/std≈1 over nonzero), measured dropout rate.
- **15 (`metrics/`):** Dice/AHD + ECE/Brier (**measured, on RAW soft probs, never thresholded**) + **G-SEG gate**
  (TC≥0.75 provisional, NETC≥0.50 per cohort incl. Ring B; healthy → ~empty TC volume) + dual DSC/Brier selection +
  the **`ET=TC−NETC` diagnostic** (reported, NOT gated). **Deliver:** the reference metric values, the G-SEG
  pass/fail example, the ET-diagnostic numbers.

### The prompt each worker gets (per /orchestrate §2 — deliverable, not instruction)

Demand **verbatim**: the test output / artifact read back (not reconstructed), the specific named numbers above, the
branch SHA, the **import-isolation proof** (`01_SHARED_CONTRACTS.md` §Import isolation — the split-brain trap), and
`STATUS: DONE | QUESTION | PREMISE-FALSE | BLOCKED`. Close with: *"If a premise here is contradicted by the code,
stop and report it with evidence — that is a success, not a failure."*

## Verify EVERY worker — ALWAYS, no exceptions (/orchestrate §3)

Before merging each returned worker: (1) the tests it claims **actually run + pass under the import-isolation
invocation** (`cd $WT && PYTHONPATH=$WT/src ~/.conda/envs/vena/bin/python -m pytest …` — a naive pytest loads half
its code from `main`, silently); (2) the named numbers **reconcile when you re-run them yourself**, never transcribed;
(3) test count strictly up, no test deleted/skipped to pass; (4) ruff clean on **touched files only**. Blame the
environment before the agent (`df -h /` first). **Watch each worker live; steer with `SendMessage` on a bad
intermediate signal instead of waiting for the end; wait silently otherwise. Two correction rounds max, then
escalate.** You own every merge.

## Exit criteria (SESSIONS §S4) — tick the row only when ALL hold

All 3 arms forward `(B,3,·)→(B,2,·)`; BSF load-coverage reported; `DML==softDice-on-hard` green; K-fold plan
deterministic + leakage-free (incl. cross-cohort dedup); metrics + G-SEG gate + dual selection + ET-diagnostic tested;
fast suite green with the count **strictly up** from baseline; ruff clean on touched files; new `segmentation` tests
counted. Then tick the S4 row in `SESSIONS.md` and append its **Orchestrator notes** (what closed, what didn't and
why, premises refuted, baseline numbers, and what S5 must know). S5 (K+1 training on Picasso, concurrent with the S2
oracle) starts only after S4 is merged green.
