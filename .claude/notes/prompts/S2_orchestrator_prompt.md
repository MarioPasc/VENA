# S2 orchestrator launch prompt — Injection wiring + region-weighting + launch the oracle matrix

Paste everything below the line into a fresh session (Opus, `xhigh`). It is self-contained.

---

You are the **orchestrator** for session **S2 — Injection wiring + region-weighting + launch oracle**
(Phase 1b) of the VENA iter-9 plan. Invoke the **`/orchestrate`** skill first and follow it: you hold
the plan, subagents write the code, **you verify every number they report against the artifact on
disk**. A worker's report is a hypothesis, never evidence.

## Read these first, in this order

1. `.claude/notes/changes/vena_new_iteration/DEVELOPMENT/SESSIONS.md` — **read the whole S1 section's
   "Orchestrator notes"**, not just the S2 row. S1 established facts you must not re-derive and traps
   that will bite you again. Then read the S2 section: its *Sequence*, *Gates*, and *Exit criteria*
   **override any plan you would invent**.
2. `.claude/notes/changes/vena_new_iteration/DEVELOPMENT/01_SHARED_CONTRACTS.md` — the fact sheet.
   Hand it to **every** worker alongside its task spec.
3. Your two task specs: `20_inject_run_wiring_2ch.md` and `21_inject_region_weighted_cfm.md`, plus
   `40_validate_soft_mask_and_injection.md` for the injection-sanity panel.
4. `scripts/mask_audit/README.md` — only if you need to re-check mask correctness (you should not; see
   below).

## State of the world — what is DONE and what is NOT

- **S1 — Oracle soft-mask: mechanically COMPLETE.** `masks/tumor_latent_soft (N,2,48,56,48)` float32,
  schema `2.1.0`, `tumor_region="tc"`, `mask_source="gt"` is cached in **all 9 cohort latent H5s
  (3,459 scans)** and passes `assert_latent_soft_mask_group_valid` 9/9. The oracle `masks/tumor_latent
  (N,3,…)` is byte-untouched beside it.
  **A full invariant audit (array `1636104`) proved the cache is bit-exact**: re-deriving from GT and
  re-pooling through the canonical `apply_crop_pad` → `avg_pool3d(4)` path reproduces it with
  `recompute_max_abs = 0.0` in every cohort; Dice and volume-ratio are 1.000 everywhere; nesting,
  continuity, mass conservation and agreement with the independent oracle group all hold.
  **Do not re-litigate whether the mask is correct — it is, measured. Consume it.**
  **The human `masks_look_valid` gate was CLOSED on 2026-07-24** (the user reviewed the QC figures and
  confirmed the masks; `masks_look_valid: true` is written into all three QC `decision.json` artifacts).
  **S1 is `☑` and requires nothing further — you are clear to launch GPU jobs.**
- **S4 — Segmenter library: COMPLETE and pushed** (`☑`). Not your concern; it shares no code with the
  injection track.
- **S5 — Segmenter training: IN PROGRESS (`◐`), NOT complete.** Arrays **`1640255` (UKB)** and
  **`1640256` (SegResNet)**, 6 tasks each, are running on Picasso at commit `34a2710`; `gseg_tc_dice
  = 0.75` remains **provisional**. **This matters to you only as GPU contention** — size and schedule
  your 5 oracle jobs knowing ~12 segmenter tasks may be queued/running in the same allocation. Do not
  wait on S5 and do not touch it; the tracks only meet at S6.
- **S3 and S6 are not yours.** Do not start the oracle analysis (S3) or anything predicted-mask (S6).

## Your scope — exactly this, nothing more

Per SESSIONS §S2:

1. **Fan out 2 workers in fresh worktrees:** task **20** ∥ task **21**.
   **Merge 20 BEFORE 21** — both may touch `lightning/module.py`.
   - **20** — serve the cached mask via `data.mask_source: oracle_soft`; **two 1-channel conditioning
     specs**; the v3a-warm-start T-13 run YAML; a loginexa smoke.
   - **21** — region-weighted CFM over regions `{Brain = NOT-BG ∩ NOT-TC, TC}`, with a test proving
     **equal weights are numerically identical to the current unweighted L1** velocity loss.
2. **Task 40** — the injection-sanity panel: **step-0 identity** (`output_scale = 0` ⇒ residual 0 ⇒
   output byte-identical to v3a) and **residual locality** (in-TC vs out-of-TC residual energy).
   Run this **before** spending GPU-days.
3. `[O]` **loginexa smoke** (`test-picasso-loginexa` skill) — the T-13 YAML builds the 2-ch ControlNet
   (`conditioning_in_channels == 2`) and runs 2 optimiser steps.
4. `[O]` **Launch the 5-job oracle matrix** (`picasso-sbatch`), all v3a-warm-start + fresh 2-ch
   ControlNet on `data.mask_source: oracle_soft`:

   | job | trunk | region_weights | purpose |
   |---|---|---|---|
   | **J0** | **freeze** | `{brain:1, tc:1}` | ControlNet-only **lower bound** (no trunk drift, no loss help) |
   | **J1** | joint-low-LR | `{brain:1, tc:1}` | equal-weight ceiling; **J0→J1 = the freeze→joint gain** |
   | **J2** | joint-low-LR | `{brain:1, tc:5}` | RW sweep |
   | **J3** | joint-low-LR | `{brain:1, tc:10}` | RW sweep |
   | **J4** | joint-low-LR | `{brain:1, tc:20}` | RW sweep (**watch FP-safety** at the top weight) |

   Only **J0 is freeze** (needs no trunk EMA); **J1–J4 are joint** and require v3a's
   `trunk_ema_snapshot.pt`. LR = linear warmup → cosine (v3a's `warmup_steps: 1000`, `scheduler: cosine`).
5. `[O]` **Monitor** — one persistent watch matching **every terminal state AND failure states**, not
   just the happy path.

**Out of scope:** S3 analysis, the segmenter, predicted masks, CFG/SPADE/WT-weight ablations (S7),
and any "improvement" to the mask derivation. If you believe scope must change, say so and stop —
do not silently widen it.

## Load-bearing facts — get these wrong and you burn GPU-days

1. **Channel 0 = TC (tumour core = NETC+ET), NOT WT.** 81% of WT is non-enhancing edema, which is what
   drove the original tumour-skip failure. Several specs still say `[WT,NETC]` / `m_wt_soft` — **read
   them as `[TC,NETC]` / `m_tc_soft`**. Region loss weights TC, not WT. Report **PSNR_ET**, never
   PSNR_WT (PSNR_WT is a known metric trap — it averages necrosis+edema+enhancement).
2. **A 2-channel mask MUST be two 1-channel specs**: `mask:tc_soft:identity` + `mask:netc_soft:identity`.
   `ConditioningAssembler.channels_per_spec` uses the **`mask_channels` constructor default (=1)**, not
   the runtime tensor shape — a single 2-ch spec silently under-counts `total_channels` and the hint-net
   first conv is built wrong.
3. **Latent grid is `(48,56,48)`**; `rflow.base_img_size_numel = 129024 = 48×56×48` is **CORRECT** —
   there is no mismatch to "fix". Any `(60,60,40)` / `144000` in an older spec is stale.
4. **`use_timestep_transform: true` requires the sampler to receive `input_img_size_numel`.** Omit it
   and MONAI divides `None / int`, every per-patient exhaustive-val silently fails with a WARNING, and
   you get an **empty `metrics.csv` and no figures**. Verify one early cadence epoch actually wrote
   non-empty per-patient rows — this is an explicit S2 exit criterion.
5. **Raise EarlyStopping patience to ~400–500.** The harder objective (adding enhancement) transiently
   *raises* `train/total_epoch` before it improves; patience 250 risks a premature stop. Keep every
   epoch checkpoint — exhaustive-val PSNR_ET is the real selection signal, not train loss.
6. **Picasso A100 selector:** use **untyped `--gres=gpu:N` + `--constraint=a100`**.
   `--gres=gpu:A100:N` matches **nothing** (A100 nodes advertise untyped `gpu:8`) and a bare
   `--constraint=dgx` silently lands on B200. **The root `CLAUDE.md` is wrong on this** — trust the
   `picasso-sbatch` skill and this line.
7. **Picasso's `sbatch` wrapper emits ANSI colour codes** even with `--parsable`. Strip them, assert the
   job id is numeric, and check `scontrol show job` does not report `Dependency=(null)`. Run
   `sbatch --test-only` before every real submission.
8. **A smoke must exercise the failing path.** S5 lost both K+1 arrays in 90 s because both smokes used
   `batch_size: 1`, which never stacks two samples and so could not reproduce a cross-cohort collate
   crash. Shrink epochs/patients/patch to go fast — **never** shrink a dimension to the value that
   disables the code path (batch→1, cohorts→1, GPUs→1).
9. **⚠ Offline-augmented latent H5s (`*_latents_aug.h5`) do NOT carry `masks/tumor_latent_soft`.** They
   were not in the mask-derive registry. **If your run YAML enables offline augmentation, the mask will
   be missing or inconsistently transformed.** Decide explicitly: either disable offline aug for the
   oracle matrix, or cache+transform the mask there first. Do not discover this at epoch 1.
10. **Known, accepted mask limits (from the audit — do not "fix" these):** a TC below ~1–2 latent voxels
    (≲130 image voxels) is unrepresentable on the `(48,56,48)` grid; **BraTS-PED latent fidelity is
    systematically degraded** (`lat_iou_tc` 0.596 vs 0.82–0.86 for adult glioma cohorts, 19.6% TC-empty)
    and 2 BraTS-PED GT labels are genuinely defective. BraTS-PED is `test_only`, so training is unaffected.

## Canonical Picasso paths & immutability

**v3a warm-start source — READ-ONLY, NEVER ALTER:**

```
/mnt/home/users/tic_163_uma/mpascual/fscratch/checkpoints/vena_project/2026-06-24_16-00-46_s1_v3a_concat_only_fft_ef000c9f/
```

Every T-13 YAML sets `run.resume_from` to the **absolute path of this dir's `ema_best.ckpt`**;
WARM_START is weights-only (fresh optimiser/EMA/RNG). J1–J4 additionally read its
`trunk_ema_snapshot.pt`. **If any tool would write inside this directory, STOP** — that is a
`PREMISE-FALSE`/BLOCKED report, not an action.

**All new run output goes here, nowhere else:**

```
/mnt/home/users/tic_163_uma/mpascual/execs/vena/experiments/<run_id>/
```

Confirm the engine's experiments-root resolves there before launch and that `run_id` does not collide.
Run validation work from `fscratch/repos/VENA-validation` (a real git repo, so `git rev-parse` resolves),
**not** `fscratch/repos/VENA`. Picasso python: `fscratch/conda_envs/vena/bin/python`.

## Gates — check before spawning, and again before the GPU launch

- **Human mask-review gate: ✅ ALREADY CLOSED (2026-07-24) — do not ask the user again.**
  `masks_look_valid: true` in all three QC `decision.json` files under
  `/media/mpascual/Sandisk2TB/research/vena/results/prior/tumor/gt/`
  (`t1c/2026-07-23T08-41-50Z/`, `t2f/2026-07-22T22-11-15Z/`, `2026-07-22T21-53-06Z/`). Nothing in S1
  blocks you; proceed straight through to the GPU launch once your own exit criteria (1)-(3) hold.
- v3a source dir present and read-only; `ema_best.ckpt` for all 5, `trunk_ema_snapshot.pt` for J1–J4.
- Clean tree; record baseline `pytest -m "not slow and not gpu"` and `ruff` counts, and `df -h /`,
  before spawning. The repo is **not** globally ruff-clean (~475 pre-existing) — that is not yours.

## Worker discipline

- Spawn with `Agent(subagent_type: "general-purpose", model: "opus", isolation: "worktree")`, both in
  **one message** so they run concurrently. Give each: its `NN_*.md`, `01_SHARED_CONTRACTS.md`, its
  lane, what it must not touch, and the load-bearing facts above that apply to it.
- **Worktree stale-base trap:** `isolation: "worktree"` cuts from the **session base commit**. Before
  trusting any output, verify `git -C $WT merge-base --is-ancestor <main-sha> HEAD`. After any merge to
  `main`, every other worktree is stale — make the next worker `git merge main` first, with proof.
- **Split-brain import trap:** `vena` is an editable install pinned to the main checkout. Require each
  worker to paste the import-isolation proof from `01_SHARED_CONTRACTS.md` (`PYTHONPATH=$WORKTREE/src`,
  assert both `vena` and `routines` resolve inside the worktree).
- `--basetemp=/home/mpascual/.pytest-tmp-<slug>`, **unique per agent**, never on `/`.
- Phrase work as a **deliverable** ("report the artifact path read back from `readlink LATEST`, and
  these specific numbers, which I will check"), not an instruction. Close every worker prompt with:
  *"Report `STATUS: DONE | QUESTION | PREMISE-FALSE | BLOCKED`. If a premise here is contradicted by the
  code or the data, stop and report it with evidence — that is a successful outcome, not a failure."*
- **You own merges**, serially, re-running the suite after each. Never lower the test count. Two
  correction rounds max, then escalate to the user.

## Exit criteria — tick the S2 row only when ALL hold

1. Tasks 20, 21, 40 merged to `main`, fast suite green (count ≥ baseline), ruff clean on touched files.
2. **Step-0 identity verified**: `output_scale = 0` ⇒ ControlNet residual exactly 0 ⇒ output identical
   to v3a. Re-derive this yourself.
3. **loginexa smoke green**: 2-ch ControlNet builds (`conditioning_in_channels == 2`) and 2 optimiser
   steps run, no shape error.
4. **5 Picasso jobs RUNNING**, job ids recorded (ANSI-stripped, numeric), `Dependency` not `(null)`,
   writing into `…/execs/vena/experiments/<run_id>/`.
5. **Exhaustive-val cadence verified writing non-empty per-patient `metrics.csv`** on one early epoch
   (the `use_timestep_transform` empty-CSV trap).
6. Persistent monitor armed on all terminal **and** failure states.
7. SESSIONS.md S2 "Orchestrator notes" appended (append-only): what closed, what did not and why, any
   premise refuted, real baseline numbers, and what S3 must know. Update any memory that is now wrong.
