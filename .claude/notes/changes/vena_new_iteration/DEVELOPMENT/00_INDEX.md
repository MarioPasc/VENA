# DEVELOPMENT — segmentation submodule + mask-injection task graph

> **What this is.** The implementation task graph for the iter-8 design
> (`../segmenter_conditioning_design.md`). Each `NN_*.md` file is a **self-contained spec** a coding agent
> executes cold via the `/orchestrate` skill. Two independent tracks: **SEG** (`src/vena/segmentation/`, the
> new segmenter) and **INJECT** (`src/vena/model/fm/*` + `routines/fm/train/*`, the v3a `[WT,NETC]` ControlNet
> resume). The tracks share **nothing** at code level and can run fully in parallel.
>
> **Authority.** `../segmenter_conditioning_design.md` (design) and `01_SHARED_CONTRACTS.md` (facts) win over
> anything an agent infers. Where a task premise is contradicted by the code, the agent **stops and reports it**
> (that is a success, per `/orchestrate` §2).

## How to consume this with `/orchestrate`

Give each agent **exactly**: its `NN_*.md` path, `01_SHARED_CONTRACTS.md`, its lane, and what it must not touch.
Spawn parallel agents in **worktrees** (`isolation: "worktree"`), honour the split-brain import-isolation proof
(`01_SHARED_CONTRACTS.md` §Import isolation), and **merge serially**. Verify every reported number against the
artifact on disk — a subagent's report is a hypothesis.

## Waves & dependency graph

```
WAVE 0  (1 task, MERGE FIRST — establishes the package + config surface everyone imports)
  10_seg_scaffold_and_config        src/vena/segmentation/ skeleton + Pydantic config + registries + __all__

WAVE 1  (PARALLEL — each owns disjoint files; merge serially)
  ── SEG leaves (own one subfolder each under src/vena/segmentation/) ──
  11_seg_models                     models/     BSF-SwinUNETR (Arm A/B) + SegResNet Arm C          [Phase-2]
  12_seg_soft_targets               targets/    SDT (per-component/geodesic) → sigmoid, harmonise   [Phase-1]
  13_seg_loss                       engine/loss.py   DML + CE (+ focal-CE, Tversky)                 [Phase-2]
  14_seg_data_kfold                 data/       K-fold OOF splits (⊆ FM-train) + dataset + augment  [Phase-2]
  15_seg_metrics                    metrics/    Dice/AHD + ECE/Brier + G-SEG gate + dual selection  [Phase-2]
  16_seg_soft_derivation            derivation/ per-class temperature + avg-pool→latent + ensemble  [Phase-1: pool; P2: temp/ens]
  ── INJECT (own files under model/fm + routines/fm/train; may all touch module.py → merge 20 first) ──
  20_inject_run_wiring_2ch          serve cached soft [WT,NETC] by mask_source + two-spec + T-13 YAML [Phase-1]
  21_inject_region_weighted_cfm     region-weighted CFM, EQUAL {brain,wt} weights default (≡ L1)      [Phase-1]
  22_inject_mask_perturbation       latent mask-perturbation transform — for T-06                     [Phase-2]

WAVE 2  (SEQUENTIAL — integration + routines, after Wave-1 deps merge)
  19_mask_derive_and_cache          routines/segmentation/mask_derive/ — SOURCE-AGNOSTIC derive+cache to latent H5.
                                    GT path (deps 10,12,16) = Phase-1, NO segmenter; predicted (+17) = Phase-2.  [both]
  40_validate_soft_mask_and_injection  VALIDATION — visual QC + latent-embedding viz + injection sanity  [Phase-1 LAUNCH GATE]
  17_seg_engine_train_predict       engine/{train,predict}.py   integrates 11+12+13+14+16               [Phase-2]
  18_seg_train_routine              routines/segmentation/train/  (train-only; mask-write is in 19)      [Phase-2]

WAVE 3  (DEFERRED — NOT the headline; build so the code exists when gated in)
  30_inject_cfg_inference_DEFERRED  CFG-at-inference guidance + noise-level output_scale (FP-gated)      [Phase-3]
```

> **Execution order is in `SESSIONS.md`** (the orchestrator-facing session plan). This INDEX is the *code-dependency*
> graph; `SESSIONS.md` groups these tasks into orchestrator sessions along the **Phase-1 (oracle) → Phase-2
> (segmenter+deployable) → Phase-3 (ablations)** arc. `[Phase-N]` tags above map each task to its phase.

### Dependency edges (explicit)

| Task | Hard deps | Parallel-safe with | Notes |
|---|---|---|---|
| 10 | — | — | **Wave 0. Merge before any Wave-1 SEG task.** |
| 11 | 10 | 12,13,14,15,16, all INJECT | owns `models/` only |
| 12 | 10 | 11,13,14,15,16, all INJECT | owns `targets/` only |
| 13 | 10 | 11,12,14,15,16, all INJECT | owns `engine/loss.py` only |
| 14 | 10 | 11,12,13,15,16, all INJECT | owns `data/` only |
| 15 | 10 | 11,12,13,14,16, all INJECT | owns `metrics/` only |
| 16 | 10 | 11,12,13,14,15, all INJECT | owns `derivation/` only |
| 20 | existing FM code (wiring); **19** for the run's cache | all SEG | **merge before 21/22** (all three may edit `lightning/module.py`) |
| 21 | 20 (soft) | 22, all SEG | owns `controlnet/losses/` + a loss-wiring block in `module.py` |
| 22 | 20 (soft) | 21, all SEG | owns a new augment transform + a perturbation hook |
| 19 | 10,12,16 (GT path); **+17** (predicted path) | 20/21 wiring | source-agnostic derive+cache; **GT path = Phase-1, no segmenter** |
| 17 | 11,12,13,14,16 | — | segmenter engine (train+predict) |
| 18 | 17 (+10,13,14,15) | — | segmenter TRAIN routine (train-only; mask-write is 19) |
| 40 | 19 (+20,21 for injection panel) | — | **Phase-1 launch gate** (visual + latent-embedding + injection sanity) |
| 30 | 20 | — | **deferred**; do not run in the T-13/T-06 headline |

### Minimal critical path to a runnable T-13 oracle experiment (Phase-1, NO segmenter)

`10 → {12 ∥ 16} → 19(source:gt)` derives+caches the SDT-soft GT `[WT,NETC]` into every latent H5, then
`20 → 21` produces `picasso_ref_v1_v3a+cn[WT,NETC]_fft.yaml` (v3a warm-start + fresh 2-ch ControlNet + equal-weight
region CFM, `mask_source: oracle_soft`), then `40` visually validates the masks + injection **before launch**. The
segmenter (11/13/14/15/17/18) is **not on this path** — the injection mechanism is validated on the oracle first.

### Minimal critical path to the deployable T-06 arm (Phase-2, both tracks)

Segmenter: `10 → {11 ∥ 13 ∥ 14 ∥ 15} → 17 → 18` (K+1 models trained). Then `19(source:predicted)` caches
`masks/tumor_latent_pred`, and the T-06 run reuses `20` with `mask_source: predicted` + `22` (mask perturbation).
Because 19 is **source-agnostic**, the oracle→deployable swap is a one-line `mask_source` change — the run code is
byte-identical.

## Global acceptance criteria (the batch is done when)

1. `~/.conda/envs/vena/bin/python -m pytest -m "not slow and not gpu" -q` — **green**, test count **strictly up**
   from the recorded baseline (no test deleted or skipped to pass). New `segmentation` marker registered.
2. `ruff check` **clean on every touched file** (repo is not globally clean — see `/orchestrate` §1).
3. `src/vena/segmentation/` importable; `from vena.segmentation import SegmentationConfig, get_segmentation_model`
   resolves; the segmenter training smoke runs on a 4-patient synthetic fixture in < 5 min.
4. The T-13 oracle run YAML config-validates and builds the model (2-ch `[WT,NETC]` ControlNet, `conv_in`
   `in_channels` correct) in a loginexa/CPU smoke; a 2-step train loop runs.
5. Every new H5 producer passes its `assert_*_valid`; every routine writes `decision.json` with a bumped
   `schema_version`; the mask cache is `(2,48,56,48)` float32.
6. Each task's own **Tests** section passes with the stated known-property assertions.

## Task-file template (every `NN_*.md` follows it)

`Objective · Track/Wave/Deps · Read-and-verify-first · Files to create/modify · Interface & contract ·
Implementation notes · Acceptance criteria (numbered, checkable) · Tests (property → assertion) · Do NOT touch ·
Report format (artifact path read back, named numbers, import-isolation proof, STATUS)`.
