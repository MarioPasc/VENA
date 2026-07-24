# 18 — Routine: segmenter training

**Track/Wave/Deps.** SEG · **Wave 2 (sequential)** · deps: 17 (+ 10, 13, 14, 15). Owns
`routines/segmentation/train/`. (The mask→latent-H5 write moved to **task 19**, which is source-agnostic and runs
GT-first without the segmenter.)

## 🔧 ITER-9 HARNESS ADDENDUM (2026-07-23)

**Parallel-launch.** SEG Wave-2, after task 17; **no oracle dependency**.
**🔴 TEMPERATURE DROPPED (Q5).** Remove `temperatures.json` from the artifacts and `T_TC`/`T_NETC` from `decision.json`.
`decision.json` (segmenter schema — its own `schema_version`) carries: backbone arm, `fold`, ckpt SHA-256, `k_folds`,
seed, corpus registry, `selection_metric`, and the per-cohort {TC,NETC} **Dice/AHD/ECE/Brier (incl. Ring B)** + the
**ET=TC−NETC diagnostic** (reported). Calibration measured, not corrected — **no temperature fields**.
**Reuse:** the routine pattern (`preflight-pattern.md`: one positional YAML arg, frozen Pydantic config `from_yaml`,
`Engine.run()->Path`, no import-time side effects, `vena-segmentation-train` console script); `SegTrainer` (17);
G-SEG (15); `routines/fm/train/` as the `decision.json` / CLI idiom reference.

**Sharper acceptance (all must hold):**
1. `RoutineConfig.from_yaml` round-trips; `cli.py` takes exactly one positional arg; importing the engine triggers no
   CUDA and no checkpoint load.
2. `smoke.yaml` trains fold 0 on a 4-patient synthetic subset in **< 5 min** → checkpoint + `decision.json` +
   `fold_plan.json` (**NO `temperatures.json`**).
3. `decision.json` carries the G-SEG table + ET diagnostic + fold + ckpt SHA + a `schema_version`; **no temperature
   keys**; `fold` is a config field so a Picasso array trains the K+1 models as separate tasks.

**Definition of done:** all 3 green, `decision.json` has no temperature fields, ruff-clean, console script registered.

## Objective
A thin routine (`preflight-pattern.md`) that trains **one** segmenter model (a fold, or the `all_train` model) from
a YAML, writing a checkpoint + fitted per-class temperatures + a `decision.json` with the per-cohort G-SEG report.
Design authority: Part B.a/B.b, B.f-§7.

## Read and verify first
- `01_SHARED_CONTRACTS.md` (routine pattern; H5; splits); `.claude/rules/preflight-pattern.md`.
- Task 17 (`SegTrainer`), task 15 (metrics/G-SEG), task 10 (`SegmentationConfig`).
- An existing routine for the `cli.py` + engine + `decision.json` idiom (`routines/fm/train/`).

## Files to create
```
routines/segmentation/train/{__init__.py,cli.py,configs/{default.yaml,smoke.yaml},engine/{__init__.py,train_engine.py}}
```
Modify: `pyproject.toml` (console script `vena-segmentation-train`).

## Interface & contract
- `vena-segmentation-train <yaml>` → `SegmentationConfig.from_yaml` → `SegTrainer(cfg, fold).fit()` → writes
  `experiments/segmentation/<run_id>/` with `checkpoints/`, `logs/train.log`, `metrics/*.csv`, `temperatures.json`,
  `fold_plan.json`, and `decision.json`.
- **`decision.json`** (segmenter schema — its own `schema_version`, not the FM one): backbone arm, `fold`,
  ckpt SHA-256, `k_folds`, per-cohort {TC,NETC} Dice/AHD/ECE/Brier (incl. Ring B), `T_TC`/`T_NETC`,
  `selection_metric`, seed, corpus registry.
- `cli.py` one positional arg; no heavy work at import; `Engine.run() -> Path`.
- **SLURM-friendly**: `fold` is a config field so a Picasso array trains the K+1 models as separate tasks.

## Acceptance criteria
1. `RoutineConfig.from_yaml` round-trips; `cli.py` takes exactly one positional arg; import has no side effects.
2. `smoke.yaml` trains fold 0 on a 4-patient synthetic subset in **< 5 min**, producing a checkpoint +
   `decision.json` + `temperatures.json`.
3. `decision.json` carries the G-SEG table + temperatures + fold + ckpt SHA; `schema_version` present.

## Tests (`tests/routines/segmentation/test_train_routine.py`; `pytestmark = pytest.mark.segmentation`)
- **config/CLI contract**: `from_yaml`; one positional arg; import-time side-effect-free (import engine → no CUDA).
- **decision.json schema**: after a stubbed/short run, assert all required keys present + types.
- **smoke** (`slow`): 4-patient synthetic end-to-end < 5 min → checkpoint + artifacts exist (readback).

## Do NOT touch
INJECT-track files; the FM `decision.json` schema; task-19's derive/cache routine; real cohort data in tests.

## Report format
Readback run-dir, the decision.json keys, the smoke timing, import-isolation proof, ruff-clean, `STATUS`.
