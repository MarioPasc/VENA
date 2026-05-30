# Routines & Pre-Flight Pattern

A *routine* is a runnable task (run a pre-flight check, train a baseline, evaluate a model, …) that wraps a configurable engine. All routines live under `routines/` and follow the same layout.

```
routines/
├── preflights/                 # pre-flight gates (must pass before Phase-3 training)
│   ├── vessel_mask/            # vesselness method QC (Frangi/Jerman/nnU-Net)
│   ├── maisi_vae/              # MAISI-V2 VAE audit on UCSF-PDGM modalities
│   └── shortcut_diag/          # §6.5 healthy-control diagnostic feasibility
├── data/                       # Phase 1 — data assembly (UCSF-PDGM H5, Málaga manifest)
├── pipeline/                   # Phase 2 — preprocessing, mask extract, MAISI I/O
├── training/                   # Phase 3 — fm_baseline / fm_mask / fm_full / ablations
├── eval/                       # Phase 4 — internal / shortcut / downstream
├── external/                   # Phase 5 — Málaga quantitative + reader study
└── release/                    # Phase 6 — paper figures, public release
```

See `training-stages.md` for the dependency graph between phases.

Each routine has the same internal layout:

```
routines/<bucket>/<name>/
├── __init__.py
├── cli.py                      # entrypoint: `python -m routines.<bucket>.<name>.cli <yaml>`
├── configs/                    # one YAML per concrete invocation
│   ├── default.yaml
│   └── smoke.yaml              # optional, fast local sanity-check
├── slurm/                      # Picasso submission scripts (Singularity, no Docker)
│   ├── launcher_<name>.sh
│   └── worker_<name>.sh
└── engine/
    ├── __init__.py             # re-exports `<Name>Engine` and `<Name>RoutineConfig`
    └── <name>_engine.py        # thin orchestrator that imports library code from src/
```

**Library implementations live in `src/vena/<area>/<name>/`** (e.g. `src/vena/preflight/vessel_mask/{frangi,jerman,nnunet,visualize}.py`). The engine module is a **thin wrapper** that wires library functions to the YAML config, runs them in order, and writes the artifact. This separation keeps library code importable and unit-testable without invoking the CLI.

## Invariants

1. **`cli.py` takes one positional argument**: the path to a YAML config. No other flags. Logging level is read from the YAML.
2. **The engine module exports two public symbols**: a frozen `<Name>RoutineConfig` dataclass (with a `from_yaml(path)` classmethod) and an `<Name>Engine` class with a single `run() -> Path` method that returns the produced artifact path.
3. **Configs are reproducible.** Persist every parameter that influenced the output into the artifact directory: a copy of the resolved YAML, an ISO-8601 timestamp, the git commit SHA, the resolved checkpoint paths, the env name (`vena`).
4. **Validate on close.** If the routine produces an H5, the engine calls `assert_<artifact>_valid(path)` (see `h5-design-principles.md`) before returning. If it produces another artifact type (JSON, figures), assert that every deliverable specified in the routine's `decision.json` schema is present.
5. **Console scripts.** Register each routine in `pyproject.toml` `[project.scripts]` as `vena-<bucket>-<name>` (e.g. `vena-preflight-maisi-vae = "routines.preflights.maisi_vae.cli:main"`) so both forms work:
   - `vena-preflight-maisi-vae cfg.yaml`
   - `python -m routines.preflights.maisi_vae.cli cfg.yaml`
6. **No heavy work at import time.** `cli.py` and `engine/__init__.py` must not load checkpoints, instantiate models, or call `cuda` at module scope. All side effects live inside `Engine.run()`.
7. **One routine, one responsibility.** Split modes into separate routines that share a library module — never add a multi-mode flag.

## Pre-flight specifics

Pre-flight schemas are defined in this rule (no separate `docs/checks/` directory exists for VENA yet; the proposal — `/media/mpascual/Sandisk2TB/research/vena/docs/proposal.md` — is the source of truth). **When `docs/checks/` lands, it supersedes the schemas below**; until then, this file pins the contract between pre-flights and downstream routines.

Dependency order: **`maisi_vae` (independent) ∥ `vessel_mask` (independent) ∥ `shortcut_diag` (independent) → Phase-3 training**.

Each pre-flight routine writes its deliverables to:

```
artifacts/<routine>/<UTC-timestamp>/
├── report.md           # human-readable, includes inlined figures
├── figures/            # PNGs / PDFs referenced by report.md
├── tables/             # CSVs of raw numbers
└── decision.json       # MACHINE-READABLE contract for downstream consumers
```

`decision.json` is the **machine-readable contract** consumed by downstream training routines. Initial schemas (subject to refinement once the corresponding `docs/checks/*.md` spec is written):

### `preflights/maisi_vae` (proposal §3.4)
```json
{
  "schema_version": "1.0",
  "vae_fine_tune": false,
  "fine_tune_target": "none|brain|swan",
  "latent_aug_safe": ["bias_field", "gamma", "intensity_shift"],
  "latent_scale": [s0, s1, s2, s3],
  "recon_psnr_db": {"t1pre": 33.1, "t1c": 32.7, "t2": 30.0, "flair": 30.2, "swan": 22.5},
  "swan_ood_flag": false,
  "checkpoint_path": "/abs/path/to/autoencoder_v2.pt",
  "checkpoint_sha256": "<sha256>"
}
```
Drives whether to encode SWAN through MAISI (`swan_ood_flag == false`) or only via the binary mask (default architecture, per proposal §8 risk row).

### `preflights/vessel_mask` (proposal §3.2)
```json
{
  "schema_version": "1.0",
  "vesselness_method": "frangi|jerman|nnunet",
  "sigma_range_mm": [0.5, 2.5],
  "soft_mask_threshold": 0.15,
  "dice_vs_hand_label": 0.71,
  "ahd_mm_vs_hand_label": 1.8,
  "n_hand_label_cases": 20,
  "passes_cmbb_rejection": true
}
```
Selects the vesselness operator used by `routines/pipeline/mask_extract`. `passes_cmbb_rejection == false` is a hard fail — the model would learn shortcuts from CMBs / iron-rich GM (proposal §6.5 risk).

### `preflights/shortcut_diag` (proposal §6.5)
```json
{
  "schema_version": "1.0",
  "protocol_feasible": true,
  "control_cohort_path": "/abs/path/to/healthy/control/h5",
  "n_controls": 12,
  "ground_truth_enhancement": "none",
  "evaluation_metric": "false_positive_enhancement_volume_ml"
}
```
Drives whether the healthy-control diagnostic in `routines/eval/shortcut` can be executed; `protocol_feasible == false` parks the diagnostic until a control cohort is sourced.

A downstream consumer never reads `report.md` programmatically. It loads `decision.json`, asserts `schema_version`, and uses the keys. `report.md` exists for the human reviewer.

## Hard rules

- **Pre-flights are gating, enforced at engine startup.** A Phase-3 (and later) routine calls a `_assert_preflight_gates(cfg)` helper at the top of `Engine.run()`, *before* any side effect. The helper loads each pre-flight's `decision.json` and raises a routine-specific `PreflightGateError` (e.g. `routines/fm/train/exceptions.py::PreflightGateError`) on any unmet condition. The canonical reference is `routines.fm.train.engine._assert_preflight_gates`: it requires `data.preflight_decision_path` when `data.augmentation_config_path` is set and checks that every requested augmentation appears in `decision.json["latent_safe_augmentations"]`.
- **Pre-flight outputs are immutable once written.** A re-run produces a new timestamped directory under `artifacts/<routine>/`. Never overwrite.
- **Latest pointer.** `artifacts/<routine>/LATEST` is a symlink to the most-recent timestamped directory. Consumers default to following the symlink and can be pinned to a specific timestamp via the YAML config (`preflight_artifact_path: artifacts/preflights/maisi_vae/2026-05-20T14-32-00Z/`).

## `decision.json` for training routines

Phase-3 routines also emit `decision.json` so external-validation and reader-study routines can verify exactly which weights and gates produced a given run. The canonical schema (`routines.fm.train` v0.3.0) is:

```json
{
  "schema_version": "0.3.0",
  "produced_at": "<ISO-8601-UTC>",
  "producer": "routines.fm.train:0.3.0",
  "run_id": "<UTC>_<stage>_<short-sha>",
  "run_dir": "/abs/path/to/experiments/<run_id>",
  "stage": "s1|s2|s3",
  "seed": 1337,
  "corpus_registry": "routines/fm/train/configs/corpus/corpus_<host>.json",
  "cohorts_used": ["UCSF-PDGM", "BraTS-GLI"],
  "trunk_checkpoint": "/abs/path/to/diff_unet_3d_rflow-mr.pt",
  "trunk_checkpoint_sha256": "<sha256>",
  "trunk_trainable": true,
  "vae_checkpoint": "/abs/path/to/autoencoder_v2.pt",
  "vae_checkpoint_sha256": "<sha256>",
  "loss_stage": "s1",
  "ema_decay": 0.9999,
  "augmentation_config_path": "routines/fm/train/configs/augmentations/<name>.yaml",
  "augmentation_preflight_path": "/abs/path/to/.../decision.json",
  "exhaustive_val_enabled": true
}
```

Bump `schema_version` on any breaking change. Add fields freely; never repurpose an existing key.

## Reference

The first routine to land — likely `routines/preflights/maisi_vae/` or `routines/data/ucsf_pdgm_h5/` — becomes the canonical example of this pattern. New routines copy its layout. The current canonical examples are `routines/fm/train/` (engine + preflight gate + decision payload) and `routines/preflights/latent_aug_equivariance/` (preflight that emits `latent_safe_augmentations`).
