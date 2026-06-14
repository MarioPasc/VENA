---
name: integrate-competitor
description: |
  Onboard a new competitor model for VENA benchmarking. Walks the 7-step recipe —
  vendor → no-augment wrapper → routine → server-3/loginexa/Picasso platforms →
  inference → docs. Anchored on the pGAN-cGAN (Dar et al., 2019) reference
  integration. Triggers on "add competitor X", "integrate baseline Y", "vendor
  Z model", "benchmark against ...", or any request to add a new image-domain or
  latent-domain synthesis competitor to VENA.
when_to_use: |
  Use whenever the user proposes adding a new published competitor model
  (GAN, diffusion, flow-matching, U-Net baseline) to be evaluated on VENA's
  UCSF-PDGM data. Do NOT use for VENA's own model variants.
---

# Integrate a competitor model into VENA

The reference integration is **pGAN-cGAN (Dar et al., 2019, IEEE TMI 38(10):2375–2388,
DOI 10.1109/TMI.2019.2901750)**. Read its validation note for the worked example:

- `.claude/notes/validation/pgan_cgan.md`
- `src/external/pgan_cgan/{UPSTREAM.md, PATCHES.md}`
- `src/vena/competitors/pgan_cgan/`
- `routines/competitors/pgan_cgan/`

Every new competitor follows the same 7-step recipe. Do not improvise — match the
existing layout and conventions exactly. If you have to deviate, document the deviation
in the validation note.

## The 7 steps

### Step 1 — Vendor upstream

```
src/external/<name>/
├── upstream/                 # cloned snapshot of the repo
├── UPSTREAM.md               # repo URL, vendored SHA, date, licence, scope
├── UPSTREAM_SHA.txt          # short SHA (commit-hash) — engine reads this for decision.json
└── PATCHES.md                # exhaustive list of in-place patches applied
```

- `git clone --depth 1 <upstream> upstream/` then `git rev-parse HEAD > UPSTREAM_SHA.txt`
  then `rm -rf upstream/.git`. The upstream becomes a frozen snapshot, not a live submodule.
- Strip `.pyc` and `__pycache__/` directories from the clone.
- Confirm the licence permits vendoring; record it in `UPSTREAM.md`. MIT, BSD, Apache-2.0
  are unconditional. GPL requires the integration to remain GPL — flag and ask the user.

### Step 2 — Apply torch-2.x compatibility patches in place

Old codebases routinely break on torch 2.x. Apply patches under `src/external/<name>/upstream/`
directly; never monkey-patch at runtime. Document every change in `PATCHES.md`.

Common patches you'll need:

- `cuda(..., async=True)` → `non_blocking=True` (`async` reserved in py3.7+).
- `Variable(...)` removed; `volatile=True` → `torch.no_grad()`.
- `tensor.data[0]` → `tensor.item()`.
- `init.normal`/`init.constant`/`init.xavier_normal`/etc. → in-place `_` variants.
- py3 integer division for slice indexing (`/2` → `//2`).
- `range(...)` → `list(range(...))` when fed to `random.shuffle`.

The integration test of correctness is "model still trains and loss descends" on a
4-epoch smoke. If it doesn't, you've broken numerics — bisect the patches.

### Step 3 — Build the no-augmentation wrapper library

```
src/vena/competitors/<name>/
├── __init__.py               # re-exports the public API: Dataset, train_*, run_inference
├── dataset.py                # torch.utils.data.Dataset reading VENA data directly
├── runner.py                 # imports vendored model, drives training, writes CSVs/checkpoints
└── inference.py              # loads trained weights, synthesises 3D volumes for N patients
```

Hard constraints:

1. **Deterministic dataset.** No augmentation. Repeat reads of the same index return
   byte-identical tensors. Pin this with a unit test
   (`test_<name>_dataset_is_deterministic`).
2. **Match the channel dim contract of the competitor.** pGAN took N input channels →
   1 output. If the competitor expects different shapes, adapt with pad/crop in the
   dataset, not in the model.
3. **Normalise via `vena.common.percentile_normalise`.** Per-patient thresholds, cached
   at init time, applied per-slice — this matches what VENA's own FM trainer sees.
   Image-domain models then rescale `[0, 1] → [-1, 1]` for tanh outputs (pGAN); latent
   models skip this and read from `*_latents.h5` instead.
4. **No `vena.model.fm.*` imports in the wrapper.** Competitor wrappers depend only on
   `vena.common` and `vena.data.*`.
5. **Open H5 lazily in workers.** Use a `self._h5 = None` field and a `_open()` method —
   h5py file handles are not picklable across `num_workers > 0`. Override `__getstate__`
   to drop the handle.

### Step 4 — Build the routine

```
routines/competitors/<name>/
├── __init__.py               # exports <Name>Config + <Name>Engine
├── cli.py                    # vena-competitor-<name> — one positional YAML arg
├── engine.py                 # Pydantic config + engine.run() -> Path
├── infer_cli.py              # vena-competitor-<name>-infer (optional but recommended)
├── configs/
│   ├── smoke_server3_4ep.yaml
│   ├── smoke_loginexa_4ep.yaml
│   └── picasso_full.yaml
├── server3/launcher_<name>_server3_4ep.sh
├── loginexa/launcher_<name>_loginexa_4ep.sh
└── slurm/runs/{launcher,worker}_<name>_picasso_full.sh
```

Engine contract (mirror `routines.competitors.pgan_cgan.engine`):

- Pydantic `<Name>Config` with `from_yaml(path)` classmethod. Frozen, validated.
- `<Name>Engine(cfg, config_yaml_path)` with a single `run() -> Path` method.
- Generate a run_id: `<UTC>_competitor_<name>_<tag>_<short-sha>`.
- Write a preliminary `decision.json` (schema 1.0, `completed=false`) BEFORE training so
  a crashed run is still tracked.
- Persist the resolved config (`config.original.yaml`, `config.resolved.json`).
- Call `train_<name>(runner_cfg, run_dir)`.
- On success, rewrite `decision.json` with `completed=true`.

Register the console script in `pyproject.toml [project.scripts]` exactly as
`vena-competitor-<name> = "routines.competitors.<name>.cli:main"` and the inference
sibling as `vena-competitor-<name>-infer`.

### Step 5 — Per-platform launchers

The three platforms have different transports. Match the templates exactly.

**Server-3 (ICAI workstation, RTX 4090, no SLURM).**
- Pattern: rsync repo → ssh → `screen -dmS <session>` → return.
- Use `screen` (tmux not installed). Session anchor: `vena-<name>-smoke`.
- Pre-warm any pretrained-weight cache on icai-server (it has internet).
- GPU 0 is the FM-train target; check which GPU is free with `nvidia-smi --query-gpu=memory.free --format=csv,noheader`.
- Conda env: `~/.conda/envs/vena/bin/python` on `icai-server`.

**Loginexa (Picasso V100-DGXS-32GB interactive node).**
- Loginexa is NOT a SLURM partition. It is an SSH-accessible interactive node at
  `10.248.7.200`, reachable from Picasso login via `ssh loginexa`.
- Pattern: invoke launcher on Picasso login → pre-warm cache locally (login node has
  internet) → ssh loginexa → `tmux new-session -d`.
- Conda env: **`vena-v100`** (sm_70 / torch 2.7.1+cu126). The prod `vena` env (torch
  2.12+cu130) does NOT work on V100 — it drops sm_70 kernels.
- Picks freest of 4 V100s by `nvidia-smi memory.free`.
- 30-minute wallclock convention (not a hard kill).

**Picasso (A100 DGX, full training).**
- Slurm with `#SBATCH --constraint=dgx --gres=gpu:N --time=Xd-HH:MM:SS --mem=...G
  --cpus-per-task=...`. Omitting `--constraint=dgx` defaults to CPU and fails.
- Launcher pre-warms cache on login node, then `sbatch` the worker.
- Worker pattern: conda discovery boilerplate + `cd $REPO_DIR` +
  `export PYTHONPATH=$REPO_DIR/src:$REPO_DIR:$PYTHONPATH` + python invocation.
- Conda env: `/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/vena/bin/python`.

### Step 6 — Validate the integration

Acceptance is THREE platforms, in this order:

1. **Server-3 4-epoch smoke** — loss descends, checkpoints + metrics CSV + decision.json
   written, `pGAN-train completed`-style sentinel in the log. Wallclock ≤ 5 min on RTX
   4090.
2. **Loginexa 4-epoch smoke** — same artifact set, wallclock ≤ 30 min on V100.
3. **Picasso full submission** — `bash <launcher> --dry-run` resolves the sbatch
   command with correct paths. Do NOT submit the full run as part of the integration
   PR; the user controls when to spend the A100 budget.

After each training run that produced checkpoints, **run inference on 10 val patients**
(or test if explicitly asked):

```
vena-competitor-<name>-infer \
    --run-dir   <run_dir> \
    --image-h5  <UCSFPDGM_image.h5> \
    --epoch     latest \
    --n-patients 10 \
    --phase     val
```

Verify outputs: 10 NIfTI volumes + PNG midslices + `metrics.csv` + `summary.json`.

### Step 7 — Document

Two artifacts MUST be produced:

1. `.claude/notes/validation/<name>.md` — implementation log with file:line references,
   patch summary, data contract, training contract, per-platform recipe, and any
   gotchas. The pGAN-cGAN note is the template.
2. Update `src/external/<name>/UPSTREAM.md` and `PATCHES.md` if anything changed during
   validation.

## Dependency installs you may need

Each new competitor potentially pulls in a new top-level dep. Declare it in
`pyproject.toml [project.dependencies]` and install in the same change. Don't `pip
install` ad-hoc on the cluster envs — they'll drift.

For each env that needs the new dep, install with the cuda-tag matching that env's torch:

```bash
# server-3 (torch 2.5.1+cu121):
ssh icai-server "~/.conda/envs/vena/bin/pip install <pkg> --extra-index-url https://download.pytorch.org/whl/cu121"

# Picasso vena-v100 (torch 2.7.1+cu126, V100 sm_70):
ssh picasso "/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/vena-v100/bin/pip install <pkg> --extra-index-url https://download.pytorch.org/whl/cu126"

# Picasso vena (torch 2.12.0+cu130, A100 sm_80):
ssh picasso "/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/vena/bin/pip install <pkg> --extra-index-url https://download.pytorch.org/whl/cu130"
```

## Anti-patterns (do not do)

1. **Re-introducing augmentation inside the wrapper.** Even "harmless" random flips
   contaminate the comparison.
2. **Monkey-patching the upstream at import time.** Patch in-place; document.
3. **Sharing `vena.model.fm.*` internals.** Competitor wrappers must be self-contained;
   they import only `vena.common` and `vena.data.*`.
4. **Inventing a new run_id format.** Always `<UTC>_competitor_<name>_<tag>_<sha>`.
5. **Hardcoding paths inside the engine.** Paths live in YAML configs; the engine reads
   the config.
6. **Skipping the dry-run step on Picasso.** The sbatch flag drift (e.g. partition
   renames) silently bites you.
7. **Modifying `src/external/<other-competitor>/` while integrating yours.** Each
   competitor's vendored snapshot is independent.
8. **Submitting the full Picasso run without the user's go-ahead.** Acceptance is the
   dry-run; the user gates the actual A100 spend.

## Mental checklist

- [ ] Upstream cloned, SHA recorded, `.pyc`/`.git` removed.
- [ ] `PATCHES.md` lists every in-place change.
- [ ] Wrapper `dataset.py` returns deterministic tensors with no augmentation.
- [ ] Unit test pins determinism + correct shape/range.
- [ ] Engine writes `decision.json` v1.0 with the competitor block.
- [ ] Three platform launchers exist and dry-run cleanly.
- [ ] Console scripts registered in `pyproject.toml`.
- [ ] Server-3 4-epoch smoke completed; artifacts verified.
- [ ] Loginexa 4-epoch smoke completed; artifacts verified.
- [ ] Picasso launcher dry-run validated.
- [ ] Inference run on 10 val patients after each training run.
- [ ] Validation note in `.claude/notes/validation/<name>.md`.
