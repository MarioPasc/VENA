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

## The pGAN-cGAN integration as a reusable template

Each file under `src/vena/competitors/pgan_cgan/` and
`routines/competitors/pgan_cgan/` is a usable scaffold; copy it into the new
competitor's directory and edit the marked points. The boundaries between
files are stable and were validated end-to-end across all three platforms.

### File-by-file map (treat as the canonical scaffold)

| File | Role | What to keep, what to adapt |
|---|---|---|
| `src/external/<name>/upstream/` | Vendored upstream snapshot, frozen. | Replace; patch torch-API drift in-place; never monkey-patch at runtime. |
| `src/external/<name>/UPSTREAM.md`, `UPSTREAM_SHA.txt`, `PATCHES.md` | Vendor metadata. | Replace verbatim — same fields, new values. |
| `src/vena/competitors/<name>/__init__.py` | Public API re-exports. | Keep shape: `Dataset`, `MultiCohortDataset`, `train_*`, `run_inference`. |
| `src/vena/competitors/<name>/dataset.py` | Per-cohort + multi-cohort H5 → competitor-format batches. | Replace the batch-formatting (`__getitem__` return dict, channel layout, range rescale). Keep the percentile-norm pipeline, the lazy-open + `__getstate__` pattern, the longitudinal id resolver, the flat-splits fallback, and the `MultiCohortImageSliceDataset` shape (cohort loop → skip-with-WARNING → ConcatDataset). |
| `src/vena/competitors/<name>/runner.py` | Programmatic training loop: build options Namespace, import vendored model, drive epoch loop, write CSVs, save `best` + `latest`. | Replace the import + opt builder + the inner step body. Keep the epoch-CSV / step-CSV writers, the `best_loss` / `patience` early-stop block, the `max_epochs` cap, the `_verify_vgg_cache()`-style pretrained-weight check, the sentinel log line. |
| `src/vena/competitors/<name>/inference.py` | Load `best_net_G.pth` (or epoch N), run on N val patients, write NIfTI + PNG + metrics CSV. | Replace the generator build + forward pass. Keep the percentile-norm parity, the crop-back to native shape, the per-patient PSNR/SSIM. |
| `routines/competitors/<name>/cli.py` | One-positional-arg CLI; rich logging; calls Engine. | Copy verbatim — only the module path changes. |
| `routines/competitors/<name>/engine.py` | Pydantic config (`DataCfg`, `HyperParamsCfg`, `RuntimeCfg`), run-id generator, `decision.json` writer, preflight checks, build runner SimpleNamespace, call `train_<name>`. | Replace the hyperparameter fields; keep the multi-cohort / single-cohort either-or validator, the `_preflight` grandparent check, the `_write_decision` schema-1.0 block, and the `_short_git_sha` / `_file_sha256` helpers. |
| `routines/competitors/<name>/infer_cli.py` | CLI wrapper around `run_inference`. | Copy verbatim — only the import path changes. |
| `routines/competitors/<name>/configs/{smoke_server3_4ep,smoke_loginexa_2ep,picasso_full}.yaml` | Per-platform YAML configs. | Copy + adapt the hyperparams + corpus_registry per platform. Smokes use `num_workers: 0` and `max_patients_per_cohort: 1`. Picasso full uses `corpus_picasso.json`, `max_epochs: 10000`, `patience: 100`, `num_workers: 8`, `batchSize: 4`, fold 0, seed 1337 — these are the paired-comparison axes. |
| `routines/competitors/<name>/server3/launcher_<name>_server3_4ep.sh` | rsync → ssh icai-server → `screen -dmS` → exit. | Copy verbatim — change session name + config path. Pre-warm VGG-style caches on icai-server (has internet). |
| `routines/competitors/<name>/loginexa/launcher_<name>_loginexa_2ep.sh` | ssh picasso → warm cache on login node → ssh loginexa → `tmux new-session -d` → exit. | Copy verbatim — change session name + config path + python interpreter pin (`vena-v100` env). Auto-pick freest V100 by `nvidia-smi memory.free`. |
| `routines/competitors/<name>/slurm/runs/{launcher,worker}_<name>_picasso_full.sh` | sbatch launcher + worker with `--constraint=dgx --partition=gpu_partition --gres=gpu:1 --time=7-00:00:00`. | Copy verbatim — change paths + env name (`vena` for A100). |
| `tests/competitors/<name>/test_dataset.py`, `test_inference.py`, `test_multicohort.py` | Synthetic-H5 fixtures + behavioural assertions. | Copy + adapt. The longitudinal-resolver, flat-splits, and missing-cohort tests carry over essentially unchanged. |
| `.claude/notes/validation/<name>.md` | Implementation log, paired-comparison axes table, per-platform recipes, gotchas. | Replace per-competitor body; keep the section structure (Scope / Code layout / Patches / Data contract / Per-platform recipe / Paired axes / Things to watch). |
| `pyproject.toml` `[project.scripts]` | Console scripts `vena-competitor-<name>` + `vena-competitor-<name>-infer`. | Add two lines per competitor. |

### Boundary that must not move

`vena.competitors.<name>` depends on `vena.common.percentile_normalise` and
the vendored upstream — **nothing from `vena.model.fm.*`, `vena.preflight.*`,
or another `vena.competitors.<other>.*`**. The competitor wrappers form a fan
of independent leaves under `vena.competitors`, so deleting one never
breaks the others.

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
2. **Match VENA's training corpus.** Production runs **must** read the same
   `routines/fm/train/configs/corpus/corpus_<host>.json` that VENA's FM trainer reads
   — never hand-roll a competitor-specific corpus, never train on a single cohort when
   VENA trains on the union. Build a `MultiCohort<X>Dataset` that takes a corpus
   registry path, filters by `role=="cv"`, and concatenates per-cohort datasets via
   `torch.utils.data.ConcatDataset`. Skip with WARNING (not error) when a cohort's
   `image_h5` is missing on the current platform or its `splits/cv/fold_<k>/<phase>`
   is empty — this is what lets the same corpus JSON drive heterogeneous environments.
   Keep a single-cohort path on the dataset class for fast sanity smokes.
3. **Match the channel dim contract of the competitor.** pGAN took N input channels →
   1 output. If the competitor expects different shapes, adapt with pad/crop in the
   dataset, not in the model. The padder must handle native shape heterogeneity
   across cohorts (UCSF-PDGM is 240×240, BraTS-GLI is 182×218 — both pad to 256).
4. **Normalise via `vena.common.percentile_normalise`.** Per-patient thresholds, cached
   at init time, applied per-slice — this matches what VENA's own FM trainer sees.
   Image-domain models then rescale `[0, 1] → [-1, 1]` for tanh outputs (pGAN); latent
   models skip this and read from `*_latents.h5` instead.
5. **No `vena.model.fm.*` imports in the wrapper.** Competitor wrappers depend only on
   `vena.common` and `vena.data.*`.
6. **Open H5 lazily in workers.** Use a `self._h5 = None` field and a `_open()` method —
   h5py file handles are not picklable across `num_workers > 0`. Override `__getstate__`
   to drop the handle. **Do NOT pass `swmr=True`** to a non-SWMR-written file; some h5py
   builds deadlock on the no-op handshake under multiprocessing.
7. **Track `best` and `latest` checkpoints, not just `latest`.** Save `best_net_*` on
   every improvement of an epoch-level metric (G_L1 for pGAN, the equivalent for your
   model). `latest` is the resume point; `best` is the evaluation point. The runner
   should accept a `patience` knob (epochs without improvement → early stop) and a
   `max_epochs` cap that mirrors VENA's reference recipe.

### Step 3.3 — One-to-one vs many-to-one input handling

VENA's task is **multi-contrast** (`{T1pre, T2, FLAIR, …}` → `T1c`), but most
published synthesis competitors are **one-to-one** (one source modality → one
target). Before building the wrapper, identify which paradigm the competitor
was trained for. Reading the upstream `--input_nc` flag is not enough; trace
where it lands in the model:

- **`input_nc` = number of source modalities** (channels are modalities) →
  **many-to-one** model. McCaD, CFM (latent), TumorFlow, MM-GAN, MM-pix2pix
  fall here. The wrapper hands the model `(B, M, H, W)` with M = number of
  modalities; the comparison fits VENA's data on a single run.
- **`input_nc` = number of neighbouring 2D slices of one modality** (channels
  are axial neighbours of the same volume) → **one-to-one** model. **pGAN**
  (Dar 2019), Replica, pix2pix-MR, plain CycleGAN are here. The model only
  knows how to translate one modality. Feeding it stacked multi-modal inputs
  is out-of-distribution and **not** what the authors validated.

**Treat one-to-one models as a panel**, never as a single multi-modal run.
For VENA's `{T1pre, T2, FLAIR} → T1c` task that means three independent
training runs (`t1pre→t1c`, `t2→t1c`, `flair→t1c`), each with its own
`picasso_full_<source>.yaml`, run id (`tag: full_<src>_<tgt>`), and inference
output directory. Report results per-source in the paper's competitor table.
Reviewers expect this; mixing modalities in `input_nc` would be a fairness
red flag.

### Step 3.3.5 — Image-domain vs latent-domain competitor

A second axis (orthogonal to one-to-one vs many-to-one) is whether the
competitor operates in **image space** (pGAN, McCaD, Replica, Pix2pix) or
**latent space** (T1C-RFlow, CFM, TumorFlow, latent DDPM). The two paths
have a different data contract that flows through every layer:

| Aspect | Image-domain | Latent-domain |
|---|---|---|
| Dataset reads | `cohort.image_h5` | `cohort.latent_h5` |
| `__getitem__` returns | `(B, C, H, W[, D])` slices/volumes in `[-1, 1]` or `[0, 1]` | `{patient_id, z_<mod>: (C, h, w, d)}` from `latents/<mod>` |
| `image_size`, padding, percentile-norm | live in the dataset | **absent** — latents are pre-shaped by VENA's encoder |
| Training-time augmentation pressure | available (be disciplined: turn it OFF) | available via stochastic `z = μ + σε` per step — paper-faithful for some upstreams, off by default in VENA's stored z (document this deviation) |
| Sampler | model forward only | model forward + scheduler step loop (FM/diffusion) |
| **Inference decode for image-space metrics** | none — model already lives in image space | mandatory — see Step 3.6 |
| `corpus_path_overrides` keyed on | `image_h5` | `latent_h5` |
| Multi-cohort skip-with-WARNING | "image_h5 missing" | "latent_h5 missing" *or* "no `latent_h5` field in registry entry" (some test_only cohorts only carry image_h5) |

The latent-domain reader skeleton is shorter than the image-domain one (no
`_pad_to`, no per-patient percentile cache, no slice index — one item =
one patient). Reuse `T1CRFlowLatentDataset` from
`src/vena/competitors/t1c_rflow/dataset.py` as the canonical template.

### Step 3.4 — Cohort schema heterogeneity (real, hit on pGAN integration)

The VENA corpus is **not** schema-uniform. Two failure modes silently drop
cohorts at run start if the wrapper assumes UCSF-PDGM's schema for all of
them. The pGAN reference integration fixed both inside
`src/vena/competitors/pgan_cgan/dataset.py`; future competitors must
implement the equivalent fallbacks (or import the same dataset class).

1. **Longitudinal cohorts store scan-level `/ids` but patient-level splits.**
   BraTS-GLI: `/ids[i] = "BraTS-GLI-00000-000"` (scan with `-NNN` session
   suffix), `splits/cv/fold_0/train[j] = "BraTS-GLI-00000"` (patient). Same
   pattern in LUMIERE with a `Patient-001__week-...` form. Resolution: when
   an exact `pid in /ids` fails, try prefix-match (`pid + "-"` or
   `pid + "_"`) and concatenate every matching scan. Skipping the prefix
   match drops 815 + 64 ≈ 879 patients (≈170 K slices) silently. The
   `longitudinal: true` field in the corpus registry is the flag that
   tells you to expect this.

2. **Small cohorts use a flat `splits/<phase>` schema, not k-fold.**
   REMBRANDT (N=63) stores its single 53/5/5 train/val/test split at
   `splits/{train,val,test}`; there is no `splits/cv/fold_<k>/...` at all.
   Resolution: prefer the k-fold path, fall back to the flat path. The
   manifest comment ("N=63 too small for nested CV") is the canonical
   warning. Other small cohorts may follow the same convention.

When in doubt, walk the H5 once before integrating a new dataset:

```python
with h5py.File(path, "r") as f:
    print(sorted(f.keys()), sorted(f["splits"].keys()) if "splits" in f else [])
    sample_id = f["ids"][0].decode()
    sample_split = next(iter(f.get("splits/cv/fold_0/train", f.get("splits/train", []))), b"")
    print("ids[0]:", sample_id, "vs split[0]:", sample_split)
```

If the two strings differ structurally, you have the BraTS-GLI/LUMIERE
case. If `splits/cv/fold_0/train` doesn't exist, you have the REMBRANDT
case. Either way the wrapper must handle it before going to production —
both `UCSFPDGMSliceDataset` (per-cohort) and `MultiCohortImageSliceDataset`
should *skip with WARNING*, not raise, when a single cohort fails to
resolve. Pin every fallback with a unit test under
`tests/competitors/<name>/`.

### Step 3.5 — h5py + multiprocessing trap (real, hit on pGAN integration)

h5py file handles + Python's `multiprocessing` start method (fork on Linux) +
a `ConcatDataset` over multiple H5s **deadlocks the DataLoader** when several
worker processes open the same file family concurrently. The symptom: training
loop hangs mid-epoch, CPU pegged at 100 % across workers, no Traceback, log
silent for many minutes. We hit this on the first multi-cohort pGAN smoke
(4 workers × 4 cohorts → 16 concurrent handles → freeze at step ~950).

Mitigation by config:
- **Smokes**: set `num_workers: 0` (the loop runs in the main process).
- **Production**: keep `num_workers: 8` only if the slowdown is a real cost;
  start with `num_workers: 0` and bump only after confirming no deadlocks on a
  full epoch.
- The dataset's `_open()` must open in plain `"r"` mode (no SWMR).

### Step 3.6 — Latent-domain inference primitives (real, hit on T1C-RFlow integration)

When a latent-domain competitor needs to decode predicted latents back to
image space for PSNR/SSIM, the canonical decode pipeline is
**`vena.common.MaisiDecoder(load_autoencoder(ckpt))` + `decode_box`**, plus
`build_crop_spec_from_h5` / `load_real_t1c_box` from
`vena.model.fm.eval.exhaustive` for the real-T1c reference. Three traps
along that path cost ~10 min of ssh round-trip the first time they hit:

1. **`vena.common.load_autoencoder(checkpoint_path, …)` has no default
   checkpoint.** It accepts `(checkpoint_path, device, arch_config, arch_overrides)`
   as positional/keyword. Surface the path in the wrapper's
   `infer_cli.py` as `--vae-checkpoint` (required, with a clear error from
   `run_inference` if absent) and document the per-platform path in the
   validation note. Canonical paths live in `src/external/LINKS.md`:
   - server-3: `/media/hddb/mario/checkpoints/MAISI_V2_RM/NV-Generate-MR/models/autoencoder_v2.pt`
   - Picasso: `/mnt/home/users/tic_163_uma/mpascual/fscratch/checkpoints/NV-Generate-MR/models/autoencoder_v2.pt`

2. **`AutoencoderHandle` is *not* a decoder.** It carries `model`,
   `device`, `checkpoint_path`, `checkpoint_sha256`, `arch_kwargs` — no
   `.decoder` attribute. Wrap it: `vae_decoder = MaisiDecoder(handle)`.
   `MaisiDecoder` accepts sliding-ROI / overlap / mode / precision kwargs;
   the defaults `(20, 20, 8)`, `0.4`, `gaussian`, `autocast` match VENA's
   exhaustive-val.

3. **`vena.common.percentile_normalise` expects a 5-D `(B, C, H, W, D)`
   tensor** when `foreground_only=True`, but `load_real_t1c_box` returns
   3-D `(H, W, D)`. Wrap before, unwrap after:
   `percentile_normalise(real[None, None], lower=0, upper=99.5,
   foreground_only=True)[0, 0]`. Without the wrap you get a cryptic
   `expects (B,C,H,W,D); got shape (192, 224, 192)` and **the patient is
   skipped** — the inference run completes successfully with 0 rows in
   `metrics.csv`, which reads like a silent success. Guard against this
   with a post-condition: `assert summary["n_patients_succeeded"] > 0` in
   the wrapper or the validation watcher.

These three are pre-emptable in the inference.py skeleton; copy
`src/vena/competitors/t1c_rflow/inference.py` verbatim for the next
latent-domain competitor.

### Step 3.7 — Upstream code vs paper text incoherency check (real, hit on T1C-RFlow integration)

Many synthesis papers ship code where the released script does **not**
match what the paper text claims. **VENA policy (locked 2026-06-15):
follow the peer-reviewed paper text. The code is not peer-reviewed.**
When the wrapper diverges from the upstream-as-released, that is by
design — and it is the *code* that is suspect, not the wrapper. The
load-bearing axis to check is whether the code does something more
favourable than the paper text claims, since that is method laundering
even when unintentional.

Open `UPSTREAM.md` with a *Differences between upstream paper text and
upstream code* section. Tabulate every divergence with:

| Column | What goes in it |
|---|---|
| Axis | what is disagreeing (architecture, scheduler kwarg, loss, optimiser, etc.) |
| Paper text | verbatim quote + section number |
| Released code | line reference in `src/external/<name>/upstream/<file>.py` |
| Load-bearing? | YES / NO / MARGINAL — does this affect reported metrics? |
| Direction of advantage | which version gives better numbers, and roughly how much |

Backbone-required overrides (config-flag-bug fixed at runtime, e.g.
T1C-RFlow's `use_discrete_timesteps`) are **not** load-bearing — call them
out anyway. Architectural divergences (channel counts, attention) and
training-data choices (un-announced VAE retraining on test-adjacent
data) **are** load-bearing — those are the ones that change the result.

When following the paper, document the wrapper's architecture choice in
the validation note's paired-axes table and in
`decision.json["competitor"]["deviations"]` with a key like
`unet_architecture: paper_faithful_3level`. The code-version may be
re-introduced later as a **separate ablation row**, never as the headline
number.

Triage protocol — once vendored, before writing the runner:

```bash
# Quick diff between paper claim and code reality.
grep -nE "in_channels|num_channels|attention|num_train_timesteps|use_discrete|sample_method|autocast|GradScaler|lr|weight_decay|EMA|num_res_blocks" \
    src/external/<name>/upstream/train_*.py \
    src/external/<name>/upstream/**/configs/*.json
```

Every line that contradicts a number in the paper text gets a row in the
incoherency table. If the load-bearing column has any YES, the user must
be flagged: the wrapper's defaults follow paper text on those rows; ask
the user to confirm before writing any "follow the code instead" override.

#### T1C-RFlow incoherency table (worked example)

| # | Axis | Paper text | Released code | Load-bearing? | Direction |
|---|---|---|---|---|---|
| 1 | U-Net | `[128, 128, 256]`, 3 levels, 2 res-blocks, no attention | `[64, 128, 256, 512]`, 4 levels, attention at last 2, **178.6 M params** | **HIGH** | code > paper text (≥1-2 dB) |
| 2 | `use_discrete_timesteps` | implies discrete | JSON `false`, script forces `True` | NO | backbone constraint |
| 3 | AMP | §4 silent | `GradScaler` + `autocast(fp16)` | LOW | ≤0.1 dB |
| 4 | VAE | cites Guo 2024 MAISI | ships `autoencoder_epoch273.pt` (their retraining) | **HIGH** | code > paper text, ceiling-shift |

Wrapper takes rows 1 and 4 from the paper (architecture: **49.6 M params**;
VAE: VENA's MAISI-V2 — symmetric to our own model). Rows 2 and 3 keep
the code's runtime behaviour because both are forced by the backbone /
hardware.

### Step 3.8 — Direct MONAI scheduler instantiation (real, hit on T1C-RFlow integration)

VENA's `vena.model.fm.sampler.rflow.RFlowEngine` is a thin wrapper around
`monai.networks.schedulers.rectified_flow.RFlowScheduler` that **does not
expose** `use_timestep_transform` or `base_img_size_numel`. Several
latent-flow upstream scripts (including T1C-RFlow's
`train_rflow.py:136-143`) set both. Instantiate the MONAI scheduler
directly in the wrapper's `runner.py` and pin the paper's exact kwargs.
Do **not** extend `RFlowEngine` — it serves VENA's FM trainer and is not
meant to grow per-competitor knobs.

Same call site for inference: build the scheduler in `inference.py` with
the same kwargs and use `scheduler.set_timesteps(num_inference_steps=K,
input_img_size_numel=numel(z_curr))` per the upstream's `test_*.py`
pattern (T1C-RFlow `test_rflow.py:269-283`).

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

**Picasso path case-sensitivity (real, hit on T1C-RFlow integration).**
`experiments_root` on Picasso must use `…/execs/vena/experiments/…`
(**lowercase** `vena`). The directory `…/execs/VENA/…` (uppercase) does
not exist. The engine's `_preflight` fails fast on
`experiments_root.parent.parent.exists()` so the mismatch surfaces
immediately, but a sibling pGAN config under the same routine tree may
use the wrong case and template-copy errors will silently propagate. When
copying YAMLs across competitors, grep for `execs/VENA` and lowercase
every occurrence. (The log directory under `…/execs/VENA/logs/` is fine —
launchers `mkdir -p` it on the fly.)

### Step 5 — Per-platform launchers

The three platforms have different transports. Match the templates exactly.

**Server-3 (ICAI workstation, RTX 4090, no SLURM).**
- Pattern: rsync repo → ssh → `screen -dmS <session>` → return.
- Use `screen` (tmux not installed). Session anchor: `vena-<name>-smoke`.
- Pre-warm any pretrained-weight cache on icai-server (it has internet). **No
  VGG pre-warm needed for non-perceptual competitors** (FM-only, RFlow-only,
  diffusion-only) — drop the step from copied launcher templates.
- GPU 0 is the FM-train target; check which GPU is free with `nvidia-smi --query-gpu=memory.free --format=csv,noheader`.
- Conda env: `~/.conda/envs/vena/bin/python` on `icai-server`.
- **rsync excludes (real, hit on T1C-RFlow integration).** When the
  vendored upstream ships Git LFS payloads (T1C-RFlow has an 80 MB VAE
  checkpoint at `src/external/<name>/upstream/checkpoints/`), add
  `--exclude="src/external/<name>/upstream/checkpoints/"` to the launcher's
  rsync excludes. VENA uses its own MAISI-V2 checkpoint via
  `src/external/LINKS.md`; the upstream LFS payload is dead weight on the
  remote and a slow first-rsync. Re-`pip install -e .` on the remote after
  the first rsync of a new competitor so the entry points get registered.

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

Acceptance is THREE platforms in **strict order**: server-3 → loginexa →
Picasso. Each stage gates the next: if a smoke fails, fix the bug before
moving up. Do NOT skip a tier because "it'll be the same on the next one" —
each platform has caught at least one bug during the pGAN integration
(server-3 caught the multi-worker h5py deadlock, loginexa caught the V100
sm_70 / cu130 mismatch, Picasso caught the longitudinal-id resolver gap and
the REMBRANDT flat-split fallback gap).

1. **Server-3 multi-cohort smoke (4 epochs, 1 patient/cohort)** — uses
   `corpus_server3.json`; loss descends across the cohort union; checkpoints +
   metrics CSV + decision.json written; sentinel in log. Wallclock ≤ 10 min on
   RTX 4090 with `num_workers: 0`.
2. **Loginexa multi-cohort smoke (2 epochs)** — uses `corpus_picasso.json` via
   the shared Lustre mount; same artifact set; wallclock ≤ 10 min on V100. The
   shorter 2-epoch budget exists because loginexa shares GPUs with other users
   and the 30-min wallclock convention applies.
3. **Picasso full submission** — actually `sbatch` (no dry-run): once
   `squeue -j <id>` shows `RUNNING` and the python log emits the
   `MultiCohortImageSliceDataset[train/fold0]` line with the expected cohort
   count, the integration is live. The user gates the budget by requesting the
   submission; they don't need to be in the loop for the sbatch itself once
   they've authorised it.

### Step 6.1 — Watcher pattern (`/loop`-driven async monitoring)

A 4-epoch smoke takes minutes; a Picasso job takes hours-to-days. The agent
must NOT block the conversation by polling. The canonical pattern for every
submitted job is **one background watcher + one ScheduleWakeup backstop**:

1. After every submission (server-3 / loginexa / Picasso), `Bash run_in_background=true`
   a polling loop:

   ```bash
   for i in $(seq 1 N); do
     if ssh <host> "grep -q '<sentinel>' <log>"; then
       echo "[$(date +%H:%M:%S)] DONE iter $i"
       exit 0
     fi
     if ssh <host> "grep -q 'Traceback' <log>"; then
       echo "[$(date +%H:%M:%S)] TRACEBACK iter $i"
       exit 1
     fi
     sleep <T>
   done
   ```

   The sentinel for VENA competitor runs is `pGAN-train completed`-style
   (replace per competitor). `T` is 8–12 s for smokes, 30–60 s for Picasso.

2. Call `ScheduleWakeup` with `prompt: <<autonomous-loop-dynamic>>` so the
   harness wakes the agent again at a heartbeat interval if the watcher hangs
   or the SSH connection drops. Heartbeat budget: 60–270 s during active
   polling (cache stays warm), 1200–1800 s when the only signal is the
   Monitor or the watcher itself. Don't pick 300 s — it pays the cache miss
   without amortising.

3. On wake (`<task-notification>` for the watcher, or the dynamic loop
   trigger), check the watcher output file, decide:
   - **DONE** → run inference on 10 val patients with `--epoch best`,
     verify NIfTI + PNG + metrics, mark task complete, advance to the next
     platform tier.
   - **TRACEBACK** → fetch the failing log block, fix the bug, kill the
     job (`scancel <id>` or `screen -X -S <session> quit`), resubmit.
   - **TIMEOUT** → re-arm a new watcher with a longer window, schedule
     a heartbeat, return.

4. Two watchers can run concurrently if smokes are independent (server-3 +
   loginexa share no GPUs); a unified poller (`if [[ S3 -gt 0 && LX -gt 0 ]]`)
   is fine. Picasso runs are long-lived — set Monitor-style heartbeats at
   1200 s.

5. NEVER chain `sleep` calls in the foreground — the harness blocks long
   leading sleeps. Run the wait in `run_in_background: true` OR use the
   Monitor tool with an `until <check>; do sleep <T>; done` loop.

This is the same pattern the `server3` skill encodes for VENA's own FM
trainer; the competitor skill reuses it verbatim.

After each training run that produced checkpoints, **run inference on 10 val patients**
using the `best` epoch (the early-stopping-selected weights), not `latest`:

```
vena-competitor-<name>-infer \
    --run-dir   <run_dir> \
    --image-h5  <UCSFPDGM_image.h5> \
    --epoch     best \
    --n-patients 10 \
    --phase     val
```

Verify outputs: 10 NIfTI volumes + PNG midslices + `metrics.csv` + `summary.json`.

### Paired comparison axes (must match VENA's reference run)

The default VENA reference is `routines/fm/train/configs/runs/picasso_s1_1000ep_fft.yaml`.
Audit each axis when authoring the competitor's `picasso_full.yaml`:

| Axis | Match exactly | Match conceptually | Differ by design |
|---|---|---|---|
| seed | ✅ | | |
| fold | ✅ | | |
| corpus_registry | ✅ | | |
| max_epochs | ✅ | | |
| patience | ✅ (on epoch-mean train loss analogue) | | |
| save cadence | ✅ | | |
| batch_size (physical) | ✅ | | |
| num_workers | ✅ | | |
| walltime | ✅ | | |
| input modalities + masks | | | ⚠ paper-faithful set |
| output domain (latent vs image) | | | ⚠ paper-faithful |
| target metric | | ⚠ best-epoch selector | |

Differences in the right two columns are explicit choices that go in the
validation note — they are not bugs.

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
9. **Calling `load_autoencoder()` without `checkpoint_path`.** It has no
   default. Surface the path in `--vae-checkpoint` (required) and read
   from `src/external/LINKS.md` per platform. See Step 3.6.
10. **Treating `AutoencoderHandle` as a decoder.** It is not. Wrap it:
    `MaisiDecoder(handle)`. See Step 3.6.
11. **Passing 3-D tensors to `percentile_normalise(foreground_only=True)`.**
    It expects 5-D `(B, C, H, W, D)`. Wrap with `[None, None]` before and
    `[0, 0]` after. The 3-D failure mode silently drops every patient and
    your `metrics.csv` ends up empty — the inference job *exits 0* and
    looks successful. See Step 3.6.
12. **Skipping the post-inference success assertion.** After
    `run_inference` returns, the wrapper (or the watcher) must verify
    `summary.json["n_patients_succeeded"] > 0`. Without this, a per-patient
    exception that hits every patient is invisible: the wall log says
    "inference complete: wrote 0 rows" and exits 0.

## Mental checklist

- [ ] Upstream cloned, SHA recorded, `.pyc`/`.git` removed.
- [ ] `PATCHES.md` lists every in-place change.
- [ ] Wrapper `dataset.py` returns deterministic tensors with no augmentation.
- [ ] `MultiCohort<X>Dataset` reads `corpus_<host>.json` (same as VENA's FM trainer).
- [ ] Cohort fallback: missing H5 / empty split → WARNING, not crash.
- [ ] `_open()` uses plain `"r"` mode, never SWMR.
- [ ] Smoke configs set `num_workers: 0` (multi-cohort h5py deadlock).
- [ ] Runner saves `best_net_*` on epoch-metric improvement and supports `patience`.
- [ ] Unit tests pin: determinism, shape/range, multi-cohort concat, missing-cohort fallback, **longitudinal patient→scan resolver (BraTS-GLI, LUMIERE), flat-splits fallback (REMBRANDT)**.
- [ ] **One-to-one vs many-to-one disambiguated** (see Step 3.3). If one-to-one, the picasso job count = number of source modalities (3 for VENA's standard input set), each with its own config + run id + inference dir.
- [ ] **Image-domain vs latent-domain disambiguated** (see Step 3.3.5). Latent-domain reads `cohort.latent_h5`, returns `(C, h, w, d)` per patient, needs the Step 3.6 decode trio at inference.
- [ ] **Upstream paper-vs-code incoherency table** in `UPSTREAM.md` enumerated (see Step 3.7). Wrapper follows the code.
- [ ] **Latent-domain inference primitives wired correctly** (see Step 3.6):
      `--vae-checkpoint` is required; `MaisiDecoder(load_autoencoder(ckpt))`;
      `percentile_normalise(real[None,None], …)[0,0]`; post-condition
      `summary.json["n_patients_succeeded"] > 0` asserted.
- [ ] **Picasso `experiments_root` uses lowercase `…/execs/vena/…`**, not `VENA` (see Step 4 note).
- [ ] **rsync excludes the upstream's LFS payload** at
      `src/external/<name>/upstream/checkpoints/` if present (see Step 5
      server-3 note).
- [ ] Engine writes `decision.json` v1.0 with the competitor + corpus_registry block.
- [ ] Three platform launchers exist and dry-run cleanly.
- [ ] Console scripts registered in `pyproject.toml`.
- [ ] Server-3 multi-cohort smoke completed; artifacts verified.
- [ ] Loginexa multi-cohort smoke completed; artifacts verified.
- [ ] Picasso job actually `sbatch`-submitted (not dry-run) once user authorises;
      `scontrol show job` confirms RUNNING on `gpu_partition` with `--constraint=dgx`.
- [ ] Inference run on 10 val patients with `--epoch best` after each training run.
- [ ] Validation note in `.claude/notes/validation/<name>.md`.
- [ ] Pair-comparison axes table in the validation note flags every divergence
      from VENA's reference run.
