# Offline augmentation — handover

> Status as of 2026-06-04 ~10:35 CEST. Hand-off to a continuing agent.

## Read first

1. `.claude/notes/augmentation_approach/unrefined_proposal.md` — original
   Claude-Opus-Web research note (literature + first design draft).
2. `.claude/notes/augmentation_approach/refined_proposal.md` — refined K=4
   variant menu, library choice, empirical timings.
3. `/home/mpascual/.claude/plans/context-you-have-gained-cheerful-fountain.md`
   — the approved plan I executed against.
4. **This file** — what is actually built, what is currently running, what
   is left.

## TL;DR

- Library + routine + tests + new H5 schemas + train-side wiring **all
  landed locally** and on server 3. Local pytest suite (304 tests in
  `tests/data tests/model/fm tests/routines/fm`) was green at every step.
- **One smoke run passed** end-to-end on server 3 (`UCSF-PDGM` × 2
  patients × K=4 variants). All four variants cleared the per-cohort
  PSNR/SSIM gate.
- **Full bank build is in flight** on server 3, cohort-by-cohort, 2 ranks
  per cohort. Completed: UCSF-PDGM, IvyGAP. In flight: BraTS-GLI image
  merge, LUMIERE rank0+rank1 bank-build. Not started yet: UPENN-GBM merge
  (rank shards on disk waiting), LUMIERE encode + merge.
- **Two QC bugs found late** (more below):
  1. The QC figure rendered `original = augmented = aug_image_normed` →
     the leftmost panel showed the augmented image, not the clean source.
     The aug-H5 data is correctly augmented (verified — see
     `/tmp/verify_aug_applied.py`); the figure was just visually
     ambiguous.
  2. v4 hyperparameters had `elastic_prob = affine_prob = 0.7`, so
     `(1 - 0.7)² ≈ 9 %` of v4 rows had neither transform fire → byte-
     identical to source. Bumped both to `1.0`.
  Fixes landed in `routines/offline_aug/maisi/engine/offline_aug_engine.py`
  and `routines/offline_aug/maisi/configs/aug_pipelines/k4_v1.yaml`, both
  rsynced to server 3 but **the banks already on disk were built with the
  buggy config and are not regenerated** — non-fatal for training (v4
  identity rows just look like extra v0 samples).
- **Open work**: finish current bank-build pipeline, merge remaining
  cohorts, update `corpus_server3.json`, launch the 4-epoch smoke training
  run (`server3_4epoch_aug_offline.yaml`), and optionally regenerate the
  QC figures with the fixed code.

## Code that landed

All paths relative to `/home/mpascual/research/code/VENA`.

### New library — `src/vena/data/augment/`

The flat `augment/` module was split into:

- `online/` — verbatim move of the existing `{base, pipeline, config,
  tracker}.py` + `transforms/{flip, gamma, rotate, translate}.py`. All
  internal imports rewritten from `vena.data.augment.X` to
  `vena.data.augment.online.X`. The top-level `vena.data.augment.__init__`
  is a shim that re-exports the online surface so legacy callers
  (`from vena.data.augment import AugmentationTracker, …`) keep working.
- `offline/` — new:
  - `variants.py` — `make_variant_v1..v4(p_overrides) → tio.Compose`.
    v1 bias+gamma, v2 hist-shift+brightness, v3 noise+anisotropy+blur+motion,
    v4 elastic+affine (joint).
  - `torchio_adapters.py` — `MonaiHistogramShift` wraps
    `monai.transforms.RandHistogramShift` as a `tio.IntensityTransform`
    (TorchIO has no native monotonic remap).
  - `bank_builder.py` — `OfflineAugBankBuilder.build()` produces a single
    per-cohort + per-rank `<COHORT>_image_aug_rank{rank}.h5`.
    `merge_aug_image_h5_shards()` concatenates rank shards row-wise.
- `online/tracker.py` — `AugmentationTracker` kept; added
  `VariantTracker` callback that writes `metrics/variants_per_epoch.csv`
  (columns `epoch,variant,count`). It reads `batch["_aug_variant"]`,
  written by the offline-augmented dataset.

### New H5 schema — `src/vena/data/h5/augmented/`

- `image_domain.py` — `build_aug_image_manifest(cohort, modalities)`,
  `assert_aug_image_h5_valid(...)`. Schema invariants:
  - `schema_version="0.1.0"`, `domain="image"`, `cohort=<name>`,
    `expected_shape=(192,224,192)` for **every** cohort. The bank-builder
    crops to the box before augmenting so the downstream
    `LatentH5Converter` sees `crop/origin = (0,0,0)` and the crop step is
    a no-op.
  - Required aug-specific root attrs: `source_image_h5_path` (yes —
    initially forgot this in the merge writer; see Caveat #4 below),
    `source_image_h5_sha256`, `aug_config_json`, `aug_config_sha256`,
    `variants_json`, `seed`, `world_size`, `rank`.
- `latent_domain.py` — `build_aug_latent_manifest(...)`,
  `assert_aug_latent_h5_valid(...)`. Same shape conventions as the clean
  latent H5 (`latents/<mod>` is `(N, 4, 48, 56, 48)`,
  `masks/tumor_latent` is `(N, 3, 48, 56, 48)`). **Forbidden** groups:
  `patients/*`, `splits/*` — partitioning lives on the clean latent H5 and
  is keyed via `source_row_index`.

### `LatentH5Converter` patch

`src/vena/data/h5/latent_domain/convert.py`:
- `LatentH5Config` gained `aug_mode: bool = False` and
  `aug_config_sha256: str | None = None`.
- When `aug_mode=True`, the converter (a) builds the aug-latent manifest
  instead of the clean one, (b) skips `_copy_csr`, `_copy_splits`,
  `_write_empty_csr`, `_create_priors_placeholder`, and (c) stamps
  `source_aug_image_h5_path/sha256` + `aug_config_sha256` + `variants_json`
  as root attrs. `_copy_metadata` is reused unchanged — the aug-latent
  manifest declares `source_row_index`, `variants`, `aug_params_json` as
  `kind="metadata"` and the converter's existing metadata-copy iterates
  them.

### Registry — `src/vena/data/registry/models.py`

`CohortEntry` gained two optional fields:

```python
image_aug_h5: Path | None = None
latent_aug_h5: Path | None = None
```

New helper `CorpusRegistry.cv_cohorts_with_aug()`.

### Training engine — `routines/fm/train/engine.py`

- `_DataCfg` gained
  `use_offline_augmented_data: bool = False` and
  `variant_weights: dict[str, float] = {"v0":0.2,"v1":0.2,...}`.
- `_assert_preflight_gates(cfg)` now also calls
  `_assert_offline_aug_gate(cfg)` when the flag is on (checks every cv
  cohort has `latent_aug_h5` populated and the file exists; checks
  variant_weights are non-empty and assign non-zero weight to at least
  one non-v0 variant).
- `_build_decision_payload` bumped from `"0.4.0"` → `"0.5.0"`. New keys:
  `use_offline_augmented_data`, `variant_weights`, `aug_image_h5_paths`,
  `aug_latent_h5_paths`.
- `VariantTracker` callback is appended to the Lightning callback list
  when `use_offline_augmented_data=True`.

### Train DataModule — `src/vena/model/fm/lightning/data.py`

New class `OfflineAugmentedLatentH5Dataset` wraps a clean
`LatentH5Dataset` plus an aug-latent H5 handle. `__getitem__` samples
`vN` ∈ `variant_weights`, reads from the clean H5 for `v0` or from the
aug H5 for `v1..vK` (joined via `source_row_index` ↔ `ids[i]`), then runs
the online `AugmentationPipeline` on top. Sets
`batch["_aug_variant"] = vN`.

`MultiCohortLatentDataModule.__init__` gained
`use_offline_augmented_data: bool` and
`variant_weights: dict[str, float] | None`. In `setup()`, when the flag
is on, the **train** dataset is built as
`OfflineAugmentedLatentH5Dataset(cohort.latent_h5, cohort.latent_aug_h5,
train_scan_ids, ...)`. **Val and test** datasets always read the clean
H5 — leakage-proof by construction.

### Routine — `routines/offline_aug/maisi/`

```
routines/offline_aug/maisi/
├── cli.py
├── configs/
│   ├── aug_pipelines/k4_v1.yaml
│   ├── merges/<cohort>_merge.yaml      # one per cv cohort
│   ├── smoke_ucsf_pdgm.yaml
│   ├── {ucsf_pdgm,brats_gli,upenn_gbm,ivy_gap,lumiere}_rank{0,1}.yaml
├── engine/
│   ├── __init__.py
│   └── offline_aug_engine.py            # OfflineAugMaisiRoutineConfig + Engine
└── figures.py                           # AugRoundtripRow + render
```

`scripts/gen_offline_aug_configs.py` regenerates the per-cohort + per-rank
YAMLs (regenerate if the cohort list or path convention changes; don't
hand-edit individual YAMLs).

`scripts/merge_offline_aug_shards.py` is the merge driver: reads a
`merges/<cohort>_merge.yaml` and concatenates the rank shards into
`<cohort>_{image,latents}_aug.h5`.

`scripts/regen_aug_qc_figures.py` is the **post-hoc figure regenerator**.
Takes the merged image+latent H5 + the clean source H5 + the AE
checkpoint and renders 4 figures × 4 variants per cohort. Use it on the
banks that landed with the buggy QC code.

### Tests

- `tests/data/h5/augmented/test_manifests.py` — 6 unit tests for the new
  H5 manifest validators (image + latent).
- `tests/data/augment/offline/test_variants.py` — 10 unit tests for the
  variant builders (input-only invariants for v1/v2/v3, joint warp for
  v4, mask stays integer).

Run: `~/.conda/envs/vena/bin/python -m pytest tests/ -m "not slow and not gpu" -q`.

## Server 3 — current bank-build state

Storage root: `/media/hddb/mario/data/GLIOMAS/<COHORT>/h5/`.
Log root: `/media/hddb/mario/smoke_logs/offline_aug/<cohort>_rankN.log`.
Artifact root: `/media/hddb/mario/artifacts/offline_aug/maisi/<cohort>/LATEST/`.

| Cohort     | Bank build | Encode | Merge | Validated |
|------------|-----------|--------|-------|-----------|
| UCSF-PDGM  | ✅ both ranks | ✅ | ✅ | ✅ |
| BraTS-GLI  | ✅ both ranks | ✅ | ⏳ image ~70%, latent not started | — |
| UPENN-GBM  | ✅ both ranks | ✅ | ⏳ shards on disk; merge not launched | — |
| IvyGAP     | ✅ both ranks | ✅ | ✅ | ✅ |
| LUMIERE    | ⏳ both ranks at 73% bank-build | — | — | — |

Live PIDs (likely still alive when the next agent arrives):
- `1673509` — BraTS-GLI merge (state: D, disk-bound, ~2.5h in)
- `1737707` — LUMIERE rank0 bank-build (CPU)
- `1738856` — LUMIERE rank1 bank-build (CPU)

GPU 0 and GPU 1 are both idle right now (LUMIERE not yet in encode
phase). The BraTS-GLI merge runs CPU-only.

## Empirical throughput / timings

Measured on server 3 (RTX 4090, autocast + fp16 GroupNorm):

| Step                                | Throughput |
|-------------------------------------|------------|
| Image-domain augmentation (CPU)     | ~5–7 s/row (1 row = 1 scan × 1 variant) |
| MAISI encode per modality           | ~0.92–1.05 s |
| MAISI encode per scan (4 mods)      | ~3.65–4.5 s |
| `sha256_file` on the 82 GB shard    | ~10–15 min (CRC bottleneck in the engine's decision.json step) |
| Merge image-aug H5 (165 GB)         | ~3 h (gzip recompress; disk-bound) |

For the full K=4 × CV cohorts at server-3 scan-level sharding (2 ranks /
cohort, cohorts sequential):
- UCSF-PDGM: ~1 h total
- BraTS-GLI: ~5 h total (the long pole)
- UPENN-GBM: ~1 h total
- IvyGAP: ~5 min total
- LUMIERE: ~1.5 h total

## Caveats encountered

### 1. Dual `vena/` ↔ `src/vena/` layout on icai-server

The server keeps a top-level `vena/` directory at
`/home/mariopascual/projects/VENA/vena/` IN ADDITION to the editable
install of `src/vena/`. When `cwd = /home/mariopascual/projects/VENA`,
`sys.path[0]` is empty (= cwd), and Python finds the top-level `vena/`
**before** the editable install's `src/vena/`. The two diverged whenever
I rsynced only to `src/vena/...`.

**Symptom**: `ModuleNotFoundError: No module named 'vena.data.augment.offline'`
when running `python -m routines.offline_aug.maisi.cli ...` from the
project root.

**Fix**: rsync to BOTH paths:
```bash
rsync -av <local-file> icai-server:/home/mariopascual/projects/VENA/src/<path>
rsync -av <local-file> icai-server:/home/mariopascual/projects/VENA/vena/<path-without-src>
```

Always wipe `__pycache__` after each sync:
`find . -name __pycache__ -path "*augment*" -exec rm -rf {} +`.

(Already captured in `memory/reference_icai_server.md`.)

### 2. Multi-line `bash -c ...` over ssh corrupts `&` background

Sending a multi-line `nohup python ... & RANK0=$!; \\` over ssh
occasionally launched only one of the two parallel processes
(line-2-not-found errors on the second redirect). Workaround: launch
ranks 0 and 1 in **two separate ssh invocations**.

### 3. ControlMaster hang under heavy server load

After several hours of heavy CPU + disk I/O, the SSH ControlMaster
connection stopped multiplexing new sessions (Bash calls returning 0
bytes for minutes). **Workaround**: append
`-o ControlPath=none -o ConnectTimeout=15` to every ssh call once the
load reports `load average > 10`.

### 4. Merge writer missed `source_image_h5_path` root attr

First BraTS-GLI / UCSF-PDGM merge failed validation:
```
H5ValidationError: missing aug-specific root attr: source_image_h5_path
```
Fixed in `vena/data/augment/offline/bank_builder.py::merge_aug_image_h5_shards`:
now reads `ref_src_path = str(f0.attrs["source_image_h5_path"])` from the
first shard and writes it into the merged file. The existing UCSF-PDGM
merged H5 was patched in-place with a one-liner
(`f.attrs["source_image_h5_path"] = ...`). The fix has been rsynced;
the BraTS-GLI merge currently in flight is using the fixed version.

### 5. QC figure rendered `original = augmented` (the user-reported bug)

`_run_qc` in the engine was passing the augmented image as both
`original` and `augmented` to `AugRoundtripRow`, so the rendered figure
showed identical panels in columns 1–2. The actual H5 content is
correctly augmented (PSNR(source, aug) on a representative UCSF-PDGM row
was 22–38 dB for v1/v2/v3 inputs).

**Fixes (already in main code, synced to server 3)**:
- `offline_aug_engine.py` now reads the clean source row from
  `cfg.source_image_h5`, applies the same `crop/origin` the bank-builder
  used (via `_box_native_numpy`), and passes the clean image as
  `original`.
- `scripts/regen_aug_qc_figures.py` does the same retrofit on already-
  built banks. **Not yet run** (interrupted while launching it for
  UCSF-PDGM + IvyGAP — see "Continuing work" below).

### 6. v4 had ≈9 % byte-identical-to-source rows

`elastic_prob=0.7` × `affine_prob=0.7` ⇒ (0.3)² = 9 % chance neither
fires → identity. Verified on UCSF-PDGM's stored aug-H5: 1 of 4 sampled
v4 rows came out PSNR-infinite vs source.

**Fix**: `routines/offline_aug/maisi/configs/aug_pipelines/k4_v1.yaml`
now sets `elastic_prob: 1.0` and `affine_prob: 1.0`. Synced to server 3.
**The already-built banks were not regenerated** — those ~9 % v4
identity rows are equivalent to extra v0 samples at training time, which
is acceptable for our K=4 setup.

### 7. `mkdir -p` race in multi-line ssh bash

`mkdir -p ... && nohup python ... > log.log 2>&1 & disown` occasionally
hit a transient "No such file or directory" for the log target. Fix:
ensure `mkdir -p` of the log dir is its own statement before the
launches.

### 8. `_box_native_numpy` wrong subpath rsync

`rsync -av routines/offline_aug/maisi/engine/offline_aug_engine.py
routines/offline_aug/maisi/configs/aug_pipelines/k4_v1.yaml
icai-server:.../routines/offline_aug/maisi/` collapsed both files into
`maisi/` (not `maisi/engine/` and `maisi/configs/aug_pipelines/`). When
running the regen script, the server kept loading the OLD
`offline_aug_engine.py` (without `_box_native_numpy`). **Workaround**:
always rsync each source file with its complete destination subpath, one
file per `rsync` call.

Suspect file artifacts left on server 3 from this:
- `/home/mariopascual/projects/VENA/routines/offline_aug/maisi/offline_aug_engine.py`
  (deleted in last cleanup)
- `/home/mariopascual/projects/VENA/routines/offline_aug/maisi/k4_v1.yaml`
  (deleted in last cleanup)
- `/home/mariopascual/projects/VENA/routines/offline_aug/maisi/engine/smoke_ucsf_pdgm.yaml`
  (deleted in last cleanup)

## Continuing work — checklist for the next agent

### A. Finish current bank build

1. **Monitor LUMIERE bank-build** (PIDs 1737707, 1738856). When both
   exit, launch the LUMIERE merge:
   ```bash
   ssh -o ControlPath=none icai-server \
     'cd /home/mariopascual/projects/VENA && \
      CUDA_VISIBLE_DEVICES= nohup ~/.conda/envs/vena/bin/python -u \
        scripts/merge_offline_aug_shards.py \
        routines/offline_aug/maisi/configs/merges/lumiere_merge.yaml \
        --overwrite > /media/hddb/mario/smoke_logs/offline_aug/lumiere_merge.log \
        2>&1 & disown; echo $!'
   ```

2. **Once BraTS-GLI merge finishes** (PID 1673509), launch the UPENN-GBM
   merge — same pattern, point at
   `routines/offline_aug/maisi/configs/merges/upenn_gbm_merge.yaml`.

3. After each merge: validate with
   `vena.data.h5.augmented.assert_aug_{image,latent}_h5_valid` and delete
   the rank shards (`*_rank{0,1}.h5`) to free disk space.

### B. Update the corpus registry

For every cv cohort, add to
`routines/fm/train/configs/corpus/corpus_server3.json`:

```json
"image_aug_h5": "/media/hddb/mario/data/GLIOMAS/<COHORT>/h5/<stem>_image_aug.h5",
"latent_aug_h5": "/media/hddb/mario/data/GLIOMAS/<COHORT>/h5/<stem>_latents_aug.h5"
```

Stems (must match the cohort key on disk):
- UCSF-PDGM → `ucsf_pdgm`
- BraTS-GLI → `brats_gli`
- UPENN-GBM → `upenn_gbm`
- IvyGAP → `ivy_gap`
- LUMIERE → `lumiere`

Verify with:
```python
from vena.data.registry import load_registry
r = load_registry("routines/fm/train/configs/corpus/corpus_server3.json")
print([c.name for c in r.cv_cohorts_with_aug()])
# Expected: ['UCSF-PDGM','BraTS-GLI','UPENN-GBM','IvyGAP','LUMIERE']
```

### C. (Optional) Regenerate QC figures with the fix

```bash
ssh -o ControlPath=none icai-server \
  'cd /home/mariopascual/projects/VENA && \
   CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 nohup \
     ~/.conda/envs/vena/bin/python -u scripts/regen_aug_qc_figures.py \
     --cohort UCSF-PDGM \
     --source-image-h5 /media/hddb/mario/data/GLIOMAS/UCSF_PDGM/h5/UCSFPDGM_image.h5 \
     --image-aug-h5 /media/hddb/mario/data/GLIOMAS/UCSF_PDGM/h5/ucsf_pdgm_image_aug.h5 \
     --latent-aug-h5 /media/hddb/mario/data/GLIOMAS/UCSF_PDGM/h5/ucsf_pdgm_latents_aug.h5 \
     --autoencoder /media/hddb/mario/checkpoints/MAISI_V2_RM/NV-Generate-MR/models/autoencoder_v2.pt \
     --output-dir /media/hddb/mario/artifacts/offline_aug/qc_regen/UCSF-PDGM \
     --n-per-variant 4 \
     --device cuda:0 \
   > /media/hddb/mario/smoke_logs/offline_aug/qc_regen_ucsf_pdgm.log \
     2>&1 & disown; echo $!'
```

Repeat for IvyGAP, BraTS-GLI, UPENN-GBM, LUMIERE after each lands. Each
takes ~3–5 min. The regenerated figures land in
`/media/hddb/mario/artifacts/offline_aug/qc_regen/<COHORT>/` with a
`summary.md` listing per-(variant, patient) PSNR(source vs augmented).
PSNR should be ~20–40 dB for v1/v2/v3 inputs (lower = stronger augmen-
tation); for v4 it should be **finite** for every row now that
prob=1.0.

The user reported "in any of the figures I can't see a change" — that
was the bug. After regen, every row should show a clearly different
column 2 (augmented).

### D. Launch the 4-epoch smoke training run

Once the corpus JSON is updated:

```bash
ssh -o ControlPath=none icai-server \
  'cd /home/mariopascual/projects/VENA && \
   mkdir -p /media/hddb/mario/smoke_logs && \
   LOG=/media/hddb/mario/smoke_logs/aug_offline_4ep_$(date +%Y%m%d_%H%M%S).log && \
   nohup ~/.conda/envs/vena/bin/python -m routines.fm.train.cli \
     routines/fm/train/configs/runs/server3_4epoch_aug_offline.yaml \
     > "$LOG" 2>&1 & disown; echo "LOG=$LOG"'
```

Verify per `~/.claude/skills/server3` (the project skill):
- `<run>/logs/train.log` ends with `"FM-train completed"`.
- `metrics/train_step.csv` populated.
- `metrics/augmentations_per_epoch.csv` shows online combos
  (`flip_lr`, `translate`).
- `metrics/variants_per_epoch.csv` (NEW) shows rows `v0..v4` with
  non-zero counts approximately respecting `variant_weights`.
- `decision.json` has `schema_version: "0.5.0"`,
  `use_offline_augmented_data: true`.
- `exhaustive_val/epoch_NNN/metrics.csv` has `wc -l > 1` per the
  server3 skill's subprocess-silent-fail trap.

### E. (Optional) Leakage-proof diagnostic

A one-shot script that, for every cv cohort, opens
`<cohort>_latents_aug.h5`, reads the set of `ids`, opens
`<cohort>_latents.h5`, reads `splits/test`, and asserts the intersection
is empty. The bank-builder already enforces this at write time (test
patients are excluded from `_resolve_rows`); this is belt-and-braces
against a future regression.

## Files modified this session (local + server 3)

```
src/vena/data/augment/__init__.py                              # shim
src/vena/data/augment/online/*                                 # moved here
src/vena/data/augment/online/tracker.py                        # +VariantTracker
src/vena/data/augment/offline/__init__.py                      # NEW
src/vena/data/augment/offline/variants.py                      # NEW
src/vena/data/augment/offline/torchio_adapters.py              # NEW
src/vena/data/augment/offline/bank_builder.py                  # NEW (+ later patch for merge)
src/vena/data/h5/augmented/__init__.py                         # NEW
src/vena/data/h5/augmented/image_domain.py                     # NEW
src/vena/data/h5/augmented/latent_domain.py                    # NEW
src/vena/data/h5/latent_domain/convert.py                      # patched: aug_mode
src/vena/data/registry/models.py                               # +image_aug_h5, +latent_aug_h5
src/vena/model/fm/lightning/data.py                            # +OfflineAugmentedLatentH5Dataset
routines/fm/train/engine.py                                    # +offline_aug gates, decision v0.5.0
routines/fm/train/configs/runs/server3_4epoch_aug_offline.yaml # NEW smoke YAML
routines/offline_aug/__init__.py                               # NEW
routines/offline_aug/maisi/                                    # NEW routine
scripts/gen_offline_aug_configs.py                             # NEW config generator
scripts/merge_offline_aug_shards.py                            # NEW merge driver
scripts/regen_aug_qc_figures.py                                # NEW figure retrofit
scripts/bench_encode_aug.py                                    # NEW empirical benchmark
tests/data/h5/augmented/test_manifests.py                      # NEW
tests/data/augment/offline/test_variants.py                    # NEW
pyproject.toml                                                 # +torchio dep, +console scripts
```

## Useful commands

```bash
# Local smoke tests (no GPU)
~/.conda/envs/vena/bin/python -m pytest \
  tests/data/h5/augmented tests/data/augment/offline tests/data/registry \
  tests/model/fm tests/routines/fm \
  -m "not gpu and not slow" -q

# Verify a built bank
~/.conda/envs/vena/bin/python -c "
from vena.data.h5.augmented import assert_aug_image_h5_valid, assert_aug_latent_h5_valid
mods = ['t1pre','t1c','t2','flair']
assert_aug_image_h5_valid('<cohort>_image_aug.h5', '<COHORT>', mods)
assert_aug_latent_h5_valid('<cohort>_latents_aug.h5', '<COHORT>', mods, 3)
print('OK')
"

# Probe per-row augmentation magnitude on a built bank
# (see /tmp/verify_aug_applied.py — was a throwaway, easy to recreate)

# Server-3 process snapshot (use direct ssh under load)
ssh -o ControlPath=none icai-server \
  'pgrep -af "offline_aug|merge_offline_aug" | grep -v "bash -c"; \
   nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; \
   df -h /media/hddb | tail -1'
```

## Memory entries worth adding

If not already saved, the next agent should consider committing:

- "Server 3 has a dual `vena/` ↔ `src/vena/` layout — rsync code to BOTH
  paths or runtime imports go stale."
- "`rsync -av A.py B.yaml host:dir/` collapses both files into `dir/`,
  even if their source-side directories differ. Always specify the
  complete destination path per-file."
- "Per-rank aug-H5 merges write `source_image_h5_path` as a root attr —
  do not skip it; `assert_aug_image_h5_valid` rejects without it."

---

This document and the plan file together should be enough for a fresh
agent to resume.
