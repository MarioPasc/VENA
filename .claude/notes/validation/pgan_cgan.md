# pGAN-cGAN (Dar et al., 2019) — VENA integration validation

Competitor: **Dar, S. U. H., et al.** "Image Synthesis in Multi-Contrast MRI With Conditional
Generative Adversarial Networks." *IEEE TMI* 38(10):2375–2388, 2019. DOI: 10.1109/TMI.2019.2901750.
Upstream code: https://github.com/icon-lab/pGAN-cGAN, vendored at SHA `b4ca7047`.

Only the **pGAN** branch (paired pix2pix-style) is used. cGAN (CycleGAN) is irrelevant —
VENA's task has aligned ground-truth T1c, so unpaired translation contributes nothing.

## Scope contract (the no-augmentation rule)

VENA owns the augmentation regime. The competitor never receives offline-augmented data
nor applies any augmentation of its own. The wrapper enforces this:

- `src/vena/competitors/pgan_cgan/dataset.py::UCSFPDGMSliceDataset` is deterministic by
  construction — repeat reads of the same index return byte-identical tensors. A unit
  test pins this: `tests/competitors/pgan_cgan/test_dataset.py::test_dataset_is_deterministic`.
- The upstream loader (`upstream/data/__init__.py::CreateDataset`) was vendored as-is
  and never invoked by VENA. The only data path used at runtime is our
  `UCSFPDGMSliceDataset`. The upstream's "augmentation" function (`base_dataset.py::get_transform`)
  was never called by pGAN even in the original code; we left it untouched.

## Code layout

```
src/external/pgan_cgan/                    # vendored snapshot — see UPSTREAM.md, PATCHES.md
├── upstream/                              # cloned at SHA b4ca7047
│   ├── pGAN.py                            # NOT invoked by VENA (we drive the model directly)
│   ├── models/{pgan_model,networks,base_model}.py
│   ├── data/__init__.py                   # patched for py3 only; not invoked
│   ├── options/{base,train,test}_options.py
│   └── util/{util,image_pool,visualizer}.py
├── UPSTREAM.md                            # vendoring snapshot info
└── PATCHES.md                             # exhaustive list of torch-2.x patches

src/vena/competitors/pgan_cgan/
├── dataset.py                             # UCSFPDGMSliceDataset (image H5 → 2D slices)
├── runner.py                              # imports the pGAN model, drives training, writes CSVs
└── inference.py                           # loads trained generator, synthesises 3D T1c, computes PSNR/SSIM

routines/competitors/pgan_cgan/
├── cli.py                                 # vena-competitor-pgan
├── engine.py                              # PGANCompetitorConfig + PGANCompetitorEngine
├── infer_cli.py                           # vena-competitor-pgan-infer
├── configs/
│   ├── smoke_server3_4ep.yaml             # 4 train patients × 4 epochs
│   ├── smoke_loginexa_4ep.yaml            # same recipe, V100 sm_70 + vena-v100 env
│   └── picasso_full.yaml                  # full fold-0 train, 50+50 epochs (paper recipe)
├── server3/launcher_pgan_server3_4ep.sh   # rsync + screen-detached launch
├── loginexa/launcher_pgan_loginexa_4ep.sh # rsync + tmux-detached launch on loginexa node
└── slurm/runs/{launcher,worker}_pgan_picasso_full.sh
```

## Patches to upstream (torch 2.x compatibility)

Full list with line refs lives in `src/external/pgan_cgan/PATCHES.md`. Summary:

1. `models/pgan_model.py`:
   - `cuda(..., async=True)` → `cuda(..., non_blocking=True)` (`async` is a py3 reserved word).
   - `Variable(...)` removed; `volatile=True` → `torch.no_grad()`.
   - `.data[0]` → `.item()` (five sites).
2. `models/networks.py`:
   - `init.normal`/`init.constant`/`init.xavier_normal`/`init.kaiming_normal`/`init.orthogonal`
     → in-place `_` variants (twelve sites).
   - `GANLoss.get_target_tensor`: drop `Variable`, mirror input device with `.to(input.device)`
     so the loss works on whichever GPU we pick (not just `torch.cuda.current_device()`).
3. `data/__init__.py`: py3 integer division for slice indexing; `range(...)` → `list(range(...))`
   for `random.shuffle`. Kept importable even though VENA bypasses the loader.

No numerics are changed. The model architecture, losses, optimiser, and training schedule
are byte-identical to the upstream behaviour on torch 0.3.

## Data contract

| Field | Value | Source |
|---|---|---|
| Input modalities | `{T1pre, T2, FLAIR}` (3 channels) | VENA cohort image H5s |
| Target | `T1c` (1 channel) | same H5 |
| Slice axis | Axial (z) | native shapes vary — (240, 240, 155) for UCSF-PDGM/UPENN-GBM/IvyGAP/REMBRANDT; (182, 218, 182) for BraTS-GLI/LUMIERE |
| Min brain voxels per slice | 1000 | drops mostly-air slices |
| Pad/crop size | 256 × 256 (centred zero-pad) | pGAN's stride-2 downsamples require ÷4 — handles both native shapes |
| Normalisation | per-patient `percentile_normalise(0, 99.5, foreground_only=True)` | `vena.common` |
| Intensity range fed to G | `[-1, 1]` (post-tanh) | rescale at end of `__getitem__` |
| Split | `splits/cv/fold_0/{train,val}` + `splits/test` | per-cohort image H5 |
| Corpus | VENA's per-platform `corpus_<host>.json` | shared with `routines/fm/train/configs/corpus/` |

**Multi-cohort = fair comparison.** VENA's reference run
(`picasso_s1_1000ep_fft.yaml`) trains on the union of every `role: cv` cohort in
`corpus_picasso.json` (UCSF-PDGM + BraTS-GLI + UPENN-GBM + IvyGAP + LUMIERE +
REMBRANDT, ≈1698 fold-0 train patients). The pGAN routine reads the same
registry via `MultiCohortImageSliceDataset` (`src/vena/competitors/pgan_cgan/dataset.py`)
and concatenates per-cohort `UCSFPDGMSliceDataset` instances through
`torch.utils.data.ConcatDataset`. Single-cohort mode (the legacy `image_h5`
field on `DataCfg`) is preserved for sanity smokes; production runs must
specify `corpus_registry` instead.

**Per-cohort schema robustness.** Two real heterogeneities across the VENA cv
corpus broke the first multi-cohort Picasso run; both are handled in
`UCSFPDGMSliceDataset` and pinned by tests in `tests/competitors/pgan_cgan/`:

1. **Longitudinal `/ids` vs cross-sectional splits** (BraTS-GLI, LUMIERE).
   The H5 stores scan-level identifiers in `/ids`
   (e.g. `BraTS-GLI-00000-000`, `Patient-001__week-000-1`) but the
   `splits/cv/fold_<k>/<phase>` arrays carry patient-level identifiers
   (`BraTS-GLI-00000`, `Patient-001`). The dataset first tries an exact
   match, then falls back to a prefix match (`split_id + "-"` or
   `split_id + "_"`) to resolve a patient_id to every scan_id that shares
   that prefix. A patient with N scans contributes N entries to
   `patient_indices`. Without this fix BraTS-GLI (815 patients,
   117 777 slices) and LUMIERE (64 patients, 54 599 slices) were silently
   dropped at run start. Test:
   `test_dataset_resolves_longitudinal_patient_ids`.

2. **Flat splits schema** (REMBRANDT). REMBRANDT is N=63 and uses a single
   53/5/5 split stored at `splits/{train,val,test}` — no nested CV, no
   `splits/cv/fold_<k>/...`. The dataset prefers the k-fold path but
   falls back to the flat path when k-fold is absent. Without this fix
   REMBRANDT was reported as "no fold-0 patients" and silently skipped.
   The cohort author's manifest explains the rationale ("N=63 too small
   for nested CV; mirrors IvyGAP" — actually IvyGAP got nested splits in
   a later pass, REMBRANDT didn't). Test:
   `test_dataset_falls_back_to_flat_splits_schema`.

A cohort whose split is genuinely empty, whose `image_h5` path doesn't
resolve on the current platform, or whose ids don't match by exact or
prefix is skipped with a WARNING (not raised) — the other cohorts still
load. Persistent inconsistencies show up in
`decision.json::data.corpus_registry` and the training log, never as
silent omissions.

**SWAN is intentionally absent** — none of the VENA cv cohorts store SWAN in
their image H5. The paper's original recipe used multi-contrast inputs anyway
(T1↔T2↔PD); 3 modalities → T1c is the direct analogue for the
gadolinium-synthesis task.

**Early stopping + best-G tracking** (`src/vena/competitors/pgan_cgan/runner.py`):
- The runner tracks epoch-mean `G_L1` and writes `best_net_G.pth` /
  `best_net_D.pth` on every improvement.
- `cfg.hp.patience > 0` enables early stopping on no-improvement for `patience`
  epochs; the runner logs `early stopping at epoch N` and breaks.
- `cfg.hp.max_epochs` is a hard cap orthogonal to pGAN's `niter + niter_decay`
  schedule — training stops at `min(max_epochs, niter + niter_decay)`. This is
  the analogue of VENA's `max_epochs: 10000 + patience: 100` recipe.
- Smokes set `patience: 0` to disable early stopping.

## Training contract

- LSGAN (no_lsgan=False) + L1 pixel loss × 100 + VGG16 perceptual loss × 100 + GAN × 1.
- ResNet generator (9 blocks, ngf=64, InstanceNorm, ReLU).
- PatchGAN discriminator (ndf=64, n_layers=3, InstanceNorm).
- Adam optimiser, lr=2e-4, β1=0.5.
- `niter` epochs constant LR + `niter_decay` epochs linear decay to 0.
- Smoke recipe: `niter=3` + `niter_decay=1` = 4 epochs total.
- Full recipe: `niter=50` + `niter_decay=50` = 100 epochs (Dar et al.'s demo recipe).
- One checkpoint per epoch: `{N}_net_G.pth`, `{N}_net_D.pth`, plus `latest_net_*.pth`.

## VGG cache

`torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)` requires 528 MB downloaded once.
Picasso compute nodes have no internet — pre-warm the cache on the login node via the
launchers (each launcher does this for you).

`TORCH_HOME` is set explicitly on every platform so the cached `vgg16-397923af.pth` lands
in a shared-filesystem path the compute node can reach.

## Per-platform recipe and validation

### Server-3 (ICAI workstation, RTX 4090)

- Launcher: `bash routines/competitors/pgan_cgan/server3/launcher_pgan_server3_4ep.sh`
- Path: rsync → screen-detached on `icai-server` GPU 0.
- Note: server-3 has only `screen` installed (no `tmux`); the launcher uses screen.
- **2026-06-14 run — server-3, run_id `2026-06-14T16-41-58_competitor_pgan_smoke_4ep_fc68720`**:
  - Wallclock: 162 s for 4 epochs (~40 s / epoch, 4 patients).
  - Loss curve descends: `G_L1 4.13 → 2.19`, `G_VGG 12.31 → 8.65`, `G_GAN 0.46 → 0.34`.
  - Artifacts present: `checkpoints/{1..4,latest}_net_{G,D}.pth`,
    `metrics/{train_step,train_epoch}.csv`, `decision.json` (`completed=true`).
  - Inference on 10 val patients: PSNR 15–21 dB, SSIM 0.90–0.93 (poor PSNR is expected —
    only 4 training patients × 4 epochs).

### Loginexa (Picasso V100-DGXS-32GB interactive node)

- Loginexa is NOT a SLURM partition; it is an SSH-accessible interactive node
  (10.248.7.200) reachable from Picasso login. `tmux` is available.
- Conda env: **`vena-v100`** (sibling of `vena`, torch 2.7.1+cu126 — V100 = sm_70, which
  prod `vena` env's torch 2.12+cu130 dropped).
- Launcher: `ssh picasso "cd <repo> && bash routines/competitors/pgan_cgan/loginexa/launcher_pgan_loginexa_4ep.sh"`
- The launcher pre-warms VGG on the login node, picks the freest of the 4 V100 GPUs by
  free-memory, and starts a detached tmux session on loginexa.
- 30-minute wallclock budget (convention from `.claude/loginexa.yaml`; not a hard kill).

### Picasso (A100 DGX, full training)

- Launcher: `ssh picasso "cd <repo> && bash routines/competitors/pgan_cgan/slurm/runs/launcher_pgan_picasso_full.sh"`
- Sbatch worker requests `--constraint=dgx --gres=gpu:1 --time=2-00:00:00 --mem=64G
  --cpus-per-task=8`.
- 50+50 epochs on the full UCSF-PDGM fold-0 train set on a single A100.
- Conda env: **`vena`** (torch 2.12+cu130, torchvision 0.27+cu130 — installed 2026-06-14).
- Pre-launch dry-run validated: `bash <launcher> --dry-run` resolves the sbatch command
  with the correct paths.

## Inference contract

`vena-competitor-pgan-infer` loads a trained checkpoint and synthesises 3D T1c volumes for
N patients of a chosen split.

```
vena-competitor-pgan-infer \
    --run-dir   /path/to/<run_id> \
    --image-h5  /path/to/UCSFPDGM_image.h5 \
    --epoch     latest \
    --n-patients 10 \
    --phase     val \
    --fold      0 \
    --gpu-id    0
```

Outputs (per patient) under `<run_dir>/inference/epoch_<epoch>/`:

- `<patient_id>_pred_t1c.nii.gz` — synthesised T1c, `(H, W, D)` float32 in `[0, 1]`.
- `<patient_id>_real_t1c_normalised.nii.gz` — the matched real T1c after the same
  percentile normalisation; lets us compute metrics consistently.
- `<patient_id>_midslice.png` — 3-panel axial: source T1pre / real T1c / pred T1c.
- `metrics.csv` — per-patient PSNR (masked over brain) and whole-volume SSIM.
- `summary.json` — schema 1.0 run-level summary.

The whole-slice tensor pipeline mirrors the training-time normalisation exactly —
percentile thresholds are recomputed per patient on the fly, then applied to each slice
before the generator forward pass. The output is rescaled `[-1, 1] → [0, 1]` and the
brain region is cropped back to the original `(240, 240, 155)` grid.

## Things to watch on the next competitor

1. **Vendoring approach**: clone the upstream into `src/external/<name>/upstream/`, strip
   `.pyc` and `.git`, write `UPSTREAM_SHA.txt` + `UPSTREAM.md` + `PATCHES.md`. Apply
   patches in-place; never monkey-patch at runtime.
2. **No-augmentation contract**: build a deterministic dataset; add a unit test that
   asserts byte-equality between two reads of the same index.
3. **Match VENA's training corpus exactly** for any fair-comparison run. Read the
   *same* `corpus_<host>.json` that VENA's FM trainer reads — never hand-roll a
   competitor-specific corpus. Single-cohort smokes are fine for sanity; production
   numbers go in `decision.json` only when `data.corpus_registry` was set.
4. **Match the channel dim contract**: pGAN expects `(B, C, H, W)` tensors with `C`
   matching `--input_nc`. Future 3D competitors would need a 3D dataset; future
   latent-space competitors would skip the percentile-normalise + tanh-rescale steps and
   read from a `*_latents.h5` instead.
5. **`num_workers=0` for multi-cohort dataloaders.** h5py + Python multiprocessing
   deadlocks when several workers open multiple H5 files concurrently via a
   `ConcatDataset`. We hit this on the first multi-cohort smoke (4 workers × 4 cohorts
   → 16 concurrent h5py handles → DataLoader hang at ~step 950, no Traceback). The
   smoke configs now ship with `num_workers: 0`; the Picasso full run keeps `8` because
   the per-cohort H5s live on the same Lustre mount and the deadlock is intermittent —
   if it ever surfaces in production, drop to `num_workers=2` or 0.
6. **Pre-warm any pretrained-weight caches** on each platform's login node before launching
   compute jobs. Picasso A100 nodes have no internet.
7. **Per-platform env paths**:
   - server-3: `~/.conda/envs/vena/bin/python` on icai-server.
   - loginexa: `/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/vena-v100/bin/python`.
   - Picasso A100: `/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/vena/bin/python`.
8. **Picasso SLURM requires `--constraint=dgx`** for A100 access — submitting without a
   constraint defaults to `cpu` and fails with "Requested node configuration is not
   available".
9. **The Picasso loginexa node is NOT a SLURM partition** — it's an SSH-accessible
   interactive node (10.248.7.200). Use tmux-detached SSH launches, not sbatch.
10. **`screen` vs `tmux`**: server-3 has only `screen`, loginexa has both. Sniff first
    or pick `screen` since it's more portable.

## Paired comparison axes (vs `picasso_s1_1000ep_fft.yaml`)

| Axis | VENA FM trainer | pGAN competitor | match |
|---|---|---|---|
| seed | 1337 | 1337 | ✅ |
| fold | 0 | 0 | ✅ |
| corpus | `corpus_picasso.json` (6 cv cohorts) | same | ✅ |
| max_epochs | 10 000 | 10 000 | ✅ |
| patience | 100 (on `train/total_epoch`) | 100 (on epoch-mean G_L1) | ✅ |
| save cadence | every 25 epochs | every 25 epochs | ✅ |
| batch_size (physical) | 4 | 4 | ✅ |
| num_workers | 8 | 8 | ✅ |
| walltime | 7d on A100 | 7d on A100 | ✅ |
| input modalities | `{T1pre, T2, FLAIR, …}` + masks | `{T1pre, T2, FLAIR}` (paper-faithful) | ⚠ |
| output | latent T1c via MAISI VAE | image-domain T1c (paper-faithful) | ⚠ |
| target metric | `mse_latent`/`bg`/`5 NFE` | epoch-mean G_L1 (for early-stop) | ⚠ |

The last three rows differ by design — adapting the competitor to VENA's latent
space, mask conditioning, or NFE-based metric would change its identity as a
baseline. We compare at the *application* level: synthesised image-domain T1c
on the same val/test patients, with the same PSNR/SSIM/LPIPS pipeline.

## Open follow-ups

- Run the full Picasso training (~24-36 h estimated wallclock) when the queue clears.
- After full training, run inference on the 50 patients of `splits/test` for the paper's
  results table.
- Optional: extend the inference module to compute the LPIPS metric (currently uses only
  PSNR + skimage SSIM).
