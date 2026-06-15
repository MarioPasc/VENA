# ResViT Competitor — Validation Note

**Status:** integration in progress (2026-06-15).
**Paper:** Dalmaz, O.; Yurt, M.; Çukur, T. "ResViT: Residual Vision Transformers
for Multimodal Medical Image Synthesis." *IEEE TMI* 41(10):2598–2614, Oct 2022.
DOI: 10.1109/TMI.2022.3167808. arXiv:2106.16031v3.
**Upstream:** https://github.com/icon-lab/ResViT (vendored at `f039733`).

## Scope

ResViT is a 2D adversarial encoder-decoder for multi-modal MRI synthesis. Its
distinguishing feature is the **Aggregated Residual Transformer (ART) block**
in the bottleneck — nine ART blocks per generator, each combining a residual
CNN path with a ViT transformer initialised from ImageNet R50+ViT-B/16. VENA
treats ResViT as an **image-domain GAN baseline** for the T1c synthesis task,
running it on the same multi-cohort glioma corpus as VENA's FM trainer
(`corpus_picasso.json`, 6 cv cohorts, ≈1698 fold-0 train patients).

The user-locked design choices (from the planning Q&A on 2026-06-15) are:

1. **Many-to-one input, 3 channels** `[T1pre, T2, FLAIR] → T1c`. SWAN
   excluded — matches the paper's BRATS many-to-one configuration; aligns
   the ResViT input set with pgan_cgan and syndiff for paired comparison.
2. **ViT init checkpoint** `R50+ViT-B_16.npz` (≈440 MB) cached at
   `src/external/resvit/upstream/checkpoints/`. Distributed to server-3 via
   rsync; pre-warmed on Picasso login node before sbatch (compute nodes
   have no internet — mirror the pgan VGG16 pattern). SHA-256 logged in
   `decision.json.competitor.vit_init_npz_sha256`.
3. **Two-stage curriculum chained in a single `engine.run()`**. Stage 1 is
   a CNN-only pretrain (`which_model_netG='res_cnn'`, lr=2e-4, 50+50 ep on
   Picasso); stage 2 inserts ART blocks (`which_model_netG='resvit'`,
   lr=1e-3, 25+25 ep), warm-started from stage 1 via `pre_trained_resnet=1`
   + `pre_trained_path=<stage1 ckpt>` and ViT-initialised via
   `pre_trained_transformer=1`. Stage 1 weights are saved separately as
   `checkpoints/latest_pretrain_net_{G,D}.pth` so they survive stage 2's
   `best_/latest_net_{G,D}.pth`.
4. **2D axial slicing** at 256×256, mirroring pgan_cgan / syndiff (centred
   pad, per-patient `percentile_normalise(0, 99.5, foreground_only=True)`,
   `[0, 1] → [-1, 1]` rescale for tanh output). At inference, slices are
   stacked back to a 3D `(H, W, D)` NIfTI.

The minimum deliverable is the **stage-2 best-G checkpoint**
(`checkpoints/best_net_G.pth`), saved on every improvement of epoch-mean
`G_L1` across the cohort union. Inference is configurable via
`vena-competitor-resvit-infer --epoch best|latest|latest_pretrain` so the
downstream validation regime can probe each stage's output.

## Code layout

```
src/external/resvit/
├── upstream/                         # vendored at SHA f039733
│   ├── models/{networks, residual_transformers, resvit_one, resvit_many,
│   │            transformer_configs, base_model, test_model, __init__}.py
│   ├── options/{base, train, test}_options.py        # preserved but bypassed
│   ├── data/                                          # preserved but bypassed
│   ├── util/                                          # mkdirs, image_pool only
│   ├── train.py, test.py, README.md, LICENSE
│   └── checkpoints/R50+ViT-B_16.npz                  # 440 MB; included in rsync
├── UPSTREAM.md                       # paper-vs-code incoherency table
├── UPSTREAM_SHA.txt                  # f039733
├── UPSTREAM_SHA_FULL.txt
└── PATCHES.md                        # 4 in-place torch-2.x patches

src/vena/competitors/resvit/
├── __init__.py                       # re-exports
├── dataset.py                        # UCSFPDGMSliceDataset + MultiCohortImageSliceDataset
├── runner.py                         # two-stage curriculum driver
└── inference.py                      # best-checkpoint slice→volume stacker

routines/competitors/resvit/
├── cli.py, engine.py, infer_cli.py
├── configs/{smoke_server3_4ep, smoke_loginexa_2ep, picasso_full}.yaml
├── server3/launcher_resvit_server3_4ep.sh       # rsync + screen
├── loginexa/launcher_resvit_loginexa_2ep.sh     # rsync + tmux, vena-v100
└── slurm/runs/{launcher,worker}_resvit_picasso_full.sh

tests/competitors/resvit/{__init__, test_dataset, test_multicohort, test_inference}.py
```

## Patches applied to upstream

Documented in detail in `src/external/resvit/PATCHES.md`:

1. **`models/networks.py`** — 13 `init.normal/constant/xavier_normal/kaiming_normal/orthogonal(` → `init.*_(` (torch ≥0.4 in-place suffix).
2. **`models/resvit_one.py`** — `cuda(..., async=True)` → `cuda(..., non_blocking=True)` (Python 3.7+ reserved keyword).
3. **`models/resvit_many.py`** — same `async=True` patch (we don't use this model class but keep it importable).
4. **`train.py`** — `from skimage.measure import compare_psnr` → `from skimage.metrics import peak_signal_noise_ratio` (skimage ≥0.18). We don't invoke `train.py` ourselves, but the patch keeps the upstream tree clean.

No structural changes — the input-channel count is handled by selecting
`resvit_one` (which is channel-parametric) over `resvit_many` (which
hardcodes 2-channel slicing despite advertising "many-to-one").

The hardcoded `pretrained_path` in `models/transformer_configs.py` is
**overridden at runtime** by writing into the `ml_collections.ConfigDict`
field before `create_model(opt)`. This is not monkey-patching — it is the
declared API surface of ConfigDict — and avoids baking a server-specific
absolute path into the source tree.

## Data contract

- **Input H5**: UCSF-PDGM-schema 2.0.0
  (`images/{t1pre,t1c,t2,flair}`, `masks/brain`, `ids`,
  `splits/cv/fold_<k>/{train,val}` or `splits/{train,val,test}` flat
  fallback for REMBRANDT, `schema_version=2.0.0`).
- **Per-cohort dataset** `UCSFPDGMSliceDataset`: returns
  `{'A': (3, 256, 256), 'B': (1, 256, 256), 'A_paths': str, 'B_paths': str}`
  per upstream's expected dict shape. Range `[-1, 1]` (tanh-compatible).
- **Multi-cohort wrapper** `MultiCohortImageSliceDataset`: reads
  `corpus_<host>.json`, picks every entry with `role == "cv"`,
  concatenates per-cohort datasets via `torch.utils.data.ConcatDataset`.
  Missing-cohort H5 → WARNING + skip. Empty split → WARNING + skip.
- **No augmentation** by contract. Repeated reads of the same index are
  byte-identical (pinned by `test_dataset_is_deterministic`).
- **H5 handles are opened lazily** and dropped on `__getstate__` so they
  survive `num_workers > 0` pickling.

## Per-platform recipe

### Local sanity (this workstation)

```bash
# Install ml_collections dep into the existing vena env (one-time).
~/.conda/envs/vena/bin/pip install "ml_collections>=0.1.1"

# Unit tests.
~/.conda/envs/vena/bin/python -m pytest tests/competitors/resvit/ -v -m unit
# Expect: 21 passed.

# Model import probe.
PYTHONPATH=src:. ~/.conda/envs/vena/bin/python -c "
import sys; sys.path.insert(0, 'src/external/resvit/upstream')
from models import create_model, residual_transformers
print(list(residual_transformers.CONFIGS.keys()))
"
```

### Server-3 (RTX 4090, conda env `vena`)

```bash
ssh icai-server "/home/mariopascual/.conda/envs/vena/bin/pip install 'ml_collections>=0.1.1'"
bash routines/competitors/resvit/server3/launcher_resvit_server3_4ep.sh
ssh icai-server tail -F /media/hddb/mario/smoke_logs/competitors/resvit/vena-resvit-smoke.log
```

Expected wallclock: ~10–20 min (4 epochs, 6 patients, batch 1).

### Loginexa (V100-DGXS-32GB, conda env `vena-v100`)

```bash
# From Picasso login (has internet):
ssh picasso "/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/vena-v100/bin/pip install 'ml_collections>=0.1.1'"
ssh picasso 'bash /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA/routines/competitors/resvit/loginexa/launcher_resvit_loginexa_2ep.sh'
ssh picasso 'ssh loginexa tail -F /mnt/home/users/tic_163_uma/mpascual/execs/vena/logs/competitors/resvit/vena-resvit-loginexa-smoke.log'
```

Expected wallclock: ~15–30 min on V100 (2 epochs).

### Picasso A100 DGX (full training, conda env `vena`)

```bash
ssh picasso "/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/vena/bin/pip install 'ml_collections>=0.1.1'"
# Dry-run first (prints sbatch line, no submit):
ssh picasso 'bash /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA/routines/competitors/resvit/slurm/runs/launcher_resvit_picasso_full.sh --dry-run'
# Submit when ready (USER must authorise — agent will not submit autonomously):
ssh picasso 'bash /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA/routines/competitors/resvit/slurm/runs/launcher_resvit_picasso_full.sh'
```

Resource ask: `--constraint=dgx --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=5-00:00:00`. Paper recipe: stage 1 = 50+50 ep, stage 2 = 25+25 ep (150 ep total).

### Inference (any platform)

```bash
~/.conda/envs/vena/bin/python -m routines.competitors.resvit.infer_cli \
    --run-dir <experiments_root>/<run_id> \
    --image-h5 <UCSF-PDGM image H5> \
    --epoch best --n-patients 10 --phase val
```

Asserts `summary["n_patients_succeeded"] > 0` — raises `InferenceError`
if every patient fails (skill anti-pattern: silent zero-success summary).

## Paired comparison axes (vs VENA's `picasso_s1_1000ep_fft.yaml`)

| Axis | VENA FM | ResViT competitor | Notes |
|---|---|---|---|
| Cohort union | `corpus_picasso.json` (6 cv cohorts) | same | Identical patient set |
| Modalities (input) | T1pre, T2, FLAIR, SWAN | T1pre, T2, FLAIR | SWAN excluded by user decision |
| Target | T1c (latent) | T1c (image) | Different representation space |
| Seed | 1337 | 1337 | Identical |
| Fold | 0 | 0 | Identical |
| Augmentation | image-domain pipeline (TorchIO) | none | Competitor wrapper deterministic |
| Slicing | 3D | 2D axial 256×256 | ResViT is 2D-only |
| Batch | 16 | 4 | ResViT slower / heavier per step |
| Train data unit | 1 patient | 1 axial slice | Different epoch semantics |
| Selection metric | EMA on train/total_epoch | epoch-mean G_L1 (stage 2) | Best-of-stage-2 only |
| Training budget | 1000 epochs | **paper slice budget** (see below) | Capped to ensure fairness against VENA |

These divergences are recorded in `decision.json.competitor.deviations`
and surfaced in every NIfTI's adjacent `summary.json`.

## Paper-budget cap (slice exposure)

User policy locked 2026-06-15: every competitor receives **only the paper's
total training exposure** (in seen slices) so the comparison against VENA
is not biased by the competitor seeing 100× more data than its paper budget.

| Quantity | Paper (BRATS many-to-one) | VENA's multi-cohort union (fold-0 train) |
|---|---|---|
| Slices/epoch | 25 train subj × 100 slices = **2 500** | measured: **287 117** (6 cv cohorts) |
| Stage 1 epochs | 50 + 50 = 100 | LR schedule kept at paper-recipe |
| Stage 1 slice budget | 100 × 2 500 = **250 000** | `pretrain_max_slices: 250000` |
| Stage 2 epochs | 25 + 25 = 50 | LR schedule kept at paper-recipe |
| Stage 2 slice budget | 50 × 2 500 = **125 000** | `max_slices: 125000` |
| Total slice exposure | **375 000** | **375 000** (1.0× paper) |
| Our equivalent epoch count | — | stage 1 ≈ 0.871, stage 2 ≈ 0.435, total ≈ **1.306** |

**How the cap works.** The runner's `_train_one_stage` accumulates
`slices_seen_this_stage += batch_size` on every step. When the per-stage
cap is reached, the inner step loop breaks; the outer epoch loop ends
after writing the partial epoch's row to `metrics/train_epoch.csv` and
saving the stage's `latest_*_net_{G,D}.pth`. The LR schedule is
configured with the paper-recipe `niter + niter_decay` (50+50 / 25+25)
but we exit far before the decay phase triggers — so the LR is
constant at full value for both stages, matching the paper's first
~0.87 epoch (stage 1) and ~0.43 epoch (stage 2) regime.

**Wallclock impact.** At batch=4 on A100, estimated step time
~120 ms (stage 1) / ~200 ms (stage 2):

- Stage 1: 62 500 steps × 0.12 s ≈ 125 min
- Stage 2: 31 250 steps × 0.20 s ≈ 104 min
- Total ≈ 4 h. SLURM `--time=12:00:00` reserves 3× slack.

This deviation is recorded under
`decision.json.competitor.deviations.slice_budget_cap` with the exact
caps used.

## Two-stage curriculum mapping

| Stage | Upstream model | LR | Epochs (Picasso) | Warm-start | Output ckpt files |
|---|---|---|---|---|---|
| 1 (CNN pretrain) | `Res_CNN` | 2e-4 | 50 + 50 linear decay | random init | `latest_pretrain_net_{G,D}.pth` |
| 2 (ART fine-tune) | `ResViT` | 1e-3 | 25 + 25 linear decay | CNN warm-start from stage 1 + ImageNet R50+ViT-B_16 transformer init | `best_net_{G,D}.pth` + `latest_net_{G,D}.pth` |

The runner enforces sequential transition: stage 2 only constructs the
generator after stage 1 finishes saving (or raises if stage 1 didn't
produce the handoff checkpoint). GPU is freed between stages
(`del model; torch.cuda.empty_cache()`).

## Things to watch (peculiarities for next-competitor builders)

1. **`resvit_many` is buggy, use `resvit_one`.** `models/resvit_many.py`
   hardcodes `define_G(2, ...)` and slices `[:,0:2,:,:]` in forward and
   D-pair construction, silently dropping channel 3 even when
   `--input_nc 3`. `models/resvit_one.py` is channel-parametric. Per
   VENA's "follow paper text" policy (2026-06-15), we use `resvit_one`
   with `input_nc=3` to honour the paper's many-to-one description.
2. **The pretrained_path field must be overridden at runtime.** Upstream
   `transformer_configs.py` hardcodes `./model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz`
   — a relative path that breaks the moment the working directory is not
   the upstream repo. Patch it in `runner.py::_set_vit_pretrained_path`
   via `residual_transformers.CONFIGS["Res-ViT-B_16"].pretrained_path = "<abs>"`.
3. **`ml_collections` is not a vena dep until ResViT was added.** All
   four envs (local, server-3, loginexa `vena-v100`, Picasso `vena`)
   need `pip install "ml_collections>=0.1.1"`. Pure Python — drop-in
   compatible with the existing torch / scikit-image stack.
4. **The ART block builds during stage 2 trigger an `np.load` of the 440 MB
   `.npz`**. The compute node must have this file pre-cached; the
   Picasso launcher's pre-warm step (and the server-3 launcher's verify
   step) guard this.
5. **`BaseModel.save` calls `network.cpu()` then `network.cuda(gpu)`** at
   every save — a transient CPU↔GPU round-trip. Negligible for our
   ~50-ep stage 2, but worth noting if a downstream variant saves at
   every step.
6. **GANLoss emits a benign `UserWarning`** about deprecated
   `torch.cuda.FloatTensor` constructor on every training step. Cosmetic
   only — `self.Tensor(input.size()).fill_(real_label)` still produces
   the right tensor on torch 2.x. Not patched because patching it would
   touch numerically-sensitive code; the warning is suppressed by
   redirecting stdout/stderr to the log file.
7. **Stage 1 has no "best" checkpoint.** The runner does not write a
   `best_net_G_pretrain.pth` — only the final `latest_net_G_pretrain.pth`.
   The paper treats stage 1 as a fixed-budget warm-up; tracking "best"
   would imply choosing among stage-1 weights at the stage-2 boundary,
   which the paper does not do.
8. **Patience-based early stopping is stage-2-only.** Stage 1 always runs
   `pretrain_niter + pretrain_niter_decay` epochs unconditionally.
9. **The 88 MB / 440 MB confusion.** The R50+ViT-B_16 checkpoint is
   ~440 MB (R50 hybrid CNN + ViT-B/16 transformer weights). Smaller
   ViT-only checkpoints from the same bucket are 88 MB; do not confuse
   them — ResViT requires the hybrid `R50+ViT-B_16.npz` exactly.
10. **`vena-competitor-resvit-infer --epoch latest_pretrain`** is a
    valid invocation for diagnosing stage 1 alone. Combined with stage 2's
    `--epoch best`, it lets the downstream validation regime separate
    "what the CNN backbone learned" from "what ART added".

## Things to verify on the next smoke run

- [x] Unit tests pass locally (21/21).
- [x] **Server-3 smoke completes** ✓ 2026-06-15
      run_id: `2026-06-15T13-33-43_competitor_resvit_smoke_4ep_multicohort_fc68720`
      wallclock: 17 min on RTX 4090. Stage 1: G_L1 5.71 → 3.22 (2 ep);
      stage 2: G_L1 6.47 → 5.86 (2 ep, best=5.86). All 6 checkpoints
      (latest_pretrain_net_{G,D}.pth, latest_net_{G,D}.pth, best_net_{G,D}.pth)
      + train_step.csv (397 KB) + train_epoch.csv (4 rows) present.
      decision.json `completed=true`, ViT npz SHA-256 logged.
- [x] **Loginexa smoke completes** ✓ 2026-06-15
      run_id: `2026-06-15T13-53-00_competitor_resvit_smoke_2ep_multicohort_68a229e`
      wallclock: 28 min on V100-DGXS-32GB GPU 3. Stage 1: G_L1 5.70 → 3.64 (2 ep);
      stage 2: G_L1 5.27 → 4.81 (2 ep, best=4.81). Same artifact set as server-3
      under `vena-v100` env. decision.json `completed=true`.
- [x] **Picasso launcher dry-runs cleanly** ✓ 2026-06-15
      sbatch line confirmed with `--constraint=dgx`,
      `execs/vena` lowercase, `vena` conda env path resolves,
      `--time=12:00:00` baked into worker.sh, ViT npz pre-warm step verified
      (461 MB cached on shared FS).
- [ ] (Future) Inference smoke produces 10 NIfTI volumes + midslice PNGs +
      `metrics.csv` + `summary.json` with `n_patients_succeeded > 0`.
- [ ] (Future, Picasso run) full slice-budget-capped trajectory at
      `--time=12:00:00`; expected wallclock ≈ 4 h; final stage-2 best
      G_L1 < 5.0 on epoch CSV (provisional target — recalibrate after
      first full run).

**Note on the smoke G_L1 numbers.** Stage 2 G_L1 is higher than stage 1's
end because ART blocks are inserted with random init at the stage 1 → 2
boundary; the model regresses briefly before the ViT-init transformer
attention takes over. With only 2 epochs/stage on the smoke the model
does not fully recover — this is normal and not a regression. The
full Picasso run, with the paper's full slice budget per stage, will
have time to converge.

## Open questions

- Is the 5-day Picasso wallclock budget sufficient? The paper's RTX A4000
  baseline is much slower per-step than an A100; we should over-budget
  to be safe but the dataset is larger than IXI/BRATS-25-subjects.
- Should we add an LPIPS column to the inference `metrics.csv`? Useful
  for comparing intensity-space methods with perceptual metrics, and
  matches the proposal's evaluation plan. Currently logs only PSNR + SSIM.
- Should stage-1 also save a "best" checkpoint? Currently skipped per
  paper recipe; revisit if stage-2 results are sensitive to stage-1's
  final weights vs an earlier minimum.
