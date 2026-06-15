# SynDiff — VENA competitor integration log

## Scope

**Paper.** Özbey M., Dalmaz O., Dar S.U.H., Bedel H.A., Özturk Ş., Güngör A.,
Çukur T. "Unsupervised Medical Image Translation with Adversarial Diffusion
Models." *IEEE Transactions on Medical Imaging*, 2023. arXiv:2207.08208v3.

**Upstream repository.** <https://github.com/icon-lab/SynDiff>
SHA pinned at `fff3d8449e8c7ba38339be2f9ffd4aa5572beb4b` (short `fff3d844`).
Snapshot date 2026-06-15.

**Architecture.** Bilateral adversarial diffusion: 4 generators (2 diffusive
NCSN++ + 2 non-diffusive ResNet-6blocks) + 4 discriminators
(2 time-conditional `Discriminator_large` + 2 PatchGAN cycle discriminators).
At training time all 8 networks update jointly; only `gen_diffusive_1` is
used at inference.

**Training regime.** From scratch on VENA's UCSF-PDGM + BraTS-GLI multi-cohort
corpus, fold 0, seed 1337, mirroring `corpus_picasso.json`. Upstream pretrained
checkpoints (IXI T1↔PD, BraTS T1↔T2) are **not used** — fair comparison
requires same data regime.

**Modality scope.** Three-pair panel (mirrors `pgan_cgan`):
- `t1pre → t1c`
- `t2 → t1c`
- `flair → t1c`

The paper does NOT test contrast-enhanced synthesis. Any T1c number from
SynDiff is a published-architecture extrapolation; record this caveat in any
downstream report.

## Licence caveat

Top-level `LICENSE` is MIT. **Per-file headers diverge:**

- `backbones/ncsnpp_generator_adagn.py`, `backbones/discriminator.py`,
  `utils/EMA.py`: NVIDIA Source Code License — non-commercial research only
  (adapted from NVlabs `denoising-diffusion-gan` and `LSGM`).
- `backbones/generator_resnet.py`, `backbones/layerspp.py`, `backbones/layers.py`,
  `backbones/up_or_down_sampling.py`, `backbones/dense_layer.py`: Apache 2.0
  (Google Research `score_sde_pytorch`).
- `backbones/im2im.py`: BSD-2 (pytorch-CycleGAN-and-pix2pix fork).
- `utils/op/upfirdn2d.*`, `utils/op/fused_*`: MIT (StyleGAN2 fused ops).

The top-level MIT header does NOT override the per-file NVIDIA headers.
**Practical implication**: usable for research only. Any future commercial
VENA release that bundles SynDiff weights or backbone code needs a backbone
replacement or explicit NVIDIA grant. Flagged in `decision.json` under
`competitor.license_caveat`.

## Code layout

```
src/external/syndiff/
├── UPSTREAM.md         # repo URL, SHA, license per file, paper-vs-code coherency
├── UPSTREAM_SHA.txt    # short SHA — embedded in decision.json
├── PATCHES.md          # P1 (EMA torch-2.x), P2 (set_detect_anomaly removal), C1/C2 contingencies
├── __init__.py
└── upstream/           # frozen snapshot (.git stripped)

src/vena/competitors/syndiff/
├── __init__.py         # public surface
├── dataset.py          # SynDiffSliceDataset + MultiCohortSynDiffSliceDataset (2-tuple contract)
├── runner.py           # train_syndiff — 8 networks, DDP-stripped, VENA CSV logging
└── inference.py        # run_inference — loads best_gen_diffusive_1, 4-step reverse sampling

routines/competitors/syndiff/
├── cli.py              # vena-competitor-syndiff
├── engine.py           # SynDiffCompetitorConfig + SynDiffCompetitorEngine
├── infer_cli.py        # vena-competitor-syndiff-infer
├── configs/{smoke_server3_t1pre_4ep, smoke_loginexa_t1pre_2ep,
│            picasso_full_{t1pre,t2,flair}}.yaml
├── server3/launcher_syndiff_server3_t1pre_4ep.sh
├── loginexa/launcher_syndiff_loginexa_t1pre_2ep.sh
└── slurm/runs/{launcher,worker}_syndiff_picasso_full_{t1pre,t2,flair}.sh

tests/competitors/syndiff/
├── test_dataset.py     # determinism, longitudinal-id, flat-splits fallback
├── test_multicohort.py # concat, missing-cohort warn, path overrides
└── test_inference.py   # PSNR / SSIM math + crop
```

## Patches applied (`src/external/syndiff/PATCHES.md`)

| ID | File | Reason | Risk |
|---|---|---|---|
| P1 | `utils/EMA.py` | Torch ≥ 2.0: `Optimizer` base class introduced private hook OrderedDicts (`_optimizer_step_pre_hooks`, `_optimizer_state_dict_*_hooks`, `_optimizer_load_state_dict_*_hooks`). `EMA` never calls `super().__init__()`. Without the patch every `EMA(opt).state_dict()` call raises `AttributeError` (upstream issue #42). | Low — only adds empty hook dicts and `defaults` mirror. |
| P2 | `train.py` line 582 | Removed `torch.autograd.set_detect_anomaly(True)` (upstream issue #43). Our runner reimplements the training loop and never invokes `train.py`; patch kept for cleanliness so the vendored reference is loss-free. | None — runner-side reimplementation already excludes it. |

Contingencies documented but not enabled:
- C1: Pure-PyTorch fallback for `utils/op/upfirdn2d` and `fused_bias_act` if
  CUDA compile fails on a future toolkit. Monkey-patch `utils/op/__init__.py`
  to use `upfirdn2d_native` (slower, no CUDA build).
- C2: Skip `utils/utils.py` (hard TF dependency for `restore_checkpoint`).
  Our runner never imports it.

## Conda envs — user must build before running

The **runtime build of StyleGAN2 fused-CUDA ops** (`utils/op/upfirdn2d_kernel.cu`,
`fused_bias_act_kernel.cu`) requires `ninja` in the conda env. Adding these
deps to the main `vena` / `vena-v100` envs would pollute the FM trainer's
build cache. The user requested **dedicated isolated envs**:

### `vena-syndiff` — Picasso A100 + server-3 RTX 4090

```bash
# Picasso (run on login node, then copy the prefix to fscratch)
mamba create -p /mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/vena-syndiff python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/vena-syndiff

# A100-compatible torch (cu124 wheel works on Picasso DGX node driver as of 2026-06)
pip install torch==2.4.* torchvision --index-url https://download.pytorch.org/whl/cu124
pip install ninja h5py numpy scipy scikit-image nibabel rich pydantic omegaconf pyyaml pytest matplotlib

cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA
pip install -e ".[dev]" --no-deps     # --no-deps keeps the pinned torch
```

```bash
# Server-3 (run as user mariopascual)
mamba create -n vena-syndiff python=3.10 -y
conda activate vena-syndiff

# RTX 4090 (Ada, sm_89) — cu124 wheel
pip install torch==2.4.* torchvision --index-url https://download.pytorch.org/whl/cu124
pip install ninja h5py numpy scipy scikit-image nibabel rich pydantic omegaconf pyyaml pytest matplotlib

cd /home/mariopascual/projects/VENA
pip install -e ".[dev]" --no-deps
```

### `vena-v100-syndiff` — loginexa V100 sm_70

```bash
# On loginexa node
mamba create -p /mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/vena-v100-syndiff python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/vena-v100-syndiff

# cu121 — keeps sm_70 kernels (cu13 drops them)
pip install torch==2.4.* torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ninja h5py numpy scipy scikit-image nibabel rich pydantic omegaconf pyyaml pytest matplotlib

cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA
pip install -e ".[dev]" --no-deps
```

### First-run verification (StyleGAN2 fused-op build)

After env creation, before launching the smoke, verify the build succeeds:

```bash
# Server-3 / Picasso login node / loginexa — whichever you just installed
TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_extensions/vena-syndiff" \
    "$REMOTE_PYTHON" -c "
import sys
sys.path.insert(0, '$REPO_DIR/src/external/syndiff/upstream')
from utils.op import upfirdn2d, fused_act  # builds via ninja on first call
print('fused ops built OK')
"
```

A successful build leaves `.so` files under `${TORCH_EXTENSIONS_DIR}/upfirdn2d/`
and `${TORCH_EXTENSIONS_DIR}/fused/` — subsequent imports load them from
cache. **If the build fails** (cu mismatch, missing `python3.x-dev`), apply
contingency C1 in `PATCHES.md` and re-run.

## Data contract

- **Image-domain** (not latent). Reads `cohort.image_h5`, returns 2-tuple
  `(target_slice, source_slice)` per item, both shape `(1, 256, 256)` in
  `[-1, 1]`.
- Per-patient `percentile_normalise(lower=0, upper=99.5, foreground_only=True)`
  on the brain-stripped foreground (matches VENA's encoding convention).
- Centred zero-pad / centred crop to 256×256. Image size is **hard-baked**
  in the 6-level NCSN++ trunk (attn_resolutions `(16,)` requires
  `image_size % 32 == 0`).
- Deterministic — pinned by `test_dataset_is_deterministic`. No augmentation.
- Longitudinal `<patient>-<scan>` prefix-match resolver +
  flat-splits fallback (`splits/{train,val,test}` schema) — same semantics
  as the pGAN integration.

## Training contract

- Single-GPU only. Upstream's `DistributedSampler`, `broadcast_params`, and
  `DistributedDataParallel` wraps are stripped. README example uses
  `--num_process_per_node 1` so this matches the canonical invocation.
- 8 networks, 8 optimisers, 8 cosine LR schedulers. Diffusive-generator
  optimisers + non-diffusive-generator optimisers are EMA-wrapped via the
  patched `utils/EMA.py`.
- Loss: `errG = λ * errG_cycle + errG_adv + errG_cycle_adv + λ * errG_L1`
  with `λ = lambda_l1_loss = 0.5` (paper §IV.B). R1 gradient penalty on
  diffusive discriminators every `lazy_reg = 10` steps with `r1_gamma = 1.0`
  (README values).
- Per-step CSV at `metrics/train_step.csv`; per-epoch at `metrics/train_epoch.csv`.
- Best-epoch selection on `epoch G_L1 mean` (forward L1 vs GT on the target
  side). Periodic checkpoints every `save_epoch_freq` epochs; `latest_*`
  symlink-style overwrite on every epoch.
- Sentinel log line: `syndiff-train completed`.

## Per-platform recipe

### Server-3 (RTX 4090, vena-syndiff env)

```bash
# After creating vena-syndiff on server-3 per recipe above:
bash routines/competitors/syndiff/server3/launcher_syndiff_server3_t1pre_4ep.sh --dry-run   # plan
bash routines/competitors/syndiff/server3/launcher_syndiff_server3_t1pre_4ep.sh             # submit
ssh icai-server screen -ls                                                                  # confirm
ssh icai-server tail -F /media/hddb/mario/smoke_logs/competitors/syndiff/vena-syndiff-smoke.log
```

Wallclock expectation: 4 epochs × ~3-6 min/epoch ≈ 15-30 min on 1 patient/cohort
× 6 cohorts. Heavier than pGAN (8 networks per step vs 2).

### Loginexa (V100, vena-v100-syndiff env)

```bash
# After creating vena-v100-syndiff on loginexa per recipe above:
bash routines/competitors/syndiff/loginexa/launcher_syndiff_loginexa_t1pre_2ep.sh --dry-run
bash routines/competitors/syndiff/loginexa/launcher_syndiff_loginexa_t1pre_2ep.sh
ssh loginexa tmux ls
ssh loginexa tail -F /mnt/home/users/tic_163_uma/mpascual/execs/vena/logs/competitors/syndiff/vena-syndiff-loginexa-t1pre-smoke.log
```

### Picasso (A100, vena-syndiff env)

Three panel jobs — submit individually (or all three at once if the queue is
short). Authorise before submitting.

```bash
# t1pre → t1c
bash routines/competitors/syndiff/slurm/runs/launcher_syndiff_picasso_full_t1pre.sh --dry-run
bash routines/competitors/syndiff/slurm/runs/launcher_syndiff_picasso_full_t1pre.sh

# t2 → t1c
bash routines/competitors/syndiff/slurm/runs/launcher_syndiff_picasso_full_t2.sh --dry-run
bash routines/competitors/syndiff/slurm/runs/launcher_syndiff_picasso_full_t2.sh

# flair → t1c
bash routines/competitors/syndiff/slurm/runs/launcher_syndiff_picasso_full_flair.sh --dry-run
bash routines/competitors/syndiff/slurm/runs/launcher_syndiff_picasso_full_flair.sh
```

SLURM resource request: `--constraint=dgx --gres=gpu:1 --time=4-00:00:00
--mem=64G --cpus-per-task=8`. At 50 epochs and batch_size=1, expected
wallclock 24-48 h on A100 (8 networks per step, R1 penalty every 10 steps).

## Paired comparison axes (vs VENA FM run)

| Axis | Value | Match |
|---|---|---|
| Cohort registry | `corpus_picasso.json` | ✓ |
| Fold | 0 | ✓ |
| Seed | 1337 | ✓ |
| Modality target | t1c | ✓ |
| Source modality | t1pre / t2 / flair (panel) | one source per run (paper limit) |
| Image space | image-domain 2D | competitor-specific (paper limit) |
| Augmentation | none | ✓ (deterministic dataset) |
| Eval split | `splits/cv/fold_0/val` (smoke); `splits/test` (final) | ✓ |

## Things to watch

1. **StyleGAN2 fused-op build at first import.** Compilation takes ~30–60 s
   on first run. If it fails (CUDA mismatch, missing python3-dev), enable
   contingency C1 in `PATCHES.md` (pure-PyTorch fallback) — slower but no
   CUDA build.
2. **Non-converging diffusion / NaN losses.** Upstream issues #43 (NaN with
   `set_detect_anomaly`) and #53 (low PSNR on T1↔T2 IXI even with README
   command). Our runner removes `set_detect_anomaly` (P2). If NaN losses
   appear post-smoke, the next bisect candidates are: (a) `r1_gamma` too
   high for the cohort intensity range — try `0.05` per the train.py default
   instead of the README's `1.0`; (b) `lazy_reg` too sparse — try `5`;
   (c) AMP / mixed precision — currently disabled (the runner uses fp32).
3. **EMA swap-back bug.** When saving best/periodic, we swap EMA weights in,
   save, then swap back. If a crash happens between swap-in and swap-back,
   the next epoch trains on the EMA shadow. Acceptable for our budget;
   document if it shows up.
4. **8 networks → 8× checkpoint disk.** Each periodic save writes 4
   generator state-dicts (the 4 discriminators are not saved — they are
   discarded after training). At ~50 MB / NCSN++ generator + ~5 MB / ResNet
   generator, one tag ≈ 110 MB. With `save_epoch_freq=5` and 50 epochs →
   10 periodic tags → ~1.1 GB per run; ×3 panel runs → ~3.3 GB total.
   `experiments_root` on `fscratch` accommodates this comfortably.
5. **Best-epoch selection on epoch G_L1 mean.** Training loss is a proxy
   for synthesis quality, not the true objective. Per skill convention, the
   exhaustive eval routine (TBD) drives the actual ranking; `best_*.pth`
   is the safer-than-random starting point.
6. **Inference NIfTI affine is identity.** We don't carry per-patient
   spacing through the H5; downstream evaluation pairs predictions with the
   real T1c NIfTI by `patient_id`, so affine alignment isn't load-bearing.
   Flag if a downstream consumer assumes scanner-space affines.

## Open follow-ups

- Run server-3 smoke once `vena-syndiff` is built on icai-server.
- Run loginexa smoke once `vena-v100-syndiff` is built.
- Submit Picasso panel (t1pre / t2 / flair) after both smokes pass.
- After Picasso runs land, write a brief comparison vs `pgan_cgan` (PSNR /
  SSIM table, computational footprint). Not in scope for this integration.
