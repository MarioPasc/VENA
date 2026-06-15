# ResViT upstream snapshot

| Field | Value |
|---|---|
| Repository | https://github.com/icon-lab/ResViT |
| Vendored at SHA | `f039733` |
| Vendored on | 2026-06-15 |
| Licence | Custom academic licence (see `upstream/LICENSE`) — academic / non-commercial research permitted; vendoring with attribution permitted. ResViT's own header reuses code from pGAN-cGAN (same group) and pix2pix / pytorch-CycleGAN (BSD-2-Clause). |
| Citation | Dalmaz, O.; Yurt, M.; Çukur, T. "ResViT: Residual Vision Transformers for Multimodal Medical Image Synthesis." *IEEE TMI* 41(10):2598-2614, October 2022. DOI: 10.1109/TMI.2022.3167808. arXiv:2106.16031v3. |

## Scope of use in VENA

Only the **two-stage curriculum** is used (the paper's flagship recipe):
1. **Stage 1 (CNN pretrain).** Train `Res_CNN` (residual-CNN encoder-decoder, no ART
   blocks) for `pretrain_niter + pretrain_niter_decay` epochs. Saved as
   `latest_net_G_pretrain.pt` and `latest_net_D_pretrain.pt`.
2. **Stage 2 (ART fine-tune).** Construct `ResViT` (same encoder-decoder with 9 ART
   blocks in the bottleneck), warm-start its CNN weights from the stage-1 generator
   (`pre_trained_resnet=1`), and load the ImageNet R50+ViT-B_16 transformer weights
   from `checkpoints/R50+ViT-B_16.npz` (`pre_trained_transformer=1`). Train for
   `niter + niter_decay` epochs. Saved as `best_net_G.pt` / `latest_net_G.pt`.

Both stages are driven by a single VENA `engine.run()` call — the second stage
launches automatically when the first finishes, in the same Picasso job. This is the
2026-06-15 user decision (see plan file
`/home/mpascual/.claude/plans/context-we-have-finished-snappy-tiger.md`).

VENA invokes **`models.resvit_one`** with `input_nc=3`, even for the many-to-one
task. Rationale below in the paper-vs-code table — `resvit_one` is the
input_nc-parametric path; `resvit_many` hardcodes 2-channel slicing despite the
README example invoking it with `--input_nc 3`.

The upstream code targets PyTorch ≥1.7.1 / Python ≥3.6.9. Minimal in-place patches
were applied to make it run on PyTorch 2.x — see `PATCHES.md`.

## Paper-vs-code incoherency table

| Axis | Paper (peer-reviewed) | Released code | VENA choice | Rationale |
|---|---|---|---|---|
| Many-to-one input channel count | "ART blocks aggregate variable source modalities via channel encoding" | `models/resvit_many.py` hardcodes `define_G(2, ...)` and slices `[:,0:2,:,:]` in `forward`/`backward_{D,G}` — strictly 2-source regardless of `--input_nc`. `models/resvit_one.py` correctly threads `opt.input_nc` end-to-end. | Use `resvit_one` with `input_nc=3` for `[T1pre, T2, FLAIR]→T1c`. | Follow paper text (VENA policy 2026-06-15). `resvit_one` is the channel-parametric path; `resvit_many` is structurally identical *except* for the hardcoded slicing, and using it would silently reduce the input to 2 channels. |
| ART block count | 9 (paper §III.A) | 9 (default `n_art_blocks` inside `residual_transformers.ResViT.__init__`) | 9 (no change) | Consistent. |
| Transformer config | "Base" (L=12, N_D=768, 12 heads); "Large" tested in ablation | `vit_name='Res-ViT-B_16'` default → `get_resvit_b16_config()` returns L=12, N_D=768, 12 heads, MLP=3072 | `vit_name='Res-ViT-B_16'` | Consistent. |
| MLP hidden | Paper text: 3073 (typo for 3072?) | 3072 in `transformer_configs.py` | 3072 | Paper text appears to be a typo; code value matches `R50+ViT-B_16.npz` checkpoint shape. |
| Pretrained ViT path | `R50+ViT-B_16.npz` from Google `vit_models/imagenet21k/` | Hardcoded `'./model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'` in `transformer_configs.py::get_resvit_b16_config` | Override at runtime by writing the absolute path into `residual_transformers.CONFIGS['Res-ViT-B_16'].pretrained_path` before `create_model(opt)`. Default is `src/external/resvit/upstream/checkpoints/R50+ViT-B_16.npz` (vendored). | Hardcoded relative path would break under VENA's `cwd`-independent invocation. |
| Loss recipe | $\mathcal{L}_{adv} + \lambda_{pix} \mathcal{L}_{L1}$; $\lambda_{pix}=100$ near-optimal across tasks | `lambda_A=100` (acts as `lambda_pix`); `lambda_adv=1`; `lambda_vgg=1` *but no VGG loss appears in `optimize_parameters`* | `lambda_A=100`, `lambda_adv=1`. Drop `lambda_vgg` (dead in code) | Avoid adding an unmotivated VGG term. |
| Two-stage curriculum | Stage 1: CNN-only pretrain (LR=2e-4, 50+50 epochs); Stage 2: ART fine-tune (LR=1e-3, 25+25 epochs) | README documents identical recipe; the engine entrypoint `train.py` is single-stage — the user runs `train.py` twice with different `--which_model_netG` and `--pre_trained_path` | Encoded as two calls inside VENA's `runner.py::train_resvit`. Stage 1 uses `which_model_netG='res_cnn'`; stage 2 uses `which_model_netG='resvit'` with `pre_trained_resnet=1`, `pre_trained_transformer=1`, and `pre_trained_path=<stage1 latest_net_G_pretrain.pt>`. | User-locked decision: single Picasso job. |
| LR schedule | Linear decay starting at `niter` | `lr_policy='lambda'` with linear decay over `niter_decay` epochs | Identical | Consistent. |
| Data loader | Paired 2D axial slices, 256×256, R+G+B encoded for many-to-one | `data/__init__.py::CreateDataset` expects `.mat` with `data_x` / `data_y` arrays | VENA bypasses entirely — uses `MultiCohortImageSliceDataset` returning the same `{'A', 'B', 'A_paths', 'B_paths'}` dict shape | Same pattern as pgan_cgan and syndiff. |
| Augmentation | Not described in paper recipe | `data/aligned_dataset.py::__getitem__` applies random crops and 50% horizontal flips | VENA's wrapper is deterministic — no augmentation by contract | VENA owns the augmentation regime; competitor wrappers do not augment (skill anti-pattern #1). |

## What is NOT modified

- The ResViT model architecture and ART blocks (`models/networks.py`,
  `models/residual_transformers.py`, `models/resvit_one.py`,
  `models/resvit_many.py`, `models/transformer_configs.py`).
- The training-options definitions (`options/{base,train,test}_options.py`).
  VENA constructs a `types.SimpleNamespace` that matches the same field names but
  does not invoke the upstream argparse parser.
- The `data/` directory. Preserved as documentation of the original `.mat`
  loader contract; never imported by VENA.

## Reproducing the snapshot

```bash
cd src/external/resvit
git clone --depth 1 https://github.com/icon-lab/ResViT.git upstream
cd upstream && git checkout f039733
rm -rf .git
# Apply patches listed in PATCHES.md in order.
mkdir -p checkpoints
curl -sSL -o checkpoints/R50+ViT-B_16.npz \
    https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz
sha256sum checkpoints/R50+ViT-B_16.npz > checkpoints/R50+ViT-B_16.npz.sha256
```

The `.npz` checkpoint (~330 MB) is rsync-excluded on first sync to server-3 /
loginexa (skill convention for upstream LFS-like payloads). It is downloaded
fresh on each platform that has internet access (server-3 directly, Picasso
login + loginexa via the launcher's pre-warm step before sbatch / tmux). The
compute nodes have no internet.
