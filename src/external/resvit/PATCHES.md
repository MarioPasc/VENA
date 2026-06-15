# Patches applied to vendored ResViT upstream

All patches are applied in-place under `upstream/`. Each is the minimal change
needed to run the ResViT model on PyTorch 2.x (torch 2.5+, the version VENA
pins). No numerics are changed by these patches — only API drift fixes.

## Patch 1 — `upstream/models/networks.py`

### 1a. `init.normal`, `init.constant`, `init.xavier_normal`, `init.kaiming_normal`, `init.orthogonal` → in-place `_` variants

The non-in-place names were removed in torch 0.4. Only the underscore-suffixed
variants remain (`init.normal_`, `init.constant_`, `init.xavier_normal_`,
`init.kaiming_normal_`, `init.orthogonal_`). 13 call sites patched across
`weights_init_normal`, `weights_init_xavier`, `weights_init_kaiming`,
`weights_init_orthogonal`.

## Patch 2 — `upstream/models/resvit_one.py`

### 2a. `cuda(..., async=True)` → `cuda(..., non_blocking=True)`

`async` became a reserved keyword in Python 3.7. The CUDA tensor copy keyword
was renamed in torch 0.4. Two call sites in `set_input` (one each for
`input_A` and `input_B`).

## Patch 3 — `upstream/models/resvit_many.py`

### 3a. `cuda(..., async=True)` → `cuda(..., non_blocking=True)`

Identical to Patch 2 in the sibling many-to-one model file. Two call sites in
`set_input`.

VENA invokes `resvit_one`, not `resvit_many`, even for the many-to-one task —
see the paper-vs-code incoherency table in `UPSTREAM.md`. The patch is applied
anyway so that the file remains importable and the upstream snapshot stays
torch-2.x-clean.

## Patch 4 — `upstream/train.py`

### 4a. `from skimage.measure import compare_psnr as psnr` → `from skimage.metrics import peak_signal_noise_ratio as psnr`

`skimage.measure.compare_psnr` was removed in scikit-image 0.18 (released
2020-11). Replaced by `skimage.metrics.peak_signal_noise_ratio`. VENA does not
invoke `train.py` — its own `runner.py` drives the training loop — but the
patch keeps the file importable so the upstream snapshot is torch-2.x-clean.

## What is NOT patched (and why)

- **`from torch.autograd import Variable`** and `Variable(x)` wrapping in
  `models/resvit_{one,many}.py`. `Variable` is a no-op alias for `Tensor` since
  torch 0.4 — works without warning on torch 2.x.
- **`Image.BICUBIC` in `data/{aligned_dataset.py, aligned_dataset_old.py,
  base_dataset.py}`**. PIL ≥9.1 still accepts the legacy enum (with a
  `DeprecationWarning`). VENA does not import these modules — they are
  preserved for documentation purposes only.
- **`networks.py::print_network` Python 2-style print formatting**. Already
  syntactically valid in Python 3.
- **`util/visualizer.py` Visdom / dominate / HTML logging.** VENA's runner
  writes losses to CSV directly and never instantiates `Visualizer`.
- **`util/util.py::mkdirs`**. Used by `BaseOptions.parse`; VENA replaces
  options parsing with a `SimpleNamespace`, so `mkdirs` is unused. Preserved
  as-is — does not crash on import.
- **`models/test_model.py`**. Inference helper for the single-dataset mode;
  VENA's `inference.py` implements its own 3-D-volume stacker path.

## Pre-trained ViT path override (runtime, not patched)

`models/transformer_configs.py::get_resvit_b16_config` hardcodes
`pretrained_path = './model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'`, a
path relative to the upstream repo's old layout. **Patching this file
hardcodes a server-specific absolute path into the source tree**, which would
break the moment the repo is rsync'd to a different host (server-3 / loginexa
/ Picasso each have different mount points).

Instead, VENA overrides the field at runtime in `runner.py`, before calling
`models.create_model(opt)`:

```python
from external.resvit.upstream.models import residual_transformers
residual_transformers.CONFIGS["Res-ViT-B_16"].pretrained_path = str(cfg.vit_init_npz)
```

This is a public attribute on the `ml_collections.ConfigDict`; no monkey-patch
is involved. The same pattern is documented as the right move in
`.claude/skills/integrate-competitor/SKILL.md`'s Step-3 guidance against
runtime monkey-patching — config-dict mutation is not monkey-patching.
