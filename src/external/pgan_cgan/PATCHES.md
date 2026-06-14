# Patches applied to vendored pGAN-cGAN upstream

All patches are applied in-place under `upstream/`. Each is the minimal change needed
to run the pGAN model on PyTorch 2.x (torch 2.5+, the version VENA pins). No numerics
are changed by these patches — only API drift fixes.

## Patch 1 — `upstream/models/pgan_model.py`

### 1a. `cuda(..., async=True)` → `cuda(..., non_blocking=True)`

`async` became a reserved keyword in Python 3.7. The CUDA tensor copy keyword was
renamed in torch 0.4.

### 1b. Remove `Variable(...)` wrapping; replace `volatile=True` with `torch.no_grad()`

`Variable` is a no-op alias for `Tensor` since torch 0.4. `volatile=True` was removed
the same release. The `test()` method now wraps inference in `torch.no_grad()`.

### 1c. `.data[0]` → `.item()` in `get_current_errors`

Five call sites for loss scalar extraction.

## Patch 2 — `upstream/models/networks.py`

### 2a. `init.normal`, `init.constant`, `init.xavier_normal`, `init.kaiming_normal`, `init.orthogonal` → in-place `_` variants

The non-in-place names were removed in torch 0.4. Only the underscore-suffixed variants
remain (`init.normal_`, `init.constant_`, etc.).

### 2b. `GANLoss.get_target_tensor` — drop `Variable(...)`, mirror input device

`Variable(real_tensor, requires_grad=False)` → `real_tensor.to(input.device);
real_tensor.requires_grad_(False)`. The `.to(input.device)` is the substantive change:
the original code created the label on the default CUDA device (set by
`torch.cuda.set_device`), which works only when the model is on that exact device. The
patched form mirrors the input device so the loss works on whichever GPU the runner
chose (relevant on server-3 where the chosen GPU may not be cuda:0).

## Patch 3 — `upstream/data/__init__.py`

### 3a. Py3 integer division for slice indexing

`np.array(f['data_x']).shape[3]/2` (float in py3) → `// 2`; same for `opt.input_nc/2`
and `opt.output_nc/2`. Refactored to use `half_in = opt.input_nc//2` and
`half_out = opt.output_nc//2` for readability.

### 3b. `range(...)` → `list(range(...))` for `random.shuffle`

`random.shuffle` requires a mutable sequence; `range` is immutable in py3.

VENA does not call this loader, but the patches keep the upstream loader importable so
nothing breaks if a downstream consumer drops in a `.mat` file by the original recipe.

## What is NOT patched

- `torchvision.models.vgg16(pretrained=True)` (deprecated since torchvision 0.13). It
  still works with a `UserWarning`. VENA pre-downloads the VGG16 weights into a
  project-local `TORCH_HOME` (`~/.cache/torch/`) before any Picasso compute-node run,
  because compute nodes have no internet access. The runner emits a clear error if the
  cache is missing.
- The `cGAN.py` entrypoint and `models/cgan_model.py`. Unused by VENA.
- The `util/visualizer.py` Visdom-based logging. VENA bypasses this; its runner writes
  losses to CSV directly.
