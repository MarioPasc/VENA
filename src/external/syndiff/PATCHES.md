# SynDiff — in-place patches applied to `upstream/`

Patches are listed in the order they were applied. Each block contains:
the file path, why, and the exact code change. Anyone re-vendoring the
snapshot (different SHA) must re-apply these before the runner / inference
modules will load.

## P1 — `utils/EMA.py`: torch-2.x optimiser hook attrs

**Why.** `EMA` subclasses `torch.optim.Optimizer` but never calls
`super().__init__(...)`. PyTorch ≥ 2.0 added several private hook attributes
to the `Optimizer` base class that are set in `__init__` — `state_dict()` /
`load_state_dict()` / `step()` machinery now reads them unconditionally. Under
torch 2.x, calling `EMA(opt, decay).state_dict()` raises
`AttributeError: 'EMA' object has no attribute '_optimizer_state_dict_pre_hooks'`
(upstream issue #42). Reported on every torch 2.x install.

**Fix.** After `self.param_groups = opt.param_groups` in `EMA.__init__`,
initialise the hook OrderedDicts to mirror `Optimizer.__init__` (we cannot
call `super().__init__()` cleanly because the parent expects `params` and
`defaults`, which the wrapper sources from `opt`). We also expose
`self.defaults = opt.defaults` so any external code that asks the wrapped
optimiser for its defaults sees the inner optimiser's values.

```python
# inserted at the end of EMA.__init__, just below self.param_groups = opt.param_groups
self.defaults = opt.defaults
self._optimizer_step_pre_hooks = OrderedDict()
self._optimizer_step_post_hooks = OrderedDict()
self._optimizer_state_dict_pre_hooks = OrderedDict()
self._optimizer_state_dict_post_hooks = OrderedDict()
self._optimizer_load_state_dict_pre_hooks = OrderedDict()
self._optimizer_load_state_dict_post_hooks = OrderedDict()
self._patch_step_function = lambda: None
```

`OrderedDict` is added to the imports at the top of the file.

## P2 — `train.py`: drop `set_detect_anomaly(True)` (hygiene, not load-bearing for us)

**Why.** Line 582 of `train.py` hard-codes
`torch.autograd.set_detect_anomaly(True)` inside every iteration of the
training loop. This is debugging-only — it disables several autograd
optimisations and triggers NaN-detection asserts that crash the run when a
single intermediate tensor is non-finite (upstream issue #43, also reported by
several users in #44 and #51). Our runner under
`src/vena/competitors/syndiff/runner.py` reimplements the loop and does not
call this function. The patch is applied only so the vendored `train.py`
remains a clean reference for anyone re-validating it directly.

**Fix.** Delete the line `torch.autograd.set_detect_anomaly(True)` (line 582
of the snapshot). No semantic change to our runner — we never invoke this
file.

## Contingencies (not applied by default)

### C1 — StyleGAN2 fused ops fall back to pure-PyTorch

If `utils/op/upfirdn2d.cpp` or `fused_bias_act.cpp` fail to compile against
the platform's CUDA toolkit at first import (most likely on Picasso if the
A100 module's CUDA mismatches the conda env's torch wheel), monkey-patch
`utils/op/__init__.py` to alias the fused entrypoints to the pure-PyTorch
reference implementations already in `utils/op/upfirdn2d.py` (function
`upfirdn2d_native`). Slower (≈2-4× per layer) but builds nowhere. Document
the activation in `.claude/notes/validation/syndiff.md` when used.

### C2 — drop the `utils/utils.py` TensorFlow dependency

The vendored `utils/utils.py` carries `import tensorflow as tf` at module top
(for a `tf.io.gfile`-backed checkpoint helper neither our runner nor
inference modules call). Our wrappers import only `from utils.EMA import EMA`
— never `utils.utils` — so the TF import is never triggered. If a future
patch needs to import anything else from `utils/`, gate or remove the TF
import; do not add `tensorflow` to the conda env.
