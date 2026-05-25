# Coding Standards

Project conda env: `vena`. All Python invocations use `~/.conda/envs/vena/bin/python`.

1. **Type hints** on ALL function signatures and return types. Python 3.10+ syntax (`X | None`, `list[int]`, `tuple[str, ...]`).
2. **Google-style docstrings** on all public functions and classes. NumPy-style is also acceptable when the function is mathematical (state `Parameters`, `Returns`, `Raises`).
3. **Brief inline comments** on non-obvious code only. Explain *why*, never *what*.
4. **Logging** via Python `logging` module with `rich` handler. INFO for training events, DEBUG for shapes/values. Never `print()` in library code; `print()` is acceptable only in scripts and notebooks.
5. **No magic numbers** — all hyperparams from YAML configs via OmegaConf or Pydantic. Hyperparams that crossed routine boundaries must round-trip into the produced artifact's attrs.
6. **Libraries-first**. Prefer well-maintained external libraries (MONAI, einops, torchmetrics, nibabel, SimpleITK, lpips, piq, h5py, scikit-image, hydra-core) over hand-rolled implementations — they are tested, peer-reviewed, and designed for the task. When adding a new dependency:
   - (a) declare it in `pyproject.toml` `[project.dependencies]` with a version pin,
   - (b) `~/.conda/envs/vena/bin/pip install -e .` (or `pip install <pkg>`) in the same change,
   - (c) state the rationale in the commit body (one sentence: which task it solves and why the project should not roll its own).
   Never `pip install` ad-hoc without updating `pyproject.toml`.
7. **3D throughout the core pipeline.** MAISI VAE, the FM generator, and ControlNet conditioning all operate on 3D tensors. No 2D-slice operations except in clearly-labelled evaluation utilities (e.g. 2.5D LPIPS aggregation, axial-slice reader-study export).
8. **Frozen pretrained models are immutable.** Never edit code under `src/external/` and never write to checkpoint paths declared in `src/external/LINKS.md`. Adapter wrappers (e.g. ControlNet head built around the MAISI VAE) live in `src/vena/adapters/`. See `external-deps.md`.
9. **Tests use pytest** under the `vena` env:
   ```
   ~/.conda/envs/vena/bin/python -m pytest tests/ -v
   ```
   Markers: `unit`, `preflight_maisi`, `preflight_vessel`, `preflight_aug`, `fm`, `controlnet`, `gpu`, `slow`.
10. **Keep functions atomic** — one conceptual task per function. Cyclomatic complexity stays low; extract helpers rather than nesting `if`/`for` beyond two levels.
11. **Shape contracts.** Functions that consume or return tensors must either type-hint their shapes in the docstring (e.g. `x: Float["B C H W D"]`) or assert them with `einops.rearrange`-style pattern checks at function boundaries. State the expected shape upfront when refactoring.
12. **Custom exceptions per module** (e.g. `class PreflightError(Exception): ...`, `class H5SchemaError(Exception): ...`, `class VesselMaskError(Exception): ...`). Library code never raises bare `Exception`.
13. **Numerical guard clauses.** Epsilon clamping where division occurs (`x / (denom + 1e-8)`). Explicit `torch.no_grad()` / `.detach()` / `torch.cuda.empty_cache()` around inference paths that handle large 3D volumes.
14. **One class per file** when the class exceeds ~100 lines. Submodule organization under a namespace package (`src/vena/<area>/<name>.py`).
