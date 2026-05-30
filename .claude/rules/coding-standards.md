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
15. **`except` blocks name the exception types they expect.** Never `except Exception:` or bare `except:` in library code unless the body re-raises or logs at WARNING+ with the original exception. The acceptable forms are (a) `except (Foo, Bar) as exc: logger.debug("...", exc)` for a known-narrow swallow, (b) `except Exception as exc: logger.warning("..."); raise` for a last-ditch annotated re-raise. Bare `pass` after `except Exception` is forbidden — the bug that motivates this rule is a stripped import that silently dropped CSV columns until a real run surfaced it.
16. **No nested function definitions inside loops.** A helper defined inside a `for`/`while` is re-built every iteration and is invisible to unit tests. Lift it to module scope (use a leading underscore for module-private). The `_finite_mean` / `_finite_std` pattern in `src/vena/model/fm/lightning/module.py` is the canonical example.
17. **Module docstrings must match the implementation.** When changing a class's contract (e.g. flipping the trunk-trainable default, changing what is checkpointed), update the module docstring in the same change. A stale docstring counts as a bug under this rule — the reviewer is expected to flag it.
18. **External writes to a private attribute (`module._foo = ...`) are forbidden.** If a sibling engine needs to drive a `LightningModule`'s internal state, add a public method on the module (e.g. `compute_val_conditioning(batch)` for the validation conditioning cache). The module owns its own private state.
19. **`from __future__ import annotations` everywhere.** Use string-annotated forward references; put type-only imports under `if TYPE_CHECKING:` so runtime startup stays minimal. Ruff F821 false-positives on TYPE_CHECKING-only names are acceptable and may be suppressed with `# noqa: F821` when the runtime works.
20. **Stale-code hygiene.** Empty files, `__pycache__`-only directories, and `[project.scripts]` entries pointing at deleted modules are deletable on sight. Before deletion, `rg` the codebase to confirm no live caller — a deletion that breaks the build is a one-line revert, but a leftover stub that silently shadows a canonical implementation is a real bug.
