---
name: refactor
description: Refactor Python modules for production quality (VENA)
---

# Python Module Refactoring

You are a world-class Python engineer. Refactor the target module following these rules strictly.

## Checklist (ALL must be satisfied)

- [ ] No hardcoded values — all config via OmegaConf / Pydantic / Hydra YAML.
- [ ] Type hints on every function signature and return type (Python 3.10+ syntax).
- [ ] Google-style docstrings on all public functions and classes.
- [ ] Shape assertions at tensor function boundaries.
- [ ] Atomic function design (one conceptual task per function).
- [ ] Library calls (MONAI, einops, torch.nn.functional, torchmetrics, scikit-image, SimpleITK) preferred over manual implementations.
- [ ] Logging via Python `logging` module — no `print()` in library code.
- [ ] Custom exceptions for domain-specific errors (never bare `Exception`).
- [ ] `@dataclass(frozen=True)` (or Pydantic `BaseModel`) for configuration containers.
- [ ] Numerical guard clauses (epsilon clamping where division occurs).
- [ ] If a new library is introduced, declared in `pyproject.toml` `[project.dependencies]` and installed in the `vena` env in the same change.
- [ ] Tensor handling discipline: `.detach()`, `torch.no_grad()`, `torch.cuda.empty_cache()` around inference paths on large 3D volumes.
- [ ] No 2D operations inside the core pipeline (see `coding-standards.md`, item 7).
- [ ] No edits under `src/external/` (other than `LINKS.md`) and no writes to checkpoint paths (see `external-deps.md`).

## Process

1. Read the target file completely.
2. Identify all violations of the checklist.
3. Propose the refactoring plan (what changes, in what order, and why).
4. Implement the refactoring.
5. Run: `~/.conda/envs/vena/bin/python -m py_compile <file>` to verify syntax.
6. Run: `~/.conda/envs/vena/bin/python -m pytest tests/ -x -q` if relevant tests exist.
7. Run: `~/.conda/envs/vena/bin/python -m ruff check <file>` and `ruff format <file>`.

$ARGUMENTS
