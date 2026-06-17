# Patches applied to `src/external/lpix2pix_3d/upstream/`

Patches are applied in-place at vendoring time. Runtime monkey-patching is
forbidden (skill anti-pattern 2).

## None at vendoring (2026-06-17)

Both vendored files (`train_pix2pix_t1n_t2f.py`, `test_pix2pix_t1n_t2f.py`)
are byte-identical to upstream SHA `fc8314f6`. The VENA wrapper at
`vena.competitors.lpix2pix_3d` does not import from these files at runtime
— it re-implements `GeneratorUNetWrapper` and `PatchDiscriminator3D` against
MONAI primitives directly (see UPSTREAM.md "What is invoked from VENA").

The vendored files are kept on disk for **reference and reproducibility
only** so a reviewer can audit:

1. The architecture and loss form against `src/vena/competitors/lpix2pix_3d/runner.py`.
2. The Isola *et al.* 2017 Pix2Pix recipe (BCE + λ·L1 with λ=100).
3. The Eidex *et al.* 2025 baseline configuration (channel-concat
   conditioning on MAISI-V2 latents).

## What is NOT modified

- `train_pix2pix_t1n_t2f.py` — preserved verbatim.
- `test_pix2pix_t1n_t2f.py` — preserved verbatim.

## Contingencies (not active by default)

### C1 — torch-API drift on a future env

`train_pix2pix_t1n_t2f.py:28` imports `from torch.amp import autocast,
GradScaler`. If a future torch version moves these symbols, the
**wrapper** code (not the vendored file) is updated — never patch the
vendored file. The vendored file's only role is reference.

## Verification

The vendored files are never executed by VENA's import graph or test
suite. Verification is satisfied if:

1. `git diff` against the cloned upstream tree at SHA `fc8314f6` is
   empty for both files.
2. `grep -r "from .upstream import"` under `src/vena/competitors/lpix2pix_3d/`
   returns no hits.
