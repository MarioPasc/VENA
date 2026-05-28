# VENA

> **V**essel **E**ncoded **N**eural **A**ugmentation — SWAN-conditioned latent flow matching for gadolinium-free synthesis of T1 post-contrast brain MRI.

## End goal

Replace the gadolinium-based contrast agent in brain-tumour follow-up MRI with a generative model that synthesises T1 post-contrast ($\widehat{T_{1c}}$) from a pre-contrast multimodal input plus a **vessel prior $M_v$ extracted from SWAN/SWI** and a tumour-shape prior $M_{\text{tum}}$. The model is a **conditional latent flow-matching** generator on top of the **MAISI-V2 VAE-GAN** latent space, with a **ControlNet-style** branch that injects the vessel and tumour masks. Internal validation on UCSF-PDGM ($N{=}501$, GE 3 T), external validation on a Málaga in-house cohort (multi-vendor, glioma + meningioma). Target deliverable: MICCAI 2026 or *MedIA*/*IEEE TMI* journal submission.

The differentiator versus prior work (Kleesiek 2019, Preetha 2021, McCaD 2024, CFM 2025, T1C-RFlow 2025, TumorFlow 2026) is **explicit vessel-aware conditioning via SWAN**, in latent space, with vessel-resolved evaluation — not just whole-volume PSNR/SSIM.

## Documentation source-of-truth

| Asset | Path |
|---|---|
| Proposal (method, losses, evaluation, ablations, timeline) | `/media/mpascual/Sandisk2TB/research/vena/docs/proposal.md` |
| Literature review (CNN/GAN → diffusion → flow-matching → vessel methods) | `/media/mpascual/Sandisk2TB/research/vena/docs/literature.md` |
| External code, checkpoints, datasets — canonical paths | `src/external/LINKS.md` |
| Project rules (enforced) | `.claude/rules/` |

Whenever this file drifts from the proposal, **the proposal wins**.

## Project rules

Read these before non-trivial work:

- [`.claude/rules/coding-standards.md`](.claude/rules/coding-standards.md) — Python conventions, env, testing, library-first policy.
- [`.claude/rules/preflight-pattern.md`](.claude/rules/preflight-pattern.md) — Routine layout (`routines/<bucket>/<name>/`), thin engine over `src/vena/` library code, `decision.json` contract.
- [`.claude/rules/h5-design-principles.md`](.claude/rules/h5-design-principles.md) — Schema-versioned, self-describing HDF5 artifacts (applies to the UCSF-PDGM cache at `UCSFPDGM_image.h5`).
- [`.claude/rules/training-stages.md`](.claude/rules/training-stages.md) — Six-phase timeline (data → pipeline → training → internal val → external val → writing) and the canonical routine names that map onto it.
- [`.claude/rules/external-deps.md`](.claude/rules/external-deps.md) — How to consume frozen MAISI-V2 weights and UCSF-PDGM data; what `src/external/` is and is not.
- [`.claude/rules/model-coding-standards.md`](.claude/rules/model-coding-standards.md) — FM-generator conventions (`src/vena/model/fm/`): training-only module + offloaded validation, metric-CSV logging, EMA/grad-accum, intensity-space parity, async second-GPU exhaustive validation.

## Quick context

| Property | Value |
|---|---|
| Modality (input core) | $T_{1\text{pre}}$, T2, FLAIR, SWAN |
| Modality (target) | $T_{1c}$ (post-contrast) |
| Conditioning priors | Vessel mask $M_v$ (Frangi on SWAN, soft); tumour mask $M_{\text{tum}}$ (BraTS-style segmenter) |
| Native shape (UCSF-PDGM, isotropic 1 mm) | ~`(240, 240, 155)` after skull-strip / co-registration |
| Latent (MAISI-V2 VAE) | 4× spatial compression, 4 channels |
| Generator | Latent FM (rectified-flow / linear interpolant), DiT or U-Net trunk |
| Conditioning route | ControlNet branch on masks; encoder for $T_{1\text{pre}}$ |
| Loss (default) | $\mathcal{L}_{\text{CFM}}$ (latent); ablate $+ \lambda_v \mathcal{L}_{\text{vessel}} + \lambda_t \mathcal{L}_{\text{tum}}$ on decoded output |
| Inference | Rectified-flow sampling, 1–10 Heun/Euler steps; <10 s/volume on A100 |
| Train cohort | UCSF-PDGM, 400 train / 50 val / 50 test (patient-level split) |
| External cohort | Hospital U. Regional de Málaga (glioma + meningioma, multi-vendor) |
| Conda env | `vena` (Python ≥3.11) |

## Layout

```
VENA/
├── CLAUDE.md                       # this file
├── pyproject.toml                  # deps, ruff, pytest, mypy, console scripts
├── src/
│   ├── external/LINKS.md           # canonical external paths (MAISI, UCSF-PDGM, Picasso mirror)
│   └── vena/                       # library code: importable, unit-testable
│       ├── common/maisi.py         # (planned) MAISI encode/decode primitive
│       ├── adapters/               # (planned) wrappers around external code (never edit src/external/)
│       ├── preflight/              # (planned) library impls of pre-flights
│       ├── data/                   # (planned) UCSF-PDGM H5 converter + loaders
│       ├── vessel/                 # (planned) Frangi / Jerman / nnU-Net vesselness wrappers
│       ├── flow/                   # (planned) latent FM trainer, samplers
│       └── eval/                   # (planned) metrics, reader-study utilities
├── routines/                       # CLI entrypoints — one YAML arg, thin engine
│   ├── preflights/                 # gating checks (vessel mask QC, MAISI audit, etc.)
│   └── <phase>/                    # phase routines per training-stages.md
├── artifacts/<routine>/<UTC>/      # report.md, figures/, tables/, decision.json
├── tests/                          # pytest, markers per pyproject.toml
└── .claude/
    ├── rules/                      # project-wide rules (above)
    ├── skills/                     # /explore /test /dl-scientist /refactor
    └── hooks/                      # session hooks (compact-context.sh)
```

## Conda env and quick commands

```bash
# Activate
source ~/.conda/envs/vena/bin/activate           # or: conda activate vena

# Install (CUDA wheel selected at install time — see pyproject.toml notes)
~/.conda/envs/vena/bin/pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cu124  # RTX 4060
~/.conda/envs/vena/bin/pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cu121  # RTX 3060
# Picasso A100 inside NGC Singularity: torch already in image, use --no-deps then install non-torch deps.

# Test
~/.conda/envs/vena/bin/python -m pytest -m "not slow and not gpu" -v --tb=short
~/.conda/envs/vena/bin/python -m pytest -m "fm and gpu" -v

# Lint / format
~/.conda/envs/vena/bin/python -m ruff check src/ routines/ tests/
~/.conda/envs/vena/bin/python -m ruff format src/ routines/ tests/
```

## Hardware

- Local A: RTX 4060 8 GB (Debian 12, CUDA 12.4). For development, smoke runs, light visualisation.
- Local B: RTX 3060 12 GB (CUDA 12.2). Larger smoke batches.
- Picasso (UMA HPC): SLURM, **Singularity only** (no Docker), 8× A100 40 GB per node (effective ~39 GB). Full training target: ~5 days on 4× A100 (per proposal §5).

See `src/external/LINKS.md` for the Picasso mirror of MAISI weights and the UCSF-PDGM H5 cache.

## Conventions in one line each

- **Library code in `src/vena/`, routines in `routines/<bucket>/<name>/` are thin engines** — see `.claude/rules/preflight-pattern.md`.
- **Frozen pretrained models are immutable** — never edit `src/external/*` (other than `LINKS.md`), never write to checkpoint paths.
- **Pre-flights are gating** — downstream routines load `decision.json` and assert at startup.
- **3D throughout** — no 2D ops in the core pipeline; 2.5D only in clearly-labelled evaluation utilities.
- **YAML-driven** — every hyperparameter through OmegaConf/Pydantic; round-trip into the produced artifact.
- **Conventional commits**, no force-push, no co-author trailer.

## Open questions tracked in `DECISIONS.md`

This file does not exist yet; it will be created the first time a non-trivial architectural decision is made (vesselness method choice, residual-vs-direct target parameterisation, ControlNet skip-connection scheme, etc.). One entry per decision: date, options considered, choice, rationale, reversibility.
