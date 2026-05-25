---
name: explore
description: Deep codebase exploration for VENA (SWAN-conditioned latent FM for T1Gd synthesis)
---

# VENA Codebase Exploration

Thoroughly explore the VENA codebase to answer the query.

## Project at a glance

VENA — **V**essel **E**ncoded **N**eural **A**ugmentation — is a **gadolinium-free T1 post-contrast synthesis** project: given $\{T_{1\text{pre}}, T_2, \text{FLAIR}, \text{SWAN}\}$ plus a vessel mask $M_v$ (Frangi on SWAN) and a tumour mask $M_{\text{tum}}$, predict $\widehat{T_{1c}}$. Method: **3D conditional latent flow-matching** generator on top of the **MAISI-V2 VAE-GAN** latent space, with **ControlNet-style** conditioning on the masks. Training: UCSF-PDGM ($N{=}501$, GE 3 T). External validation: Hospital U. Regional de Málaga (multi-vendor, glioma + meningioma).

| Property | Value |
|---|---|
| Inputs (full config) | $T_{1\text{pre}}, T_2, \text{FLAIR}, \text{SWAN}, M_v, M_{\text{tum}}$ |
| Target | $T_{1c}$ |
| Native shape | ~`(240, 240, 155)`, isotropic 1 mm |
| Latent (MAISI-V2 VAE) | 4× spatial compression, 4 channels |
| Train cohort | UCSF-PDGM, 400 train / 50 val / 50 test (patient-level split) |
| External cohort | Hospital U. Regional de Málaga (in progress) |
| Conda env | `vena` |

## Pre-flight gates (must inspect before assuming any architecture decision)

Three pre-flight checks gate Phase-3 training. Each writes a `decision.json` under `artifacts/preflights/<name>/LATEST/`. **When the query touches architecture, pretrained models, vesselness, or training stages, read the relevant `decision.json` first** — do not infer from code or proposal alone.

| Check | Spec (in proposal) | Decision keys |
|---|---|---|
| `preflights/maisi_vae` | §3.4 — encoding through MAISI VAE-GAN | `vae_fine_tune`, `swan_ood_flag`, `latent_aug_safe`, `latent_scale[4]` |
| `preflights/vessel_mask` | §3.2 — vessel-mask extraction on SWAN | `vesselness_method` (frangi/jerman/nnunet), Dice, `passes_cmbb_rejection` |
| `preflights/shortcut_diag` | §6.5 — healthy-control diagnostic | `protocol_feasible`, `control_cohort_path` |

Dependency: pre-flights are independent and may run in parallel. All gate Phase-3 training (see `.claude/rules/training-stages.md`).

## Key locations

- `CLAUDE.md` (repo root) — hub of paths, conventions, quick commands, end-goal statement.
- `src/external/LINKS.md` — canonical external dependency paths (MAISI, UCSF-PDGM, Picasso mirror).
- `src/vena/` — library code; importable, unit-testable.
  - `common/maisi.py` (planned) — shared MAISI VAE encode/decode primitives.
  - `adapters/` (planned) — ControlNet head and other wrappers around external code.
  - `preflight/{vessel_mask,maisi_vae,shortcut_diag}/` (planned) — library implementations of each pre-flight.
  - `vessel/` (planned) — Frangi / Jerman / nnU-Net vesselness wrappers (scikit-image, MONAI, nnUNet).
  - `data/` (planned) — UCSF-PDGM H5 converter and loaders.
  - `flow/` (planned) — latent FM trainer, samplers (rectified-flow / Heun / Euler).
  - `eval/` (planned) — metrics, reader-study utilities, FastSurfer-LIT runner.
- `routines/` — CLI entrypoints (one YAML arg, thin engine wrappers).
  - `routines/preflights/{vessel_mask,maisi_vae,shortcut_diag}/`
  - `routines/{data,pipeline,training,eval,external,release}/<name>/` — phase routines, see `training-stages.md`.
- `artifacts/<routine>/<UTC-timestamp>/` — outputs: `report.md`, `figures/`, `tables/`, `decision.json`.
- `tests/` — pytest. Markers: `unit`, `preflight_maisi`, `preflight_vessel`, `preflight_aug`, `fm`, `controlnet`, `gpu`, `slow`.

## Documentation source-of-truth

External docs at `/media/mpascual/Sandisk2TB/research/vena/docs/`:

- `proposal.md` — full method, loss formulation, ablations, validation protocol, timeline.
- `literature.md` — CNN/GAN era → conditional diffusion → flow-matching/rectified-flow → architectural template (TumorFlow, MAISI) → vessel methods (Frangi, Jerman, shearlet, Vessel-CAPTCHA, DeepVesselNet).

## Critical conventions

- **3D throughout** — no 2D ops in the core pipeline. 2.5D LPIPS and axial-slice reader-study exports are the only exceptions, in clearly-labelled evaluation utilities.
- **Frozen models** (MAISI VAE) are never written to. Adapter wrappers go in `src/vena/adapters/`. The MAISI-V2 Flow-Matching checkpoint is kept as a reference (initialisation candidate); the project trains its own conditional FM head.
- **No unified H5 schema.** Each H5 artifact owns its layout but satisfies the principles in `.claude/rules/h5-design-principles.md`. The UCSF-PDGM image cache is the working example.
- **Conda env:** `vena`. Pytest: `~/.conda/envs/vena/bin/python -m pytest tests/ -v`.
- **Library code lives in `src/vena/`**, routines are thin wrappers. Never put implementation logic inside `routines/<name>/engine/`.

## Loss components (cheat sheet)

Total: $\mathcal{L} = \mathcal{L}_{\text{CFM}} + \lambda_v\,\mathcal{L}_{\text{vessel}} + \lambda_t\,\mathcal{L}_{\text{tum}}$, default $\lambda_v = \lambda_t = 0$ (proposal §4.4).

- $\mathcal{L}_{\text{CFM}}$ — conditional flow matching on latents (linear interpolant / rectified flow). Generator takes $[z_t, \text{enc}(T_{1\text{pre}}), \text{enc}(T_2), \text{enc}(\text{FLAIR}), \text{ControlNet}(M_v, M_{\text{tum}})]$ via additive ControlNet skip connections.
- $\mathcal{L}_{\text{vessel}} = \|M_v \odot (\widehat{T_{1c}} - T_{1c})\|_1$ — region-weighted L1 in pixel space, masks derived from the **target** $T_{1c}$ (not from SWAN, to avoid input–target shortcut, proposal §4.4).
- $\mathcal{L}_{\text{tum}} = \|M_{\text{tum}} \odot (\widehat{T_{1c}} - T_{1c})\|_1$ — analogous tumour-region term.
- Optional ablation: residual target $\Delta z = z_{T_{1c}} - z_{T_{1\text{pre}}}$ instead of direct $z_{T_{1c}}$ (proposal §4.2).

Inference: rectified-flow sampling, 1–10 Heun or Euler steps, EMA weights (decay 0.9999), <10 s/volume on A100.

$ARGUMENTS
