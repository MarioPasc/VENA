# 3D-DiT upstream snapshot

| Field | Value |
|---|---|
| Repository | https://github.com/zacheidex/An-Efficient-3D-Latent-Diffusion-Model-for-T1-contrast-Enhanced-MRI-Generation |
| Vendored at SHA | `fc8314f60d877f9ee55996f960f89b17b269200f` |
| Vendored on | 2026-06-15 |
| Licence | **None at HEAD `fc8314f6`** — no LICENSE file at the repository root. Default copyright (all-rights-reserved) applies. Vendored under assumed academic-use intent of the arXiv preprint; awaiting an explicit MIT/Apache-2.0 grant from the authors. The DiT-3D model definition (`dit3d.py`) is itself adapted from Meta Platforms' DiT (Peebles & Xie 2023) per the copyright header inside the file. |
| Method paper | Peebles, W., & Xie, S. "Scalable Diffusion Models with Transformers." *ICCV 2023*. arXiv:2212.09748. (DiT backbone.) |
| Method paper (3D) | Mo, S., Xie, E., Chu, R., Hong, L., Niessner, M., & Li, Z. "DiT-3D: Exploring Plain Diffusion Transformers for 3D Shape Generation." *NeurIPS 2023*. arXiv:2307.01831. (3D variant of DiT.) |
| Baseline reference | Eidex, Z. *et al.* 2025, "An Efficient 3D Latent Diffusion Model for T1-contrast Enhanced MRI Generation," arXiv:2509.24194 — §4 uses DiT-3D as the transformer-backbone diffusion baseline. |

## Why a separate vendoring from `src/external/t1c_rflow/`

The same upstream repository (Eidex *et al.* 2025) ships both:

1. A rectified-flow U-Net trainer (`train_rflow.py`) — vendored at
   `src/external/t1c_rflow/upstream/` and consumed by the
   `vena.competitors.t1c_rflow` wrapper.
2. A DiT-3D backbone (`dit3d.py`, `dit3d_wrapper.py`, `test_dit.py`) —
   referenced by the T1C-RFlow paper §4 as the "transformer-backbone
   diffusion" baseline. **This** snapshot vendors only those three files.

Each competitor's vendored snapshot is independent so deleting one cannot
break the others (skill anti-pattern 7). The two snapshots are byte-identical
at SHA `fc8314f6` for the DiT-3D files; we keep them duplicated rather than
cross-importing.

## Citation

Cite the DiT-3D row of VENA's competitor table with the following BibTeX
(Peebles & Xie for the DiT backbone, Eidex *et al.* for the 3D adaptation +
MAISI-latent training recipe we follow):

```bibtex
@inproceedings{peebles2023scalable,
  title     = {Scalable Diffusion Models with Transformers},
  author    = {Peebles, William and Xie, Saining},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2023},
  eprint    = {2212.09748},
  archivePrefix = {arXiv}
}

@article{eidex2025efficient3dlatentdiffusion,
  title   = {An Efficient 3D Latent Diffusion Model for T1-contrast Enhanced MRI Generation},
  author  = {Eidex, Zach and Safari, Mojtaba and Ding, Jie and Qiu, Richard
             and Roper, Justin and Yu, David and Shu, Hui-Kuo and Tian, Zhen
             and Mao, Hui and Yang, Xiaofeng},
  journal = {arXiv preprint arXiv:2509.24194},
  year    = {2025}
}
```

## Scope of use in VENA

The VENA wrapper (`src/vena/competitors/dit_3d/`) imports the `DiT3DWrapper`
class **from this vendored snapshot at runtime** — unlike T1C-RFlow which
rebuilds everything from MONAI primitives. Reason: `DiT3DWrapper` is a 10-line
shim around the local `DiT` class; reproducing it in `vena.competitors`
would duplicate the file with no architectural change.

The wrapper:

- Trains DiT-3D over VENA's frozen MAISI-V2 latents
  (`vena.common.load_autoencoder`).
- Uses VENA's RFlow scheduler primitives
  (`monai.networks.schedulers.rectified_flow.RFlowScheduler` — same as the
  T1C-RFlow wrapper) with the paper-pinned kwargs.
- Conditions by channel-wise concat of the T1pre + FLAIR latents to the
  noisy T1c latent: `x = [z_t, z_T1pre, z_FLAIR]` along channel axis,
  giving `in_channels = latent_channels × (1 + len(cond)) = 12`. This
  matches T1C-RFlow's `train_rflow.py:129-202` and is the same
  conditioning route the paper's DiT-3D baseline uses (`test_dit.py:148`).
- Loss is the L1 velocity loss `F.l1_loss(v_pred, z_T1c − z_noise)` — same
  as T1C-RFlow's `train_rflow.py:207`.

## Differences between method-paper text and vendored code (incoherencies)

Following VENA policy (2026-06-15): when paper text and code disagree, we
**follow the peer-reviewed paper text**. The wrapper's defaults below are
chosen to match each method paper's stated configuration.

| # | Axis | Peebles & Xie 2023 (DiT paper) | Eidex 2025 (used as 3D-DiT baseline) | Vendored code | Wrapper default | Load-bearing? |
|---|---|---|---|---|---|---|
| 1 | Backbone params | DiT-B = `depth=12, hidden=768, num_heads=12, patch=2` (~130M for 2D); DiT-XL = `depth=28, hidden=1152, ...` | Eidex §4 cites DiT-3D as a baseline but does not specify size | `DiT3DWrapper` exposes any kwargs; `test_dit.py:181` does not pin them either | **DiT-B/4 in 3D** — patch=4, depth=12, hidden=768, num_heads=12 — paper-standard "base" model; patch=4 chosen because our latent grid is `(48,56,48)` and `48/4=12`, `56/4=14` (cleanly divisible) | LOW — model size is a recognised tunable; DiT-B is the paper's standard non-XL configuration |
| 2 | Scheduler | DDPM (paper §2.1) | RFlow (Eidex §3.2) | `test_dit.py:282` uses `RFlowScheduler` with `num_inference_steps=200` and the same kwargs as T1C-RFlow | **RFlow** — paper-symmetric with T1C-RFlow and VENA-S2 (the only axis isolated against VENA is now the backbone, not the scheduler) | NO — Eidex 2025 §4's reported DiT-3D numbers are the RFlow variant; we reproduce that |
| 3 | Loss | L2 on `ε` (DiT paper §2.2) | L1 on velocity `z_T1c − z_noise` (Eidex paper Eq. 4) | `test_dit.py` is inference-only; no loss visible in the vendored code | **L1 on velocity** — Eidex 2025 §3.2; mirrors `t1c_rflow.runner._train_t1c_rflow` so the only competitor-internal delta from T1C-RFlow is the backbone | NO — L1-velocity is the published Eidex 2025 baseline form |
| 4 | Input shape | `(B, C, H, W)` 2D | n/a | `dit3d.py:268` documents `(B, C, D, H, W)` for 3D | Wrapper passes `(B, 12, 48, 56, 48)` (concat conditioning) | NO |
| 5 | Positional embedding | learned sin-cos, fixed size after init | n/a | `dit3d.py:206` registers `pos_embed` as a non-trainable buffer of fixed shape | **All cohorts must share latent grid `(4,48,56,48)`** — VENA's schema 2.0.0 enforces this via the trunk-÷8 constraint; otherwise the wrapper would need a resize at the patch-embed step | HIGH — runtime guard in `runner.py` rejects shape mismatches |
| 6 | CFG / class label | classifier-free guidance via dropout of `y` | n/a | `dit3d.py:267` accepts `y=None` (unconditional) | `y=None` always — VENA conditioning is fully encoded in the channel-concat, no class label | NO |

If a future ablation row wants the DDPM scheduler (row 2 reverted to
Peebles & Xie 2023), it would become a `loss_form: ddpm_eps_l2` variant
in the engine's `HyperParamsCfg`; do not change the default without
explicit user authorisation.

## What is NOT modified

- DiT architecture (`DiT` class in `dit3d.py`), wrapper (`DiT3DWrapper` in
  `dit3d_wrapper.py`), positional-embedding helpers — preserved verbatim
  from upstream SHA `fc8314f6`, modulo the cosmetic print removal in
  `PATCHES.md`.

## What is invoked from VENA

| Symbol | Source file | Used by |
|---|---|---|
| `DiT3DWrapper` | `upstream/dit3d_wrapper.py` | `vena.competitors.dit_3d.runner._build_dit3d` |
| `DiT` (indirectly via `DiT3DWrapper`) | `upstream/dit3d.py` | same |

`test_dit.py` is kept as a reference for the inference loop only — its
top-level CLI is not invoked from VENA (VENA's inference path lives at
`vena.competitors.dit_3d.inference`).

## Reproducing the snapshot

```bash
cd src/external/dit_3d
git clone --depth 1 https://github.com/zacheidex/An-Efficient-3D-Latent-Diffusion-Model-for-T1-contrast-Enhanced-MRI-Generation.git tmp_clone
cd tmp_clone && git rev-parse HEAD > ../UPSTREAM_SHA.txt
cp dit3d.py dit3d_wrapper.py test_dit.py ../upstream/
cd .. && rm -rf tmp_clone
find upstream -name __pycache__ -type d -exec rm -rf {} +
find upstream -name '*.pyc' -delete
# Then re-apply the one cosmetic patch in PATCHES.md to upstream/dit3d.py.
```

## Files of interest under `upstream/`

| File | What it does |
|---|---|
| `dit3d.py` | DiT-3D model definition (`DiT`, `DiTBlock`, `PatchEmbed3D`, positional-embedding helpers). Adapted from Meta's 2D DiT (Peebles & Xie 2023) with `Conv3d` patch embedding + 3D sin-cos positional encoding. |
| `dit3d_wrapper.py` | Ten-line shim: `class DiT3DWrapper(nn.Module)` with `forward(x, t)` ignoring optional `y`. The VENA wrapper imports this class verbatim. |
| `test_dit.py` | Reference inference script (R-Flow Euler integration over 200 steps + AutoencoderKL decode). Not invoked by VENA — kept for traceability of the paper's reported DiT-3D numbers. |
