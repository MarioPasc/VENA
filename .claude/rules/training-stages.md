# Training Stages (VENA)

The full plan is specified in `/media/mpascual/Sandisk2TB/research/vena/docs/proposal.md` (sections 3–9). This rule pins the dependency order so agents do not skip phases or invent stage names. The proposal's "Timeline (indicative)" (§9) maps to six phases below; routine names are normative.

## Dependency graph

```
[ Phase 0 — Pre-flights (gating) ]
   vessel_mask  ────┐
   maisi_vae   ─────┼──► (independent; may run in parallel to Phase 1 data work)
   shortcut_diag ───┘

                    │
                    ▼
[ Phase 1 — Data assembly ]
   ucsf_pdgm_h5    (NIfTI → H5 cache: image-domain, skull-strip, co-reg, intensity-norm,
                   vessel masks, tumour masks, splits)
   malaga_manifest (CSV manifest, IRB & pseudonymisation; data not yet on disk)

                    │
                    ▼
[ Phase 2 — Pipeline implementation ]
   preprocess      (deterministic, idempotent NIfTI → torch tensor pipeline)
   mask_extract    (Frangi/Jerman/nnU-Net vesselness wrappers; tumour segmenter)
   maisi_io        (MAISI VAE encode/decode primitives, latent-stats cache)

                    │
                    ▼
[ Phase 3 — Model training ]
   fm_baseline     ($T_{1\text{pre}}$ only, no masks)        — isolates FM contribution
   fm_mask         (+ $M_v$ and/or $M_{\text{tum}}$ via ControlNet)
   fm_full         (all modalities + masks + auxiliary losses, EMA, full schedule)
   ablations       (proposal §7 axes: inputs / masks / mask-source / λ_v,t / SWAN encoding / residual)

                    │
                    ▼
[ Phase 4 — Internal validation ]
   internal_eval   (UCSF-PDGM 50-vol test: PSNR, SSIM, LPIPS, vessel-conspicuity, tumour-region metrics)
   shortcut_eval   (§6.5 — healthy-control diagnostic)
   downstream_eval (§6.4 — FastSurfer-LIT parcellation + tumour-segmenter inference on $\widehat{T_{1c}}$)

                    │
                    ▼
[ Phase 5 — External validation ]
   malaga_eval     (quantitative on glioma + meningioma)
   reader_study    (§6.3 — blinded radiologist reads, AFC + Likert)

                    │
                    ▼
[ Phase 6 — Writing & release ]
   paper           (MICCAI 2026 / MedIA / TMI manuscript)
   release         (code under permissive licence, vessel-resolved benchmark spec)
```

## Routine names (canonical)

Each stage maps to one routine under `routines/<bucket>/<name>/`, following the pattern in `preflight-pattern.md`:

| Phase | Routine | Purpose | Decision / target metric |
|---|---|---|---|
| 0 | `routines/preflights/vessel_mask` | Vesselness method QC on SWAN: Frangi vs Jerman vs (optional) nnU-Net. Dice / AHD vs hand-labels on ~20 cases. | `vesselness_method ∈ {frangi, jerman, nnunet}`, soft-mask threshold |
| 0 | `routines/preflights/maisi_vae` | MAISI-V2 VAE audit on UCSF-PDGM modalities (recon PSNR/SSIM, latent stats, latent equivariance under augs). | `vae_fine_tune: bool`, per-channel `latent_scale[4]`, list of latent-safe augs |
| 0 | `routines/preflights/shortcut_diag` | Pilot of §6.5 healthy-control diagnostic: confirm the model can be tested for SWAN dark-voxel shortcuts. | `protocol_feasible: bool`, control-cohort path |
| 1 | `routines/data/ucsf_pdgm_h5` | NIfTI → H5 converter (image-domain, all five modalities + masks + splits) | Valid H5 per `h5-design-principles.md`; 501 patients indexed |
| 1 | `routines/data/malaga_manifest` | CSV manifest builder, IRB/pseudo flags | Manifest passes schema check |
| 2 | `routines/pipeline/preprocess` | Deterministic preprocessing library smoke | Round-trip determinism on 10 cases |
| 2 | `routines/pipeline/mask_extract` | Run chosen vesselness + tumour segmenter on the H5 cache | Mask coverage stats |
| 2 | `routines/pipeline/maisi_io` | Cache MAISI latents for the train cohort | Latent H5 ready |
| 3 | `routines/training/fm_baseline` | Latent FM, $T_{1\text{pre}}$ only | Δ PSNR/SSIM vs Kleesiek 2019 baseline |
| 3 | `routines/training/fm_mask` | + mask conditioning (ControlNet) | Δ vessel-conspicuity vs `fm_baseline` |
| 3 | `routines/training/fm_full` | All modalities + auxiliary losses + EMA | Best in-domain scores |
| 3 | `routines/training/ablations` | Proposal §7 ablation matrix | 6-row ablation table |
| 4 | `routines/eval/internal` | UCSF-PDGM 50-vol test | PSNR, SSIM, LPIPS, vessel-conspicuity |
| 4 | `routines/eval/shortcut` | §6.5 diagnostic | False-positive enhancement rate ≈ 0 on controls |
| 4 | `routines/eval/downstream` | FastSurfer-LIT + tumour seg | Δ Dice vs reference $T_{1c}$ |
| 5 | `routines/external/malaga_eval` | Quantitative external | Multi-vendor + cross-pathology gap |
| 5 | `routines/external/reader_study` | Blinded reader study | AFC ≤ chance, Likert distribution |
| 6 | `routines/release/paper` | Figure & table production for manuscript | Reproducible build of every figure |
| 6 | `routines/release/release` | Tag, licence, benchmark spec | Public release artefacts |

## Dependency rules

1. **Pre-flights gate Phase 3.** Every training routine loads `artifacts/preflights/<name>/LATEST/decision.json` at startup and asserts the conditions it depends on (e.g. `vesselness_method != null` for `fm_mask`, `vae_fine_tune` matches the loaded checkpoint). Missing pre-flight ⇒ fast-fail with the missing-artifact path in the message.
2. **Phase 1 may overlap Phase 0.** Building `ucsf_pdgm_h5` does not depend on the vesselness choice — the H5 stores raw NIfTI-derived modalities. Mask layers are added after Phase 0 closes (`routines/pipeline/mask_extract`).
3. **One routine per directory.** Variants (e.g. `λ_v = 0.1` vs `λ_v = 1.0`) are separate YAML configs under the same routine — `routines/training/fm_full/configs/lambda_v_0_1.yaml`, `lambda_v_1_0.yaml`. Not separate routines.
4. **Ablation table is built from `training/ablations`.** Each row corresponds to one YAML config under `routines/training/ablations/configs/`. The six axes listed in proposal §7 (input modalities; mask conditioning; mask source; loss weighting; SWAN encoding; residual target) are the minimum set.
5. **Picasso A100-40GB sizing.** Full-system training targets ~5 days on 4× A100 (proposal §5). Each pre-flight is ≤ ~10 h. SLURM scripts must request these budgets explicitly via `--time` and `--mem` and pass `--constraint=dgx` for A100 nodes.

## How to add a new routine (procedure)

1. Read `docs/proposal.md` for the corresponding phase plus the relevant pre-flight `decision.json` files under `artifacts/preflights/<name>/LATEST/`.
2. Copy the closest existing routine as a starting skeleton (none yet — bootstrap by following `preflight-pattern.md` exactly).
3. Implement library code under `src/vena/<area>/`, never inside `routines/`.
4. Register the console script in `pyproject.toml` (`vena-<bucket>-<name>`).
5. Write a smoke YAML (`configs/smoke.yaml`) that runs end-to-end on a 4-volume subset in < 5 minutes.
6. Add a pytest under `tests/<area>/test_<name>_engine.py` (mock the model for unit; mark `slow` for GPU smoke).
