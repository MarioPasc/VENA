# External Dependencies

This project depends on **one frozen pretrained model** (MAISI-V2 VAE-GAN, with the MAISI-V2 Flow-Matching checkpoint kept as a reference) and **two datasets** (UCSF-PDGM for training and internal validation; an in-house Málaga cohort for external validation). None of these live in the repo. Their canonical paths are listed in `src/external/LINKS.md` and mirrored under "Documentation source-of-truth" in the top-level `CLAUDE.md`.

## Hard rules

1. **Never edit code under `src/external/`** except `src/external/LINKS.md` itself. The directory is a pointer index, not a vendored copy. The deny list in `.claude/settings.json` enforces this.
2. **Never write to checkpoint paths.** The deny list also blocks writes under `/media/mpascual/Sandisk2TB/checkpoints/**`. Treat the checkpoints as read-only system files.
3. **Adapter wrappers go in `src/vena/adapters/`** (or `src/vena/common/` for cross-cutting primitives), not in any external source tree. Examples:
   - `src/vena/common/maisi.py` — shared MAISI-V2 VAE-GAN encode/decode primitive (used by every routine that touches the latent space).
   - `src/vena/adapters/controlnet.py` — ControlNet-style conditioning branch built around the frozen VAE / FM trunk.
4. **External code is imported, not modified.** If an upstream change is needed (e.g. swapping MAISI's flow-matching head for ours, or freezing only the encoder while fine-tuning the decoder), do it via subclass / monkey-patch / weight surgery in an adapter module — never by editing the external source.
5. **Dataset paths come from config.** Routines accept dataset paths as YAML parameters; the default values live in the routine's `configs/default.yaml`. Never hard-code dataset paths in library code under `src/vena/`.
6. **Treat checkpoints as inputs to checksum.** When loading, log the file's SHA-256 (or its size + mtime if SHA is too slow) so the artifact's `decision.json` can record which exact weights produced the result.

## Canonical external paths (snapshot — `src/external/LINKS.md` is source of truth)

### Local workstation

| Asset | Path |
|---|---|
| MAISI-V2 source code | `/media/mpascual/Sandisk2TB/checkpoints/MAISI_V2_RM/code/NV-Generate-CTMR` |
| MAISI-V2 VAE-GAN checkpoint (encoder + decoder) | `/media/mpascual/Sandisk2TB/checkpoints/MAISI_V2_RM/NV-Generate-MR/models/autoencoder_v2.pt` |
| MAISI-V2 Flow-Matching checkpoint (reference) | `/media/mpascual/Sandisk2TB/checkpoints/MAISI_V2_RM/NV-Generate-MR/models/diff_unet_3d_rflow-mr.pt` |
| UCSF-PDGM source data (NIfTI) | `/media/mpascual/MeningD2/GLIOMA/UCSF_PDGM/source` |
| UCSF-PDGM H5 cache (image-domain, schema 2026-05-19) | `/media/mpascual/MeningD2/GLIOMA/UCSF_PDGM/h5/UCSFPDGM_image.h5` |
| Documentation (proposal, literature) | `/media/mpascual/Sandisk2TB/research/vena/docs/` |

### Picasso (UMA HPC) mirror

| Asset | Path |
|---|---|
| MAISI-V2 VAE-GAN checkpoint | `/mnt/home/users/tic_163_uma/mpascual/fscratch/checkpoints/NV-Generate-MR/models/autoencoder_v2.pt` |
| MAISI-V2 Flow-Matching checkpoint | `/mnt/home/users/tic_163_uma/mpascual/fscratch/checkpoints/NV-Generate-MR/diff_unet_3d_rflow-mr.pt` |
| UCSF-PDGM H5 cache | `/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/vena/UCSFPDGM_image.h5` |

When this table drifts from `src/external/LINKS.md`, **`src/external/LINKS.md` wins** and this file should be updated. CI may add a check that the two are in sync.

## A note on the Málaga external cohort

The Málaga in-house cohort (Hospital Universitario Regional de Málaga, proposal §2.2) is **not** yet on disk. Acquisition is governed by an active data-sharing agreement and the IBIMA-BioNAND pseudonymisation pipeline. Until it lands:

- No routine may take a Málaga path as a required input.
- The external-validation routine (`routines/external_val/`, planned) carries a `data_path` parameter that is `null` by default and a clear `MalagaNotAvailableError` fast-fail.
- Manifest format is fixed: per-study CSV with the fields enumerated in proposal §2.2 (subject_id, study_id, scan_date, pathology, who_grade, idh_mgmt_1p19q, enhancement_status, longitudinal_index, sequences_available, field_strength, scanner_vendor_model, voxel_size_mm, gd_agent, gd_dose_mmolkg, pre_post_delay_min, prior_treatments, ct_available, notes).

## Adapter module checklist

When you create a new adapter:

- [ ] Lives under `src/vena/adapters/` (or `src/vena/common/` for cross-cutting primitives).
- [ ] Imports from the external source via the path declared in `src/external/LINKS.md` (or via an installed pip package if upstream publishes one — preferred when available).
- [ ] Carries a module-level docstring stating which external commit / checkpoint version it targets.
- [ ] Has a unit test in `tests/adapters/test_<name>.py` that exercises the wrapper with a synthetic input (no checkpoint download).
- [ ] Logs the resolved checkpoint path and its checksum at first load.
