# 40 — Validation: soft-mask QC + latent-embedding visualization + injection sanity

**Track/Wave/Deps.** VALIDATION · **Phase-1 (gates the oracle GPU launch)** · deps: 19 (cached soft masks); the
injection checks additionally use 20/21. Owns `routines/segmentation/validate_masks/` (or a `src/vena/segmentation/
metrics/visualize.py` + a thin routine). Read-only w.r.t. training.

## Objective
Before spending GPU-days on the oracle runs, **eyeball that the soft masks are right** and that the injection is
wired sanely. Three deliverables: (a) **visual QC** of hard vs soft masks overlaid on anatomy; (b) a
**latent-embedding visualization** of the masks once pooled to the latent grid; (c) an **injection sanity panel**
(step-0 identity + residual locality). This is the "validation of the soft-mask approach" the user asked for
(recognising there is no deep ensemble yet — the GT oracle has no ensemble variance to show).

> **[RESOLVED 2026-07-22]** The latent-space validation = **(B) per-patient PCA/UMAP** of mask-latent vectors
> coloured by **tumour volume / cohort** (do the masks occupy a sensible manifold — do larger/necrotic tumours land
> where expected?) **+ (A) a slice montage** with a **pinned layout**:
> - **one patient per row**; patients chosen to **span small → large tumour size** (e.g. tumour-volume quantiles);
> - **5 columns per patient = 5 tumour-bearing slices** (evenly spaced through the tumour extent);
> - each cell = an **anatomy slice** (T1pre) with the **soft `[WT,NETC]` mask overlaid at α = 0.7** (perceptual
>   colormap; WT and NETC distinguishable).
>
> (C per-voxel separation and D ControlNet-residual-overlay stay available behind a config flag; **D lands in S2**
> — it needs the injection wired, so it is the injection-sanity deliverable there, not S1.)

## Read and verify first
- `01_SHARED_CONTRACTS.md`; task 19 cache (`masks/tumor_latent_soft`); the image-domain H5 (`images/*`,
  `masks/tumor`) for the anatomy overlay; the decode helper `vena.common.decode.decode_box` if showing decoded
  context; the exhaustive-val figure conventions in `vena.model.fm.eval.exhaustive` (reuse the black-background
  panel style).

## Files to create
```
src/vena/segmentation/metrics/visualize.py          # QC + embedding figure builders (pure, testable)
routines/segmentation/validate_masks/{__init__.py,cli.py,configs/default.yaml,engine/{__init__.py,validate_engine.py}}
```
Modify: `pyproject.toml` (`vena-segmentation-validate-masks`). New deps if UMAP is chosen (`umap-learn`) — declare
with rationale (coding-standards rule 6).

## Interface & contract
```python
def render_mask_qc(image, hard_mask, soft_mask_img, soft_mask_latent, *, patient_id, path) -> Path:
    # 3 rows: anatomy+hard overlay | anatomy+soft overlay (image res) | soft mask on the (60,60,40) latent grid
def render_slice_montage(patients: Sequence[PatientView], *, n_cols=5, alpha=0.7, path) -> Path:
    # PINNED LAYOUT: one patient per row, ordered by tumour volume (small→large); n_cols tumour-bearing slices
    # per row (evenly spaced through the tumour extent); each cell = T1pre slice + soft [WT,NETC] overlay @ alpha
def render_latent_embedding(mask_latents: dict[str, Tensor], meta: pd.DataFrame, *, method="pca_umap_perpatient",
                            color_by=("tumor_volume","cohort"), path) -> Path:
    # (B) 2-D PCA/UMAP of per-patient mask-latent vectors, coloured by tumour volume / cohort  [default]
def render_injection_sanity(module, batch, *, path) -> Path:   # S2 deliverable
    # step-0 identity residual heatmap + output_scale-ramp residual-locality (residual concentrated in/near WT)
```
- Figures: black background, per-slice intensity-matched windows (reuse exhaustive-val conventions), soft masks
  shown with a perceptual colormap in [0,1]; annotate patient id + tumour volume.
- The routine runs over a small configurable patient set (best/worst/random by tumour size across cohorts) and
  writes `artifacts/validate_masks/<UTC>/figures/*` + a short `report.md` + a `decision.json`
  (`masks_look_valid: bool` is a **human-set** field after review, plus machine stats: soft-mass fraction in WT,
  NETC⊆WT violation count, empty-mask count).

## Acceptance criteria
1. `render_mask_qc` produces a 3-row figure (hard / soft-image / soft-latent) for a synthetic case; the soft mask
   is visibly graded (not binary) and nested (NETC ⊆ WT).
2. `render_slice_montage` produces the **pinned layout**: rows = patients ordered by ascending tumour volume, exactly
   5 tumour-bearing slice columns per row, soft `[WT,NETC]` overlaid at α = 0.7 (assert row order + column count).
3. `render_latent_embedding` (PCA/UMAP per-patient) runs and produces a 2-D figure; per-patient points coloured by
   tumour volume / cohort.
4. `render_injection_sanity` (S2) shows the step-0 residual ≈ 0 everywhere and, at `output_scale>0`, residual energy
   **concentrated in/near WT** (report the in-WT vs out-of-WT residual-energy ratio).
5. The routine writes figures + `report.md` + `decision.json` with the machine stats.

## Tests (`tests/segmentation/metrics/test_visualize.py`; `pytestmark = pytest.mark.segmentation`; headless matplotlib `Agg`)
- **QC figure**: synthetic anatomy + soft mask → a figure file is written; assert the soft-overlay array is graded
  (has values strictly between 0 and 1) and nested.
- **embedding**: synthetic per-patient mask latents + metadata → the chosen method returns a 2-D embedding of the
  right shape; colour mapping matches the metadata.
- **injection locality**: synthetic residuals concentrated in a WT box → the in-WT/out-of-WT energy ratio > 1
  (guards the P2 locality claim); step-0 residual == 0.
- **machine stats**: soft-mass-in-WT fraction, NETC⊆WT violation count computed correctly on a constructed case.

## Do NOT touch
Training code paths; the cached masks (read-only); the FM `decision.json`.

## Report format
Readback artifact dir + the figure paths, the graded/nested checks, the in-WT residual-energy ratio, the chosen
embedding method, the machine stats, import-isolation proof, ruff-clean, `STATUS`. **This routine's `report.md` is
the human gate for the oracle launch — surface it for review.**
