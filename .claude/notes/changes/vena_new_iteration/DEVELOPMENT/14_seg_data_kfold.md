# 14 — Data: K-fold OOF splits, dataset, augmentation

**Track/Wave/Deps.** SEG · **Wave 1 (parallel)** · deps: 10 (+ task 12's `make_soft_targets` at train time, but
build/test independently with a stub). Owns `src/vena/segmentation/data/` only.

## Objective
Build the **K-fold out-of-fold** split machinery (the free-ensemble backbone), the image-domain dataset serving
`{t1pre, t2, flair}` (z-score-on-brain) + GT soft targets, and the augmentation pipeline. **Leakage is the enemy**:
segmenter folds must be a subset of the FM train split; every FM-train mask is predicted OOF; FM-val/test masks
come from an all-FM-train model. Design authority: Part B.b, B.d, **B.f-§3** (k-fold ≠ deep ensemble caveat).

## Read and verify first
- `01_SHARED_CONTRACTS.md` (cohorts; FM split source; norm worlds).
- The FM split source: `splits/{train,val,test}` in the image/latent H5 and `routines/fm/train/configs/corpus/
  corpus_*.json` — the segmenter train set **must equal** the FM train patient set (verify how patient IDs map).
- `src/vena/data/h5/*/image_domain/` readers for `{t1pre,t2,flair}` + `masks/tumor`; MONAI transforms
  (`NormalizeIntensityd` with `nonzero=True` for z-score-on-brain; spatial + intensity augmentations).

## Files to create
```
src/vena/segmentation/data/kfold.py    # deterministic patient-level K-fold OOF plan (⊆ FM-train)
src/vena/segmentation/data/dataset.py  # image-domain dataset: {t1pre,t2,flair} z-score-on-brain + GT soft target
src/vena/segmentation/data/augment.py  # intensity/bias/contrast + spatial + modality-dropout pipeline
```

## Interface & contract
```python
@dataclass(frozen=True)
class FoldPlan:
    k: int
    fm_train_ids: tuple[str, ...]
    folds: tuple[tuple[str, ...], ...]              # K disjoint held-out ID tuples, union == fm_train_ids
    fm_val_ids: tuple[str, ...]; fm_test_ids: tuple[str, ...]
def build_fold_plan(cfg: DataConfig, fm_splits: Mapping[str, Sequence[str]]) -> FoldPlan: ...
def oof_assignment(plan: FoldPlan, patient_id: str) -> int | Literal["all_train"]:
    # which fold-model predicts this patient's OOF mask; FM-val/test -> "all_train"
class SegImageDataset(Dataset):  # returns {"image": (3,H,W,D), "target": (2,H,W,D) soft, "patient_id", "brain": (1,H,W,D)}
    def __init__(self, ids, cfg, *, augment: bool, target_fn=make_soft_targets): ...
def build_augmentation(cfg: DataConfig) -> Callable: ...
```
- **Determinism**: folds depend only on `(sorted(fm_train_ids), cfg.fold_seed, cfg.k_folds)` — same inputs → same
  plan across machines/runs. Stratify by cohort if feasible (keep each cohort represented in each fold).
- **Leakage invariant**: `folds` are pairwise disjoint and their union is exactly `fm_train_ids`; no FM-val/test ID
  appears in any fold; `oof_assignment` returns `"all_train"` for FM-val/test.
- **Normalisation**: **z-score on brain** (nonzero voxels, per channel) — the `downstream_seg` convention,
  independent of the VAE 99.95. Never the VAE percentile here.
- **Augmentation** (~80% of robustness): `RandBiasField`, `RandAdjustContrast`, `RandHistogramShift`, `RandGamma`,
  `RandGaussianNoise` + flip/affine/mild-elastic + **modality-dropout** (randomly zero t2 **or** flair; t1c always
  absent). Keep targets soft through spatial transforms (no nearest-neighbour binarisation).

## Implementation notes
- Store the resolved `FoldPlan` to the run artifact (JSON) so mask provenance is auditable.
- The dataset must accept a `target_fn` so tests inject a stub and task 17 injects task-12's `make_soft_targets`.

## Acceptance criteria
1. `build_fold_plan` is deterministic (same inputs → identical `folds`); `folds` disjoint; `⋃ folds == fm_train_ids`.
2. No FM-val/test ID in any fold; `oof_assignment(FM-val id) == "all_train"`.
3. `SegImageDataset[i]` → `image (3,H,W,D)`, `target (2,H,W,D) ∈ [0,1]`, brain-masked z-score (mean≈0/std≈1 over
   nonzero), `patient_id` present.
4. Modality-dropout zeros exactly one of {t2, flair} at the configured rate; t1c never appears.

## Tests (`tests/segmentation/data/test_kfold.py`, `test_dataset.py`; `pytestmark = pytest.mark.segmentation`)
- **fold determinism + leakage (load-bearing)**: synthetic ID lists → two `build_fold_plan` calls equal; assert
  disjoint + union; inject overlapping val IDs → assert none leak into folds; `oof_assignment` correct for
  train-fold vs val/test IDs.
- **cohort coverage**: with multi-cohort synthetic IDs, each fold contains ≥1 ID from each cohort (if stratified).
- **z-score-on-brain**: a synthetic volume with a zero background + nonzero brain → after normalise, brain mean≈0
  std≈1, background untouched.
- **modality dropout**: over N draws at rate p, exactly-one-of-{t2,flair} zeroed at ≈p; t1c channel absent.
- **soft-through-spatial**: a soft target through a flip/affine stays in `[0,1]` and is not binarised.

## Do NOT touch
Anything outside `src/vena/segmentation/data/` + `tests/segmentation/data/`. Do not read real cohort data in unit
tests (use synthetic on-disk fixtures / in-memory arrays).

## Report format
Report fold sizes per cohort, the leakage-check result, z-score brain stats, dropout rate measured, import-isolation
proof, ruff-clean, `STATUS`.
