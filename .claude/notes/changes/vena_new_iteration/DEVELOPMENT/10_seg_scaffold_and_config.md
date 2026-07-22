# 10 — Segmentation submodule scaffold + config surface

**Track/Wave/Deps.** SEG · **Wave 0 (merge before any Wave-1 SEG task)** · deps: none.

## Objective
Create the `src/vena/segmentation/` package skeleton, the frozen Pydantic **`SegmentationConfig`** (with all
sub-configs) and its `from_yaml`, the **model registry**, the module exception hierarchy, and a `segmentation`
pytest marker. This is the shared contract every Wave-1 SEG task imports; it contains **no model/training logic**.

## Read and verify first
- `01_SHARED_CONTRACTS.md` (layout, coding rules, grid `(60,60,40)`).
- `src/vena/data/cohort/register_cohort.py` (or wherever `@register_cohort` lives) — **copy its decorator-registry
  pattern** for the model registry (VENA already standardises on decorator registries; do not invent a new style).
- One existing frozen-Pydantic routine config with `from_yaml` (e.g. `routines/fm/train/` config) for the
  `from_yaml` + OmegaConf-merge idiom.
- `pyproject.toml` `[tool.pytest.ini_options].markers`.

## Files to create
```
src/vena/segmentation/__init__.py          # re-export public API; __all__
src/vena/segmentation/config.py            # Pydantic configs + from_yaml
src/vena/segmentation/exceptions.py        # SegmentationError hierarchy
src/vena/segmentation/models/__init__.py   # re-export registry API
src/vena/segmentation/models/registry.py   # @register_segmentation_model + get_segmentation_model
src/vena/segmentation/targets/__init__.py  # empty __all__ (task 12 fills)
src/vena/segmentation/data/__init__.py     # empty __all__ (task 14 fills)
src/vena/segmentation/engine/__init__.py   # empty __all__ (tasks 13/17 fill)
src/vena/segmentation/derivation/__init__.py  # empty __all__ (task 16 fills)
src/vena/segmentation/metrics/__init__.py  # empty __all__ (task 15 fills)
src/vena/segmentation/py.typed             # marker
```
Modify: `pyproject.toml` (add `segmentation` marker). Do **not** add console scripts here (task 18 does).

## Interface & contract
Pydantic v2 `BaseModel`, `model_config = ConfigDict(frozen=True, extra="forbid")`. Top-level:
```python
class SegmentationConfig(BaseModel):
    model: ModelConfig
    data: DataConfig
    targets: TargetConfig
    loss: LossConfig
    train: TrainConfig
    derivation: DerivationConfig
    metrics: MetricsConfig
    seed: int = 1337
    @classmethod
    def from_yaml(cls, path: str | Path) -> "SegmentationConfig": ...   # OmegaConf load → resolve → cls(**)
```
Sub-configs (fields are the design surface — every downstream task reads these; add fields freely later, never
repurpose):
- `ModelConfig`: `name: Literal["bsf_swinunetr_brats","bsf_swinunetr_ukb","segresnet"]`, `feature_size:int=48`,
  `in_channels:int=3`, `out_channels:int=2`, `checkpoint: Path|None`, `strict_load:bool=False`,
  `deep_supervision:bool=True`.
- `DataConfig`: `corpus_registry: Path`, `image_h5_root: Path`, `modalities: tuple[str,...]=("t1pre","t2","flair")`,
  `k_folds:int=5`, `fold_seed:int=1337`, `patch_size: tuple[int,int,int]`, `cache_rate:float`, `num_workers:int`.
- `TargetConfig`: `soft:bool=True`, `sdt_sigma_vox:float=3.0`, `netc_operator: Literal["euclidean_percomponent",
  "geodesic"]="euclidean_percomponent"`, `clip_vox:float=10.0`.
- `LossConfig`: `dice_variant: Literal["dml","soft_dice","tversky","focal_tversky"]="dml"`,
  `ce_variant: Literal["ce","focal_ce"]="ce"`, `dice_weight:float=1.0`, `ce_weight:float=1.0`,
  `tversky_alpha:float=0.3`, `tversky_beta:float=0.7`, `deep_supervision_weights: tuple[float,...]`.
- `TrainConfig`: `max_epochs`, `lr`, `batch_size`, `optimizer:Literal["adamw"]`, `scheduler:Literal["cosine"]`,
  `amp:bool=True`, `val_every_epochs:int`, `early_stop_patience:int`, `calibration_split_frac:float=0.1`.
- `DerivationConfig`: `temperature: Literal["per_class","global","none"]="per_class"`, `avg_pool_stride:int=4`,
  `latent_grid: tuple[int,int,int]=(60,60,40)`, `emit_variance:bool=False`.
- `MetricsConfig`: `gseg_wt_dice:float=0.80`, `gseg_netc_dice:float=0.50`,
  `selection_metric: Literal["dice","brier","dual"]="dual"`.

Registry (mirror `@register_cohort`):
```python
def register_segmentation_model(name: str) -> Callable[[type], type]: ...
def get_segmentation_model(name: str, cfg: ModelConfig) -> torch.nn.Module: ...  # raises SegModelError on unknown
```
`exceptions.py`: `class SegmentationError(Exception)`, subclasses `SegModelError, SegDataError, SegLossError,
SegTargetError, SegDerivationError, SegMetricError`.

## Implementation notes
- `latent_grid` default `(60,60,40)` — never `(48,56,48)`.
- Registry `get_segmentation_model` must raise `SegModelError` naming the unknown key + the list of registered keys.
- Keep `models/registry.py` import-light (no torch model construction at import; the builder runs inside `get_*`).

## Acceptance criteria
1. `from vena.segmentation import SegmentationConfig, get_segmentation_model, register_segmentation_model` resolves.
2. `SegmentationConfig.from_yaml(<a written sample yaml>)` returns a frozen instance; mutating a field raises.
3. `extra="forbid"` — an unknown YAML key raises a Pydantic `ValidationError`.
4. `get_segmentation_model("nope", cfg)` raises `SegModelError` listing registered names.
5. `pytest -m segmentation` is a recognised marker (no `PytestUnknownMarkWarning`).

## Tests (`tests/segmentation/test_config.py`, `tests/segmentation/test_registry.py`; `pytestmark = pytest.mark.segmentation`)
- **round-trip**: write a minimal YAML → `from_yaml` → assert every default + overridden field; assert `frozen`
  raises on assignment.
- **schema strictness**: a YAML with a typo'd key → `ValidationError`.
- **defaults are the contract**: assert `latent_grid == (60,60,40)`, `out_channels == 2`,
  `netc_operator == "euclidean_percomponent"`, `selection_metric == "dual"` (guards against silent drift).
- **registry**: `@register_segmentation_model("dummy")` on a stub `nn.Module` factory → `get_segmentation_model
  ("dummy", cfg)` returns it; unknown name → `SegModelError`.

## Do NOT touch
Anything outside `src/vena/segmentation/` + the single `pyproject.toml` marker line + `tests/segmentation/`.

## Report format
Artifact = the package path (`readlink -f src/vena/segmentation`). Report: the exact `__all__` exported, the test
count added, the import-isolation proof, ruff-clean on touched files, `STATUS: DONE | QUESTION | PREMISE-FALSE | BLOCKED`.
