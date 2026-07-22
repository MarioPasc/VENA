# 19 — Routine: source-agnostic soft-mask derivation + latent-H5 cache

**Track/Wave/Deps.** SEG · **GT path = Phase-1 (no segmenter); predicted path = Phase-2** · deps (GT): 10, 12, 16;
deps (predicted): + 17. Owns `routines/segmentation/mask_derive/` + a `derive_latent_soft_mask` entrypoint in
`src/vena/segmentation/derivation/`.

## Objective — THE "swap" GUARANTEE
Produce the soft `[WT,NETC]` conditioning map on the latent grid `(2,60,60,40)` and **cache it into every cohort's
latent H5**, from **either** GT hard labels **or** the trained segmenter — through **one code path** so that going
from oracle to deployable is a single `source:` config change. This is the concrete implementation of the user's
requirement: *"code this so that when we have the segmenter, we only change the GT mask for the predicted one and it
behaves exactly the same."* Design authority: Part A.7, A.8-§6/§7, B.c, B.f-§4; T-04.

## Read and verify first
- `01_SHARED_CONTRACTS.md` (grid; latent H5 schema; the shared writer/validator).
- Task 12 (`make_soft_targets`: SDT→sigmoid at image res), task 16 (`pool_to_latent`, `apply_temperature`,
  `ensemble_soft`), task 17 (`predict_oof`, predicted path only).
- `src/vena/data/h5/shared/` + `augmented/latent_domain.py` — the API to **add a group to an existing latent H5**,
  extend the manifest, and extend `assert_*_valid`. Confirm how `masks/tumor_latent` (oracle avg-pool) was written
  and whether the **image→latent crop** used there must be replicated for voxel registration.

## Files to create
```
src/vena/segmentation/derivation/derive.py   # derive_latent_soft_mask(source, ...) — the shared entrypoint
routines/segmentation/mask_derive/{__init__.py,cli.py,configs/{gt.yaml,predicted.yaml,smoke.yaml},engine/{__init__.py,derive_engine.py}}
```
Modify: `pyproject.toml` (`vena-segmentation-mask-derive`); the latent-H5 manifest + validator to know the new groups.

## Interface & contract
```python
def derive_latent_soft_mask(*, source: Literal["gt","predicted"], label: NDArray | None = None,
                            seg_prediction: Tensor | None = None, temps: ClassTemperatures | None = None,
                            image: NDArray | None, cfg) -> Tensor:   # returns (2,60,60,40) soft [WT,NETC]
```
- **GT path** (`source="gt"`): `harmonise_labels` → `make_soft_targets` (SDT→sigmoid at image res, per-component/
  geodesic NETC) → `pool_to_latent`. **No temperature, no ensemble** (GT is exact). Output = the oracle soft mask.
- **Predicted path** (`source="predicted"`): segmenter logits → `apply_temperature(T_WT,T_NETC)` → sigmoid →
  `pool_to_latent` → `ensemble_soft` (K-fold mean). **Identical output format + softening character** (the segmenter
  is trained on the same SDT-soft targets, so its calibrated probs live in the same soft space).
- **Cache group naming**: GT → `masks/tumor_latent_soft`; predicted → `masks/tumor_latent_pred`. Both
  `(N,2,60,60,40)` float32 ∈ [0,1], **written beside** `masks/tumor_latent` (never replacing it). Self-describing
  attrs; bump H5 `schema_version`; root attr `mask_source` + (`predicted`) `predicted_mask_seg_sha256` / fold plan.
- **Registration invariant**: the written map voxel-aligns with `latents/*` — assert shape `(…,2,60,60,40)` and a
  centroid match against the oracle `masks/tumor_latent` WT union on a spot case.
- **Routine**: `vena-segmentation-mask-derive <yaml>`; `configs/gt.yaml` sets `source: gt` (Phase-1, no segmenter);
  `configs/predicted.yaml` sets `source: predicted` + the segmenter run/ckpts (Phase-2). Same engine, same writer.

## Implementation notes
- Idempotent: re-running replaces only the target group + re-stamps provenance; the oracle `masks/tumor_latent` and
  `latents/*` are byte-untouched.
- Softening is config-driven (`TargetConfig.soft`, `sdt_sigma_vox`, `netc_operator`) so the oracle and predicted
  maps share the exact pooling/softening settings — the "behaves the same" contract is enforced by shared config,
  not by duplicated code.

## Acceptance criteria
1. `derive_latent_soft_mask(source="gt", ...)` on a synthetic label → `(2,60,60,40)` ∈ [0,1], `NETC ≤ WT`.
2. `source="gt"` and `source="predicted"` produce **identically-shaped, identically-attributed** outputs (the swap
   guarantee) — a test that runs both on a synthetic case and asserts same shape/dtype/range/attrs schema.
3. `mask_derive` GT run on a synthetic 4-patient latent H5 writes `masks/tumor_latent_soft (4,2,60,60,40)`, attrs +
   provenance present, `schema_version` bumped, `assert_*_valid` passes, oracle group byte-identical.
4. Centroid of the written WT channel aligns with the oracle `masks/tumor_latent` WT union (registration).

## Tests (`tests/routines/segmentation/test_mask_derive.py`; `pytestmark = pytest.mark.segmentation`; H5 tmp fixture)
- **swap-invariance (load-bearing)**: GT vs predicted paths on a synthetic case → same shape/dtype/range and same
  H5-attr schema (only the group name + provenance differ). This is the guarantee the user asked for.
- **GT derivation correctness**: a synthetic label → SDT-soft → pooled `(2,60,60,40)`, nesting holds, boundary
  voxels graded.
- **H5 write + validate**: writes `masks/tumor_latent_soft`, `assert_*_valid` passes, oracle group unchanged,
  `schema_version` up, registration centroid check.
- **idempotency**: re-run replaces only the target group; a second run yields identical bytes.

## Do NOT touch
INJECT-track files; `masks/tumor_latent` (oracle avg-pool) or `latents/*`; real cohort H5s in tests.

## Report format
Readback run-dir + the written group shape/attrs, the swap-invariance result, the `assert_*_valid` result, the
oracle-untouched + registration checks, import-isolation proof, ruff-clean, `STATUS`.
