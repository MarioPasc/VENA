# 12 — Soft targets (SDT → sigmoid), label harmonisation, nesting

**Track/Wave/Deps.** SEG · **Wave 1 (parallel)** · deps: 10. Owns `src/vena/segmentation/targets/` only.

## Objective
Turn hard BraTS-style labels into **soft `[WT, NETC]` training targets** by a per-class **signed distance
transform → sigmoid**, with correct handling of **multifocal / disconnected NETC**, code-agnostic label
harmonisation, and enforced nesting `NETC ⊆ WT`. Design authority: Part B.c step 1, B.d, **B.f-§4**.

## Read and verify first
- `01_SHARED_CONTRACTS.md` (norm worlds; grid).
- The label conventions in the cohort readers (`src/vena/data/niigz/*.py`) and image-H5 `masks/tumor` — confirm the
  raw label integers per cohort so harmonisation is code-agnostic.
- `scipy.ndimage.distance_transform_edt`, `skimage.segmentation` (geodesic) availability in the env (declare any
  new dep in `pyproject.toml` with a one-line rationale, per coding-standards rule 6).

## Files to create
```
src/vena/segmentation/targets/harmonise.py    # WT=(label>0), NETC=(label==1); code-agnostic
src/vena/segmentation/targets/sdt.py          # signed distance transforms (euclidean per-component + geodesic)
src/vena/segmentation/targets/soft_targets.py # hard labels -> soft [WT,NETC] via SDT->sigmoid; nesting
```

## Interface & contract
```python
def harmonise_labels(label: NDArray) -> dict[str, NDArray]:   # {"wt": bool, "netc": bool}, WT=(label>0), NETC=(label==1)
def signed_distance(mask: NDArray[bool], *, mode: Literal["euclidean_percomponent","geodesic"],
                    image: NDArray | None = None, clip_vox: float) -> NDArray[float]:   # >0 inside, <0 outside, clipped
def soft_target(mask: NDArray[bool], *, sigma_vox: float, mode: str, image=None, clip_vox: float) -> NDArray[float]:
    # returns sigmoid(SDT / sigma_vox) in [0,1]; 0.5 at boundary
def make_soft_targets(label: NDArray, cfg: TargetConfig, image: NDArray | None = None) -> NDArray:  # (2,H,W,D) [WT,NETC]
```
- **Sign convention**: SDT `> 0` inside the region, `< 0` outside, clipped to `±clip_vox`; `sigmoid(SDT/σ)` → 0.5 at
  boundary, ~0.95 at 3σ inside, ~0.05 at 3σ outside.
- **NETC multi-component**: `euclidean_percomponent` computes the SDT **per connected component** (via
  `scipy.ndimage.label`) and takes the union max, so the space *between* two disconnected lesions is **not** scored
  as interior. `geodesic` routes distance through image intensity (needs `image`); implement to the SiNGR idea
  (normalised, per-class). WT is typically connected → plain euclidean is fine; expose the mode per-class via cfg.
- **Nesting**: enforce `NETC_soft ≤ WT_soft` elementwise (clamp NETC to WT after softening); return `(2,H,W,D)`
  with channel 0 = WT, 1 = NETC.
- **Order**: this task produces image-resolution soft targets. **Do not pool here** (task 16 pools SDT→sigmoid
  outputs to the latent grid). The canonical order is `SDT → sigmoid → (later) avg-pool`.

## Implementation notes
- Operate on float32; epsilon-guard the `/σ`.
- Keep everything soft end-to-end (SoftSeg) — never binarise inside these functions.
- Document the label integers you harmonise from (BraTS-2021 vs 2023) in the module docstring.

## Acceptance criteria
1. `make_soft_targets` returns `(2,H,W,D)` float32 in `[0,1]`, `channel0 ≥ channel1` everywhere (nesting).
2. Euclidean-per-component SDT on two disjoint spheres does **not** bridge them (mid-gap value stays "outside").
3. `soft_target` gives ≈0.5 within one voxel of the boundary and monotonically increases inward.

## Tests (`tests/segmentation/targets/test_soft_targets.py`; `pytestmark = pytest.mark.segmentation`; pure-numpy)
- **known geometry**: a solid sphere radius r in a grid → SDT at centre ≈ r (clipped), SDT at boundary ≈ 0,
  `sigmoid(SDT/σ)` ≈ 0.5 on the boundary shell (assert_allclose with a tolerance).
- **multi-component (the load-bearing test)**: two disconnected cubes → `euclidean_percomponent` yields a mid-gap
  soft value **< 0.5** (outside), whereas a naive global signed-EDT (compute it inline for contrast) yields a
  spuriously high mid-gap value. Assert the per-component version is strictly lower at the gap centre.
- **harmonise**: a synthetic label with values `{0,1,2,4}` (BraTS) → `wt == (label>0)`, `netc == (label==1)`;
  a `{0,1,2,3}` variant (BraTS-2023) → same semantics (code-agnostic).
- **nesting**: random soft WT/NETC pre-clamp with NETC>WT somewhere → after `make_soft_targets`, `NETC ≤ WT`.
- **geodesic sanity** (if implemented): with a high-intensity barrier between two label blobs, geodesic distance
  across the barrier > euclidean (routes around it).

## Do NOT touch
Anything outside `src/vena/segmentation/targets/` + `tests/segmentation/targets/`. Do not pool to latent here.

## Report format
Report the sphere/boundary numbers, the multi-component gap values (per-component vs naive), new deps added (with
rationale), import-isolation proof, ruff-clean, `STATUS`.
