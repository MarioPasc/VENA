# TASK T0 — `validation-core`: the Phase-2 substrate

**Read `01_SHARED_CONTRACTS.md` first, completely. It is not optional context.**

| | |
|---|---|
| **Model** | Opus 4.8, effort `max` |
| **Isolation** | **Main checkout** (`/home/mpascual/research/code/VENA`), no worktree. You are the only agent in this tree. |
| **Runs** | **Sequentially, before every other Phase-2 task.** Three parallel agents block on you. |
| **Lane (you own)** | `src/vena/validation/**`, `routines/validation/__init__.py`, `routines/validation/preregister/**`, `tests/validation/**`, and **`pyproject.toml`** |
| **Do not touch** | Anything else. Especially `src/vena/model/fm/**`, `src/vena/inference/**`, `CLAUDE.md`, `.claude/rules/**`, `src/external/**` |

---

## 1. Why this exists

Phase 1 froze 360 prediction H5s + 40 reference H5s (289 GB). Phase 2 must
score them. Three downstream agents (paired fidelity §4.2, spatial residual
§4.3, downstream segmentation §4.4) will run **in parallel** on top of what you
build. Your modules are their only shared substrate — if your loader is wrong,
all three are wrong, and nobody will notice because every number will still
look plausible.

You write **no metrics**. You write the substrate they all stand on.

Your API is a contract three agents code against simultaneously. **Getting the
signatures right matters more than getting the internals clever.** Once you
land, changing a signature costs three merge conflicts.

---

## 2. Deliverables

### 2.1 `src/vena/validation/io.py` — the results loader

The single most important module in Phase 2.

- `discover_shards(root: Path) -> list[ShardInfo]` — glob `<root>/*/decision.json`.
- `build_index(root: Path) -> pd.DataFrame` — one row per prediction file:
  `method, cohort, ring, nfe, shard, path, references_h5, n_scans, schema_version`.
  **Discover from disk; never hard-code the 16 methods or the 8 cohorts** (§5 of
  contracts — BraTS-PED will appear later and must flow through with no code change).
- `ScanSample` — a frozen dataclass carrying `scan_id, patient_id, cohort, ring,
  method, nfe, pred, real, brain, wt, inference_seconds, peak_vram_mb`.
  Arrays are `np.ndarray` `(240,240,155)`; `pred`/`real` f32, masks bool.
- `iter_scans(pred_path, *, reference_cache, scan_ids=None) -> Iterator[ScanSample]`
  — streams one scan at a time. **Never load a whole (N,240,240,155) dataset
  into RAM**; index row-wise (the files are chunked `(1,H,W,D)`, so a row read
  is one chunk read — 0.076 s/vol measured).
- `ReferenceCache` — LRU keyed by resolved reference path. 16 methods share one
  reference per cohort; re-reading it 45× is the difference between a 40-minute
  sweep and a 6-hour one. Cache the *joined index* always; cache volume data
  only if it fits the configured budget.

**The join (contracts §6.3) — implement exactly:**
1. `references_h5 = h5.attrs["references_h5"]`, resolved against
   `pred_path.parents[3]` (the shard root).
2. Build `{scan_id: row}` for both files. **Join on `scan_id`, never on index.**
3. Assert `schema_version == "2.0"` on both; raise a named exception otherwise.
4. If a prediction `scan_id` is absent from the reference → raise. If a
   reference `scan_id` is absent from the prediction → that scan failed at
   inference; **log at WARNING, skip it, and count it** so the engine can report
   coverage. Never silently drop.

Custom exceptions: `ValidationIOError`, `SchemaVersionError`, `JoinError`.

### 2.2 `src/vena/validation/registry.py` — method + cohort metadata

Encodes the pre-registered roles from **contracts §4.1** (already decided — do
not re-derive from the docs, they are stale):

- `MethodRole = Literal["vena", "family", "ablation", "supplementary"]`
- `MethodSpec(key, display, tier, role, selection_nfe, panel_source)` where
  `tier ∈ {"identity","2d_gan","2d_diffusion","3d_latent","vena"}`.
- `VENA_HEADLINE = "VENA-S1-v3b-rw"`.
- `COMPETITOR_FAMILY` — the 8. `ABLATION_FAMILY` — the 3. `SUPPLEMENTARY` — the 4.
- `ring_of_cohort(cohort) -> Literal["A","B"]` — but **prefer the H5's own `ring`
  root attr**; use the map only as a cross-check and raise on disagreement.
- Stable display order + a colourblind-safe colour per method, so every figure
  across all three routines orders and colours methods identically.

**A registry entry must not be needed for a method to be *loaded*.** Unknown
methods (e.g. a future BraTS-PED-era row) load fine with
`role="supplementary"` and a WARNING. Fail closed on statistics, open on I/O.

### 2.3 `src/vena/validation/regions.py` — region masks

`region_masks(brain, wt, *, dilate_k: int = 5) -> dict[str, np.ndarray]`
returning exactly:
- `brain` — the HD-BET/CBICA mask.
- `wt` — whole tumour, binary.
- `bg` — `brain & ~dilate(wt, k=dilate_k)` — the §4.3 C-noT region.

Dilation **must** use the house technique (contracts §8):
`F.max_pool3d(x.float(), kernel_size=k, stride=1, padding=k//2) > 0.5`.
`dilate_k=5` ⇒ radius 2. Make it a parameter; the engine round-trips it into
`decision.json`.

Note §4.2 also wants `brain \ WT` (**undilated**) as its background region,
while §4.3's C-noT uses the **dilated** one. Provide both; name them
unambiguously (`bg_undilated` vs `bg`), and document which proposal section
each serves. Conflating them is a silent wrong number.

### 2.4 `src/vena/validation/stats.py` — the §6 primitives

Shared by all three routines. Pure functions over numpy/pandas, no I/O.

- `collapse_to_patient(df, value_cols, *, by=("method","cohort","nfe","patient_id")) -> pd.DataFrame`
  — **mean over scans within a patient.** This is trap #4 in contracts §11 and
  the single most important function you write. LUMIERE: 72 scans → 11 patients.
- `paired_wilcoxon(vena, competitor) -> WilcoxonResult(statistic, pvalue, n)`
  — `scipy.stats.wilcoxon`, two-sided, paired on `patient_id`. Must **inner-join
  on patient_id** before testing, and assert both arms have identical patient
  sets — an unaligned paired test is nonsense. Handle the all-zero-difference
  degenerate case explicitly rather than letting scipy raise.
- `holm_bonferroni(pvalues: dict[str,float]) -> dict[str, HolmResult]`
  — wraps `statsmodels.stats.multitest.multipletests(method="holm")`. Returns
  per-comparison adjusted p + reject flag. **The dict key set defines the
  family** — the caller passes exactly one family at a time.
- `cliffs_delta(a, b) -> float` — non-parametric effect size, Cliff 1996.
  Implement via the standard dominance count; add the conventional magnitude
  thresholds (|δ|<0.147 negligible, <0.33 small, <0.474 medium, else large).
- `bootstrap_ci(values, *, n_boot=10000, ci=0.95, strata=None, seed=1337) -> (lo, hi)`
  — **patient-stratified** when `strata` (the cohort labels) is given, per
  proposal §6.2. Resample patients, never scans.
- `MCID = 0.01` on the [0,1] intensity scale (proposal §6.2) — a module
  constant with the citation in its docstring, so every routine reports
  "statistically but not clinically significant" the same way.

Reuse `spearman_with_bootstrap_ci` from
`vena.preflight.priors_validation.statistics.correlation` — **do not write a
second Spearman**. Re-export it here so the §4.3 agent has one obvious import.

### 2.5 `src/vena/validation/plotting.py` — shared figure substrate

- House style: black-background qualitative panels; gray cmap; per-slice
  `vmin/vmax` anchored to the **real** slice (contracts §9.2).
- `annotate_significance(ax, pairs, holm_results, ...)` — the `*`/`**`/`***`/
  `n.s.` bracket annotator every routine uses, so significance is marked
  identically everywhere.
- `method_palette()` / `method_order()` — from the registry.
- `render_method_comparison_figure(...)` — the **sibling** of
  `vena.model.fm.eval.exhaustive.render_comparison_figure`, keyed by **method**
  instead of NFE. Same conventions. Reuse `select_content_slices` from that
  module. **Do not modify the original** (contracts §9.2).

### 2.6 `src/vena/validation/audit.py` — the §4.1 harmonisation audit (Table S1)

Phase 1 already harmonised. This **verifies** it and produces the mandatory
cross-method audit:
- Assert per volume: inside brain ⊆ [0,1]; outside brain ≡ 0.
- Per (method, cohort): post-harmonisation moments — mean, std, p1, p50, p99 —
  with the **real T1c row as the reference**, and each method's deviation from it.
- Diagnostic rule from proposal §4.1: cross-method spread of the *mean* within a
  cohort must be < 0.05. Flag violations; **do not silently correct them.**
- A method whose p99 sits systematically below the real T1c's is
  **under-saturating enhancement** — a real, reportable failure mode, not a
  harmonisation bug.

### 2.7 `src/vena/validation/artifacts.py` — the artifact writer

Implements contracts §9 exactly. `ArtifactWriter(output_root, routine)` with
`run_dir`, `tables_dir`, `figures_dir`, `per_scan_dir`, `write_decision(dict)`,
`write_report(md)`. **Copy `_make_run_dir` verbatim** from
`routines/preflights/latent_aug_equivariance` (contracts §9) — relative LATEST
symlink, `"%Y-%m-%dT%H-%M-%SZ"`, OSError logged-and-swallowed.

`write_decision` must stamp: `schema_version`, `produced_at`, `producer`,
`git_sha`, `input_root`, per-shard `decision.json` sha256, cohorts analysed,
methods analysed, family assignment, `n_scans`, `n_patients`, and every
parameter that influenced a number.

### 2.8 `routines/validation/preregister/` — freeze the ring partitions

Closes fairness concern ⑦ (P3 pre-registration). A thin engine that resolves
the test scan/patient inventory **from the frozen predictions on disk** and
writes `ring_partitions.json` + its sha256 into the artifact folder.

Uses `vena.inference.image_dataset.resolve_test_scan_patient_pairs` as the
cross-check against what the H5s actually contain, and **raises on
disagreement** — a mismatch means the predictions do not correspond to the
splits, which is a stop-the-line finding.

CLI: `vena-validation-preregister <yaml>`.

### 2.9 `pyproject.toml` — you own this file

1. Register the `validation` pytest marker:
   `"validation: Phase-2 benchmark analysis components"`.
2. Register **all four** console scripts now, so the three parallel agents never
   touch this file and never conflict:
   ```toml
   vena-validation-preregister      = "routines.validation.preregister.cli:main"
   vena-validation-paired-fidelity  = "routines.validation.paired_fidelity.cli:main"
   vena-validation-spatial-residual = "routines.validation.spatial_residual.cli:main"
   vena-validation-downstream-seg   = "routines.validation.downstream_seg.cli:main"
   ```
   Entry points are lazy — the last three resolve to modules that do not exist
   yet, and `pip install -e .` still succeeds. That is intentional.
3. Add **no new dependencies.** Everything needed is installed (contracts §8).
   If you believe otherwise, **stop and report** rather than adding one.

---

## 3. Acceptance criteria

- [ ] `iter_scans` streams; peak RSS stays flat across a 50-scan file (measure
      it and report the number).
- [ ] Join is by `scan_id`. **Write a test that proves it**: build a synthetic
      pair whose reference rows are *shuffled* relative to the prediction rows,
      and assert the values still land on the right scan. An index-join passes
      the naive test and fails this one — that is the point.
- [ ] A prediction file missing one reference scan → WARNING + counted + skipped,
      not a crash, not a silent drop. Tested.
- [ ] `collapse_to_patient` reduces LUMIERE 72 → 11. Tested on synthetic data
      AND observed on real data in the smoke.
- [ ] `holm_bonferroni` reproduces a textbook worked example exactly.
- [ ] `cliffs_delta` returns +1 / −1 / 0 on fully-dominant / dominated /
      identical inputs.
- [ ] `bootstrap_ci` with a fixed seed is deterministic across runs.
- [ ] `region_masks(dilate_k=5)` gives exactly a radius-2 dilation — assert
      against a hand-computed single-voxel case.
- [ ] `vena-validation-preregister` runs on the **real local tree** and emits a
      `ring_partitions.json` whose counts match contracts §5 **exactly**:
      Ring A 321 scans / 247 patients, Ring B 146 / 146, total 467 / 393.
      **This is your end-to-end proof that the loader is correct.** If these
      numbers don't come out, something upstream is wrong — report it.
- [ ] The audit runs on the smoke subset and reproduces the verified fact that
      inside-brain ⊆ [0,1] and exterior ≡ 0 for both arms.
- [ ] Ruff clean; tests marked `validation`; import-isolation check pasted (you
      are in the main tree, so it should show the main checkout — confirm that).

## 4. Tests

`tests/validation/{__init__.py,conftest.py,test_io.py,test_registry.py,
test_regions.py,test_stats.py,test_audit.py,test_artifacts.py}`.
`pytestmark = pytest.mark.validation` at module level in every file.

`conftest.py` must provide a **synthetic on-disk H5 fixture** matching schema
2.0 (small, e.g. 4 scans × 16³) — a prediction file + a reference file with a
`references_h5` attr, including a longitudinal cohort (3 patients, 5 scans) and
a deliberately shuffled reference order. Every downstream agent will reuse this
fixture, so make it a reusable, well-named fixture, not a local helper.

No real data, no GPU, no network in unit tests. The real-data smoke is the
`preregister` run above.

## 5. Report back

`STATUS: DONE` plus: the module/symbol table (the three parallel agents code
against it — this is the most important part of your report), the
`ring_partitions.json` counts, the RSS measurement, the test summary, and
anything in your plan you found to be false.
