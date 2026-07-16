# Phase-2 Validation — Orchestration Ledger

*Orchestrator: Fable 5. Opened 2026-07-16. This is the master plan and the
running record. Task plans live beside it; `01_SHARED_CONTRACTS.md` is the
read-first substrate for every agent.*

## Goal

Build, test, and run the **Phase-2 analysis layer**: read the 360 frozen
prediction H5s + 40 reference H5s produced by Phase-1 inference (289 GB, 16
methods × 8 cohorts × 45 method-NFE pairs), compute the validation-proposal §4
metric suite under the §6 pre-registered statistical plan, and emit
self-contained artifact folders under
`/media/mpascual/Sandisk2TB/research/vena/results/fm/inference/analyses/`.

Phase 1 = predictions only, no metrics. Phase 2 did not exist. This is Phase 2.

---

## Facts established before planning (verified, not assumed)

| Fact | Value |
|---|---|
| Results tree | Local `…/results/fm/inference` **288 G** · Picasso `~/execs/vena/inference` **289 G**. Both complete. |
| Inventory | **360** prediction files + **40** references = 16 methods × 8 cohorts × **45 (method,NFE) pairs** |
| Corrected | An explore probe claimed NFE=100 missing for C4/C5. **False** — `nfe_100.h5` exists in all 8 cohorts. 45 pairs × 8 = 360 ✔ |
| Scans / patients | Ring A **321 / 247**, Ring B **146 / 146**, total **467 / 393** |
| Read throughput | 0.076 s/vol, **709 MB/s** decompressed from the SanDisk → a full pass is I/O-cheap (~0.4 h single-threaded for 21,015 reads) |
| Local disk | **134 G free (93% full)** → Phase 2 persists scalars/CSVs/PNGs only, **never a derived volume** |
| MeningD2 | **NOT mounted locally** → corpus H5s (multi-label tumour GT) reachable only on Picasso → §4.4 cannot run locally |
| Picasso | reachable (key auth); repo at `1ad2ba4` = local main; env `fscratch/conda_envs/vena`; `cpu_partition` 335 nodes, `gpu_partition` a100 |
| Deps | monai **1.5.2** (`MultiScaleSSIMMetric` ⇒ **MS-SSIM-3D free, no new dep**), statsmodels 0.14.6, scipy 1.15.3, sklearn (`mutual_info_regression` **IS** the KSG estimator), seaborn, torch 2.12. **nnunetv2 NOT installed.** |
| Reuse found | `ImageMetrics`, `spearman_with_bootstrap_ci` (canonical), `render_comparison_figure` + `select_content_slices`, `_make_run_dir`, `validate_predictions` |
| Worktree isolation | **`cd <wt> && PYTHONPATH=<wt>/src <vena-python>`** — verified empirically. No conda cloning needed. |

### The isolation finding (why no env cloning)

`vena` is an editable install path-pinned to the main checkout via a setuptools
`_EditableFinder` at `sys.meta_path[4]` — **after** `PathFinder[3]`, so
`sys.path` wins. `sys.path = [cwd, *PYTHONPATH, *site-packages]`. Therefore
`cwd=<worktree>` shadows `routines` and `PYTHONPATH=<worktree>/src` shadows
`vena`. Tested both ways.

**The trap:** omit `PYTHONPATH` and `routines` loads from the worktree while
`vena` loads from the main checkout — a silent split-brain import, green tests
against code you didn't write. Every parallel agent runs a mandatory
import-isolation assertion as its first command.

This saves 3 × 7.9 G conda clones and ~30 min of setup.

---

## Decisions taken (user-approved 2026-07-16) — do not re-litigate

1. **Test family.** VENA = `VENA-S1-v3b-rw`. Holm family (n=8) = C0-Identity,
   C1-pGAN-**t1pre**, C2-ResViT, C3-SynDiff-**t1pre**, C4-3D-DiT, C5-T1C-RFlow,
   C6-3D-LDDPM, C7-3D-Latent-Pix2Pix. Ablation family (n=3, separate correction)
   = v3b (region-weighting ≈ the proposal's "A1"), v3a (mask conditioning),
   S3-LPL-b2c (LPL null). The t2/flair panels are supplementary, in no family.
   *T1pre is choosable a priori (canonical pre→post CE direction), so it does
   not burn the P3 pre-registration the way test-selected "best panel" would.*
2. **§4.4 segmenter.** Pretrained fixed instrument (MONAI `brats_mri_segmentation`
   SegResNet), report Dice_real / Dice_synth / **Δ**. Deviates from proposal
   Appendix A; Appendix A's confounder is a *level* effect that cancels in the
   paired Δ, which is the quantity §4.4 actually claims. No nnU-Net install.
3. **BraTS-PED.** Build Phase 2 now; backfill its Phase-1 inference in parallel
   via a dedicated agent. Phase-2 code **discovers cohorts from disk** so PED
   flows in with no code change.
4. **Compute.** Dev + smoke locally on the SanDisk; **full sweep on Picasso**
   (`cpu_partition` for §4.2/§4.3, `gpu_partition --constraint=a100` for §4.4).
   Only place §4.4 can run.

---

## Schedule

```
                    ┌──────────────────────────────────────────┐
  NOW ─────────────►│ P1  brats-ped-backfill   (worktree)      │──► Picasso inference
                    └──────────────────────────────────────────┘    (independent, ~days)
  NOW ─────────────► T0  validation-core       (MAIN TREE, alone)
                          │ merge + verify by orchestrator
                          ▼
       ┌──────────────────┼──────────────────┐
       ▼                  ▼                  ▼
  V1 paired_fidelity  V2 spatial_residual  V3 downstream_seg     (3 worktrees, parallel)
       └──────────────────┼──────────────────┘
                          ▼  merge serially, re-test each time
                    T2  statistical synthesis + Picasso full sweep   (orchestrator)
```

| Task | Plan | Lane | Isolation |
|---|---|---|---|
| **T0** validation-core | `02_validation_core.md` | `src/vena/validation/**`, `routines/validation/preregister/**`, `tests/validation/**`, **`pyproject.toml`** | main tree |
| **V1** paired_fidelity | `03_paired_fidelity.md` | `src/vena/validation/metrics_paired.py`, `routines/validation/paired_fidelity/**`, its 2 tests | worktree |
| **V2** spatial_residual | `04_spatial_residual.md` | `src/vena/validation/spatial_residual.py`, `routines/validation/spatial_residual/**`, its 2 tests | worktree |
| **V3** downstream_seg | `05_downstream_seg.md` | `src/vena/validation/downstream_seg.py`, `routines/validation/downstream_seg/**`, its 2 tests | worktree |
| **P1** brats-ped-backfill | `06_brats_ped_backfill.md` | `routines/fm/inference/configs/picasso_ped_*.yaml` + a new launcher (**new files only**) | worktree |

### Lane analysis

V1/V2/V3 lanes are **file-disjoint**. The one shared file they would otherwise
collide on is `pyproject.toml` (`[project.scripts]` + the `validation` pytest
marker) — **T0 registers all four console scripts up front**, so no parallel
agent ever touches it. Entry points are lazy, so registering scripts whose
modules don't exist yet still installs cleanly.

P1's lane (`routines/fm/inference/configs/`) is disjoint from all Phase-2 lanes,
so it runs concurrently with everything from the start.

`CLAUDE.md`, `.claude/rules/**`, `src/external/**`, and
`render_comparison_figure` are off-limits to every agent — the orchestrator
updates docs at the end.

### Why T0 is alone and sequential

Its API is a contract three agents code against simultaneously. A signature
change after fan-out costs three merge conflicts. Its `iter_scans` join and
`collapse_to_patient` are the two places where a bug produces *plausible wrong
numbers in every downstream table* rather than a crash.

---

## The scientific traps I am holding the agents to

Documented in `01_SHARED_CONTRACTS.md` §11. The load-bearing ones:

1. **Double normalisation.** Phase 1 already harmonised
   (`percentile_normalise(lower=0.0, upper=99.5, foreground_only=True)`,
   exterior zeroed; verified on disk). Re-applying it silently changes every
   number. `data_range=1.0` fixed, never derived.
2. **Patient-level collapse.** LUMIERE is 4.5% of Ring-A *patients* but 22% of
   *scans*. Every paired test and bootstrap is per patient. Getting this wrong
   is anti-conservative — it inflates significance.
3. **Join by `scan_id`, never row index.** UCSF-PDGM happens to be aligned; that
   is luck, not a contract. T0 must prove it with a shuffled-reference test.
4. **Region-restricted SSIM.** `ImageMetrics.ssim` mean-fills outside the
   region; the rules themselves call it "a rough training-time proxy [that]
   degenerates on tiny regions". A WT is often <1% of the volume. V1 must use
   the SSIM **map** averaged in-region, or bounding-box SSIM — and justify it.
5. **Mixed BraTS label conventions.** Cohorts self-declare `label_system`:
   BraTS2021 → {1,2,**4**}; BraTS2023 → {1,2,**3**}. **Confirmed live:
   BraTS-PED is BraTS2023.** Hard-coding `label==4` silently zeroes ET-Dice on
   those cohorts.
6. **The shuffle null (§4.3.5).** The proposal's `--shuffle-null 1000` × 21,015
   volumes is infeasible *and unnecessary*: `E[ρ_S]=0` and `E[Conc(q)]=1` are
   **analytic**; the shuffle only estimates variance, and at ~1e6 voxels the
   null is tight. ~100 shuffles, **empirically justified by a convergence
   check**, logged.
7. **KSG MI cost.** k-NN in 2D at n≈1e6 × 21,015 is infeasible. Subsample
   ~20–50k voxels (unbiased for a per-voxel-distribution statistic), log it.
8. **Designed sanity anchors.** C0-Identity must be beaten by every real method
   inside the WT, and must have the *highest* `Conc(5%)`/`ρ_S` under C-noT
   (it is the bright-region error ceiling by construction), and the *largest*
   Δ-Dice. `Dice_real` must land ≈0.85 WT / 0.70–0.85 ET — near-zero means a
   wrong channel order. Each is a canary that fires before a wrong number
   reaches a table.

## Fairness limitations inherited from Phase 1 (state, do not fix)

- ⑤ VENA sampling was unseeded → predictions are not bit-reproducible;
  cross-NFE draws differ. Unfixable without re-running inference.
- ① Input-modality mismatch: VENA/ResViT see 3 modalities, C4–C7 see 2
  `{t1pre,flair}`, pGAN/SynDiff see 1. Confounds "only the generative
  formulation differs".
- ② VENA-S1-v3b/v3b-rw receive the ground-truth WT mask; no competitor does.
  **`VENA-S1-v3a` (no mask) is the no-oracle comparator — report it beside the
  headline row every time.**
- ⑥ pGAN/SynDiff ran as one-to-one single-source panels, under-conditioned vs
  the 3-input methods.

---

## Orchestrator protocol

- **Never edit a delegated agent's code.** Fixing it myself means I mis-scheduled.
- **Never trust a closing note. Re-run every check** in the agent's own worktree.
- **Merge serially**, re-testing the merged tree each time; branches were cut
  before earlier merges.
- **Two `SendMessage` rounds max** per defect, then escalate to the user.
- Merging and doc updates stay with me.
- Verify the baseline suite is green before fan-out and compare after each merge.

## Log

| When | Event |
|---|---|
| 2026-07-16 | Read proposal (1397 ll), fairness audit, results README, task-orchestrator skill. 3 explore agents → inference code map, reuse/dep inventory, on-disk probe. |
| 2026-07-16 | Verified: 360+40 files, 45 method-NFE pairs (corrected a false probe claim re NFE=100), 467 scans / 393 patients, read throughput, 134 G free, MeningD2 unmounted, Picasso has both copies + corpus H5s, BraTS-PED image+latent already encoded (`label_system: BraTS2023`). |
| 2026-07-16 | Proved the PYTHONPATH worktree-isolation recipe empirically; no conda cloning needed. |
| 2026-07-16 | 4 decisions taken with the user (family / segmenter / PED / compute). Plan files written. |
| 2026-07-16 | Installed `ruff 0.15.21` into `vena` (already declared in the `dev` extra; torch/monai unperturbed). Baseline: **943 passed, 4 deselected**; **475 pre-existing ruff errors, 70 unformatted files** → lint scoped to each agent's own files (contracts §14). |
| 2026-07-16 | Baseline recorded; **T0 + P1 launched.** |
| 2026-07-16 | **T0 returned DONE; verification rejected it.** Kept: `iter_scans` streams (RSS flat after 123 MB warm-up), the `scan_id` join + its shuffled-reference test (`test_io.py:125`), a complete `stats.py`, `preregister` reproducing 321/247 · 146/146 · 467/393 on real data, 968 passed. **Rejected — 4 defects:** ① `registry.py` is a stub (all module dicts empty at import; `method_role` is a `startswith("VENA-")` 2-way heuristic → headline indistinguishable from ablations, and the 4 supplementary panels lumped into the family ⇒ **Holm over 12 instead of 8, i.e. every p-value in the paper wrong and plausible-looking**). The pre-registration must be pinned in code, not name-derived. ② `regions.py` missing `bg_undilated` (§4.2's region) — trap #8 materialised. ③ `plotting.py` missing `annotate_significance` + palette/order (the user's mark-significance requirement). ④ no `tests/validation/conftest.py` — the shared fixture all three V-agents are told to reuse. One correction round sent (round 1 of 2). |
| 2026-07-16 | **P1 smoke verified by the orchestrator, not by the agent.** Job 1597915 COMPLETED (5:16, MaxRSS 6.16 G). 16 files, one per method at its correct `selection_nfe`; **BraTS-PED only** — proves the `cv_test: []` / `test_only: [PED]` filter on real execution (cross-read `engine.py:355/367/376`). `ring=B`, schema 2.0, `references_h5` resolves, scan/patient ids populated, harmonisation holds on both arms, **both validators CLEAN**. Lane clean (430 ins / 7 files / 0 mods); production tree intact at 360+40. P1 resumed → production shards. |
