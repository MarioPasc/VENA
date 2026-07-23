# Orchestrator session plan — VENA iter-8 (segmenter + mask-injection)

**Status:** ACTIVE (authored 2026-07-22).
**Owner:** the `/orchestrate` orchestrator. Workers never edit this file.
**Design authority:** `../segmenter_conditioning_design.md` (Part A injection, Part B segmenter, §A.8/§B.f iter-8).
**Task ledger:** `00_INDEX.md` (code-dependency graph) + the `NN_*.md` specs + `01_SHARED_CONTRACTS.md` (fact sheet).

## How to use this file (orchestrator contract)

1. At session start, find the **first unticked row** of the master table. That row is your session. Its task
   structure (`∥` / `→` / `[O]` below) overrides your own slot-filling; deviate only with a stated reason in the
   session's notes.
2. Run the `/orchestrate` preflight first (clean tree, baseline `pytest -m "not slow and not gpu"` + `ruff` counts,
   `df -h /`, `ssh picasso 'sacct …'` for anything already running). Session-specific preconditions are under *Gates*.
3. Each task's own `NN_*.md` is the authority on its acceptance + tests. Give each worker exactly: its `NN_*.md`,
   `01_SHARED_CONTRACTS.md`, its lane, and what it must not touch. **Verify every reported number against the
   artifact on disk** — a worker's report is a hypothesis (`/orchestrate` §3).
4. At session end: tick the row, and append to that session's **Orchestrator notes** (append-only) — what closed,
   what didn't and why, premises refuted, baseline numbers, and anything the *next* session must know.

**Notation.** `A ∥ B` — parallel (isolated worktrees, disjoint lanes, ≤3 workers). `A → B` — B starts only after A
is merged green. `[O]` — orchestrator-only, main tree, no workers running. `(gate)` — a human/quantitative gate.

## The arc

**Phase 1 (oracle upper bound, NO segmenter):** derive the SDT-soft `[WT,NETC]` from **GT** labels, validate it
(visual + latent-embedding), wire the v3a + fresh-ControlNet injection, launch 2–3 oracle runs, and decide whether
the injection mechanism is good enough. **Phase 2 (deployable):** build + train the segmenter, cache **predicted**
masks (a one-line `mask_source` swap from the oracle), train the deployable T-06 arm, report the oracle→predicted
gap. **Phase 3:** deferred ablations (CFG, WT up-weight sweep, SPADE).

> **🔧 ITER-9 PARALLELISATION (2026-07-23).** The SEG track (segmenter: **S4** build → **S5** train) is
> **code-independent** of the INJECT/oracle track (S1→S2→S3) — `00_INDEX.md` states the two tracks "share nothing
> at code level and can run fully in parallel," and the code confirms it (S4 tasks 11/13/14/15 depend only on task
> 10, **merged**). **S4/S5 may therefore launch NOW, concurrent with the oracle** (S1 cache + S2 oracle matrix), so
> Picasso trains the oracle *and* the K+1 segmenter models at the same time. The two tracks only **meet at S6**
> (predicted-mask cache + deployable T-06), which needs BOTH S3 (recipe) and S5 (trained segmenter). The old
> **"S4 gate = S3 verdict GO" is REMOVED** — it was a strategic risk-gate, not a code dependency; the segmenter is
> needed for the deployable arm AND for the Q6 OOD-coupling contribution regardless of the oracle verdict, and the
> `[TC,NETC]` mask semantics are LOCKED so S3 cannot invalidate the segmenter target.
> **~~The ONE hard S4 blocker is the BSF checkpoints~~ RESOLVED — BSF ckpts are present + pinned** (`src/external/LINKS.md`,
> local + Picasso). **S4 is DONE (2026-07-23, commits `eab69b8`+`6f281db`):** all three arms green, Arm B UKB loads
> 125/142 = 0.880, Arm A BraTS 182/198 = 0.919 (after matching `depths=(2,2,6,2)`). **GPU note:** S2's 5 oracle jobs +
> S5's K+1 SwinUNETR jobs run concurrently — size the Picasso allocation for both (the SwinUNETR jobs are smaller:
> 3-ch image-res, ~K+1≈6 models).

## Canonical Picasso paths & immutability (LOAD-BEARING — read before S2 / S5 / S6)

**v3a warm-start source — READ-ONLY, NEVER ALTER:**
```
/mnt/home/users/tic_163_uma/mpascual/fscratch/checkpoints/vena_project/2026-06-24_16-00-46_s1_v3a_concat_only_fft_ef000c9f/
```
- The oracle runs warm-start from this directory's `ema_best.ckpt`; the joint jobs (J1–J4) additionally read its
  `trunk_ema_snapshot.pt`. **Treat the whole directory as a frozen input** (`.claude/rules/external-deps.md`): never
  write, move, rename, or overwrite anything under it — not the ckpt, not a sibling file, not on "resume". Every
  T-13/T-06 YAML sets `run.resume_from` to the **absolute path** of this dir's `ema_best.ckpt`; WARM_START is
  **weights-only**, so the source is opened read-only and copied into fresh optimiser/EMA state.
- **⚠ If any tool would write into this path, STOP** — that is a `PREMISE-FALSE`/BLOCKED report, not an action.

**New runs instantiate HERE — writable experiments root:**
```
/mnt/home/users/tic_163_uma/mpascual/execs/vena/experiments/<run_id>/
```
- The FM engine creates a fresh `experiments/<run_id>/` per oracle job (`checkpoints/`, `logs/`, `metrics/`,
  `exhaustive_val/`, `decision.json`). All run output lands here; **nothing** the runs produce goes near the v3a
  source dir. Confirm the engine's experiments-root resolves to this path on Picasso before launch (the run YAML /
  launcher sets it), and that `run_id` does not collide with an existing dir.

## Master checklist

| ✓ | Session | Goal | Task structure | Gates |
|---|---|---|---|---|
| ☐ | **S1 — Oracle soft-mask + validation** | SDT-soft GT `[WT,NETC]` cached in every latent H5, visually + latent-embedding validated | `[O]` preflight → **10** → { **12** ∥ **16** } → **19**(source:gt) → **40**(mask QC + latent embedding) → `(gate)` human mask review | latent H5s writable; scaffold decision |
| ☐ | **S2 — Injection + launch oracle** | v3a + fresh 2-ch ControlNet wired; **5 oracle runs** launched + monitored | { **20** ∥ **21** } → **40**(injection sanity) → `[O]` loginexa smoke → `[O]` launch the 5-job matrix + Monitor | S1 masks cached+validated; v3a ckpt (+`trunk_ema_snapshot.pt` for J1–J4) on Picasso |
| ☐ | **S3 — Oracle verdict** | Injection-sufficiency verdict + region-weight/trunk pick + go/no-go for the segmenter | `[O]` harvest → `[O]` analysis (PSNR_ET / no-regression / FP-safety) → `[O]` verdict | S2 jobs terminal |
| ☑ | **S4 — Segmenter library** *(runs parallel to S1–S3, iter-9)* | BSF-SwinUNETR + SegResNet, loss, data/K-fold, metrics — built + unit-green | { **11** ∥ **13** ∥ **14** ∥ **15** } | S1 task 10 merged ✓; **BSF SSL located+pinned (UKB-SSL=headline)**; ~~S3=GO~~ removed → **FULLY UNBLOCKED** |
| ☐ | **S5 — Segmenter training + ensemble** *(may overlap S2 oracle on Picasso)* | K+1 models trained; G-SEG report; **calibration MEASURED (Q5: no temperature)** | **17** → **18** → `[O]` K+1 Picasso array + Monitor | S4 merged green |
| ☐ | **S6 — Predicted mask + deployable T-06** | `masks/tumor_latent_pred` cached; T-06 arm trained; oracle→predicted gap reported | { **19**(source:predicted) ∥ **22** } → `[O]` cache → `[O]` T-06 launch (reuse 20, `mask_source:predicted`) + Monitor | S5 G-SEG passed |
| ☐ | **S7 — Deferred levers & ablations** | CFG / WT-weight sweep / SPADE-T07 — optional, post-validation | { **30** ∥ WT-weight sweep ∥ SPADE-T07 } (any/none) | explicit human opt-in |

---

## S1 — Oracle soft-mask + validation  *(Phase 1a — the user's "step 1: derivation, step 2: validation")*

**Sequence.**
1. `[O]` **Scaffold decision (surface, non-blocking):** task 10 builds the *whole* `SegmentationConfig`
   (model/data/loss/metrics too), but S1 only exercises targets+derivation. Build the full scaffold now (it is
   cheap and S4 reuses it) unless the human prefers a minimal targets-only scaffold. Record the choice.
2. **10** (Wave 0) — package skeleton + Pydantic config + registries + `segmentation` marker. Merge before 12/16.
3. **Fan out (2 workers):** **12** (soft targets: SDT→sigmoid, per-component/geodesic NETC, harmonise, nesting) ∥
   **16** (derivation: per-class temperature + `pool_to_latent` to `(60,60,40)` + ensemble; S1 uses the **pool**).
4. **19** (source-agnostic derive+cache, run with `source: gt`) — writes `masks/tumor_latent_soft (N,2,60,60,40)`
   into every cohort latent H5 via the shared `vena.data.h5` writer/validator. **The swap guarantee lives here.**
5. **40** — visual QC + **latent-embedding visualization** (Q4 resolved: **per-patient PCA/UMAP of mask-latent
   vectors** coloured by tumour volume/cohort, **+ a slice montage**: one patient per row spanning small→large
   tumour size, **5 columns = 5 tumour-bearing slices**, each = anatomy slice with the soft `[WT,NETC]` mask
   overlaid at **α = 0.7**).
6. `(gate)` **Human mask review:** eyeball `artifacts/validate_masks/LATEST/report.md`. This gates the S2 GPU launch.

**Gates.** Latent H5s are writable (Picasso + local mirror); the crop/registration convention for
`masks/tumor_latent` is confirmed (task 19 read-and-verify). **Q1 resolved: SDT-graded oracle** (SDT→sigmoid→avg-pool
of GT — task 19 as specced).

**Exit criteria.** `masks/tumor_latent_soft (N,2,48,56,48)` present + `assert_*_valid` green in every cohort latent
H5, oracle `masks/tumor_latent` byte-untouched; `NETC_soft ≤ WT_soft`; the QC + embedding figures render and the
human review says the masks are localised and graded; `segmentation` marker registered; fast suite green, ruff clean
on touched files. *(grid corrected (60,60,40)→(48,56,48) — see 🔴 note below.)*

**Orchestrator notes (append-only).**
- **2026-07-22 — S1 in progress (`/orchestrate`, Opus 4.8 @ xhigh).** Base commit `9c90f78`. Baseline pytest
  1176 passed / 1 skipped; ruff 478 pre-existing (not ours).
- **🔴 GRID ERRATUM (LOAD-BEARING).** The served latent grid is **`(48,56,48)`, NOT `(60,60,40)`**. Verified vs
  Picasso disk (`latents/* (N,4,48,56,48)`, `masks/tumor_latent (N,3,48,56,48)`), the producer
  (`data/h5/latent_domain/manifest.py`: `LATENT_SPATIAL=(48,56,48)`, crop `(192,224,192)`, avg-pool stride 4),
  and the v3a warm-start config (`base_img_size_numel=129024=48×56×48`). `01_SHARED_CONTRACTS.md` §Geometry + every
  `NN_*.md` spec say `(60,60,40)` — WRONG (the fact sheet even flags the v3a 129024 contradiction but mislabels
  `(48,56,48)` "stale"). `(60,60,40)` survives only in a stale `data/h5/lightning/data.py` docstring. Every mask =
  `(2,48,56,48)`. Erratum handed to every downstream agent. **`01_SHARED_CONTRACTS.md` + the other specs still carry
  the wrong number — fix at session close.**
- **Task 10 (scaffold) MERGED** (`82d4675`, grid fix `bf78004`). Frozen `SegmentationConfig` + all sub-configs +
  decorator model registry + `segmentation` marker; `latent_grid` default corrected to `(48,56,48)` with a live
  drift-guard test vs `LATENT_SPATIAL`. +18 tests. Verified: import resolves, defaults exact, marker registered.
- **Task 12 (targets/) MERGED** (`d0ec86a`). SDT→sigmoid soft `[WT,NETC]`; per-component euclidean NETC does NOT
  bridge disjoint lesions (independently re-derived: mid-gap 0.065 < 0.5 vs naive 0.5; interior 0.79); harmonise
  BraTS2021 `{0,1,2,4}` + BraTS2023 `{0,1,2,3}`; nesting `NETC≤WT`. +33 tests. `scipy`/`scikit-image` already deps.
- **Task 16 (derivation/) MERGED** (`1e38a10`). Per-class temperature (independently re-derived on realistic
  miscalibration: `T_WT=1.42`, `T_NETC=0.68`, NLL down, argmax-preserving — the reported `T≈1301` was a pathological
  50%-error synthetic, correct); `pool_to_latent`→`(2,48,56,48)` reusing the exact `masks/tumor_latent` crop-then-pool
  (`apply_crop_pad` via `vena.common` + `avg_pool3d(k=4)`), registration exact; K-fold `ensemble_soft`. +28 tests.
- **Task 19 (mask_derive routine) MERGED** (`3b8ce78`). `derive_latent_soft_mask(source=gt|predicted)` — one code
  path, GT & predicted both `(2,48,56,48)` (the swap guarantee). Additive latent-H5 schema **2.1.0**
  (`masks/tumor_latent_soft`/`_pred` optional-but-validated groups; un-processed 2.0.0 H5s still validate). +8 tests
  (swap-invariance, H5 write+validate, idempotency, oracle byte-identical, registration centroid). **Out-of-lane but
  CORRECT:** de-exported the heavy `LatentH5Converter` from `data/h5/latent_domain/__init__` (was pulling MAISI model
  code into every importer, breaking the "no heavy import" rule); both real callers (`encode`, `offline_aug`) already
  use `.convert`. Verified: all affected modules import; the one data/h5 failure was a PRE-EXISTING `@slow` broken test
  (`test_ucsf_image_convert_smoke`, fails on main too, deselected from the fast suite); the 2 ruff F841 in
  `encode/maisi/engine.py` are pre-existing on main.
- **Suite green at 1263** (1176→1194→+33→+28→+8); nothing deleted/skipped; new files ruff-clean.
- **Data-flow verified for 19/40:** rows align **by id** (image H5 `ids` ↔ latent H5 `ids`); `crop/origin (N,3)` lives
  only in the image H5; image H5s LOCAL (MeningD2), latent H5s PICASSO-only. `umap` absent → task 40 uses PCA (sklearn).
- **Task 40 (validate_masks) MERGED** (`3143c61`). QC 3-row + pinned montage + PCA embedding (umap absent → PCA) +
  `render_injection_sanity` (unit-tested on synthetic residuals; real panel = S2). +19 tests. **Suite green at 1282.**
- **Local figure pass DONE + orchestrator-verified** (UCSF-PDGM, 12 patients, random spread) →
  `/media/.../results/prior/tumor/gt/2026-07-22T14-34-14Z/` (montage.png, embedding.png, 12 qc_*.png, report.md,
  decision.json). Machine stats: `netc_violation_count=0`, `empty_mask_count=0`, `soft_mass_fraction_in_wt=0.148`.
  **Visual check (I viewed the PNGs):** montage masks localised on focal lesions, ordered by ascending tumour volume,
  NETC (magenta) nested inside WT (white); QC latent-grid row shows a smoothly GRADED (48,56,48) WT heatmap with a
  smaller nested NETC blob, correctly registered. Oracle derivation is correct on real data. `masks_look_valid=null`
  (awaiting HUMAN review — the gate is the user's to close).
- **Picasso cache LAUNCHED — SLURM array `1629508` (9 tasks, one per cohort), RUNNING on `cpu_partition`.** Writes
  `masks/tumor_latent_soft (N,2,48,56,48)` into each latent H5 (exit-criterion-1). Idempotent; monitor armed for the
  terminal state. **Cost finding: the image-res SDT dominates at ~14 s/patient** → sequential would be ~14 h, so
  sharded by cohort (wall-clock ≈ BraTS-GLI ~5 h; small cohorts finish in minutes). An interactive IvyGAP verify hit
  the ssh timeout mid-derive but left the H5 byte-untouched (engine derives all-in-memory, writes at the end), so no
  partial/corrupt state. `mask_derive` needs a MINIMAL registry (`{name,image_h5,latent_h5}`, extra=forbid) — the
  standard corpus JSON won't parse. Cfgs + array worker in scratchpad + `~/execs/vena/mask_derive_cfg/`.
- **Docs corrected (commit `9eed6c1`):** the stale `(60,60,40)` → `(48,56,48)` in `01_SHARED_CONTRACTS.md`
  (§Geometry + erratum banner), `00_INDEX.md`, and `20_inject` (whose base_img_size_numel "mismatch" advice was
  wrong — `129024=48×56×48` is CORRECT, no reconciliation). Completed-spec bodies (10/16/19/40) keep their historical
  `(60,60,40)` — code + fact-sheet are canonical. Memory `reference_latent_grid_48_56_48` written.
- **S1 status:** all 5 code tasks merged (suite 1282); oracle masks produced + visually verified (localised, nested,
  graded); figures in the Sandisk GT dir awaiting **human review**; cache array running. S1 row ticks once the array
  is terminal + all 9 H5s validated (`masks/tumor_latent_soft` present, oracle byte-identical) and the human closes
  the mask-review gate.
- **Viz feedback round (2026-07-22, Opus-4.8 agent, merged `8ffd858`, seg suite 110).** Scientist review of the QC
  figures surfaced issues; root-caused + fixed (VIZ only — derivation untouched):
  - **Calibration is CORRECT, not a derivation problem** — verified `soft@0.5 volume / hard-tumour volume = 1.000`
    (globally + at max-area slice; UCSF 0290 max slice: hard NETC=34, soft NETC>0.5=34). The "prob where no tumour"
    was a slice-selection artifact + the graded halo rendered by the heatmap.
  - **Slice-selection bug:** `render_mask_qc` picked the peak-VALUE slice (`argmax(wt.max(axis=(0,1)))`, ~32-way tie
    → arbitrary) and argmaxed image vs latent INDEPENDENTLY → different z-levels. Fixed → tumour-AREA
    (`sum(axis=(0,1))`) for both, so anatomy + latent rows show the same widest slice.
  - **NETC-hard bug (pre-existing engine):** `hard_mask = (soft[0]>0.5)` passed a WT BINARY; `render_mask_qc` did
    `==1` → whole WT (3136 vox) as "NETC hard" instead of the true necrotic core (34). Fixed → thread the true
    integer BraTS label via `PatientView.hard_label`.
  - **Rendering:** continuous soft overlay (opacity ∝ prob) + contour rings at 0.25/0.5/0.75; high-contrast colors
    **WT=green (0.1,0.9,0.2) / NETC=magenta (1.0,0.1,0.6)** (retired white-ish `hot`/`cool`); montage α=0.6, 10 cols.
  - Re-rendered + I viewed: all 3 QC rows (hard/soft/latent) now agree; NETC nested; gradation explicit. Authoritative
    UCSF figures re-run into a fresh GT-dir timestamp; the buggy `2026-07-22T14-34-14Z` set is superseded.
- **🔴 CHANNEL-0 = TC, not WT (2026-07-22, Opus-4.8, merged `447e416`; full suite 1298).** Scientist caught (via
  `qc_0159`) that WT includes non-enhancing edema. Verified: **81% of WT is edema** (0159 74%, 0290 96%, 0533 77%),
  so `channel0−NETC = WT−NETC = ET+ED` was edema-dominated → the model was told to enhance mostly-edema (a driver of
  S1 tumour-skip). **Fix: channel 0 = TC (tumour core = NETC+ET = `(label>0)&(label!=2)`), excludes edema;
  `TC−NETC = ET` = true enhancing region.** Config-driven `TargetConfig.tumor_region: {wt,tc}="tc"` (wt kept for S7
  ablation). Files: config/harmonise/soft_targets + viz labels (WT→TC, hard-panel excludes edema) + `mask_derive`
  records a `tumor_region` attr. WT cache (job 1629508) **cancelled + superseded**; **re-cached all 9 with TC = job
  1630643** (monitor armed). Figures re-rendered (TC) → `2026-07-22T20-40-15Z` (verified on 0290: 96%-edema tumour now
  a tiny TC core, not the WT blob; hard TC == soft TC>0.5 exactly). Docs: fact-sheet erratum banner + this note +
  memory `[[project_channel0_tumor_core_not_wt]]`. **S2 carries this:** `m_wt_soft`→`m_tc_soft`, `[WT,NETC]`→`[TC,NETC]`,
  region-loss WT→TC, **Phase-2 segmenter target = TC** (G-SEG WT-Dice gate must become a TC-Dice gate). **Use PSNR_ET.**
  Segmenter specs (11/13/15/17/18) + config updated: predict `[TC,NETC]`, G-SEG **TC-Dice ≥ 0.75 provisional**
  (`MetricsConfig.gseg_wt_dice`→`gseg_tc_dice`, re-derive in S5 — TC harder than WT), `T_TC` temperature.
- **🔴 QC-round-2 + CACHE CLEAN REDO (2026-07-22, Opus-4.8 agent, merged `1b9f946`, seg suite 137).** Scientist flagged
  QC inconsistencies (mask in one row not another; soft looks hard). Root causes found — **one a real bug I MISSED**:
  - **Figure bug (engine):** `validate_masks._build_patient_views` set `soft_img = stack([(label>0),(label==1)])` — a
    **BINARY WT** — and passed it as `soft_mask_img`; so the montage + QC "soft" rows rendered binary WT, NOT soft TC.
    My earlier "figures look correct" was compromised (I verified the derivation data, not that the figure rendered it).
    Fixed → `make_soft_targets(label, cfg.targets)` (real soft TC).
  - **Slice bug:** `k_img`/`k_lat` argmaxed INDEPENDENTLY on different-res volumes → rows at different z. Fixed →
    crop-frame (192,224,192), ONE reference slice (max hard-TC area), latent upscaled ×4; all rows same slice+res.
  - Soft rows now colormap (TC=YlGn / NETC=RdPu, higher prob=brighter) + contours {0.25,0.5,0.75}; `check_mask_invariants`
    added (hard⊆soft, soft-continuous, latent-upscaled≈image-soft IoU).
  - **NO derivation/registration bug** (independently cross-checked: 0159 IoU 0.864 / 0.24 vox; crop preserves TC).
    0236/0485 are **genuinely 100% edema (TC=0)** → correctly blank. Faint "soft" = the `0.034` SDT floor
    (`sigmoid(-clip/σ)`) — a design detail (label-smoothing vs clean conditioning; relevant to G-SHORTCUT). **OPEN for the
    user:** clamp the soft floor to 0? (would need a derivation change + re-cache.)
  - **Cache clean redo (per user):** the WT cache 1629508 had written stale WT to 7 cohorts; 1630643 (TC) only
    overwrote 3 before cancel → MIXED state. Cancelled 1630643, **deleted `masks/tumor_latent_soft` from all 9 latent
    H5s + reset schema 2.0.0** (oracle+latents preserved, shapes verified), re-synced `1b9f946`, **re-submitted fresh
    all-TC cache = job 1630706** (worker `tumor_region: tc`). Monitor armed. Corrected figures re-rendering → GT dir.
- **⚠ S2 note (deferred):** aug latent H5s `*_latents_aug.h5` are NOT in the registry list; if S2 trains with offline
  aug, the soft mask must be present+consistently-transformed there too (task-20/DataModule concern).
- **🔴 CACHE 1630706 PARTIALLY TIMED OUT — 7/9 done, 2 re-submitted (2026-07-23, Opus-4.8 resume).** Job 1630706
  (the fresh all-TC cache) went terminal as **7 COMPLETED + 2 TIMEOUT**. Array-index→cohort map (from
  `array_mask_derive_gt.sh` `COHORTS[]`): 0=UCSF-PDGM✅ **1=BraTS-GLI⏰ 2=UPENN-GBM⏰** 3=IvyGAP✅ 4=BraTS-Africa-Glioma✅
  5=BraTS-Africa-Other✅ 6=LUMIERE✅ 7=REMBRANDT✅ 8=BraTS-PED✅. **Root cause = self-set `#SBATCH --time=08:00:00`**,
  not a partition cap (`cpu_partition MaxTime=7-00:00:00`). The two largest latent H5s exceed 8h serial at the TC
  per-component-SDT cost; both were CANCELLED "DUE TO TIME LIMIT" with **no Python error**.
- **H5 state verified on disk** (metadata scan, `scratchpad/scan_soft_meta.py`): **7/9 SOFT-OK** — schema **2.1.0**,
  `mask_source=gt`, `masks/tumor_latent_soft (N,2,48,56,48)`, `tumor_region=tc`, oracle `masks/tumor_latent (N,3,…)`
  intact. **2/9 (BraTS-GLI, UPENN-GBM) byte-untouched** — schema still 2.0.0, no soft group, oracle intact. Confirms
  the engine's write-all-in-memory-at-end design (`derive_engine.py::_process_cohort` line ~334): a timeout leaves the
  H5 clean, **never a partial/corrupt write**.
- **🔴 ROW-COUNT CORRECTION (load-bearing for sizing/re-runs).** Latent H5 `N` = **all cached scans**, NOT the deduped
  CV counts in `01_SHARED_CONTRACTS.md §Cohorts`. Measured `N`: UCSF-PDGM **495**, BraTS-GLI **1251**, UPENN-GBM **611**,
  IvyGAP **34**, BraTS-Africa-Glioma **95**, BraTS-Africa-Other **51**, LUMIERE **599**, REMBRANDT **63**, BraTS-PED **260**.
  Per-scan TC-derive cost varies wildly by cohort (~16 s BraTS-PED → ~47 s UPENN-GBM, driven by native volume size +
  #connected-components in the per-component euclidean SDT). BraTS-GLI (1251) ≈10-11 h, UPENN-GBM (611) ≈9-10 h serial.
- **RE-SUBMIT = job `1631539`** (`sbatch --array=1,2 --time=24:00:00 array_mask_derive_gt.sh`), Picasso repo HEAD
  confirmed at **`1b9f946`** (the correct all-TC commit that produced the 7 good cohorts), same idempotent engine +
  same `gt_{1,2}.yaml` (`tumor_region: tc`). RUNNING (both tasks) as of 2026-07-23T10:30Z. **Monitor armed** (task
  `b2lo3ahje`, all terminal states, ssh-per-poll so a dropped conn doesn't kill it). On completion the 2 H5s will
  match the other 7 (schema→2.1.0, region=tc). Then all-9 validated ⇒ S1 exit-criterion-1 met.
- **Figure render re-verified by eye** (I viewed montage + qc_0367; post `1b9f946`+`33f4751`): genuinely **soft TC**
  (contour rings, graded), localised on focal lesions, NETC nested inside TC, masks only on tumour-bearing slices —
  the binary-WT figure bug stays fixed. Two complete UCSF QC sets in the GT archive: default-bg
  `…/gt/2026-07-22T21-53-06Z/` (12 pat; **1 invariant flag = UCSF-PDGM-0367**) + flair-bg `…/gt/t2f/2026-07-22T22-11-15Z/`
  (12 pat; 0 flags). **0367 is BENIGN**: a multifocal small TC (main focus + 2 satellites); the 4×-avg-pool coarsens
  the satellites → latent↔image IoU 0.332 / centroid 4.79 vox. Pooling-fidelity limit on tiny multifocal cores, **not**
  a derivation/registration bug — the latent mask still marks the correct lesion. The other 4 "low IoU" rows are
  `has_tc_region:false` (100%-edema tumours, TC=0 → IoU=0 by construction). **`t1c/` + `t1pre/` anatomy-variant dirs
  under `…/gt/` are EMPTY** (the previous agent's runs were interrupted mid-render; the `anatomy_sequence` config from
  commit `33f4751` works — only flair completed).
- **⚠ S6 FINDING (parallelism, carry forward).** `MaskDeriveEngine._process_cohort` is a **serial per-scan loop** with
  no multiprocessing; `_derive_one` is independent per scan (embarrassingly parallel). The **S6 predicted-mask re-cache**
  (task 19 `source:predicted`) runs this SAME engine on all 9 cohorts and **will hit the identical 8h wall** on
  BraTS-GLI+UPENN-GBM. Fix for S6: either (a) set `--time ≥ 24h` for the big cohorts (zero-code, what we did here), or
  (b) add a `multiprocessing.Pool(cpus_per_task)` over the scan loop + bump `--cpus-per-task` (≈N×faster, de-risks the
  wall permanently). Recommend (b) for S6 (predicted derive is even costlier: adds segmenter inference per scan).
- **Human mask-review gate STILL OPEN** — user closes `masks_look_valid` in the QC `decision.json` after eyeballing the
  two UCSF sets above (24 patients spanning small→large→multifocal→100%-edema). Gates the S2 GPU launch.
- **✅ USER DECISION — KEEP the 0.034 soft-floor** (2026-07-23). The uniform far-field floor `sigmoid(-clip/σ)` (σ=3,
  clip=10 → 0.034) is **retained by design**: it is representative of a best-case segmenter output (high probability
  density inside the tumour, low-but-nonzero outside). No derivation change, **no re-cache** — job 1631539 stands.
  (Mechanistically also harmless to the oracle: a constant DC field the ControlNet's zero-init conv+bias absorbs.)
- **⚠ VERIFY-SCRIPT FALSE POSITIVE — BraTS-Africa-Other IS correct TC, not stale WT** (2026-07-23). The scratchpad
  `verify_tc_cache.py` (Picasso) flagged BraTS-Africa-Other as "WT (STALE 1629508)" — a **verdict tie-break artifact**:
  it picks the first patient with TC>500 vox and tie-breaks `iou_tc > iou_wt` → else "WT". Its sampled patient
  `BraTS-SSA-00009-000` is **edema-free** (TC==WT) → both IoU=1.000 → tie → spurious "WT". **Refuted** by a targeted
  re-check (`scratchpad/verify_bao_edema.py`) on two EDEMA-bearing patients (TC≠WT): `BraTS-SSA-00018-000` MAE(cache,TC)
  =0.00000 vs MAE(cache,WT)=0.00072; `BraTS-SSA-00040-000` (edema 216k vox) MAE(cache,TC)=0.00000 vs MAE(cache,WT)=0.027.
  Cache is **bit-exact TC**, `region_attr=tc`. **All 7 completed cohorts are clean TC, zero stale contamination.**
  *Trap for S6:* when re-running `verify_tc_cache.py` after the predicted cache, use **MAE-to-fresh-derive** as the
  TC/WT criterion, not the IoU tie-break (it misfires on every edema-free patient).
- **t1c QC background DONE + eye-verified** (2026-07-23, per user) — `validate_masks anatomy_sequence=t1c` on the **same
  12 patients** as the flair/t2f set (local UCSF image H5, git `33f4751`) → `…/gt/t1c/2026-07-23T08-41-50Z/` (14 figures,
  **0/12 invariant violations, reg_iou_low=0, netc_viol=0**). I viewed the montage: on the T1c post-contrast background
  the **green TC contour sits exactly on the bright enhancing core**, magenta NETC (necrotic, dark on T1c) nested inside,
  **edema excluded** — the strongest visual confirmation of the channel-0=TC semantics and the mask-on-enhancement check.
  Three QC backgrounds now available for the human gate: default (`2026-07-22T21-53-06Z`), flair (`t2f/2026-07-22T22-11-15Z`),
  t1c (`t1c/2026-07-23T08-41-50Z`). (t1pre left unrendered — not requested.)

---

## S2 — Injection wiring + region-weighting + launch oracle  *(Phase 1b)*

**Sequence.**
1. **Fan out (2 workers):** **20** (serve `masks/tumor_latent_soft` by `data.mask_source: oracle_soft`; two specs
   `mask:wt_soft:identity` + `mask:netc_soft:identity`; v3a-warm-start T-13 run YAML; loginexa smoke) ∥ **21**
   (region-weighted CFM, regions `{Brain = NOT-BG ∩ NOT-WT, WT}`, **equal weights ≡ L1** guaranteed by a test).
   **Merge 20 before 21** (both may touch `lightning/module.py`).
2. **40** (injection-sanity panel) — step-0 identity (`output_scale=0` → residual 0 → output == v3a) + residual
   locality (in-WT vs out-of-WT residual energy). Confirms P1/P2 before spending GPU-days.
3. `[O]` **loginexa smoke** — the T-13 YAML builds the 2-ch ControlNet (hint-net `conditioning_in_channels == 2`)
   and runs 2 optimiser steps (`test-picasso-loginexa` skill).
4. `[O]` **Launch the 5-job oracle matrix** on Picasso (`picasso-sbatch`) — all v3a-warm-start + fresh 2-ch
   ControlNet on the **oracle** SDT-soft mask (`data.mask_source: oracle_soft`), varying trunk policy × `region_weights`:

   | job | trunk | region_weights | purpose |
   |---|---|---|---|
   | **J0** | **freeze** | `{brain:1, wt:1}` | ControlNet-only **lower bound** — real floor (no trunk drift, no loss help) |
   | **J1** | joint-low-LR | `{brain:1, wt:1}` | equal-weight ceiling; **pairs with J0 → the freeze→joint gain** |
   | **J2** | joint-low-LR | `{brain:1, wt:5}` | RW sweep |
   | **J3** | joint-low-LR | `{brain:1, wt:10}` | RW sweep |
   | **J4** | joint-low-LR | `{brain:1, wt:20}` | RW sweep (**watch FP-safety** at the top weight) |

   Only **J0 is freeze** (no trunk EMA needed → sidesteps A.8-§5); **J1–J4 are joint** and require v3a's
   `trunk_ema_snapshot.pt`. **LR = linear warmup → cosine annealing** (v3a's `warmup_steps:1000` + `scheduler:cosine`).
   **⚠ Raise EarlyStopping patience** — the harder objective (adding enhancement) transiently *raises*
   `train/total_epoch` before it improves, so patience 250 risks a premature stop; use **~400–500** and keep every
   epoch checkpoint (exhaustive-val PSNR_ET is the real selection signal, not train loss). Strip the ANSI job-id
   (`/orchestrate` §4); verify `Dependency` not `(null)`; set `--time`/`--mem` per proposal §5.
5. `[O]` **Monitor** — one persistent watch matching every terminal SLURM state; `ScheduleWakeup` while pending.

**Gates.** S1 masks cached + human-validated; the **v3a source dir is present + READ-ONLY** (see "Canonical Picasso
paths & immutability") — `ema_best.ckpt` for all jobs, `trunk_ema_snapshot.pt` for J1–J4 (J0 freeze does not need
it, A.8-§5); every run YAML sets `run.resume_from` to that dir's `ema_best.ckpt` and writes output **only** to
`…/execs/vena/experiments/<run_id>/`, never the v3a source dir.

**Exit criteria.** loginexa smoke green (2-ch build + 2 steps, no shape error); step-0 identity verified; **5 Picasso
jobs RUNNING** (job ids recorded, `Dependency` clean); exhaustive-val cadence writing non-empty per-patient
`metrics.csv` (not the empty-CSV `use_timestep_transform` trap — verify one early epoch); monitor armed.

**Orchestrator notes (append-only).**
- _(empty)_

---

## S3 — Oracle results + injection-sufficiency verdict  *(Phase 1c)*

**Sequence.**
1. `[O]` **Harvest** the 2–3 runs' `exhaustive_val/epoch_NNN/metrics.csv` (per-patient × NFE) — copy back + content-hash.
2. `[O]` **Analysis (re-derive every headline number yourself):**
   - **PSNR_ET** vs v3a (target: recover most of the v3b−v3a ET gap) — **use PSNR_ET, not PSNR_WT** (the WT trap,
     `[[project_s1_v2_tumor_failure_diagnosis_2026_06_22]]`).
   - **No-regression gate (A.6):** `MS-SSIM_brain ≥ v3a − MCID` and `MAE_brain ≤ v3a + MCID`.
   - **FP-safety:** false-enhancement on GT-ET≈0 cases ≈ v3a (mask-gating must not *add* FP).
   - Per region-weight arm; which arm best trades PSNR_ET ↑ vs FP/whole-brain.
3. `[O]` **Verdict:** (a) is the ControlNet injection sufficient to place enhancement given a perfect mask? (b) which
   `region_weights` + trunk policy to carry forward; (c) **GO/NO-GO for the segmenter phase**. Record in a handoff.

**Gates.** S2 jobs terminal (COMPLETED/early-stopped) with ≥ several exhaustive-val cadences on disk.

**Exit criteria.** A table (arm × {PSNR_ET, PSNR_brain, MS-SSIM_brain, MAE_brain, FP-vol}) re-derived from the
per-patient CSVs; the oracle **upper bound** stated; the go/no-go + the chosen recipe written to the handoff; memory
updated with the verdict.

**Orchestrator notes (append-only).**
- _(empty)_

---

## S4 — Segmenter library  *(Phase 2a)*

**Sequence.**
1. **Fan out (≤4 workers, disjoint subfolders):** **11** (`models/` — BSF-SwinUNETR Arm A/B + SegResNet Arm C, fork
   `downstream_seg`) ∥ **13** (`engine/loss.py` — DML+CE, focal-CE, Tversky) ∥ **14** (`data/` — K-fold OOF ⊆
   FM-train + dataset + augmentation) ∥ **15** (`metrics/` — Dice/AHD + ECE/Brier + G-SEG + dual selection).
2. Each owns one subfolder → merges are near-conflict-free; run serially anyway, re-verify the suite each merge.

**Gates.** Task 10 (scaffold) merged (S1 ✓). **~~S3 verdict = GO~~ REMOVED (iter-9)** — S4/S5 are code-independent
of the oracle and run in parallel (see "The arc" parallelisation note; `00_INDEX.md` confirms the tracks share
nothing at code level). **BSF SSL checkpoints RESOLVED (2026-07-23) — located + pinned in `src/external/LINKS.md`**
(local + Picasso). **Arm priority: UKB-SSL = leak-free HEADLINE/PRIMARY** (`…_SSL_UKBiobank/64-gpu-model_bestValRMSE.pt`;
no BraTS patients, no T1ce — the patient-overlap + T1ce leak in BraTS-SSL is **unfixable by OOF** because it is an
SSL-stage leak), **BraTS-SSL = domain-matched comparator** (`…_SSL_BraTS/model_bestValRMSE-fold{0..4}.pt`),
**SegResNet-scratch = floor**, **finetuned = NEVER use** (L1+L2+L3). **S4 is now FULLY UNBLOCKED. Launch order:**
Arm-C (`segresnet`, no ckpt, fork `downstream_seg.py`) + 13/14/15 first (fastest to green), then Arm-B UKB-SSL
(headline) + Arm-A BraTS-SSL (comparator).

**Exit criteria.** All three arms forward `(B,3,·)→(B,2,·)`; BSF load-coverage reported; DML==soft-Dice-on-hard test
green; K-fold plan deterministic + leakage-free; metrics + G-SEG gate + dual selection tested; suite green, new
`segmentation` tests counted.

**Orchestrator notes (append-only).**
- **S4 CLOSED 2026-07-23.** All four lanes merged to `main` working tree, ruff-clean on touched files. Fast suite
  **1313 → 1453 passed** (+140: models 21, loss 26, data 55, metrics 38), 1 skipped, 21 deselected (17 = models
  `slow` real-BSF-load tests). Every load-bearing number was **re-derived by the orchestrator from the on-disk
  artifact**, not transcribed. **The merged code is UNCOMMITTED in `main`'s working tree** (4 re-wired `__init__.py`
  + 13 new lane files + 4 new `tests/segmentation/*` dirs) — commit before any S5 run.
- **⚠ Worktree stale-base incident (recovery, load-bearing lesson).** `isolation:"worktree"` cut lanes **11/14/15**
  from a **stale base `c424a1f`** (pre-`vena.segmentation` scaffold); lane **13** was correctly at `main` `33f4751`.
  The 3 stale workers silently **re-created** `config.py`/`exceptions.py`/`registry.py` instead of reusing them —
  caught by `git diff --stat main` showing 9218 deletions, NOT by any worker report. Fix: `git reset --hard main`
  per stale worktree (canonical scaffold restored, re-created dupes dropped, untracked lane files preserved), then
  resumed the workers via SendMessage to re-wire `__init__` + re-verify against canonical interfaces. **Rule for
  future orchestration: after spawning a worktree agent, assert `git -C $WT merge-base --is-ancestor <main-sha> HEAD`
  BEFORE trusting its output.**
- **Arm A load-coverage measured at 126/198 = 0.636 (S4, before fix).** The BraTS-SSL ckpt
  (`BrainSegFounder_SSL_BraTS/model_bestValRMSE-fold0.pt`, sha256 `e46d80ce75f3…`) produced 56 extra `layers3` encoder
  block keys absent from MONAI `SwinUNETR(fs=48, depths=(2,2,2,2))`. S4 test pinned 126/198 as provisional. Arm B UKB
  (`64-gpu-model_bestValRMSE.pt`, sha256 `4be92492ae4f…`) = **125/142 = 0.880**. (BSF ckpts ARE present locally.)
  **NOTE: RESOLVED in S5 parallel worker — see S5 orchestrator notes for the fix and updated numbers.**
- **What S5 must know (build/train the K+1 ensemble):**
  - `load_bsf_encoder(model.backbone, ckpt, brats_channel_slice=[0,1,3] for BraTS | None for UKB)` — **MUST pass
    `.backbone`** (inner MONAI SwinUNETR, keys `swinViT.*`), NOT the `_VenaSwinUNETR` wrapper (keys `_backbone.*`) or
    0 keys match. The builders already do this when `cfg.checkpoint` is set.
  - SwinUNETR forward needs spatial dims **divisible by 32** (use 32³/64³); SegResNet Arm C takes any 8-divisible size.
    Deep-sup return = `(logits, aux_H2, aux_H4)` (SwinUNETR via forward-hooks on `decoder2`/`decoder3`).
  - **MONAI 1.5.2 has NO `RandGammad`** — lane 14 substituted a 2nd `RandAdjustContrastd` pass (γ∈[0.5,2.0]).
  - `build_fold_plan(cfg, fm_splits, *, dedup_duplicates=None)` — transitive cross-cohort dedup exclusion is via the
    **optional kwarg**; S5 must plumb the real dedup map from the corpus dedup source (synthetic-only in unit tests).
  - Metrics are **MEASURED not corrected** (Q5 temperature dropped). Healthy-control gate threshold is a module
    constant `_HEALTHY_TC_VOLUME_THRESHOLD=0.01` in `gate.py`, NOT a `MetricsConfig` field (config has only
    `gseg_tc_dice`/`gseg_netc_dice`/`selection_metric`, `extra="forbid"`). `gseg_tc_dice=0.75` is **PROVISIONAL** —
    re-derive from measured per-cohort TC Dice in S5.

---

## S5 — Segmenter training + K-fold ensemble  *(Phase 2b)*

**Sequence.**
1. **17** (engine: `SegTrainer` one-model-per-invocation + `predict_oof` ensemble/TTA) → **18** (train routine +
   `decision.json` + `vena-segmentation-train`).
2. `[O]` **Train the K+1 models** as a Picasso array (K fold-models + the all-FM-train model); Monitor.
3. `[O]` **G-SEG evaluation** per cohort incl. Ring B (**TC Dice ≥ 0.75 provisional** — re-derive from measured TC
   Dice; TC is harder than WT — NETC Dice ≥ 0.50; healthy → ~empty **TC** volume); **(iter-9 Q5: temperature
   DROPPED — no `T_TC`/`T_NETC` fit)** report DSC **and** Brier/classwise-ECE (calibration MEASURED, not corrected)
   + **ET=TC−NETC Dice as a reported diagnostic**. **Target/gate is TC (=NETC+ET, edema excluded), not WT.**

**Gates.** S4 merged green; GPU budget on Picasso (K+1 SwinUNETR trainings) — **may run concurrently with the S2
oracle matrix** (both tracks on Picasso at once, iter-9 parallelisation); the SwinUNETR jobs are smaller (3-ch
image-res) than the FM oracle jobs, so size the joint allocation accordingly.

**Exit criteria.** K+1 checkpoints + `fold_plan.json` (no `temperatures.json` — iter-9 Q5 dropped temperature); the G-SEG table passes (or the
documented fallback to a single coarse TC channel is invoked and recorded); no in-fold self-prediction (OOF routing
asserted).

**🎫 TICKET (S5, PARALLEL — opened 2026-07-23 from S4) — BraTS-SSL SwinUNETR config mismatch.**
S4 measured, loading the BSF SSL encoders into a **standard** MONAI `SwinUNETR(feature_size=48)` via
`load_bsf_encoder(model.backbone, ckpt, brats_channel_slice=[0,1,3]|None)`: **Arm B UKB = 125/142 = 0.880** (1-ch
stem correctly skipped — expected) but **Arm A BraTS = only 126/198 = 0.636** — 72 skipped = **56 extra
`swinViT.layers3.*` keys** + 16 SSL task-head keys absent from our build. BrainSegFounder is documented as a
*standard* Swin, so **0.636 is probably a construction-config mismatch, not a genuinely deeper architecture.**
- **Hypothesis (check first, cheapest):** BSF built its SwinUNETR with **`use_v2=True`** (MONAI SwinUNETR-v2 adds
  residual conv blocks → extra per-stage keys) and/or non-default **`depths` / `num_heads`**. Our `_VenaSwinUNETR`
  uses MONAI defaults.
- **Investigation (CPU-only, runs parallel to the K+1 training array):** dump the ckpt `state_dict` keys, diff
  against `SwinUNETR(fs=48, use_v2=True)` and against the standard build; identify the config that lifts Arm A
  coverage toward ~0.92 (only the 16 SSL heads should legitimately drop). If found, set that config for **both**
  arms (UKB is the same BSF family) and re-run the S4 load-coverage test with the corrected expectation.
- **Fallback if genuinely deeper:** keep Arm A at 0.636 as a documented comparator (the headline Arm B UKB is
  unaffected at 0.880). Do NOT block S5 training on this — it only changes Arm A's encoder-init quality.
- Files: `src/vena/segmentation/models/bsf_swinunetr.py` (`_VenaSwinUNETR.__init__`, `load_bsf_encoder`);
  caveat pinned in `src/external/LINKS.md`; context in the S4 orchestrator notes above.

**Orchestrator notes (append-only).**
- **✅ TICKET RESOLVED — Arm A BraTS depths=(2,2,6,2) (2026-07-23, S5 parallel worker).**
  Empirically inferred from checkpoint key dump (key pattern `swinViT.layers{N}.0.blocks.{i}.*`):
  - **Arm A BraTS:** `depths=(2,2,6,2)` — stage 3 has **6 blocks** (indices 0–5) vs MONAI default 2. 6−2 = 4 extra
    blocks × 14 keys/block = **56 extra keys** = exactly the 56 observed missing keys. `num_heads=(3,6,12,24)`.
    No `use_v2` markers (no residual-conv keys). With `depths=(2,2,6,2)`: **182/198 (91.9%)** matched; **16 skipped**
    = SSL task heads only (`rotation_head`, `contrastive_head`, `conv.*`). No swinViT encoder keys left out.
  - **Arm B UKB:** `depths=(2,2,2,2)` (MONAI default). **125/142 (88.0%)** unchanged (1-ch stem + 16 SSL heads).
  - **Fix implemented:** module constants `_BSF_BRATS_SWIN_KW = {"depths":(2,2,6,2), "num_heads":(3,6,12,24)}` and
    `_BSF_UKB_SWIN_KW = {"depths":(2,2,2,2), "num_heads":(3,6,12,24)}` added to `bsf_swinunetr.py`. `_VenaSwinUNETR`
    accepts `swinunetr_kwargs: dict | None = None` forwarded to `SwinUNETR(**extra)`. Builders call
    `_VenaSwinUNETR(cfg, swinunetr_kwargs=_BSF_{BRATS,UKB}_SWIN_KW)`. Decoder hooks (`decoder3`/`decoder2`) confirmed
    present on both variants. Tests in `TestRealBsfLoad` updated to pin 182/198 + add `test_arm_a_matched_fraction_above_0_80`.
  - **LINKS.md caveat** updated from `⚠ OPEN TICKET` to `RESOLVED`. T1ce removal documented at
    `.claude/notes/changes/vena_new_iteration/bsf_stem_t1ce_removal.md` (drop-slice `[0,1,3]` = current = recommended).
  - **What S5 must know (updated):** Arm A builder auto-uses `depths=(2,2,6,2)` — no YAML field, no ModelConfig change.
    Deep-supervision hooks survive the deeper stage-3 (decoder blocks are MONAI decoder, independent of swinViT depth).
    Coverage Arm A: **0.919 ≥ 0.80** (confirmed). All 182 encoder keys transfer; the 16 SSL heads drop as designed.
    No action required from S5 — the fix is self-contained in the library layer.

---

## S6 — Predicted-mask cache + deployable T-06  *(Phase 2c)*

**Sequence.**
1. **Fan out (2 workers):** **19**(source:predicted) (reuse the derive/cache routine → `masks/tumor_latent_pred`,
   temperature + K-fold ensemble mean) ∥ **22** (mask-perturbation augmentation, enabled for T-06 only).
2. `[O]` **Cache** `masks/tumor_latent_pred (N,2,60,60,40)` into every latent H5 (`assert_*_valid`; oracle +
   `_soft` groups untouched).
3. `[O]` **T-06 launch** — reuse the S2 run YAML with **`data.mask_source: predicted`** + perturbation ON + the S3
   region-weight/trunk pick. This is the **one-line swap** the architecture guarantees.
4. `[O]` **Oracle→predicted gap** — PSNR_ET(T-06 predicted) vs PSNR_ET(T-13 oracle); report as a table column
   (unreported in TA-ViT = a VENA contribution).

**Gates.** S5 G-SEG passed; S3 recipe fixed.

**Exit criteria.** `masks/tumor_latent_pred` cached + validated; T-06 run launched + monitored; the oracle-vs-predicted
gap re-derived from per-patient CSVs and recorded; G-SHORTCUT (healthy-control FP≈0) checked on T-06.

**Orchestrator notes (append-only).**
- _(empty)_

---

## S7 — Deferred levers & ablations  *(Phase 3 — explicit human opt-in)*

**Sequence.** { **30** (CFG-at-inference + noise-level `output_scale`, FP-gated) ∥ **WT up-weight sweep**
(`{brain:1, wt:5/10/20}` from the coded mechanism) ∥ **SPADE/adaLN-zero T-07** ablation } — any/none, per the human.

**Gates.** Phase-1/2 validated; explicit opt-in. None of these gate the headline.

**Exit criteria.** Whatever ran is green + merged; each ablation reported against PSNR_ET **and** FP-safety; CFG
defaults remain no-ops unless a guidance sweep is explicitly requested.

**Orchestrator notes (append-only).**
- _(empty)_

---

## Planning decisions (resolved 2026-07-22)

| # | Decision | Affects | Resolved |
|---|---|---|---|
| Q1 | Oracle softening | S1 (task 19) | ✅ **SDT-graded** (SDT→sigmoid→avg-pool of GT; matches the predicted path → true swap) |
| Q2 | Region-weight sweep | S2 launch matrix | ✅ **wt ∈ {1, 5, 10, 20}** (equal + 3 up-weights) |
| Q3 | Trunk policy | S2 launch matrix | ✅ **J0 freeze @ wt:1 (floor) + J1–J4 joint-low-LR @ wt:{1,5,10,20}** = 5 jobs; LR = linear-warmup→cosine; **raise EarlyStopping patience to ~400–500** (harder objective transiently raises train loss) |
| Q4 | Latent-embedding viz | S1 (task 40) | ✅ **per-patient PCA/UMAP + slice montage** (1 patient/row, 5 tumour-slice cols, soft mask α=0.7) |

## Planning decisions (resolved 2026-07-23 — scientist audit / iter-9)

| # | Decision | Affects | Resolved |
|---|---|---|---|
| Q5 | Soft-map softeners | S5/S6 (tasks 16/17/18/15) | ✅ **σ-only; temperature DROPPED.** avg-pool / DML+CE / K-fold-mean are mandatory (no knob); **σ is the one knob** (shared oracle↔predicted so the maps match); calibration **MEASURED not corrected** (report ECE/Brier); **ET=TC−NETC reported diagnostic** (not gated); `temperature.py` becomes unused (hygiene-delete when no live caller). Design: B.c banner + B.f-§2 superseded. |
| Q6 | OOD segmenter coupling | S3/S6 + `../../article/03_generalization_ood.md` | ✅ **first-class contribution.** Oracle→predicted `PSNR_ET` gap reported **PER RING** (Ring A vs B, never pooled) + Ring-B segmenter **TC/NETC error distribution** + **localisation(segmenter)+intensity(generator)** decomposition. Segmenter is the OOD ceiling; T-13 oracle mask **leaks post-contrast** (ceiling, not deployable). Design: A.8-§7 addition; article T3.6. |
| Q7 | Vessels / normal enhancement | eval (design A.9) | ✅ **evaluate-only, NO reserved channel.** Report in-ROI enhancement fidelity of the synth T1c on vessel/dural-sinus/choroid/pituitary (reuse Frangi / `venous_atlas_build`), **per ring**. Field's #1 open gap (Moya-Sáez 2023); TA-ViT ignores it; mask-free SynCE shows whole-image models capture it. Motivates a future SWAN/vessel channel **only if** the generator misses normal enhancement. Design: A.9. |
