# VENA Phase-2 Validation — SESSION HANDOFF

*Written 2026-07-16 ~23:45 local by the orchestrator (Fable 5) at the end of a
long session, as the **single entry point** for the next session. Read this file
top to bottom before touching anything.*

> **You are the orchestrator.** You hold the plan; Opus 4.8 subagents do the
> coding. Yesterday's subagents are **gone** — a new session cannot `SendMessage`
> them. Their *branches and worktrees survive on disk*. You will spawn fresh
> agents and/or verify + merge what exists.

---

## 0. TL;DR — the five things that matter

1. **A Phase-1 bug inverts the paper's headline result.** Predictions were
   double-normalised. Fixed in the loader (`1c5d2c3`). **§4** — read it, it is
   the single most important thing in this file.
2. **Three analysis routines are written and unmerged** on agent branches. Two
   have the fix; **V3 (`downstream_seg`) does NOT and must re-run** (§6).
3. **Two Picasso smoke jobs were in flight** (1599746, 1599757). They are done by
   now. **Check them first** (§7).
4. **The full sweep is designed and built but NOT submitted.** 360-task array,
   ~10 min wall (§8).
5. **Stop when §11's acceptance criteria are met.** Not before, not after.

**Read next, in order:** this file → `01_SHARED_CONTRACTS.md` (verified facts +
traps) → `00_ORCHESTRATION.md` (full decision log). Task specs:
`02_validation_core.md`, `03_paired_fidelity.md`, `04_spatial_residual.md`,
`05_downstream_seg.md`, `06_brats_ped_backfill.md`. All in
`.claude/notes/prompts/`.

---

## 1. Scope — what this project is

**Phase 1 (done, frozen):** every method predicted $\widehat{T_{1c}}$ for every
test scan; results are 360 prediction H5s + 40 reference H5s, **289 GB**. No
metrics were computed.

**Phase 2 (this work):** read those frozen H5s, compute the validation-proposal
§4 metric suite under the §6 pre-registered statistical plan, emit self-contained
artifacts, and run the full sweep on Picasso.

Source of intent (read for *why*, never for facts — they have drifted):
- `.claude/notes/validation/validation_proposal.md` — the protocol
- `.claude/notes/validation/validation_fairness.md` — the Phase-1 fairness audit
- `/media/mpascual/Sandisk2TB/research/vena/results/fm/inference/README.md`

Three routines, one per proposal section:
| Routine | Section | Produces |
|---|---|---|
| `paired_fidelity` | §4.2 + §4.5 + §4.7 | **the primary endpoint**: MAE on brain, Ring A |
| `spatial_residual` | §4.3 | the vessel-fidelity claim (label-free) |
| `downstream_seg` | §4.4 | ΔDice (real vs synthetic T1c) |

---

## 2. Where everything is

| What | Path |
|---|---|
| Repo | `/home/mpascual/research/code/VENA` (branch `main`) |
| Predictions (local) | `/media/mpascual/Sandisk2TB/research/vena/results/fm/inference` — 288 G |
| Predictions (Picasso) | `/mnt/home/users/tic_163_uma/mpascual/execs/vena/inference` — 289 G |
| Analyses output | `<predictions_root>/analyses/` |
| Corpus H5s (multi-label GT) | Picasso `fscratch/datasets/vena/<cohort>/h5/<NAME>_image.h5` — **NOT reachable locally, MeningD2 unmounted** |
| Picasso env | `fscratch/conda_envs/vena/bin/python` |
| Picasso shared repo | `fscratch/repos/VENA` (@ `1ad2ba4` — **do not push/pull; other jobs run from it**) |
| V1's Picasso scratch checkout | `fscratch/repos/VENA-pf-agent` |
| Local env | `~/.conda/envs/vena/bin/python` (Python 3.11) |

`ssh picasso` works with key auth (`BatchMode=yes`). **Do all compute on
Picasso** — the local box is a dev workstation (RTX 4060) and long foreground
runs get reaped by the 10-minute tool timeout.

---

## 3. Verified facts — do NOT re-derive these

**16 methods × 8 cohorts × 45 (method,NFE) pairs = 360 prediction files + 40 references.**

`selection_nfe`: C0/C1×3/C2/C7 → **1**; C3×3 → **4**; C4/C5/VENA×4 → **5**; C6 → **1000**.
(C4/C5 also have NFE=100; VENA has {1,2,5,10,20}.)

**Pre-registered statistical roles (user-approved, do not re-litigate):**
- **VENA** (reference arm) = `VENA-S1-v3b-rw`
- **Competitor family (n=8, Holm)** = `C0-Identity`, `C1-pGAN-t1pre`, `C2-ResViT`,
  `C3-SynDiff-t1pre`, `C4-3D-DiT`, `C5-T1C-RFlow`, `C6-3D-LDDPM`, `C7-3D-Latent-Pix2Pix`
- **Ablation family (n=3, separate Holm)** = `VENA-S1-v3b`, `VENA-S1-v3a`, `VENA-S3-LPL-b2c`
- **Supplementary (no family)** = `C1-pGAN-{t2,flair}`, `C3-SynDiff-{t2,flair}`

*(t1pre panel chosen a priori — pGAN/SynDiff ran one-to-one; choosing the winner
on test would be oracle selection.)*

**Cohorts / rings:**
| Cohort | Ring | scans | patients |
|---|---|---:|---:|
| UCSF-PDGM | A | 50 | 50 |
| BraTS-GLI | A | 127 | 114 |
| UPENN-GBM | A | 62 | 62 |
| IvyGAP | A | 5 | 5 |
| LUMIERE | A | **72** | **11** ← longitudinal |
| REMBRANDT | A | 5 | 5 |
| BraTS-Africa-Glioma | B | 95 | 95 |
| BraTS-Africa-Other | B | 51 | 51 |
| **Ring A** | | **321** | **247** |
| **Ring B** | | **146** | **146** |
| **TOTAL** | | **467** | **393** |

BraTS-PED (260) was never inferred; backfill queued (§7).

---

## 4. ⚠ THE CRITICAL FINDING — the scoring-space rule

**Read `01_SHARED_CONTRACTS.md` §7.0 in full. Summary:**

Phase-1's `predictions/t1c_synthetic_harmonised` is **corrupted for 15 of the 16
methods**. Scoring it naively **reports that VENA loses to SOTA** — an artefact.

**Mechanism:** every method except C0 was trained to emit the *already*
percentile-normalised T1c (VENA raw mean 0.345 vs target 0.337). Each
under-saturates the bright enhancement tail → raw p99.5 < 1. `apply_harmonisation`
then stretches [p0,p99.5]→[0,1] — a per-scan affine fit to the prediction's own
histogram — inflating parenchyma to 0.491, away from a reference normalised once
from scanner units. Only **C0-Identity** emits real scanner units (raw p99.5 ≈
778–2466) and genuinely needs the mapping.

**The rule (implemented in `vena.validation.io.select_scoring_volume`, wired into
`iter_scans`):**
```python
p995 = np.percentile(raw[brain], 99.5)
if p995 <= 1.05 and raw[brain].min() >= -0.05:   # SCORING_P995_MAX / SCORING_MIN_FLOOR
    volume, mode = raw, "raw"
else:
    volume, mode = harmonised, "harmonised"
```
A **property of the volume, never a method list** — a list breaks when BraTS-PED
or a new competitor lands. `ScanSample.pred` is already the selected volume, so
consumers need no change; `pred_mode` + `raw_p995` ride along for the audit.

**Verified on real data (UCSF-PDGM):**
```
VENA-S1-v3b-rw  raw          p995 0.809   MAE 0.0572   ← BEST
VENA-S1-v3a     raw          p995 0.774   MAE 0.0631
C2-ResViT       raw          p995 0.792   MAE 0.0843
C5-T1C-RFlow    raw          p995 0.712   MAE 0.1008
C4-3D-DiT       raw          p995 0.362   MAE 0.1639
C0-Identity     harmonised   p995 777.7   MAE 0.4444   ← floor
```
VENA beats SOTA by **43%**; the mask ablation lands where theory predicts.

**Caveats to carry:**
- C0's 0.4444 is inflated by the normalisation asymmetry (harmonised T1pre mean
  0.767 vs T1c 0.337), not purely absent enhancement. Valid floor; magnitude
  overstates "cost of no synthesis".
- **Proposal §4.1 still owes a reconciliation to this rule.**
- Under-saturation (raw p99.5 < 1) is now a **reported** failure mode → §4.1
  Table S1. C4-3D-DiT reaching only 38% of the reference's dynamic range is a
  finding, not a bug.

---

## 5. What is DONE

- ✅ **T0 `validation-core` merged** (`78a160d`, `1c5d2c3`): `vena.validation.{io,
  registry,regions,stats,plotting,audit,artifacts}` + `routines/validation/preregister/`.
  - `iter_scans` streams; joins on **`scan_id`, never row index**.
  - `discover_shards` **skips shards whose `decision.json` declares
    `smoke.enabled: true`** (Picasso has a stale `smoke_loginexa/` — 96 files;
    glob matched 456 there vs 360 locally). `build_index` **raises** on duplicate
    `(method,cohort,nfe,scan_id)`.
  - `registry` pins the 4-way family (sizes asserted 8/3/4).
  - `regions.region_masks(brain, wt, *, dilate_k=5)` → `brain/wt/wt_dilated/bg/
    bg_undilated`. **`bg` = dilated (§4.3 C-noT); `bg_undilated` = §4.2.** Do not conflate.
  - `stats.collapse_to_patient` — LUMIERE 72→11 **before any test**.
  - `select_scoring_volume` (§4).
- ✅ `preregister` reproduced Ring A 321/247, Ring B 146/146, total 467/393 on real data.
- ✅ Suite **1008 passed, 4 deselected** (was 943 at session start).
- ✅ Sweep infrastructure built by V1 (§8), unmerged.
- ✅ Harmonisation audit script: `.claude/notes/validation/harmonisation_audit.py`.

**main = `1c5d2c3`** (+ ledger/doc commits).

---

## 6. Unmerged branches — verify, then merge serially

| Branch | Worktree `.claude/worktrees/…` | HEAD | +N | Has `1c5d2c3`? |
|---|---|---|---:|---|
| `worktree-agent-a0ce6b4edb85457bb` (**V1** paired_fidelity + sweep) | `agent-a0ce6b4edb85457bb` | `19fc498` | 6 | ✅ yes |
| `worktree-agent-a6ad94d803dc252bd` (**V2** spatial_residual) | `agent-a6ad94d803dc252bd` | `fdaea7c` | 4 | ✅ yes |
| `worktree-agent-ac2a2992ba7486776` (**V3** downstream_seg) | `agent-ac2a2992ba7486776` | `b7f3add` | 4 | ❌ **PRE-FIX** |
| `worktree-agent-ae7cc423beb026d01` (**P1** BraTS-PED configs) | `agent-ae7cc423beb026d01` | `c1aa43f` | 1 | ❌ pre-fix (harmless — inference configs only) |

### ⚠ V3 MUST re-run — its Dice_synth is void
V3's branch predates `1c5d2c3`, so it fed the **corrupted `_harmonised`** into the
segmenter for the synthetic arm. Its smoke used **only C0-Identity**, where
harmonised *is* the correct volume — which is why it looked fine and the defect
is invisible in its report. For the other 15 methods it is wrong.
**Action:** merge `main` into V3's branch, make it use `sample.pred` (post-merge
that is automatic), re-run its Picasso smoke with ≥1 non-C0 method
(e.g. `VENA-S1-v3b-rw`), confirm `pred_mode` counts, then merge.

Merge rule: **serially, re-running the full suite after each**; branches were cut
before earlier merges. Expect a routine-shape inconsistency to normalise:
V2 uses `engine/<name>_engine.py`; T0/V3 use flat `engine.py`.

---

## 7. Jobs in flight at handoff — CHECK THESE FIRST

```bash
ssh picasso 'sacct -j 1599746,1599757,1597923,1597924,1597925,1597926,1597927 \
  --format=JobID,JobName%26,State,Elapsed,End -P -X'
```

| Job | What | State at handoff | Where results land |
|---|---|---|---|
| **1599746** | V2 `spatial_residual` smoke | RUNNING 20:12 (2 h wall) | `execs/vena/inference/analyses/spatial_residual/<UTC>/` |
| **1599757** | V1 `paired_fidelity` smoke | RUNNING 12:47 (2 h wall) | `execs/vena/paired_fidelity_smoke/<UTC>/` |
| **1597923-27** | P1 BraTS-PED inference ×5 | **PENDING — `QOSGrpGRES`** (group GPU cap, 32 GPUs, someone else holds them) | `execs/vena/inference/picasso_ped_*/` |

**+10 h later these will be terminal.** Both smokes had 2 h walltime, so they
either COMPLETED or TIMEOUT/FAILED. The PED shards may have started, finished, or
still be queued — they are **not blocking Phase 2**.

### What to verify in the smoke outputs (this is the whole point)

**V1 `paired_fidelity`** — `decision.json` + `per_scan.csv`:
- `pred_mode_counts_by_method`: **C0-Identity → `harmonised` (all scans); the
  other 15 → `raw`.** If not, the scoring rule is not wired.
- **The C0 canary**: every real method's MAE **<** C0's. C0 ≈ 0.44; VENA ≈ 0.06.
  *(If C0 ≈ 0.339 and VENA ≈ 0.27, you are looking at a PRE-FIX artifact — check
  the timestamp.)*
- `VENA-S1-v3b-rw` should be the **best** method.
- LUMIERE: 72 scans → **11** patients in `per_patient.csv`.
- `skipped_smoke_shards` should list `smoke_loginexa`.
- **`elapsed_s` / `n_scans`** → re-measure cost/volume (see §8).

**V2 `spatial_residual`** — the open scientific question:
- Report ρ_S, Conc(1/5/10%) for C0, VENA, C5 under **C-noT**.
- **Does the C0-as-ceiling argument survive?** On *void* (pre-fix) data V2 found
  C0 is the ceiling on ρ_S (0.219) and Conc(1%) (1.535) but **NOT on Conc(5%)**
  (C5 0.997 > C0 0.911) — and *all* Conc(5%) were **< 1**, i.e. errors *avoid*
  bright voxels, the opposite of the failure mode §4.3 was built to detect.
  If that **survives on correct data**, it is a real finding: proposal §4.3.3 /
  §4.3.5 need revising and **Conc(1%), not Conc(5%), is the discriminating
  statistic**. If it evaporates, say so. **Do not force either outcome.**

---

## 8. The full sweep — designed, built, NOT submitted

**Cost (measured, from V1's `decision.json`: `elapsed_s: 1872.6`, `n_scans: 635`):**
`2.95 s/volume` wall @ ~9.8 cores = **28.8 CPU-s/volume** → 21,015 volumes ≈
**168 CPU-hours**. *(An earlier "239 CPU-h" in the log was my arithmetic error —
divided by 381, forgetting C5/VENA each ran 2 NFEs. **Re-measure from the new
smoke**: `iter_scans` now reads both `_raw` and `_harmonised`, doubling per-scan I/O.)*

**168 CPU-h is total work, not wall-clock.** Measured limits:
```
MaxTRESPU (per user): cpu=9000      MaxArraySize = 4096
cpu_partition: 335 nodes, 128+ cores, 450 GB+
GrpTRES gres/gpu=32   ← irrelevant: §4.2/§4.3 need NO GPU
```
**CPU jobs bypass the GPU cap stalling P1** — confirmed twice (1599746, 1599757
both scheduled instantly while PED sat in `QOSGrpGRES`).

**Chosen sharding: one array task per prediction file → 360 tasks, ~10 min wall**
(2,880 cores = 32% of cap). Best restart granularity: six bad files →
`--array=6,42,...`, not a 15 h redo.

V1 built it (on its branch, `fscratch/repos/VENA-pf-agent`):
- `cli_manifest.py` → `build_index` → `manifest.csv` with `task_id`
- `cli_shard.py` → one H5 per `SLURM_ARRAY_TASK_ID` → `shard_NNNN.csv` (no stats)
- `cli_merge.py` → concat → `engine.run_postprocess()`
- `launcher_paired_fidelity_sweep.sh` → `--array=0-N%120` + merge
  `--dependency=afterok:<array_id>`; `--dry-run` supported
- **`run_postprocess` runs the patient collapse + Holm correction ONCE, globally,
  in the merge — never per shard.** Sharding those would silently break the
  LUMIERE collapse and the family-of-8 correction.

§4.4 `downstream_seg` needs a **GPU** (~2.6 GPU-h, 4.0 with PED at 1.18 s/call) →
it queues behind the same `gres/gpu=32` cap.

---

## 9. How to work with the Opus 4.8 subagents

Spawn with `Agent`, `subagent_type: "general-purpose"`, `model: "opus"`,
`isolation: "worktree"`, background (default). Session effort is inherited.

**Give each agent:** its task-spec path + `01_SHARED_CONTRACTS.md`, its lane, and
what it must not touch. Nothing else — it starts cold.

### Non-negotiables (learned the hard way this session)
1. **Never trust a closing note. Re-run every check yourself.** Every real defect
   this session came from a number that didn't reconcile, never from reading code.
2. **Demand the artifact path + specific real numbers**, not "run the smoke."
   Two of three agents skipped the smoke entirely when it was phrased as an
   instruction. They cannot fake a C0 canary or a channel-order check.
3. **Before reading any number an agent reports, verify its branch:**
   ```bash
   git -C <worktree> merge-base --is-ancestor <fix-sha> HEAD && echo OK
   ```
   V2 reported results from a run that predated the fix. Real-but-stale is more
   dangerous than absent — it is persuasive.
4. **Worktree isolation**: `cd <wt> && PYTHONPATH=<wt>/src ~/.conda/envs/vena/bin/python`.
   The editable install is path-pinned to the main checkout; without `PYTHONPATH`,
   `routines` loads from the worktree and `vena` from main — silent split-brain.
   **Do not clone the conda env** (verified unnecessary).
5. **`--basetemp=/home/mpascual/.pytest-tmp-<agent-slug>` on every pytest call,
   NEVER `/tmp`.** The suite writes **31 GB/run** (`tests/competitors/*/test_multicohort.py`
   ≈1.4 GB/test × 6 families); pytest keeps 3 runs; `/tmp` is on the **137 GB
   root** and this filled it to **0 bytes**, wedging the machine. Unique per agent
   — `--basetemp` **wipes its dir at start-up**.
6. **Lint only the agent's own files.** Repo baseline is **475 pre-existing ruff
   errors / 70 unformatted files** (ruff 0.15.21). Never `ruff format` the repo.
7. **`isolation: "worktree"` cuts from the SESSION BASE commit, not current
   `main`.** After merging anything, check `git worktree list` and tell new agents
   to `git merge --ff-only main` **first**, with proof.
8. **Two correction rounds max**, then escalate to the user — **but check
   `squeue`/state before acting yourself.** I once "took over" a job one second
   after the agent had submitted it, creating a duplicate.
9. **Agents idle without a live monitor.** Several launched long jobs then stopped,
   so nothing woke them. Require a background `until` loop on `sacct`, or run your
   own waiter.
10. **All compute on Picasso via SLURM.** Local foreground runs get reaped at 10 min.

---

## 10. Mistakes made — do not repeat

**Mine (orchestrator):**
- **Wrote the shard-discovery glob from the local tree.** Wrong on Picasso
  (`smoke_loginexa`). *Local-only verification cannot validate a cluster-side
  contract.*
- **Accused T0 of a fiction closing-check.** Its "1008 passed" was honest; the
  318 errors were `ENOSPC` from the full disk. Check the environment before
  blaming the agent.
- **239 CPU-h arithmetic error** — divided by 381 instead of the recorded 635.
  Real: 168.
- **Proposed a tier-aware (latent vs image) scoring rule** from a 3-method sample.
  The 16-method audit refuted it: it is C0-vs-everyone. *Confirm across the full
  population before designing around a sample.*
- **Duplicated V1's job** by acting one second after it complied.
- **Let the smokes run locally** at 41 CPU-s/volume until the user pushed back.

**Agents':**
- V2 + V3 skipped the real-data smoke, then handed *me* the commands to run it.
- V3 had **C0-Identity backwards** ("identity copies real T1c" — it copies
  **T1pre**), and had wired `delta_wt ≈ 0` in as a join proof. That would have
  made a *correct* implementation look broken, or driven tuning until the real
  T1c leaked into the synthetic arm — plausible Dice, invalid table.
- V3 sized the sweep off **3,498 scans** (the corpus) instead of **467** (test).
- V2 reported pre-fix numbers as current.
- T0 first shipped a `registry` stub with a 2-way `startswith("VENA-")` heuristic
  — which would have Holm-corrected over 12 rows instead of 8, making **every
  p-value in the paper wrong** and plausible-looking.

**Environment traps:** full `/` (31 GB/pytest-run); pytest silently falls back to
writing `pytest-of-*/` under the CWD when TMPDIR is full (31 GB landed *in the
repo*, now gitignored); `rm` is not allowlisted and `rm -rf $HOME*` is
hard-denied — use `mv` to relocate, and ask the user to delete.

---

## 11. ACCEPTANCE CRITERIA — stop iterating when ALL hold

1. **All three routines merged to `main`**, suite green and **≥1008 passed**, ruff
   clean on validation files.
2. **Each routine has a verified real-data smoke** on Picasso whose
   `decision.json` shows:
   - `pred_mode`: C0 → `harmonised`, other 15 → `raw`
   - `skipped_smoke_shards` includes `smoke_loginexa`
   - LUMIERE 72 scans → 11 patients
   - **C0 canary passes**: every real method beats C0; `VENA-S1-v3b-rw` best
3. **V3 re-run post-fix** with ≥1 non-C0 method; `Dice_real` ≈ WT 0.85–0.92 /
   ET 0.70–0.85; WT mask-join proof Dice ≈ 1.0; ΔDice(C0) largest.
4. **V2 has answered the C0-ceiling question on correct data** — either it holds,
   or §4.3.3/§4.3.5 get a documented revision to Conc(1%).
5. **The full sweep is SUBMITTED** (360-task array + dependent merge) and its
   `decision.json` records the manifest, the accepted/skipped shards, and the
   scoring-mode counts.
6. **`preregister` has run ON PICASSO** *after* the smoke-skip fix, freezing
   `ring_partitions.json` + hash. **Order is load-bearing** — run before the fix
   and it globs 456 files and freezes a wrong pre-registration, the artifact the
   whole P3 integrity claim rests on. Then **delete
   `routines/validation/downstream_seg/configs/ring_partitions_bootstrap.json`**
   (V3's honest stopgap; its own comment says to replace it) — two sources of
   truth for the partitions is how the 3,498-vs-467 error happened.

**Not required to stop:** BraTS-PED landing (Ring B works at 146; the loader
discovers cohorts from disk, so PED flows in later with no code change);
proposal §4.1/§4.3 reconciliation (file it as a follow-up).

---

## 12. TOMORROW — do this, in order

```bash
cd /home/mpascual/research/code/VENA
git log --oneline -3          # expect 1c5d2c3 or later
git worktree list             # 4 agent worktrees survive
df -h /                       # must NOT be ~0; see §9.5
ssh picasso 'sacct -j 1599746,1599757,1597923,1597924,1597925,1597926,1597927 \
  --format=JobID,JobName%26,State,Elapsed -P -X'
```

1. **Read the two smoke results** (§7). They decide everything. If a job
   TIMEOUT'd or FAILED, read its log under `execs/vena/logs/` and resubmit —
   do **not** re-run locally.
2. **Verify V1's numbers yourself** against §7's checklist. If C0 ≈ 0.44 and
   VENA ≈ 0.06 and is best → the fix holds → merge V1.
3. **Merge V2** if its C-noT numbers are post-fix (check `pred_mode` in its CSV).
   Record its C0-ceiling verdict either way.
4. **Spawn a fresh Opus agent for V3** — it is PRE-FIX and must merge `main`,
   re-run on Picasso with a non-C0 method, and report (§6). Give it
   `05_downstream_seg.md` + `01_SHARED_CONTRACTS.md` + §4 of this file.
5. **Merge serially**, re-running the suite each time. Normalise routine shape.
6. **Run `preregister` on Picasso** (§11.6), then drop the bootstrap JSON.
7. **Submit the sweep** — `launcher_paired_fidelity_sweep.sh --dry-run` first,
   then for real. Then the spatial_residual and downstream_seg sweeps.
8. **Check P1** (`picasso_ped_*`): if the shards ran, expect **45 prediction
   files** + a BraTS-PED reference per shard, validators clean, 260 unique
   scan_id == 260 patient_id, `ring=B`, and the **existing tree still at 360+40**.
   If still `QOSGrpGRES`, leave them — not blocking.

**Housekeeping the user may want to do** (`rm` is denied to the agent):
```
rm -rf /home/mpascual/.pytest-tmp-*        # per-agent pytest temp
```

---

## 13. The one-paragraph version

Phase-1 froze 289 GB of predictions but harmonised them a second time, which
corrupted 15 of 16 methods and made VENA appear to lose to SOTA. That is fixed in
the loader (`select_scoring_volume`, `1c5d2c3`) with no re-inference needed —
`_raw` was stored. Scored correctly, **VENA-S1-v3b-rw wins by 43%**. Three
analysis routines are written; V1 and V2 are post-fix with Picasso smokes in
flight, V3 is pre-fix and must re-run. The full sweep is built (360-task array,
~10 min wall, CPU-only so it bypasses the GPU cap) but not submitted. Verify the
smokes, merge the three branches, freeze the pre-registration on Picasso, submit
the sweep. **Trust no agent's numbers without checking its branch SHA first.**
