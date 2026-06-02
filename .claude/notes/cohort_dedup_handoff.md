# Cohort deduplication — handoff

> Snapshot at 2026-06-02 ~12:54 local. Tangible deliverable: a 4-epoch
> deduplicated FM smoke run on **server3 (icai-server)** is currently
> running; we are awaiting its completion to verify the dedup wires through
> training end-to-end.

## What is the user trying to do

Before the full S2 training launch on Picasso, drop cross-cohort duplicate
patients so the same case is not seen twice. The biggest overlap is
**UCSF-PDGM ⊂ BraTS-2021 ⊂ BraTS-GLI 2023/2025**: 293/495 UCSF-PDGM patients
also appear in BraTS-GLI under renumbered IDs. The user supplied
`BraTS2021_MappingToTCIA.xlsx` (BraTS-2021 ↔ TCIA-source mapping). Hard
constraints:

* No re-running the image-domain or latent-domain H5 conversions.
* A per-cohort patient-ID allow-list must be generated **before** training
  and loaded **in memory** by the DataModule (no I/O during training).
* When a dedup file is configured, training must MANDATORILY use it (gated).

The decision direction was confirmed with the user via `AskUserQuestion`:

* `priority = [BraTS-GLI, UCSF-PDGM, IvyGAP, LUMIERE]` — keep BraTS-GLI
  whole (it has no per-patient bridge to BraTS-2021), drop the 293
  UCSF-PDGM duplicates.
* `on_unresolvable = warn` — IvyGAP has 34 xlsx-listed members but its H5
  carries no bridge to BraTS-2021 IDs (its IDs are `W<N>`, xlsx portal IDs
  are numeric); IvyGAP is kept whole with a logged warning. Supply an
  external `W<N>` ↔ portal-ID bridge later to close this gap.

## What has been done

### Library (`src/vena/preflight/cohort_dedup/`)

* `xlsx.py` — stdlib parser for `BraTS2021_MappingToTCIA.xlsx` (no openpyxl
  dep). Returns `Brats2021Mapping(rows, by_brats21_id, by_collection)`.
* `resolver.py` — `CohortClaim` per cohort (explicit `pid_to_bridge` map or
  `implicit_brats21=True` for the BraTS-GLI umbrella), priority-based
  resolution. Returns `ResolverOutput(kept, rejected, resolved, unresolvable)`.
* `decision.py` — schema v1.0 contract: `assert_dedup_decision_valid`,
  `build_allowlists`, `write_decision`, `DedupDecisionSchemaError`.
* `_report.py` — `report.md` + `figures/keep_vs_reject.png`.
* `engine.py` — `CohortDedupConfig` + `CohortDedupEngine.run()`:
  * Loads corpus registry (`require_latents=False` so we can run before
    latent encoding completes).
  * Parses xlsx via stdlib.
  * For each cohort, opens `image_h5` and reads `patients/keys` plus the
    configured `metadata/<field>` bridge (per-scan → per-patient collapse).
  * Calls resolver, writes `decision.json` (validated before commit) +
    `report.md` + bar chart.
  * Updates `<output_root>/LATEST` symlink.
* `__init__.py` exposes the public surface.

### Routine (`routines/preflights/cohort_dedup/`)

* `cli.py` — single positional YAML arg, console script
  `vena-preflight-cohort-dedup` (registered in `pyproject.toml`).
* `engine/cohort_dedup_engine.py` — thin wrapper around the library engine.
* `configs/default.yaml` — local-workstation paths.
* `configs/default_server3.yaml` — server3 paths (xlsx at
  `/media/hddb/mario/datasets/BraTS2021_MappingToTCIA.xlsx`, output at
  `/media/hddb/mario/artifacts/preflights/cohort_dedup/`).
* `configs/smoke.yaml` — same as default (no GPU work).

### FM trainer wiring (`routines/fm/train/engine.py`)

* `_DataCfg.dedup_decisions_path: Path | None = None` added.
* `_assert_preflight_gates(cfg)` — augmentation gate is unchanged; new
  `_assert_dedup_gate(cfg)` runs when `dedup_decisions_path` is set:
  1. File exists.
  2. Validates with `assert_dedup_decision_valid` (schema v1.0).
  3. `decision["corpus_registry_sha256"]` must match `sha256_file(cfg.data.corpus_registry)`.
  4. Every cv cohort in the registry must have an entry in
     `decision["cohorts"]`.
  5. `unresolvable_overlaps != []` triggers a WARNING in the run log (not
     a raise — the preflight already accepted the policy at `warn` time).
* In `run()`, after the gate, load the decision into a `dict[cohort,
  set[str]]` (`build_allowlists`) and pass to `MultiCohortLatentDataModule`.
* `_build_decision_payload` bumped schema to **`0.4.0`** with new keys
  `dedup_decision_path` + `dedup_decision_sha256`. Producer string also
  bumped.

### DataModule (`src/vena/model/fm/lightning/data.py`)

* `MultiCohortLatentDataModule.__init__` now accepts
  `dedup_allowlists: dict[str, set[str]] | None`.
* `setup()` — for each cv cohort, intersects `train/val/test_patient_keys`
  with `allowlist[cohort.name]` BEFORE `_expand_patients_to_scans`. Missing
  allow-list for a cv cohort raises `RuntimeError`. Test-only cohorts
  filtered too when an allow-list happens to be present, missing entries
  tolerated.

### Tests (all `pytestmark = pytest.mark.unit`)

* `tests/preflight/cohort_dedup/test_xlsx.py` — round-trip through a
  hand-built in-memory xlsx fixture.
* `tests/preflight/cohort_dedup/test_resolver.py` — priority direction,
  unresolvable detection with `warn` and `error` modes, lowest-priority
  fallback.
* `tests/preflight/cohort_dedup/test_decision_schema.py` — round-trip,
  validation failures (wrong version, totals mismatch, missing key, list
  length mismatch).
* `tests/routines/fm/test_dedup_gate.py` — `_assert_dedup_gate` happy path,
  missing file, SHA mismatch, cohort missing.

All 19 new tests pass locally and on server3. Full fast suite (`-m "not
slow and not gpu"`) — **353 passed, 4 deselected, no regressions** locally.

### Smoke YAMLs

* `routines/fm/train/configs/runs/smoke_s1_4ep_dedup.yaml` — local
  workstation paths (`/media/mpascual/...`). NOT yet runnable because the
  local `corpus_local.json` declares `latent_h5` paths that do not exist on
  the user's current disk (the real latents are at
  `/media/mpascual/MeningD2/MAISI_VAEGAN_LATENTS/UCSF_PDGM/UCSFPDGM_latents.h5`,
  not at the corpus-declared path). Out of scope for this delivery — the
  user switched to server3.
* `routines/fm/train/configs/runs/smoke_s1_4ep_dedup_server3.yaml` — clone
  of the canonical `smoke_s1_4ep_logging.yaml` plus the single line
  `dedup_decisions_path: /media/hddb/mario/artifacts/preflights/cohort_dedup/LATEST/decision.json`.
  Otherwise identical: 4 epochs, dual-GPU, `block_until_complete: true`,
  augmentations on, MAISI VAE decode in exhaustive_val.

### Server3-specific notes (from the project's known dual-layout caveat)

* `/home/mariopascual/projects/VENA/` has BOTH `src/vena/` and a stale
  top-level `vena/` — `import vena` resolves to the latter. After rsyncing
  new files under `src/vena/...` you MUST mirror them into
  `vena/preflight/cohort_dedup/` and `vena/model/fm/lightning/data.py` (the
  `pip install -e .` editable layout does not unshadow them). This was hit
  on first deploy; the fix is already applied.
* Tests, xlsx, and preflight decision exist on server3:
  * Code: `/home/mariopascual/projects/VENA/`
  * xlsx: `/media/hddb/mario/datasets/BraTS2021_MappingToTCIA.xlsx`
  * dedup decision: `/media/hddb/mario/artifacts/preflights/cohort_dedup/LATEST/decision.json`
    (timestamp `2026-06-02T10-37-20Z`)

## What has been verified

* The preflight runs **locally** AND on **server3**. Both produce identical
  totals: 7 cohorts, 2159 patients in, 1866 kept, **293 rejected**
  (UCSF-PDGM only). IvyGAP↔BraTS-GLI flagged as 34 unresolvable groups
  (warn-and-keep).
* 19/19 new unit tests pass on both machines.
* Full local fast suite still passes (353 tests).
* The gate logic on server3 has been exercised: the smoke run progressed
  past `_assert_preflight_gates` (one residual-overlap WARNING logged) and
  past the augmentation gate; the failure mode that blocked the local
  smoke (missing latent H5 files in the corpus registry, unrelated to the
  dedup code) does NOT apply on server3.

## What is in flight (what to verify when you take over)

A 4-epoch deduplicated FM smoke is running on server3. Launched at
~`2026-06-02 12:54 local` via the `server3` skill pattern:

```
ssh icai-server 'cd /home/mariopascual/projects/VENA && \
  nohup ~/.conda/envs/vena/bin/python -m routines.fm.train.cli \
    routines/fm/train/configs/runs/smoke_s1_4ep_dedup_server3.yaml \
    > "$LOG" 2>&1 & disown'
```

The launcher tool-notification arrived (launcher exited cleanly); the
training process is detached. The log path is in the launcher's stdout — if
you've lost it, `ls -1t /media/hddb/mario/smoke_logs/dedup_smoke_*.log |
head -1` finds the newest one.

Expected wall-clock per the timing reference in
`.claude/skills/server3/SKILL.md`: ~20 min for a 4-epoch multi-cohort run
with `block_until_complete: true` (~2.5 min/epoch training + 3–5 min/epoch
exhaustive val). A `ScheduleWakeup` is set for ~13:04 local to check progress.

### What "success" looks like (full checklist)

1. **Process gone.**
   ```bash
   ssh icai-server 'pgrep -af "vena/bin/python -m routines.fm.train.cli"'
   ```
   No output (or only this command itself).

2. **Log endswith "FM-train completed".**
   ```bash
   ssh icai-server 'tail -15 /media/hddb/mario/smoke_logs/dedup_smoke_*.log'
   ```

3. **`logs/train.log` shows the dedup wired in.** Look for two lines in the
   run-dir log:
   * `"dedup decision <path> carries 1 unresolvable overlap(s)"` (WARNING).
   * `"cohort_dedup ENABLED from <path> — per-cohort kept: {'UCSF-PDGM':
     202, 'BraTS-GLI': 1133, 'IvyGAP': 34, ...}"` (INFO).
   * Per-cohort dedup-filter lines: `"UCSF-PDGM: dedup filter kept
     train=NNN/356, val=NN/89, test=NN/50"` etc.

4. **`decision.json` in the run dir reports `schema_version: "0.4.0"`** and
   includes both `dedup_decision_path` and `dedup_decision_sha256`. Confirm:
   ```bash
   ssh icai-server "~/.conda/envs/vena/bin/python -c '
   import json, glob
   p = sorted(glob.glob(\"/media/hddb/mario/experiments/*_s1_*\"))[-1]
   d = json.load(open(p + \"/decision.json\"))
   print(\"schema_version:\", d[\"schema_version\"])
   print(\"dedup_decision_path:\", d[\"dedup_decision_path\"])
   print(\"dedup_decision_sha256:\", d[\"dedup_decision_sha256\"])
   '"
   ```

5. **`metrics/train_step.csv` and `metrics/train_epoch.csv`** present with
   N≈4 epoch rows; train loss trend decreasing.

6. **`checkpoints/`** carries `ema_epoch_001..004.ckpt`, `ema_best.ckpt`,
   `last.ckpt`, and (because `trunk.trainable=true`) `trunk_ema_snapshot.pt`.

7. **`exhaustive_val/epoch_NNN/`** per epoch — each `metrics.csv` must have
   `wc -l > 1` (header + actual rows). This is the project's "subprocess
   silent-fail trap" — exit code 0 is not enough. There should be 20
   patients × |nfe_levels| rows per epoch.

8. **`exhaustive_val/gpu_usage.log`** shows co-residency: cuda:0 holds the
   training process (~12 GB) and cuda:1 holds the val subprocess (~2–4 GB)
   during each val window.

### If the run is still running

Don't kill it. Set another `ScheduleWakeup` for ~10–15 min and recheck.
Typical full run including the four exhaustive_val passes is ~20 min from
launch.

### If the run errored

Most likely failure modes:

* **`PreflightGateError`** on the dedup gate. Check the message — likely a
  SHA mismatch because someone touched `corpus_server3.json`. Re-run the
  preflight: `ssh icai-server 'cd /home/mariopascual/projects/VENA &&
  ~/.conda/envs/vena/bin/vena-preflight-cohort-dedup
  routines/preflights/cohort_dedup/configs/default_server3.yaml'`.
* **Missing module** — e.g. `ModuleNotFoundError: No module named
  'vena.preflight.cohort_dedup'`. Solution: the rsync must mirror into the
  stale top-level `vena/` directory as well:
  ```bash
  rsync -av src/vena/preflight/cohort_dedup/ \
    icai-server:/home/mariopascual/projects/VENA/vena/preflight/cohort_dedup/
  rsync -av src/vena/model/fm/lightning/data.py \
    icai-server:/home/mariopascual/projects/VENA/vena/model/fm/lightning/data.py
  ```
* **`DedupDecisionSchemaError`** — the preflight artifact is malformed
  (e.g. someone hand-edited it). Re-run the preflight.

## Open follow-ups (NOT required for this deliverable)

* Build a `W<N>` ↔ TCIA-portal-ID bridge for IvyGAP so we can close the 34
  unresolvable groups. Likely sources: the IvyGAP clinical data dump on
  TCIA, or `routines/h5_datasets/ivy_gap/`'s reader where the portal-ID
  may already be in scope.
* The local `corpus_local.json` declares latent_h5 paths that do not
  resolve on disk (`/media/mpascual/MeningD2/GLIOMA/<cohort>/h5/<...>_latents.h5`
  vs the actual location at `/media/mpascual/MeningD2/MAISI_VAEGAN_LATENTS/...`).
  Either fix the registry or re-encode. Not gating for the server3 run.
* `routines/fm/train/configs/smoke/{smoke,smoke_short,smoke_full,...}.yaml`
  still declare the legacy `data.latents_h5` key and will fail at config
  validation. These pre-date the multi-cohort migration and likely should
  be deleted or updated to `corpus_registry`. Out of scope for dedup.

## Decision log

| Question | Answer | Date |
|---|---|---|
| Priority direction | `[BraTS-GLI, UCSF-PDGM, IvyGAP, LUMIERE]` — drop from UCSF-PDGM | 2026-06-02 |
| `on_unresolvable` | `warn` (keep IvyGAP, log the gap) | 2026-06-02 |
| Schema version of dedup `decision.json` | `1.0` | 2026-06-02 |
| Schema bump of FM-train `decision.json` | `0.3.0 → 0.4.0` | 2026-06-02 |

## File index (new + modified)

New:
* `src/vena/preflight/cohort_dedup/{__init__,xlsx,resolver,decision,_report,engine}.py`
* `routines/preflights/cohort_dedup/{__init__.py,cli.py,engine/{__init__.py,cohort_dedup_engine.py},configs/{default.yaml,default_server3.yaml,smoke.yaml}}`
* `routines/fm/train/configs/runs/smoke_s1_4ep_dedup.yaml` (local — unused)
* `routines/fm/train/configs/runs/smoke_s1_4ep_dedup_server3.yaml` (server3 — in flight)
* `tests/preflight/cohort_dedup/{__init__.py,test_xlsx.py,test_resolver.py,test_decision_schema.py}`
* `tests/routines/fm/test_dedup_gate.py`
* `artifacts/preflights/cohort_dedup/2026-06-02T10-41-12Z/{decision.json,report.md,figures/keep_vs_reject.png,...}` (local; LATEST symlink)
* `/media/hddb/mario/artifacts/preflights/cohort_dedup/2026-06-02T10-37-20Z/...` (server3; LATEST symlink)

Modified:
* `routines/fm/train/engine.py` — `_DataCfg`, `_assert_preflight_gates`,
  `_assert_dedup_gate`, `_build_decision_payload` (schema 0.4.0), `run()`.
* `src/vena/model/fm/lightning/data.py` — `MultiCohortLatentDataModule`
  ctor + `setup()`.
* `pyproject.toml` — added `vena-preflight-cohort-dedup` console script.

Planning artifact:
* `/home/mpascual/.claude/plans/context-we-have-almost-rosy-scone.md` — the
  approved implementation plan (decision-direction confirmation included).
