# 2026-06-02 — Cohort overlap incident (pre-Picasso)

## What we found

The multi-cohort training corpus (`corpus_*.json`) is NOT patient-disjoint.
Confirmed cross-cohort duplicates by joining `metadata/brats21_id` from
`UCSFPDGM_image.h5` against `BraTS2021_MappingToTCIA.xlsx`:

| Cohort | n_patients | overlapping | mechanism |
|---|---:|---:|---|
| UCSF-PDGM | 495 | **293** in BraTS-GLI | direct via `metadata/brats21_id`; xlsx says 438 UCSF cases (UCSF-PDGM + UCSF-PDGM_Additional) entered BraTS-2021; only 293 of those landed in our H5 with the field populated |
| IvyGAP | 34 | **likely 34** in BraTS-GLI | xlsx lists 34 IvyGAP TCIA rows; our IvyGAP H5 also has exactly 34 patients but its IDs are `W<N>` (TCIA portal IDs are numeric in xlsx). Cannot match without an external `W<N>` ↔ portal-ID bridge file. |
| LUMIERE | 91 | 0 | independent (Bern); no BraTS-2021 rows |
| BraTS-Africa-*, BraTS-PED | n/a | 0 | independent cohorts, never went to BraTS-2021 |

BraTS-GLI 2023/2025 is the umbrella: its 1133 patients are the entire
BraTS-2021 corpus under renumbered IDs (`BraTS-GLI-PPPPP-TTT`) plus extras.
The BraTS-GLI H5 carries no per-patient bridge to BraTS-2021 IDs, so we
cannot identify *which* BraTS-GLI patient is the renumbered UCSF-PDGM
patient without an external 2023↔2021 mapping file (not in scope today).

## Why it matters

Untreated, the FM trainer's `TemperatureBalancedSampler` sees the 293
UCSF-PDGM patients twice — once via the UCSF-PDGM cohort and once via
BraTS-GLI — and the temperature-balanced weighting (which already biases
toward patient count) compounds the double-count. Splits leak too: if
patient X is in UCSF-PDGM's val fold and its BraTS-GLI twin is in
BraTS-GLI's train fold, val metrics are optimistic for that case.

## What we shipped to mitigate

1. **`vena-preflight-cohort-dedup` preflight** — reads corpus + xlsx + each
   cohort's `metadata/<bridge>` field, runs a priority-based resolver, emits
   `decision.json` v1.0 with per-cohort `kept_patient_ids` /
   `rejected_patient_ids` arrays. Stdlib xlsx parser (no openpyxl).
2. **Hard gate at training startup**
   (`routines.fm.train.engine._assert_dedup_gate`):
   - decision.json validates against schema v1.0.
   - `decision["corpus_registry_sha256"]` matches the currently-loaded
     corpus registry's hash. Stops silent corpus-vs-decision drift.
   - Every cv cohort in the registry has an allow-list entry.
   - `unresolvable_overlaps != []` → WARNING (the preflight already accepted
     this at `on_unresolvable: warn` time).
3. **In-memory filter** — allow-lists loaded once into
   `dict[cohort, set[str]]` and passed to
   `MultiCohortLatentDataModule(dedup_allowlists=…)`. The DataModule
   intersects split patient keys BEFORE CSR-expansion to scan IDs, so
   dropped patients never reach `LatentH5Dataset` nor the sampler. No I/O
   during training.
4. **`decision.json` schema bump 0.3.0 → 0.4.0** — train runs now record
   `dedup_decision_path` + `dedup_decision_sha256`, so downstream
   external-eval / reader-study routines can re-verify which decision
   produced a given checkpoint.

## Decision direction (user-confirmed 2026-06-02)

* `priority = [BraTS-GLI, UCSF-PDGM, IvyGAP, LUMIERE]`. BraTS-GLI wins ties.
  The 293 duplicates are dropped from **UCSF-PDGM** (the side that carries
  the bridge field). BraTS-GLI is kept whole. No external 2023↔2021
  mapping file required — runs on the data we have today.
* `on_unresolvable = warn`. IvyGAP is kept whole; gap logged in
  `decision.json["unresolvable_overlaps"]`.

## Validation evidence

* Local preflight: 293 rejected ✓, IvyGAP 34 unresolvable ✓, BraTS-GLI
  kept whole ✓.
* Server3 preflight: identical numbers (decision
  `2026-06-02T10-37-20Z`).
* 19 new unit tests pass on both machines. Full fast suite green (353
  passed, 0 regressions).
* **4-epoch deduped FM smoke** on server3
  (`/media/hddb/mario/experiments/2026-06-02_10-37-57_s1_a8b73a01`):
  * UCSF-PDGM filtered to 147/356 train, 34/89 val, 21/50 test. Δ summed
    over the three splits = 293 ✓.
  * BraTS-GLI / IvyGAP / LUMIERE untouched.
  * Train loss 1.905 → 1.486 → 1.436 → 1.420 across epochs.
  * Per-cohort CFM in `metrics/train_epoch.csv`.
  * 4× `exhaustive_val/epoch_NNN/metrics.csv` with 186 lines each (not the
    silent-fail trap).
  * Run `decision.json` has `schema_version: "0.4.0"` and
    `dedup_decision_path` + `_sha256` populated.

## Remaining open

1. **IvyGAP bridge** (34 unresolved overlaps). Source: TCIA IvyGAP
   clinical CSV, or `routines/h5_datasets/ivy_gap/` reader which already
   joins on portal IDs internally.
2. **Local `corpus_local.json` stale `latent_h5` paths** block the local
   4060/3060 smoke (latents actually live at
   `/media/mpascual/MeningD2/MAISI_VAEGAN_LATENTS/UCSF_PDGM/UCSFPDGM_latents.h5`,
   not the registry-declared path). Server3 / Picasso unaffected.
3. **Legacy smoke YAMLs** (`routines/fm/train/configs/smoke/*.yaml`) still
   use the rejected `data.latents_h5` key — pre-multi-cohort migration.
   Delete or port to `corpus_registry`.

## Operational checklist for future train YAMLs

Any new YAML under `routines/fm/train/configs/runs/` that uses the
multi-cohort corpus MUST include:

```yaml
data:
  corpus_registry: routines/fm/train/configs/corpus/corpus_<host>.json
  dedup_decisions_path: <abs path to artifacts/preflights/cohort_dedup/LATEST/decision.json>
```

Refresh the dedup decision after any change to `corpus_*.json` (the gate's
SHA check will block training otherwise). Re-run:
`vena-preflight-cohort-dedup routines/preflights/cohort_dedup/configs/default_<host>.yaml`.
