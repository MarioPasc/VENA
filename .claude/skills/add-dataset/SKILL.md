---
name: add-dataset
description: Add a new MRI cohort (e.g. another BraTS subset, IvyGAP-like, longitudinal LUMIERE-like) to the VENA pipeline. Walks the 8-step playbook — clarify → cohort reader → optional HD-BET skull-strip → image-H5 converter → routine/CLI → unit tests → corpus JSON entries → encode config → bandwidth-optimised local↔server3↔Picasso transfers. Use whenever the user says "add cohort X", "integrate dataset Y", "we have a new dataset", or asks to onboard any new training/OOD cohort.
---

# Add a New Dataset to VENA

This playbook is for an autonomous agent integrating a new MRI cohort. Read it cover-to-cover before writing code. The 8 steps are dependency-ordered; do not jump ahead.

The pattern is calibrated against six prior integrations (UCSF-PDGM, BraTS-GLI, IvyGAP, BraTS-Africa-{Glioma,Other}, LUMIERE, BraTS-PED). Every rule here exists because one of those broke without it.

---

## Step 0 — Clarify in ONE batch

Before touching any file, gather requirements with a single `AskUserQuestion` call. Drip-feeding clarifications burns user time and produces inconsistent code. Required fields:

| Question | Drives |
|---|---|
| Preprocessing state: **skull-stripped / defaced / raw native** | Whether to build a Step-2 skull-strip routine + intensity policy at write time |
| Role: **`cv` (train+val+test)** or **`test_only` (OOD)** | Corpus JSON `role`, splits-writer branch, `splits/cv/fold_0` alias patch |
| Longitudinal: **yes / no** | Cohort reader: `Patient` vs `Session` dataclass; CSR `patients/{offsets,keys}` in image-H5 |
| Modalities present: any subset of **{t1pre, t1c, t2, flair, swan}** | `BRATS_*_IMAGE_SEQUENCE_MAP`, `has_swan` in corpus JSON, encode-config `modalities` list |
| Native space: **SRI24 (240,240,155)** / **MNI152 (182,218,182)** / **scanner-native** | Manifest `expected_shape`; if scanner-native, you also need an atlas registration step (proposal §3 — not currently supported) |
| Label system: **BraTS2021 `{0,1,2,4}`** / **BraTS2023 `{0,1,2,3}`** / other | Manifest `label_system`; encoder auto-remaps BraTS2023→2021 at encode time |
| If cv: splits strategy. Examples — **single 24/5/5** (IvyGAP), **400/50/50** (UCSF-PDGM), **10test + 5-fold on remaining** (LUMIERE) | `vena.data.h5.shared.splits.make_cohort_splits` arguments |
| Dedup against existing cohorts (UCSF-PDGM ⊂ BraTS-GLI is a real overlap) | Whether to invoke `routines/preflights/cohort_dedup` before splits land |
| Source root absolute path | Engine `default.yaml`, all 3 corpus JSONs |
| Destination naming: **`<slug>` (snake_case)** for paths + **`<Cohort-Tag>` (Title-Case)** for H5 attr | All file paths and the H5 root attr `cohort` |

If any answer is "I'm not sure", default to the most conservative choice and flag it: `cv → test_only`, `unknown space → scanner-native (block here, ask for atlas)`, `unknown label → block here`.

---

## Step 1 — Cohort reader

**File:** `src/vena/data/niigz/<slug>.py`.

Template (cross-sectional, BraTS-style — adapt for longitudinal or per-cohort layout):

```python
from __future__ import annotations
import logging, re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from vena.data.cohort import register_cohort
from .shared.exceptions import ModalityNotFoundError, PatientNotFoundError
from .shared.io import NiftiVolume, load_nii

logger = logging.getLogger(__name__)
Modality = Literal["t1pre", "t1c", "t2", "flair"]

_MODALITY_SUFFIX: dict[str, str] = {"t1pre":"t1n","t1c":"t1c","t2":"t2w","flair":"t2f"}
_SEG_SUFFIX = "seg"
_PATIENT_DIR_RE = re.compile(r"^<CohortPrefix>-(\d+)-(\d+)$")  # adapt per cohort

@dataclass(frozen=True)
class <Name>Patient:
    patient_id: str
    root: Path
    metadata: dict[str, Any] = field(default_factory=dict)

@register_cohort(
    "<slug>",
    pathology="glioma",   # or meningioma / metastasis / healthy / other / pediatric_glioma
    metadata={
        "release": "<dataset name + year>",
        "spacing_mm": (1.0, 1.0, 1.0),
        "atlas": "SRI24",        # or MNI152 or native
        "label_system": "BraTS2023",
    },
)
class <Name>Dataset:
    def __init__(self, source_root: Path | str) -> None:
        self.source_root = Path(source_root)
        if not self.source_root.is_dir():
            raise FileNotFoundError(f"source_root does not exist: {self.source_root}")
        self._patients = self._discover_patients()
        self._index_by_id = {p.patient_id: i for i, p in enumerate(self._patients)}

    def _discover_patients(self) -> list[<Name>Patient]:
        out: list[<Name>Patient] = []
        for d in sorted(self.source_root.iterdir()):
            if d.is_dir() and _PATIENT_DIR_RE.match(d.name):
                out.append(<Name>Patient(patient_id=d.name, root=d))
        return out

    def __len__(self) -> int: return len(self._patients)
    def __iter__(self) -> Iterator[<Name>Patient]: return iter(self._patients)
    def __getitem__(self, key: int | str) -> <Name>Patient:
        if isinstance(key, int): return self._patients[key]
        if key in self._index_by_id: return self._patients[self._index_by_id[key]]
        raise PatientNotFoundError(f"Unknown <slug> patient: {key}")
    def ids(self) -> list[str]: return [p.patient_id for p in self._patients]

    @staticmethod
    def _modality_path(p, suffix: str) -> Path:
        return p.root / f"{p.patient_id}-{suffix}.nii.gz"

    def load_modality(self, p, name: Modality) -> NiftiVolume:
        if name not in _MODALITY_SUFFIX:
            raise ModalityNotFoundError(f"Unknown modality: {name!r}")
        path = self._modality_path(p, _MODALITY_SUFFIX[name])
        if not path.exists():
            raise ModalityNotFoundError(f"Modality {name} missing for {p.patient_id}: {path}")
        return load_nii(path)

    def load_tumor_seg(self, p) -> NiftiVolume:
        path = self._modality_path(p, _SEG_SUFFIX)
        if not path.exists():
            raise ModalityNotFoundError(f"Tumour segmentation missing for {p.patient_id}: {path}")
        return load_nii(path)
```

**Longitudinal variant** (LUMIERE pattern): replace `Patient` with `Session`, add `session_id`/`patient_id`/`week`/`week_repeat` metadata fields, and expose `patient_groups() -> list[tuple[str, list[int]]]` returning `(patient_id, [row_indices])` for CSR. See `src/vena/data/niigz/lumiere.py`.

**Register the import** in `src/vena/data/niigz/__init__.py` (both the symbol import and the `__all__` entry).

---

## Step 2 — Skull-strip routine (only if step 0 said defaced/raw)

VENA's existing converters assume already-stripped data. If the source is defaced (face removed, skull retained) or fully raw, build a HD-BET preprocessing routine **before** the H5 converter sees it.

**Critical constraint**: HD-BET 2.x pulls in nnU-Net v2 + torch 2.12 (CUDA 13), which is incompatible with VENA's pinned torch 2.6+cu124. Install HD-BET in a dedicated env:

```bash
conda create -n hdbet python=3.11 -y
~/.conda/envs/hdbet/bin/pip install HD-BET --index-url https://download.pytorch.org/whl/cu124
# verify: ~/.conda/envs/hdbet/bin/python -c "import torch; print(torch.cuda.is_available())" → True
```

The runner shells out to `~/.conda/envs/hdbet/bin/hd-bet`. Reference implementation: `src/vena/preprocess/hd_bet/runner.py` (library) + `routines/preprocess/brats_ped_skullstrip/` (routine). Pattern per patient:

1. Run HD-BET on the reference modality (default `t1pre`) with `--save_bet_mask`.
2. Apply the derived brain mask voxelwise to the other 3 modalities.
3. Carry the tumour seg through unchanged.
4. Save brain mask as `{pid}-brain_mask.nii.gz` for auditing.

**Mandatory robustness**: wrap every `nibabel.load(...)` in `try: ... except (OSError, ValueError, ImageFileError, EOFError) as exc: raise HDBETError(...)` — a single corrupt source `.nii.gz` will otherwise crash the whole batch hours into the run.

**Pre-flight scan** before launching the full HD-BET run:

```bash
find <source_root> -name '*.nii.gz' | while read f; do
  file "$f" | grep -qv 'gzip compressed' && echo "CORRUPT: $f"
done
```

If a patient has any corrupt modality, move the source directory aside (`mv <pid_dir> .SKIPPED_<pid>_<reason>`) **before** the H5 converter runs, so the row never enters the encoder's iteration set.

**Pin the `ImageFileError` import** with a comment to survive the formatter:
```python
from nibabel.filebasedimages import ImageFileError  # noqa: TC002
```

The H5 converter must filter skipped patients out of `splits/*` AND `patients/keys` AND `patients/offsets` AND `ids` — encoder iterates `f["ids"].shape[0]`, not `splits/test`. Cleanest path: do the source-side `mv` before conversion so the converter never sees the bad patient.

---

## Step 3 — Image-H5 converter

**Files:** `src/vena/data/h5/<slug>/image_domain/{convert.py, manifest.py, __init__.py}`.

Mirror BraTS-Africa exactly (cross-sectional, single subset, OOD): `src/vena/data/h5/brats_africa/image_domain/convert.py`. Or LUMIERE for longitudinal multi-session CSR: `src/vena/data/h5/lumiere/image_domain/convert.py`.

Key invariants the manifest must encode (see `vena.data.h5.shared.H5Manifest`):
- `schema_version="2.0.0"`, `domain="image"`, `expected_shape=(H,W,D)`.
- Datasets: `images/{t1pre,t1c,t2,flair}` float32, `masks/{tumor,brain}` int8, `crop/origin` int32, `patients/{offsets,keys}` CSR, `ids` vlen-str.
- Root attrs: `schema_version`, `cohort` (Title-Case), `domain`, `created_at`, `producer`, `config_json`, `git_sha`, `label_system`, `crop_box=[192,224,192]` (JSON), `orientation="LPS"`, `split_role` (`internal` for cv / `external` for test_only).
- Brain mask: union of nonzero voxels across the four modalities (skull-stripped data has background == 0).
- Validate before returning: `assert_h5_valid(path, manifest)` in a try/except that unlinks the file on failure.

For cv cohorts use `vena.data.h5.shared.splits.make_cohort_splits` to derive `splits/test`, `splits/cv/fold_K/{train,val}`. For test_only cohorts write only `splits/test` (every patient).

---

## Step 4 — Routine + CLI + smoke YAML

**Files:** `routines/h5_datasets/<slug>/{cli.py, engine.py, configs/default.yaml, configs/smoke.yaml, __init__.py}`.

Mirror `routines/h5_datasets/brats_africa_glioma/` — 5-line `cli.py`, 25-line `engine.py` wrapping the library converter. The engine config is a `<Name>H5RoutineConfig(<Name>ImageH5Config)` subclass with `from_yaml(path)` classmethod.

Register the console script in `pyproject.toml`:

```toml
vena-h5-<slug>                  = "routines.h5_datasets.<slug>.cli:main"
# and if skull-strip needed:
vena-preprocess-<slug>-skullstrip = "routines.preprocess.<slug>_skullstrip.cli:main"
```

Re-run `~/.conda/envs/vena/bin/pip install -e . --no-deps` after editing `pyproject.toml` so the scripts land in PATH.

Smoke YAML (`configs/smoke.yaml`): set `limit: 4`, `output_path: /tmp/vena_smoke_h5/<slug>_image_smoke.h5`, `overwrite: true`. The smoke MUST complete in < 5 minutes on a 4060.

---

## Step 5 — Unit tests

**File:** `tests/data/cohort/test_<slug>.py`.

Module-level `pytestmark = pytest.mark.unit` (otherwise the marker-filtered fast suite skips the test silently).

Synthetic on-disk fixture with 8×8×8 NIfTI volumes (no real data). Cover: registration in `get_cohort_registry()`, `__len__`, `ids()`, `isinstance(ds, CohortProtocol)`, modality + seg loading, ID/index lookup, non-matching-dir-name rejection. ~80 lines mirroring `tests/data/cohort/test_brats_africa.py`.

Run: `~/.conda/envs/vena/bin/python -m pytest tests/data/cohort/test_<slug>.py -v`. All must pass before launching the local smoke conversion.

---

## Step 6 — Corpus JSON entries

Update **all three** corpus registries with the new cohort:

| File | Path prefix |
|---|---|
| `routines/fm/train/configs/corpus/corpus_local.json` | `/media/mpascual/MeningD2/GLIOMA/<COHORT>/h5/` |
| `routines/fm/train/configs/corpus/corpus_server3.json` | `/media/hddb/mario/data/GLIOMAS/<slug>/h5/` |
| `routines/fm/train/configs/corpus/corpus_picasso.json` | `/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/vena/<slug>/h5/` |

Schema per entry:
```json
{
  "name": "<Cohort-Tag>",
  "pathology": "glioma",                    // free string; pediatric_glioma, other_neoplasm etc. accepted
  "label_system": "BraTS2023",              // matches H5 root attr
  "role": "cv",                              // or "test_only"
  "longitudinal": false,
  "image_h5":  "<abs path>/<Cohort_Tag>_image.h5",
  "latent_h5": "<abs path>/<Cohort_Tag>_latents.h5",
  "n_patients": 260,
  "n_scans":    260,
  "modalities": ["t1pre", "t1c", "t2", "flair"],
  "has_swan":   false
}
```

Initially use a placeholder `n_patients/n_scans` from the source count; update after conversion finishes with the actual number (skipped patients drop this).

---

## Step 7 — Encode config

**File:** `routines/encode/maisi/configs/<slug>_server3.yaml`. Copy `brats_africa_glioma_server3.yaml` (test_only) or `ucsf_pdgm_server3.yaml` (cv). The intensity normalisation and precision MUST match existing cohorts for latent comparability — do not deviate without a research-rigor justification:

```yaml
modalities: [t1pre, t1c, t2, flair]
precision_mode: fp32
percentile_lower: 0.0
percentile_upper: 99.95
percentile_foreground_only: true     # all stored volumes are skull-stripped → foreground-only required
encode_full_cohort: true
roundtrip: { enabled: true, n_patients: 4, modalities: [t1pre, t1c, t2, flair] }
pca:       { enabled: true, pooling: global_avg, n_components: 2 }
```

---

## Step 8 — Bandwidth-optimised transfers

**Order matters.** The image-H5 has no dependency on the encode; ship it to Picasso *while* server3 encodes. Naively serial: ~30 min. Parallelised: ~15 min.

```
─────────────────────────────────────────────────────────────────────
T0  | local: image-H5 written
T1  | LAUNCH IN PARALLEL:
    |   (a) rsync image-H5 → server3
    |   (b) scp     image-H5 → Picasso       (USE LOCAL→PICASSO DIRECT)
T2  | (a) lands → push VENA repo to server3 → pip install -e .
T3  | server3: encode → produces latent-H5 (+ roundtrip QC)
T4  | server3 → Picasso: scp -3 latent-H5    (RELAYS THROUGH LOCAL)
─────────────────────────────────────────────────────────────────────
```

### Concrete commands

Patch image-H5 with `splits/cv/fold_0` aliases BEFORE shipping (test_only cohorts only — required by trainer's exhaustive_val):

```python
import h5py
p = "<local image H5>"
with h5py.File(p, "a") as f:
    test_ids = [x.decode() if isinstance(x, bytes) else x for x in f["splits/test"][:]]
    if "splits/cv" in f: del f["splits/cv"]
    g = f.create_group("splits/cv/fold_0")
    vlen = h5py.string_dtype(encoding="utf-8")
    g.create_dataset("val",   data=test_ids, dtype=vlen)
    g.create_dataset("train", data=[],        dtype=vlen)
```

Launch the two transfers in parallel:

```bash
# (a) image → server3 — use BARE host alias (config sets user=mariopascual)
nohup rsync -avP <local image>  icai-server:/media/hddb/mario/data/GLIOMAS/<slug>/h5/ \
    > /tmp/vena_<slug>_logs/rsync_image.log 2>&1 &

# (b) image → Picasso (direct from local; agent has mpascual@picasso key)
nohup scp <local image> mpascual@picasso3.scbi.uma.es:/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/vena/<slug>/h5/ \
    > /tmp/vena_<slug>_logs/scp_image_picasso.log 2>&1 &
```

Push the VENA repo to server3 with sources WITHOUT trailing slashes (trailing-slash flattens the nested layout):

```bash
rsync -az --exclude='.git' --exclude='__pycache__' --exclude='.pytest_cache' \
      --exclude='*.egg-info' --exclude='artifacts' --exclude='experiments' \
      src routines tests pyproject.toml CLAUDE.md .claude \
      icai-server:/home/mariopascual/projects/VENA/
ssh icai-server 'cd /home/mariopascual/projects/VENA && ~/.conda/envs/vena/bin/pip install -e . --no-deps'
ssh icai-server '~/.conda/envs/vena/bin/python -m pytest tests/data/cohort/test_<slug>.py -q'
```

Launch encode on server3 (background; nohup so the SSH can return):

```bash
ssh icai-server '
  mkdir -p /media/hddb/mario/results/vena/encode_<slug>_maisi /media/hddb/mario/smoke_logs;
  cd /home/mariopascual/projects/VENA &&
  nohup ~/.conda/envs/vena/bin/python -m routines.encode.maisi.cli \
       routines/encode/maisi/configs/<slug>_server3.yaml \
       > /media/hddb/mario/smoke_logs/encode_<slug>_$(date +%Y%m%d_%H%M%S).log 2>&1 &
'
```

Verify roundtrip QC (must pass before pushing the latent H5):

```bash
ssh icai-server 'cat /media/hddb/mario/results/vena/encode_<slug>_maisi/LATEST/tables/roundtrip_metrics.csv | head -10'
# Acceptance: MSE < ~1e-3 per modality (≈30 dB PSNR with data_range=1.0).
```

Patch latent-H5 with the same `splits/cv/fold_0` aliases (test_only cohorts only):

```bash
ssh icai-server '~/.conda/envs/vena/bin/python <<PY
import h5py
p="/media/hddb/mario/data/GLIOMAS/<slug>/h5/<Cohort_Tag>_latents.h5"
with h5py.File(p,"a") as f:
    test_ids = [x.decode() if isinstance(x,bytes) else x for x in f["splits/test"][:]]
    if "splits/cv" in f: del f["splits/cv"]
    g = f.create_group("splits/cv/fold_0")
    vlen = h5py.string_dtype(encoding="utf-8")
    g.create_dataset("val",   data=test_ids, dtype=vlen)
    g.create_dataset("train", data=[],        dtype=vlen)
PY'
```

Ship latent-H5 from server3 to Picasso via `scp -3` (routes through local — no intermediate disk):

```bash
scp -3 icai-server:/media/hddb/mario/data/GLIOMAS/<slug>/h5/<Cohort_Tag>_latents.h5 \
       mpascual@picasso3.scbi.uma.es:/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/vena/<slug>/h5/
```

Update corpus JSONs' `n_patients` and `n_scans` to the actual counts produced.

---

## Verification checklist (gate each step)

- [ ] Step 1: `tests/data/cohort/test_<slug>.py` passes (`pytest -m unit -k <slug>`).
- [ ] Step 2 (if skull-strip): smoke 4-patient run completes; HD-BET report shows `n_ok == n_total`; mask-applied modalities have plausible brain-volume fraction (~15–25% of voxels for adult, can be lower for pediatric).
- [ ] Step 3+4: smoke 4-patient H5 conversion succeeds; H5 root attrs include `schema_version=2.0.0` and `n_patients_written == 4`.
- [ ] Step 5: unit tests green.
- [ ] Step 6: all 3 corpus JSONs valid (Pydantic model validates at trainer load).
- [ ] Step 8 (a)/(b) in parallel: image-H5 sha256 matches on local, server3, Picasso.
- [ ] Encode: `roundtrip_metrics.csv` shows MSE < 1e-3 on every modality × patient.
- [ ] Latent-H5 patched + transferred: `n_patients` consistent between image-H5 and latent-H5.

If any gate fails, stop and ask the user — do not paper over with workarounds.

---

## Common cohort archetypes (pick the closest reference)

| Archetype | Reference cohort | Lift |
|---|---|---|
| Adult HGG cross-sectional, BraTS-style, OOD test | BraTS-Africa-Glioma | minimal — change paths + regex |
| Adult HGG longitudinal, multi-session, CV | LUMIERE | CSR `patients/{offsets,keys}` + `Session` dataclass |
| Adult glioma cross-sectional with rich metadata, CV | UCSF-PDGM | per-row `metadata/*` (age, sex, who_grade, IDH/MGMT/1p19q) |
| Pediatric / unusual pathology, BraTS-style, defaced, OOD test | BraTS-PED | + Step 2 HD-BET skull-strip routine |
| Multi-vendor academic, longitudinal, multi-pathology | (not yet — Málaga in-house) | requires manifest CSV + dedup |

---

## Reference paths (snapshot — `src/external/LINKS.md` wins on drift)

| Asset | Path |
|---|---|
| MAISI VAE-GAN (local) | `/media/mpascual/Sandisk2TB/checkpoints/MAISI_V2_RM/NV-Generate-MR/models/autoencoder_v2.pt` |
| MAISI VAE-GAN (server3) | `/media/hddb/mario/checkpoints/MAISI_V2_RM/NV-Generate-MR/models/autoencoder_v2.pt` |
| MAISI VAE-GAN (Picasso) | `/mnt/home/users/tic_163_uma/mpascual/fscratch/checkpoints/NV-Generate-MR/models/autoencoder_v2.pt` |
| HD-BET CLI (local) | `~/.conda/envs/hdbet/bin/hd-bet` |
| Picasso SSH | `mpascual@picasso3.scbi.uma.es` (key-based via ssh-agent) |
| icai-server SSH | bare alias `icai-server` → `mariopascual@150.214.214.47:33430` |

---

## If the user is in autonomous mode

Schedule wakeups at 10–20 min intervals while HD-BET / encode / rsync runs in the background. Use `<<autonomous-loop-dynamic>>` as the prompt. Pattern from prior sessions:

- HD-BET full run: ~14 s/patient TTA on RTX 4060 → ~1 h per 250 patients. Wakeup 20 min.
- Image-H5 convert: ~1 patient/s → ~5 min for 261. Wakeup 7 min.
- rsync 5 GB local → server3: ~10 MB/s → ~9 min. Wakeup 12 min.
- Encode 260 patients on RTX 4090: ~10 min. Wakeup 12 min.
- scp -3 server3→Picasso 2 GB: ~10 min. Wakeup 15 min.

Mark task #N completed only after the verification gate for that step passes — not at command launch.
