# Adding a new cohort to VENA

This recipe describes how to add a new pathology cohort (BraTS-MEN, Málaga,
mets, ...) so the rest of the training/eval pipeline picks it up automatically.
The work is partitioned across two layers:

1. **NIfTI-source reader** — a class satisfying
   `vena.data.cohort.CohortProtocol`, registered with `register_cohort`.
2. **Image-domain H5 converter** — a routine under `routines/h5_datasets/<name>/`
   that turns NIfTI into the project's H5 schema.

Once both exist, the cohort plugs into the latent pipeline (`routines/encode/maisi`),
the training data path (`MultiCohortLatentDataModule`), and exhaustive validation
without any changes to the FM trainer.

## 1. NIfTI reader

Create `src/vena/data/niigz/<name>.py`:

```python
from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vena.data.cohort import register_cohort

logger = logging.getLogger(__name__)

_PATIENT_DIR_RE = re.compile(r"^<your-patient-folder-pattern>$")


@dataclass(frozen=True)
class MyCohortPatient:
    """Per-patient handle (cohort-specific fields go here)."""

    patient_id: str
    root: Path
    metadata: dict[str, Any] = field(default_factory=dict)


@register_cohort("my_cohort", pathology="meningioma")
class MyCohortDataset:
    """Index of <Cohort> at <source_root>."""

    def __init__(self, source_root: Path | str) -> None:
        self.source_root = Path(source_root)
        if not self.source_root.is_dir():
            raise FileNotFoundError(f"source_root does not exist: {self.source_root}")
        self._patients = self._discover()
        self._index_by_id = {p.patient_id: i for i, p in enumerate(self._patients)}
        logger.info("%s: discovered %d patients", type(self).__name__, len(self._patients))

    def _discover(self) -> list[MyCohortPatient]: ...   # cohort-specific glob

    def __len__(self) -> int: return len(self._patients)
    def __iter__(self) -> Iterator[MyCohortPatient]: return iter(self._patients)
    def __getitem__(self, key: int | str) -> MyCohortPatient: ...   # ditto
    def ids(self) -> list[str]: return [p.patient_id for p in self._patients]
```

**Conformance check**: `from vena.data.cohort import CohortProtocol;
isinstance(MyCohortDataset(...), CohortProtocol)` should return `True` at
runtime (the protocol is `@runtime_checkable`).

## 2. H5 converter

Mirror the existing `routines/h5_datasets/ucsf_pdgm/` layout under
`routines/h5_datasets/<name>/`:

* `cli.py` — argparse entry point (`vena-h5-<name>`).
* `engine/` — thin engine wrapping the library-side converter at
  `src/vena/data/h5/<name>/image_domain/convert.py`.
* `configs/default.yaml` — typed via Pydantic `_RoutineConfig`.
* The latent-domain converter is **shared** at
  `src/vena/data/h5/latent_domain/convert.py`; do not duplicate.

Register the CLI in `pyproject.toml [project.scripts]`:

```
vena-h5-my-cohort = "routines.h5_datasets.my_cohort.cli:main"
```

## 3. Add to the corpus registry

Append a cohort entry to `routines/fm/train/configs/corpus/corpus_<host>.json`
(one entry per host: `local`, `server3`, ...):

```json
{
  "name": "MyCohort",
  "pathology": "meningioma",
  "label_system": "BraTS2024",
  "role": "external",
  "longitudinal": false,
  "image_h5": "/path/to/MyCohort_image.h5",
  "latent_h5": "/path/to/MyCohort_latents.h5",
  "n_patients": 0,
  "n_scans": 0,
  "modalities": ["t1pre", "t1c", "t2", "flair"],
  "has_swan": false
}
```

`role` is either `cv` (contributes to train/val/test) or `external`
(test-only, used by the external validation routine).

## 4. Tests

Add `tests/data/niigz/test_<name>.py` exercising the NIfTI reader on a
synthetic 2-patient fixture (no real data). Add
`tests/data/h5/test_<name>_image_convert_smoke.py` mirroring the UCSF-PDGM
smoke. Mark `@pytest.mark.unit`.

## What you do **not** need to touch

* `FMLightningModule` — cohort-agnostic, reads latent samples through
  `MultiCohortLatentDataset`.
* `MultiCohortLatentDataModule` — discovers cohorts via the registry; the
  `TemperatureBalancedSampler` weights them by `tau`.
* The exhaustive-validation engine — already iterates the registry.
* Any callback or metric module.
