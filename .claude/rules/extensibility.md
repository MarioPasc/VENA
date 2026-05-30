# Extensibility Patterns

How to add new pathology cohorts, new datasets, and how to share frozen-model
primitives across the pipeline. Pins the contracts introduced in the
pre-long-run hardening pass.

## `src/vena/common/` is the canonical adapter surface

1. **All cross-module access to the frozen MAISI VAE goes through
   `vena.common`**, never directly through `vena.model.autoencoder.maisi.*`.
   The package re-exports `MaisiEncoder`, `MaisiDecoder`, `load_autoencoder`,
   `AutoencoderHandle`, `percentile_normalise`, `DepthPad`, `CropPadSpec`,
   `apply_crop_pad`, plus the `SPATIAL_COMPRESSION` / `LATENT_CHANNELS`
   constants. Reach into the autoencoder subpackage only when extending the
   adapter itself.
2. **Decode helpers live in `vena.common.decode`**, never duplicated in
   engines:
   - `decode_box(decoder, latent, crop_spec, *, return_seconds=False)` for the
     full-volume brain-box path (exhaustive validation, external eval).
   - `decode_depth_identity(decoder, latent)` for the in-process training-time
     proxy where the latent was stored under the depth-pad convention.
   New decode paths get a new named helper here; never inline `MaisiDecoder.decode`
   in a routine engine.
3. **Adding a new cross-cutting primitive** (e.g. a shared metric helper, a
   shared sampler wrapper): add the symbol to `vena.common.__init__.__all__`
   *and* keep its source under whichever subpackage already owns the
   implementation. `vena.common` is a re-export layer, not a code-hosting one.

## `src/vena/data/cohort/` — adding a new pathology

1. **Every NIfTI cohort reader registers itself** via
   `@register_cohort("<name>", pathology="...")` at import time. The decorator
   is in `vena.data.cohort.register_cohort`. Look up the registry via
   `vena.data.cohort.get_cohort_registry()`.
2. **Cohort handles satisfy `CohortPatient`**: `patient_id: str`,
   `root: Path`, `metadata: dict[str, Any]` (use `field(default_factory=dict)`
   when no metadata exists yet). The dataset class satisfies
   `CohortProtocol[T]` structurally: `source_root: Path`, `__len__`,
   `__iter__`, `__getitem__`, `ids()`.
3. **Pathology labels are a closed `Literal`** in `vena.data.cohort.protocol`.
   Adding a new pathology means extending the literal in the same change as
   the cohort decorator — never pass an unannounced string.
4. **Adding a cohort is the recipe in `src/vena/data/cohort/HOWTO.md`**:
   write a `niigz` reader, write a `routines/h5_datasets/<name>/` converter,
   add a registry entry to `routines/fm/train/configs/corpus/corpus_*.json`,
   write tests. The FM trainer, validation, and registry-driven DataModule
   require no changes.
5. **Tests required**: `tests/data/cohort/test_<name>.py` exercising the
   protocol against a synthetic on-disk fixture (no real cohort data); mark
   `pytestmark = pytest.mark.unit`.

## Multi-cohort data path

1. **`MultiCohortLatentDataModule` is the only training DataModule.** The
   legacy single-cohort `LatentH5DataModule` was removed; the
   `tests/model/fm/test_legacy_dataset_removed.py` guard fails if anyone
   re-introduces it. The per-cohort `LatentH5Dataset` *is* retained — it is
   the building block consumed by `MultiCohortLatentDataset`.
2. **`data.corpus_registry` is required** in every training YAML.
   `data.latents_h5` is rejected at config-validation time with a clear
   error pointing at `routines/fm/train/configs/corpus/`.
3. **Single-cohort runs use a single-cohort registry** (a JSON listing only
   that cohort), not the legacy single-H5 path.
4. **Exhaustive-val takes the same registry**:
   `exhaustive_val.corpus_registry` is required when
   `exhaustive_val.enabled=true`. Legacy `exhaustive_val.image_h5` is rejected
   with a similar fast-fail.
5. **Per-cohort weighting is sampling-side only.** `TemperatureBalancedSampler`
   weights by `N_c^tau`; no per-cohort loss weighting in the LightningModule.

## Adding a new cross-cutting helper

When the same logic appears twice (e.g. decoding, normalisation, region
masking), move it into the closest existing namespace and import from there.
Acceptable destinations, in order of preference:

1. `vena.common` (or `vena.common.<topic>`) — for primitives reused across
   layers (data ↔ model ↔ routine).
2. `vena.model.fm.<area>` — for FM-only helpers.
3. `vena.data.<area>` — for data-only helpers.

Never copy-paste a helper into an engine. Never reach into a sibling routine
to import its helpers.
