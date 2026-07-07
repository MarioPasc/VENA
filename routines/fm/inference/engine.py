"""Unified validation-inference engine.

Single positional YAML arg (per ``preflight-pattern.md`` invariant 1).
Sweeps every method × cohort × test patient × NFE, writes
per-(method × cohort × NFE) prediction H5 files conforming to validation
§5.3, renders one multi-method comparison PNG per cohort, and emits a
``decision.json`` summarising the run.

The engine is **single-GPU sequential** — each adapter teardown frees
VRAM before the next method's setup. This matches the user's stated
constraint ("using only 1 GPU sequentially") and the loginexa smoke
budget.
"""

from __future__ import annotations

import datetime as _dt  # Py3.10 compat — datetime.UTC is 3.11+; use _dt.datetime.utcnow()
import hashlib
import json
import logging
import subprocess
from pathlib import Path

# Anchor the _dt usage so ruff autoflake does not strip the import.
_DATETIME_MOD = _dt
from typing import Any, Literal

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field

from routines.fm.inference.exceptions import (
    CohortFilterError,
    InferenceConfigError,
    ModelRegistryError,
)
from vena.data.registry import CohortEntry, CorpusRegistry, load_registry
from vena.inference import (
    InferenceModel,
    InferenceResult,
    get_inference_factory,
)

# vena.inference.figure transitively imports vena.model.fm.eval which
# pulls MAISI. Defer the import to first use so a competitor-env process
# (vena-syndiff, vena-v100-syndiff) that has figure.enabled=false does
# not pay the import cost — and does not surface a missing MAISI dep.
from vena.inference.h5_writer import (
    PRODUCER,
    PerPatientRecord,
    assert_predictions_valid,
    write_predictions_h5,
)
from vena.inference.image_dataset import (
    harmonised_modalities_for_record,
    resolve_test_scan_patient_pairs,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- config


class _CohortsFilter(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    cv_test: list[str] | None = None
    test_only: list[str] | None = None
    exclude: list[str] = Field(default_factory=list)


class _MethodsFilter(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    include: list[str] | None = None
    exclude: list[str] = Field(default_factory=list)


class _SmokeCfg(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    enabled: bool = False
    n_patients_per_cohort: int = 1
    use_selection_nfe_only: bool = True


class _FigureCfg(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    enabled: bool = True
    n_slices: int = 7
    slice_offset: int = 10


class InferenceJobConfig(BaseModel):
    """Single-YAML schema for the unified inference routine."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id_tag: str
    output_root: Path
    corpus_registry: Path
    models_yaml: Path
    fold: int = 0
    device: str = "cuda:0"
    warmup_passes: int = 0
    cohorts: _CohortsFilter = Field(default_factory=_CohortsFilter)
    methods: _MethodsFilter = Field(default_factory=_MethodsFilter)
    smoke: _SmokeCfg = Field(default_factory=_SmokeCfg)
    nfe_override: list[int] | None = None
    figure: _FigureCfg = Field(default_factory=_FigureCfg)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @classmethod
    def from_yaml(cls, path: Path | str) -> InferenceJobConfig:
        path = Path(path)
        if not path.is_file():
            raise InferenceConfigError(f"config YAML not found: {path}")
        with path.open("r") as f:
            return cls.model_validate(yaml.safe_load(f))


class _MethodEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str
    type: str
    kwargs: dict[str, Any] = Field(default_factory=dict)


class _ModelsYAML(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: str = "1.0"
    methods: list[_MethodEntry]


# ---------------------------------------------------------------------------- engine


class InferenceEngine:
    """Multi-method × cohort × patient × NFE sequential sweep."""

    def __init__(self, cfg: InferenceJobConfig) -> None:
        self.cfg = cfg
        self.run_dir = (cfg.output_root / cfg.run_id_tag).expanduser().resolve()
        # (cohort, scan_id) -> true CSR patient key, so each predictions-H5
        # row records its patient for Phase-2 patient-level pooling and
        # patient-stratified bootstrap (validation §6.4).
        self._scan_to_patient: dict[tuple[str, str], str] = {}

    # ------------------------------------------------------------------ entry

    def run(self) -> Path:
        torch.set_float32_matmul_precision("high")
        self._prepare_run_dir()
        self._attach_file_logger()
        logger.info("inference routine start | run_dir=%s", self.run_dir)

        registry = self._load_corpus()
        cohorts = self._resolve_cohorts(registry)
        models_yaml = self._load_models_yaml()
        method_entries = self._filter_methods(models_yaml.methods)

        # Resolve (scan_id, patient_id) pairs per cohort up-front (smoke
        # truncates here). Scan IDs drive the per-scan inference; the patient
        # key is stashed so each predictions-H5 row records its true patient
        # for Phase-2 patient-level pooling (validation §6.4).
        cohort_patients: dict[str, list[str]] = {}
        for c in cohorts:
            pairs = resolve_test_scan_patient_pairs(c, fold=self.cfg.fold)
            if self.cfg.smoke.enabled:
                pairs = pairs[: self.cfg.smoke.n_patients_per_cohort]
            if not pairs:
                logger.warning("cohort '%s' has no resolved test patients; skipping", c.name)
                continue
            for scan_id, patient_id in pairs:
                self._scan_to_patient[(c.name, scan_id)] = patient_id
            cohort_patients[c.name] = [scan_id for scan_id, _ in pairs]
            logger.info(
                "cohort '%s': %d test scans to process", c.name, len(cohort_patients[c.name])
            )

        # Pre-compute reference modalities + masks once per (cohort, patient) so
        # every method's H5 row gets the same reference block byte-for-byte.
        reference_cache = self._build_reference_cache(cohorts, cohort_patients)

        # Per-method sweep. The adapter teardown frees GPU memory before the
        # next adapter's setup so we never have two on the device at once.
        per_cohort_selection_pred: dict[str, dict[str, tuple[str, int, float, torch.Tensor]]] = {
            c: {} for c in cohort_patients
        }
        per_method_payload: dict[str, dict[str, Any]] = {}
        git_sha = _git_sha_or_none()

        failed_methods: list[tuple[str, str]] = []
        for entry in method_entries:
            try:
                adapter = self._instantiate(entry)
            except Exception as exc:
                logger.warning(
                    "method '%s' (type='%s') failed to instantiate (%s) — skipping",
                    entry.name,
                    entry.type,
                    exc,
                )
                failed_methods.append((entry.name, f"instantiate: {exc}"))
                continue
            try:
                logger.info("method '%s' (type='%s') setup", entry.name, entry.type)
                try:
                    adapter.setup()
                except Exception as exc:
                    logger.warning(
                        "method '%s': setup() raised %s: %s — skipping",
                        entry.name,
                        type(exc).__name__,
                        exc,
                    )
                    failed_methods.append((entry.name, f"setup: {exc}"))
                    continue

                nfe_list = self._effective_nfes(adapter)
                self._warmup(adapter, cohorts, cohort_patients, nfe_list)
                per_method_payload[entry.name] = {
                    "type": entry.type,
                    "kwargs": entry.kwargs,
                    "selection_nfe": adapter.selection_nfe,
                    "nfe_list": list(nfe_list),
                }

                for cohort in cohorts:
                    pids = cohort_patients.get(cohort.name)
                    if not pids:
                        continue
                    records_by_nfe: dict[int, list[PerPatientRecord]] = {n: [] for n in nfe_list}
                    for pid in pids:
                        try:
                            self._predict_one_patient(
                                adapter,
                                cohort,
                                pid,
                                nfe_list,
                                reference_cache,
                                records_by_nfe,
                                per_cohort_selection_pred,
                            )
                        except Exception as exc:
                            logger.warning(
                                "method '%s' cohort '%s' patient '%s': %s — skipping",
                                entry.name,
                                cohort.name,
                                pid,
                                exc,
                            )

                    # Flush per-NFE H5s for this cohort. Per-file write/validate
                    # failures are logged and skipped so one bad (method,cohort,nfe)
                    # tuple does not kill the rest of the sweep.
                    for nfe in nfe_list:
                        records = records_by_nfe[nfe]
                        if not records:
                            continue
                        out_path = (
                            self.run_dir
                            / "predictions"
                            / entry.name
                            / cohort.name
                            / f"nfe_{int(nfe):03d}.h5"
                        )
                        try:
                            write_predictions_h5(
                                out_path,
                                records,
                                method=entry.name,
                                cohort=cohort.name,
                                nfe=int(nfe),
                                ring=_ring_for_role(cohort.role),
                                git_sha=git_sha,
                                run_id_tag=self.cfg.run_id_tag,
                                extra_config={
                                    "method_type": entry.type,
                                    "kwargs": entry.kwargs,
                                    "selection_nfe": adapter.selection_nfe,
                                },
                            )
                            assert_predictions_valid(out_path)
                            logger.info("wrote %s (%d records)", out_path, len(records))
                        except Exception as exc:
                            logger.warning(
                                "H5 write/validate failed for %s @ NFE=%d (%s) — skipping",
                                out_path,
                                int(nfe),
                                exc,
                            )
                            failed_methods.append(
                                (entry.name, f"h5_validate {cohort.name}/nfe_{nfe}: {exc}")
                            )
            finally:
                try:
                    adapter.teardown()
                except Exception as exc:
                    logger.warning("teardown of '%s' raised: %s", entry.name, exc)

        # Per-cohort comparison figures.
        if self.cfg.figure.enabled:
            self._render_figures(
                cohorts,
                cohort_patients,
                reference_cache,
                per_cohort_selection_pred,
                per_method_payload,
            )

        decision_path = self._write_decision(
            cohorts, cohort_patients, method_entries, per_method_payload, git_sha
        )
        logger.info("inference routine complete | decision=%s", decision_path)
        return self.run_dir

    # ------------------------------------------------------------------ helpers

    def _prepare_run_dir(self) -> None:
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "predictions").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "figures").mkdir(parents=True, exist_ok=True)

    def _attach_file_logger(self) -> None:
        log_path = self.run_dir / "logs" / "inference.log"
        handler = logging.FileHandler(log_path)
        handler.setLevel(getattr(logging, self.cfg.log_level))
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s :: %(message)s"))
        root = logging.getLogger()
        root.addHandler(handler)
        root.setLevel(getattr(logging, self.cfg.log_level))

    def _load_corpus(self) -> CorpusRegistry:
        try:
            return load_registry(self.cfg.corpus_registry, require_latents=False)
        except Exception as exc:
            raise InferenceConfigError(
                f"failed to load corpus registry {self.cfg.corpus_registry}: {exc}"
            ) from exc

    def _resolve_cohorts(self, registry: CorpusRegistry) -> list[CohortEntry]:
        excluded = set(self.cfg.cohorts.exclude)
        wanted: list[CohortEntry] = []
        registry_by_name = {c.name: c for c in registry.cohorts}
        if self.cfg.cohorts.cv_test is None:
            wanted.extend(c for c in registry.cv_cohorts() if c.name not in excluded)
        else:
            for name in self.cfg.cohorts.cv_test:
                if name in excluded:
                    continue
                if name not in registry_by_name:
                    raise CohortFilterError(f"cv_test cohort '{name}' not in corpus registry")
                c = registry_by_name[name]
                if c.role != "cv":
                    raise CohortFilterError(f"cohort '{name}' has role '{c.role}', expected 'cv'")
                wanted.append(c)
        if self.cfg.cohorts.test_only is None:
            wanted.extend(c for c in registry.test_cohorts() if c.name not in excluded)
        else:
            for name in self.cfg.cohorts.test_only:
                if name in excluded:
                    continue
                if name not in registry_by_name:
                    raise CohortFilterError(f"test_only cohort '{name}' not in corpus registry")
                c = registry_by_name[name]
                if c.role != "test_only":
                    raise CohortFilterError(
                        f"cohort '{name}' has role '{c.role}', expected 'test_only'"
                    )
                wanted.append(c)
        if not wanted:
            raise CohortFilterError("no cohorts selected after filtering")
        logger.info("cohorts in scope: %s", [c.name for c in wanted])
        return wanted

    def _load_models_yaml(self) -> _ModelsYAML:
        path = self.cfg.models_yaml
        if not path.is_file():
            raise ModelRegistryError(f"models YAML missing: {path}")
        with path.open("r") as f:
            raw = yaml.safe_load(f)
        try:
            return _ModelsYAML.model_validate(raw)
        except Exception as exc:
            raise ModelRegistryError(f"models YAML {path} failed schema validation: {exc}") from exc

    def _filter_methods(self, methods: list[_MethodEntry]) -> list[_MethodEntry]:
        excluded = set(self.cfg.methods.exclude)
        include = self.cfg.methods.include
        if include is not None:
            include_set = set(include)
            filtered = [m for m in methods if m.name in include_set and m.name not in excluded]
        else:
            filtered = [m for m in methods if m.name not in excluded]
        if not filtered:
            raise ModelRegistryError("no methods selected after include/exclude filters")
        logger.info("methods in scope: %s", [m.name for m in filtered])
        return filtered

    def _instantiate(self, entry: _MethodEntry) -> InferenceModel:
        factory = get_inference_factory(entry.type)
        kwargs = dict(entry.kwargs)
        kwargs.setdefault("name", entry.name)
        kwargs.setdefault("device", self.cfg.device)
        return factory(**kwargs)

    def _effective_nfes(self, adapter: InferenceModel) -> tuple[int, ...]:
        if self.cfg.smoke.enabled and self.cfg.smoke.use_selection_nfe_only:
            return (adapter.selection_nfe,)
        if self.cfg.nfe_override is not None:
            requested = tuple(int(n) for n in self.cfg.nfe_override)
            # Honour the adapter's nfe_list constraint: a 1-shot model
            # (Pix2Pix, GAN, identity) collapses any override to (1,).
            if adapter.nfe_list == (1,):
                return (1,)
            return requested
        return tuple(adapter.nfe_list)

    def _warmup(
        self,
        adapter: InferenceModel,
        cohorts: list[CohortEntry],
        cohort_patients: dict[str, list[str]],
        nfe_list: tuple[int, ...],
    ) -> None:
        if self.cfg.warmup_passes <= 0:
            return
        # Use the first available (cohort, patient) for warmup.
        for c in cohorts:
            pids = cohort_patients.get(c.name)
            if pids:
                logger.info(
                    "method '%s': %d warmup passes on %s/%s",
                    adapter.name,
                    self.cfg.warmup_passes,
                    c.name,
                    pids[0],
                )
                for _ in range(self.cfg.warmup_passes):
                    try:
                        adapter.predict(c, pids[0], nfe_list[0])
                    except Exception as exc:
                        logger.warning(
                            "method '%s' warmup raised: %s — continuing", adapter.name, exc
                        )
                        return
                return

    def _predict_one_patient(
        self,
        adapter: InferenceModel,
        cohort: CohortEntry,
        patient_id: str,
        nfe_list: tuple[int, ...],
        reference_cache: dict[tuple[str, str], dict[str, Any]],
        records_by_nfe: dict[int, list[PerPatientRecord]],
        per_cohort_selection_pred: dict[str, dict[str, tuple[str, int, float, torch.Tensor]]],
    ) -> None:
        ref = reference_cache[(cohort.name, patient_id)]
        for nfe in nfe_list:
            result: InferenceResult = adapter.predict(cohort, patient_id, int(nfe))
            # The harmonised volume may live on the box geometry (latent-tier)
            # OR the native geometry (image-tier). Both pass through
            # apply_harmonisation, so dtype + range are correct; we only need
            # to coerce to numpy in the H5's expected shape (= the reference
            # T1c's shape since the H5 row is shape-aligned across modalities).
            target_shape = ref["t1c_real_harmonised"].shape
            harmonised_np = _to_target_shape(result.t1c_synthetic_harmonised, target_shape)
            raw_np = _to_target_shape(result.t1c_synthetic_raw, target_shape)

            records_by_nfe[nfe].append(
                PerPatientRecord(
                    patient_id=self._scan_to_patient.get((cohort.name, patient_id), patient_id),
                    scan_id=patient_id,
                    cohort=cohort.name,
                    t1c_synthetic_harmonised=harmonised_np,
                    t1c_synthetic_raw=raw_np,
                    t1c_real_harmonised=ref["t1c_real_harmonised"],
                    t1pre_harmonised=ref["t1pre_harmonised"],
                    t2_harmonised=ref["t2_harmonised"],
                    flair_harmonised=ref["flair_harmonised"],
                    brain_mask=ref["brain"],
                    wt_mask=ref["wt"],
                    inference_seconds=result.inference_seconds,
                    peak_vram_mb=result.peak_vram_mb,
                )
            )
            # Cache the selection-NFE prediction for the comparison figure.
            if int(nfe) == int(adapter.selection_nfe):
                if patient_id not in per_cohort_selection_pred[cohort.name]:
                    per_cohort_selection_pred[cohort.name][patient_id] = {}  # type: ignore[assignment]
                bucket = per_cohort_selection_pred[cohort.name][patient_id]
                if not isinstance(bucket, dict):  # paranoia: typing seam
                    bucket = {}
                # Cache as (method_name, nfe, seconds, volume); engine
                # writes one entry per method per patient.
                bucket[adapter.name] = (
                    adapter.name,
                    int(nfe),
                    float(result.inference_seconds),
                    torch.from_numpy(harmonised_np),
                )
                per_cohort_selection_pred[cohort.name][patient_id] = bucket  # type: ignore[assignment]

    def _build_reference_cache(
        self,
        cohorts: list[CohortEntry],
        cohort_patients: dict[str, list[str]],
    ) -> dict[tuple[str, str], dict[str, Any]]:
        cache: dict[tuple[str, str], dict[str, Any]] = {}
        for c in cohorts:
            for pid in cohort_patients.get(c.name, []):
                try:
                    (
                        t1pre_h,
                        t2_h,
                        flair_h,
                        t1c_h,
                        brain,
                        wt,
                    ) = harmonised_modalities_for_record(c.image_h5, pid)
                except Exception as exc:
                    logger.warning(
                        "reference build failed for %s/%s: %s — dropping patient",
                        c.name,
                        pid,
                        exc,
                    )
                    continue
                cache[(c.name, pid)] = {
                    "t1pre_harmonised": t1pre_h,
                    "t2_harmonised": t2_h,
                    "flair_harmonised": flair_h,
                    "t1c_real_harmonised": t1c_h,
                    "brain": brain,
                    "wt": wt,
                }
        return cache

    def _render_figures(
        self,
        cohorts: list[CohortEntry],
        cohort_patients: dict[str, list[str]],
        reference_cache: dict[tuple[str, str], dict[str, Any]],
        per_cohort_selection_pred: dict[str, dict[str, tuple[str, int, float, torch.Tensor]]],
        per_method_payload: dict[str, dict[str, Any]],
    ) -> None:
        for c in cohorts:
            pids = cohort_patients.get(c.name)
            if not pids:
                continue
            # Pick the patient with the most methods predicted (smoke usually =1 patient).
            best_pid = pids[0]
            best_count = -1
            patients_dict = per_cohort_selection_pred.get(c.name, {})
            for pid in pids:
                bucket = patients_dict.get(pid, {})
                if isinstance(bucket, dict) and len(bucket) > best_count:
                    best_count = len(bucket)
                    best_pid = pid
            ref = reference_cache.get((c.name, best_pid))
            if ref is None:
                continue
            method_predictions = []
            bucket = patients_dict.get(best_pid, {})
            if not isinstance(bucket, dict):
                continue
            for method_name in per_method_payload:
                if method_name not in bucket:
                    continue
                name, nfe, seconds, vol = bucket[method_name]
                method_predictions.append((name, vol, int(nfe), float(seconds)))
            if not method_predictions:
                continue
            out_path = self.run_dir / "figures" / f"{c.name}.png"
            from vena.inference.figure import render_multi_method_figure

            render_multi_method_figure(
                cohort=c.name,
                patient_id=best_pid,
                real_t1c=ref["t1c_real_harmonised"],
                method_predictions=method_predictions,
                out_path=out_path,
                n_slices=self.cfg.figure.n_slices,
                slice_offset=self.cfg.figure.slice_offset,
                title_suffix=("smoke" if self.cfg.smoke.enabled else None),
            )
            logger.info("wrote figure %s (%d methods)", out_path, len(method_predictions))

    def _write_decision(
        self,
        cohorts: list[CohortEntry],
        cohort_patients: dict[str, list[str]],
        method_entries: list[_MethodEntry],
        per_method_payload: dict[str, dict[str, Any]],
        git_sha: str | None,
    ) -> Path:
        registry_hash = _file_sha256(self.cfg.corpus_registry)
        models_yaml_hash = _file_sha256(self.cfg.models_yaml)
        payload = {
            "schema_version": "1.0",
            "produced_at": _dt.datetime.utcnow().isoformat() + "Z",
            "producer": PRODUCER,
            "run_id_tag": self.cfg.run_id_tag,
            "run_dir": str(self.run_dir),
            "git_sha": git_sha,
            "corpus_registry": str(self.cfg.corpus_registry),
            "corpus_registry_sha256": registry_hash,
            "models_yaml": str(self.cfg.models_yaml),
            "models_yaml_sha256": models_yaml_hash,
            "cohorts": {
                c.name: {
                    "role": c.role,
                    "image_h5": str(c.image_h5),
                    "latent_h5": str(c.latent_h5),
                    "patient_ids": cohort_patients.get(c.name, []),
                }
                for c in cohorts
            },
            "methods": [
                {
                    "name": m.name,
                    "type": m.type,
                    "kwargs": m.kwargs,
                    "selection_nfe": per_method_payload.get(m.name, {}).get("selection_nfe"),
                    "nfe_list": per_method_payload.get(m.name, {}).get("nfe_list"),
                }
                for m in method_entries
            ],
            "smoke": self.cfg.smoke.model_dump(),
            "device": self.cfg.device,
        }
        path = self.run_dir / "decision.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return path


# ---------------------------------------------------------------------------- helpers


def _ring_for_role(role: str) -> str:
    return {"cv": "A", "test_only": "B"}.get(role, "?")


def _file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha_or_none() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent),
        )
        return out.stdout.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _to_target_shape(vol: Any, target_shape: tuple[int, ...]) -> Any:
    """Centre-crop or zero-pad ``vol`` to ``target_shape`` in 3D.

    The latent-tier adapters return the decoded box volume (often equal
    to the native shape after crop_spec un-pad). The image-tier
    adapters return the native ``masks/brain`` shape directly. In both
    cases the predictions H5 schema expects the reference modality's
    shape; we coerce here so the row is shape-aligned across
    ``synth/raw/real/refs/masks``.
    """
    import numpy as np

    arr = vol.detach().cpu().numpy() if isinstance(vol, torch.Tensor) else np.asarray(vol)
    arr = arr.astype(np.float32, copy=False)
    if arr.shape == tuple(target_shape):
        return arr
    out = np.zeros(tuple(target_shape), dtype=np.float32)
    common = tuple(min(a, b) for a, b in zip(arr.shape, target_shape, strict=True))
    src = tuple(
        slice((arr.shape[i] - common[i]) // 2, (arr.shape[i] - common[i]) // 2 + common[i])
        for i in range(3)
    )
    dst = tuple(
        slice((target_shape[i] - common[i]) // 2, (target_shape[i] - common[i]) // 2 + common[i])
        for i in range(3)
    )
    out[dst] = arr[src]
    return out


__all__ = [
    "InferenceEngine",
    "InferenceJobConfig",
]
