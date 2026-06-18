"""Engine for the ``decoder_lpl_profile`` preflight.

Orchestrates the §4.1 + §4.2 + §4.7b sweeps documented in
``.claude/notes/changes/decoder_perceptual_loss_s3.md`` over N patients
× 5 augmentation variants, optionally sharded by cohort across 4
loginexa V100 GPUs.

This engine is intentionally pluggable: phase-1 and phase-2 model
artefacts (VAE handle, S1 module, sampler) are constructed lazily so a
smoke run can pass a stub VAE + stub model_call without instantiating
the real MAISI checkpoints. The production path constructs them via
``load_autoencoder`` + :class:`FMLightningModule.from_checkpoint`.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from vena.data.h5.shared import now_iso_utc

from .aggregate import aggregate, update_latest_symlink
from .patient_sampler import PatientPick, select_patients_by_strata
from .phase1_feature_stats import (
    per_channel_feature_stats,
    per_patient_block_magnitude,
)
from .phase2_separation import (
    error_concentration,
    pre_post_separation,
    x1_reliability_vs_t,
)
from .phase3_drift import (
    compute_drift,
    empty_wt_rate,
    per_cohort_w_nw_ratio,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic config
# ---------------------------------------------------------------------------


class _PatientSelectionCfg(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    n_per_cohort: int = 3
    cv_cohorts: list[str] = Field(default_factory=list)
    volume_strata: list[str] = Field(default_factory=lambda: ["small", "median", "large"])
    variants: list[str] = Field(default_factory=lambda: ["v0", "v1", "v2", "v3", "v4"])


class _CohortPaths(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str
    latent_h5: Path
    latent_aug_h5: Path | None = None


class _ProbeCfg(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    blocks: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    max_block: int = 5
    grad_checkpoint: bool = False


class _Phase2Cfg(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    sampler: str = "euler"
    nfe: int = 10
    t_sweep: list[float] = Field(
        default_factory=lambda: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    )
    enabled: bool = True


class _Phase3Cfg(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    drift_threshold: float = 0.20
    drift_fail_fraction: float = 0.25
    inter_cohort_spread_threshold: float = 1.5


class _EmitCfg(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    composite_figures: bool = False
    per_channel_distribution: bool = True


class _TrunkSpec(BaseModel):
    """Subset of the FM train ``model.trunk`` block needed to rebuild S1."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    checkpoint: Path
    arch_json: Path | None = None
    arch_overrides: dict[str, Any] = Field(default_factory=dict)
    class_token: int = 9
    spacing_mm: list[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0])
    trainable: bool = True
    regime: str = "fft"
    peft: dict[str, Any] | None = None


class _ControlNetSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    conditioning_inputs: list[str] = Field(default_factory=list)
    arch_overrides: dict[str, Any] = Field(default_factory=dict)


class _S1ModelCfg(BaseModel):
    """Mirror of the FM train ``model`` block + ``rflow`` + ``ema``."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    trunk: _TrunkSpec
    controlnet: _ControlNetSpec
    rflow: dict[str, Any] = Field(default_factory=dict)
    ema: dict[str, Any] = Field(default_factory=dict)


class DecoderLplProfileConfig(BaseModel):
    """Validated config schema for the preflight."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    output_root: Path
    vae_checkpoint: Path | None = None
    s1_checkpoint: Path | None = None
    s1_trunk_ema_snapshot: Path | None = None
    s1_model: _S1ModelCfg | None = None

    cohorts: list[_CohortPaths] = Field(default_factory=list)
    seed: int = 42
    device: str = "cuda:0"

    shard_id: int | None = None
    n_shards: int = 1

    patient_selection: _PatientSelectionCfg = Field(default_factory=_PatientSelectionCfg)
    probe: _ProbeCfg = Field(default_factory=_ProbeCfg)
    phase2: _Phase2Cfg = Field(default_factory=_Phase2Cfg)
    phase3: _Phase3Cfg = Field(default_factory=_Phase3Cfg)
    emit: _EmitCfg = Field(default_factory=_EmitCfg)

    @field_validator("output_root", mode="before")
    @classmethod
    def _to_path(cls, v: Any) -> Path:
        return Path(v)

    @classmethod
    def from_yaml(cls, path: Path | str) -> DecoderLplProfileConfig:
        text = Path(path).read_text()
        data = yaml.safe_load(text) or {}
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# Per-cell processing — pure function so a smoke run can drive it directly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PatientCell:
    cohort: str
    patient_id: str
    variant: str
    row_index: int  # row in the *source* H5 (clean or aug)
    wt_volume: float
    stratum: str


def _open_latent_h5(path: Path):
    return h5py.File(path, "r")


def _read_clean_latent(h5: Any, modality: str, row: int) -> torch.Tensor:
    return torch.from_numpy(h5[f"latents/{modality}"][row]).unsqueeze(0).float()


def _read_clean_masks(h5: Any, row: int) -> tuple[torch.Tensor, torch.Tensor]:
    tumor_lat = torch.from_numpy(h5["masks/tumor_latent"][row]).unsqueeze(0).float()
    soft = tumor_lat.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
    m_wt = (soft >= 0.5).float()
    m_brain = torch.from_numpy(h5["masks/brain_latent"][row]).unsqueeze(0).float()
    if m_brain.ndim == 4:
        m_brain = m_brain.unsqueeze(1)  # (1, 1, H, W, D)
    return m_wt, m_brain


def _aug_rows_for_patient(aug_h5: Any, patient_id: str) -> dict[str, int]:
    """Return ``{variant: row_index}`` for one patient in an aug H5."""
    ids = aug_h5["ids"][:]
    variants = aug_h5["variants"][:]
    out: dict[str, int] = {}
    for i, (raw_id, raw_v) in enumerate(zip(ids, variants)):
        pid = raw_id.decode() if isinstance(raw_id, bytes) else str(raw_id)
        var = raw_v.decode() if isinstance(raw_v, bytes) else str(raw_v)
        if pid == patient_id and var not in out:
            out[var] = i
    return out


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


# Optional artefact bundle. Production callers construct via
# ``_build_model_artefacts`` below; smokes pass a stub.
@dataclass
class ModelArtefacts:
    """Pluggable model bundle for the engine.

    The production engine builds these from a config; tests can construct
    a stub bundle in ~10 lines and drive ``run_cell`` directly.
    """

    feature_extractor_factory: Callable[
        [],
        Any,  # ContextManager yielding a FeatureExtractor closure
    ]
    sample_x1_s1: Callable[[torch.Tensor, dict[str, torch.Tensor]], torch.Tensor]
    velocity_call: Callable[[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]], torch.Tensor]
    build_conditioning: Callable[[Any, int], dict[str, torch.Tensor]]
    device: torch.device


class DecoderLplProfileEngine:
    """Pre-flight engine — emits the §4 deliverables under ``output_root``."""

    def __init__(
        self,
        cfg: DecoderLplProfileConfig,
        config_yaml_path: Path | None = None,
        *,
        model_artefacts: ModelArtefacts | None = None,
    ) -> None:
        self.cfg = cfg
        self.config_yaml_path = config_yaml_path
        self._model_artefacts = model_artefacts

    # --------------------------------------------------------------
    # Run loop
    # --------------------------------------------------------------

    def run(self) -> Path:
        """Process every (patient, variant) cell, write CSV partials.

        Returns the artefact directory. The aggregator is invoked at the
        end of the *single-shard* path so a smoke produces all
        deliverables in one go. A sharded run (``shard_id != None``)
        writes only its slice; a separate ``aggregate`` invocation must
        consolidate later.
        """
        out_dir = self._make_run_dir()
        self._persist_config(out_dir)

        # Slice cohorts by shard id (round-robin) when sharded.
        cohort_slice = self._cohort_slice()

        artefacts = self._model_artefacts or self._build_model_artefacts()

        # Per-shard CSV partials live under either ``shard_<i>/tables/``
        # (sharded) or ``tables/`` (single-shard).
        tables_dir = self._tables_dir(out_dir)
        tables_dir.mkdir(parents=True, exist_ok=True)

        # Accumulators (one row per call → write all at the end).
        rows_per_block_magnitude: list[dict] = []
        rows_per_channel: list[dict] = []
        rows_outlier_threshold: list[dict] = []
        rows_pre_post: list[dict] = []
        rows_error_conc: list[dict] = []
        rows_x1_t: list[dict] = []
        # WT volumes for empty_wt_rate.
        wt_volumes: dict[tuple[str, str], float] = {}
        # Per-(cohort, patient, variant) distance lookup for phase 3.
        distance_lookup: dict[tuple[str, str, str], dict[int, dict[str, float]]] = defaultdict(dict)

        blocks = tuple(self.cfg.probe.blocks)

        for cohort in cohort_slice:
            try:
                self._process_cohort(
                    cohort=cohort,
                    artefacts=artefacts,
                    out_dir=out_dir,
                    blocks=blocks,
                    rows_per_block_magnitude=rows_per_block_magnitude,
                    rows_per_channel=rows_per_channel,
                    rows_outlier_threshold=rows_outlier_threshold,
                    rows_pre_post=rows_pre_post,
                    rows_error_conc=rows_error_conc,
                    rows_x1_t=rows_x1_t,
                    wt_volumes=wt_volumes,
                    distance_lookup=distance_lookup,
                )
            except FileNotFoundError as exc:
                logger.warning("cohort %s skipped — file missing: %s", cohort.name, exc)
                continue

        # ---- Phase 3 aggregation (on this shard's data).
        drift_rows = compute_drift(distance_lookup, blocks=blocks)
        ratio_rows = per_cohort_w_nw_ratio(distance_lookup, block=5, variant="v0")
        empty_rows = empty_wt_rate(wt_volumes, cohorts=[c.name for c in cohort_slice])

        # ---- Persist shard CSVs.
        self._write_table(tables_dir / "per_block_magnitude.csv", rows_per_block_magnitude)
        self._write_table(tables_dir / "per_channel_L_dec_distribution.csv", rows_per_channel)
        self._write_table(tables_dir / "outlier_threshold.csv", rows_outlier_threshold)
        self._write_table(tables_dir / "pre_post_separation.csv", rows_pre_post)
        self._write_table(tables_dir / "error_concentration.csv", rows_error_conc)
        self._write_table(tables_dir / "x1_reliability_vs_t.csv", rows_x1_t)
        self._write_table(tables_dir / "drift_per_patient_variant.csv", drift_rows)
        self._write_table(tables_dir / "per_cohort_W_nW_ratio.csv", ratio_rows)
        self._write_table(tables_dir / "empty_wt_rate.csv", empty_rows)

        # ---- Aggregate in single-shard mode.
        if self.cfg.shard_id is None:
            decision = aggregate(
                out_dir,
                cohorts=[c.name for c in self.cfg.cohorts],
                variants=tuple(self.cfg.patient_selection.variants),
            )
            update_latest_symlink(out_dir)
            logger.info(
                "single-shard aggregate complete: decision @ %s, allowed=%s",
                out_dir / "decision.json",
                decision.allowed_variants,
            )
        else:
            print(f"SHARD_DONE shard_id={self.cfg.shard_id} out={out_dir}", flush=True)

        return out_dir

    # --------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------

    def _make_run_dir(self) -> Path:
        timestamp = now_iso_utc().replace(":", "-")
        root = Path(self.cfg.output_root) / timestamp
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _persist_config(self, out_dir: Path) -> None:
        (out_dir / "config_resolved.json").write_text(
            json.dumps(self.cfg.model_dump(mode="json"), indent=2, default=str)
        )

    def _tables_dir(self, out_dir: Path) -> Path:
        if self.cfg.shard_id is not None:
            return out_dir / f"shard_{self.cfg.shard_id}" / "tables"
        return out_dir / "tables"

    def _cohort_slice(self) -> list[_CohortPaths]:
        if self.cfg.shard_id is None or self.cfg.n_shards <= 1:
            return list(self.cfg.cohorts)
        # Round-robin assignment so heavy cohorts spread across shards.
        return [
            c for i, c in enumerate(self.cfg.cohorts) if i % self.cfg.n_shards == self.cfg.shard_id
        ]

    @staticmethod
    def _write_table(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            # Write an empty file with no header — aggregator handles it.
            path.write_text("")
            return
        fieldnames = list(rows[0])
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    def _build_model_artefacts(self) -> ModelArtefacts:
        """Production VAE / S1 module / sampler / closures build."""
        from vena.common import load_autoencoder
        from vena.model.fm.inference import get_sampler
        from vena.model.fm.lightning import FMLightningModule
        from vena.model.fm.lpl import decoder_feature_extractor
        from vena.model.fm.maisi.config import TrunkConfig

        if (
            self.cfg.vae_checkpoint is None
            or self.cfg.s1_checkpoint is None
            or self.cfg.s1_model is None
        ):
            raise RuntimeError(
                "Real-data preflight requires vae_checkpoint + s1_checkpoint"
                " + s1_model in the YAML. Pass model_artefacts=... for smokes."
            )

        device = torch.device(self.cfg.device)
        handle = load_autoencoder(self.cfg.vae_checkpoint, device=str(device))

        s1 = self.cfg.s1_model
        trunk_cfg = TrunkConfig(
            checkpoint=s1.trunk.checkpoint,
            arch_json=s1.trunk.arch_json,
            arch_overrides=dict(s1.trunk.arch_overrides),
            class_token=int(s1.trunk.class_token),
            spacing_mm=list(s1.trunk.spacing_mm),
            trainable=bool(s1.trunk.trainable),
            regime=str(s1.trunk.regime),
            peft=dict(s1.trunk.peft) if s1.trunk.peft else None,
        )
        module = FMLightningModule(
            trunk_config=trunk_cfg,
            conditioning_specs=list(s1.controlnet.conditioning_inputs),
            stage="S1",
            controlnet_arch_overrides=dict(s1.controlnet.arch_overrides),
            rflow_cfg=dict(s1.rflow),
            ema_cfg=dict(s1.ema),
            region_resolver=None,
            vae_decoder=None,
        )
        module = module.to(device)
        if self.cfg.s1_trunk_ema_snapshot is not None:
            module.set_pending_trunk_ema_snapshot(Path(self.cfg.s1_trunk_ema_snapshot))
        module.setup()

        # Load the ControlNet EMA shadow from the S1 Lightning ckpt.
        ckpt = torch.load(self.cfg.s1_checkpoint, map_location=device, weights_only=False)
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        ema_prefix = "ema._ema.ema_model."
        ema_state = {
            k.removeprefix(ema_prefix): v for k, v in state_dict.items() if k.startswith(ema_prefix)
        }
        if not ema_state:
            raise RuntimeError(
                f"No keys matching {ema_prefix!r} in {self.cfg.s1_checkpoint};"
                " is this a Lightning ckpt from the S1 run?"
            )
        missing, unexpected = module.ema.ema_model.load_state_dict(ema_state, strict=False)
        logger.info(
            "ControlNet EMA shadow loaded: %d tensors, %d missing, %d unexpected",
            len(ema_state),
            len(missing),
            len(unexpected),
        )

        # Fall back to the Lightning ckpt's trunk_ema state when no R6
        # snapshot was supplied (production S1 runs predate R6).
        if module.trunk_ema is not None and self.cfg.s1_trunk_ema_snapshot is None:
            trunk_ema_prefix = "trunk_ema._ema.ema_model."
            trunk_ema_state = {
                k.removeprefix(trunk_ema_prefix): v
                for k, v in state_dict.items()
                if k.startswith(trunk_ema_prefix)
            }
            if trunk_ema_state:
                module.trunk_ema.ema_model.load_state_dict(trunk_ema_state, strict=False)
                logger.info(
                    "Trunk EMA shadow loaded from Lightning ckpt: %d tensors",
                    len(trunk_ema_state),
                )
            else:
                logger.warning(
                    "Trunk is trainable but no trunk_ema state in ckpt;"
                    " sampling uses the live trunk."
                )

        module.eval()
        sampler = get_sampler("euler")(scheduler=module.rflow.scheduler)
        blocks = frozenset(int(b) for b in self.cfg.probe.blocks)
        max_block = int(self.cfg.probe.max_block)
        grad_ckpt = bool(self.cfg.probe.grad_checkpoint)
        nfe = int(self.cfg.phase2.nfe)

        def feature_extractor_factory():
            return decoder_feature_extractor(
                handle,
                blocks=blocks,
                max_block=max_block,
                grad_checkpoint=grad_ckpt,
            )

        def build_conditioning(src_h5: Any, row: int) -> dict[str, torch.Tensor]:
            out: dict[str, Any] = {"patient_id": "preflight"}
            for mod in ("t1pre", "t2", "flair"):
                arr = src_h5[f"latents/{mod}"][row]
                out[f"z_{mod}"] = torch.from_numpy(arr).unsqueeze(0).float().to(device)
            tumor_lat = src_h5["masks/tumor_latent"][row]
            soft = np.clip(tumor_lat.sum(axis=0, keepdims=True), 0.0, 1.0)
            m_wt = (soft >= 0.5).astype(np.float32)
            out["m_wt"] = torch.from_numpy(m_wt).unsqueeze(0).float().to(device)
            if "masks/brain_latent" in src_h5:
                brain = src_h5["masks/brain_latent"][row]
                brain_t = torch.from_numpy(brain).float()
                if brain_t.ndim == 3:
                    brain_t = brain_t.unsqueeze(0)
                out["m_brain"] = brain_t.unsqueeze(0).to(device)
            return out

        def sample_x1_s1(
            z_target: torch.Tensor, conditioning_batch: dict[str, torch.Tensor]
        ) -> torch.Tensor:
            module.compute_val_conditioning(conditioning_batch)
            x0 = torch.randn_like(z_target)
            model_call = module._make_ema_call()
            with torch.inference_mode():
                return sampler.sample(model_call, x0, num_inference_steps=nfe)

        def velocity_call(
            x_t: torch.Tensor,
            t_tensor: torch.Tensor,
            conditioning_batch: dict[str, torch.Tensor],
        ) -> torch.Tensor:
            module.compute_val_conditioning(conditioning_batch)
            model_call = module._make_ema_call()
            with torch.no_grad():
                return model_call(x_t, t_tensor)

        return ModelArtefacts(
            feature_extractor_factory=feature_extractor_factory,
            sample_x1_s1=sample_x1_s1,
            velocity_call=velocity_call,
            build_conditioning=build_conditioning,
            device=device,
        )

    # --------------------------------------------------------------
    # Per-cohort iteration
    # --------------------------------------------------------------

    def _process_cohort(
        self,
        *,
        cohort: _CohortPaths,
        artefacts: ModelArtefacts,
        out_dir: Path,
        blocks: tuple[int, ...],
        rows_per_block_magnitude: list[dict],
        rows_per_channel: list[dict],
        rows_outlier_threshold: list[dict],
        rows_pre_post: list[dict],
        rows_error_conc: list[dict],
        rows_x1_t: list[dict],
        wt_volumes: dict[tuple[str, str], float],
        distance_lookup: dict[tuple[str, str, str], dict[int, dict[str, float]]],
    ) -> None:
        # When an augmented latent H5 exists, constrain the sampler to its
        # intersection with the clean H5 ids so phase 1/2/3 see all 5
        # variants per patient (the §4.7b drift gate is variant-dense; a
        # patient with only v0 contributes nothing to drift).
        eligible_ids: set[str] | None = None
        if cohort.latent_aug_h5 is not None:
            with _open_latent_h5(cohort.latent_aug_h5) as aug_h5:
                aug_ids_raw = aug_h5["ids"][:]
                eligible_ids = {
                    (i.decode() if isinstance(i, bytes) else str(i)) for i in aug_ids_raw
                }
            logger.info(
                "[cohort %s] aug-coverage filter: %d unique patient ids in aug H5",
                cohort.name,
                len(eligible_ids),
            )
        picks = select_patients_by_strata(
            cohort.latent_h5,
            n_per_cohort=self.cfg.patient_selection.n_per_cohort,
            volume_strata=self.cfg.patient_selection.volume_strata,
            eligible_ids=eligible_ids,
            seed=self.cfg.seed,
        )
        if not picks:
            logger.warning("cohort %s: no patients picked", cohort.name)
            return
        shard_tag = f"shard{self.cfg.shard_id}" if self.cfg.shard_id is not None else "single"
        logger.info(
            "[%s] cohort %s: %d patients, variants=%s",
            shard_tag,
            cohort.name,
            len(picks),
            self.cfg.patient_selection.variants,
        )

        with _open_latent_h5(cohort.latent_h5) as clean_h5:
            aug_h5_ctx = (
                _open_latent_h5(cohort.latent_aug_h5) if cohort.latent_aug_h5 is not None else None
            )
            try:
                for pick in picks:
                    self._process_patient(
                        cohort=cohort,
                        pick=pick,
                        artefacts=artefacts,
                        clean_h5=clean_h5,
                        aug_h5=aug_h5_ctx,
                        blocks=blocks,
                        rows_per_block_magnitude=rows_per_block_magnitude,
                        rows_per_channel=rows_per_channel,
                        rows_outlier_threshold=rows_outlier_threshold,
                        rows_pre_post=rows_pre_post,
                        rows_error_conc=rows_error_conc,
                        rows_x1_t=rows_x1_t,
                        wt_volumes=wt_volumes,
                        distance_lookup=distance_lookup,
                    )
            finally:
                if aug_h5_ctx is not None:
                    aug_h5_ctx.close()

    def _process_patient(
        self,
        *,
        cohort: _CohortPaths,
        pick: PatientPick,
        artefacts: ModelArtefacts,
        clean_h5: Any,
        aug_h5: Any | None,
        blocks: tuple[int, ...],
        rows_per_block_magnitude: list[dict],
        rows_per_channel: list[dict],
        rows_outlier_threshold: list[dict],
        rows_pre_post: list[dict],
        rows_error_conc: list[dict],
        rows_x1_t: list[dict],
        wt_volumes: dict[tuple[str, str], float],
        distance_lookup: dict[tuple[str, str, str], dict[int, dict[str, float]]],
    ) -> None:
        wt_volumes[(cohort.name, pick.patient_id)] = pick.wt_volume

        # Resolve per-variant source.
        sources: dict[str, tuple[Any, int]] = {"v0": (clean_h5, pick.row_index)}
        if aug_h5 is not None:
            aug_map = _aug_rows_for_patient(aug_h5, pick.patient_id)
            for v in self.cfg.patient_selection.variants:
                if v == "v0":
                    continue
                if v in aug_map:
                    sources[v] = (aug_h5, aug_map[v])
        # Process v0 first so phase 2/3 can use its features for drift /
        # cohort-ratio anchoring.
        ordered_variants = ["v0"] + [v for v in self.cfg.patient_selection.variants if v != "v0"]

        device = artefacts.device
        shard_tag = f"shard{self.cfg.shard_id}" if self.cfg.shard_id is not None else "single"

        for variant in ordered_variants:
            if variant not in sources:
                continue
            _t_cell = time.perf_counter()
            src_h5, row = sources[variant]
            z_t1c = _read_clean_latent(src_h5, "t1c", row).to(device)
            z_t1pre = _read_clean_latent(src_h5, "t1pre", row).to(device)
            m_wt, m_brain = _read_clean_masks(src_h5, row)
            m_wt = m_wt.to(device)
            m_brain = m_brain.to(device)

            with artefacts.feature_extractor_factory() as extract:
                # --- Phase 1: magnitude + per-channel stats + outlier_k.
                mag = per_patient_block_magnitude(extract, z_t1c, blocks=blocks)
                pc = per_channel_feature_stats(extract, z_t1c, blocks=blocks)
                for blk in blocks:
                    rows_per_block_magnitude.append(
                        {
                            "cohort": cohort.name,
                            "patient_id": pick.patient_id,
                            "variant": variant,
                            "block_idx": int(blk),
                            "mean_norm": mag[blk]["mean_norm"],
                            "std_norm": mag[blk]["std_norm"],
                            "p99_norm": mag[blk]["p99_norm"],
                        }
                    )
                    if self.cfg.emit.per_channel_distribution:
                        for c_idx in range(pc[blk]["mean_abs"].shape[0]):
                            rows_per_channel.append(
                                {
                                    "cohort": cohort.name,
                                    "patient_id": pick.patient_id,
                                    "variant": variant,
                                    "block_idx": int(blk),
                                    "channel_idx": int(c_idx),
                                    "mean_L_dec": float(pc[blk]["mean_abs"][c_idx]),
                                    "p99_L_dec": float(pc[blk]["p99_abs"][c_idx]),
                                    "mad": float(pc[blk]["mad"][c_idx]),
                                }
                            )
                    # Pre-aggregate the per-block recommended k.
                    rows_outlier_threshold.append(
                        {
                            "cohort": cohort.name,
                            "patient_id": pick.patient_id,
                            "variant": variant,
                            "block_idx": int(blk),
                            "mad_median": float(np.median(pc[blk]["mad"])),
                            "recommended_k": 5.0,
                        }
                    )

                # --- Phase 2: pre/post separation + error concentration + t-sweep.
                if self.cfg.phase2.enabled:
                    sep = pre_post_separation(
                        extract,
                        z_t1c,
                        z_t1pre,
                        blocks=blocks,
                        m_wt_lat=m_wt,
                        m_brain_lat=m_brain,
                    )
                    for blk in blocks:
                        for region, dist in sep[blk].items():
                            rows_pre_post.append(
                                {
                                    "cohort": cohort.name,
                                    "patient_id": pick.patient_id,
                                    "variant": variant,
                                    "block_idx": int(blk),
                                    "region": region,
                                    "sep_dist": float(dist),
                                }
                            )
                        # Phase-3 distance_lookup uses pre/post WT/notWT.
                        distance_lookup[(cohort.name, pick.patient_id, variant)][int(blk)] = {
                            "WT": float(sep[blk]["WT"]),
                            "notWT": float(sep[blk]["notWT"]),
                        }

                    conditioning = artefacts.build_conditioning(src_h5, row)
                    conditioning = {
                        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                        for k, v in conditioning.items()
                    }
                    x1_hat_s1 = artefacts.sample_x1_s1(z_t1c, conditioning)
                    err = error_concentration(
                        extract,
                        z_t1c,
                        x1_hat_s1,
                        blocks=blocks,
                        m_wt_lat=m_wt,
                        m_brain_lat=m_brain,
                    )
                    for blk in blocks:
                        for region, dist in err[blk].items():
                            rows_error_conc.append(
                                {
                                    "cohort": cohort.name,
                                    "patient_id": pick.patient_id,
                                    "variant": variant,
                                    "block_idx": int(blk),
                                    "region": region,
                                    "residual_dist": float(dist),
                                }
                            )

                    rel = x1_reliability_vs_t(
                        extract,
                        artefacts.velocity_call,
                        z_t1c,
                        conditioning,
                        t_sweep=tuple(self.cfg.phase2.t_sweep),
                        blocks=blocks,
                        m_brain_lat=m_brain,
                    )
                    for t_val, per_block in rel.items():
                        for blk, dist in per_block.items():
                            rows_x1_t.append(
                                {
                                    "cohort": cohort.name,
                                    "patient_id": pick.patient_id,
                                    "variant": variant,
                                    "t": float(t_val),
                                    "block_idx": int(blk),
                                    "feature_distance_to_target": float(dist),
                                }
                            )
            elapsed = time.perf_counter() - _t_cell
            logger.info(
                "[%s] CELL_DONE cohort=%s patient=%s variant=%s elapsed=%.2fs",
                shard_tag,
                cohort.name,
                pick.patient_id,
                variant,
                elapsed,
            )


__all__ = [
    "DecoderLplProfileConfig",
    "DecoderLplProfileEngine",
    "ModelArtefacts",
]
