"""V3 normalisation audit engine.

Sweeps the registered normalisation variants across N=30 UCSF-PDGM
patients (main) and N=5 per smoke cohort (winner-only verification),
computing per-variant VAE round-trip / image-space / latent-space /
distribution metrics. Emits a ``decision.json`` consumed by the v3
encode-rollout decision.

The engine deliberately uses ``MaisiEncoder.encode(normalise=False)``
after externally applying the variant's normalisation so the variant is
the only knob being swept. The frozen MAISI-V2 VAE is loaded once and
re-used across variants and patients.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from vena.common import (
    SPATIAL_COMPRESSION,
    AutoencoderHandle,
    CropPadSpec,
    MaisiDecoder,
    MaisiEncoder,
    apply_crop_pad,
    load_autoencoder,
)
from vena.model.autoencoder.maisi.encode.masks import (
    AbstractMaskDownsampler,
    get_downsampler,
)

from .config import NormalizationAuditConfig, SmokeCohortSpec
from .decision import (
    DECISION_PRODUCER,
    DECISION_SCHEMA_VERSION,
    NormalizationAuditDecisionV1,
    PerVariantMetrics,
    SmokeCohortVerdict,
    write_decision_json,
)
from .distribution import HIST_BINS, HIST_RANGE, histogram_normalised, kl_divergence
from .figures import (
    render_intensity_histograms,
    render_kl_divergence_bar,
    render_per_region_psnr_bar,
    render_recon_grid,
    render_signal_ratio_scatter,
)
from .metrics import (
    RegionMasks,
    build_region_masks,
    compute_per_region_round_trip,
    whole_volume_ssim,
)
from .signal import (
    image_space_contrast,
    latent_space_contrast,
    signal_ratios,
)
from .variants import NormalizationVariant, get_variant_registry

logger = logging.getLogger(__name__)

LATENT_SPATIAL: tuple[int, int, int] = (48, 56, 48)


@dataclass(frozen=True)
class _PatientRecord:
    """Per-patient inputs assembled once and reused across all variants."""

    patient_id: str
    spec: CropPadSpec
    images_box: dict[str, torch.Tensor]  # (1, 1, *box) float32 RAW intensities, box space
    brain_box: torch.Tensor  # (1, 1, *box) float32 brain mask in box space
    tumor_box: torch.Tensor  # (1, 1, *box) int64 tumor labels in box space
    regions_box: RegionMasks
    regions_latent: RegionMasks  # downsampled to LATENT_SPATIAL
    n_et_voxels_image: int


def _sha256_file(path: Path) -> str:
    """SHA-256 of a file (used to stamp the VAE checkpoint)."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha(repo_root: Path) -> str | None:
    """Best-effort git SHA at the repo root; returns ``None`` if unavailable."""
    git_dir = repo_root / ".git"
    if not git_dir.exists():
        return None
    head = (git_dir / "HEAD").read_text().strip()
    if head.startswith("ref:"):
        ref = head.split(" ", 1)[1].strip()
        ref_path = git_dir / ref
        if ref_path.exists():
            return ref_path.read_text().strip()
    return head


@contextmanager
def _open_image_h5(path: Path) -> Any:
    """Lightweight context manager for an image H5 in read-only mode."""
    f = h5py.File(path, "r")
    try:
        yield f
    finally:
        f.close()


def _box_target_shape(h5: Any) -> tuple[int, int, int]:
    """Extract target box shape from image-H5 ``crop_box`` root attr.

    The attribute is a JSON-encoded ``[H, W, D]`` list (legacy schema) or a
    ``{"target_shape": [H, W, D]}`` dict (planned schema). Both forms are
    accepted.
    """
    raw = h5.attrs.get("crop_box", None)
    if raw is None:
        raise RuntimeError("image H5 lacks `crop_box` root attribute")
    if isinstance(raw, (bytes, np.bytes_)):
        raw = raw.decode()
    elif not isinstance(raw, str):
        raw = str(raw)
    blob = json.loads(raw)
    if isinstance(blob, list):
        seq = blob
    elif isinstance(blob, dict) and "target_shape" in blob:
        seq = blob["target_shape"]
    else:
        raise RuntimeError(f"image H5 `crop_box` attr has unexpected shape: {blob!r}")
    return tuple(int(x) for x in seq)  # type: ignore[return-value]


def _native_shape(h5: Any, modality: str) -> tuple[int, int, int]:
    return tuple(int(x) for x in h5[f"images/{modality}"].shape[1:])  # type: ignore[return-value]


def _read_ids(h5: Any) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in h5["ids"][:]]


def _sample_patient_ids(
    all_ids: list[str], n: int, seed: int, explicit: list[str] | None
) -> list[str]:
    if explicit:
        missing = [pid for pid in explicit if pid not in all_ids]
        if missing:
            raise ValueError(f"patient_ids not found in H5: {missing[:5]}")
        return list(explicit)
    if n >= len(all_ids):
        return list(all_ids)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(all_ids), size=n, replace=False)
    return [all_ids[i] for i in sorted(indices.tolist())]


def _load_patient(
    h5: Any,
    pid: str,
    modalities: list[str],
    box_shape: tuple[int, int, int],
    device: torch.device,
    mask_downsampler: AbstractMaskDownsampler,
) -> _PatientRecord:
    """Read + crop one patient's modalities + masks; build region masks."""
    ids = _read_ids(h5)
    try:
        row = ids.index(pid)
    except ValueError as exc:
        raise KeyError(f"patient {pid} not found in image H5") from exc
    nshape = _native_shape(h5, modalities[0])
    crop_origin = tuple(int(v) for v in h5["crop/origin"][row])
    spec = CropPadSpec(
        crop_origin=crop_origin,  # type: ignore[arg-type]
        native_shape=nshape,
        target_shape=box_shape,
    )
    # Raw modalities.
    imgs_box: dict[str, torch.Tensor] = {}
    for m in modalities:
        arr = np.asarray(h5[f"images/{m}"][row], dtype=np.float32)
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device, non_blocking=True)
        imgs_box[m] = apply_crop_pad(t, spec)
    # Brain mask (box space).
    brain_arr = np.asarray(h5["masks/brain"][row], dtype=np.float32)
    brain_t = torch.from_numpy(brain_arr).unsqueeze(0).unsqueeze(0).to(device, non_blocking=True)
    brain_box = apply_crop_pad(brain_t, spec)
    # Tumor labels (box space). Handle BraTS-2023 3→4 remap.
    label_system = str(h5.attrs.get("label_system", "")) or ""
    seg_arr = np.asarray(h5["masks/tumor"][row], dtype=np.int64)
    if label_system == "BraTS2023":
        seg_arr = np.where(seg_arr == 3, 4, seg_arr)
    seg_t = torch.from_numpy(seg_arr).unsqueeze(0).unsqueeze(0).to(device)
    seg_box = apply_crop_pad(seg_t.float(), spec).to(torch.int64)
    # Region masks at box resolution.
    regions_box = build_region_masks(seg_box, brain_box.bool())
    # Latent region masks via per-class avg-pool + threshold at 0.5.
    onehot_latent = mask_downsampler.downsample(seg_box, target_shape=LATENT_SPATIAL)
    # onehot_latent: (1, 3, h, w, d) float; channels = NETC, ED, ET (label codes 1,2,4)
    netc_l = onehot_latent[:, 0:1] >= 0.5
    ed_l = onehot_latent[:, 1:2] >= 0.5
    et_l = onehot_latent[:, 2:3] >= 0.5
    wt_l = netc_l | ed_l | et_l
    # Brain in latent space: avg-pool the brain mask too.
    brain_latent_pooled = torch.nn.functional.avg_pool3d(
        brain_box, kernel_size=SPATIAL_COMPRESSION, stride=SPATIAL_COMPRESSION
    )
    brain_l = brain_latent_pooled >= 0.5
    bnwt_l = brain_l & ~wt_l
    regions_latent = RegionMasks(brain=brain_l, netc=netc_l, ed=ed_l, et=et_l, wt=wt_l, bnwt=bnwt_l)

    n_et_voxels_image = int(regions_box.et.sum().item())

    return _PatientRecord(
        patient_id=pid,
        spec=spec,
        images_box=imgs_box,
        brain_box=brain_box,
        tumor_box=seg_box,
        regions_box=regions_box,
        regions_latent=regions_latent,
        n_et_voxels_image=n_et_voxels_image,
    )


@dataclass
class _VariantAccumulator:
    """Streaming aggregator for one variant across patients (main sweep)."""

    rows: list[dict[str, float]]
    hist_per_modality: dict[str, list[np.ndarray]]
    seen_et_patients: int = 0
    seen_et_patients_large: int = 0


def _decide_winner(
    metrics_per_variant: dict[str, PerVariantMetrics],
) -> tuple[str | None, str]:
    """Pick the winner per the C1..C7 acceptance gate.

    Tie-break order (per spec §3): lowest KL → highest C4 ratio → cheapest
    implementation (alphabetical id for tie-stable choice).
    """
    passing = [vid for vid, m in metrics_per_variant.items() if m.passes_all and vid != "V0"]
    if not passing:
        # Fallback path. Diagnose why each non-V0 variant failed.
        reasons: list[str] = []
        for vid, m in metrics_per_variant.items():
            if vid == "V0":
                continue
            failed_criteria = [
                name
                for name, ok in [
                    ("C1", m.passes_c1_mae_whole),
                    ("C2", m.passes_c2_mae_et),
                    ("C3", m.passes_c3_kl),
                    ("C4", m.passes_c4_image_signal),
                    ("C5", m.passes_c5_latent_signal),
                    ("C7", m.passes_c7_psnr_whole),
                ]
                if not ok
            ]
            reasons.append(f"{vid}: fails {','.join(failed_criteria) or 'unknown'}")
        rationale = "No variant passed C1-C7; V0 retained as fallback. " + " | ".join(reasons)
        return None, rationale

    # Tie-break.
    def _sort_key(vid: str) -> tuple[float, float, str]:
        m = metrics_per_variant[vid]
        # Lower KL is better → use kl_max directly.
        # Higher C4 is better → use -ratio.
        return (m.kl_divergence_max, -m.image_signal_ratio_et_over_bnwt, vid)

    ordered = sorted(passing, key=_sort_key)
    winner = ordered[0]
    rationale = (
        f"Winner: {winner}. Passed C1-C7. Tie-break: lowest KL_max="
        f"{metrics_per_variant[winner].kl_divergence_max:.3f} nats, "
        f"image C4 ratio={metrics_per_variant[winner].image_signal_ratio_et_over_bnwt:.3f}."
    )
    return winner, rationale


class NormalizationAuditEngine:
    """Orchestrate the per-variant audit and emit ``decision.json`` + figures."""

    def __init__(self, cfg: NormalizationAuditConfig) -> None:
        self.cfg = cfg
        self._device = torch.device(cfg.device)

    # ------------------------------------------------------------------ entry
    def run(self) -> Path:
        torch.set_float32_matmul_precision("high")
        cfg = self.cfg

        # Resolve artifact dir (timestamped).
        timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
        out_root = Path(cfg.out_dir)
        out_dir = out_root / timestamp
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "figures").mkdir(exist_ok=True)
        (out_dir / "tables").mkdir(exist_ok=True)
        (out_dir / "logs").mkdir(exist_ok=True)

        # Attach a file log handler.
        log_path = out_dir / "logs" / "audit.log"
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logging.getLogger().addHandler(fh)
        logger.info("Artifact dir: %s", out_dir)

        # Persist resolved YAML + git sha.
        (out_dir / "config.json").write_text(
            json.dumps(cfg.model_dump(mode="json"), indent=2, default=str)
        )
        repo_root = Path(__file__).resolve().parents[4]
        git_sha = _git_sha(repo_root)
        if git_sha:
            (out_dir / "git_sha.txt").write_text(git_sha + "\n")

        # Load VAE.
        logger.info("Loading MAISI VAE: %s", cfg.vae_checkpoint)
        handle: AutoencoderHandle = load_autoencoder(
            cfg.vae_checkpoint,
            device=self._device,
            arch_config=cfg.vae_arch_config,
        )
        # Build encoder + decoder. Variants apply normalisation externally;
        # we set the encoder's percentile params to V0 defaults so the
        # ``normalise=False`` path stays well-formed even if a future code
        # path forgets to pass ``normalise=False``.
        encoder = MaisiEncoder(
            handle=handle,
            percentile_lower=0.0,
            percentile_upper=99.5,
            percentile_foreground_only=True,
        )
        decoder = MaisiDecoder(handle=handle)
        mask_downsampler = get_downsampler(
            "per_class_avg_pool", spatial_compression=SPATIAL_COMPRESSION
        )

        vae_sha = handle.checkpoint_sha256

        # Run main sweep on the production cohort.
        main_metrics, hists_per_variant = self._run_main_sweep(
            out_dir=out_dir,
            encoder=encoder,
            decoder=decoder,
            mask_downsampler=mask_downsampler,
        )

        # Render figures.
        self._render_figures(out_dir=out_dir, metrics=main_metrics, hists=hists_per_variant)

        # Decide winner.
        winner, rationale = _decide_winner(main_metrics)
        fallback_used = winner is None

        # Smoke verification on the winner.
        smoke_verdicts: list[SmokeCohortVerdict] = []
        if winner is not None and cfg.smoke_cohorts:
            smoke_verdicts = self._run_smoke_sweep(
                out_dir=out_dir,
                encoder=encoder,
                decoder=decoder,
                mask_downsampler=mask_downsampler,
                winner=winner,
            )
            # Promote fallback if any smoke cohort fails.
            if not all(v.passes for v in smoke_verdicts):
                fallback_used = True
                rationale += (
                    f" Smoke verification failed on "
                    f"{[v.cohort for v in smoke_verdicts if not v.passes]}; "
                    f"falling back to V0."
                )
                winner = None

        decision = NormalizationAuditDecisionV1(
            produced_at=datetime.now(UTC).isoformat(),
            producer=DECISION_PRODUCER,
            schema_version=DECISION_SCHEMA_VERSION,  # type: ignore[arg-type]
            git_sha=git_sha,
            vae_checkpoint=str(cfg.vae_checkpoint),
            vae_checkpoint_sha256=vae_sha,
            main_cohort=cfg.main_cohort_name,
            n_patients_main=main_metrics[next(iter(main_metrics))].n_patients
            if main_metrics
            else 0,
            patient_seed=cfg.patient_seed,
            variants_tested=list(main_metrics.keys()),
            metrics_per_variant=main_metrics,
            acceptance_thresholds={
                "c1_mae_whole_max": cfg.c1_mae_whole_max,
                "c2_mae_et_max": cfg.c2_mae_et_max,
                "c3_kl_max_nats": cfg.c3_kl_max_nats,
                "c4_image_signal_ratio_min": cfg.c4_image_signal_ratio_min,
                "c5_latent_signal_ratio_min": cfg.c5_latent_signal_ratio_min,
                "c7_psnr_whole_min_db": cfg.c7_psnr_whole_min_db,
            },
            winner=winner,
            winner_rationale=rationale,
            fallback_used=fallback_used,
            smoke_cohorts=smoke_verdicts,
            next_action=(
                "re_encode_all_cohorts"
                if (winner is not None and not fallback_used)
                else ("fall_back_to_v0" if fallback_used else "manual_review")
            ),
        )
        write_decision_json(out_dir / "decision.json", decision)

        self._write_report(out_dir=out_dir, decision=decision)

        # LATEST symlink.
        latest = out_root / "LATEST"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(out_dir.name)
        logger.info("LATEST -> %s", out_dir.name)

        logging.getLogger().removeHandler(fh)
        fh.close()
        return out_dir

    # ------------------------------------------------------------------ main
    def _run_main_sweep(
        self,
        *,
        out_dir: Path,
        encoder: MaisiEncoder,
        decoder: MaisiDecoder,
        mask_downsampler: AbstractMaskDownsampler,
    ) -> tuple[dict[str, PerVariantMetrics], dict[str, dict[str, np.ndarray]]]:
        cfg = self.cfg
        registry = get_variant_registry()
        missing = [v for v in cfg.variants_to_test if v not in registry]
        if missing:
            raise ValueError(f"variants_to_test contains unregistered ids: {missing}")

        with _open_image_h5(cfg.image_h5_main) as f:
            all_ids = _read_ids(f)
            box_shape = _box_target_shape(f)
            chosen = _sample_patient_ids(
                all_ids, cfg.n_patients_main, cfg.patient_seed, cfg.patient_ids_main
            )
            logger.info(
                "Main sweep: cohort=%s, n=%d (of %d), variants=%s",
                cfg.main_cohort_name,
                len(chosen),
                len(all_ids),
                cfg.variants_to_test,
            )

            # Pre-load every patient once. ~ B * (4 mods + brain + tumor) tensors live on GPU.
            patient_records: list[_PatientRecord] = []
            for pid in chosen:
                rec = _load_patient(
                    h5=f,
                    pid=pid,
                    modalities=cfg.modalities,
                    box_shape=box_shape,
                    device=self._device,
                    mask_downsampler=mask_downsampler,
                )
                patient_records.append(rec)

        # Per-variant streaming aggregator.
        accumulators: dict[str, _VariantAccumulator] = {
            vid: _VariantAccumulator(rows=[], hist_per_modality={m: [] for m in cfg.modalities})
            for vid in cfg.variants_to_test
        }

        # Per-variant first-patient recon for figure rendering.
        first_recon_per_variant: dict[
            str, tuple[str, dict[str, np.ndarray], dict[str, np.ndarray]]
        ] = {}

        target = cfg.target_modality

        for vid in cfg.variants_to_test:
            variant = registry[vid]
            logger.info("=== Variant %s — %s", vid, variant.description)
            v_t0 = time.time()
            for rec in patient_records:
                p_t0 = time.time()
                masks_arg = {"brain": rec.brain_box}
                imgs_normed = variant.apply(rec.images_box, masks_arg)
                # Encode + decode each modality.
                latents: dict[str, torch.Tensor] = {}
                recons: dict[str, torch.Tensor] = {}
                for m in cfg.modalities:
                    z = encoder.encode(
                        imgs_normed[m],
                        crop_spec=None,  # already in box space; bypass crop step
                        normalise=False,
                        mode="full",
                    ).latent  # (1, C, h, w, d)
                    # If full-volume box encode disagrees with depth-pad path,
                    # MaisiEncoder.encode with crop_spec=None falls back to its
                    # legacy depth-pad path. To stay on the box path explicitly
                    # we pass a synthetic spec that maps identity.
                    latents[m] = z
                    # Build a non-padded CropPadSpec for the decode path.
                    spec_box = CropPadSpec(
                        crop_origin=(0, 0, 0),
                        native_shape=tuple(imgs_normed[m].shape[2:]),  # type: ignore[arg-type]
                        target_shape=tuple(imgs_normed[m].shape[2:]),  # type: ignore[arg-type]
                    )
                    recon = decoder.decode(z, crop_spec=spec_box, mode="full").image
                    recons[m] = recon

                # Round-trip metrics on T1c (target modality).
                rt = compute_per_region_round_trip(
                    pred=recons[target][0, 0],
                    ref=imgs_normed[target][0, 0],
                    regions=rec.regions_box,
                    data_range=1.0,
                )
                ssim_whole = whole_volume_ssim(
                    pred=recons[target][0, 0],
                    ref=imgs_normed[target][0, 0],
                    brain_mask=rec.brain_box[0, 0],
                    data_range=1.0,
                )

                # Image-space contrast.
                img_c = image_space_contrast(
                    t1c_normalised=imgs_normed["t1c"][0, 0],
                    t1pre_normalised=imgs_normed["t1pre"][0, 0],
                    regions=rec.regions_box,
                )
                img_ratio = signal_ratios(img_c, image=True)

                # Latent-space contrast.
                lat_c = latent_space_contrast(
                    z_t1c=latents["t1c"],
                    z_t1pre=latents["t1pre"],
                    regions_latent=rec.regions_latent,
                )
                lat_ratio = signal_ratios(lat_c, image=False)

                # Histogram per modality.
                for m in cfg.modalities:
                    h = histogram_normalised(
                        imgs_normed[m][0, 0],
                        mask=rec.brain_box[0, 0] > 0,
                    )
                    accumulators[vid].hist_per_modality[m].append(h)

                # Row.
                row = {
                    "variant_id": vid,
                    "patient_id": rec.patient_id,
                    "n_et_voxels_image": rec.n_et_voxels_image,
                    "ssim_whole": ssim_whole,
                    **rt,
                    **img_c,
                    **img_ratio,
                    **lat_c,
                    **lat_ratio,
                }
                accumulators[vid].rows.append(row)
                if rec.n_et_voxels_image > 0:
                    accumulators[vid].seen_et_patients += 1
                    if rec.n_et_voxels_image >= cfg.et_voxel_threshold_large:
                        accumulators[vid].seen_et_patients_large += 1

                if rec.patient_id == patient_records[0].patient_id:
                    # Capture for recon-grid figure.
                    first_recon_per_variant[vid] = (
                        rec.patient_id,
                        {m: imgs_normed[m][0, 0].detach().cpu().numpy() for m in cfg.modalities},
                        {m: recons[m][0, 0].detach().cpu().numpy() for m in cfg.modalities},
                    )

                del imgs_normed, latents, recons
                if self._device.type == "cuda":
                    torch.cuda.empty_cache()
                logger.info("  %s %s in %.1fs", vid, rec.patient_id, time.time() - p_t0)
            logger.info("=== Variant %s done in %.1fs", vid, time.time() - v_t0)

        # Persist per-(variant, patient) rows.
        for vid, acc in accumulators.items():
            self._write_per_variant_csv(out_dir / "tables" / f"per_patient_{vid}.csv", acc.rows)

        # Render recon grids for every variant on the first patient.
        for vid, (pid, reals, preds) in first_recon_per_variant.items():
            render_recon_grid(
                real_volumes=reals,
                recon_volumes=preds,
                variant_id=vid,
                patient_id=pid,
                out_path=out_dir / "figures" / f"recon_grid_{vid}.png",
            )

        # Compute per-variant aggregates + KL vs V0.
        v0_hists_per_modality: dict[str, np.ndarray] = {}
        if "V0" in accumulators:
            for m in self.cfg.modalities:
                if accumulators["V0"].hist_per_modality[m]:
                    stacked = np.stack(accumulators["V0"].hist_per_modality[m])
                    v0_hists_per_modality[m] = stacked.mean(axis=0)
                    v0_hists_per_modality[m] = (
                        v0_hists_per_modality[m] / v0_hists_per_modality[m].sum()
                    )

        per_variant_metrics: dict[str, PerVariantMetrics] = {}
        hist_per_variant_for_fig: dict[str, dict[str, np.ndarray]] = {}
        for vid, acc in accumulators.items():
            variant = registry[vid]
            metrics, hist_per_mod = self._aggregate_variant(
                variant=variant,
                rows=acc.rows,
                hist_per_modality=acc.hist_per_modality,
                v0_hists_per_modality=v0_hists_per_modality,
                cfg=cfg,
            )
            per_variant_metrics[vid] = metrics
            hist_per_variant_for_fig[vid] = hist_per_mod

        return per_variant_metrics, hist_per_variant_for_fig

    # ----------------------------------------------------------------- smoke
    def _run_smoke_sweep(
        self,
        *,
        out_dir: Path,
        encoder: MaisiEncoder,
        decoder: MaisiDecoder,
        mask_downsampler: AbstractMaskDownsampler,
        winner: str,
    ) -> list[SmokeCohortVerdict]:
        cfg = self.cfg
        registry = get_variant_registry()
        variant = registry[winner]
        verdicts: list[SmokeCohortVerdict] = []
        for spec in cfg.smoke_cohorts:
            verdict = self._run_smoke_cohort(
                spec=spec,
                variant=variant,
                encoder=encoder,
                decoder=decoder,
                mask_downsampler=mask_downsampler,
            )
            verdicts.append(verdict)
        # Persist smoke verdicts CSV.
        smoke_path = out_dir / "tables" / "smoke_cohorts.csv"
        self._write_smoke_csv(smoke_path, verdicts)
        return verdicts

    def _run_smoke_cohort(
        self,
        *,
        spec: SmokeCohortSpec,
        variant: NormalizationVariant,
        encoder: MaisiEncoder,
        decoder: MaisiDecoder,
        mask_downsampler: AbstractMaskDownsampler,
    ) -> SmokeCohortVerdict:
        cfg = self.cfg
        with _open_image_h5(spec.image_h5) as f:
            all_ids = _read_ids(f)
            box_shape = _box_target_shape(f)
            chosen = _sample_patient_ids(all_ids, spec.n_patients, cfg.patient_seed, None)
            mae_w_list: list[float] = []
            mae_et_list: list[float] = []
            ratio_list: list[float] = []
            for pid in chosen:
                rec = _load_patient(
                    h5=f,
                    pid=pid,
                    modalities=cfg.modalities,
                    box_shape=box_shape,
                    device=self._device,
                    mask_downsampler=mask_downsampler,
                )
                masks_arg = {"brain": rec.brain_box}
                imgs_normed = variant.apply(rec.images_box, masks_arg)
                z_t1c = encoder.encode(
                    imgs_normed[cfg.target_modality],
                    crop_spec=None,
                    normalise=False,
                    mode="full",
                ).latent
                spec_box = CropPadSpec(
                    crop_origin=(0, 0, 0),
                    native_shape=tuple(imgs_normed[cfg.target_modality].shape[2:]),  # type: ignore[arg-type]
                    target_shape=tuple(imgs_normed[cfg.target_modality].shape[2:]),  # type: ignore[arg-type]
                )
                rec_t1c = decoder.decode(z_t1c, crop_spec=spec_box, mode="full").image
                rt = compute_per_region_round_trip(
                    pred=rec_t1c[0, 0],
                    ref=imgs_normed[cfg.target_modality][0, 0],
                    regions=rec.regions_box,
                    data_range=1.0,
                )
                mae_w_list.append(rt["mae_whole"])
                mae_et_list.append(rt["mae_et"])
                img_c = image_space_contrast(
                    t1c_normalised=imgs_normed["t1c"][0, 0],
                    t1pre_normalised=imgs_normed["t1pre"][0, 0],
                    regions=rec.regions_box,
                )
                r = signal_ratios(img_c, image=True)["image_signal_ratio_et_over_bnwt"]
                if r == r:  # not NaN
                    ratio_list.append(r)
                del imgs_normed, z_t1c, rec_t1c
                if self._device.type == "cuda":
                    torch.cuda.empty_cache()
        mae_w = float(np.nanmean(mae_w_list)) if mae_w_list else float("nan")
        mae_et = float(np.nanmean(mae_et_list)) if mae_et_list else float("nan")
        ratio = float(np.nanmean(ratio_list)) if ratio_list else float("nan")
        passes = (
            (mae_w <= cfg.c1_mae_whole_max)
            and (mae_et <= cfg.c2_mae_et_max)
            and (ratio >= cfg.c4_image_signal_ratio_min)
        )
        return SmokeCohortVerdict(
            cohort=spec.name,
            n_patients=len(chosen),
            mae_whole=mae_w,
            mae_et=mae_et,
            image_signal_ratio_et_over_bnwt=ratio,
            passes=passes,
        )

    # ------------------------------------------------------------- aggregate
    def _aggregate_variant(
        self,
        *,
        variant: NormalizationVariant,
        rows: list[dict[str, float]],
        hist_per_modality: dict[str, list[np.ndarray]],
        v0_hists_per_modality: dict[str, np.ndarray],
        cfg: NormalizationAuditConfig,
    ) -> tuple[PerVariantMetrics, dict[str, np.ndarray]]:
        def _nanmean(key: str, where: list[dict[str, float]] | None = None) -> float:
            src = rows if where is None else where
            vals = [r[key] for r in src if key in r and r[key] == r[key]]
            return float(np.nanmean(vals)) if vals else float("nan")

        large_rows = [
            r for r in rows if r.get("n_et_voxels_image", 0) >= cfg.et_voxel_threshold_large
        ]

        # Per-modality KL vs V0.
        hist_per_mod_aggregate: dict[str, np.ndarray] = {}
        kl_per_modality: dict[str, float] = {}
        for m in self.cfg.modalities:
            if hist_per_modality[m]:
                stacked = np.stack(hist_per_modality[m])
                p = stacked.mean(axis=0)
                p = p / p.sum()
                hist_per_mod_aggregate[m] = p
                if m in v0_hists_per_modality:
                    kl_per_modality[m] = kl_divergence(p, v0_hists_per_modality[m])
                else:
                    kl_per_modality[m] = 0.0  # V0 vs itself
            else:
                kl_per_modality[m] = float("nan")
        kl_max = (
            float(np.nanmax(list(kl_per_modality.values()))) if kl_per_modality else float("nan")
        )

        mae_whole = _nanmean("mae_whole")
        mae_et = _nanmean("mae_et")
        mae_netc = _nanmean("mae_netc")
        mae_ed = _nanmean("mae_ed")
        mae_bnwt = _nanmean("mae_bnwt")
        psnr_whole = _nanmean("psnr_whole_db")
        psnr_et = _nanmean("psnr_et_db")
        ssim_whole = _nanmean("ssim_whole")
        img_diff_et = _nanmean("image_mean_abs_diff_et")
        img_diff_bnwt = _nanmean("image_mean_abs_diff_bnwt")
        img_ratio = _nanmean("image_signal_ratio_et_over_bnwt")
        lat_delta_et = _nanmean("latent_mean_abs_delta_et")
        lat_delta_bnwt = _nanmean("latent_mean_abs_delta_bnwt")
        lat_ratio = _nanmean("latent_signal_ratio_et_over_bnwt")
        lat_abs_t1c_et = _nanmean("latent_mean_abs_t1c_et")
        lat_abs_t1pre_et = _nanmean("latent_mean_abs_t1pre_et")

        img_ratio_large = (
            _nanmean("image_signal_ratio_et_over_bnwt", where=large_rows) if large_rows else None
        )
        lat_ratio_large = (
            _nanmean("latent_signal_ratio_et_over_bnwt", where=large_rows) if large_rows else None
        )

        # Acceptance bits — evaluate against the LARGE-ET stratum for C4/C5
        # when at least 3 patients qualify, else the full sample.
        c4_eval = (
            img_ratio_large if (img_ratio_large is not None and len(large_rows) >= 3) else img_ratio
        )
        c5_eval = (
            lat_ratio_large if (lat_ratio_large is not None and len(large_rows) >= 3) else lat_ratio
        )
        passes_c1 = mae_whole <= cfg.c1_mae_whole_max if mae_whole == mae_whole else False
        passes_c2 = mae_et <= cfg.c2_mae_et_max if mae_et == mae_et else False
        passes_c3 = kl_max <= cfg.c3_kl_max_nats if kl_max == kl_max else False
        passes_c4 = c4_eval >= cfg.c4_image_signal_ratio_min if c4_eval == c4_eval else False
        passes_c5 = c5_eval >= cfg.c5_latent_signal_ratio_min if c5_eval == c5_eval else False
        passes_c7 = psnr_whole >= cfg.c7_psnr_whole_min_db if psnr_whole == psnr_whole else False
        passes_all = all([passes_c1, passes_c2, passes_c3, passes_c4, passes_c5, passes_c7])

        metrics = PerVariantMetrics(
            variant_id=variant.id,
            variant_version=variant.variant_version,
            params=dict(variant.params),
            n_patients=len(rows),
            mae_whole=mae_whole,
            mae_et=mae_et,
            mae_netc=mae_netc,
            mae_ed=mae_ed,
            mae_bnwt=mae_bnwt,
            psnr_whole_db=psnr_whole,
            psnr_et_db=None if psnr_et != psnr_et else psnr_et,
            ssim_whole=ssim_whole,
            image_mean_abs_diff_et=img_diff_et,
            image_mean_abs_diff_bnwt=img_diff_bnwt,
            image_signal_ratio_et_over_bnwt=img_ratio,
            latent_mean_abs_delta_et=lat_delta_et,
            latent_mean_abs_delta_bnwt=lat_delta_bnwt,
            latent_signal_ratio_et_over_bnwt=lat_ratio,
            latent_mean_abs_t1c_et=lat_abs_t1c_et,
            latent_mean_abs_t1pre_et=lat_abs_t1pre_et,
            kl_divergence_per_modality=kl_per_modality,
            kl_divergence_max=kl_max,
            image_signal_ratio_large_et_stratum=img_ratio_large,
            latent_signal_ratio_large_et_stratum=lat_ratio_large,
            n_patients_large_et=len(large_rows),
            passes_c1_mae_whole=bool(passes_c1),
            passes_c2_mae_et=bool(passes_c2),
            passes_c3_kl=bool(passes_c3),
            passes_c4_image_signal=bool(passes_c4),
            passes_c5_latent_signal=bool(passes_c5),
            passes_c7_psnr_whole=bool(passes_c7),
            passes_all=bool(passes_all),
        )
        return metrics, hist_per_mod_aggregate

    # ------------------------------------------------------------------ io
    def _write_per_variant_csv(self, path: Path, rows: list[dict[str, float]]) -> None:
        if not rows:
            path.write_text("")
            return
        import csv

        keys: list[str] = []
        for r in rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in keys})

    def _write_smoke_csv(self, path: Path, verdicts: list[SmokeCohortVerdict]) -> None:
        import csv

        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["cohort", "n_patients", "mae_whole", "mae_et", "c4_ratio", "passes"])
            for v in verdicts:
                w.writerow(
                    [
                        v.cohort,
                        v.n_patients,
                        v.mae_whole,
                        v.mae_et,
                        v.image_signal_ratio_et_over_bnwt,
                        v.passes,
                    ]
                )

    def _render_figures(
        self,
        *,
        out_dir: Path,
        metrics: dict[str, PerVariantMetrics],
        hists: dict[str, dict[str, np.ndarray]],
    ) -> None:
        figs_dir = out_dir / "figures"
        # Per-region PSNR bars.
        psnr_dict = {
            vid: {
                "whole": m.psnr_whole_db,
                "et": (m.psnr_et_db if m.psnr_et_db is not None else float("nan")),
                "netc": float("nan"),  # only whole/et computed at aggregate level
                "ed": float("nan"),
                "bnwt": float("nan"),
            }
            for vid, m in metrics.items()
        }
        render_per_region_psnr_bar(
            psnr_per_variant=psnr_dict,
            threshold_db=self.cfg.c7_psnr_whole_min_db,
            out_path=figs_dir / "per_region_psnr_bar.png",
        )
        # Signal-ratio scatter.
        render_signal_ratio_scatter(
            image_ratios={vid: m.image_signal_ratio_et_over_bnwt for vid, m in metrics.items()},
            latent_ratios={vid: m.latent_signal_ratio_et_over_bnwt for vid, m in metrics.items()},
            c4_threshold=self.cfg.c4_image_signal_ratio_min,
            c5_threshold=self.cfg.c5_latent_signal_ratio_min,
            out_path=figs_dir / "signal_ratio_scatter.png",
        )
        # KL bar.
        render_kl_divergence_bar(
            kl_per_variant_per_modality={
                vid: dict(m.kl_divergence_per_modality) for vid, m in metrics.items()
            },
            threshold_nats=self.cfg.c3_kl_max_nats,
            out_path=figs_dir / "distribution_kl_bar.png",
        )
        # Intensity histograms (one per modality).
        bin_edges = np.linspace(HIST_RANGE[0], HIST_RANGE[1], HIST_BINS + 1)
        bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        for m in self.cfg.modalities:
            per_v: dict[str, np.ndarray] = {}
            for vid in metrics.keys():
                if m in hists.get(vid, {}):
                    per_v[vid] = hists[vid][m]
            if not per_v:
                continue
            render_intensity_histograms(
                histograms_per_variant=per_v,
                modality=m,
                bin_centres=bin_centres,
                v0_percentile_cuts=None,
                out_path=figs_dir / f"intensity_histogram_{m}.png",
            )

    def _write_report(self, *, out_dir: Path, decision: NormalizationAuditDecisionV1) -> None:
        lines: list[str] = []
        lines.append(f"# V3 normalisation audit — {decision.main_cohort}")
        lines.append("")
        lines.append(f"- produced_at: {decision.produced_at}")
        lines.append(f"- producer: {decision.producer}")
        lines.append(f"- git_sha: {decision.git_sha or 'n/a'}")
        lines.append(f"- VAE checkpoint: `{decision.vae_checkpoint}`")
        lines.append(f"- VAE SHA-256: `{decision.vae_checkpoint_sha256[:16]}…`")
        lines.append(f"- n_patients_main: {decision.n_patients_main}")
        lines.append("")
        lines.append("## Acceptance thresholds")
        lines.append("")
        for k, v in decision.acceptance_thresholds.items():
            lines.append(f"- `{k}` = {v}")
        lines.append("")
        lines.append("## Per-variant metrics (main cohort)")
        lines.append("")
        lines.append(
            "| variant | n | mae_whole | mae_et | psnr_whole_db | C4 ratio (large-ET) | C5 ratio (large-ET) | KL_max | passes_all |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for vid, m in decision.metrics_per_variant.items():
            c4 = m.image_signal_ratio_large_et_stratum
            c5 = m.latent_signal_ratio_large_et_stratum
            lines.append(
                f"| {vid} | {m.n_patients} | {m.mae_whole:.4f} | {m.mae_et:.4f} | "
                f"{m.psnr_whole_db:.2f} | "
                f"{m.image_signal_ratio_et_over_bnwt:.3f} "
                f"({(c4 if c4 is not None else float('nan')):.3f}) | "
                f"{m.latent_signal_ratio_et_over_bnwt:.3f} "
                f"({(c5 if c5 is not None else float('nan')):.3f}) | "
                f"{m.kl_divergence_max:.3f} | {'✓' if m.passes_all else '✗'} |"
            )
        lines.append("")
        lines.append("## Decision")
        lines.append("")
        if decision.winner is None:
            lines.append(f"**No winner** — fallback to V0. {decision.winner_rationale}")
        else:
            lines.append(f"**Winner: {decision.winner}** — {decision.winner_rationale}")
        lines.append("")
        if decision.smoke_cohorts:
            lines.append("## Smoke cohorts (winner)")
            lines.append("")
            lines.append("| cohort | n | mae_whole | mae_et | C4 ratio | passes |")
            lines.append("|---|---:|---:|---:|---:|---|")
            for v in decision.smoke_cohorts:
                lines.append(
                    f"| {v.cohort} | {v.n_patients} | {v.mae_whole:.4f} | "
                    f"{v.mae_et:.4f} | {v.image_signal_ratio_et_over_bnwt:.3f} | "
                    f"{'✓' if v.passes else '✗'} |"
                )
            lines.append("")
        lines.append("## Figures")
        lines.append("")
        figs = sorted((out_dir / "figures").glob("*.png"))
        for p in figs:
            lines.append(f"- `figures/{p.name}`")
        (out_dir / "report.md").write_text("\n".join(lines) + "\n")
        os.sync()


__all__ = ["NormalizationAuditEngine"]
