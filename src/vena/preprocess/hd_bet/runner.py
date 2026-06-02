"""HD-BET v2 batch skull-strip runner for BraTS-style cohorts.

Pipeline per patient
--------------------

1. Run ``hd-bet`` (subprocess; isolated conda env) on the reference modality
   (default: ``t1n``) with ``--save_bet_mask`` to obtain both the stripped
   reference NIfTI and a binary brain mask in the source space.
2. Load the mask, apply it (voxelwise multiplication) to the remaining
   modalities listed in :attr:`HDBETSkullStripConfig.also_apply_to`, and
   write each stripped output to the mirrored patient directory.
3. Copy the tumour segmentation through unchanged (the BraTS labels are
   already constrained to brain tissue by design).
4. Persist the brain mask itself in the destination tree as
   ``{pid}-brain_mask.nii.gz`` for downstream auditing.

The runner consumes BraTS-style trees::

    <source_root>/<pid>/<pid>-{t1n,t1c,t2w,t2f,seg}.nii.gz

and writes the same layout to ``dest_root``.

All work happens in worker subprocesses. The reference HD-BET run uses one
GPU at a time (HD-BET internally loads a model into VRAM); the cheap
mask-apply + seg-copy step runs serially after each HD-BET completes per
patient so a single ``n_jobs`` setting bounds GPU contention.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from nibabel.filebasedimages import ImageFileError
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class HDBETError(RuntimeError):
    """Raised on any HD-BET subprocess or IO failure."""


_DEFAULT_PATIENT_DIR_REGEX = r"^BraTS-(SSA|GLI|PED|MET|MEN)-\d+-\d+$"
_PATIENT_DIR_RE = re.compile(_DEFAULT_PATIENT_DIR_REGEX)

# Modality slug → BraTS file-suffix component.
_DEFAULT_SUFFIX_MAP: dict[str, str] = {
    "t1pre": "t1n",
    "t1c": "t1c",
    "t2": "t2w",
    "flair": "t2f",
}
_SEG_SUFFIX = "seg"
_DEFAULT_MODALITY_FILENAME_TEMPLATE = "{pid}-{suffix}.nii.gz"
_DEFAULT_SEG_FILENAME_TEMPLATE = "{pid}-{seg_suffix}.nii.gz"
_DEFAULT_BRAIN_MASK_FILENAME_TEMPLATE = "{pid}-brain_mask.nii.gz"


class HDBETSkullStripConfig(BaseModel):
    """Resolved configuration for one HD-BET batch run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_root: Path = Field(
        ...,
        description="Directory containing BraTS-style per-patient subdirectories.",
    )
    dest_root: Path = Field(
        ...,
        description="Output root; one mirror subdirectory per patient.",
    )

    hdbet_python: Path = Field(
        default=Path("/home/mpascual/.conda/envs/hdbet/bin/python"),
        description="Path to a python interpreter that exposes the `hd-bet` CLI.",
    )
    hdbet_cli: Path = Field(
        default=Path("/home/mpascual/.conda/envs/hdbet/bin/hd-bet"),
        description="Path to the hd-bet executable script.",
    )
    device: str = Field(
        default="cuda:0",
        description="Device for HD-BET. 'cuda', 'cuda:N', 'cpu', or 'mps'.",
    )
    disable_tta: bool = Field(
        default=False,
        description="Disable test-time augmentation (~2x faster, small quality drop).",
    )

    reference_modality: str = Field(
        default="t1pre",
        description="Modality slug to feed HD-BET as the brain reference.",
    )
    also_apply_to: tuple[str, ...] = Field(
        default=("t1c", "t2", "flair"),
        description="Modality slugs to which the derived brain mask is applied.",
    )
    suffix_map: dict[str, str] = Field(
        default_factory=lambda: dict(_DEFAULT_SUFFIX_MAP),
        description="H5 modality slug → BraTS NIfTI file suffix.",
    )
    patient_dir_regex: str = Field(
        default=_DEFAULT_PATIENT_DIR_REGEX,
        description=(
            "Regex matching per-patient directory names under source_root. "
            "Default matches BraTS-{SSA,GLI,PED,MET,MEN}-NNNNN-NNN; override "
            "for cohorts with different ID conventions (e.g. REMBRANDT)."
        ),
    )
    modality_filename_template: str = Field(
        default=_DEFAULT_MODALITY_FILENAME_TEMPLATE,
        description=(
            "Format string for modality NIfTI filenames; {pid} and {suffix} "
            "are substituted. Default '{pid}-{suffix}.nii.gz' matches BraTS; "
            "REMBRANDT uses '{pid}_{suffix}_LPS_rSRI.nii.gz'."
        ),
    )
    seg_filename_template: str = Field(
        default=_DEFAULT_SEG_FILENAME_TEMPLATE,
        description=(
            "Format string for the tumour-seg filename; {pid} and {seg_suffix} "
            "are substituted. Default '{pid}-{seg_suffix}.nii.gz'."
        ),
    )
    seg_suffix: str = Field(
        default=_SEG_SUFFIX,
        description=(
            "Token substituted into seg_filename_template's {seg_suffix} slot "
            "(e.g. 'seg' for BraTS, 'GlistrBoost_out' for REMBRANDT)."
        ),
    )
    brain_mask_filename_template: str = Field(
        default=_DEFAULT_BRAIN_MASK_FILENAME_TEMPLATE,
        description=(
            "Format string for the persisted brain-mask filename; {pid} "
            "is substituted. Default '{pid}-brain_mask.nii.gz'."
        ),
    )

    n_jobs: int = Field(
        default=1,
        ge=1,
        description=(
            "Number of HD-BET subprocesses to run in parallel. Each holds the "
            "HD-BET model in VRAM, so >1 only makes sense if `device` rotates "
            "or if you accept GPU-share contention."
        ),
    )
    overwrite: bool = Field(default=False)
    limit: int | None = Field(
        default=None,
        description="If set, only process the first ``limit`` patients (smoke).",
    )
    log_level: str = "INFO"

    def to_json(self) -> str:
        return self.model_dump_json()


# ----------------------------------------------------------------------------
# Worker-side helpers
# ----------------------------------------------------------------------------


def _patient_modality_path(
    patient_root: Path,
    pid: str,
    suffix: str,
    template: str = _DEFAULT_MODALITY_FILENAME_TEMPLATE,
) -> Path:
    return patient_root / template.format(pid=pid, suffix=suffix)


def _patient_seg_path(
    patient_root: Path,
    pid: str,
    seg_suffix: str,
    template: str = _DEFAULT_SEG_FILENAME_TEMPLATE,
) -> Path:
    return patient_root / template.format(pid=pid, seg_suffix=seg_suffix)


def _load_volume(path: Path) -> tuple[NDArray[Any], Any, Any]:
    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj)
    return arr, img.affine, img.header


def _save_volume(
    path: Path,
    arr: NDArray[Any],
    affine: Any,
    header: Any,
    *,
    dtype: np.dtype[Any] | None = None,
) -> None:
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    img = nib.Nifti1Image(arr, affine=affine, header=header)
    img.set_data_dtype(arr.dtype)
    nib.save(img, str(path))


def _hdbet_outputs(out_ref: Path) -> tuple[Path, Path]:
    """HD-BET v2 writes ``<stem>.nii.gz`` (stripped image) and
    ``<stem>_bet.nii.gz`` (mask) when ``--save_bet_mask`` is passed.
    """
    stem = out_ref.name
    if stem.endswith(".nii.gz"):
        base = stem[: -len(".nii.gz")]
    else:
        base = out_ref.stem
    return out_ref, out_ref.with_name(f"{base}_bet.nii.gz")


def _process_one(
    cfg: HDBETSkullStripConfig,
    pid: str,
    src_root: Path,
) -> dict[str, Any]:
    """Strip one patient. Runs HD-BET on the reference modality, then applies
    the derived mask to the remaining modalities and copies the segmentation.
    """
    ref_suffix = cfg.suffix_map[cfg.reference_modality]
    src_ref = _patient_modality_path(src_root, pid, ref_suffix, cfg.modality_filename_template)
    if not src_ref.exists():
        raise HDBETError(f"{pid}: missing reference modality {src_ref.name}")

    dst_dir = cfg.dest_root / pid
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_ref = _patient_modality_path(dst_dir, pid, ref_suffix, cfg.modality_filename_template)

    if dst_ref.exists() and not cfg.overwrite:
        logger.info("skip %s (output present)", pid)
        return {"patient_id": pid, "status": "skipped", "reason": "output present"}

    cmd = [
        str(cfg.hdbet_cli),
        "-i",
        str(src_ref),
        "-o",
        str(dst_ref),
        "-device",
        cfg.device,
        "--save_bet_mask",
    ]
    if cfg.disable_tta:
        cmd.append("--disable_tta")

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=900,
        )
    except subprocess.TimeoutExpired as exc:
        raise HDBETError(f"{pid}: HD-BET timed out after 900s") from exc

    if proc.returncode != 0:
        raise HDBETError(
            f"{pid}: hd-bet exited {proc.returncode}\n"
            f"stdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
        )

    stripped_ref_path, mask_path = _hdbet_outputs(dst_ref)
    if not stripped_ref_path.exists() or not mask_path.exists():
        raise HDBETError(
            f"{pid}: HD-BET reported success but outputs missing "
            f"({stripped_ref_path.exists()}, {mask_path.exists()})"
        )

    # Persist the brain mask under a canonical name.
    canonical_mask = dst_dir / cfg.brain_mask_filename_template.format(pid=pid)
    if canonical_mask.exists():
        canonical_mask.unlink()
    mask_path.rename(canonical_mask)

    try:
        mask_arr, mask_affine, mask_header = _load_volume(canonical_mask)
    except (OSError, ValueError, ImageFileError, EOFError) as exc:
        raise HDBETError(f"{pid}: failed to load brain mask: {exc}") from exc
    brain = (mask_arr > 0).astype(np.uint8)

    # Apply mask to remaining modalities. IO errors on a single modality (e.g.
    # a truncated source .nii.gz from a partial download) become HDBETError
    # entries on the runner's per-patient error list rather than crashing the
    # whole batch.
    for slug in cfg.also_apply_to:
        suffix = cfg.suffix_map[slug]
        src_mod = _patient_modality_path(src_root, pid, suffix, cfg.modality_filename_template)
        if not src_mod.exists():
            raise HDBETError(f"{pid}: missing modality {src_mod.name}")
        try:
            arr, affine, header = _load_volume(src_mod)
        except (OSError, ValueError, ImageFileError, EOFError) as exc:
            raise HDBETError(
                f"{pid}: failed to load source {slug} ({src_mod.name}): {exc}"
            ) from exc
        if arr.shape != brain.shape:
            raise HDBETError(f"{pid}: shape mismatch on {slug}: {arr.shape} vs mask {brain.shape}")
        stripped = np.asarray(arr) * brain
        dst_mod = _patient_modality_path(dst_dir, pid, suffix, cfg.modality_filename_template)
        try:
            _save_volume(dst_mod, stripped, affine, header, dtype=np.float32)
        except OSError as exc:
            raise HDBETError(f"{pid}: failed to write {dst_mod.name}: {exc}") from exc

    # Carry tumour seg through (segmentation labels already lie inside brain).
    src_seg = _patient_seg_path(src_root, pid, cfg.seg_suffix, cfg.seg_filename_template)
    if src_seg.exists():
        dst_seg = _patient_seg_path(dst_dir, pid, cfg.seg_suffix, cfg.seg_filename_template)
        shutil.copyfile(src_seg, dst_seg)

    # The HD-BET-stripped reference modality is already saved at dst_ref; ensure
    # its dtype is float32 for downstream consistency.
    ref_arr, ref_affine, ref_header = _load_volume(stripped_ref_path)
    _save_volume(stripped_ref_path, ref_arr, ref_affine, ref_header, dtype=np.float32)

    elapsed = time.monotonic() - t0
    return {
        "patient_id": pid,
        "status": "ok",
        "elapsed_s": round(elapsed, 2),
        "brain_vox": int(brain.sum()),
    }


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------


class HDBETSkullStripRunner:
    """Discover BraTS-style patients and run HD-BET on each."""

    def __init__(self, cfg: HDBETSkullStripConfig) -> None:
        self.cfg = cfg

    def discover(self) -> list[tuple[str, Path]]:
        """Return ``[(patient_id, patient_root), ...]`` sorted by id."""
        regex = re.compile(self.cfg.patient_dir_regex)
        out: list[tuple[str, Path]] = []
        for d in sorted(self.cfg.source_root.iterdir()):
            if not d.is_dir():
                continue
            if regex.match(d.name) is None:
                continue
            out.append((d.name, d))
        return out

    def run(self) -> Path:
        cfg = self.cfg
        if not cfg.source_root.is_dir():
            raise HDBETError(f"source_root does not exist: {cfg.source_root}")
        if not Path(cfg.hdbet_cli).exists():
            raise HDBETError(f"hd-bet CLI not found at {cfg.hdbet_cli}")
        cfg.dest_root.mkdir(parents=True, exist_ok=True)

        patients = self.discover()
        if cfg.limit is not None:
            patients = patients[: cfg.limit]
        if not patients:
            raise HDBETError(f"No BraTS-style patients under {cfg.source_root}")
        n = len(patients)
        logger.info("HD-BET skull-strip: %d patient(s) → %s", n, cfg.dest_root)
        logger.info("device=%s, n_jobs=%d, disable_tta=%s", cfg.device, cfg.n_jobs, cfg.disable_tta)

        results: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        t0 = time.monotonic()
        log_every = max(1, n // 50)

        if cfg.n_jobs == 1:
            for i, (pid, root) in enumerate(patients, start=1):
                self._process_with_log(pid, root, i, n, t0, log_every, results, errors)
        else:
            with ThreadPoolExecutor(max_workers=cfg.n_jobs) as pool:
                futures = {pool.submit(_process_one, cfg, pid, root): pid for pid, root in patients}
                for i, fut in enumerate(as_completed(futures), start=1):
                    pid = futures[fut]
                    try:
                        results.append(fut.result())
                    except HDBETError as exc:
                        errors.append({"patient_id": pid, "error": str(exc)})
                        logger.warning("skip %s: %s", pid, exc)
                    self._maybe_log_progress(i, n, t0, log_every, len(errors))

        report = {
            "config_json": cfg.to_json(),
            "n_total": n,
            "n_ok": len(results),
            "n_errors": len(errors),
            "errors": errors,
            "results": results,
        }
        report_path = cfg.dest_root / "hd_bet_report.json"
        with report_path.open("w") as f:
            json.dump(report, f, indent=2)
        logger.info("Wrote HD-BET report: %s (%d/%d ok)", report_path, len(results), n)
        return cfg.dest_root

    def _process_with_log(
        self,
        pid: str,
        root: Path,
        i: int,
        n: int,
        t0: float,
        log_every: int,
        results: list[dict[str, Any]],
        errors: list[dict[str, str]],
    ) -> None:
        try:
            results.append(_process_one(self.cfg, pid, root))
        except HDBETError as exc:
            errors.append({"patient_id": pid, "error": str(exc)})
            logger.warning("skip %s: %s", pid, exc)
        self._maybe_log_progress(i, n, t0, log_every, len(errors))

    @staticmethod
    def _maybe_log_progress(i: int, n: int, t0: float, log_every: int, n_errors: int) -> None:
        if i % log_every == 0 or i == n:
            elapsed = time.monotonic() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            eta = (n - i) / rate if rate > 0 else float("inf")
            logger.info(
                "progress %d/%d (%.1f%%) rate=%.2f patients/s eta=%.0fs errors=%d",
                i,
                n,
                100.0 * i / n,
                rate,
                eta,
                n_errors,
            )
            sys.stdout.flush()


def discover_brats_patients(source_root: Path) -> Iterable[tuple[str, Path]]:
    """Public helper to enumerate BraTS-style patient dirs."""
    for d in sorted(Path(source_root).iterdir()):
        if d.is_dir() and _PATIENT_DIR_RE.match(d.name):
            yield d.name, d
