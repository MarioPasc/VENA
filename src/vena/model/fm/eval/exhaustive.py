"""Library helpers for the exhaustive image-space validation routine.

These are the pure, unit-testable pieces of the exhaustive validation:

* :func:`load_real_t1c_normalised` — read a patient's raw T1c from the
  image-domain H5 and apply the *exact* normalisation the MAISI encoder applied
  to its input (``percentile_normalise`` with ``lower=0, upper=99.5``), so the
  decoded prediction (already in ``[0, 1]``) and the reference live in the same
  intensity space. This is what makes the PSNR/SSIM a true end-to-end measure
  rather than a proxy.
* :func:`full_volume_psnr_ssim` — whole-volume 3-D PSNR/SSIM on ``[0, 1]``
  volumes via :class:`vena.model.fm.metrics.ImageMetrics`.
* :func:`select_content_slices` — choose the axial slice indices for the
  qualitative figure (content range, offset inward, equispaced).
* :func:`render_comparison_figure` — the (1 + ``len(nfe_levels)``)-row
  (real + one row per NFE level) × N-slice comparison panel with per-NFE
  generation+decode time annotations. Row count adapts to the configured NFE
  levels (e.g. 6 rows for ``{1, 2, 5, 10, 20}``).
* :func:`write_latent_preds_h5` — schema-versioned latent-prediction cache.

The orchestration (model build, sampling, decoding) lives in the routine
engine ``routines.fm.exhaustive_val.engine`` so this module stays import-light
and testable without checkpoints.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import torch

from vena.model.autoencoder.maisi.preprocessing import (
    CropPadSpec,
    apply_crop_pad,
    percentile_normalise,
)

logger = logging.getLogger(__name__)


class ExhaustiveValError(Exception):
    """Raised on unrecoverable exhaustive-validation errors."""


def load_real_t1c_normalised(
    image_h5: Path | str,
    patient_id: str,
    *,
    percentile_lower: float = 0.0,
    percentile_upper: float = 99.5,
    foreground_only: bool = True,
) -> torch.Tensor:
    """Load and normalise a patient's reference T1c from the image-domain H5.

    The normalisation mirrors the MAISI encoder exactly
    (``percentile_normalise(lower=0, upper=99.5, foreground_only=True)`` over
    the skull-stripped brain foreground), so the decoded prediction (already in
    ``[0, 1]``) and the reference live in the same intensity space.

    Parameters
    ----------
    image_h5 : Path | str
        Path to the cohort's image H5 (``images/t1c`` raw intensities, ``ids``).
    patient_id : str
        Patient ID as found under ``/ids``.
    percentile_lower, percentile_upper, foreground_only : float, float, bool
        Forwarded to :func:`percentile_normalise`; defaults match the encoder
        (foreground_only=True for skull-stripped volumes).

    Returns
    -------
    torch.Tensor
        Shape ``(H, W, D)`` float32 in ``[0, 1]``.

    Raises
    ------
    ExhaustiveValError
        If the patient ID is absent from the image H5.
    """
    image_h5 = Path(image_h5)
    with h5py.File(image_h5, "r") as f:
        ids = [b.decode() if isinstance(b, bytes) else str(b) for b in f["ids"][:]]
        idx_by_id = {pid: i for i, pid in enumerate(ids)}
        if patient_id not in idx_by_id:
            raise ExhaustiveValError(f"patient '{patient_id}' not found in {image_h5}/ids")
        raw = f["images/t1c"][idx_by_id[patient_id]]  # (H, W, D) raw float32
    vol = torch.from_numpy(np.ascontiguousarray(raw)).float()
    norm = percentile_normalise(
        vol[None, None],  # (1, 1, H, W, D)
        lower=percentile_lower,
        upper=percentile_upper,
        foreground_only=foreground_only,
    )
    return norm[0, 0].contiguous()


def load_real_t1c_box(
    image_h5: Path | str,
    patient_id: str,
    crop_spec: CropPadSpec,
    *,
    percentile_lower: float = 0.0,
    percentile_upper: float = 99.5,
    foreground_only: bool = True,
) -> torch.Tensor:
    """Load, crop to the box, and normalise a patient's reference T1c.

    Reads the native T1c from the image-domain H5, constructs the
    ``CropPadSpec`` from the stored ``crop/origin``, then applies
    :func:`apply_crop_pad` so the result occupies the same box space as the
    VAE-decoded prediction. Normalisation uses ``percentile_normalise`` with
    ``foreground_only=True`` over the skull-stripped brain — identical to the
    encoder's input transform.

    Parameters
    ----------
    image_h5 : Path | str
        Path to the cohort's image H5 (schema 2.0.0: ``images/t1c``,
        ``crop/origin`` int32 ``(N, 3)``, root attr ``crop_box`` JSON list).
    patient_id : str
        Patient ID (matched via ``/ids``).
    crop_spec : CropPadSpec
        Pre-built crop geometry (caller reads ``crop/origin`` and ``crop_box``
        from the H5 and passes them here). Separating this from the H5 read
        keeps the helper pure and unit-testable.
    percentile_lower, percentile_upper, foreground_only : float, float, bool
        Forwarded to :func:`percentile_normalise`.

    Returns
    -------
    torch.Tensor
        Shape ``(*crop_spec.target_shape,)`` float32 in ``[0, 1]``.

    Raises
    ------
    ExhaustiveValError
        If the patient ID is absent from the image H5.
    """
    image_h5 = Path(image_h5)
    with h5py.File(image_h5, "r") as f:
        ids = [b.decode() if isinstance(b, bytes) else str(b) for b in f["ids"][:]]
        idx_by_id = {pid: i for i, pid in enumerate(ids)}
        if patient_id not in idx_by_id:
            raise ExhaustiveValError(f"patient '{patient_id}' not found in {image_h5}/ids")
        raw = f["images/t1c"][idx_by_id[patient_id]]  # (H, W, D) raw float32
    vol = torch.from_numpy(np.ascontiguousarray(raw)).float()
    # Crop/pad to the common box shape before normalisation so percentiles are
    # computed over the box (consistent with the encoder path).
    box_5d = apply_crop_pad(vol[None, None], crop_spec)  # (1, 1, *target_shape)
    norm = percentile_normalise(
        box_5d,
        lower=percentile_lower,
        upper=percentile_upper,
        foreground_only=foreground_only,
    )
    return norm[0, 0].contiguous()


def build_crop_spec_from_h5(
    image_h5: Path | str,
    patient_id: str,
) -> CropPadSpec:
    """Build a :class:`CropPadSpec` for one patient from the image-domain H5.

    Reads ``crop/origin[row]`` (int32 per-axis start) and the ``crop_box``
    root attribute (JSON-encoded ``[H, W, D]`` target shape), then infers
    the native volume shape from ``images/t1c.shape[1:]``.

    Parameters
    ----------
    image_h5 : Path | str
        Path to the cohort's image H5 (schema 2.0.0).
    patient_id : str
        Patient ID (matched via ``/ids``).

    Returns
    -------
    CropPadSpec

    Raises
    ------
    ExhaustiveValError
        If the patient ID is absent from the H5 or required fields are missing.
    """
    import json as _json

    image_h5 = Path(image_h5)
    with h5py.File(image_h5, "r") as f:
        ids = [b.decode() if isinstance(b, bytes) else str(b) for b in f["ids"][:]]
        idx_by_id = {pid: i for i, pid in enumerate(ids)}
        if patient_id not in idx_by_id:
            raise ExhaustiveValError(f"patient '{patient_id}' not found in {image_h5}/ids")
        row = idx_by_id[patient_id]
        try:
            crop_origin = tuple(int(v) for v in f["crop/origin"][row])
        except KeyError as exc:
            raise ExhaustiveValError(
                f"image H5 {image_h5} missing 'crop/origin' (schema 2.0.0 required)"
            ) from exc
        try:
            crop_box_raw = f.attrs["crop_box"]
            if isinstance(crop_box_raw, str):
                target_shape = tuple(int(v) for v in _json.loads(crop_box_raw))
            else:
                target_shape = tuple(int(v) for v in crop_box_raw)
        except KeyError as exc:
            raise ExhaustiveValError(
                f"image H5 {image_h5} missing 'crop_box' root attribute (schema 2.0.0 required)"
            ) from exc
        native_shape = tuple(int(v) for v in f["images/t1c"].shape[1:])

    return CropPadSpec(
        crop_origin=crop_origin,  # type: ignore[arg-type]
        native_shape=native_shape,  # type: ignore[arg-type]
        target_shape=target_shape,  # type: ignore[arg-type]
    )


def full_volume_psnr_ssim(
    pred: torch.Tensor,
    real: torch.Tensor,
    image_metrics: object,
) -> tuple[float, float]:
    """Whole-volume PSNR/SSIM between two ``[0, 1]`` volumes of shape ``(H,W,D)``.

    Parameters
    ----------
    pred, real : torch.Tensor
        ``(H, W, D)`` float tensors in ``[0, 1]`` on the same device.
    image_metrics : ImageMetrics
        A :class:`vena.model.fm.metrics.ImageMetrics` instance (``data_range``
        must be ``1.0``).

    Returns
    -------
    tuple[float, float]
        ``(psnr_db, ssim)``.
    """
    p = pred[None, None]
    r = real[None, None]
    mask = torch.ones_like(p, dtype=torch.bool)
    psnr = float(image_metrics.psnr(p, r, mask).reshape(-1)[0].item())
    ssim = float(image_metrics.ssim(p, r, mask).reshape(-1)[0].item())
    return psnr, ssim


def select_content_slices(
    reference: torch.Tensor | np.ndarray,
    n_slices: int = 10,
    offset: int = 10,
) -> list[int]:
    """Pick ``n_slices`` equispaced axial indices inside the content range.

    The content range is the first/last axial slice (last axis) with any
    non-zero voxel; it is shrunk inward by ``offset`` on each side (so the
    figure avoids near-empty end slices), then ``n_slices`` equispaced indices
    are taken. Degenerate ranges fall back to a centred window.

    Parameters
    ----------
    reference : Tensor | ndarray
        ``(H, W, D)`` reference volume; axial axis is the last one.
    n_slices : int
        Number of slice indices to return.
    offset : int
        Inward shrink applied to both ends of the content range.

    Returns
    -------
    list[int]
        ``n_slices`` ascending axial indices in ``[0, D)``.
    """
    arr = reference.detach().cpu().numpy() if isinstance(reference, torch.Tensor) else reference
    depth = arr.shape[-1]
    has_content = np.array([bool(np.any(arr[..., k] > 0)) for k in range(depth)])
    nz = np.flatnonzero(has_content)
    if nz.size == 0:
        lo, hi = 0, depth - 1
    else:
        lo, hi = int(nz[0]), int(nz[-1])
    lo2, hi2 = lo + offset, hi - offset
    if hi2 <= lo2:  # offset collapsed the range — fall back to the raw content span
        lo2, hi2 = lo, hi
    if hi2 <= lo2:  # still degenerate (single content slice)
        lo2, hi2 = 0, depth - 1
    idx = np.linspace(lo2, hi2, num=n_slices)
    return [round(float(v)) for v in idx]


def render_comparison_figure(
    real: torch.Tensor,
    synth_by_nfe: dict[int, torch.Tensor],
    time_by_nfe: dict[int, float],
    slice_indices: list[int],
    *,
    patient_id: str,
    mean_ssim: float,
    title_tag: str,
    out_path: Path | str,
) -> Path:
    """Render the (1 + ``len(synth_by_nfe)``)-row comparison panel and save it.

    Row 0 is the real T1c; subsequent rows are the synthesised T1c at each NFE
    level in *descending* order (highest NFE first), each annotated with its
    generation+decode wall-clock time. Columns are the chosen axial slices. The
    row count adapts to however many NFE levels are provided.

    Parameters
    ----------
    real : Tensor
        ``(H, W, D)`` reference volume in ``[0, 1]``.
    synth_by_nfe : dict[int, Tensor]
        NFE level -> ``(H, W, D)`` synthesised volume in ``[0, 1]``.
    time_by_nfe : dict[int, float]
        NFE level -> generation+decode seconds (for that patient).
    slice_indices : list[int]
        Axial indices (columns).
    patient_id, mean_ssim, title_tag : str, float, str
        Figure title metadata (``title_tag`` e.g. "best" / "worst").
    out_path : Path | str
        PNG destination.

    Returns
    -------
    Path
        ``out_path``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nfes = sorted(synth_by_nfe.keys(), reverse=True)
    rows = ["real", *nfes]
    n_rows, n_cols = len(rows), len(slice_indices)
    real_np = real.detach().cpu().float().numpy()

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.4 * n_cols, 1.5 * n_rows), squeeze=False)
    for r, row in enumerate(rows):
        if row == "real":
            vol = real_np
            row_label = "Real T1c"
        else:
            vol = synth_by_nfe[row].detach().cpu().float().numpy()
            row_label = f"Synth NFE={row}\n(t={time_by_nfe.get(row, float('nan')):.2f}s)"
        for c, k in enumerate(slice_indices):
            ax = axes[r][c]
            ax.imshow(np.rot90(vol[..., k]), cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"z={k}", fontsize=7)
            if c == 0:
                ax.set_ylabel(row_label, fontsize=8)
    fig.suptitle(
        f"{title_tag.upper()} — {patient_id}  (mean SSIM={mean_ssim:.4f})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_latent_preds_h5(
    path: Path | str,
    entries: list[tuple[str, int, np.ndarray]],
    *,
    epoch: int,
    run_id: str,
    ema_snapshot_sha: str | None = None,
    extra_attrs: dict[str, object] | None = None,
) -> Path:
    """Write predicted latents to a schema-versioned H5.

    Layout: ``/predictions/{patient_id}/nfe_{N}`` float16 gzip-4, one group per
    patient. Root attrs follow ``.claude/rules/h5-design-principles.md``.

    Parameters
    ----------
    path : Path | str
        Output H5 path.
    entries : list[tuple[str, int, ndarray]]
        ``(patient_id, nfe, latent[C,h,w,d])`` records.
    epoch, run_id : int, str
        Provenance.
    ema_snapshot_sha : str | None
        SHA of the EMA snapshot that produced these predictions.
    extra_attrs : dict | None
        Additional root attrs.

    Returns
    -------
    Path
        ``path``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = "1.0"
        f.attrs["created_at"] = datetime.now(UTC).isoformat()
        f.attrs["producer"] = "routines.fm.exhaustive_val:latent_preds"
        f.attrs["epoch"] = int(epoch)
        f.attrs["run_id"] = run_id
        if ema_snapshot_sha:
            f.attrs["ema_snapshot_sha256"] = ema_snapshot_sha
        for k, v in (extra_attrs or {}).items():
            f.attrs[k] = v
        grp_root = f.create_group("predictions")
        grp_root.attrs["description"] = "Predicted T1c latents, one group per patient."
        for pid, nfe, latent in entries:
            grp = grp_root.require_group(pid)
            grp.attrs["patient_id"] = pid
            key = f"nfe_{int(nfe)}"
            if key in grp:
                del grp[key]
            dset = grp.create_dataset(
                key,
                data=np.ascontiguousarray(latent.astype(np.float16)),
                dtype="float16",
                compression="gzip",
                compression_opts=4,
            )
            dset.attrs["units"] = "dimensionless"
            dset.attrs["description"] = "MAISI VAE latent of predicted T1c"
            dset.attrs["dtype"] = "float16"
            dset.attrs["nfe"] = int(nfe)
    return path
