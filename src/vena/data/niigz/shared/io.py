"""Generic NIfTI I/O primitives shared across cohort loaders.

Wraps `nibabel.load` / `nibabel.save` in a frozen `NiftiVolume` dataclass so the
rest of the codebase moves a single, self-describing object instead of an
``(array, affine, header)`` tuple.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from nibabel.nifti1 import Nifti1Header
from numpy.typing import NDArray

from .exceptions import NiigzLoadError


@dataclass(frozen=True)
class NiftiVolume:
    """Volume + physical metadata loaded from a single ``.nii.gz`` file.

    Attributes
    ----------
    array : np.ndarray
        Voxel data, shape ``(X, Y, Z)`` for 3D volumes.
    affine : np.ndarray
        4x4 voxel-to-world affine.
    header : nib.Nifti1Header
        Full NIfTI header. Kept so downstream writes can preserve provenance.
    path : Path
        Source file path; used by logs / manifests.
    spacing_mm : tuple[float, float, float]
        Voxel spacing in millimetres along each axis (extracted from the header).
    """

    array: NDArray[Any]
    affine: NDArray[np.floating[Any]]
    header: Nifti1Header
    path: Path
    spacing_mm: tuple[float, float, float]


def load_nii(path: Path | str) -> NiftiVolume:
    """Load a NIfTI volume from disk.

    Parameters
    ----------
    path
        Filesystem path to a ``.nii`` or ``.nii.gz`` file.

    Returns
    -------
    NiftiVolume
        Frozen container with array, affine, header, source path and voxel spacing.

    Raises
    ------
    NiigzLoadError
        If the path does not exist or `nibabel` fails to load it.
    """
    p = Path(path)
    if not p.exists():
        raise NiigzLoadError(f"NIfTI file does not exist: {p}")
    try:
        img = nib.load(str(p))
    except Exception as exc:  # nibabel raises various subclasses
        raise NiigzLoadError(f"Failed to load {p}: {exc}") from exc

    # `get_fdata()` always returns float64; downstream code casts as needed.
    array = np.asarray(img.dataobj)
    affine = np.asarray(img.affine, dtype=np.float64)
    header = img.header
    zooms = header.get_zooms()[:3]
    spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
    return NiftiVolume(array=array, affine=affine, header=header, path=p, spacing_mm=spacing)


def save_nii(
    array: NDArray[Any],
    affine: NDArray[np.floating[Any]],
    header: Nifti1Header | None,
    path: Path | str,
) -> Path:
    """Write a NIfTI volume to disk, preserving the source affine.

    The header is cloned from the input when provided and its ``datatype`` /
    ``bitpix`` are updated to match ``array.dtype`` so the file round-trips
    cleanly under `nibabel.load`.

    Parameters
    ----------
    array
        Voxel data to write.
    affine
        4x4 voxel-to-world transform to attach.
    header
        Optional template header (e.g. from a sibling modality). If None, a
        fresh `Nifti1Header` is built.
    path
        Output path; parent directories are created on demand.

    Returns
    -------
    Path
        Resolved output path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if header is None:
        new_header = Nifti1Header()
    else:
        new_header = header.copy()
    new_header.set_data_dtype(array.dtype)

    img = nib.Nifti1Image(array, affine, header=new_header)
    nib.save(img, str(out))
    return out
