"""ANTsPy atlas → subject registration with disk-cached warps.

Each subject's transform is cached under
``<cache_root>/<subject_sha>/`` keyed by the SHA-256 of the T1pre bytes, so
re-runs are nearly free. ``register_mni_to_subject`` computes the warp from
the MNI152NLin2009cAsym T1 template into subject T1pre space; downstream
helpers reuse the cached forward transform to warp atlas label maps.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import ants
import nibabel as nib
import numpy as np
from numpy.typing import NDArray

from vena.data.niigz import NiftiVolume

logger = logging.getLogger(__name__)

RegistrationKind = Literal["rigid", "affine", "syn"]
_ANTSPY_TYPE = {"rigid": "Rigid", "affine": "Affine", "syn": "SyN"}


@dataclass(frozen=True)
class AtlasWarpResult:
    """Resolved warp from MNI152 template → subject T1pre space."""

    subject_id: str
    kind: RegistrationKind
    fwd_transforms: tuple[str, ...]
    inv_transforms: tuple[str, ...]
    fixed_path: Path  # subject T1pre on disk (the registration target)
    moving_path: Path  # MNI152 template (the moving image)
    cache_dir: Path


def _hash_volume(volume: NiftiVolume, n_bytes: int = 1 << 20) -> str:
    """Cheap SHA-256 of the first N bytes of a NIfTI file (path-based)."""
    h = hashlib.sha256()
    with volume.path.open("rb") as f:
        h.update(f.read(n_bytes))
    return h.hexdigest()[:16]


def _save_volume_temp(volume: NiftiVolume, out_path: Path) -> Path:
    """Write a NiftiVolume to ``out_path`` if not already present."""
    if out_path.exists():
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(np.asarray(volume.array), volume.affine, volume.header),
        str(out_path),
    )
    return out_path


def register_mni_to_subject(
    t1pre: NiftiVolume,
    mni_template_path: Path,
    cache_root: Path,
    *,
    kind: RegistrationKind = "affine",
    subject_id: str | None = None,
) -> AtlasWarpResult:
    """Register the MNI152 template *into* subject T1pre space.

    The transforms are cached under ``cache_root/<subject_sha>_<kind>/``.

    Parameters
    ----------
    t1pre
        Subject's T1pre volume (the fixed image in this registration).
    mni_template_path
        MNI152NLin2009cAsym T1 template (the moving image).
    cache_root
        Root cache directory; populated lazily.
    kind
        Registration type. ``"affine"`` is the routine default (rigid+affine
        composite, ~10-30 s/subject on a modern CPU). ``"syn"`` adds
        non-linear refinement (~5-10 min/subject).
    """
    sid = subject_id or t1pre.path.stem.split("_")[0]
    digest = _hash_volume(t1pre)
    sub_cache = Path(cache_root) / f"{sid}_{digest}_{kind}"
    sub_cache.mkdir(parents=True, exist_ok=True)

    fwd_file = sub_cache / "fwd_transforms.txt"
    inv_file = sub_cache / "inv_transforms.txt"
    fixed_cache = sub_cache / "fixed_t1pre.nii.gz"
    moving_cache = sub_cache / "moving_mni152.nii.gz"

    if fwd_file.exists() and inv_file.exists() and fixed_cache.exists():
        logger.info("atlas warp cache hit: %s", sub_cache)
        return AtlasWarpResult(
            subject_id=sid,
            kind=kind,
            fwd_transforms=tuple(fwd_file.read_text().splitlines()),
            inv_transforms=tuple(inv_file.read_text().splitlines()),
            fixed_path=fixed_cache,
            moving_path=moving_cache,
            cache_dir=sub_cache,
        )

    _save_volume_temp(t1pre, fixed_cache)
    if not moving_cache.exists():
        shutil.copy(mni_template_path, moving_cache)

    fixed_ants = ants.image_read(str(fixed_cache))
    moving_ants = ants.image_read(str(moving_cache))
    reg = ants.registration(
        fixed=fixed_ants,
        moving=moving_ants,
        type_of_transform=_ANTSPY_TYPE[kind],
        outprefix=str(sub_cache) + "/ants_",
    )
    # Persist transform-file paths so cache loads can reuse them without
    # re-running the registration.
    fwd = tuple(str(p) for p in reg["fwdtransforms"])
    inv = tuple(str(p) for p in reg["invtransforms"])
    fwd_file.write_text("\n".join(fwd))
    inv_file.write_text("\n".join(inv))
    logger.info("registered %s with %s; %d fwd transforms", sid, kind, len(fwd))
    return AtlasWarpResult(
        subject_id=sid,
        kind=kind,
        fwd_transforms=fwd,
        inv_transforms=inv,
        fixed_path=fixed_cache,
        moving_path=moving_cache,
        cache_dir=sub_cache,
    )


def warp_label_to_subject(
    warp: AtlasWarpResult,
    label_path: Path,
    *,
    interpolator: str = "nearestNeighbor",
) -> NDArray[np.int32]:
    """Warp an MNI-space label image into the subject's T1pre space.

    Uses ``ants.apply_transforms`` with the *forward* transform list
    (template→subject). ``"nearestNeighbor"`` is the default — labels are
    integer-valued, so linear interpolation would corrupt the encoding.
    """
    fixed = ants.image_read(str(warp.fixed_path))
    moving = ants.image_read(str(label_path))
    warped = ants.apply_transforms(
        fixed=fixed,
        moving=moving,
        transformlist=list(warp.fwd_transforms),
        interpolator=interpolator,
    )
    arr = warped.numpy().astype(np.int32)
    return arr
