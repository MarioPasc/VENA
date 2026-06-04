"""Offline-augmentation variant builders (K = 4: ``v1``..``v4``).

Each variant is a TorchIO :class:`~torchio.Compose` ready to apply to a
:class:`~torchio.Subject` whose intensity images are tagged
``t1pre``/``t1c``/``t2``/``flair`` and whose tumour segmentation is the
:class:`~torchio.LabelMap` tagged ``tumor``.

Variant menu (locked by ``.claude/notes/augmentation_approach/refined_proposal.md``):

* ``v1`` field/scanner — bias field + small gamma, **inputs only**.
* ``v2`` contrast-shape — monotonic histogram remap + brightness/contrast,
  **inputs only**.
* ``v3`` SNR/resolution — Gaussian noise + anisotropy + blur + low-prob
  motion, **inputs only**.
* ``v4`` anatomy — light elastic + small-angle affine, joint over the
  full Subject (inputs **and** target **and** mask).

The per-transform probabilities here are the "probability of firing
inside a variant", not the probability of choosing the variant at
train-time (that is controlled by ``variant_weights`` in the FM trainer's
data config). Default values are set to the refined-proposal defaults
and can be overridden per-variant via ``hp_overrides``.
"""

from __future__ import annotations

import random
from typing import Any

import torchio as tio

from vena.data.augment.offline.torchio_adapters import MonaiHistogramShift

VARIANT_NAMES: tuple[str, ...] = ("v1", "v2", "v3", "v4")
"""Canonical, ordered list of bank variant names."""

VARIANT_INPUT_ONLY: dict[str, bool] = {
    "v1": True,
    "v2": True,
    "v3": True,
    "v4": False,
}
"""Whether each variant applies to inputs only (True) or the full Subject (False)."""

_INPUT_KEYS: tuple[str, ...] = ("t1pre", "t2", "flair")
"""Modalities forwarded into the network as inputs; the target is ``t1c``."""


def _v1(hp: dict[str, Any]) -> tio.Compose:
    """Field/scanner: bias field + small gamma (inputs only)."""
    bias_order = int(hp.get("bias_order", 3))
    bias_coeff = tuple(hp.get("bias_coefficients", (-0.5, 0.5)))
    gamma_log = tuple(hp.get("gamma_log_range", (-0.3, 0.3)))
    gamma_prob = float(hp.get("gamma_prob", 0.5))
    transforms: list[tio.Transform] = [
        tio.RandomBiasField(
            coefficients=bias_coeff,
            order=bias_order,
            include=list(_INPUT_KEYS),
        ),
        tio.RandomGamma(
            log_gamma=gamma_log,
            p=gamma_prob,
            include=list(_INPUT_KEYS),
        ),
    ]
    return tio.Compose(transforms)


def _v2(hp: dict[str, Any]) -> tio.Compose:
    """Contrast-shape: monotonic histogram remap + brightness/contrast (inputs only)."""
    n_ctrl_low = int(hp.get("hist_shift_n_control_low", 8))
    n_ctrl_high = int(hp.get("hist_shift_n_control_high", 12))
    bc_log_range = tuple(hp.get("brightness_contrast_log_range", (-0.15, 0.15)))
    bc_prob = float(hp.get("brightness_contrast_prob", 0.4))
    transforms: list[tio.Transform] = [
        MonaiHistogramShift(
            num_control_points=(n_ctrl_low, n_ctrl_high),
            prob=1.0,
            include=list(_INPUT_KEYS),
        ),
        # Brightness/contrast is implemented as a second gamma on a tighter
        # range; the multiplicative effect on top of the histogram shift
        # exercises a different region of the transfer-function family.
        tio.RandomGamma(
            log_gamma=bc_log_range,
            p=bc_prob,
            include=list(_INPUT_KEYS),
        ),
    ]
    return tio.Compose(transforms)


def _v3(hp: dict[str, Any]) -> tio.Compose:
    """SNR/resolution: noise + anisotropy + blur + low-prob motion (inputs only)."""
    noise_std = tuple(hp.get("noise_std", (0.0, 0.05)))
    aniso_axes = tuple(hp.get("anisotropy_axes", (0, 1, 2)))
    aniso_down = tuple(hp.get("anisotropy_downsampling", (1.5, 4.0)))
    aniso_prob = float(hp.get("anisotropy_prob", 0.7))
    blur_std = tuple(hp.get("blur_std", (0.0, 1.5)))
    blur_prob = float(hp.get("blur_prob", 0.4))
    motion_translation = float(hp.get("motion_translation", 5.0))
    motion_degrees = float(hp.get("motion_degrees", 5.0))
    motion_num = tuple(hp.get("motion_num_transforms", (1, 3)))
    motion_prob = float(hp.get("motion_prob", 0.1))
    transforms: list[tio.Transform] = [
        tio.RandomNoise(
            std=noise_std,
            include=list(_INPUT_KEYS),
        ),
        tio.RandomAnisotropy(
            axes=aniso_axes,
            downsampling=aniso_down,
            p=aniso_prob,
            include=list(_INPUT_KEYS),
        ),
        tio.RandomBlur(
            std=blur_std,
            p=blur_prob,
            include=list(_INPUT_KEYS),
        ),
        # TorchIO's RandomMotion requires `num_transforms: int` (single
        # value). To exercise the (1, 3) range from the refined proposal,
        # sample once per `_v3` call from Python's `random` (seeded
        # upstream by bank_builder per (rank, src_idx, variant)).
        tio.RandomMotion(
            degrees=motion_degrees,
            translation=motion_translation,
            num_transforms=random.randint(int(motion_num[0]), int(motion_num[1])),
            p=motion_prob,
            include=list(_INPUT_KEYS),
        ),
    ]
    return tio.Compose(transforms)


def _v4(hp: dict[str, Any]) -> tio.Compose:
    """Anatomy: light elastic + small-angle affine (joint over the full Subject).

    No ``include=`` filter — TorchIO applies spatial transforms consistently
    to every member of the Subject, with nearest-neighbour resampling for
    the LabelMap (tumour mask). The crop+pad invariant of the aug-image H5
    (``expected_shape=(192, 224, 192)``, ``crop_origin=(0, 0, 0)``) is
    preserved because TorchIO's affine/elastic resample in place at the
    same grid.
    """
    elastic_num_ctrl = int(hp.get("elastic_num_control_points", 7))
    elastic_max_disp = float(hp.get("elastic_max_displacement", 4.0))
    elastic_locked = bool(hp.get("elastic_locked_borders", True))
    elastic_prob = float(hp.get("elastic_prob", 0.7))
    affine_scales = tuple(hp.get("affine_scales", (0.9, 1.1)))
    affine_degrees = float(hp.get("affine_degrees", 10.0))
    affine_translation = float(hp.get("affine_translation_voxels", 8.0))
    affine_prob = float(hp.get("affine_prob", 0.7))
    transforms: list[tio.Transform] = [
        tio.RandomElasticDeformation(
            num_control_points=elastic_num_ctrl,
            max_displacement=elastic_max_disp,
            locked_borders=2 if elastic_locked else 0,
            p=elastic_prob,
            image_interpolation="linear",
            label_interpolation="nearest",
        ),
        tio.RandomAffine(
            scales=affine_scales,
            degrees=affine_degrees,
            translation=affine_translation,
            p=affine_prob,
            image_interpolation="linear",
            label_interpolation="nearest",
        ),
    ]
    return tio.Compose(transforms)


_BUILDERS = {"v1": _v1, "v2": _v2, "v3": _v3, "v4": _v4}


def make_variant(
    name: str,
    hp_overrides: dict[str, Any] | None = None,
) -> tio.Compose:
    """Build one variant's TorchIO Compose.

    Parameters
    ----------
    name : str
        One of :data:`VARIANT_NAMES`.
    hp_overrides : dict | None
        Per-variant hyperparameter overrides. Unknown keys are ignored
        (forward-compatible with new transform knobs); known keys override
        the defaults declared in the variant builder.

    Returns
    -------
    tio.Compose
        Composed transform ready to apply to a TorchIO :class:`Subject`.

    Raises
    ------
    KeyError
        If ``name`` is not in :data:`VARIANT_NAMES`.
    """
    if name not in _BUILDERS:
        raise KeyError(f"unknown variant {name!r}; available: {VARIANT_NAMES}")
    return _BUILDERS[name](dict(hp_overrides or {}))
