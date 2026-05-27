"""Image-to-latent downsampling operators.

The registry pattern parallels ``src/vena/model/autoencoder/maisi/encode/masks/``
(the encoding pipeline's per-class avg-pool downsampler). Adding a new
operator is one file plus a one-line registration in :func:`get_downsampler`.

Currently supported names:

* ``identity``    — no spatial transform. Use when the input is already at
                    latent resolution (the tumour-mask path in S1).
* ``nearest``     — nearest-neighbour interpolation to a fixed factor or shape.
* ``trilinear``   — trilinear interpolation (soft maps such as Frangi
                    vesselness, perfusion scores).
* ``avg_pool``    — strided average pooling (matches the per-class avg-pool
                    used by the latent-H5 converter; equivalent to MAISI's
                    ``F.avg_pool3d`` with the VAE compression factor).
* ``vae``         — encode through the frozen MAISI VAE. Intended for
                    image-domain priors that must align with the trunk's
                    latent statistics. Not used in S1.

Each operator implements :class:`AbstractDownsampler` and consumes
``(B, C, H, W, D)`` image-space tensors, returning ``(B, C, h, w, d)`` latent
tensors with the spatial factor or shape specified at construction.
"""

from typing import Any

from .base import AbstractDownsampler
from .identity import IdentityDownsampler
from .nearest import NearestDownsampler
from .pooling import AvgPoolDownsampler
from .trilinear import TrilinearDownsampler

__all__ = [
    "AbstractDownsampler",
    "IdentityDownsampler",
    "NearestDownsampler",
    "TrilinearDownsampler",
    "AvgPoolDownsampler",
    "get_downsampler",
]


def get_downsampler(name: str, **kwargs: Any) -> AbstractDownsampler:
    """Construct a downsampler by registry name.

    Parameters
    ----------
    name : str
        One of ``"identity"``, ``"nearest"``, ``"trilinear"``, ``"avg_pool"``.
        ``"vae"`` is reserved and raises ``NotImplementedError`` until the
        image-prior pathway lands.
    **kwargs
        Operator-specific kwargs, e.g. ``factor=4`` for resamplers,
        ``threshold=0.5`` for nearest with binarisation.

    Raises
    ------
    ValueError
        Unknown name.
    NotImplementedError
        For reserved-but-not-yet-implemented operators (``"vae"``).
    """
    name = name.lower()
    if name == "identity":
        return IdentityDownsampler(**kwargs)
    if name == "nearest":
        return NearestDownsampler(**kwargs)
    if name == "trilinear":
        return TrilinearDownsampler(**kwargs)
    if name in ("avg_pool", "avgpool"):
        return AvgPoolDownsampler(**kwargs)
    if name == "vae":
        raise NotImplementedError(
            "VAE-based image-to-latent downsampling lands with the prior-maps "
            "image-domain pathway; not used by S1."
        )
    raise ValueError(
        f"unknown downsampler '{name}'; choose from "
        "{identity, nearest, trilinear, avg_pool}"
    )
