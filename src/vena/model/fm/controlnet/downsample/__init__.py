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
* ``zero_out``    — drop the conditioning signal but preserve the channel
                    slot. Used by the S1 baseline so warm-start to S2/S3
                    (which switch back to ``identity`` or ``lift_to_4ch``)
                    requires no channel-dim surgery.
* ``lift_to_4ch`` — learned ``Conv3d(1, 4, kernel_size=1)`` that lifts a
                    1-channel mask to 4 channels (matching the latent
                    modalities' energy). Reserved for S2/S3.
* ``vae``         — encode through the frozen MAISI VAE. Intended for
                    image-domain priors that must align with the trunk's
                    latent statistics. Not used in S1.

Each operator implements :class:`AbstractDownsampler` and consumes
``(B, C, H, W, D)`` image-space tensors, returning ``(B, C', h, w, d)`` latent
tensors (``C' = C`` for stateless operators; ``C' != C`` only for
channel-lifting operators that expose ``out_channels``).
"""

from typing import Any

from .base import AbstractDownsampler
from .identity import IdentityDownsampler
from .lift import LiftTo4ChDownsampler
from .nearest import NearestDownsampler
from .pooling import AvgPoolDownsampler
from .trilinear import TrilinearDownsampler
from .zero_out import ZeroOutDownsampler

__all__ = [
    "AbstractDownsampler",
    "AvgPoolDownsampler",
    "IdentityDownsampler",
    "LiftTo4ChDownsampler",
    "NearestDownsampler",
    "TrilinearDownsampler",
    "ZeroOutDownsampler",
    "get_downsampler",
]


def get_downsampler(name: str, **kwargs: Any) -> AbstractDownsampler:
    """Construct a downsampler by registry name.

    Parameters
    ----------
    name : str
        One of ``"identity"``, ``"nearest"``, ``"trilinear"``, ``"avg_pool"``,
        ``"zero_out"``, ``"lift_to_4ch"``. ``"vae"`` is reserved and raises
        ``NotImplementedError`` until the image-prior pathway lands.
    **kwargs
        Operator-specific kwargs, e.g. ``factor=4`` for resamplers,
        ``threshold=0.5`` for nearest with binarisation, ``out_channels=N``
        for ``lift_to_4ch``.

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
    if name in ("zero_out", "zeroout"):
        return ZeroOutDownsampler(**kwargs)
    if name in ("lift_to_4ch", "lift4ch"):
        return LiftTo4ChDownsampler(**kwargs)
    if name == "vae":
        raise NotImplementedError(
            "VAE-based image-to-latent downsampling lands with the prior-maps "
            "image-domain pathway; not used by S1."
        )
    raise ValueError(
        f"unknown downsampler '{name}'; choose from "
        "{identity, nearest, trilinear, avg_pool, zero_out, lift_to_4ch}"
    )
