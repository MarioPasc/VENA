"""Optimally Oriented Flux (OOF) vesselness.

Implements the multi-scale OOF response of Law & Chung 2008 in 3D. OOF is
defined as the outward gradient flux through the surface of a sphere of radius
``r`` centred at every voxel; the resulting 3 × 3 symmetric flux matrix
``Q(x, r)`` has eigenvalues whose two largest magnitudes characterise the
across-vessel curvature, identically to the Hessian-based Frangi response but
without the second-derivative noise amplification that plagues Hessian
estimators on SWI / SWAN.

We compute ``Q`` in the Fourier domain using the closed-form expression

.. math::

    \\widehat{Q}_{ij}(\\boldsymbol\\omega; r) = \\widehat{I}(\\boldsymbol\\omega)
        \\,\\cdot\\, \\frac{4 \\pi r \\, j_{1}(|\\boldsymbol\\omega|\\,r)}
                          {|\\boldsymbol\\omega|^{2}}
        \\,\\cdot\\, \\omega_{i}\\, \\omega_{j} ,

where :math:`j_{1}` is the spherical Bessel function of the first kind, order
one. The six independent components ``Q_{00}, Q_{11}, Q_{22}, Q_{01}, Q_{02},
Q_{12}`` are recovered by a single inverse FFT each, and the per-voxel
eigendecomposition is delegated to :func:`numpy.linalg.eigvalsh` (closed-form
for 3 × 3 symmetric matrices).

After the eigendecomposition, the response is divided by the sphere area
:math:`4 \\pi r^{2}` (Law-Chung 2008 eq. 16) to obtain the per-unit-area
response. Without this normalisation the response grows as :math:`O(r^{2})`
and the multi-scale ``max`` is biased toward the largest scale instead of
the scale that best matches the local vessel radius.

ITK ecosystem note
------------------
``itk-tubetk`` (5.4 / 1.4.1) exposes Hessian-based vesselness operators
(`HessianToObjectnessMeasureImageFilter`, `MultiScaleHessianBasedMeasureImageFilter`,
`Hessian3DToVesselnessMeasureImageFilter`) and a diffusion-based enhancer
(`EnhanceTubesUsingDiffusion`). It does **not** ship the OOF filter. The OOF
formulation lives in the unmerged ITK external module `ITKOptimallyOrientedFlux`
and is not Python-bound. We therefore compute the OOF kernel directly through
:mod:`scipy.fft` and :func:`scipy.special.spherical_jn`, and use ITK only for
the optional VED-style preprocessor (see
:mod:`vena.vessel_priors.preprocessing.anisotropic_diffusion`).

References
----------
Law, M. W. K. & Chung, A. C. S. *Three Dimensional Curvilinear Structure
Detection Using Optimally Oriented Flux*. ECCV 2008, LNCS 5305, pp. 368–382.
https://doi.org/10.1007/978-3-540-88693-8_27

Bériault, S., Xiao, Y., Collins, D. L. & Pike, G. B. *Automatic
SWI Venography Segmentation Using Conditional Random Fields*. IEEE TMI 34(12),
2015. https://doi.org/10.1109/TMI.2015.2447006 (empirical SWI validation of
OOF against Frangi / Sato / Jerman).
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.fft import fftn, fftfreq, ifftn
from scipy.special import spherical_jn

from vena.vessel_priors.abc_model import (
    AbstractVesselModel,
    VesselInput,
    VesselOutput,
)

logger = logging.getLogger(__name__)


class OOFVesselModel(AbstractVesselModel):
    """Multi-scale Optimally Oriented Flux on a brain-masked SWI volume.

    Parameters
    ----------
    sigma_min_mm, sigma_max_mm
        Inclusive radius range, in millimetres, sampled at ``sigma_steps``
        values. The OOF "scale" corresponds to the sphere radius ``r`` over
        which the outward flux is computed (Law & Chung 2008, §2). Proposal
        default: ``[0.5, 2.5]`` (same range as Frangi for fair comparison).
    sigma_steps
        Number of radii uniformly between min and max (default 5).
    black_ridges
        ``True`` selects dark tubular structures (SWI veins appear dark on
        magnitude; SWAN-derived priors use this convention). Proposal default.
    threshold
        Soft-response threshold used to derive the binary mask (default
        ``0.15``).
    normalize
        Intensity normalisation prior to filtering. ``"percentile"`` clips at
        ``percentile_clip`` then min-max-scales to ``[0, 1]`` within the brain
        mask. ``"minmax"`` skips the clip.
    percentile_clip
        Lower / upper percentile when ``normalize == "percentile"``.
    eps
        Small additive constant used to stabilise the divisions
        ``1 / |omega|`` and ``1 / max(response)``.
    voxel_spacing_mm
        Optional override ``(sx, sy, sz)``. When ``None``, the input volume's
        own ``spacing_mm`` is used. UCSF-PDGM is 1 mm isotropic so the default
        is safe; for anisotropic data the radii are scaled by the mean voxel
        size.
    """

    name: ClassVar[str] = "oof"

    def __init__(
        self,
        sigma_min_mm: float = 0.5,
        sigma_max_mm: float = 2.5,
        sigma_steps: int = 5,
        black_ridges: bool = True,
        threshold: float = 0.15,
        normalize: Literal["minmax", "percentile"] = "percentile",
        percentile_clip: tuple[float, float] = (0.5, 99.5),
        eps: float = 1e-8,
        voxel_spacing_mm: tuple[float, float, float] | None = None,
    ) -> None:
        if sigma_min_mm <= 0 or sigma_max_mm <= 0:
            raise ValueError("radius values must be positive")
        if sigma_max_mm < sigma_min_mm:
            raise ValueError("sigma_max_mm must be >= sigma_min_mm")
        if sigma_steps < 1:
            raise ValueError("sigma_steps must be >= 1")
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("threshold must lie in [0, 1]")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.sigma_min_mm = float(sigma_min_mm)
        self.sigma_max_mm = float(sigma_max_mm)
        self.sigma_steps = int(sigma_steps)
        self.black_ridges = bool(black_ridges)
        self.threshold = float(threshold)
        self.normalize = normalize
        self.percentile_clip = (float(percentile_clip[0]), float(percentile_clip[1]))
        self.eps = float(eps)
        self.voxel_spacing_mm = (
            tuple(float(s) for s in voxel_spacing_mm) if voxel_spacing_mm else None
        )

    # ------------------------------------------------------------------ helpers

    def _radii_voxels(
        self, spacing_mm: tuple[float, float, float]
    ) -> NDArray[np.float64]:
        s = self.voxel_spacing_mm or spacing_mm
        mean_mm = float(np.mean(s))
        radii_mm = np.linspace(self.sigma_min_mm, self.sigma_max_mm, self.sigma_steps)
        return radii_mm / mean_mm

    def _normalize(
        self,
        img: NDArray[np.float32],
        brain: NDArray[np.bool_],
    ) -> NDArray[np.float32]:
        in_brain = img[brain]
        if in_brain.size == 0:
            return img
        if self.normalize == "percentile":
            lo = float(np.percentile(in_brain, self.percentile_clip[0]))
            hi = float(np.percentile(in_brain, self.percentile_clip[1]))
        else:
            lo = float(in_brain.min())
            hi = float(in_brain.max())
        if hi <= lo:
            return np.zeros_like(img)
        out = np.clip((img - lo) / (hi - lo + self.eps), 0.0, 1.0).astype(np.float32)
        out *= brain.astype(np.float32)
        return out

    # ------------------------------------------------------------------ core OOF

    def _oof_multiscale(
        self,
        image: NDArray[np.float32],
        radii: NDArray[np.float64],
    ) -> NDArray[np.float32]:
        """Compute multi-scale OOF, returning the max per-voxel response.

        The eigenvalue convention follows Law & Chung 2008: at every voxel, the
        OOF matrix ``Q`` has eigenvalues ``λ_a ≤ λ_b ≤ λ_c`` sorted ascending.
        For a *dark* tube on a lighter background (SWI veins) the across-vessel
        eigenvalues become large *negative*, hence

        - ``black_ridges=True``  → response = ``max(0, -(λ_a + λ_b) / 2)``
        - ``black_ridges=False`` → response = ``max(0,  (λ_b + λ_c) / 2)``

        The maximum across radii produces the final scale-invariant response.
        """
        x = image.astype(np.float32, copy=False)
        shape = x.shape
        # Use real-to-complex FFT for memory efficiency; spectrum is conjugate-
        # symmetric and the real iFFT recovers the spatial-domain response.
        spectrum = fftn(x.astype(np.complex64), norm="backward")

        # Angular-frequency grid (rad / voxel) on the same shape as `spectrum`.
        omega_axes = [2.0 * np.pi * fftfreq(n).astype(np.float32) for n in shape]
        ox, oy, oz = np.meshgrid(*omega_axes, indexing="ij")
        omega2 = ox * ox + oy * oy + oz * oz
        omega = np.sqrt(omega2)
        # Avoid 1/0 at DC; the kernel is set to zero there anyway via masking.
        omega_safe = np.where(omega > self.eps, omega, np.float32(1.0))

        best = np.zeros(shape, dtype=np.float32)

        for r in radii:
            kr = omega_safe * np.float32(r)
            # spherical_jn returns float64; cast to float32 to keep RAM in check.
            j1 = spherical_jn(1, kr).astype(np.float32)
            # OOF Fourier kernel (Law-Chung 2008, eq. 13 in the derivation): the
            # convolution kernel for the OOF matrix is
            #   F{K_ij}(ω) = (4π r · j_1(r|ω|) / |ω|²) · ω_i ω_j .
            # The integration with F{∂_i I} = i ω_i F{I} contributes one factor
            # of ω_i / |ω| through the matrix-vector contraction, giving the
            # final per-component scaling below.
            scaling = (4.0 * np.pi * r) * (j1 / (omega_safe * omega_safe))
            # Kill the DC component to avoid the (1/|ω|²) singularity at ω=0.
            scaling = np.where(omega > self.eps, scaling, np.float32(0.0))

            # Six independent components of the symmetric OOF matrix Q.
            # Each requires one inverse FFT.
            comps: dict[tuple[int, int], NDArray[np.float32]] = {}
            pairs = [
                (0, 0, ox, ox),
                (1, 1, oy, oy),
                (2, 2, oz, oz),
                (0, 1, ox, oy),
                (0, 2, ox, oz),
                (1, 2, oy, oz),
            ]
            for i, j, ai, aj in pairs:
                kernel = (scaling * ai * aj).astype(np.complex64)
                qij = np.real(ifftn(spectrum * kernel, norm="backward")).astype(
                    np.float32
                )
                comps[(i, j)] = qij

            # Build per-voxel symmetric 3 × 3 matrices and eigendecompose.
            mat = np.empty(shape + (3, 3), dtype=np.float32)
            mat[..., 0, 0] = comps[(0, 0)]
            mat[..., 1, 1] = comps[(1, 1)]
            mat[..., 2, 2] = comps[(2, 2)]
            mat[..., 0, 1] = mat[..., 1, 0] = comps[(0, 1)]
            mat[..., 0, 2] = mat[..., 2, 0] = comps[(0, 2)]
            mat[..., 1, 2] = mat[..., 2, 1] = comps[(1, 2)]
            del comps

            evs = np.linalg.eigvalsh(mat).astype(np.float32)  # ascending
            del mat
            l_a = evs[..., 0]
            l_b = evs[..., 1]
            l_c = evs[..., 2]

            # Scale-normalise by the sphere area (Law-Chung 2008 eq. 16). Without
            # this, the response grows ~r² and the multi-scale max collapses to
            # the largest radius.
            area_norm = np.float32(1.0 / (4.0 * np.pi * float(r) * float(r)))
            if self.black_ridges:
                response = (
                    np.maximum(0.0, -(l_a + l_b) * np.float32(0.5)) * area_norm
                )
            else:
                response = (
                    np.maximum(0.0, (l_b + l_c) * np.float32(0.5)) * area_norm
                )

            np.maximum(best, response, out=best)
            logger.debug(
                "OOF r=%.3f vox: response p99=%.4g", float(r), float(np.percentile(response, 99))
            )

        return best

    # ------------------------------------------------------------------ predict

    def predict(self, x: VesselInput) -> VesselOutput:
        swi = x.swi
        brain = (x.brain_mask > 0).astype(bool)
        if brain.shape != swi.array.shape:
            raise ValueError(
                f"brain mask shape {brain.shape} != SWI shape {swi.array.shape}"
            )

        img = swi.array.astype(np.float32, copy=False)
        normalised = self._normalize(img, brain)

        radii = self._radii_voxels(swi.spacing_mm)
        logger.info(
            "OOF: pid=%s radii(vox)=%s black_ridges=%s",
            x.patient_id,
            np.round(radii, 3).tolist(),
            self.black_ridges,
        )

        raw = self._oof_multiscale(normalised, radii)
        raw *= brain.astype(np.float32)
        peak = float(raw.max())
        soft = raw / (peak + self.eps) if peak > 0 else raw

        binary = ((soft >= self.threshold) & brain).astype(np.uint8)

        params: dict[str, Any] = {
            "sigma_min_mm": self.sigma_min_mm,
            "sigma_max_mm": self.sigma_max_mm,
            "sigma_steps": self.sigma_steps,
            "black_ridges": self.black_ridges,
            "threshold": self.threshold,
            "normalize": self.normalize,
            "percentile_clip": list(self.percentile_clip),
            "eps": self.eps,
            "voxel_spacing_mm": list(swi.spacing_mm),
            "radii_voxel": [float(v) for v in radii],
            "raw_response_peak": peak,
        }

        self._validate_output(soft, binary, swi)

        return VesselOutput(
            soft=soft.astype(np.float32),
            binary=binary,
            affine=swi.affine.copy(),
            threshold=self.threshold,
            params=params,
        )
