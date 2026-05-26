"""Correctness tests for :class:`vena.vessel_priors.models.OOFVesselModel`.

These tests do not require ground-truth labels. They rely on analytical
properties of the Law & Chung 2008 OOF response:

1. **Peak radius** — for an infinite cylinder of radius ``a``, the OOF response
   at the cylinder centre peaks when the sphere radius ``r`` matches ``a``.
   See Law-Chung 2008 Fig. 3.
2. **Orientation invariance** — OOF is rotation-equivariant by construction
   (the kernel only depends on the spectral magnitude and the outer product
   ``ω ω^T``, both of which are rotation-equivariant). The coefficient of
   variation of the on-axis response across orientations should be small.
3. **Adjacency separation** — two parallel cylinders at separation
   ``d > 2 a`` should produce two distinct response peaks (the published
   reason OOF beats Frangi near deep medullary veins; Bériault 2015 §IV.D).
4. **Sign convention** — ``black_ridges=True`` lights up dark tubes and
   suppresses bright tubes.
5. **Translation equivariance** — shifting the input shifts the response
   identically (within FFT wraparound effects near the boundary).
"""

from __future__ import annotations

import numpy as np
import pytest

from vena.data.niigz import NiftiVolume
from vena.preflight.vessel_mask import (
    cylinder_volume,
    parallel_cylinders_volume,
    rotated_cylinder_volume,
    smooth_cylinder_volume,
)
from vena.prior_maps.vessel_priors.abc_model import VesselInput
from vena.prior_maps.vessel_priors.models import OOFVesselModel


def _run_oof(
    image: np.ndarray,
    radii_mm: list[float],
    *,
    black_ridges: bool = True,
    normalize: str = "minmax",
) -> np.ndarray:
    """Run OOF on ``image`` with brain_mask = ones; returns the soft response.

    ``predict`` divides by the per-volume peak so its output is always in
    ``[0, 1]``. This is what production code consumes and what we test for
    sign / orientation / adjacency / translation properties.
    """
    H, W, D = image.shape
    vol = NiftiVolume(
        array=image.astype(np.float32),
        affine=np.eye(4),
        header=None,
        path=None,
        spacing_mm=(1.0, 1.0, 1.0),
    )
    brain = np.ones((H, W, D), dtype=np.uint8)
    model = OOFVesselModel(
        sigma_min_mm=float(min(radii_mm)),
        sigma_max_mm=float(max(radii_mm)),
        sigma_steps=len(radii_mm),
        black_ridges=black_ridges,
        threshold=0.5,
        normalize=normalize,
    )
    out = model.predict(
        VesselInput(swi=vol, brain_mask=brain, patient_id="synth")
    )
    return out.soft


def _run_oof_raw(
    image: np.ndarray,
    radius_vox: float,
    *,
    black_ridges: bool = True,
) -> np.ndarray:
    """Compute the *un-normalised* multi-scale OOF response at a single radius.

    For peak-radius diagnostics we need the raw response; the model's
    ``predict`` divides by the per-volume peak, which saturates the on-tube
    voxel to ``1.0`` at every single-scale call and masks the peak structure.
    Calling :meth:`OOFVesselModel._oof_multiscale` directly skips the
    normalisation and gives us the eigenvalue-derived OOF response that the
    Law-Chung 2008 analytical curve is defined on.
    """
    model = OOFVesselModel(
        sigma_min_mm=radius_vox,
        sigma_max_mm=radius_vox,
        sigma_steps=1,
        black_ridges=black_ridges,
        threshold=0.5,
        normalize="minmax",
    )
    return model._oof_multiscale(image.astype(np.float32), np.asarray([float(radius_vox)]))


# ---------------------------------------------------------------- Peak radius


@pytest.mark.unit
@pytest.mark.preflight_vessel
@pytest.mark.parametrize("a_mm", [2.5, 3.0, 3.5])
def test_oof_peak_at_cylinder_radius(a_mm: float) -> None:
    """Raw OOF response at cylinder centre peaks when sphere radius ≈ cylinder radius.

    We test against the un-normalised :meth:`OOFVesselModel._oof_multiscale`
    output because the model's ``predict`` rescales each volume so the peak
    is exactly ``1.0`` — which destroys the per-scale peak structure that
    Law & Chung 2008 Fig. 3 describes.

    The cylinder is anti-aliased via :func:`smooth_cylinder_volume`. A hard
    binary cylinder on a 1 mm grid has effective radius ``a + 0.5`` mm under
    voxel-area integration, which biases the empirical peak by half a voxel
    independent of the OOF kernel.

    Restricted to ``a_mm ≥ 2.5`` (≥ 2.5 × voxel size). At smaller radii the
    cylinder cross-section is under-resolved — Law & Chung 2008 §4.1 require
    ``a ≥ 2 × spacing`` for clean peak recovery; on a 1 mm grid that means the
    peak shifts by up to one voxel for ``a ∈ [1, 2] mm``. The orientation,
    adjacency, and sign tests below cover the small-radius regime via property
    checks rather than direct peak localisation.

    Tolerance: 0.5 mm (two sweep steps). The OOF peak should lock onto the
    nominal cylinder radius to within voxel quantisation.
    """
    size = 56
    img = smooth_cylinder_volume(
        size=size, radius_mm=a_mm, axis=2, transition_mm=0.5
    )
    centre = size // 2
    sweep = np.arange(0.5, 5.01, 0.25)
    responses = []
    for r in sweep:
        raw = _run_oof_raw(img, float(r))
        responses.append(float(raw[centre, centre, centre]))
    responses = np.asarray(responses)
    r_peak = float(sweep[int(np.argmax(responses))])
    assert abs(r_peak - a_mm) <= 0.5, (
        f"Expected peak at r≈{a_mm} mm but found {r_peak} mm. Sweep "
        f"responses: {dict(zip(sweep.tolist(), responses.tolist(), strict=False))}"
    )


# ---------------------------------------------------------------- Orientation invariance


@pytest.mark.unit
@pytest.mark.preflight_vessel
def test_oof_orientation_invariance() -> None:
    """On-axis OOF response is invariant to cylinder orientation within ~10 %."""
    a = 1.5
    radii = [1.0, 1.5, 2.0]
    directions: list[tuple[float, float, float]] = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
    ]
    size = 40
    centre = size // 2
    responses: list[float] = []
    for d in directions:
        img = rotated_cylinder_volume(size=size, radius_mm=a, direction=d)
        soft = _run_oof(img, radii)
        responses.append(float(soft[centre, centre, centre]))
    arr = np.asarray(responses)
    mean = float(arr.mean())
    cv = float(arr.std() / max(mean, 1e-8))
    assert mean > 0.4, f"On-axis response too low (mean={mean:.3f})"
    # OOF is theoretically rotation-equivariant; the residual CV comes from
    # voxelisation of the rotated cylinder (which is not band-limited).
    assert cv < 0.15, (
        f"Orientation CV {cv:.3f} above 0.15 — OOF should be near rotation-"
        f"invariant. Responses: {dict(zip(directions, responses, strict=False))}"
    )


# ---------------------------------------------------------------- Adjacency


@pytest.mark.unit
@pytest.mark.preflight_vessel
def test_oof_adjacency_separates_close_vessels() -> None:
    """Two parallel dark cylinders at ``d = 4 a`` give two distinct peaks.

    The trough at the midpoint must lie strictly below both peaks. This is
    the canonical OOF-vs-Frangi failure mode; Frangi merges the two responses
    at this separation, OOF does not.
    """
    a = 1.0
    d = 4.0 * a
    size = 40
    img = parallel_cylinders_volume(
        size=size, radius_mm=a, separation_mm=d, axis=2, offset_axis=1
    )
    soft = _run_oof(img, [0.5, 1.0, 1.5])
    cx = size // 2
    # Centres of the two cylinders on offset_axis=1, at cx ± d/2.
    half_sep = int(round(d / 2.0))
    peak_a = float(soft[cx, cx + half_sep, cx])
    peak_b = float(soft[cx, cx - half_sep, cx])
    trough = float(soft[cx, cx, cx])
    assert peak_a > 0.5
    assert peak_b > 0.5
    # The trough should be at least 30 % below the weaker peak.
    weaker_peak = min(peak_a, peak_b)
    assert trough < 0.7 * weaker_peak, (
        f"Adjacency separation failed: peak_a={peak_a:.3f} peak_b={peak_b:.3f} "
        f"trough={trough:.3f}; trough must be < 0.7×min(peaks)"
    )


# ---------------------------------------------------------------- Sign convention


@pytest.mark.unit
@pytest.mark.preflight_vessel
def test_oof_sign_convention_black_ridges_only_lights_dark() -> None:
    """``black_ridges=True`` lights up dark cylinders and suppresses bright ones."""
    size = 32
    centre = size // 2
    dark = cylinder_volume(size=size, radius_mm=1.5, background=0.8, foreground=0.1)
    bright = cylinder_volume(size=size, radius_mm=1.5, background=0.1, foreground=0.8)
    soft_dark = _run_oof(dark, [1.0, 1.5], black_ridges=True)
    soft_bright = _run_oof(bright, [1.0, 1.5], black_ridges=True)
    on_dark = float(soft_dark[centre, centre, centre])
    on_bright = float(soft_bright[centre, centre, centre])
    assert on_dark > 0.5
    assert on_bright < 0.05, (
        f"black_ridges=True should suppress bright tubes; got on_bright={on_bright:.3f}"
    )


@pytest.mark.unit
@pytest.mark.preflight_vessel
def test_oof_sign_convention_white_ridges_only_lights_bright() -> None:
    """``black_ridges=False`` lights up bright cylinders and suppresses dark ones."""
    size = 32
    centre = size // 2
    dark = cylinder_volume(size=size, radius_mm=1.5, background=0.8, foreground=0.1)
    bright = cylinder_volume(size=size, radius_mm=1.5, background=0.1, foreground=0.8)
    soft_dark = _run_oof(dark, [1.0, 1.5], black_ridges=False)
    soft_bright = _run_oof(bright, [1.0, 1.5], black_ridges=False)
    assert float(soft_bright[centre, centre, centre]) > 0.5
    assert float(soft_dark[centre, centre, centre]) < 0.05


# ---------------------------------------------------------------- Translation equivariance


@pytest.mark.unit
@pytest.mark.preflight_vessel
def test_oof_translation_equivariance() -> None:
    """Shifting input shifts the response identically (away from the boundary)."""
    a = 1.5
    size = 40
    shift = 3
    img1 = cylinder_volume(size=size, radius_mm=a, axis=2)
    img2 = np.roll(img1, shift=shift, axis=0)
    soft1 = _run_oof(img1, [1.0, 1.5])
    soft2 = _run_oof(img2, [1.0, 1.5])
    soft1_shifted = np.roll(soft1, shift=shift, axis=0)
    # Compare in the interior; FFT is naturally periodic so a circular shift
    # is exactly equivariant, but the input cylinder was clipped at the volume
    # boundary so we still check away from the edges to keep the assertion
    # interpretable.
    interior = (slice(8, size - 8),) * 3
    np.testing.assert_allclose(
        soft2[interior], soft1_shifted[interior], atol=2e-3, rtol=5e-2
    )


# ---------------------------------------------------------------- Smoke test guard


@pytest.mark.unit
@pytest.mark.preflight_vessel
def test_oof_off_tube_response_is_zero() -> None:
    """Response in pure-background regions is essentially zero."""
    size = 32
    img = cylinder_volume(size=size, radius_mm=1.5, axis=2)
    soft = _run_oof(img, [1.0, 1.5])
    # Voxel far from the cylinder axis (corner of the cube).
    off = float(soft[2, 2, 2])
    assert off < 0.05, f"Off-tube response should be ~0, got {off:.3f}"
