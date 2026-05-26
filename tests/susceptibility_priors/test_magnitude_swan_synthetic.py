"""Correctness tests for :class:`MagnitudeSwanSusceptibilityModel`.

Tests:

1. **Channel contract** — both ``sus`` and ``itss`` present, float32, shape
   match, both non-negative.
2. **Sus lights up dark voxels** — a synthetic dark patch on a bright
   background produces a higher ``sus`` value at the patch than elsewhere.
3. **ITSS is zero outside the tumour mask** — tumour-gated by design.
4. **ITSS responds to dark foci inside the tumour** — a sub-region with
   lowest 10% of in-tumour intensities triggers non-zero smoothed ITSS at
   those voxels.
5. **Empty tumour mask** — ITSS channel returns all zeros and the model logs
   a warning (we do not assert on the log; just non-crash).
6. **Sus background is approximately zero** — over a bright uniform patch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vena.data.niigz import NiftiVolume
from vena.prior_maps.susceptibility_priors.abc_model import SusceptibilityInput
from vena.prior_maps.susceptibility_priors.models import (
    MagnitudeSwanSusceptibilityModel,
)


def _make_volume(arr: np.ndarray) -> NiftiVolume:
    return NiftiVolume(
        array=arr.astype(np.float32),
        affine=np.eye(4),
        header=None,
        path=Path("/synthetic/swi.nii.gz"),
        spacing_mm=(1.0, 1.0, 1.0),
    )


@pytest.fixture
def synthetic_inputs():
    shape = (32, 32, 32)
    brain = np.zeros(shape, dtype=np.uint8)
    brain[4:-4, 4:-4, 4:-4] = 1
    tumour = np.zeros(shape, dtype=np.uint8)
    tumour[12:20, 12:20, 12:20] = 1
    # Bright background SWAN ~ 1000; ITSS-like dark foci ~ 50 inside the tumour.
    swi = np.zeros(shape, dtype=np.float32)
    swi[brain > 0] = 1000.0
    return swi, brain, tumour


def test_output_contract(synthetic_inputs):
    swi_arr, brain, tumour = synthetic_inputs
    out = MagnitudeSwanSusceptibilityModel().predict(
        SusceptibilityInput(
            swi=_make_volume(swi_arr),
            brain_mask=brain,
            tumour_mask=tumour,
            patient_id="synth",
        )
    )
    assert set(out.channels) >= {"sus", "itss"}
    for arr in out.channels.values():
        assert arr.shape == swi_arr.shape
        assert arr.dtype == np.float32
        assert np.isfinite(arr).all()
        assert arr.min() >= -1e-6
    assert out.binary is not None and out.binary.dtype == np.uint8


def test_sus_lights_dark_patch():
    shape = (32, 32, 32)
    brain = np.ones(shape, dtype=np.uint8)
    tumour = np.zeros(shape, dtype=np.uint8)
    swi = np.full(shape, 1000.0, dtype=np.float32)
    swi[14:18, 14:18, 14:18] = 50.0  # dark patch (vein-like)
    out = MagnitudeSwanSusceptibilityModel(sigma_mm=1.0).predict(
        SusceptibilityInput(
            swi=_make_volume(swi),
            brain_mask=brain,
            tumour_mask=tumour,
            patient_id="synth",
        )
    )
    dark = out.channels["sus"][15, 15, 15]
    bright = out.channels["sus"][6, 6, 6]
    assert dark > 0.5
    assert bright < 0.05


def test_itss_is_zero_outside_tumour(synthetic_inputs):
    swi_arr, brain, tumour = synthetic_inputs
    swi_arr = swi_arr.copy()
    swi_arr[14:17, 14:17, 14:17] = 50.0  # dark foci inside the tumour
    out = MagnitudeSwanSusceptibilityModel().predict(
        SusceptibilityInput(
            swi=_make_volume(swi_arr),
            brain_mask=brain,
            tumour_mask=tumour,
            patient_id="synth",
        )
    )
    outside_tumour = tumour == 0
    assert np.all(out.channels["itss"][outside_tumour] == 0)


def test_itss_responds_to_in_tumour_dark_foci(synthetic_inputs):
    swi_arr, brain, tumour = synthetic_inputs
    swi_arr = swi_arr.copy()
    # Tumour = 8^3 = 512 voxels. Use a sparse dark patch (< 10% of tumour) so
    # the in-tumour q10 cut-off lies strictly between dark and bright voxels.
    # Patch 2*4*4 = 32 voxels ≈ 6.25% of the tumour.
    swi_arr[12:14, 14:18, 14:18] = 50.0
    out = MagnitudeSwanSusceptibilityModel(sigma_mm=1.0).predict(
        SusceptibilityInput(
            swi=_make_volume(swi_arr),
            brain_mask=brain,
            tumour_mask=tumour,
            patient_id="synth",
        )
    )
    # Some ITSS density should appear at the dark-foci voxels.
    inner = out.channels["itss"][13, 16, 16]
    assert inner > 0.1


def test_empty_tumour_mask_yields_zero_itss():
    shape = (32, 32, 32)
    brain = np.ones(shape, dtype=np.uint8)
    tumour = np.zeros(shape, dtype=np.uint8)
    swi = np.full(shape, 1000.0, dtype=np.float32)
    out = MagnitudeSwanSusceptibilityModel().predict(
        SusceptibilityInput(
            swi=_make_volume(swi),
            brain_mask=brain,
            tumour_mask=tumour,
            patient_id="synth",
        )
    )
    assert np.allclose(out.channels["itss"], 0.0)


def test_bright_uniform_region_has_low_sus():
    shape = (32, 32, 32)
    brain = np.ones(shape, dtype=np.uint8)
    tumour = np.zeros(shape, dtype=np.uint8)
    swi = np.full(shape, 1000.0, dtype=np.float32)
    out = MagnitudeSwanSusceptibilityModel().predict(
        SusceptibilityInput(
            swi=_make_volume(swi),
            brain_mask=brain,
            tumour_mask=tumour,
            patient_id="synth",
        )
    )
    # No structure, no contrast → percentile-norm + (1 - .) maps everything to
    # the same low value; sus stays close to zero.
    assert out.channels["sus"].max() < 0.05
