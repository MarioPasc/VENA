"""Correctness tests for :class:`Alsop2015PerfusionModel`.

The model performs NAWM-median normalisation of the CBF map followed by a
tanh squash. The tests verify:

1. **Channel contract** — both ``cbf_rel`` and ``cbf`` present, float32, shape
   match, finite, ``cbf`` in ``[-1, 1]``.
2. **NAWM identity** — when CBF is uniformly equal to the NAWM reference, the
   ``cbf_rel`` channel returns 1 in the parenchyma and ``cbf`` returns
   ``tanh(1/3)`` ≈ 0.32.
3. **Tumour exclusion from NAWM** — putting an outlier patch in the tumour
   region must NOT shift the NAWM reference (since NAWM = parenchyma − tumour).
4. **Brain masking** — voxels outside the brain mask must be 0 in both channels.
5. **Range bounds** — high-CBF region (5× NAWM) yields ``cbf_rel ≈ 5`` and
   ``cbf ≈ tanh(5/3) ≈ 0.927``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vena.data.niigz import NiftiVolume
from vena.prior_maps.perfusion_priors.abc_model import PerfusionInput
from vena.prior_maps.perfusion_priors.models import Alsop2015PerfusionModel


def _make_volume(arr: np.ndarray) -> NiftiVolume:
    return NiftiVolume(
        array=arr.astype(np.float32),
        affine=np.eye(4),
        header=None,
        path=Path("/synthetic/asl.nii.gz"),
        spacing_mm=(1.0, 1.0, 1.0),
    )


@pytest.fixture
def synthetic_inputs():
    shape = (32, 32, 32)
    brain = np.zeros(shape, dtype=np.uint8)
    brain[4:-4, 4:-4, 4:-4] = 1
    parenchyma = brain.copy()
    tumour = np.zeros(shape, dtype=np.uint8)
    tumour[12:20, 12:20, 12:20] = 1
    # ASL: uniform CBF = 50 ml/100g/min everywhere inside the brain.
    asl = np.zeros(shape, dtype=np.float32)
    asl[brain > 0] = 50.0
    return asl, brain, parenchyma, tumour


def test_output_contract(synthetic_inputs):
    asl_arr, brain, parenchyma, tumour = synthetic_inputs
    model = Alsop2015PerfusionModel()
    out = model.predict(
        PerfusionInput(
            asl=_make_volume(asl_arr),
            brain_mask=brain,
            parenchyma_mask=parenchyma,
            tumour_mask=tumour,
            patient_id="synth-0001",
        )
    )
    assert set(out.channels) >= {"cbf_rel", "cbf"}
    for arr in out.channels.values():
        assert arr.shape == asl_arr.shape
        assert arr.dtype == np.float32
        assert np.isfinite(arr).all()
    assert out.channels["cbf"].min() >= -1.0 - 1e-6
    assert out.channels["cbf"].max() <= 1.0 + 1e-6
    assert out.binary is None


def test_nawm_identity_yields_unit_relative(synthetic_inputs):
    asl_arr, brain, parenchyma, tumour = synthetic_inputs
    model = Alsop2015PerfusionModel(squash_const=3.0)
    out = model.predict(
        PerfusionInput(
            asl=_make_volume(asl_arr),
            brain_mask=brain,
            parenchyma_mask=parenchyma,
            tumour_mask=tumour,
            patient_id="synth",
        )
    )
    nawm = (parenchyma > 0) & (tumour == 0)
    np.testing.assert_allclose(out.channels["cbf_rel"][nawm].mean(), 1.0, rtol=0, atol=1e-5)
    np.testing.assert_allclose(
        out.channels["cbf"][nawm].mean(), np.tanh(1.0 / 3.0), rtol=0, atol=1e-5
    )


def test_tumour_outliers_do_not_shift_nawm_reference(synthetic_inputs):
    asl_arr, brain, parenchyma, tumour = synthetic_inputs
    # Put a huge CBF outlier inside the tumour. Because NAWM = parenchyma − tumour,
    # the reference median must be unaffected.
    asl_arr = asl_arr.copy()
    asl_arr[tumour > 0] = 5000.0
    model = Alsop2015PerfusionModel()
    out = model.predict(
        PerfusionInput(
            asl=_make_volume(asl_arr),
            brain_mask=brain,
            parenchyma_mask=parenchyma,
            tumour_mask=tumour,
            patient_id="synth",
        )
    )
    assert pytest.approx(out.params["cbf_nawm_reference"], rel=1e-6) == 50.0


def test_outside_brain_is_zero(synthetic_inputs):
    asl_arr, brain, parenchyma, tumour = synthetic_inputs
    model = Alsop2015PerfusionModel()
    out = model.predict(
        PerfusionInput(
            asl=_make_volume(asl_arr),
            brain_mask=brain,
            parenchyma_mask=parenchyma,
            tumour_mask=tumour,
            patient_id="synth",
        )
    )
    outside = brain == 0
    assert np.allclose(out.channels["cbf_rel"][outside], 0.0)
    assert np.allclose(out.channels["cbf"][outside], 0.0)


def test_high_cbf_region_saturates_appropriately(synthetic_inputs):
    asl_arr, brain, parenchyma, tumour = synthetic_inputs
    asl_arr = asl_arr.copy()
    asl_arr[16:20, 16:20, 16:20] = 250.0  # 5× the NAWM value (50)
    model = Alsop2015PerfusionModel(squash_const=3.0)
    out = model.predict(
        PerfusionInput(
            asl=_make_volume(asl_arr),
            brain_mask=brain,
            parenchyma_mask=parenchyma,
            tumour_mask=tumour,
            patient_id="synth",
        )
    )
    # Inside the high-CBF region cbf_rel should be ~5 and cbf ~ tanh(5/3).
    hi = (slice(17, 19), slice(17, 19), slice(17, 19))  # interior of the patch
    np.testing.assert_allclose(out.channels["cbf_rel"][hi].mean(), 5.0, atol=1e-5)
    np.testing.assert_allclose(out.channels["cbf"][hi].mean(), np.tanh(5.0 / 3.0), atol=1e-5)
