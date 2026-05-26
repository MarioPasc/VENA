"""Correctness tests for :class:`NAWMNormalizedCellularityModel`.

Tests:

1. **Channel contract** — both ``adc_rel`` and ``cell`` present, float32, shape
   match, ``cell`` ∈ ``[0, 1]``, ``adc_rel`` ≥ 0.
2. **Cell channel is zero outside the tumour mask** — tumour-gated by design.
3. **Restricted-diffusion regions yield ``cell`` near 1** — when the tumour
   ADC is much lower than the NAWM reference.
4. **Free-diffusion (necrosis) regions yield ``cell`` near 0** — high ADC
   inside tumour should give ``cell`` close to 0.
5. **NAWM identity** — when ADC equals the NAWM reference everywhere, the
   ``adc_rel`` channel returns 1 across the brain and ``cell`` returns 0.5
   inside the tumour (sigmoid at zero argument).
6. **Binary mask is the tumour mask** — for the QC collage contour.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vena.data.niigz import NiftiVolume
from vena.prior_maps.cellularity_priors.abc_model import CellularityInput
from vena.prior_maps.cellularity_priors.models import (
    NAWMNormalizedCellularityModel,
)


def _make_volume(arr: np.ndarray) -> NiftiVolume:
    return NiftiVolume(
        array=arr.astype(np.float32),
        affine=np.eye(4),
        header=None,
        path=Path("/synthetic/adc.nii.gz"),
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
    adc = np.zeros(shape, dtype=np.float32)
    adc[brain > 0] = 1.0e-3  # uniform NAWM-like ADC
    return adc, brain, parenchyma, tumour


def test_output_contract(synthetic_inputs):
    adc_arr, brain, parenchyma, tumour = synthetic_inputs
    out = NAWMNormalizedCellularityModel().predict(
        CellularityInput(
            adc=_make_volume(adc_arr),
            brain_mask=brain,
            parenchyma_mask=parenchyma,
            tumour_mask=tumour,
            patient_id="synth",
        )
    )
    assert set(out.channels) >= {"adc_rel", "cell"}
    for arr in out.channels.values():
        assert arr.shape == adc_arr.shape
        assert arr.dtype == np.float32
        assert np.isfinite(arr).all()
    assert out.channels["cell"].min() >= -1e-6
    assert out.channels["cell"].max() <= 1.0 + 1e-6
    assert out.channels["adc_rel"].min() >= -1e-6
    assert out.binary is not None and out.binary.dtype == np.uint8


def test_cell_is_zero_outside_tumour(synthetic_inputs):
    adc_arr, brain, parenchyma, tumour = synthetic_inputs
    out = NAWMNormalizedCellularityModel().predict(
        CellularityInput(
            adc=_make_volume(adc_arr),
            brain_mask=brain,
            parenchyma_mask=parenchyma,
            tumour_mask=tumour,
            patient_id="synth",
        )
    )
    outside_tumour = tumour == 0
    assert np.allclose(out.channels["cell"][outside_tumour], 0.0)


def test_restricted_diffusion_yields_high_cell(synthetic_inputs):
    adc_arr, brain, parenchyma, tumour = synthetic_inputs
    adc_arr = adc_arr.copy()
    # Restrict diffusion strongly inside the tumour: ADC drops to 30% of NAWM.
    # sigmoid((1 - 0.3) * adc_nawm / (0.2 * adc_nawm)) = sigmoid(3.5) ≈ 0.97
    adc_arr[tumour > 0] = 0.3e-3
    out = NAWMNormalizedCellularityModel(sigma_adc_fraction=0.2).predict(
        CellularityInput(
            adc=_make_volume(adc_arr),
            brain_mask=brain,
            parenchyma_mask=parenchyma,
            tumour_mask=tumour,
            patient_id="synth",
        )
    )
    interior = out.channels["cell"][tumour > 0]
    assert interior.mean() > 0.9


def test_necrosis_yields_low_cell(synthetic_inputs):
    adc_arr, brain, parenchyma, tumour = synthetic_inputs
    adc_arr = adc_arr.copy()
    adc_arr[tumour > 0] = 2.5e-3  # necrotic-like high ADC inside tumour
    out = NAWMNormalizedCellularityModel(sigma_adc_fraction=0.2).predict(
        CellularityInput(
            adc=_make_volume(adc_arr),
            brain_mask=brain,
            parenchyma_mask=parenchyma,
            tumour_mask=tumour,
            patient_id="synth",
        )
    )
    interior = out.channels["cell"][tumour > 0]
    assert interior.mean() < 0.1


def test_nawm_identity(synthetic_inputs):
    adc_arr, brain, parenchyma, tumour = synthetic_inputs
    out = NAWMNormalizedCellularityModel(sigma_adc_fraction=0.2).predict(
        CellularityInput(
            adc=_make_volume(adc_arr),
            brain_mask=brain,
            parenchyma_mask=parenchyma,
            tumour_mask=tumour,
            patient_id="synth",
        )
    )
    nawm = (parenchyma > 0) & (tumour == 0)
    np.testing.assert_allclose(out.channels["adc_rel"][nawm].mean(), 1.0, atol=1e-5)
    # sigmoid(0) = 0.5 inside the tumour (ADC equals NAWM reference there too)
    np.testing.assert_allclose(out.channels["cell"][tumour > 0].mean(), 0.5, atol=1e-5)


def test_binary_matches_tumour_mask(synthetic_inputs):
    adc_arr, brain, parenchyma, tumour = synthetic_inputs
    out = NAWMNormalizedCellularityModel().predict(
        CellularityInput(
            adc=_make_volume(adc_arr),
            brain_mask=brain,
            parenchyma_mask=parenchyma,
            tumour_mask=tumour,
            patient_id="synth",
        )
    )
    np.testing.assert_array_equal(out.binary, tumour)
