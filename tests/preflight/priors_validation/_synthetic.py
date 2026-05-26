"""Shared synthetic-input fixtures for the validation-routine unit tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from vena.data.niigz import NiftiVolume
from vena.preflight.priors_validation.core.dataclasses import (
    SubjectInputs,
    SubjectMetadata,
)


def _mk_vol(arr: NDArray) -> NiftiVolume:
    return NiftiVolume(
        array=arr.astype(np.float32),
        affine=np.eye(4),
        header=None,
        path=Path("/synthetic/test.nii.gz"),
        spacing_mm=(1.0, 1.0, 1.0),
    )


def build_synthetic_subject(
    *,
    shape: tuple[int, int, int] = (32, 32, 32),
    cbf_nawm: float = 22.0,
    cbf_gm: float = 55.0,
    cbf_csf: float = 1.0,
    adc_nawm_mm2_s: float = 0.8e-3,
    adc_gm_mm2_s: float = 0.85e-3,
    adc_csf_mm2_s: float = 3.0e-3,
    delta_strength: float = 1.0,
    seed: int = 0,
) -> SubjectInputs:
    """Build a synthetic subject with known physical ground-truth maps.

    Layout: brain mask occupies the inner 24³ cube; cortex is a 4-voxel shell;
    NAWM is the next 8 voxels in; ventricles are the inner 8³ core. Tumour is
    a 6³ block placed near the cortex/NAWM boundary.
    """
    rng = np.random.default_rng(seed)
    brain = np.zeros(shape, dtype=np.uint8)
    brain[4:-4, 4:-4, 4:-4] = 1
    # ventricles core
    ventricles = np.zeros(shape, dtype=np.uint8)
    ventricles[12:-12, 12:-12, 12:-12] = 1
    # cortex shell = brain minus 2-voxel inner cube
    inner = np.zeros(shape, dtype=np.uint8)
    inner[6:-6, 6:-6, 6:-6] = 1
    cortex = brain & (~inner.astype(bool))
    # white matter = inner minus ventricles
    wm = inner & (~ventricles.astype(bool))
    # parenchyma = cortex + wm (everything not ventricles inside brain)
    parenchyma = ((cortex.astype(bool) | wm.astype(bool)) & brain.astype(bool)).astype(np.uint8)
    tumour = np.zeros(shape, dtype=np.uint8)
    tumour[8:14, 8:14, 8:14] = 1
    # ----- CBF map ml/100g/min -----
    cbf = np.zeros(shape, dtype=np.float32)
    cbf[cortex.astype(bool)] = cbf_gm
    cbf[wm.astype(bool)] = cbf_nawm
    cbf[ventricles.astype(bool)] = cbf_csf
    cbf[tumour > 0] = cbf_gm * 1.5
    cbf += rng.normal(scale=2.0, size=shape).astype(np.float32)
    cbf = np.clip(cbf, 0.0, None) * (brain > 0)
    # ----- ADC map mm²/s -----
    adc = np.zeros(shape, dtype=np.float32)
    adc[cortex.astype(bool)] = adc_gm_mm2_s
    adc[wm.astype(bool)] = adc_nawm_mm2_s
    adc[ventricles.astype(bool)] = adc_csf_mm2_s
    adc[tumour > 0] = 0.6e-3  # restricted inside tumour
    # Scale the additive noise to the signal magnitude so the synthetic
    # remains realistic across the full range of ``adc_nawm_mm2_s`` values
    # we feed in (including the 1e-9 quirk-detector test case).
    adc_noise_scale = 0.05 * float(adc_nawm_mm2_s)
    adc += rng.normal(scale=adc_noise_scale, size=shape).astype(np.float32)
    adc = np.clip(adc, 0.0, None) * (brain > 0)
    # ----- T1pre / T1Gd -----
    t1pre = (rng.normal(loc=600.0, scale=50.0, size=shape).astype(np.float32)) * (brain > 0)
    # Enhancement strongest in tumour, also slightly in sinus (we don't model
    # a separate sinus region in the synthetic subject — keep it simple).
    t1gd = t1pre + delta_strength * 200.0 * (tumour > 0).astype(np.float32)
    # ----- derived priors (CBF squashed, cell, sus, itss, etc.) -----
    cbf_nawm_obs = float(np.median(cbf[wm.astype(bool) & (tumour == 0)]))
    cbf_rel = (cbf / max(cbf_nawm_obs, 1e-6)).astype(np.float32) * (brain > 0)
    cbf_chan = np.tanh(cbf_rel / 3.0).astype(np.float32) * (brain > 0)
    cell = (
        (tumour > 0).astype(np.float32) * 0.85
        + rng.normal(scale=0.05, size=shape).astype(np.float32)
    ) * (tumour > 0)
    sus = (rng.uniform(0, 0.1, size=shape).astype(np.float32)) * (brain > 0)
    itss = ((tumour > 0).astype(np.float32) * 0.4) * (tumour > 0)
    adc_rel = (adc / max(float(np.median(adc[wm.astype(bool) & (tumour == 0)])), 1e-9)).astype(
        np.float32
    ) * (brain > 0)
    derived = {
        "cbf_rel": _mk_vol(cbf_rel),
        "cbf": _mk_vol(cbf_chan),
        "cell": _mk_vol(cell),
        "sus": _mk_vol(sus),
        "itss": _mk_vol(itss),
        "adc_rel": _mk_vol(adc_rel),
    }
    meta = SubjectMetadata(
        subject_id="SYNTH-0001",
        age=55.0,
        sex="M",
        scanner="synthetic",
        field_strength_t=3.0,
        pathology="glioblastoma",
        who_grade=4,
    )
    return SubjectInputs(
        subject_id="SYNTH-0001",
        t1pre=_mk_vol(t1pre),
        t1gd=_mk_vol(t1gd),
        brain_mask=_mk_vol(brain.astype(np.float32)),
        parenchyma_mask=_mk_vol(parenchyma.astype(np.float32)),
        tumour_mask=_mk_vol(tumour.astype(np.float32)),
        cbf=_mk_vol(cbf),
        adc=_mk_vol(adc),
        chi=None,
        swan_mag=None,
        derived_priors=derived,
        metadata=meta,
    )


def build_synthetic_context_from(subject: SubjectInputs):
    """Build a TestContext bypassing atlas registration — for unit tests only.

    Atlas labels are filled with synthetic HO-subcortical labels so that the
    range/T2 ROI lookups resolve. NAWM mask = wm region; ventricle mask =
    ventricles region.
    """
    from vena.preflight.priors_validation.atlases.registry import HO_SUB, VENOUS_INHOUSE
    from vena.preflight.priors_validation.preprocessing import robust_zscore
    from vena.preflight.priors_validation.tests.base import TestContext

    shape = subject.t1pre.array.shape
    brain = np.asarray(subject.brain_mask.array) > 0
    parenchyma = np.asarray(subject.parenchyma_mask.array) > 0
    ventricles = brain & (~parenchyma)

    # Build a HO-subcortical-like label map: 1 = WM, 2 = cortex, 3 = vent
    ho_sub = np.zeros(shape, dtype=np.int32)
    inner = np.zeros(shape, dtype=bool)
    inner[6:-6, 6:-6, 6:-6] = True
    cortex = brain & (~inner)
    wm = inner & (~ventricles)
    ho_sub[wm] = 1
    ho_sub[cortex] = 2
    ho_sub[ventricles] = 3

    # Venous in-house: a small superior strip (rows near z=top) as fake sinus
    venous = np.zeros(shape, dtype=np.int32)
    venous[14:18, 14:18, -8:-4] = 1

    delta_t1 = (
        robust_zscore(np.asarray(subject.t1gd.array), brain)
        - robust_zscore(np.asarray(subject.t1pre.array), brain)
    ).astype(np.float32)

    return TestContext(
        subject=subject,
        atlas_labels={HO_SUB: ho_sub, VENOUS_INHOUSE: venous},
        nawm_mask=wm,
        ventricle_mask=ventricles,
        delta_t1=delta_t1,
        atlas_registration_dice=0.95,
    )
