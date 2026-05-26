"""T3 (T1Gd coherence) against synthetic ground-truth subjects.

Covers spec §9.1 cases: pure noise → |ρ| small; perfect predictor → ρ ≈ 1;
inverted sign → ``wrong_sign`` failure.

These tests exercise the ``sus`` channel rather than ``cbf`` because the
synthetic subject populates the raw ``cbf`` slot with a non-correlated map
and T3's ``_prior_volume_for("cbf", sub)`` would pick that up first.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vena.data.niigz import NiftiVolume
from vena.preflight.priors_validation.tests import T3T1GdCoherence

from ._synthetic import build_synthetic_context_from, build_synthetic_subject


def _outcomes_for(outs, prior, roi):
    for o in outs:
        if o.prior_id == prior and o.roi_id == roi:
            return o
    return None


def _patch_channel(subj, name, arr):
    subj.derived_priors[name] = NiftiVolume(
        array=arr.astype(np.float32),
        affine=np.eye(4),
        header=None,
        path=Path(f"/synthetic/{name}.nii.gz"),
        spacing_mm=(1.0, 1.0, 1.0),
    )


def test_t3_perfect_predictor_yields_rho_close_to_one():
    subj = build_synthetic_subject(delta_strength=1.0)
    ctx = build_synthetic_context_from(subj)
    sus = np.asarray(subj.derived_priors["sus"].array).copy()
    tumour_mask = np.asarray(subj.tumour_mask.array) > 0
    sus[tumour_mask] = np.asarray(ctx.delta_t1)[tumour_mask]
    _patch_channel(subj, "sus", sus)
    outs = list(T3T1GdCoherence(n_boot=50, seed=0).run(subj, ctx))
    o = _outcomes_for(outs, "sus", "tum")
    assert o is not None and o.metric_value is not None, "missing (sus, tum)"
    assert o.metric_value > 0.9, f"expected ρ → 1, got {o.metric_value}"


def test_t3_pure_noise_yields_small_rho():
    subj = build_synthetic_subject()
    ctx = build_synthetic_context_from(subj)
    rng = np.random.default_rng(0)
    noise = rng.uniform(-1, 1, size=subj.t1pre.array.shape).astype(np.float32)
    _patch_channel(subj, "sus", noise)
    outs = list(T3T1GdCoherence(n_boot=50, seed=0).run(subj, ctx))
    o = _outcomes_for(outs, "sus", "tum")
    assert o is not None and o.metric_value is not None
    assert abs(o.metric_value) < 0.2, f"pure noise should yield |ρ| ≲ 0.15, got {o.metric_value}"


def test_t3_inverted_sign_flags_wrong_sign():
    subj = build_synthetic_subject(delta_strength=1.0)
    ctx = build_synthetic_context_from(subj)
    sus = np.asarray(subj.derived_priors["sus"].array).copy()
    tumour_mask = np.asarray(subj.tumour_mask.array) > 0
    sus[tumour_mask] = -np.asarray(ctx.delta_t1)[tumour_mask]
    _patch_channel(subj, "sus", sus)
    outs = list(T3T1GdCoherence(n_boot=50, seed=0).run(subj, ctx))
    o = _outcomes_for(outs, "sus", "tum")
    assert o is not None and o.metric_value is not None
    assert o.metric_value < -0.8, f"expected ρ ≈ −1, got {o.metric_value}"
    assert (o.extras or {}).get("failure_mode") == "wrong_sign", (
        f"expected wrong_sign, got {(o.extras or {}).get('failure_mode')}"
    )
    assert not o.passed
