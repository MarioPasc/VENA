"""Tests for the lambda_img schedule (vena.model.fm.lpl.schedule)."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from vena.model.fm.lpl import LambdaImgSchedule, compute_lambda_img

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# Validator
# --------------------------------------------------------------------------


def test_constant_default_is_valid() -> None:
    s = LambdaImgSchedule()
    assert s.kind == "constant"
    assert compute_lambda_img(s, 0) == 1.0
    assert compute_lambda_img(s, 100) == 1.0


def test_negative_warmup_rejected() -> None:
    with pytest.raises(ValidationError):
        LambdaImgSchedule(kind="linear", warmup_epochs=-1)


def test_lambda_max_below_min_rejected() -> None:
    with pytest.raises(ValidationError):
        LambdaImgSchedule(lambda_min=1.0, lambda_max=0.5)


def test_negative_lambda_rejected() -> None:
    with pytest.raises(ValidationError):
        LambdaImgSchedule(lambda_min=-0.1, lambda_max=1.0)


def test_cosine_requires_total_epochs() -> None:
    with pytest.raises(ValidationError):
        LambdaImgSchedule(kind="cosine_with_anneal", warmup_epochs=10)


def test_cosine_total_must_exceed_warmup() -> None:
    with pytest.raises(ValidationError):
        LambdaImgSchedule(kind="cosine_with_anneal", warmup_epochs=50, total_epochs=50)


def test_zero_sigmoid_slope_rejected() -> None:
    with pytest.raises(ValidationError):
        LambdaImgSchedule(kind="sigmoid", sigmoid_slope_k=0.0)


# --------------------------------------------------------------------------
# Linear
# --------------------------------------------------------------------------


def test_linear_boundary_endpoints() -> None:
    s = LambdaImgSchedule(kind="linear", warmup_epochs=30, lambda_min=0.0, lambda_max=1.0)
    assert compute_lambda_img(s, 0) == pytest.approx(0.0)
    assert compute_lambda_img(s, 15) == pytest.approx(0.5)
    assert compute_lambda_img(s, 30) == pytest.approx(1.0)
    assert compute_lambda_img(s, 31) == pytest.approx(1.0)
    assert compute_lambda_img(s, 1000) == pytest.approx(1.0)


def test_linear_monotonic_non_decreasing_during_warmup() -> None:
    s = LambdaImgSchedule(kind="linear", warmup_epochs=10, lambda_min=0.0, lambda_max=2.0)
    prev = -1.0
    for e in range(15):
        v = compute_lambda_img(s, e)
        assert v >= prev
        prev = v


def test_linear_zero_warmup_jumps_to_max() -> None:
    s = LambdaImgSchedule(kind="linear", warmup_epochs=0, lambda_min=0.0, lambda_max=3.0)
    assert compute_lambda_img(s, 0) == pytest.approx(3.0)
    assert compute_lambda_img(s, 1000) == pytest.approx(3.0)


def test_negative_epoch_clamped_to_zero() -> None:
    s = LambdaImgSchedule(kind="linear", warmup_epochs=10, lambda_min=0.2, lambda_max=1.0)
    assert compute_lambda_img(s, -5) == pytest.approx(0.2)


# --------------------------------------------------------------------------
# Sigmoid
# --------------------------------------------------------------------------


def test_sigmoid_midpoint_is_average() -> None:
    s = LambdaImgSchedule(
        kind="sigmoid",
        warmup_epochs=20,
        lambda_min=0.0,
        lambda_max=1.0,
        sigmoid_slope_k=0.3,
    )
    # At midpoint = warmup/2 = 10, sigmoid(0) = 0.5 → λ = 0.5 * (max - min) + min
    assert compute_lambda_img(s, 10) == pytest.approx(0.5)


def test_sigmoid_bounded() -> None:
    s = LambdaImgSchedule(kind="sigmoid", warmup_epochs=20, lambda_min=0.3, lambda_max=1.2)
    for e in range(0, 200):
        v = compute_lambda_img(s, e)
        assert v >= s.lambda_min - 1e-9
        assert v <= s.lambda_max + 1e-9


def test_sigmoid_monotonic_non_decreasing() -> None:
    s = LambdaImgSchedule(kind="sigmoid", warmup_epochs=30, sigmoid_slope_k=0.4)
    prev = -1.0
    for e in range(0, 60):
        v = compute_lambda_img(s, e)
        assert v >= prev - 1e-9
        prev = v


# --------------------------------------------------------------------------
# Cosine with anneal
# --------------------------------------------------------------------------


def test_cosine_warmup_then_anneal_shape() -> None:
    s = LambdaImgSchedule(
        kind="cosine_with_anneal",
        warmup_epochs=50,
        lambda_min=0.3,
        lambda_max=1.0,
        total_epochs=1000,
    )
    assert compute_lambda_img(s, 0) == pytest.approx(0.3)
    assert compute_lambda_img(s, 50) == pytest.approx(1.0)
    # Anneal phase reaches lambda_min at total_epochs.
    assert compute_lambda_img(s, 1000) == pytest.approx(0.3)
    # Past total_epochs, value clamps at the anneal end.
    assert compute_lambda_img(s, 2000) == pytest.approx(0.3)


def test_cosine_mid_anneal_is_between_max_and_min() -> None:
    s = LambdaImgSchedule(
        kind="cosine_with_anneal",
        warmup_epochs=50,
        lambda_min=0.0,
        lambda_max=1.0,
        total_epochs=1000,
    )
    mid = compute_lambda_img(s, 525)
    assert 0.0 <= mid <= 1.0
    # Cosine ½(1 + cos(π·475/950)) = ½(1 + cos(π/2)) = 0.5
    assert mid == pytest.approx(0.5, abs=1e-3)


# --------------------------------------------------------------------------
# Constant
# --------------------------------------------------------------------------


def test_constant_ignores_lambda_min() -> None:
    s = LambdaImgSchedule(kind="constant", lambda_min=0.2, lambda_max=1.5)
    assert compute_lambda_img(s, 0) == pytest.approx(1.5)
    assert compute_lambda_img(s, 999) == pytest.approx(1.5)


# --------------------------------------------------------------------------
# Unknown kind defence
# --------------------------------------------------------------------------


def test_unknown_kind_typecheck_at_construct() -> None:
    with pytest.raises(ValidationError):
        LambdaImgSchedule(kind="garbage")  # type: ignore[arg-type]


def test_finite_outputs_for_all_kinds() -> None:
    for kind in ("constant", "linear", "sigmoid", "cosine_with_anneal"):
        kwargs = dict(kind=kind, warmup_epochs=10, lambda_min=0.0, lambda_max=1.0)
        if kind == "cosine_with_anneal":
            kwargs["total_epochs"] = 100
        s = LambdaImgSchedule(**kwargs)  # type: ignore[arg-type]
        for e in (0, 5, 10, 50, 200):
            v = compute_lambda_img(s, e)
            assert math.isfinite(v)
