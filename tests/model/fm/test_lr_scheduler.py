"""Unit tests for the LR-lambda dispatch (2026-06-09 overhaul, CHANGE 1).

The pure-function ``_lr_lambda(scheduler, step, warmup, max_steps)`` lives in
``vena.model.fm.lightning.module``. Exercise every branch without spinning
up Lightning so the suite stays a fast CPU pass.
"""

from __future__ import annotations

import math

import pytest

from vena.model.fm.lightning.module import _lr_lambda

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


def test_warmup_linear_ramp() -> None:
    """Linear ramp from 0 at step=0 to 1 at step=warmup_steps."""
    warmup = 100
    max_steps = 1000
    assert _lr_lambda("cosine", 0, warmup, max_steps) == 0.0
    assert _lr_lambda("cosine", warmup // 2, warmup, max_steps) == pytest.approx(0.5)
    # At step==warmup, the warmup branch falls through to the decay branch
    # which evaluates progress==0 → lambda==1 for cosine.
    assert _lr_lambda("cosine", warmup, warmup, max_steps) == pytest.approx(1.0)


def test_warmup_zero_uses_decay_immediately() -> None:
    """warmup_steps=0 means the very first step already counts as decay."""
    assert _lr_lambda("cosine", 0, 0, 1000) == pytest.approx(1.0)
    assert _lr_lambda("cosine", 1000, 0, 1000) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Cosine
# ---------------------------------------------------------------------------


def test_cosine_decay_endpoints() -> None:
    """Cosine goes 1 → 0 from end-of-warmup → max_steps."""
    warmup, max_steps = 100, 1100
    assert _lr_lambda("cosine", warmup, warmup, max_steps) == pytest.approx(1.0)
    assert _lr_lambda("cosine", max_steps, warmup, max_steps) == pytest.approx(0.0)


def test_cosine_midpoint() -> None:
    """At progress 0.5, lambda == 0.5 (cos(π/2)+1)/2."""
    warmup, max_steps = 0, 1000
    mid = max_steps // 2
    assert _lr_lambda("cosine", mid, warmup, max_steps) == pytest.approx(0.5, abs=1e-6)


def test_cosine_monotonic_decreasing_post_warmup() -> None:
    """Strictly non-increasing once decay starts."""
    warmup, max_steps = 0, 1000
    prev = math.inf
    for step in range(0, max_steps + 1, 50):
        val = _lr_lambda("cosine", step, warmup, max_steps)
        assert val <= prev + 1e-9, f"cosine lambda increased at step {step}"
        prev = val


# ---------------------------------------------------------------------------
# Polynomial (back-compat)
# ---------------------------------------------------------------------------


def test_polynomial_endpoints() -> None:
    warmup, max_steps = 100, 1100
    assert _lr_lambda("polynomial", warmup, warmup, max_steps) == pytest.approx(1.0)
    assert _lr_lambda("polynomial", max_steps, warmup, max_steps) == pytest.approx(0.0)


def test_polynomial_midpoint() -> None:
    """Linear → at progress 0.5, lambda == 0.5."""
    warmup, max_steps = 0, 1000
    assert _lr_lambda("polynomial", 500, warmup, max_steps) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Constant
# ---------------------------------------------------------------------------


def test_constant_after_warmup() -> None:
    """`constant` scheduler returns 1.0 for every post-warmup step."""
    for step in (100, 500, 999, 10_000):
        assert _lr_lambda("constant", step, 100, 1000) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Unknown scheduler — the bug-prevention check
# ---------------------------------------------------------------------------


def test_unknown_scheduler_raises() -> None:
    """The silent-fallthrough fallback was the 2026-06-07 LR misconfiguration."""
    with pytest.raises(ValueError, match="unknown LR scheduler"):
        _lr_lambda("bogus", 100, 100, 1000)


def test_unknown_scheduler_not_raised_during_warmup() -> None:
    """During warmup the scheduler is not consulted — invalid name OK."""
    # If the warmup branch ran first and short-circuited, no ValueError is
    # raised. This documents the expected behaviour explicitly.
    assert _lr_lambda("bogus", 50, 100, 1000) == pytest.approx(0.5)
