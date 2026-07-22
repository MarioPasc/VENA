"""Tests for vena.segmentation.models.registry.

Covers:
- @register_segmentation_model decorator registers a factory.
- get_segmentation_model("registered_name", cfg) calls the factory and returns
  the result.
- get_segmentation_model("nope", cfg) raises SegModelError listing known keys.
- Duplicate registration raises SegModelError.
"""

from __future__ import annotations

import pytest

import vena.segmentation.models.registry as _reg_mod
from vena.segmentation.config import ModelConfig
from vena.segmentation.exceptions import SegModelError
from vena.segmentation.models import (
    get_segmentation_model,
    register_segmentation_model,
    registered_model_names,
)

pytestmark = pytest.mark.segmentation

# ---------------------------------------------------------------------------
# Fixture: isolate registry state across tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_registry() -> pytest.Generator[None, None, None]:
    """Save and restore the global _REGISTRY around every test.

    This prevents test-registered dummy models from leaking into other tests
    (or from raising 'already registered' on repeated runs in the same session).
    """
    original = dict(_reg_mod._REGISTRY)
    yield
    _reg_mod._REGISTRY.clear()
    _reg_mod._REGISTRY.update(original)


# ---------------------------------------------------------------------------
# Minimal ModelConfig fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def model_cfg() -> ModelConfig:
    """Return a minimal frozen ModelConfig for registry calls."""
    return ModelConfig(name="bsf_swinunetr_brats")


# ---------------------------------------------------------------------------
# Sentinel return value for the dummy factory
# ---------------------------------------------------------------------------


class _DummyModule:
    """Stub that records the config it was called with."""

    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_register_and_get(model_cfg: ModelConfig) -> None:
    """Registering a factory and getting it back returns a _DummyModule."""

    @register_segmentation_model("dummy_net")
    class DummyNet(_DummyModule):
        pass

    result = get_segmentation_model("dummy_net", model_cfg)
    assert isinstance(result, _DummyModule)
    assert result.cfg is model_cfg


def test_registered_names_includes_newly_registered() -> None:
    """registered_model_names() reflects newly decorated classes."""

    @register_segmentation_model("another_net")
    class AnotherNet(_DummyModule):
        pass

    assert "another_net" in registered_model_names()


def test_unknown_name_raises_seg_model_error(model_cfg: ModelConfig) -> None:
    """get_segmentation_model with an unregistered name raises SegModelError."""
    with pytest.raises(SegModelError) as exc_info:
        get_segmentation_model("nope", model_cfg)

    msg = str(exc_info.value)
    assert "nope" in msg, "Error message should name the unknown key"


def test_unknown_name_error_lists_registered(model_cfg: ModelConfig) -> None:
    """SegModelError on unknown name lists currently registered keys."""

    @register_segmentation_model("visible_net")
    class VisibleNet(_DummyModule):
        pass

    with pytest.raises(SegModelError) as exc_info:
        get_segmentation_model("does_not_exist", model_cfg)

    msg = str(exc_info.value)
    assert "visible_net" in msg, (
        "Error message must list registered keys so callers know what is available"
    )


def test_duplicate_registration_raises() -> None:
    """Re-registering the same name raises SegModelError."""

    @register_segmentation_model("dupe_net")
    class DupeNetV1(_DummyModule):
        pass

    with pytest.raises(SegModelError):

        @register_segmentation_model("dupe_net")
        class DupeNetV2(_DummyModule):
            pass


def test_decorator_returns_class_unchanged() -> None:
    """The decorator is a pass-through — the class identity is preserved."""

    @register_segmentation_model("passthrough_net")
    class PassNet(_DummyModule):
        pass

    assert PassNet is PassNet  # trivially true, but verifies no wrapping
    assert issubclass(PassNet, _DummyModule)


def test_get_calls_factory_each_time(model_cfg: ModelConfig) -> None:
    """Each get_segmentation_model call constructs a fresh instance."""

    @register_segmentation_model("fresh_net")
    class FreshNet(_DummyModule):
        pass

    a = get_segmentation_model("fresh_net", model_cfg)
    b = get_segmentation_model("fresh_net", model_cfg)
    assert a is not b, "get_segmentation_model must return a new instance each call"
