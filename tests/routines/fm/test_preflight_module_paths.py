"""Unit tests for B11 resolved module path checks (CWD-independent).

``_assert_module_paths_in_repo_root()`` verifies that both ``routines`` and
``vena`` are imported from within the expected repository root, catching
stale editable installs or wrong PYTHONPATH values.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from routines.fm.train.engine import _assert_module_paths_in_repo_root

pytestmark = pytest.mark.unit


class TestModulePathsInRepoRoot:
    """The guard fires when a module lives outside the repo root."""

    def test_passes_with_correct_paths(self) -> None:
        """Both vena and routines are in the correct repo; must not raise."""
        # The test itself runs from the repo, so the guard passes as-is.
        _assert_module_paths_in_repo_root()  # should not raise

    def test_raises_when_vena_outside_repo(self) -> None:
        """If vena.__file__ is outside the repo root, AssertionError fires."""

        fake_vena = MagicMock()
        fake_vena.__file__ = "/some/other/installation/vena/__init__.py"

        with patch.dict("sys.modules", {"vena": fake_vena}):
            with pytest.raises(AssertionError, match="vena"):
                _assert_module_paths_in_repo_root()

    def test_raises_when_routines_outside_repo(self) -> None:
        """If routines.__file__ is outside the repo root, AssertionError fires."""

        fake_routines = MagicMock()
        fake_routines.__file__ = "/totally/different/repo/routines/__init__.py"

        with patch.dict("sys.modules", {"routines": fake_routines}):
            with pytest.raises(AssertionError, match="routines"):
                _assert_module_paths_in_repo_root()

    def test_error_message_mentions_pythonpath(self) -> None:
        """Error message must mention PYTHONPATH to guide the user."""
        fake_routines = MagicMock()
        fake_routines.__file__ = "/wrong/place/routines/__init__.py"

        with patch.dict("sys.modules", {"routines": fake_routines}):
            with pytest.raises(AssertionError, match="PYTHONPATH"):
                _assert_module_paths_in_repo_root()
