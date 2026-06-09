"""Compatibility shim — the styles now live in `vena.model.fm.post_train.plotting_styles`.

Re-exports the public surface so any external notebook / script that imports
from the original `routines.fm.post_train.plotting_styles` path keeps working.
"""

from __future__ import annotations

from vena.model.fm.post_train.plotting_styles import *  # noqa: F403
from vena.model.fm.post_train.plotting_styles import __all__  # noqa: F401
