"""HD-BET brain extraction adapter (subprocess wrapper around the
``hd-bet`` CLI installed in a dedicated conda env).

Why a subprocess wrapper instead of a Python import: HD-BET 2.x pulls in
nnU-Net v2, which carries a heavy dependency tree and can pin torch
versions that conflict with the project env. Isolating HD-BET in its own
env (``~/.conda/envs/hdbet``) keeps the project env clean.
"""

from __future__ import annotations

from .runner import (
    HDBETError,
    HDBETSkullStripConfig,
    HDBETSkullStripRunner,
)

__all__ = [
    "HDBETError",
    "HDBETSkullStripConfig",
    "HDBETSkullStripRunner",
]
