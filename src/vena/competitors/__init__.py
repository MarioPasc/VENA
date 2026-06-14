"""Wrappers that feed VENA data into vendored competitor models.

Each competitor has its own subpackage under ``vena.competitors.<name>``. The
contract is the same for every wrapper:

1. A ``dataset.py`` module that exposes a ``torch.utils.data.Dataset`` reading
   directly from VENA's data path (image H5 for image-domain models, latent H5
   for latent-domain models).
2. **No augmentations** are applied. VENA owns the augmentation regime; the
   competitor's loader is fed only deterministic, normalised volumes.
3. A ``runner.py`` module exposing a ``train(cfg, run_dir)`` function that
   imports the vendored competitor model, builds it, drives the training loop,
   and writes ``metrics/train_step.csv``, ``checkpoints/*.pth``, and
   ``decision.json`` under ``run_dir``.

The actual competitor source code lives under ``src/external/<competitor>/``.
This package is the bridge.
"""

from __future__ import annotations
