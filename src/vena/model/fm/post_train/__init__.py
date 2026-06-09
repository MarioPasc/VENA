"""Post-training analysis and plotting for VENA FM runs.

This package consumes a finished `experiments/<run_id>/` directory and produces
the figures under `<run_dir>/plots/`. The public entrypoint is
`PostTrainRunner`, which the train engine calls after `trainer.fit()`.

Library code only — the CLI / Pydantic config wrappers live in
`routines/fm/post_train/`.
"""

from __future__ import annotations

from vena.model.fm.post_train.runner import PostTrainRunner, render_post_train_plots

__all__ = ["PostTrainRunner", "render_post_train_plots"]
