"""VENA wrapper around the vendored SynDiff competitor.

Public surface kept narrow on purpose — engines, CLIs and tests pull only
these names. SynDiff is image-domain (2D slices, ``[-1, 1]``) and one-to-one
(exactly two contrasts per run, bidirectional cycle-consistency). The
contract convention used throughout is:

    contrast1  ==  the *target* modality   (e.g. ``"t1c"``)
    contrast2  ==  the *source* modality   (e.g. ``"t1pre"`` / ``"t2"`` / ``"flair"``)

so ``gen_diffusive_1`` synthesises the target given the source — the model
the inference module loads.
"""

from __future__ import annotations

from .dataset import (
    DatasetError,
    MultiCohortSynDiffSliceDataset,
    SynDiffSliceDataset,
)
from .inference import InferenceError, run_inference
from .runner import SynDiffRunnerError, train_syndiff

__all__ = [
    "DatasetError",
    "InferenceError",
    "MultiCohortSynDiffSliceDataset",
    "SynDiffRunnerError",
    "SynDiffSliceDataset",
    "run_inference",
    "train_syndiff",
]
