"""Exceptions raised by the FM training routine."""

from __future__ import annotations


class FMTrainError(Exception):
    """Base for errors raised inside the FM training routine."""


class PreflightGateError(FMTrainError):
    """A pre-flight gate failed.

    Raised when the training config depends on a pre-flight artifact
    (vessel mask QC, latent-aug equivariance, MAISI VAE audit, ...) that
    either does not exist or does not declare the required permissions.
    """


class InvalidResumeFromError(FMTrainError):
    """``run.resume_from`` could not be classified.

    Raised by :func:`routines.fm.train.engine._classify_resume_from` when the
    YAML value is neither one of the literal keywords (``baseline`` / ``latest``
    / ``best``), nor a run_id matching ``<UTC>_<stage>_<tag>_<sha>``, nor an
    absolute path to an existing ``.ckpt`` file. The engine does **not**
    silently fall back to fresh — an unrecognised value is a config bug.
    """
