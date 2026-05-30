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
