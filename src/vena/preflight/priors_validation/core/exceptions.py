"""Exception hierarchy for the priors-validation preflight.

Every exception carries the ``subject_id`` (when applicable), the optional
``prior_id`` it pertains to, and an actionable ``message`` per spec §2.2.
"""

from __future__ import annotations


class ValidationException(Exception):
    """Base class for all priors-validation exceptions.

    Parameters
    ----------
    subject_id
        Cohort identifier of the subject the failure pertains to.
    prior_id
        Identifier of the prior channel involved, or ``None`` when not
        attributable to a single prior.
    message
        Actionable human-readable explanation.
    """

    def __init__(
        self,
        subject_id: str,
        message: str,
        *,
        prior_id: str | None = None,
    ) -> None:
        self.subject_id = subject_id
        self.prior_id = prior_id
        self.message = message
        prefix = f"[{subject_id}"
        if prior_id is not None:
            prefix += f"/{prior_id}"
        prefix += "]"
        super().__init__(f"{prefix} {message}")


class PriorMissingError(ValidationException):
    """Raised when a required prior is absent for a subject."""


class AtlasRegistrationError(ValidationException):
    """Raised when atlas-to-subject registration QC fails (Dice < 0.85).

    Aborts the remaining tests *for that subject* per spec §8.4 but does not
    propagate to other subjects.
    """


class InsufficientCohortError(ValidationException):
    """Raised when the cohort is too small for cohort-level aggregation."""


class InvalidThresholdError(ValidationException):
    """Raised when a literature threshold is missing or malformed in config."""
