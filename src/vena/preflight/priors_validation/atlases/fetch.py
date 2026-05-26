"""Atlas provisioning — wraps nilearn / templateflow downloads with caching.

Downloads land under ``<atlases_root>/{nilearn,templateflow}/``. Every call
returns the resolved file path; absence of the file triggers a download.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AtlasBundle:
    """All atlas paths needed by the priors-validation routine."""

    mni152_t1w: Path
    ho_cort_path: Path
    ho_sub_path: Path
    ho_cort_labels: list[str]
    ho_sub_labels: list[str]
    venous_inhouse: Path | None
    extras: dict[str, Path]


def ensure_atlases(
    atlases_root: Path,
    venous_inhouse_path: Path | None = None,
) -> AtlasBundle:
    """Ensure every atlas the routine needs is on disk; return their paths.

    Parameters
    ----------
    atlases_root
        Root directory under which atlases are cached. Nilearn and templateflow
        each get their own subdirectory so re-runs hit the cache.
    venous_inhouse_path
        Optional path to the in-house venous mask NIfTI (output of the
        ``venous_atlas_build`` routine). When ``None``, T3 R_sinus rows fall
        back to ``not_applicable`` in subject reports.
    """
    atlases_root = Path(atlases_root)
    atlases_root.mkdir(parents=True, exist_ok=True)
    nilearn_dir = atlases_root / "nilearn"
    templateflow_dir = atlases_root / "templateflow"
    nilearn_dir.mkdir(parents=True, exist_ok=True)
    templateflow_dir.mkdir(parents=True, exist_ok=True)

    # Route templateflow downloads to the project atlas cache.
    os.environ.setdefault("TEMPLATEFLOW_HOME", str(templateflow_dir))
    import templateflow.api as tf

    mni152_t1w = Path(
        tf.get(
            "MNI152NLin2009cAsym",
            resolution=1,
            desc="brain",
            suffix="T1w",
            extension=".nii.gz",
        )
    )
    logger.info("MNI152NLin2009cAsym T1w brain: %s", mni152_t1w)

    from nilearn.datasets import fetch_atlas_harvard_oxford

    ho_cort = fetch_atlas_harvard_oxford("cort-maxprob-thr25-1mm", data_dir=str(nilearn_dir))
    ho_sub = fetch_atlas_harvard_oxford("sub-maxprob-thr25-1mm", data_dir=str(nilearn_dir))
    # In recent nilearn, ``maps`` is a ``Nifti1Image``; ``filename`` is the
    # on-disk NIfTI path. Older versions exposed ``maps`` as a path string.
    ho_cort_path = Path(ho_cort.filename if hasattr(ho_cort, "filename") else ho_cort.maps)
    ho_sub_path = Path(ho_sub.filename if hasattr(ho_sub, "filename") else ho_sub.maps)
    logger.info("Harvard-Oxford cort: %s", ho_cort_path)
    logger.info("Harvard-Oxford sub : %s", ho_sub_path)

    venous = None
    if venous_inhouse_path is not None:
        p = Path(venous_inhouse_path)
        if p.exists():
            venous = p
        else:
            logger.warning(
                "Venous in-house atlas not found at %s — T3 R_sinus rows will "
                "report not_applicable.",
                p,
            )

    return AtlasBundle(
        mni152_t1w=mni152_t1w,
        ho_cort_path=ho_cort_path,
        ho_sub_path=ho_sub_path,
        ho_cort_labels=list(ho_cort.labels),
        ho_sub_labels=list(ho_sub.labels),
        venous_inhouse=venous,
        extras={},
    )
