"""Shared presentation glue for the VENA article studies.

Single source of truth for two things the individual study modules must agree on:

1. The **pixel/latent generation-space grouping** (the article's primary axis for
   Study 1). Note this is *not* the registry's ``tier`` field — ``tier`` is the
   generation *formulation* (``flow`` / ``gan`` / ``diffusion`` / ``null``),
   whereas the article groups by the *space* the model generates in.
2. Locating the frozen per-scan CSVs under the local results archive.

**Do not add statistics here.** Selection-NFE reduction, bootstrap CI, paired
Wilcoxon, Cliff's delta and Holm correction all live in ``vena.validation.stats``
and ``vena.validation.spatial_residual`` and must be imported from there — two
copies of a statistic drift (see ``.claude/skills/orchestrate/SKILL.md`` §7).
"""

from __future__ import annotations

from pathlib import Path

from vena.validation.spatial_residual import _filter_to_selection_nfe

# Public alias for the one selection-NFE reducer, so every study shares it.
filter_to_selection_nfe = _filter_to_selection_nfe

# Generation-space grouping. ``reference`` = the null pass-through baseline.
DOMAIN: dict[str, str] = {
    "C0-Identity": "reference",
    "C1-pGAN-t1pre": "pixel",
    "C1-pGAN-t2": "pixel",
    "C1-pGAN-flair": "pixel",
    "C2-ResViT": "pixel",
    "C3-SynDiff-t1pre": "pixel",
    "C3-SynDiff-t2": "pixel",
    "C3-SynDiff-flair": "pixel",
    "C4-3D-DiT": "latent",
    "C5-T1C-RFlow": "latent",
    "C6-3D-LDDPM": "latent",
    "C7-3D-Latent-Pix2Pix": "latent",
    "VENA-S1-v3b-rw": "latent",
    "VENA-S1-v3b": "latent",
    "VENA-S1-v3a": "latent",
    "VENA-S3-LPL-b2c": "latent",
}

DOMAIN_ORDER: tuple[str, ...] = ("reference", "pixel", "latent")


def domain_of(method: str) -> str:
    """Return the generation-space group of ``method`` (defaults to ``latent``)."""
    return DOMAIN.get(method, "latent")


ANALYSES_ROOT = Path("/media/mpascual/Sandisk2TB/research/vena/results/fm/inference/analyses")


def per_scan_csv(routine: str, filename: str) -> Path:
    """Resolve ``<routine>/LATEST/per_scan/<filename>`` under the local archive.

    Raises:
        FileNotFoundError: if the resolved path does not exist.
    """
    path = (ANALYSES_ROOT / routine / "LATEST" / "per_scan" / filename).resolve()
    if not path.exists():
        raise FileNotFoundError(f"per-scan CSV not found: {path}")
    return path
