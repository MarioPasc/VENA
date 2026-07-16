"""Phase-2 method and cohort metadata registry.

Pre-registered statistical roles are **pinned as module-level constants** at
import time (proposal P3, 2026-07-16).  They are never derived from name
patterns alone — ``startswith("VENA-")`` cannot distinguish the headline
from ablations and would Holm-correct over 12 rows instead of 8, making
every p-value in the paper wrong in a way that looks perfectly plausible.

:func:`load_partitions` populates the cohort-ring mapping (``COHORT_RING``,
``RING_A_COHORTS``, ``RING_B_COHORTS``) from ``ring_partitions.json`` written
by ``vena-validation-preregister``.  It also extends ``METHOD_ROLE`` and
``SELECTION_NFE`` for any discovered-but-unregistered methods (e.g. future
BraTS-PED backfill rows) without overwriting the pre-registered values.

:func:`method_role` returns the **four-way** pre-registered role:
``"vena"`` | ``"family"`` | ``"ablation"`` | ``"supplementary"``.
Unknown methods log a WARNING and return ``"supplementary"`` (fail-open for
I/O so BraTS-PED-era rows do not crash the loader) but must never silently
enter a Holm family — statistical routines must assert the role before testing.

Family-size guards are module-level assertions: adding a method to
``COMPETITOR_FAMILY`` or ``ABLATION_FAMILY`` is a CI failure unless the
assertion constant is also updated.  This is deliberate — those sizes are
pre-registered and cannot be changed silently.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pre-registered family constants
# Do NOT change the family sizes without bumping decision.json schema_version
# and re-running vena-validation-preregister.
# ---------------------------------------------------------------------------

#: The headline VENA model — reference arm of every paired Wilcoxon test.
VENA_HEADLINE: str = "VENA-S1-v3b-rw"

#: Competitor family — 8 members (Holm family size for §6.1 paired tests).
#: C0=null floor, C1/C2/C7=GAN, C3/C4/C6=diffusion, C5=flow-matching SOTA.
COMPETITOR_FAMILY: tuple[str, ...] = (
    "C0-Identity",
    "C1-pGAN-t1pre",
    "C2-ResViT",
    "C3-SynDiff-t1pre",
    "C4-3D-DiT",
    "C5-T1C-RFlow",
    "C6-3D-LDDPM",
    "C7-3D-Latent-Pix2Pix",
)

#: Ablation family — 3 members, own Holm correction (§6.2).
#: v3b isolates region-weighting (A1), v3a isolates mask conditioning,
#: LPL-b2c is the LPL null arm.
ABLATION_FAMILY: tuple[str, ...] = (
    "VENA-S1-v3b",
    "VENA-S1-v3a",
    "VENA-S3-LPL-b2c",
)

#: Supplementary rows — reported in tables but enter NO Holm family.
#: Single-source pGAN/SynDiff panels; t1pre is the canonical direction
#: (choosable a priori, no oracle selection).
SUPPLEMENTARY: tuple[str, ...] = (
    "C1-pGAN-t2",
    "C1-pGAN-flair",
    "C3-SynDiff-t2",
    "C3-SynDiff-flair",
)

# Guard: future method additions cannot silently resize a Holm family.
assert len(COMPETITOR_FAMILY) == 8, (
    f"COMPETITOR_FAMILY must have exactly 8 members (Holm n); "
    f"got {len(COMPETITOR_FAMILY)}.  Update the assertion constant "
    f"only after a pre-registration amendment."
)
assert len(ABLATION_FAMILY) == 3, (
    f"ABLATION_FAMILY must have exactly 3 members; got {len(ABLATION_FAMILY)}."
)

# ---------------------------------------------------------------------------
# MethodRole literal and MethodSpec dataclass
# ---------------------------------------------------------------------------

MethodRole = Literal["vena", "family", "ablation", "supplementary"]


@dataclass(frozen=True)
class MethodSpec:
    """Pre-registered metadata for a single method.

    Parameters
    ----------
    key :
        Exact on-disk method directory name (hyphen-separated, matches
        ``predictions/<key>/`` in the inference tree).
    display :
        Short human-readable label for figures and tables.
    tier :
        Methodological tier: ``"null"`` | ``"gan"`` | ``"diffusion"`` |
        ``"flow"``.  Used for colour grouping.
    role :
        Pre-registered statistical role.
    selection_nfe :
        Single NFE used in the headline table for this method (verified on
        disk 2026-07-16; see SHARED_CONTRACTS §4).
    panel_source :
        Input modality variant for single-source models (pGAN, SynDiff).
        ``None`` for multi-source and VENA methods.  The t1pre panel is the
        canonical family entry; t2/flair go to SUPPLEMENTARY.
    """

    key: str
    display: str
    tier: str
    role: MethodRole
    selection_nfe: int
    panel_source: str | None


# ---------------------------------------------------------------------------
# All 16 pre-registered methods in canonical display order
# ---------------------------------------------------------------------------

#: Registry of all pre-registered methods.
#: Order: VENA headline → competitor family (C0→C7) → ablations → supplementary.
#: Stable across runs — role-grouped by design, not alphabetical.
METHOD_SPECS: tuple[MethodSpec, ...] = (
    # ---- Headline ----
    MethodSpec("VENA-S1-v3b-rw", "VENA (ours)", "flow", "vena", 5, None),
    # ---- Competitor family (n=8) ----
    MethodSpec("C0-Identity", "Identity", "null", "family", 1, None),
    MethodSpec("C1-pGAN-t1pre", "pGAN-T1pre", "gan", "family", 1, "t1pre"),
    MethodSpec("C2-ResViT", "ResViT", "gan", "family", 1, None),
    MethodSpec("C3-SynDiff-t1pre", "SynDiff-T1pre", "diffusion", "family", 4, "t1pre"),
    MethodSpec("C4-3D-DiT", "3D-DiT", "diffusion", "family", 5, None),
    MethodSpec("C5-T1C-RFlow", "T1C-RFlow", "flow", "family", 5, None),
    MethodSpec("C6-3D-LDDPM", "3D-LDDPM", "diffusion", "family", 1000, None),
    MethodSpec("C7-3D-Latent-Pix2Pix", "Latent-Pix2Pix", "gan", "family", 1, None),
    # ---- Ablation family (n=3) ----
    MethodSpec("VENA-S1-v3b", "VENA-v3b (no RW)", "flow", "ablation", 5, None),
    MethodSpec("VENA-S1-v3a", "VENA-v3a (no mask)", "flow", "ablation", 5, None),
    MethodSpec("VENA-S3-LPL-b2c", "VENA-LPL-b2c (null)", "flow", "ablation", 5, None),
    # ---- Supplementary (n=4) ----
    MethodSpec("C1-pGAN-t2", "pGAN-T2", "gan", "supplementary", 1, "t2"),
    MethodSpec("C1-pGAN-flair", "pGAN-FLAIR", "gan", "supplementary", 1, "flair"),
    MethodSpec("C3-SynDiff-t2", "SynDiff-T2", "diffusion", "supplementary", 4, "t2"),
    MethodSpec("C3-SynDiff-flair", "SynDiff-FLAIR", "diffusion", "supplementary", 4, "flair"),
)

_METHOD_SPEC_MAP: dict[str, MethodSpec] = {s.key: s for s in METHOD_SPECS}

# ---------------------------------------------------------------------------
# Module-level constants pre-populated at import time from METHOD_SPECS
# ---------------------------------------------------------------------------

#: Per-method NFE selected at pre-registration.  Pre-populated from METHOD_SPECS;
#: extended (never overwritten) by load_partitions for unregistered methods.
SELECTION_NFE: dict[str, int] = {s.key: s.selection_nfe for s in METHOD_SPECS}

#: Method-name → four-way role.  Pre-populated at import; extended lazily by
#: load_partitions for newly discovered methods.
METHOD_ROLE: dict[str, str] = {s.key: s.role for s in METHOD_SPECS}

# ---------------------------------------------------------------------------
# Cohort-ring mapping (populated by load_partitions, empty at import)
# ---------------------------------------------------------------------------

#: Cohort-name → ring letter ("A" or "B").
#: Populated from ring_partitions.json by :func:`load_partitions`.
COHORT_RING: dict[str, str] = {}

#: Ring-A cohort names (cv_test role).
RING_A_COHORTS: frozenset[str] = frozenset()

#: Ring-B cohort names (test_only / OOD role).
RING_B_COHORTS: frozenset[str] = frozenset()


# ---------------------------------------------------------------------------
# Pure helpers — no I/O
# ---------------------------------------------------------------------------


def method_role(method: str) -> MethodRole:
    """Return the pre-registered four-way statistical role for *method*.

    Parameters
    ----------
    method :
        Exact method key (e.g. ``"VENA-S1-v3b-rw"``).

    Returns
    -------
    MethodRole
        One of ``"vena"``, ``"family"``, ``"ablation"``, ``"supplementary"``.

    Notes
    -----
    Unknown methods log a WARNING and return ``"supplementary"`` — fail-open
    for I/O (BraTS-PED backfill rows must not crash the loader).  Statistical
    routines must assert ``role == "family"`` before including a method in a
    Holm family; fail-open here does not mean fail-open in statistics.
    """
    spec = _METHOD_SPEC_MAP.get(method)
    if spec is None:
        logger.warning(
            "Unknown method %r — assigning supplementary role. "
            "If this is a new method (e.g. BraTS-PED backfill), add it to "
            "METHOD_SPECS in vena.validation.registry.",
            method,
        )
        return "supplementary"
    return spec.role


def ring_of_cohort(cohort: str, h5_ring_attr: str | None = None) -> str:
    """Return the ring letter (``"A"`` or ``"B"``) for *cohort*.

    Prefers the prediction H5's own ``ring`` root attribute (written by Phase 1
    and therefore authoritative) over the static ``COHORT_RING`` map (written by
    preregister from those same attributes).  Raises on disagreement so silent
    drift is caught immediately.

    Parameters
    ----------
    cohort :
        Cohort name as it appears in H5 metadata.
    h5_ring_attr :
        Value of ``h5file.attrs["ring"]`` from the prediction H5, or ``None``
        if the caller has not opened the file.

    Returns
    -------
    str
        ``"A"`` or ``"B"``.

    Raises
    ------
    ValueError
        If the H5 attr and the static preregistered map disagree.
    KeyError
        If neither source is available (unknown cohort + preregister not run).
    """
    static = COHORT_RING.get(cohort)

    if h5_ring_attr is not None:
        if static is not None and h5_ring_attr != static:
            raise ValueError(
                f"Ring disagreement for cohort {cohort!r}: "
                f"H5 ring={h5_ring_attr!r}, preregistered ring={static!r}. "
                "Re-run vena-validation-preregister to resync."
            )
        return h5_ring_attr

    if static is not None:
        return static

    raise KeyError(
        f"Unknown cohort {cohort!r} and no H5 ring attr provided. "
        "Run vena-validation-preregister first, or pass h5_ring_attr."
    )


def method_order() -> list[str]:
    """Return the canonical display order for all pre-registered methods.

    Order: VENA headline → competitor family (C0…C7) → ablations → supplementary.
    Use this for all figures and tables so methods appear consistently.
    Stable across runs — role-grouped, not alphabetical.

    Returns
    -------
    list[str]
        List of method keys in canonical display order.
    """
    return [s.key for s in METHOD_SPECS]


def method_palette() -> dict[str, str]:
    """Return a colourblind-safe hex colour for each pre-registered method.

    Uses the Wong (2011) 8-colour palette for the competitor family (mapped
    exactly to the 8 family members).  VENA gets a distinct teal.  Ablations
    get muted tones.  Supplementary get grey shades.

    Reference: Wong B (2011) Color blindness. Nature Methods 8:441.
    DOI: 10.1038/nmeth.1618

    Returns
    -------
    dict[str, str]
        ``{method_key: "#RRGGBB"}``.  All 16 pre-registered methods are
        present.
    """
    # Wong (2011) 8-colour palette — safe for deuteranopia/protanopia/tritanopia.
    _wong = (
        "#E69F00",  # orange
        "#56B4E9",  # sky blue
        "#009E73",  # bluish green
        "#F0E442",  # yellow
        "#0072B2",  # blue
        "#D55E00",  # vermilion
        "#CC79A7",  # reddish purple
        "#999999",  # grey (8th slot)
    )
    palette: dict[str, str] = {}

    # VENA headline — distinct teal (clearly primary).
    palette[VENA_HEADLINE] = "#00BCD4"

    # Competitor family — one Wong colour each (exactly 8, strict zip).
    for method, colour in zip(COMPETITOR_FAMILY, _wong, strict=True):
        palette[method] = colour

    # Ablations — muted olive / desaturated versions.
    _ablation_colours = ("#8D7B2A", "#3A7D5E", "#7B3D7D")
    for method, colour in zip(ABLATION_FAMILY, _ablation_colours, strict=True):
        palette[method] = colour

    # Supplementary — light to mid greys.
    _supp_colours = ("#DDDDDD", "#BBBBBB", "#888888", "#666666")
    for method, colour in zip(SUPPLEMENTARY, _supp_colours, strict=True):
        palette[method] = colour

    return palette


# ---------------------------------------------------------------------------
# Partition loader — fills cohort-ring constants from ring_partitions.json
# ---------------------------------------------------------------------------


def load_partitions(path: Path) -> None:
    """Populate cohort-ring constants from ``ring_partitions.json``.

    Extends ``METHOD_ROLE`` and ``SELECTION_NFE`` for any discovered-but-
    unregistered methods without overwriting the pre-registered values.

    Parameters
    ----------
    path :
        Path to ``ring_partitions.json`` as written by
        ``vena-validation-preregister``.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the JSON is malformed.
    """
    global COHORT_RING, RING_A_COHORTS, RING_B_COHORTS, METHOD_ROLE, SELECTION_NFE

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"ring_partitions.json not found: {path}")

    try:
        with path.open() as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"ring_partitions.json is not valid JSON: {exc}") from exc

    new_cohort_ring: dict[str, str] = {}
    rings: dict = data.get("rings", {})
    for ring_letter, ring_data in rings.items():
        for cohort_name in ring_data.get("cohorts", {}).keys():
            new_cohort_ring[cohort_name] = ring_letter

    COHORT_RING = new_cohort_ring
    RING_A_COHORTS = frozenset(c for c, r in COHORT_RING.items() if r == "A")
    RING_B_COHORTS = frozenset(c for c, r in COHORT_RING.items() if r == "B")

    # Extend METHOD_ROLE: pre-registered roles take priority; only add new ones.
    new_method_role: dict[str, str] = dict(METHOD_ROLE)
    for m in data.get("methods", []):
        if m not in new_method_role:
            new_method_role[m] = method_role(m)
    METHOD_ROLE = new_method_role

    # Extend SELECTION_NFE: pre-registered NFEs take priority.
    new_sel: dict[str, int] = dict(SELECTION_NFE)
    for m, nfe in data.get("selection_nfe", {}).items():
        if m not in new_sel and nfe is not None:
            new_sel[m] = int(nfe)
    SELECTION_NFE = new_sel

    logger.info(
        "Loaded ring partitions: %d cohorts (%d Ring A, %d Ring B), %d methods",
        len(COHORT_RING),
        len(RING_A_COHORTS),
        len(RING_B_COHORTS),
        len(METHOD_ROLE),
    )
