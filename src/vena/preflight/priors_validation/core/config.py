"""Literature constants and thresholds (versioned).

Every numeric threshold this preflight uses lives here, with citation. Each
constant carries ``value``, ``unit``, ``source_doi``, ``version_introduced``
so future updates are auditable per spec §4.

Aliases used throughout:

* ``"cbf"``    — raw quantitative CBF map (ml/100g/min); spec §4.1.
* ``"adc"``    — raw ADC map (mm²/s); spec §4.2.
* ``"chi"``    — raw QSM map (ppm); spec §4.3. Not used in v0 (no phase data).

Derived (NAWM-normalised) channels live under their own names: ``"cbf_rel"``,
``"adc_rel"``, ``"cell"``, ``"sus"``, ``"itss"``, ``"vessel_soft"`` etc., but
T1 range thresholds only apply to the raw physical-units inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

ROUTINE_VERSION = "0.1.0"
CONFIG_VERSION_INTRODUCED = "v0.1"


@dataclass(frozen=True)
class RangeThreshold:
    """An acceptable-range threshold for one (prior, ROI) cell."""

    lo: float
    hi: float
    unit: str
    source_doi: str
    version_introduced: str = CONFIG_VERSION_INTRODUCED


# --------------------------------------------------------------------------
# T1 — range sanity (spec §§4.1, 4.2, 4.3 tables)
# --------------------------------------------------------------------------

# CBF (ml/100 g/min). Alsop 2015 consensus + Yen 2002 + Wamelink 2024.
RANGE_CBF: dict[str, RangeThreshold] = {
    "cortical_gm": RangeThreshold(
        lo=35.0,
        hi=80.0,
        unit="ml/100g/min",
        source_doi="10.1002/mrm.25197",
    ),
    "nawm": RangeThreshold(
        lo=12.0,
        hi=35.0,
        unit="ml/100g/min",
        source_doi="10.1002/mrm.25197",
    ),
    "whole_brain": RangeThreshold(
        lo=25.0,
        hi=55.0,
        unit="ml/100g/min",
        source_doi="10.1002/mrm.10140",
    ),
    "cerebellum": RangeThreshold(
        lo=30.0,
        hi=70.0,
        unit="ml/100g/min",
        source_doi="10.1002/mrm.25197",
    ),
    "hgg_tumour_core": RangeThreshold(
        lo=40.0,
        hi=200.0,
        unit="ml/100g/min",
        source_doi="10.1148/radiol.230793",
    ),
}

# ADC (10^-3 mm²/s — values stored in 10^-3 units; algorithm scales).
RANGE_ADC: dict[str, RangeThreshold] = {
    "nawm": RangeThreshold(
        lo=0.55,
        hi=1.15,
        unit="1e-3 mm^2/s",
        source_doi="10.1148/radiol.13130420",
    ),
    "cortical_gm": RangeThreshold(
        lo=0.6,
        hi=1.0,
        unit="1e-3 mm^2/s",
        source_doi="10.1148/radiol.13130420",
    ),
    "ventricles": RangeThreshold(
        lo=2.5,
        hi=3.4,
        unit="1e-3 mm^2/s",
        source_doi="10.1148/radiol.13130420",
    ),
    "hgg_cellular": RangeThreshold(
        lo=0.5,
        hi=1.1,
        unit="1e-3 mm^2/s",
        source_doi="10.1016/S0895-6111(00)00067-2",
    ),
    "necrotic_core": RangeThreshold(
        lo=1.6,
        hi=5.0,
        unit="1e-3 mm^2/s",
        source_doi="10.1016/S0895-6111(00)00067-2",
    ),
}

# QSM (ppm). Wang & Liu 2015, Liu 2011, Bilgic 2012.
RANGE_CHI: dict[str, RangeThreshold] = {
    "nawm": RangeThreshold(
        lo=-0.06,
        hi=-0.005,
        unit="ppm",
        source_doi="10.1002/mrm.25358",
    ),
    "cortical_gm": RangeThreshold(
        lo=-0.03,
        hi=0.03,
        unit="ppm",
        source_doi="10.1002/mrm.25358",
    ),
    "globus_pallidus": RangeThreshold(
        lo=0.06,
        hi=0.30,
        unit="ppm",
        source_doi="10.1016/j.neuroimage.2011.07.077",
    ),
    "substantia_nigra": RangeThreshold(
        lo=0.05,
        hi=0.25,
        unit="ppm",
        source_doi="10.1016/j.neuroimage.2011.07.077",
    ),
    "red_nucleus": RangeThreshold(
        lo=0.04,
        hi=0.22,
        unit="ppm",
        source_doi="10.1016/j.neuroimage.2011.07.077",
    ),
    "dentate_nucleus": RangeThreshold(
        lo=0.04,
        hi=0.20,
        unit="ppm",
        source_doi="10.1016/j.neuroimage.2011.07.077",
    ),
    "venous_sinus": RangeThreshold(
        lo=0.10,
        hi=0.65,
        unit="ppm",
        source_doi="10.1002/mrm.25358",
    ),
}

RANGE_TABLE: dict[str, dict[str, RangeThreshold]] = {
    "cbf": RANGE_CBF,
    "adc": RANGE_ADC,
    "chi": RANGE_CHI,
}

# Diagnostic table (spec §5.1) — symptom → probable cause.
DIAGNOSTIC_HINTS: dict[tuple[str, str, str], str] = {
    ("cbf", "nawm", "above"): "NAWM CBF too high → wrong M_0 scaling or labelling efficiency α",
    ("cbf", "nawm", "below"): "NAWM CBF too low → inverted label-control subtraction",
    (
        "cbf",
        "cortical_gm",
        "above",
    ): "Cortical CBF too high → check labelling efficiency / partial-volume correction",
    (
        "cbf",
        "cortical_gm",
        "below",
    ): "Cortical CBF too low → suspect M_0 scaling or partial-volume correction",
    (
        "chi",
        "globus_pallidus",
        "below",
    ): "GP χ near zero → QSM dipole-inversion regularisation too aggressive",
    ("adc", "ventricles", "below"): "CSF ADC too low → wrong b-value pair or EPI distortion",
    ("adc", "nawm", "below"): "NAWM ADC too low → restricted-diffusion artefact or wrong b-value",
    ("adc", "nawm", "above"): "NAWM ADC too high → suspect tissue mis-classification",
}

# --------------------------------------------------------------------------
# T2 — atlas localisation: expected magnitude orderings per prior (spec §5.2)
# --------------------------------------------------------------------------

# Each entry: ordered list of ROI names by *expected magnitude* (high → low).
T2_EXPECTED_ORDERINGS: dict[str, list[str]] = {
    "cbf": [
        "choroid_plexus",
        "pituitary",
        "cortical_gm",
        "basal_ganglia",
        "nawm",
        "ventricles",
    ],
    "chi": [
        "globus_pallidus",
        "substantia_nigra",
        "red_nucleus",
        "dentate_nucleus",
        "putamen",
        "caudate",
        "nawm",
        "ventricles",
    ],
    "adc": ["ventricles", "cortical_gm", "nawm"],
    # Derived NAWM-relative channels: ordering is preserved by construction;
    # adc_rel is the cleanest derived channel for T2 (cbf_rel less so because
    # it can saturate at the upper clip). We test the raw quantitative maps
    # via the "cbf"/"adc"/"chi" entries above, but the derived "adc_rel" still
    # has a meaningful expected order across ventricles > GM > NAWM.
    "adc_rel": ["ventricles", "cortical_gm", "nawm"],
}

# --------------------------------------------------------------------------
# T3 — T1Gd-coherence sign + magnitude bands (spec §5.3)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CoherenceExpectation:
    """Expected sign and |ρ| band for one (prior, ROI) cell of T3."""

    sign: int  # +1, 0, or -1
    rho_lo: float
    rho_hi: float
    note: str = ""


# Keys: (prior_id, roi_id). roi_id ∈ {tum, sinus, healthy, pituitary}.
T3_EXPECTATIONS: dict[tuple[str, str], CoherenceExpectation] = {
    # CBF
    ("cbf", "tum"): CoherenceExpectation(
        sign=+1, rho_lo=0.3, rho_hi=1.0, note="HGG tumour core perfused fraction"
    ),
    ("cbf", "sinus"): CoherenceExpectation(
        sign=+1, rho_lo=0.4, rho_hi=1.0, note="dural sinuses bright on T1Gd"
    ),
    ("cbf", "healthy"): CoherenceExpectation(
        sign=+1, rho_lo=0.1, rho_hi=0.3, note="weakly positive in NAWM+GM"
    ),
    ("cbf", "pituitary"): CoherenceExpectation(
        sign=+1, rho_lo=0.4, rho_hi=1.0, note="pituitary has no BBB"
    ),
    ("cbf_rel", "tum"): CoherenceExpectation(sign=+1, rho_lo=0.3, rho_hi=1.0, note=""),
    ("cbf_rel", "sinus"): CoherenceExpectation(sign=+1, rho_lo=0.4, rho_hi=1.0, note=""),
    ("cbf_rel", "healthy"): CoherenceExpectation(sign=+1, rho_lo=0.1, rho_hi=0.3, note=""),
    ("cbf_rel", "pituitary"): CoherenceExpectation(sign=+1, rho_lo=0.4, rho_hi=1.0, note=""),
    # Cellularity
    ("cell", "tum"): CoherenceExpectation(
        sign=+1, rho_lo=0.0, rho_hi=1.0, note="positive within enhancing subregion only"
    ),
    ("cell", "sinus"): CoherenceExpectation(sign=0, rho_lo=-0.1, rho_hi=0.1, note=""),
    ("cell", "healthy"): CoherenceExpectation(sign=0, rho_lo=-0.1, rho_hi=0.1, note=""),
    ("cell", "pituitary"): CoherenceExpectation(sign=0, rho_lo=-0.1, rho_hi=0.1, note=""),
    # Susceptibility (sub-A: sus / itss)
    ("sus", "tum"): CoherenceExpectation(
        sign=+1, rho_lo=0.0, rho_hi=1.0, note="weakly positive (ITSS-driven)"
    ),
    ("sus", "sinus"): CoherenceExpectation(
        sign=+1, rho_lo=0.5, rho_hi=1.0, note="strong positive (venous deoxyHb)"
    ),
    ("sus", "healthy"): CoherenceExpectation(sign=+1, rho_lo=0.0, rho_hi=0.3, note=""),
    ("sus", "pituitary"): CoherenceExpectation(sign=0, rho_lo=-0.1, rho_hi=0.1, note=""),
    # Susceptibility QSM (deferred in v0; entries kept for protocol completeness)
    ("chi_pos", "tum"): CoherenceExpectation(sign=+1, rho_lo=0.0, rho_hi=1.0, note=""),
    ("chi_pos", "sinus"): CoherenceExpectation(sign=+1, rho_lo=0.5, rho_hi=1.0, note=""),
    ("chi_pos", "healthy"): CoherenceExpectation(sign=+1, rho_lo=0.0, rho_hi=0.3, note=""),
    ("chi_pos", "pituitary"): CoherenceExpectation(sign=0, rho_lo=-0.1, rho_hi=0.1, note=""),
    ("chi_neg", "tum"): CoherenceExpectation(
        sign=-1, rho_lo=-1.0, rho_hi=-0.0, note="calcification suppresses enhancement"
    ),
    # ADC raw
    ("adc", "tum"): CoherenceExpectation(
        sign=-1, rho_lo=-1.0, rho_hi=-0.0, note="high ADC = necrosis = no enhancement"
    ),
    ("adc_rel", "tum"): CoherenceExpectation(sign=-1, rho_lo=-1.0, rho_hi=-0.0, note=""),
    # ITSS (sub-A susceptibility, in-tumour gated channel)
    ("itss", "tum"): CoherenceExpectation(sign=+1, rho_lo=0.0, rho_hi=1.0, note=""),
}

# --------------------------------------------------------------------------
# T4 — cross-modal coupling thresholds (spec §5.4)
# --------------------------------------------------------------------------

T4_ITSS_CBF_RHO_MIN = 0.3  # ρ_min for ITSS↔CBF inside tumour
T4_CELL_CBF_RHO_MIN_HGG = 0.3
T4_CELL_CBF_RHO_MIN_LGG = 0.1
T4_CALCIFICATION_DELTA_T1_MAX = 0.2  # z-units

# --------------------------------------------------------------------------
# T5 — reproducibility ICC targets (spec §4.4 + §5.5)
# --------------------------------------------------------------------------

T5_ICC_TARGETS: dict[str, float] = {
    "cbf": 0.7,
    "adc": 0.85,
    "chi": 0.85,
    "cbf_rel": 0.7,
    "adc_rel": 0.85,
    "sus": 0.85,
    "itss": 0.85,
}

# --------------------------------------------------------------------------
# Statistical interpretation (spec §4.4)
# --------------------------------------------------------------------------

ICC_INTERPRETATION = (
    (0.50, "poor"),
    (0.75, "moderate"),
    (0.90, "good"),
    (1.00, "excellent"),
)

# Cohort pass-rate thresholds (spec §§5.1–5.4)
COHORT_PASS_RATE_T1 = 0.90
COHORT_PASS_RATE_T2 = 0.95
COHORT_PASS_RATE_T3 = 0.75
COHORT_PASS_RATE_T4 = 0.70

# Per-subject T2 ρ pass threshold (spec §5.2)
T2_RHO_MIN = 0.7

# Effect-size guards (spec §6.2): if cohort-median |ρ| < this, the prior is
# flagged as not predictively informative regardless of p-values.
EFFECT_SIZE_MIN_FOR_INFORMATIVE = 0.1

# --------------------------------------------------------------------------
# Other constants
# --------------------------------------------------------------------------

BOOTSTRAP_RESAMPLES_DEFAULT = 1000
FDR_Q_DEFAULT = 0.05
TRAINING_CLEARANCE_THRESHOLD_DEFAULT = 0.9
