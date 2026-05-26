"""Atlas registry — maps logical ROI names to ``(atlas_id, label_value)``.

For v0.1 we use Harvard-Oxford subcortical and cortical, plus the in-house
venous mask. The CIT168 deep-GM atlas (Pauli 2018) and the JHU ICBM-DTI-81
white-matter labels are noted as ``[deferred]`` — both are easily added.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AtlasROI:
    """One logical ROI mapped to one atlas image + the label values it covers.

    A ROI may be the union of multiple integer labels in the same atlas
    (e.g. left + right hemispheres).
    """

    atlas_id: str
    label_values: tuple[int, ...]
    description: str = ""


# Atlas IDs used in the registry below.
HO_CORT = "harvard_oxford_cort"  # 48 cortical labels
HO_SUB = "harvard_oxford_sub"  # 21 subcortical labels (lateralised)
VENOUS_INHOUSE = "venous_inhouse"  # binary mask, label 1


# Harvard-Oxford subcortical label values (from the maxprob-thr25-1mm atlas
# inspected on 2026-05-26; see comments in atlases/__init__.py for the full
# enumeration).
HO_SUB_LABELS = {
    "Left Cerebral White Matter": 1,
    "Left Cerebral Cortex": 2,
    "Left Lateral Ventricle": 3,
    "Left Thalamus": 4,
    "Left Caudate": 5,
    "Left Putamen": 6,
    "Left Pallidum": 7,
    "Brain-Stem": 8,
    "Left Hippocampus": 9,
    "Left Amygdala": 10,
    "Left Accumbens": 11,
    "Right Cerebral White Matter": 12,
    "Right Cerebral Cortex": 13,
    "Right Lateral Ventricle": 14,
    "Right Thalamus": 15,
    "Right Caudate": 16,
    "Right Putamen": 17,
    "Right Pallidum": 18,
    "Right Hippocampus": 19,
    "Right Amygdala": 20,
    "Right Accumbens": 21,
}


def atlas_label_value(name: str) -> int:
    """Look up a single HO-subcortical label by its (verbose) name."""
    if name not in HO_SUB_LABELS:
        raise KeyError(f"unknown HO-subcortical label: {name!r}")
    return HO_SUB_LABELS[name]


# Logical ROI registry. Keys are ROI ids used throughout the test code.
ATLAS_REGISTRY: dict[str, AtlasROI] = {
    # T1 / T2 ROIs — built from Harvard-Oxford subcortical
    "cortical_gm": AtlasROI(
        atlas_id=HO_SUB,
        label_values=(
            HO_SUB_LABELS["Left Cerebral Cortex"],
            HO_SUB_LABELS["Right Cerebral Cortex"],
        ),
        description="Cortical grey matter (HO subcortical L+R Cerebral Cortex)",
    ),
    "nawm": AtlasROI(
        atlas_id=HO_SUB,
        label_values=(
            HO_SUB_LABELS["Left Cerebral White Matter"],
            HO_SUB_LABELS["Right Cerebral White Matter"],
        ),
        description="NAWM proxy (HO subcortical L+R Cerebral White Matter, refined by parenchyma∖tumour at use)",
    ),
    "ventricles": AtlasROI(
        atlas_id=HO_SUB,
        label_values=(
            HO_SUB_LABELS["Left Lateral Ventricle"],
            HO_SUB_LABELS["Right Lateral Ventricle"],
        ),
        description="Lateral ventricles (HO subcortical L+R Lateral Ventricle)",
    ),
    "basal_ganglia": AtlasROI(
        atlas_id=HO_SUB,
        label_values=(
            HO_SUB_LABELS["Left Caudate"],
            HO_SUB_LABELS["Right Caudate"],
            HO_SUB_LABELS["Left Putamen"],
            HO_SUB_LABELS["Right Putamen"],
            HO_SUB_LABELS["Left Pallidum"],
            HO_SUB_LABELS["Right Pallidum"],
        ),
        description="Basal ganglia (caudate + putamen + pallidum, L+R)",
    ),
    "putamen": AtlasROI(
        atlas_id=HO_SUB,
        label_values=(HO_SUB_LABELS["Left Putamen"], HO_SUB_LABELS["Right Putamen"]),
    ),
    "caudate": AtlasROI(
        atlas_id=HO_SUB,
        label_values=(HO_SUB_LABELS["Left Caudate"], HO_SUB_LABELS["Right Caudate"]),
    ),
    "globus_pallidus": AtlasROI(
        atlas_id=HO_SUB,
        label_values=(HO_SUB_LABELS["Left Pallidum"], HO_SUB_LABELS["Right Pallidum"]),
        description="Globus pallidus (HO subcortical L+R Pallidum)",
    ),
    "brainstem": AtlasROI(
        atlas_id=HO_SUB,
        label_values=(HO_SUB_LABELS["Brain-Stem"],),
    ),
    # In-house venous mask (binary; label 1 if present)
    "sinus": AtlasROI(
        atlas_id=VENOUS_INHOUSE,
        label_values=(1,),
        description="Dural sinus mask (in-house, built from UCSF-PDGM T1Gd MIPs)",
    ),
    # Pituitary, choroid plexus, cerebellum, substantia nigra, red nucleus,
    # dentate nucleus — DEFERRED: not in HO subcortical; need CIT168 / cerebellum
    # atlas / AAL extensions. Tests that depend on these ROIs gracefully skip
    # via applicable() when the ROI is missing.
}
