"""Phase-2 benchmark validation substrate.

Public re-exports for downstream agents (paired fidelity §4.2, spatial
residual §4.3, downstream segmentation §4.4).
"""

from __future__ import annotations

from vena.validation.artifacts import (
    make_run_dir,
    symlink_latest,
    write_decision_json,
    write_per_scan_csv,
)
from vena.validation.audit import audit_harmonisation
from vena.validation.io import (
    ReferenceCache,
    ScanSample,
    ShardInfo,
    build_index,
    discover_shards,
    iter_scans,
)
from vena.validation.plotting import (
    annotate_significance,
    method_order,
    method_palette,
    setup_figure_style,
)
from vena.validation.regions import region_masks
from vena.validation.registry import (
    ABLATION_FAMILY,
    COHORT_RING,
    COMPETITOR_FAMILY,
    METHOD_ROLE,
    METHOD_SPECS,
    RING_A_COHORTS,
    RING_B_COHORTS,
    SELECTION_NFE,
    SUPPLEMENTARY,
    VENA_HEADLINE,
    MethodRole,
    MethodSpec,
    load_partitions,
    method_role,
    ring_of_cohort,
)
from vena.validation.stats import (
    MCID,
    HolmResult,
    SpearmanResult,
    WilcoxonResult,
    bootstrap_ci,
    cliffs_delta,
    collapse_to_patient,
    holm_bonferroni,
    paired_wilcoxon,
    spearman_with_bootstrap_ci,
)

__all__ = [
    "ABLATION_FAMILY",
    "COHORT_RING",
    "COMPETITOR_FAMILY",
    "MCID",
    "METHOD_ROLE",
    "METHOD_SPECS",
    "RING_A_COHORTS",
    "RING_B_COHORTS",
    "SELECTION_NFE",
    "SUPPLEMENTARY",
    "VENA_HEADLINE",
    "HolmResult",
    "MethodRole",
    "MethodSpec",
    "ReferenceCache",
    "ScanSample",
    "ShardInfo",
    "SpearmanResult",
    "WilcoxonResult",
    "annotate_significance",
    "audit_harmonisation",
    "bootstrap_ci",
    "build_index",
    "cliffs_delta",
    "collapse_to_patient",
    "discover_shards",
    "holm_bonferroni",
    "iter_scans",
    "load_partitions",
    "make_run_dir",
    "method_order",
    "method_palette",
    "method_role",
    "paired_wilcoxon",
    "region_masks",
    "ring_of_cohort",
    "setup_figure_style",
    "spearman_with_bootstrap_ci",
    "symlink_latest",
    "write_decision_json",
    "write_per_scan_csv",
]
