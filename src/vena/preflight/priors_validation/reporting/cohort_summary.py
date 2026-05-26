"""Cohort PDF + Parquet + cohort.json + decision.json (spec §7.2 + §7.3)."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ..core.config import (
    COHORT_PASS_RATE_T1,
    COHORT_PASS_RATE_T2,
    COHORT_PASS_RATE_T3,
    COHORT_PASS_RATE_T4,
    EFFECT_SIZE_MIN_FOR_INFORMATIVE,
)
from ..core.dataclasses import CohortReport

_STYLES = getSampleStyleSheet()


def _outcomes_dataframe(report: CohortReport) -> pd.DataFrame:
    rows = []
    for vr in report.subjects:
        for o in vr.outcomes:
            rows.append(
                {
                    "subject_id": vr.subject_id,
                    "test_id": o.test_id,
                    "prior_id": o.prior_id,
                    "roi_id": o.roi_id,
                    "metric_name": o.metric_name,
                    "metric_value": (
                        float(o.metric_value)
                        if o.metric_value is not None and np.isfinite(float(o.metric_value))
                        else float("nan")
                    ),
                    "threshold": str(o.threshold),
                    "passed": bool(o.passed),
                    "severity": o.severity,
                    "diagnostic": o.diagnostic,
                }
            )
    return pd.DataFrame(rows)


def _decision_json(report: CohortReport) -> dict:
    """Build the machine-readable decision contract consumed by Phase-3 training."""
    return {
        "schema_version": "1.0",
        "n_subjects": report.n_subjects,
        "n_subjects_applicable": report.n_subjects_applicable,
        "cohort_pass_rate_overall": float(report.cohort_pass_rate_overall),
        "per_test_pass_rate": {
            k: (float(v) if v is not None else None) for k, v in report.per_test_pass_rate.items()
        },
        "per_prior_clearance": dict(report.per_prior_clearance),
        "atlas_versions": dict(report.atlas_versions),
        "routine_version": report.routine_version,
        "training_clearance": bool(report.training_clearance),
        "warnings": list(report.warnings),
    }


def _cohort_pdf(report: CohortReport, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
    )
    h1 = _STYLES["Heading1"]
    h2 = _STYLES["Heading2"]
    body = _STYLES["BodyText"]
    story: list = []

    story.append(Paragraph("Priors validation — cohort summary", h1))
    story.append(
        Paragraph(
            f"N subjects: {report.n_subjects}; applicable: {report.n_subjects_applicable}; "
            f"routine v{report.routine_version}",
            body,
        )
    )
    story.append(
        Paragraph(
            f"<b>Overall pass rate:</b> {report.cohort_pass_rate_overall:.1%}  |  "
            f"<b>Training clearance:</b> {'YES' if report.training_clearance else 'NO'}",
            body,
        )
    )
    story.append(
        Paragraph(
            "<i>This routine validates the priors, not the synthesiser. "
            "It is a sanity + informativeness check; passing does not imply "
            "model utility.</i>",
            body,
        )
    )
    story.append(Spacer(1, 4 * mm))

    # Per-test cohort pass-rate table
    story.append(Paragraph("Per-test cohort pass rates", h2))
    thresholds = {
        "T1_range_sanity": COHORT_PASS_RATE_T1,
        "T2_atlas_localisation": COHORT_PASS_RATE_T2,
        "T3_t1gd_coherence": COHORT_PASS_RATE_T3,
        "T4_cross_modal": COHORT_PASS_RATE_T4,
        "T5_reproducibility": None,
    }
    data = [["Test", "Cohort pass rate", "Spec target", "Status"]]
    for tid, target in thresholds.items():
        rate = report.per_test_pass_rate.get(tid)
        rate_str = f"{rate:.1%}" if rate is not None else "—"
        tgt_str = f"{target:.0%}" if target is not None else "—"
        if rate is None or target is None:
            status = "N/A"
        elif rate >= target:
            status = "OK"
        else:
            status = "BELOW TARGET"
        data.append([tid, rate_str, tgt_str, status])
    t = Table(data, colWidths=[60 * mm, 35 * mm, 30 * mm, 35 * mm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 4 * mm))

    # Per-prior clearance
    story.append(Paragraph("Per-prior clearance", h2))
    data = [["Prior", "Clearance"]]
    for prior, status in sorted(report.per_prior_clearance.items()):
        data.append([prior, status])
    t = Table(data, colWidths=[80 * mm, 40 * mm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 4 * mm))

    # T3 (prior, ROI) cohort-median |rho|
    story.append(Paragraph("T3 — Cohort-median |ρ| with effect-size guard", h2))
    t3_rows = defaultdict(list)
    for vr in report.subjects:
        for o in vr.outcomes:
            if o.test_id == "T3_t1gd_coherence" and o.metric_value is not None:
                t3_rows[(o.prior_id, o.roi_id)].append(float(o.metric_value))
    data = [["Prior × ROI", "Cohort median |ρ|", "N", "Informative?"]]
    for (prior, roi), vals in sorted(t3_rows.items()):
        arr = np.array([v for v in vals if np.isfinite(v)])
        med = float(np.median(np.abs(arr))) if arr.size else float("nan")
        informative = "yes" if med >= EFFECT_SIZE_MIN_FOR_INFORMATIVE else "no"
        data.append([f"{prior} / {roi}", f"{med:.3f}", str(arr.size), informative])
    if len(data) > 1:
        t = Table(data, colWidths=[80 * mm, 40 * mm, 15 * mm, 30 * mm])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ]
            )
        )
        story.append(t)

    # Warnings + outlier list
    if report.warnings:
        story.append(Spacer(1, 4 * mm))
        story.append(Paragraph("Cohort-level warnings", h2))
        for w in report.warnings:
            story.append(Paragraph(f"• {w}", body))

    outliers = [vr for vr in report.subjects if not vr.overall_passed]
    if outliers:
        story.append(Spacer(1, 4 * mm))
        story.append(Paragraph(f"Outlier subjects (n={len(outliers)})", h2))
        for vr in outliers[:30]:
            reasons = "; ".join(
                f"{o.test_id}:{o.metric_name}={o.diagnostic[:60]}"
                for o in vr.outcomes
                if o.severity == "error" and not o.passed
            )[:300]
            story.append(Paragraph(f"<b>{vr.subject_id}</b>: {reasons or '(aborted)'}", body))

    doc.build(story)
    return out_path


def write_cohort_outputs(report: CohortReport, out_dir: Path) -> dict[str, Path]:
    """Write cohort PDF + parquet + cohort.json + decision.json into ``out_dir``.

    Returns a dict mapping artefact name → resolved path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / "cohort_results.parquet"
    df = _outcomes_dataframe(report)
    df.to_parquet(parquet_path, index=False)

    cohort_json_path = out_dir / "cohort.json"
    cohort_payload = {
        "schema_version": "1.0",
        "report": {
            "n_subjects": report.n_subjects,
            "n_subjects_applicable": report.n_subjects_applicable,
            "per_test_pass_rate": {
                k: (float(v) if v is not None else None)
                for k, v in report.per_test_pass_rate.items()
            },
            "per_prior_clearance": dict(report.per_prior_clearance),
            "cohort_pass_rate_overall": float(report.cohort_pass_rate_overall),
            "atlas_versions": dict(report.atlas_versions),
            "routine_version": report.routine_version,
            "training_clearance": bool(report.training_clearance),
            "warnings": list(report.warnings),
        },
        "subjects": [
            {
                "subject_id": vr.subject_id,
                "overall_passed": vr.overall_passed,
                "aborted": vr.aborted,
                "abort_reason": vr.abort_reason,
                "failed_priors": sorted(vr.failed_priors),
                "outcomes": [asdict(o) for o in vr.outcomes],
            }
            for vr in report.subjects
        ],
    }
    cohort_json_path.write_text(json.dumps(cohort_payload, indent=2, sort_keys=True, default=str))

    decision_path = out_dir / "decision.json"
    decision_path.write_text(
        json.dumps(_decision_json(report), indent=2, sort_keys=True, default=str)
    )

    pdf_path = _cohort_pdf(report, out_dir / "cohort_summary.pdf")

    return {
        "cohort_pdf": pdf_path,
        "cohort_parquet": parquet_path,
        "cohort_json": cohort_json_path,
        "decision_json": decision_path,
    }
