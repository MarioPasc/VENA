"""Per-subject PDF + JSON report (spec §7.1)."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ..core.dataclasses import (
    SubjectInputs,
    TestOutcome,
    ValidationResult,
)
from ._plots import mosaic_three_axis

_STYLES = getSampleStyleSheet()


def _format_value(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if not np.isfinite(v):
            return "nan"
        return f"{v:.3g}"
    return str(v)


def _format_threshold(t) -> str:
    if t is None:
        return "—"
    if isinstance(t, tuple):
        return f"[{_format_value(t[0])}, {_format_value(t[1])}]"
    return _format_value(t)


def _outcomes_to_table(outcomes: list[TestOutcome], headers: list[str]) -> Table:
    data = [headers]
    for o in outcomes:
        data.append(
            [
                o.prior_id or "—",
                o.roi_id or "—",
                _format_value(o.metric_value),
                _format_threshold(o.threshold),
                "PASS" if o.passed else ("WARN" if o.severity == "warning" else "FAIL"),
                Paragraph(o.diagnostic, _STYLES["BodyText"]),
            ]
        )
    style = TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]
    )
    for i, o in enumerate(outcomes, start=1):
        if o.severity == "error" and not o.passed:
            style.add("BACKGROUND", (0, i), (-1, i), colors.mistyrose)
        elif o.severity == "warning":
            style.add("BACKGROUND", (0, i), (-1, i), colors.lightyellow)
    t = Table(data, colWidths=[18 * mm, 22 * mm, 18 * mm, 22 * mm, 14 * mm, 80 * mm])
    t.setStyle(style)
    return t


def write_per_subject_json(result: ValidationResult, out_path: Path) -> Path:
    """Dump the subject's outcomes as a versioned JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "subject_id": result.subject_id,
        "overall_passed": result.overall_passed,
        "aborted": result.aborted,
        "abort_reason": result.abort_reason,
        "failed_priors": sorted(result.failed_priors),
        "outcomes": [asdict(o) for o in result.outcomes],
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return out_path


def write_per_subject_pdf(
    inputs: SubjectInputs,
    result: ValidationResult,
    delta_t1,
    out_path: Path,
    *,
    figures_dir: Path | None = None,
) -> Path:
    """Render the per-subject one-page-ish PDF."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
    )
    story: list = []
    h2 = _STYLES["Heading2"]
    body = _STYLES["BodyText"]

    meta = inputs.metadata
    status_word = "PASS" if result.overall_passed else ("ABORTED" if result.aborted else "FAIL")
    available_priors = sorted(inputs.derived_priors.keys())
    raw_avail = [m for m in ("cbf", "adc", "chi", "swan_mag") if getattr(inputs, m) is not None]
    header_lines = [
        f"<b>Subject:</b> {result.subject_id}  |  <b>Status:</b> {status_word}",
        f"<b>Age:</b> {meta.age or '—'}  <b>Sex:</b> {meta.sex or '—'}  "
        f"<b>Scanner:</b> {meta.scanner or '—'}  "
        f"<b>Field:</b> {meta.field_strength_t or '—'} T",
        f"<b>Pathology:</b> {meta.pathology or '—'}  <b>WHO grade:</b> {meta.who_grade or '—'}",
        f"<b>Raw priors:</b> {', '.join(raw_avail) or '—'}",
        f"<b>Derived priors:</b> {', '.join(available_priors) or '—'}",
    ]
    if result.abort_reason:
        header_lines.append(f"<b>Aborted:</b> {result.abort_reason}")
    for line in header_lines:
        story.append(Paragraph(line, body))
    story.append(Spacer(1, 4 * mm))

    by_test: dict[str, list[TestOutcome]] = defaultdict(list)
    for o in result.outcomes:
        by_test[o.test_id].append(o)

    headers = ["Prior", "ROI", "Metric", "Threshold", "Result", "Diagnostic"]
    for test_id, label in [
        ("T1_range_sanity", "T1 — Quantitative range sanity"),
        ("T2_atlas_localisation", "T2 — Anatomical localisation"),
        ("T3_t1gd_coherence", "T3 — T1Gd coherence"),
        ("T4_cross_modal", "T4 — Cross-modal coupling"),
        ("T5_reproducibility", "T5 — Test-retest reproducibility"),
    ]:
        story.append(Paragraph(label, h2))
        outs = by_test.get(test_id, [])
        if not outs:
            story.append(Paragraph("Not applicable for this subject.", body))
            story.append(Spacer(1, 3 * mm))
            continue
        # Sort error rows first, then warning, then info
        sev_order = {"error": 0, "warning": 1, "info": 2}
        outs_sorted = sorted(
            outs, key=lambda o: (sev_order.get(o.severity, 3), o.prior_id or "", o.roi_id or "")
        )
        story.append(_outcomes_to_table(outs_sorted, headers))
        story.append(Spacer(1, 3 * mm))

    # Visual QC strip
    if figures_dir is not None:
        mosaic_path = figures_dir / f"{result.subject_id}_qc.png"
        vols: dict = {"T1pre": np.asarray(inputs.t1pre.array)}
        vols["T1Gd"] = np.asarray(inputs.t1gd.array)
        if delta_t1 is not None:
            vols["ΔT1 (z)"] = np.asarray(delta_t1)
        for name in ("cbf", "cell", "sus", "itss", "adc_rel", "vessel_soft"):
            if name in inputs.derived_priors:
                vols[name] = np.asarray(inputs.derived_priors[name].array)
        try:
            mosaic_three_axis(vols, np.asarray(inputs.brain_mask.array), mosaic_path)
            story.append(PageBreak())
            story.append(Paragraph("Visual QC strip", h2))
            story.append(
                Image(str(mosaic_path), width=170 * mm, height=200 * mm, kind="proportional")
            )
        except Exception as exc:
            story.append(Paragraph(f"Mosaic rendering failed: {exc}", body))

    doc.build(story)
    return out_path
