"""Markdown + bar-chart writer for the cohort_dedup preflight."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_report(run_dir: Path, payload: dict[str, Any]) -> None:
    """Write ``report.md`` + ``figures/keep_vs_reject.png`` under ``run_dir``."""
    cohorts = payload["cohorts"]
    names = list(cohorts.keys())
    kept = [cohorts[n]["n_kept"] for n in names]
    rejected = [cohorts[n]["n_rejected"] for n in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = list(range(len(names)))
    ax.bar(x, kept, label="kept", color="#3a7d44")
    ax.bar(x, rejected, bottom=kept, label="rejected", color="#c44536")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("patients")
    ax.set_title("Cohort dedup: kept vs rejected per cohort")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "keep_vs_reject.png", dpi=120)
    plt.close(fig)

    lines: list[str] = []
    lines.append("# Cohort deduplication report")
    lines.append("")
    lines.append(f"- produced_at: {payload['produced_at']}")
    lines.append(f"- producer: {payload['producer']}")
    lines.append(f"- corpus_registry: `{payload['corpus_registry_path']}`")
    lines.append(f"- mapping_xlsx: `{payload['mapping_xlsx_path']}`")
    lines.append(f"- priority: `{payload['priority']}`")
    lines.append(f"- policy: `{payload['policy']}`")
    lines.append("")
    t = payload["totals"]
    lines.append(
        f"**Totals**: {t['n_cohorts']} cohorts | "
        f"in: {t['n_patients_total_in']} | "
        f"kept: {t['n_patients_total_kept']} | "
        f"rejected: {t['n_patients_total_rejected']}"
    )
    lines.append("")
    lines.append("## Per-cohort")
    lines.append("")
    lines.append("| Cohort | n_total | n_kept | n_rejected | bridge_field |")
    lines.append("|---|---:|---:|---:|---|")
    for name in names:
        e = cohorts[name]
        lines.append(
            f"| {name} | {e['n_total']} | {e['n_kept']} | {e['n_rejected']} | "
            f"`{e['bridge_field']}` |"
        )
    lines.append("")
    lines.append("![Kept vs rejected](figures/keep_vs_reject.png)")
    lines.append("")
    lines.append("## Resolved overlaps")
    lines.append("")
    audit = payload.get("overlap_audit") or []
    if not audit:
        lines.append("(none)")
    else:
        lines.append("| bridge | TCIA source | kept | dropped |")
        lines.append("|---|---|---|---|")
        for entry in audit[:50]:
            dropped = ", ".join(f"{c}:{p}" for c, p in entry["dropped"])
            kept_pid = entry.get("kept_pid") or "(implicit)"
            lines.append(
                f"| `{entry['bridge']}` | {entry.get('tcia_source', '?')} | "
                f"{entry['kept_cohort']}:{kept_pid} | {dropped} |"
            )
        if len(audit) > 50:
            lines.append("")
            lines.append(f"(showing 50 of {len(audit)}; full list in `decision.json`)")
    lines.append("")
    lines.append("## Unresolvable overlaps")
    lines.append("")
    ur = payload.get("unresolvable_overlaps") or []
    if not ur:
        lines.append("(none)")
    else:
        for u in ur:
            lines.append(
                f"- **{u['cohort_a']} <-> {u['cohort_b']}** "
                f"({u['n_candidate_groups']} candidate groups): {u['reason']}"
            )
    lines.append("")

    (run_dir / "report.md").write_text("\n".join(lines))
