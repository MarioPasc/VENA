"""Article-figure/table studies built on the frozen validation per-scan CSVs.

Each study consumes a routine's `<routine>/LATEST/per_scan/*.csv` (produced by the
Phase-2 sweep) and renders the paper's tables and figures. Studies never re-run a
sweep and never reimplement statistics: method metadata comes from
``vena.validation.registry`` and all statistics from ``vena.validation.stats`` /
``vena.validation.spatial_residual`` (single code path — see
``.claude/skills/orchestrate/SKILL.md`` §7). Cross-study presentation glue lives
in :mod:`routines.validation.studies._shared`.
"""
