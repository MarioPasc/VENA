"""Unit tests for the ``decoder_lpl_profile`` decision.json writer + aggregator.

Covers:

* ``DecoderLplDecisionV1`` validates the v1.0 contract on a hand-built
  payload.
* ``write_decision_json`` round-trips through ``assert_*_valid``.
* ``aggregate`` consumes synthetic CSVs in ``tmp_path``, computes a
  sensible recipe, and produces every deliverable file (decision.json,
  report.md, all six figures).
* ``aggregate`` honours the v4 brain-mask hard gate: a synthetic
  3× ratio inflation on v4 forces ``v4_brain_mask_status='broken_drop_v4'``
  and drops v4 from ``allowed_variants``.
* ``update_latest_symlink`` atomically points ``LATEST`` at the new dir.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from vena.preflight.decoder_lpl_profile import (
    DECISION_SCHEMA_VERSION,
    DecoderLplDecisionV1,
    aggregate,
    assert_decoder_lpl_decision_valid,
    update_latest_symlink,
    write_decision_json,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Decision schema round-trip
# ---------------------------------------------------------------------------


def _minimal_decision() -> DecoderLplDecisionV1:
    return DecoderLplDecisionV1(
        schema_version=DECISION_SCHEMA_VERSION,
        produced_at="2026-06-18T12:00:00Z",
        producer="vena.preflight.decoder_lpl_profile:1.0",
        n_patients_run=18,
        patients_per_cohort={"UCSF-PDGM": 3},
        A_recommended=[2, 5],
        w_l={2: 1.0, 5: 1.0},
        t_min=0.7,
        outlier_k={2: 5.0, 5: 5.0},
        region_recipe={
            "alpha_wt": 2.0,
            "alpha_notwt": 3.0,
            "soft_region": False,
            "per_cohort_overrides": None,
        },
        allowed_variants=["v0", "v1", "v2", "v3"],
        v4_brain_mask_status="broken_drop_v4",
    )


def test_decision_round_trip(tmp_path: Path) -> None:
    decision = _minimal_decision()
    path = tmp_path / "decision.json"
    write_decision_json(path, decision)
    parsed = assert_decoder_lpl_decision_valid(path)
    assert parsed.schema_version == "1.0"
    assert parsed.allowed_variants == ["v0", "v1", "v2", "v3"]
    assert parsed.v4_brain_mask_status == "broken_drop_v4"


def test_decision_rejects_bad_schema_version(tmp_path: Path) -> None:
    """A future schema_version like ``'2.0'`` must fail validation."""
    payload = _minimal_decision().model_dump(mode="json")
    payload["schema_version"] = "2.0"
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(Exception):  # pydantic.ValidationError
        assert_decoder_lpl_decision_valid(path)


def test_decision_w_l_int_keys_coerced() -> None:
    """JSON deserialises dict keys as strings; the validator must coerce."""
    payload = _minimal_decision().model_dump(mode="json")
    payload["w_l"] = {"2": 1.0, "5": 1.0}
    parsed = DecoderLplDecisionV1.model_validate(payload)
    assert set(parsed.w_l) == {2, 5}


# ---------------------------------------------------------------------------
# Aggregator on synthetic shard CSVs
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _synthesise_cell_csvs(
    out_dir: Path,
    *,
    cohorts: tuple[str, ...] = ("UCSF-PDGM", "BraTS-GLI"),
    patients_per_cohort: int = 2,
    variants: tuple[str, ...] = ("v0", "v1", "v4"),
    inflate_v4: bool = True,
) -> None:
    """Write a complete set of single-shard CSVs under ``tables/``.

    Constructed so that:
    * v4 ratio = 3 × v0 ratio at block 5 (so the v4 gate trips).
    * Block 5 error concentration > block 2 (so A_recommended chooses both).
    * The x̂_1 reliability monotonically decreases (so t_min knee is in range).
    """
    tables = out_dir / "tables"
    blocks = (0, 1, 2, 3, 4, 5)
    ts = (0.3, 0.5, 0.7, 0.9)

    mag_rows = []
    pc_rows = []
    out_rows = []
    sep_rows = []
    err_rows = []
    rel_rows = []
    drift_rows = []
    ratio_rows = []
    empty_rows = []

    for cohort in cohorts:
        for p in range(patients_per_cohort):
            pid = f"{cohort}-{p:02d}"
            for v in variants:
                v4_factor = 3.0 if (v == "v4" and inflate_v4) else 1.0
                # mag rows: block-5 magnitude > block-2 (matching N=4 pilot).
                for blk in blocks:
                    mag = 1.5 if blk == 2 else 2.5 if blk == 5 else 1.0
                    mag_rows.append(
                        {
                            "cohort": cohort,
                            "patient_id": pid,
                            "variant": v,
                            "block_idx": blk,
                            "mean_norm": mag,
                            "std_norm": 0.1,
                            "p99_norm": mag * 1.5,
                        }
                    )
                    # outlier table.
                    out_rows.append(
                        {
                            "cohort": cohort,
                            "patient_id": pid,
                            "variant": v,
                            "block_idx": blk,
                            "mad_median": 0.5,
                            "recommended_k": 5.0,
                        }
                    )
                    # per-channel
                    for c_idx in range(4):
                        pc_rows.append(
                            {
                                "cohort": cohort,
                                "patient_id": pid,
                                "variant": v,
                                "block_idx": blk,
                                "channel_idx": c_idx,
                                "mean_L_dec": mag * (1.0 + 0.1 * c_idx),
                                "p99_L_dec": mag * 1.5,
                                "mad": 0.5,
                            }
                        )
                # Pre/post + error concentration — make the WT residual
                # larger at blocks {2, 5} so they win the A_recommended pick.
                for blk in blocks:
                    wt_dist = (1.0 if blk in (2, 5) else 0.5) * v4_factor
                    nw_dist = 1.0 if blk in (2, 5) else 0.5
                    sep_rows.append(
                        {
                            "cohort": cohort,
                            "patient_id": pid,
                            "variant": v,
                            "block_idx": blk,
                            "region": "WT",
                            "sep_dist": wt_dist,
                        }
                    )
                    sep_rows.append(
                        {
                            "cohort": cohort,
                            "patient_id": pid,
                            "variant": v,
                            "block_idx": blk,
                            "region": "notWT",
                            "sep_dist": nw_dist,
                        }
                    )
                    err_rows.append(
                        {
                            "cohort": cohort,
                            "patient_id": pid,
                            "variant": v,
                            "block_idx": blk,
                            "region": "WT",
                            "residual_dist": wt_dist * 0.5,
                        }
                    )
                    err_rows.append(
                        {
                            "cohort": cohort,
                            "patient_id": pid,
                            "variant": v,
                            "block_idx": blk,
                            "region": "notWT",
                            "residual_dist": nw_dist * 0.5,
                        }
                    )
                # x_1 reliability — monotonically decreasing in t.
                for t in ts:
                    for blk in blocks:
                        rel_rows.append(
                            {
                                "cohort": cohort,
                                "patient_id": pid,
                                "variant": v,
                                "t": t,
                                "block_idx": blk,
                                "feature_distance_to_target": 1.0 - 0.5 * t,
                            }
                        )

    _write_csv(tables / "per_block_magnitude.csv", mag_rows)
    _write_csv(tables / "per_channel_L_dec_distribution.csv", pc_rows)
    _write_csv(tables / "outlier_threshold.csv", out_rows)
    _write_csv(tables / "pre_post_separation.csv", sep_rows)
    _write_csv(tables / "error_concentration.csv", err_rows)
    _write_csv(tables / "x1_reliability_vs_t.csv", rel_rows)

    # Phase 3 tables — the aggregator can also derive ratio_rows from
    # sep_rows but it expects pre-computed CSVs from the per-shard run.
    # Pre-fill them so the test mirrors the production layout.
    for cohort in cohorts:
        for p in range(patients_per_cohort):
            pid = f"{cohort}-{p:02d}"
            v0_b5 = next(
                r
                for r in sep_rows
                if r["cohort"] == cohort
                and r["patient_id"] == pid
                and r["variant"] == "v0"
                and r["block_idx"] == 5
                and r["region"] == "WT"
            )
            v0_b5_nw = next(
                r
                for r in sep_rows
                if r["cohort"] == cohort
                and r["patient_id"] == pid
                and r["variant"] == "v0"
                and r["block_idx"] == 5
                and r["region"] == "notWT"
            )
            ratio_v0 = v0_b5["sep_dist"] / v0_b5_nw["sep_dist"]
            for v in variants:
                if v == "v0":
                    continue
                v_sep = next(
                    r
                    for r in sep_rows
                    if r["cohort"] == cohort
                    and r["patient_id"] == pid
                    and r["variant"] == v
                    and r["block_idx"] == 5
                    and r["region"] == "WT"
                )
                v_nw = next(
                    r
                    for r in sep_rows
                    if r["cohort"] == cohort
                    and r["patient_id"] == pid
                    and r["variant"] == v
                    and r["block_idx"] == 5
                    and r["region"] == "notWT"
                )
                ratio_v = v_sep["sep_dist"] / v_nw["sep_dist"]
                drift = abs(ratio_v - ratio_v0) / abs(ratio_v0)
                drift_rows.append(
                    {
                        "cohort": cohort,
                        "patient_id": pid,
                        "variant": v,
                        "block_idx": 5,
                        "ratio_variant": ratio_v,
                        "ratio_v0": ratio_v0,
                        "drift_value": drift,
                        "passes_gate": drift < 0.2,
                    }
                )
    for cohort in cohorts:
        ratio_rows.append(
            {
                "cohort": cohort,
                "n_patients": patients_per_cohort,
                "ratio_median": 1.0,
                "ratio_p25": 0.9,
                "ratio_p75": 1.1,
            }
        )
        empty_rows.append(
            {
                "cohort": cohort,
                "n_total": patients_per_cohort,
                "n_empty_wt": 0,
                "fraction": 0.0,
            }
        )
    _write_csv(tables / "drift_per_patient_variant.csv", drift_rows)
    _write_csv(tables / "per_cohort_W_nW_ratio.csv", ratio_rows)
    _write_csv(tables / "empty_wt_rate.csv", empty_rows)


def test_aggregate_emits_every_deliverable(tmp_path: Path) -> None:
    """End-to-end: synthetic CSVs → decision.json + report.md + figures."""
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    _synthesise_cell_csvs(out_dir, inflate_v4=True)

    decision = aggregate(out_dir, cohorts=["UCSF-PDGM", "BraTS-GLI"])

    # decision.json validates against the schema.
    parsed = assert_decoder_lpl_decision_valid(out_dir / "decision.json")
    assert parsed.A_recommended  # non-empty
    assert set(parsed.w_l) == set(parsed.A_recommended)

    # report.md exists and references at least one figure.
    report = (out_dir / "report.md").read_text()
    assert "decoder_lpl_profile" in report
    assert "Figures" in report

    # All six aggregate figures should be present.
    figs = out_dir / "figures"
    for name in (
        "magnitude_curve.png",
        "channel_concentration_block2_vs_block5.png",
        "separation_per_region.png",
        "t_min_knee.png",
        "drift_heatmap.png",
        "inter_cohort_ratio_box.png",
    ):
        assert (figs / name).is_file(), f"missing figure {name}"

    # Synthesised data: v4 inflation triggers the brain-mask hard gate.
    assert parsed.v4_brain_mask_status == "broken_drop_v4"
    assert "v4" not in parsed.allowed_variants


def test_aggregate_v4_ok_when_no_inflation(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    _synthesise_cell_csvs(out_dir, inflate_v4=False)
    decision = aggregate(out_dir, cohorts=["UCSF-PDGM", "BraTS-GLI"])
    assert decision.v4_brain_mask_status == "ok"
    assert "v4" in decision.allowed_variants


def test_update_latest_symlink(tmp_path: Path) -> None:
    root = tmp_path / "preflights"
    a = root / "2026-06-18T00-00-00Z"
    a.mkdir(parents=True)
    update_latest_symlink(a)
    assert (root / "LATEST").is_symlink()
    assert (root / "LATEST").resolve() == a.resolve()

    # Atomic replace: second call points to the new dir.
    b = root / "2026-06-19T00-00-00Z"
    b.mkdir()
    update_latest_symlink(b)
    assert (root / "LATEST").resolve() == b.resolve()
