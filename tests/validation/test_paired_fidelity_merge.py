"""Tests for the paired_fidelity sweep merge step's audit trail.

The sweep's merged ``decision.json`` is the artifact an auditor reads to
confirm the stale ``smoke_loginexa`` shard was excluded (SHARED_CONTRACTS
§3.1).  The exclusion happens at manifest-generation time, so the merge must
re-derive and report it rather than assert an empty list -- reporting ``[]``
would claim nothing was excluded, i.e. that the contamination the discovery
contract exists to prevent was not prevented.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vena.validation.io import discover_shards

pytestmark = pytest.mark.validation


def _write_shard(root: Path, tag: str, *, smoke: bool) -> None:
    d = root / tag
    d.mkdir(parents=True)
    payload: dict[str, object] = {"schema_version": "1.0", "run_id_tag": tag}
    if smoke:
        payload["smoke"] = {"enabled": True, "n_patients_per_cohort": 1}
    (d / "decision.json").write_text(json.dumps(payload))


def test_merge_recovers_skipped_smoke_shards_from_the_data_root(tmp_path: Path) -> None:
    """The merge can re-derive the skipped tags without the manifest's help.

    This is the property cli_merge relies on: it calls discover_shards on the
    same data_root the manifest step used, so the two cannot disagree.
    """
    root = tmp_path / "inference"
    root.mkdir()
    _write_shard(root, "picasso_shard_a_cheap", smoke=False)
    _write_shard(root, "picasso_ped_b_vena", smoke=False)
    _write_shard(root, "smoke_loginexa", smoke=True)

    discovery = discover_shards(root)

    assert discovery.skipped_smoke == ["smoke_loginexa"]
    accepted_tags = sorted(s.root.name for s in discovery.accepted)
    # The BraTS-PED backfill writes picasso_ped_* tags and is production: the
    # filter is the self-declared smoke flag, never a shard-name pattern.
    assert accepted_tags == ["picasso_ped_b_vena", "picasso_shard_a_cheap"]


def test_merge_does_not_hardcode_an_empty_skipped_list() -> None:
    """Guard: cli_merge must not regress to ``skipped_smoke_shards=[]``.

    A literal empty list here silently converts "a smoke shard was excluded"
    into "no shard was excluded" in the paper's headline artifact.  The
    hard-coded form shipped once already and read as correct because the
    exclusion genuinely had happened upstream -- only the report was false.
    """
    src = Path(__file__).resolve().parents[2] / "routines/validation/paired_fidelity/cli_merge.py"
    text = src.read_text()

    assert "skipped_smoke_shards=[]" not in text, (
        "cli_merge hard-codes an empty skipped_smoke_shards; it must pass the "
        "list re-derived from discover_shards(cfg.data_root)."
    )
    assert "discover_shards" in text, "cli_merge must re-derive the skipped shards from the root."
