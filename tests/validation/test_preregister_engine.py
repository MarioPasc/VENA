"""Tests for the preregister engine's selection_nfe freezing.

The pre-registration artifact is what the P3 pre-registration-integrity claim
rests on, so the NFE each method is scored at must be legible in the artifact
itself -- not left implicit in ``vena.validation.registry.SELECTION_NFE``,
which an auditor could only recover by checking out code at the recorded
``git_sha``.

The engine's own conftest fixture is deliberately NOT reused here: it writes
``C0-Identity`` at NFE 5, whereas C0 only ever exists at NFE 1 on real disk
(SHARED_CONTRACTS §4).  These fixtures mirror the real NFE grid so the
on-disk-availability assertion is exercised against realistic input.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from routines.validation.preregister.engine import (
    PreregisterConfig,
    PreregisterEngine,
    PreregisterError,
)

pytestmark = pytest.mark.validation

_H = _W = _D = 8

#: (method, nfe_on_disk) mirroring the real grid for these two methods.
_C0 = "C0-Identity"
_VENA = "VENA-S1-v3b-rw"


def _vlen(grp: h5py.Group, name: str, values: list[str]) -> None:
    grp.create_dataset(
        name, data=np.asarray(values, dtype=object), dtype=h5py.string_dtype(encoding="utf-8")
    )


def _write_pred(path: Path, *, method: str, cohort: str, nfe: int, ring: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scan_ids, patient_ids = ["s1", "s2"], ["p1", "p2"]
    n = len(scan_ids)
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = "2.0"
        f.attrs["method"] = method
        f.attrs["cohort"] = cohort
        f.attrs["nfe"] = nfe
        f.attrs["ring"] = ring
        f.attrs["references_h5"] = f"references/{cohort}.h5"
        g = f.create_group("predictions")
        vol = np.zeros((n, _H, _W, _D), dtype=np.float32)
        g.create_dataset("t1c_synthetic_harmonised", data=vol)
        g.create_dataset("t1c_synthetic_raw", data=vol.copy())
        gm = f.create_group("masks")
        gm.create_dataset("brain", data=np.ones((n, _H, _W, _D), dtype=np.int8))
        gm.create_dataset("wt", data=np.ones((n, _H, _W, _D), dtype=np.int8))
        gt = f.create_group("metadata")
        _vlen(gt, "scan_id", scan_ids)
        _vlen(gt, "patient_id", patient_ids)
        _vlen(gt, "cohort", [cohort] * n)
        gt.create_dataset("nfe", data=np.full(n, nfe, dtype=np.int32))
        gt.create_dataset("inference_seconds", data=np.ones(n, dtype=np.float32))
        gt.create_dataset("peak_vram_mb", data=np.ones(n, dtype=np.float32))


def _write_ref(path: Path, *, cohort: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scan_ids, patient_ids = ["s1", "s2"], ["p1", "p2"]
    n = len(scan_ids)
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = "2.0"
        f.attrs["cohort"] = cohort
        g = f.create_group("reference")
        vol = np.zeros((n, _H, _W, _D), dtype=np.float32)
        for name in (
            "t1c_real_harmonised",
            "t1pre_harmonised",
            "t2_harmonised",
            "flair_harmonised",
        ):
            g.create_dataset(name, data=vol.copy())
        gm = f.create_group("masks")
        gm.create_dataset("brain", data=np.ones((n, _H, _W, _D), dtype=np.int8))
        gm.create_dataset("wt", data=np.ones((n, _H, _W, _D), dtype=np.int8))
        gt = f.create_group("metadata")
        _vlen(gt, "scan_id", scan_ids)
        _vlen(gt, "patient_id", patient_ids)
        _vlen(gt, "cohort", [cohort] * n)


def _build_tree(root: Path, grid: dict[str, list[int]]) -> Path:
    """Write a production shard exposing *grid* = {method: [nfe, ...]}."""
    shard = root / "prod_shard"
    shard.mkdir(parents=True)
    (shard / "decision.json").write_text(
        json.dumps({"schema_version": "1.0", "run_id_tag": "prod_shard"})
    )
    _write_ref(shard / "references" / "UCSF-PDGM.h5", cohort="UCSF-PDGM")
    for method, nfes in grid.items():
        for nfe in nfes:
            _write_pred(
                shard / "predictions" / method / "UCSF-PDGM" / f"nfe_{nfe:03d}.h5",
                method=method,
                cohort="UCSF-PDGM",
                nfe=nfe,
                ring="A",
            )
    return root


def _run(root: Path, out: Path) -> dict:
    engine = PreregisterEngine(
        PreregisterConfig(inference_root=root, output_root=out, corpus_registry=None)
    )
    run_dir = engine.run()
    return json.loads((run_dir / "ring_partitions.json").read_text())


def test_selection_nfe_is_frozen_into_the_artifact(tmp_path: Path) -> None:
    """The artifact records each method's pre-registered scoring NFE."""
    root = _build_tree(tmp_path / "inf", {_C0: [1], _VENA: [1, 2, 5, 10, 20]})
    payload = _run(root, tmp_path / "out")

    # C0 is scored at NFE 1, VENA at its selection NFE 5 -- not merely the
    # only, nor the largest, NFE present on disk.
    assert payload["selection_nfe"] == {_C0: 1, _VENA: 5}


def test_selection_nfe_absent_from_disk_is_a_stop_the_line_error(tmp_path: Path) -> None:
    """A method whose pre-registered NFE was never inferred must fail loudly.

    Silently freezing selection_nfe=5 when only NFE 100 exists would filter the
    method to zero rows in the headline table -- C0 would vanish and its canary
    would never fire.
    """
    root = _build_tree(tmp_path / "inf", {_C0: [1], _VENA: [1, 2, 10, 20]})  # no NFE 5
    with pytest.raises(PreregisterError, match="only NFEs"):
        _run(root, tmp_path / "out")


def test_unregistered_method_is_a_stop_the_line_error(tmp_path: Path) -> None:
    """A discovered method with no pre-registered NFE must fail loudly.

    Assigning its NFE after seeing test scores would be oracle selection.
    """
    root = _build_tree(tmp_path / "inf", {_C0: [1], "C9-Unregistered-Model": [1]})
    with pytest.raises(PreregisterError, match="no pre-registered selection_nfe"):
        _run(root, tmp_path / "out")
