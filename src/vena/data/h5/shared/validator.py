"""Validator pair for H5 artifacts produced by this codebase.

Implements the ``validate_<artifact>`` / ``assert_<artifact>_valid`` pattern
required by ``.claude/rules/h5-design-principles.md`` principle 7.

Used at two points:

1. **Producer-side**: the converter calls ``assert_h5_valid(out, manifest)``
   before returning the artifact path. A non-conformant file must never reach
   disk in a "successful" state.
2. **Consumer-side**: downstream loaders call ``assert_h5_valid`` at open-time
   to fail fast on version mismatches or corrupted files.

Validation is structural: it checks that every dataset declared in the
manifest exists with the declared dtype and the expected leading dimension,
and that the file's root attributes are present and consistent with the
embedded manifest. It does not look at voxel values.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np

from .exceptions import H5ValidationError
from .schema import DTypeTag, H5Manifest

logger = logging.getLogger(__name__)

_REQUIRED_ROOT_ATTRS = (
    "schema_version",
    "cohort",
    "domain",
    "created_at",
    "producer",
    "config_json",
    "manifest_json",
    "git_sha",
)


def _numpy_dtype_matches(dset: h5py.Dataset, tag: DTypeTag) -> bool:
    if tag == "vlen-str":
        return h5py.check_string_dtype(dset.dtype) is not None
    return dset.dtype == np.dtype(tag)


def validate_h5(path: Path | str, manifest: H5Manifest) -> list[str]:
    """Return a list of human-readable violations; empty list means valid.

    Never raises (other than for unreadable files). Callers wanting a hard
    failure should use ``assert_h5_valid``.
    """
    path = Path(path)
    violations: list[str] = []

    if not path.exists():
        return [f"file does not exist: {path}"]

    with h5py.File(path, "r") as f:
        # ---- root attrs --------------------------------------------------
        for attr in _REQUIRED_ROOT_ATTRS:
            if attr not in f.attrs:
                violations.append(f"missing root attr: {attr}")

        # version + cohort + domain must match the supplied manifest
        if "schema_version" in f.attrs and f.attrs["schema_version"] != manifest.schema_version:
            violations.append(
                f"schema_version mismatch: file={f.attrs['schema_version']!r} "
                f"manifest={manifest.schema_version!r}"
            )
        if "cohort" in f.attrs and f.attrs["cohort"] != manifest.cohort:
            violations.append(
                f"cohort mismatch: file={f.attrs['cohort']!r} manifest={manifest.cohort!r}"
            )
        if "domain" in f.attrs and f.attrs["domain"] != manifest.domain:
            violations.append(
                f"domain mismatch: file={f.attrs['domain']!r} manifest={manifest.domain!r}"
            )

        # manifest_json must parse and round-trip
        if "manifest_json" in f.attrs:
            try:
                embedded = H5Manifest.from_json(str(f.attrs["manifest_json"]))
            except Exception as exc:
                violations.append(f"manifest_json failed to parse: {exc}")
            else:
                if embedded.schema_version != manifest.schema_version:
                    violations.append(
                        "embedded manifest_json schema_version disagrees with caller's manifest"
                    )

        # ---- datasets ----------------------------------------------------
        n_scans = _infer_n_scans(f, manifest)
        spatial = manifest.expected_shape

        for spec in manifest.datasets:
            if spec.path not in f:
                violations.append(f"missing dataset: {spec.path}")
                continue
            dset = f[spec.path]
            if not isinstance(dset, h5py.Dataset):
                violations.append(f"{spec.path}: expected Dataset, got {type(dset).__name__}")
                continue
            if not _numpy_dtype_matches(dset, spec.dtype):
                violations.append(
                    f"{spec.path}: dtype mismatch — file={dset.dtype} manifest={spec.dtype}"
                )
            # leading-dim consistency
            if spec.leading_dim is not None and n_scans is not None:
                if dset.shape[0] != n_scans:
                    violations.append(
                        f"{spec.path}: leading dim {dset.shape[0]} != n_scans={n_scans}"
                    )
            # spatial-shape contract for image / mask kinds
            if spec.kind in {"image", "mask"} and spatial is not None:
                if tuple(dset.shape[1:]) != tuple(spatial):
                    violations.append(
                        f"{spec.path}: spatial shape {dset.shape[1:]} != expected {spatial}"
                    )
            # required self-describing attrs (principle 4)
            for attr in ("units", "description", "dtype"):
                if attr not in dset.attrs:
                    violations.append(f"{spec.path}: missing attr {attr!r}")

    return violations


def _infer_n_scans(f: h5py.File, manifest: H5Manifest) -> int | None:
    """Use the first dataset with a ``leading_dim`` to fix ``n_scans``.

    Returns ``None`` if no such dataset is present, in which case
    leading-dim checks are skipped (the file is empty of stacked data).
    """
    for spec in manifest.datasets:
        if spec.leading_dim is None:
            continue
        if spec.path in f:
            dset = f[spec.path]
            if isinstance(dset, h5py.Dataset) and dset.shape:
                return int(dset.shape[0])
    return None


def assert_h5_valid(path: Path | str, manifest: H5Manifest) -> None:
    """Raise ``H5ValidationError`` listing every violation; succeed silently."""
    violations = validate_h5(path, manifest)
    if violations:
        joined = "\n  - ".join(violations)
        raise H5ValidationError(
            f"H5 artifact failed validation against {manifest.cohort}/{manifest.domain} "
            f"manifest (v{manifest.schema_version}):\n  - {joined}"
        )
    logger.debug("H5 valid: %s", path)
