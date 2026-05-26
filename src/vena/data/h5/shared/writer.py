"""Thin writer wrapper around ``h5py`` that enforces our schema policy.

Every dataset gets:

* ``units``, ``description``, ``dtype`` attributes (principle 4 of
  ``.claude/rules/h5-design-principles.md``);
* ``leading_dim`` when supplied;
* ``gzip`` compression level 4 on bulk arrays (≥ 2 elements in the leading
  axis or > 1 MiB total);
* ``chunks=(1, *rest)`` whenever the dataset has a leading axis matching
  ``leading_dim`` — reading one scan is one chunk read (principle 6).

The writer also stamps every required root attribute on file creation so the
producer cannot forget any of them.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

from .schema import DatasetSpec, DTypeTag, H5Manifest

logger = logging.getLogger(__name__)

_VLEN_STR = h5py.special_dtype(vlen=str)


def _dtype_to_numpy(tag: DTypeTag) -> Any:
    """Resolve a manifest dtype tag to its numpy / h5py-special equivalent."""
    if tag == "vlen-str":
        return _VLEN_STR
    return np.dtype(tag)


class H5Writer:
    """Context-managed wrapper around ``h5py.File`` opened in write mode.

    Parameters
    ----------
    path : Path
        Destination file. Parent directories are created on demand. If the
        file exists the constructor raises unless ``overwrite=True``.
    manifest : H5Manifest
        Manifest serialised into the file as the ``manifest_json`` root attr.
    config_json : str
        JSON dump of the producing config (e.g. resolved Pydantic config).
    producer : str
        Identifier of the producing module + version, e.g.
        ``"vena.data.h5.ucsf_pdgm.image_domain.convert:0.1.0"``.
    created_at : str
        ISO-8601 UTC timestamp.
    git_sha : str | None
        ``HEAD`` SHA of the producing repo, or ``None`` outside a repo.
    overwrite : bool
        Whether to unlink an existing file before opening.
    """

    def __init__(
        self,
        path: Path,
        *,
        manifest: H5Manifest,
        config_json: str,
        producer: str,
        created_at: str,
        git_sha: str | None,
        overwrite: bool = False,
    ) -> None:
        self.path = Path(path)
        self.manifest = manifest
        self._config_json = config_json
        self._producer = producer
        self._created_at = created_at
        self._git_sha = git_sha
        self._overwrite = overwrite
        self._f: h5py.File | None = None

    # ------------------------------------------------------------------ open

    def __enter__(self) -> H5Writer:
        if self.path.exists():
            if not self._overwrite:
                raise FileExistsError(
                    f"Output H5 already exists: {self.path}. "
                    "Pass overwrite=True or delete it first."
                )
            self.path.unlink()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = h5py.File(self.path, "w")
        self._stamp_root_attrs()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._f is not None:
            try:
                self._f.flush()
            finally:
                self._f.close()
                self._f = None

    @property
    def file(self) -> h5py.File:
        if self._f is None:
            raise RuntimeError("H5Writer used outside of `with` block")
        return self._f

    # ------------------------------------------------------------------ root

    def _stamp_root_attrs(self) -> None:
        f = self.file
        f.attrs["schema_version"] = self.manifest.schema_version
        f.attrs["cohort"] = self.manifest.cohort
        f.attrs["domain"] = self.manifest.domain
        f.attrs["created_at"] = self._created_at
        f.attrs["producer"] = self._producer
        f.attrs["config_json"] = self._config_json
        f.attrs["manifest_json"] = self.manifest.to_json()
        f.attrs["git_sha"] = self._git_sha if self._git_sha is not None else "unknown"

    # ------------------------------------------------------------------ alloc

    def create_stacked(
        self,
        spec: DatasetSpec,
        n: int,
        spatial_shape: tuple[int, ...],
    ) -> h5py.Dataset:
        """Allocate ``spec`` as a stacked tensor of shape ``(n, *spatial_shape)``.

        Compression and chunking defaults from this writer apply. The dataset is
        zero-initialised; callers fill rows in arbitrary order via item indexing.
        """
        shape = (n, *spatial_shape)
        chunks = (1, *spatial_shape)
        dset = self.file.create_dataset(
            spec.path,
            shape=shape,
            dtype=_dtype_to_numpy(spec.dtype),
            chunks=chunks,
            compression="gzip",
            compression_opts=4,
        )
        self._stamp_dataset_attrs(dset, spec)
        return dset

    def create_1d(self, spec: DatasetSpec, n: int) -> h5py.Dataset:
        """Allocate a per-scan 1D dataset (e.g. ``ids``, ``metadata/age``)."""
        dset = self.file.create_dataset(
            spec.path,
            shape=(n,),
            dtype=_dtype_to_numpy(spec.dtype),
        )
        self._stamp_dataset_attrs(dset, spec)
        return dset

    def write_vlen_str_1d(self, path: str, values: list[str]) -> h5py.Dataset:
        """Write a 1D vlen-str dataset without manifest enforcement.

        Used for splits (e.g. ``splits/test``, ``splits/cv/fold_0/train``) which
        carry self-describing attrs but are not stacked image data.
        """
        dset = self.file.create_dataset(
            path,
            data=np.asarray(values, dtype=object),
            dtype=_VLEN_STR,
        )
        dset.attrs["units"] = "dimensionless"
        dset.attrs["description"] = "Patient identifiers (UCSF-PDGM-NNNN)."
        dset.attrs["dtype"] = "vlen-str"
        return dset

    # ------------------------------------------------------------------ attrs

    def _stamp_dataset_attrs(self, dset: h5py.Dataset, spec: DatasetSpec) -> None:
        dset.attrs["units"] = spec.units
        dset.attrs["description"] = spec.description
        dset.attrs["dtype"] = spec.dtype
        if spec.leading_dim is not None:
            dset.attrs["leading_dim"] = spec.leading_dim


@contextmanager
def open_writer(
    path: Path,
    *,
    manifest: H5Manifest,
    config_json: str,
    producer: str,
    created_at: str,
    git_sha: str | None,
    overwrite: bool = False,
) -> Iterator[H5Writer]:
    """Functional alias for ``with H5Writer(...) as w:``."""
    w = H5Writer(
        path,
        manifest=manifest,
        config_json=config_json,
        producer=producer,
        created_at=created_at,
        git_sha=git_sha,
        overwrite=overwrite,
    )
    with w as opened:
        yield opened


def assign_row(dset: h5py.Dataset, index: int, array: NDArray[Any]) -> None:
    """Write one row into a stacked dataset, with a shape sanity check.

    Avoids the silent broadcasting that ``dset[i] = array`` allows when the
    spatial dims of ``array`` happen to match a permutation of the dataset's
    spatial dims.
    """
    expected = tuple(dset.shape[1:])
    if tuple(array.shape) != expected:
        raise ValueError(
            f"shape mismatch writing to {dset.name}[{index}]: "
            f"got {array.shape}, expected {expected}"
        )
    dset[index] = array
