"""Pydantic schema for self-describing H5 manifests.

A *manifest* describes — once, in code — what an H5 file produced by this
codebase looks like: which datasets it contains, their dtypes, units, and a
short human description. The manifest is **serialised into the H5 file itself**
as the ``manifest_json`` root attribute, so a consumer that has access to only
the file (no source checkout) can still walk the structure and interpret every
field. This is the concrete realisation of principles 1, 3, 4 and 7 from
``.claude/rules/h5-design-principles.md``.

Manifests are dataset-specific: each cohort/domain pair (e.g. UCSF-PDGM ×
image-domain, UCSF-PDGM × latent-domain, Málaga × image-domain) ships its own
manifest module under ``vena.data.h5.<cohort>.<domain>.manifest`` and re-exports
a module-level ``<NAME>_MANIFEST`` constant. The shared layer only constrains
the *structure* of a manifest, not its contents.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .exceptions import H5SchemaError

# H5-native dtype tags used in every manifest. ``vlen-str`` maps to
# ``h5py.special_dtype(vlen=str)`` at write time; the rest map directly to
# numpy dtypes of the same name.
DTypeTag = Literal[
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "bool",
    "vlen-str",
]

DatasetKind = Literal["image", "mask", "metadata", "id", "split"]


class DatasetSpec(BaseModel):
    """Self-describing entry for one HDF5 dataset.

    Attributes
    ----------
    path : str
        Slash-delimited HDF5 path, e.g. ``"images/t1pre"`` or ``"metadata/age"``.
    dtype : DTypeTag
        On-disk dtype tag.
    kind : DatasetKind
        Coarse semantic role; lets consumers filter (e.g. "all images").
    units : str
        Physical units of the stored values, or ``"dimensionless"`` /
        ``"label"`` / ``"intensity_au"`` for unitless quantities.
    description : str
        One-sentence semantic meaning. Written verbatim into the dataset's
        ``description`` attr.
    leading_dim : str | None
        Name of the first axis when the dataset is stacked across scans
        (e.g. ``"n_scans"``). ``None`` for scalars or single-vector datasets.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    dtype: DTypeTag
    kind: DatasetKind
    units: str
    description: str
    leading_dim: str | None = None

    @field_validator("path")
    @classmethod
    def _no_leading_slash(cls, v: str) -> str:
        if v.startswith("/"):
            raise ValueError(f"dataset path must be relative (no leading '/'): {v!r}")
        if not v:
            raise ValueError("dataset path must be non-empty")
        return v


class H5Manifest(BaseModel):
    """Top-level description of an H5 file produced by this codebase.

    Attributes
    ----------
    schema_version : str
        Semantic version of the manifest itself; bump on breaking changes.
    cohort : str
        Cohort tag (e.g. ``"UCSF-PDGM"``, ``"Malaga"``).
    domain : str
        Representation domain (``"image"`` or ``"latent"``).
    expected_shape : tuple[int, ...] | None
        Fixed spatial shape every image/mask dataset must match, or ``None``
        when the producer accepts variable-shape volumes (per-patient group
        layout). For UCSF-PDGM image-domain this is ``(240, 240, 155)``.
    datasets : list[DatasetSpec]
        Every dataset the file must contain. The order is preserved on write.
    splits_spec : dict[str, str]
        Free-form documentation of the layout under ``splits/``. Each key is
        a top-level split group, each value a one-line description. Stored
        only for human reference; ``validate_h5`` does not enforce it.
    extras : dict[str, str]
        Cohort-specific notes (e.g. ``{"intensity_policy": "raw, no rescale"}``)
        that should round-trip into the file's ``manifest_json`` attr.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str
    cohort: str
    domain: Literal["image", "latent"]
    expected_shape: tuple[int, ...] | None = None
    datasets: list[DatasetSpec]
    splits_spec: dict[str, str] = Field(default_factory=dict)
    extras: dict[str, str] = Field(default_factory=dict)

    @field_validator("datasets")
    @classmethod
    def _unique_paths(cls, v: list[DatasetSpec]) -> list[DatasetSpec]:
        seen: set[str] = set()
        for d in v:
            if d.path in seen:
                raise ValueError(f"duplicate dataset path in manifest: {d.path!r}")
            seen.add(d.path)
        return v

    # ----- convenience accessors ---------------------------------------------

    def by_kind(self, kind: DatasetKind) -> list[DatasetSpec]:
        return [d for d in self.datasets if d.kind == kind]

    def get(self, path: str) -> DatasetSpec:
        for d in self.datasets:
            if d.path == path:
                return d
        raise H5SchemaError(f"no dataset {path!r} in manifest")

    def to_json(self) -> str:
        """Serialise to a stable JSON string suitable for an H5 root attr."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, s: str) -> H5Manifest:
        return cls.model_validate_json(s)
