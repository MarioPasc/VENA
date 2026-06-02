"""Stdlib parser for the BraTS-2021 ↔ TCIA mapping xlsx.

Deliberately uses only ``zipfile`` + ``xml.etree.ElementTree`` so the
cohort_dedup preflight carries no openpyxl dependency. The file
``BraTS2021_MappingToTCIA.xlsx`` is small (~80 KB, 1486 rows, 8 columns) and
parses in a fraction of a second.

Expected sheet ``TCIA`` with header columns

* ``Data Collection (as on TCIA+additional)``
* ``Site ID``
* ``PatientID on TCIA Radiology Portal``
* ``Study date (m/d/yyyy) per PatientID``
* ``BraTS2021 ID``
* ``Segmentation (Task 1) Cohort``
* ``MGMT (Task 2) Cohort``
* ``MGMT value``

Only the first sheet is read.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

_NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

# Header names as they appear in BraTS2021_MappingToTCIA.xlsx.
HDR_DATA_COLLECTION = "Data Collection (as on TCIA+additional)"
HDR_SITE_ID = "Site ID"
HDR_PORTAL_ID = "PatientID on TCIA Radiology Portal"
HDR_STUDY_DATE = "Study date (m/d/yyyy) per PatientID"
HDR_BRATS21_ID = "BraTS2021 ID"
HDR_SEG_COHORT = "Segmentation (Task 1) Cohort"
HDR_MGMT_COHORT = "MGMT (Task 2) Cohort"
HDR_MGMT_VALUE = "MGMT value"


@dataclass(frozen=True)
class MappingRow:
    """One row of the xlsx — a BraTS-2021 patient and its TCIA-side origin."""

    brats21_id: str
    data_collection: str
    site_id: str | None
    portal_id: str | None
    study_date: str | None
    seg_cohort: str | None
    mgmt_cohort: str | None
    mgmt_value: str | None


@dataclass(frozen=True)
class Brats2021Mapping:
    """Indexed view over the xlsx.

    Attributes
    ----------
    rows
        Every non-empty row of the xlsx, in sheet order.
    by_brats21_id
        Index keyed by ``BraTS2021_NNNNN``.
    by_collection
        ``data_collection -> frozenset[brats21_id]``.
    """

    rows: tuple[MappingRow, ...]
    by_brats21_id: dict[str, MappingRow] = field(repr=False)
    by_collection: dict[str, frozenset[str]] = field(repr=False)


def _col_index(ref: str) -> int:
    m = re.match(r"([A-Z]+)(\d+)", ref or "")
    if not m:
        return 0
    col = 0
    for ch in m.group(1):
        col = col * 26 + (ord(ch) - ord("A") + 1)
    return col - 1


def _load_shared(z: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in z.namelist():
        return []
    ss = ET.fromstring(z.read("xl/sharedStrings.xml"))
    out: list[str] = []
    for si in ss.findall("s:si", _NS):
        parts = [t.text or "" for t in si.findall(".//s:t", _NS)]
        out.append("".join(parts))
    return out


def _iter_rows(sheet_xml: bytes, shared: list[str]) -> list[list[str | None]]:
    sx = ET.fromstring(sheet_xml)
    rows: list[list[str | None]] = []
    for row in sx.findall("s:sheetData/s:row", _NS):
        cells: dict[int, str | None] = {}
        for c in row.findall("s:c", _NS):
            ref = c.attrib.get("r", "")
            ctype = c.attrib.get("t", "n")
            v = c.find("s:v", _NS)
            inline = c.find("s:is", _NS)
            if v is None and inline is None:
                val: str | None = None
            elif ctype == "s" and v is not None:
                val = shared[int(v.text or "0")]
            elif ctype == "str" and v is not None:
                val = v.text
            elif ctype == "inlineStr" and inline is not None:
                val = "".join(t.text or "" for t in inline.findall(".//s:t", _NS))
            else:
                val = v.text if v is not None else None
            cells[_col_index(ref)] = val
        if not cells:
            continue
        max_c = max(cells)
        rows.append([cells.get(i) for i in range(max_c + 1)])
    return rows


def parse_brats2021_mapping(path: Path | str) -> Brats2021Mapping:
    """Parse the BraTS-2021 ↔ TCIA xlsx into an indexed mapping.

    Parameters
    ----------
    path
        Absolute or repo-relative xlsx path.

    Returns
    -------
    Brats2021Mapping
        Indexed rows + by-BraTS21-ID and by-collection lookup tables.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the sheet is missing the expected header columns or is empty.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"mapping xlsx not found: {path}")
    with zipfile.ZipFile(path) as z:
        shared = _load_shared(z)
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        rel_map = {
            r.attrib["Id"]: r.attrib["Target"]
            for r in rels.findall(
                "{http://schemas.openxmlformats.org/package/2006/relationships}Relationship"
            )
        }
        sheet_node = wb.find("s:sheets/s:sheet", _NS)
        if sheet_node is None:
            raise ValueError(f"no sheet found in xlsx {path}")
        rid = sheet_node.attrib[
            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
        ]
        target = rel_map[rid].lstrip("/")
        sheet_path = target if target.startswith("xl/") else f"xl/{target}"
        sheet_xml = z.read(sheet_path)
        all_rows = _iter_rows(sheet_xml, shared)

    if not all_rows:
        raise ValueError(f"xlsx {path} is empty")
    header = [(h or "").strip() for h in all_rows[0]]
    required = (HDR_DATA_COLLECTION, HDR_BRATS21_ID)
    missing = [h for h in required if h not in header]
    if missing:
        raise ValueError(
            f"xlsx {path} is missing required column(s): {missing}; got header {header}"
        )

    idx = {h: header.index(h) for h in header if h}

    def _get(row: list[str | None], col: str) -> str | None:
        i = idx.get(col)
        if i is None or i >= len(row):
            return None
        v = row[i]
        if v in ("", None):
            return None
        return str(v).strip() or None

    rows: list[MappingRow] = []
    by_b21: dict[str, MappingRow] = {}
    by_coll: dict[str, set[str]] = {}
    for r in all_rows[1:]:
        b21 = _get(r, HDR_BRATS21_ID)
        if not b21:
            continue
        row = MappingRow(
            brats21_id=b21,
            data_collection=_get(r, HDR_DATA_COLLECTION) or "",
            site_id=_get(r, HDR_SITE_ID),
            portal_id=_get(r, HDR_PORTAL_ID),
            study_date=_get(r, HDR_STUDY_DATE),
            seg_cohort=_get(r, HDR_SEG_COHORT),
            mgmt_cohort=_get(r, HDR_MGMT_COHORT),
            mgmt_value=_get(r, HDR_MGMT_VALUE),
        )
        rows.append(row)
        by_b21[b21] = row
        by_coll.setdefault(row.data_collection, set()).add(b21)

    return Brats2021Mapping(
        rows=tuple(rows),
        by_brats21_id=by_b21,
        by_collection={k: frozenset(v) for k, v in by_coll.items()},
    )
