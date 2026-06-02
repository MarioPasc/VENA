"""Round-trip tests for the stdlib xlsx parser.

We fabricate a tiny xlsx fixture in memory (zipfile + sharedStrings + sheet)
so the parser is exercised without depending on the real
BraTS2021_MappingToTCIA.xlsx file.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from vena.preflight.cohort_dedup.xlsx import (
    HDR_BRATS21_ID,
    HDR_DATA_COLLECTION,
    HDR_MGMT_COHORT,
    HDR_MGMT_VALUE,
    HDR_PORTAL_ID,
    HDR_SEG_COHORT,
    HDR_SITE_ID,
    HDR_STUDY_DATE,
    parse_brats2021_mapping,
)

pytestmark = pytest.mark.unit


def _build_xlsx(rows: list[list[str]]) -> bytes:
    """Build a minimal valid xlsx with one sheet and the given rows."""
    shared: list[str] = []
    pos: dict[str, int] = {}

    def _intern(s: str) -> int:
        if s not in pos:
            pos[s] = len(shared)
            shared.append(s)
        return pos[s]

    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    sheet_rows: list[str] = []
    for r_idx, row in enumerate(rows, start=1):
        cells: list[str] = []
        for c_idx, val in enumerate(row):
            ref = f"{chr(ord('A') + c_idx)}{r_idx}"
            idx = _intern(val)
            cells.append(f'<c r="{ref}" t="s"><v>{idx}</v></c>')
        sheet_rows.append(f'<row r="{r_idx}">{"".join(cells)}</row>')
    sheet_xml = (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f"<worksheet {ns}><sheetData>"
        f"{''.join(sheet_rows)}"
        f"</sheetData></worksheet>"
    )
    ss_si = "".join(f"<si><t>{s}</t></si>" for s in shared)
    shared_xml = (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<sst {ns} count="{len(shared)}" uniqueCount="{len(shared)}">{ss_si}</sst>'
    )
    workbook_xml = (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<workbook {ns} xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets><sheet name="TCIA" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        "</Relationships>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>'
        "</Types>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("xl/workbook.xml", workbook_xml)
        z.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        z.writestr("xl/sharedStrings.xml", shared_xml)
        z.writestr("xl/worksheets/sheet1.xml", sheet_xml)
    return buf.getvalue()


def _write_xlsx(tmp_path: Path, rows: list[list[str]]) -> Path:
    p = tmp_path / "mapping.xlsx"
    p.write_bytes(_build_xlsx(rows))
    return p


HEADER = [
    HDR_DATA_COLLECTION,
    HDR_SITE_ID,
    HDR_PORTAL_ID,
    HDR_STUDY_DATE,
    HDR_BRATS21_ID,
    HDR_SEG_COHORT,
    HDR_MGMT_COHORT,
    HDR_MGMT_VALUE,
]


def test_parse_returns_indexed_rows(tmp_path: Path) -> None:
    p = _write_xlsx(
        tmp_path,
        [
            HEADER,
            ["UCSF-PDGM", "18", "57", "2020-01-01", "BraTS2021_00000", "Training", "Training", "1"],
            ["IvyGAP", "21", "169", "2018-05-20", "BraTS2021_00100", "Training", "Training", "0"],
            [
                "UCSF-PDGM_Additional",
                "18",
                "96",
                "2020-02-01",
                "BraTS2021_00001",
                "Validation",
                "Validation",
                "0",
            ],
        ],
    )
    mapping = parse_brats2021_mapping(p)
    assert len(mapping.rows) == 3
    assert mapping.by_brats21_id["BraTS2021_00000"].data_collection == "UCSF-PDGM"
    assert mapping.by_brats21_id["BraTS2021_00100"].data_collection == "IvyGAP"
    assert set(mapping.by_collection) == {"UCSF-PDGM", "UCSF-PDGM_Additional", "IvyGAP"}
    assert mapping.by_collection["UCSF-PDGM"] == frozenset({"BraTS2021_00000"})


def test_missing_required_column_raises(tmp_path: Path) -> None:
    bad_header = [HDR_DATA_COLLECTION, HDR_SITE_ID]  # missing BraTS2021 ID
    p = _write_xlsx(tmp_path, [bad_header, ["UCSF-PDGM", "18"]])
    with pytest.raises(ValueError, match="missing required column"):
        parse_brats2021_mapping(p)


def test_nonexistent_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse_brats2021_mapping(tmp_path / "does_not_exist.xlsx")
