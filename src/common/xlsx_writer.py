"""Minimal Excel (.xlsx) writer on the Python standard library.

Houdini's Python has no spreadsheet package, so Read PVD writes the Office
Open XML parts itself: one worksheet per table, numbers as numbers (a NaN or
an infinity becomes an empty cell, which Excel plots as a gap), text as
inline strings, a bold frozen header row and column widths. No charts and no
formulas: the workbook holds data to plot. Rows may be any iterable (a
generator for large tables); they are streamed into the zip file.

    book = XlsxWorkbook()
    book.add_sheet("Data", [["Step", "Time (s)"], [0, 0.0], [1, 0.5]])
    book.save("results.xlsx", title="...", creator="Read PVD")
"""

import datetime
import math
import os
import re
import zipfile

XLSX_MAX_ROWS = 1048576
XLSX_MAX_COLUMNS = 16384
_XLSX_SHEET_FORBIDDEN = re.compile(r"[\[\]:*?/\\]")
# characters XML 1.0 cannot carry at all
_XLSX_XML_INVALID = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f￾￿]")


class XlsxLimitError(ValueError):
    """A table does not fit in an Excel worksheet."""


def xlsx_column_name(index):
    """0 -> A, 25 -> Z, 26 -> AA (Excel column letters)."""
    name = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def xlsx_sheet_name(title, taken=()):
    """A legal, unique (case-insensitively) worksheet name for `title`:
    at most 31 characters, none of [ ] : * ? / \\, not starting or ending
    with an apostrophe."""
    name = _XLSX_SHEET_FORBIDDEN.sub("_", str(title)).strip().strip("'")
    name = _XLSX_XML_INVALID.sub("", name)[:31].strip().strip("'") or "Sheet"
    lowered = {value.lower() for value in taken}
    candidate, number = name, 2
    while candidate.lower() in lowered:
        suffix = f" ({number})"
        candidate = name[:31 - len(suffix)].rstrip() + suffix
        number += 1
    return candidate


def _xlsx_escape(text):
    text = _XLSX_XML_INVALID.sub("", str(text))
    return (text.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _xlsx_cell(reference, value, style):
    """One <c> element, or "" for an empty cell."""
    if value is None:
        return ""
    attributes = f' r="{reference}"' + (f' s="{style}"' if style else "")
    if isinstance(value, str):
        text = _xlsx_escape(value)
        if not text:
            return ""
        space = ' xml:space="preserve"' if text != text.strip() else ""
        return (f'<c{attributes} t="inlineStr"><is><t{space}>{text}</t>'
                '</is></c>')
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _xlsx_cell(reference, str(value), style)
    if not math.isfinite(number):
        return ""
    if isinstance(value, (bool, int)) or (
            hasattr(value, "dtype") and value.dtype.kind in "iub"):
        return f"<c{attributes}><v>{int(value)}</v></c>"
    return f"<c{attributes}><v>{number!r}</v></c>"


class XlsxWorkbook:
    """Collect tables, then save() them as one .xlsx file."""

    def __init__(self):
        self._sheets = []   # (name, rows, widths, header_rows)

    def add_sheet(self, title, rows, widths=None, header_rows=1):
        """Add a worksheet; returns the (legal, unique) name it got.
        `widths` are column widths in characters (None = Excel's default);
        the first `header_rows` rows are bold and stay in view (frozen)."""
        name = xlsx_sheet_name(title, [sheet[0] for sheet in self._sheets])
        self._sheets.append((name, rows, list(widths or ()),
                             max(0, int(header_rows))))
        return name

    @property
    def sheet_names(self):
        return [sheet[0] for sheet in self._sheets]

    @staticmethod
    def _write_sheet(handle, rows, widths, header_rows):
        def put(text):
            handle.write(text.encode("utf-8"))

        put('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<worksheet xmlns="http://schemas.openxmlformats.org/'
            'spreadsheetml/2006/main" xmlns:r="http://schemas.'
            'openxmlformats.org/officeDocument/2006/relationships">')
        if header_rows:
            cell = f"A{header_rows + 1}"
            put('<sheetViews><sheetView workbookViewId="0">'
                f'<pane ySplit="{header_rows}" topLeftCell="{cell}" '
                'activePane="bottomLeft" state="frozen"/>'
                f'<selection pane="bottomLeft" activeCell="{cell}" '
                f'sqref="{cell}"/></sheetView></sheetViews>')
        put('<sheetFormatPr defaultRowHeight="15"/>')
        if widths:
            put("<cols>" + "".join(
                f'<col min="{index + 1}" max="{index + 1}" '
                f'width="{float(width):.2f}" customWidth="1"/>'
                for index, width in enumerate(widths) if width) + "</cols>")
        put("<sheetData>")
        count = 0
        buffer = []
        for count, row in enumerate(rows, 1):
            if count > XLSX_MAX_ROWS:
                raise XlsxLimitError(
                    f"more than {XLSX_MAX_ROWS:,} rows, Excel's limit")
            row = list(row)
            if len(row) > XLSX_MAX_COLUMNS:
                raise XlsxLimitError(
                    f"{len(row):,} columns, more than Excel's limit of "
                    f"{XLSX_MAX_COLUMNS:,}")
            style = 1 if count <= header_rows else 0
            cells = "".join(
                _xlsx_cell(f"{xlsx_column_name(column)}{count}", value, style)
                for column, value in enumerate(row))
            buffer.append(f'<row r="{count}">{cells}</row>')
            if len(buffer) >= 2000:
                put("".join(buffer))
                buffer = []
        put("".join(buffer))
        put("</sheetData></worksheet>")
        return count

    def save(self, path, title="", creator=""):
        """Write the workbook to `path` (replacing it); returns the number
        of rows written per sheet."""
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
        sheet_types = "".join(
            f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.'
            'spreadsheetml.worksheet+xml"/>'
            for index in range(1, len(self._sheets) + 1))
        sheets = "".join(
            f'<sheet name="{_xlsx_escape(name)}" sheetId="{index}" '
            f'r:id="rId{index}"/>'
            for index, (name, *_) in enumerate(self._sheets, 1))
        relations = "".join(
            f'<Relationship Id="rId{index}" Type="http://schemas.'
            'openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{index}.xml"/>'
            for index in range(1, len(self._sheets) + 1))
        relations += (
            f'<Relationship Id="rId{len(self._sheets) + 1}" Type="http://'
            'schemas.openxmlformats.org/officeDocument/2006/relationships/'
            'styles" Target="styles.xml"/>')
        counts = []
        temporary = path + ".tmp"
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "[Content_Types].xml",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                '<Types xmlns="http://schemas.openxmlformats.org/package/2006/'
                'content-types"><Default Extension="rels" ContentType="'
                'application/vnd.openxmlformats-package.relationships+xml"/>'
                '<Default Extension="xml" ContentType="application/xml"/>'
                '<Override PartName="/xl/workbook.xml" ContentType="'
                'application/vnd.openxmlformats-officedocument.spreadsheetml.'
                'sheet.main+xml"/><Override PartName="/xl/styles.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.'
                'spreadsheetml.styles+xml"/><Override PartName="/docProps/'
                'core.xml" ContentType="application/vnd.openxmlformats-'
                'package.core-properties+xml"/><Override PartName="/docProps/'
                'app.xml" ContentType="application/vnd.openxmlformats-'
                'officedocument.extended-properties+xml"/>'
                + sheet_types + "</Types>")
            archive.writestr(
                "_rels/.rels",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                '<Relationships xmlns="http://schemas.openxmlformats.org/'
                'package/2006/relationships"><Relationship Id="rId1" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/'
                'relationships/officeDocument" Target="xl/workbook.xml"/>'
                '<Relationship Id="rId2" Type="http://schemas.openxmlformats.'
                'org/package/2006/relationships/metadata/core-properties" '
                'Target="docProps/core.xml"/><Relationship Id="rId3" Type="'
                'http://schemas.openxmlformats.org/officeDocument/2006/'
                'relationships/extended-properties" Target="docProps/app.xml"'
                '/></Relationships>')
            archive.writestr(
                "docProps/core.xml",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.'
                'org/package/2006/metadata/core-properties" xmlns:dc="http://'
                'purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/'
                'dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-'
                f'instance"><dc:title>{_xlsx_escape(title)}</dc:title>'
                f'<dc:creator>{_xlsx_escape(creator)}</dc:creator>'
                f'<dcterms:created xsi:type="dcterms:W3CDTF">{stamp}'
                '</dcterms:created><dcterms:modified xsi:type="dcterms:'
                f'W3CDTF">{stamp}</dcterms:modified></cp:coreProperties>')
            archive.writestr(
                "docProps/app.xml",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                '<Properties xmlns="http://schemas.openxmlformats.org/'
                'officeDocument/2006/extended-properties"><Application>'
                f'{_xlsx_escape(creator or "Read PVD")}</Application>'
                '</Properties>')
            archive.writestr(
                "xl/workbook.xml",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                '<workbook xmlns="http://schemas.openxmlformats.org/'
                'spreadsheetml/2006/main" xmlns:r="http://schemas.'
                'openxmlformats.org/officeDocument/2006/relationships">'
                f'<bookViews><workbookView/></bookViews><sheets>{sheets}'
                '</sheets></workbook>')
            archive.writestr(
                "xl/_rels/workbook.xml.rels",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                '<Relationships xmlns="http://schemas.openxmlformats.org/'
                f'package/2006/relationships">{relations}</Relationships>')
            archive.writestr(
                "xl/styles.xml",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                '<styleSheet xmlns="http://schemas.openxmlformats.org/'
                'spreadsheetml/2006/main"><fonts count="2"><font><sz val="11"'
                '/><name val="Calibri"/><family val="2"/></font><font><b/><sz '
                'val="11"/><name val="Calibri"/><family val="2"/></font>'
                '</fonts><fills count="2"><fill><patternFill patternType="'
                'none"/></fill><fill><patternFill patternType="gray125"/>'
                '</fill></fills><borders count="1"><border><left/><right/>'
                '<top/><bottom/><diagonal/></border></borders><cellStyleXfs '
                'count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId='
                '"0"/></cellStyleXfs><cellXfs count="2"><xf numFmtId="0" '
                'fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId='
                '"0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont='
                '"1"/></cellXfs><cellStyles count="1"><cellStyle name="Normal"'
                ' xfId="0" builtinId="0"/></cellStyles></styleSheet>')
            for index, (name, rows, widths, header_rows) in enumerate(
                    self._sheets, 1):
                with archive.open(f"xl/worksheets/sheet{index}.xml", "w",
                                  force_zip64=True) as handle:
                    counts.append(self._write_sheet(handle, rows, widths,
                                                    header_rows))
        os.replace(temporary, path)
        return dict(zip(self.sheet_names, counts))
