"""Convert EDGAR filing HTML into ground-truth markdown.

This is the shortcut that makes a 50-document corpus feasible. EDGAR filings
are HTML with real <table> markup, so the structure you are trying to recover
from the PDF is already explicit in the source. Convert the HTML once, and you
have gold for both the clean and degraded renderings of that filing.

Two warnings that matter more than the code:

1. **This produces a draft, not ground truth.** Filings use tables for layout
   as well as for data, contain formatting artifacts, and represent negatives as
   parenthesized numbers. Every generated file needs a human pass. The value is
   that you are correcting rather than transcribing.

2. **Generated gold can flatter parsers that share assumptions with it.** If
   your converter drops the same empty spacer columns pdfplumber drops, `pylib`
   scores artificially well. Spot-check against the rendered PDF, not just
   against the HTML.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from lxml import html as lxml_html

# Filings are full of layout tables: single-column wrappers, spacer rows, and
# tables holding one paragraph. None of those are data.
MIN_DATA_ROWS = 2
MIN_DATA_COLS = 2


@dataclass
class GoldDocument:
    markdown: str
    n_tables: int
    n_headings: int
    n_paragraphs: int


def clean_cell(text: str) -> str:
    """Normalize one cell's text.

    Filings pad cells with non-breaking spaces and split currency symbols into
    their own columns. Both are presentation, not content.
    """
    text = text.replace("\xa0", " ")
    text = " ".join(text.split())
    return text.replace("|", r"\|")


# Cover pages and filer-status blocks are built from checkbox glyphs.
CHECKBOX_CHARS = set("\u2610\u2611\u2612\u25a1\u2713\u2714\u00d7")

# Table-of-contents rows are dominated by "Item N." / "Part N" labels.
_TOC_ROW = re.compile(r"^\s*(item|part)\s+[0-9ivx]+", re.I)


def _is_layout_table(rows: list[list[str]]) -> bool:
    """Heuristic: does this table carry data, or is it page furniture?

    No heuristic gets this fully right. Filings use tables for the cover page,
    the filer-status checkboxes, the table of contents, and page layout, and
    some of those are structurally indistinguishable from small data tables.
    The rules below catch the common cases; the rest is what the human review
    pass in docs/annotation-guide.md exists for.
    """
    if len(rows) < MIN_DATA_ROWS:
        return True
    width = max((len(r) for r in rows), default=0)
    if width < MIN_DATA_COLS:
        return True

    cells = [c for row in rows for c in row]
    if not cells:
        return True
    filled = [c for c in cells if c.strip()]
    if not filled or len(filled) / len(cells) < 0.25:
        return True

    # Checkbox glyphs mean a form control block, not data.
    if any(ch in CHECKBOX_CHARS for c in filled for ch in c):
        return True

    # Table of contents: most rows begin with an Item/Part label.
    labelled = sum(1 for row in rows if row and _TOC_ROW.match(row[0]))
    if labelled >= max(3, len(rows) * 0.5):
        return True

    # Financial tables contain numbers. A table with almost no numeric cells is
    # usually an address block, a signature block, or a list rendered as a
    # table. This is the weakest rule here, so keep the threshold low.
    numeric = sum(1 for c in filled if any(ch.isdigit() for ch in c))
    return numeric / len(filled) < 0.10


# Currency and percent symbols that filings place in their own column.
_SYMBOL_ONLY = {"$", "%", "\u20ac", "\u00a3", "\u00a5", "(", ")"}


def _merge_symbol_columns(rows: list[list[str]]) -> list[list[str]]:
    """Fold standalone '$' / '%' columns into the adjacent value.

    Filings put the currency symbol in its own table cell for alignment. Left
    alone this doubles the column count and, worse, shifts headers out of
    register: a `colspan` header lands above the '$' column instead of above
    the number it labels.

    This is an annotation decision, not a fact. It is recorded in
    docs/annotation-guide.md because your parsers may or may not make the same
    call, and a benchmark where gold and prediction disagree about what counts
    as a column is measuring the disagreement rather than the backend.
    """
    if not rows:
        return rows
    width = max(len(r) for r in rows)
    padded = [r + [""] * (width - len(r)) for r in rows]

    # Classify by looking at adjacent column pairs, not at column totals.
    #
    # The obvious rule -- "most values in this column are '$'" -- fails on real
    # filings. Companies print the currency symbol only on the first line of a
    # section and on totals, not on every row, so a genuine '$' column may be
    # only a third symbols. Worse, colspan header origins land in that same
    # column, so it also contains date labels.
    #
    # What is reliable: in every row where the symbol column and the value
    # column are *both* populated, the left one is always just a symbol.
    symbol_cols: list[int] = []
    for c in range(width - 1):
        paired = [
            padded[r][c].strip()
            for r in range(len(padded))
            if padded[r][c].strip() and padded[r][c + 1].strip()
        ]
        if paired and all(v in _SYMBOL_ONLY for v in paired):
            symbol_cols.append(c)

    if not symbol_cols:
        return padded

    out = []
    for row in padded:
        new_row: list[str] = []
        carry = ""
        for c, cell in enumerate(row):
            if c in symbol_cols:
                # Carries the symbol on data rows and the colspan header on
                # header rows, which is what re-aligns the header over its
                # value column.
                carry = cell.strip()
                continue
            merged = f"{carry}{cell}".strip() if carry else cell
            new_row.append(merged)
            carry = ""
        if carry:
            new_row.append(carry)
        out.append(new_row)
    return out


def _drop_empty_columns(rows: list[list[str]]) -> list[list[str]]:
    """Remove columns that are empty in every row.

    Filings interleave empty columns for alignment and put '$' in its own
    column. Keeping them makes the markdown unreadable, but dropping them is a
    judgment call that must be recorded in the annotation guide, because your
    parsers may or may not make the same call.
    """
    if not rows:
        return rows
    width = max(len(r) for r in rows)
    padded = [r + [""] * (width - len(r)) for r in rows]
    keep = [
        i for i in range(width)
        if any(padded[r][i].strip() for r in range(len(padded)))
    ]
    return [[row[i] for i in keep] for row in padded]


def table_to_rows(table_el) -> list[list[str]]:
    """Flatten an HTML table to a row matrix, expanding colspan/rowspan."""
    grid: dict[tuple[int, int], str] = {}
    occupied: set[tuple[int, int]] = set()

    for r, tr in enumerate(table_el.iter("tr")):
        col = 0
        for cell in tr:
            if str(cell.tag) not in {"td", "th"}:
                continue
            while (r, col) in occupied:
                col += 1
            text = clean_cell(" ".join(cell.itertext()))
            try:
                colspan = max(1, int(cell.get("colspan", 1)))
                rowspan = max(1, int(cell.get("rowspan", 1)))
            except ValueError:
                colspan = rowspan = 1
            for dr in range(rowspan):
                for dc in range(colspan):
                    occupied.add((r + dr, col + dc))
                    # Place the text once, at the span's origin, and leave the
                    # continuation cells empty. Repeating it across the span
                    # looks harmless but is not: filings use colspan for
                    # alignment, so `colspan=3` on every cell triples the
                    # table width and fills it with duplicated text. Leaving
                    # continuations empty lets _drop_empty_columns collapse
                    # pure-layout padding while preserving genuine spans whose
                    # other columns carry data.
                    grid[(r + dr, col + dc)] = text if (dr, dc) == (0, 0) else ""
            col += colspan

    if not grid:
        return []
    n_rows = max(r for r, _ in grid) + 1
    n_cols = max(c for _, c in grid) + 1
    return [[grid.get((r, c), "") for c in range(n_cols)] for r in range(n_rows)]


def rows_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    norm = [r + [""] * (width - len(r)) for r in rows]
    header, *body = norm
    out = ["| " + " | ".join(header) + " |"]
    out.append("| " + " | ".join(["---"] * width) + " |")
    out.extend("| " + " | ".join(r) + " |" for r in body)
    return "\n".join(out)


def _looks_like_heading(text: str) -> bool:
    if not (3 < len(text) < 120):
        return False
    if re.match(r"^(item|part)\s+[0-9ivx]+[.:]?", text, re.I):
        return True
    words = text.split()
    if len(words) > 12:
        return False
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(c.isupper() for c in letters) / len(letters) > 0.8


def html_to_gold(html_text: str | bytes) -> GoldDocument:
    """Convert filing HTML into markdown ground truth.

    Accepts bytes or str. Modern EDGAR filings are inline XBRL: XHTML carrying
    an `<?xml ... encoding="..."?>` declaration. lxml refuses to parse those
    from a str, because the declaration's encoding would contradict Python's
    already-decoded string. Passing bytes lets lxml honor the declaration.
    """
    if isinstance(html_text, str):
        # Strip a leading XML declaration rather than re-encoding blindly, so
        # a str caller still works.
        html_text = re.sub(r"^\s*<\?xml[^>]*\?>", "", html_text, count=1)
        html_text = html_text.encode("utf-8")

    tree = lxml_html.fromstring(html_text)

    # iXBRL filings wrap machine-readable facts in ix: elements. The header
    # block holds hidden tagged values that are not part of the visible
    # document, so including them would create ground truth for text that never
    # appears in the rendered PDF.
    for xpath, ns in (
        ("//script | //style", None),
        ("//ix:header", {"ix": "http://www.xbrl.org/2013/inlineXBRL"}),
    ):
        try:
            found = tree.xpath(xpath, namespaces=ns) if ns else tree.xpath(xpath)
        except Exception:  # noqa: BLE001 - namespace absent in non-XBRL filings
            continue
        for bad in found:
            parent = bad.getparent()
            if parent is not None:
                parent.remove(bad)

    body = tree.find(".//body")
    root = body if body is not None else tree

    blocks: list[str] = []
    n_tables = n_headings = n_paragraphs = 0
    seen_tables: set[int] = set()

    for el in root.iter():
        tag = str(el.tag).lower()

        if tag == "table":
            # Nested tables are common; only take the outermost.
            if any(id(a) in seen_tables for a in el.iterancestors()):
                continue
            seen_tables.add(id(el))
            # Order matters. Colspan continuation columns sit between the
            # currency symbol and its number, so merging first dumps the '$'
            # into the empty padding column. Drop padding, then merge, then
            # drop again since merging can empty a column.
            rows = table_to_rows(el)
            rows = _drop_empty_columns(rows)
            rows = _merge_symbol_columns(rows)
            rows = _drop_empty_columns(rows)
            if _is_layout_table(rows):
                continue
            md = rows_to_markdown(rows)
            if md:
                blocks.append(md)
                n_tables += 1

        elif tag in {"h1", "h2", "h3", "h4"}:
            text = clean_cell(" ".join(el.itertext()))
            if text:
                level = int(tag[1])
                blocks.append(f"{'#' * level} {text}")
                n_headings += 1

        elif tag in {"p", "div"}:
            if el.find(".//table") is not None:
                continue  # handled when we reach the table itself
            text = clean_cell("".join(el.itertext()))
            if len(text) < 20:
                continue
            if blocks and blocks[-1].endswith(text):
                continue  # nested div emitting the same text twice
            if _looks_like_heading(text):
                blocks.append(f"## {text}")
                n_headings += 1
            else:
                blocks.append(text)
                n_paragraphs += 1

    return GoldDocument(
        markdown="\n\n".join(blocks) + "\n",
        n_tables=n_tables,
        n_headings=n_headings,
        n_paragraphs=n_paragraphs,
    )


def convert_file(html_path: Path, out_path: Path) -> GoldDocument:
    # Read bytes, not text: filings declare their own encoding and lxml must
    # be allowed to honor it.
    gold = html_to_gold(html_path.read_bytes())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(gold.markdown, encoding="utf-8")
    return gold
