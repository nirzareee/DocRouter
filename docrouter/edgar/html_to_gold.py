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


def _is_layout_table(rows: list[list[str]]) -> bool:
    """Heuristic: does this table carry data, or is it page furniture?"""
    if len(rows) < MIN_DATA_ROWS:
        return True
    width = max((len(r) for r in rows), default=0)
    if width < MIN_DATA_COLS:
        return True
    # A table where almost every cell is empty is a spacing device.
    cells = [c for row in rows for c in row]
    if not cells:
        return True
    filled = sum(1 for c in cells if c.strip())
    return filled / len(cells) < 0.25


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
                    # Spanned cells repeat their value: a flat markdown table
                    # cannot express the span, so repetition is the honest
                    # lossy choice. Document it in the annotation guide.
                    grid[(r + dr, col + dc)] = text if (dr, dc) == (0, 0) else text
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


def html_to_gold(html_text: str) -> GoldDocument:
    """Convert filing HTML into markdown ground truth."""
    tree = lxml_html.fromstring(html_text)

    for bad in tree.xpath("//script | //style | //ix:header", namespaces={
        "ix": "http://www.xbrl.org/2013/inlineXBRL"
    }):
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
            rows = _drop_empty_columns(table_to_rows(el))
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
    gold = html_to_gold(html_path.read_text(encoding="utf-8", errors="replace"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(gold.markdown, encoding="utf-8")
    return gold
