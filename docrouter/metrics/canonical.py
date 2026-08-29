"""Table canonicalization, applied identically to predictions and ground truth.

A benchmark must not reward a backend for happening to share a formatting
convention with the annotator. Filings put the currency symbol in its own
column; whether that column survives into markdown is a presentation choice.
The gold generator merges it. pdfplumber does not. Scored raw, a backend that
extracted every number correctly lost 0.4 TEDS purely on that difference.

So both sides are normalized before comparison. This is standard practice for
document-conversion benchmarks: the metric should measure what the backend
recovered, not which house style it happens to match.

What is normalized (structure and presentation):
  - columns empty in every row are dropped
  - columns holding only a currency or percent symbol are merged right

What is NOT normalized (content, which is what is being measured):
  - cell text, numbers, negatives, spelling, ordering
"""

from __future__ import annotations

SYMBOL_ONLY = {"$", "%", "\u20ac", "\u00a3", "\u00a5"}


def drop_empty_columns(rows: list[list[str]]) -> list[list[str]]:
    """Remove columns that are empty in every row."""
    if not rows:
        return rows
    width = max(len(r) for r in rows)
    padded = [r + [""] * (width - len(r)) for r in rows]
    keep = [
        i for i in range(width)
        if any(padded[r][i].strip() for r in range(len(padded)))
    ]
    if not keep:
        return []
    return [[row[i] for i in keep] for row in padded]


def merge_symbol_columns(rows: list[list[str]]) -> list[list[str]]:
    """Fold currency/percent-only columns into the value to their right.

    Detection compares adjacent column pairs rather than column totals.
    Filings print '$' only on the first row of a section and on totals, so a
    genuine currency column may be a minority of symbols; and colspan header
    origins land in the same column, adding date labels to it. What holds
    reliably: in every row where both columns are populated, the left one is
    only ever a symbol.
    """
    if not rows:
        return rows
    width = max(len(r) for r in rows)
    padded = [r + [""] * (width - len(r)) for r in rows]

    symbol_cols: list[int] = []
    for c in range(width - 1):
        paired = [
            padded[r][c].strip()
            for r in range(len(padded))
            if padded[r][c].strip() and padded[r][c + 1].strip()
        ]
        if paired and all(v in SYMBOL_ONLY for v in paired):
            symbol_cols.append(c)

    if not symbol_cols:
        return padded

    out = []
    for row in padded:
        new_row: list[str] = []
        carry = ""
        for c, cell in enumerate(row):
            if c in symbol_cols:
                # Carries the symbol on data rows and a colspan header on
                # header rows, which re-aligns the header over its value.
                carry = cell.strip()
                continue
            new_row.append(f"{carry}{cell}".strip() if carry else cell)
            carry = ""
        if carry:
            new_row.append(carry)
        out.append(new_row)
    return out


def canonicalize(rows: list[list[str]]) -> list[list[str]]:
    """Normalize a table for comparison.

    Order matters: colspan continuation columns sit between a currency symbol
    and its number, so merging before dropping puts the '$' in the padding
    column instead of on the value.
    """
    rows = drop_empty_columns(rows)
    rows = merge_symbol_columns(rows)
    rows = drop_empty_columns(rows)
    return rows
