"""Column detection for PDFs.

pdfplumber's `extract_text(layout=True)` preserves *spatial* position, which on
a two-column page means it splices the columns together character by character:

    the quarter and found that headcount growth accounted fothre t hper omdaujcotr

That is the reading-order failure the whole project is about, and pure-Python
extraction has no layout model to prevent it. So we build a minimal one.

The approach is geometric, not learned: find the vertical gutter (a band of
whitespace no word crosses), split the page into full-width bands and columnar
bands, and emit them in human reading order. This is roughly what a document
layout model does with a neural network, done with a histogram instead. It
handles clean digital two-column documents and degrades on anything else, which
is exactly the boundary a router should learn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Column:
    x0: float
    x1: float


def find_gutter(
    words: list[dict[str, Any]],
    page_width: float,
    min_gap_ratio: float = 0.025,
    search_band: tuple[float, float] = (0.30, 0.70),
) -> tuple[float, float] | None:
    """Find the widest vertical whitespace band near the page's middle.

    Returns (gutter_x0, gutter_x1), or None if the page is single-column.
    """
    if len(words) < 10:
        return None

    resolution = 200
    occupied = [False] * resolution
    for w in words:
        start = max(0, int(w["x0"] / page_width * resolution))
        end = min(resolution - 1, int(w["x1"] / page_width * resolution))
        for i in range(start, end + 1):
            occupied[i] = True

    lo = int(search_band[0] * resolution)
    hi = int(search_band[1] * resolution)

    best: tuple[int, int] | None = None
    run_start: int | None = None
    for i in range(lo, hi + 1):
        if not occupied[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                if best is None or (i - run_start) > (best[1] - best[0]):
                    best = (run_start, i)
                run_start = None
    if run_start is not None:
        if best is None or (hi + 1 - run_start) > (best[1] - best[0]):
            best = (run_start, hi + 1)

    if best is None:
        return None
    width_ratio = (best[1] - best[0]) / resolution
    if width_ratio < min_gap_ratio:
        return None

    return (best[0] / resolution * page_width, best[1] / resolution * page_width)


def order_words(
    words: list[dict[str, Any]],
    page_width: float,
    line_tol: float = 3.0,
) -> list[list[dict[str, Any]]]:
    """Return words grouped into lines, in reading order.

    Single-column pages come back top-to-bottom. Two-column pages come back as
    full-width bands and columnar bands interleaved in document order, with the
    left column read fully before the right column within each band.
    """
    if not words:
        return []

    gutter = find_gutter(words, page_width)
    if gutter is None:
        return _lines(words, line_tol)

    gx0, gx1 = gutter
    spanning = [w for w in words if w["x0"] < gx0 and w["x1"] > gx1]

    # Full-width bands are the y-ranges occupied by words that cross the gutter
    # (titles, section headers, wide tables). Everything between them is
    # columnar.
    bands: list[tuple[float, float]] = []
    for w in spanning:
        top, bottom = w["top"], w["bottom"]
        merged = False
        for i, (bt, bb) in enumerate(bands):
            if top <= bb + line_tol and bottom >= bt - line_tol:
                bands[i] = (min(bt, top), max(bb, bottom))
                merged = True
                break
        if not merged:
            bands.append((top, bottom))
    bands.sort()

    page_top = min(w["top"] for w in words)
    page_bottom = max(w["bottom"] for w in words)

    # Build the alternating sequence of regions down the page.
    regions: list[tuple[float, float, bool]] = []
    cursor = page_top
    for bt, bb in bands:
        if bt > cursor:
            regions.append((cursor, bt, False))
        regions.append((bt, bb, True))
        cursor = bb
    if cursor < page_bottom:
        regions.append((cursor, page_bottom, False))

    out: list[list[dict[str, Any]]] = []
    for top, bottom, full_width in regions:
        in_region = [
            w for w in words
            if w["top"] >= top - line_tol and w["bottom"] <= bottom + line_tol
        ]
        if not in_region:
            continue
        if full_width:
            out.extend(_lines(in_region, line_tol))
        else:
            left = [w for w in in_region if w["x1"] <= gx1]
            right = [w for w in in_region if w["x0"] >= gx0]
            out.extend(_lines(left, line_tol))
            out.extend(_lines(right, line_tol))
    return out


def _lines(
    words: list[dict[str, Any]], line_tol: float
) -> list[list[dict[str, Any]]]:
    """Group words into lines by vertical position, left to right within a line."""
    if not words:
        return []
    ordered = sorted(words, key=lambda w: (w["top"], w["x0"]))
    lines: list[list[dict[str, Any]]] = [[ordered[0]]]
    for w in ordered[1:]:
        if abs(w["top"] - lines[-1][0]["top"]) <= line_tol:
            lines[-1].append(w)
        else:
            lines.append([w])
    return [sorted(line, key=lambda w: w["x0"]) for line in lines]


def lines_to_paragraphs(
    lines: list[list[dict[str, Any]]], gap_factor: float = 1.6
) -> list[str]:
    """Join lines into paragraphs, breaking on unusually large vertical gaps."""
    if not lines:
        return []

    texts = [" ".join(w["text"] for w in line) for line in lines]
    tops = [line[0]["top"] for line in lines]
    heights = [line[0]["bottom"] - line[0]["top"] for line in lines]
    median_h = sorted(heights)[len(heights) // 2] if heights else 12.0

    paragraphs: list[str] = []
    current = [texts[0]]
    for i in range(1, len(texts)):
        gap = tops[i] - tops[i - 1]
        # A negative gap means we jumped back up the page: a new column started.
        if gap < 0 or gap > median_h * gap_factor:
            paragraphs.append(" ".join(current))
            current = [texts[i]]
        else:
            current.append(texts[i])
    paragraphs.append(" ".join(current))
    return [p for p in paragraphs if p.strip()]
