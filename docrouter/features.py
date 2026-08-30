"""Cheap document features, extracted before any expensive parse.

The router's entire value depends on this file being fast. A feature that
requires running a backend is not a routing feature: by the time you have it,
you have already paid the cost the router exists to avoid.

Everything here is computed from a sample of pages using the PDF's own
structures -- character counts, ruling lines, image coverage, word geometry --
and targets well under a second on a 500-page filing. Compare that to Docling's
~180s and the asymmetry that makes routing worthwhile is obvious.

The strongest signal by far is `has_text_layer`. In the benchmark it separates
the clean and degraded conditions perfectly, and the cheap backends score
0.000 on every document where it is False.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Pages sampled for the per-page statistics. Fixed rather than proportional so
# feature cost stays flat as documents get longer.
SAMPLE_PAGES = 8

# Below this many characters per sampled page, treat the document as having no
# usable text layer. Not zero: scanned PDFs often carry a few stray glyphs from
# headers, stamps, or a partial OCR pass.
TEXT_LAYER_MIN_CHARS = 50


@dataclass
class DocumentFeatures:
    doc_key: str
    page_count: int
    file_size_kb: float

    has_text_layer: bool
    chars_per_page: float

    # Table proxies. Ruling lines are what line-based table detection keys on;
    # documents with none force a text-alignment strategy or a model.
    lines_per_page: float
    rects_per_page: float

    # Layout proxies.
    est_columns: int
    words_per_page: float

    # Scan proxy: rasterized pages are one full-page image and nothing else.
    image_area_ratio: float

    # How long the features themselves took. This is the router's overhead and
    # belongs in any honest cost accounting.
    feature_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def is_scanned(self) -> bool:
        """Best cheap guess at whether this document needs OCR."""
        return not self.has_text_layer or self.image_area_ratio > 0.8


def extract_features(path: str | Path, doc_key: str | None = None) -> DocumentFeatures:
    """Compute routing features for one document."""
    import pdfplumber

    from .layout import find_gutter

    path = Path(path)
    start = time.perf_counter()

    with pdfplumber.open(path) as pdf:
        page_count = len(pdf.pages)
        # Sample from across the document, not just the front: cover pages are
        # unrepresentative, and a filing's tables cluster in the middle.
        if page_count <= SAMPLE_PAGES:
            indices = list(range(page_count))
        else:
            step = page_count / SAMPLE_PAGES
            indices = [min(int(i * step), page_count - 1) for i in range(SAMPLE_PAGES)]

        chars = lines = rects = words_total = 0
        image_area = page_area = 0.0
        column_votes: list[int] = []

        for i in indices:
            page = pdf.pages[i]
            chars += len(page.chars)
            lines += len(page.lines)
            rects += len(page.rects)

            page_area += page.width * page.height
            for img in page.images:
                w = abs(img.get("x1", 0) - img.get("x0", 0))
                h = abs(img.get("bottom", 0) - img.get("top", 0))
                image_area += w * h

            words = page.extract_words() if page.chars else []
            words_total += len(words)
            if len(words) >= 20:
                column_votes.append(2 if find_gutter(words, page.width) else 1)

    n = max(len(indices), 1)
    chars_per_page = chars / n

    features = DocumentFeatures(
        doc_key=doc_key or path.stem,
        page_count=page_count,
        file_size_kb=path.stat().st_size / 1024,
        has_text_layer=chars_per_page >= TEXT_LAYER_MIN_CHARS,
        chars_per_page=chars_per_page,
        lines_per_page=lines / n,
        rects_per_page=rects / n,
        est_columns=(
            max(set(column_votes), key=column_votes.count) if column_votes else 1
        ),
        words_per_page=words_total / n,
        image_area_ratio=(image_area / page_area) if page_area else 0.0,
        feature_time_ms=(time.perf_counter() - start) * 1000,
    )
    return features


def extract_corpus_features(corpus_root: str | Path) -> list[DocumentFeatures]:
    """Compute features for every document in a corpus directory."""
    docs = sorted((Path(corpus_root) / "docs").iterdir())
    out = []
    for doc in docs:
        if doc.name.startswith(".") or doc.suffix.lower() != ".pdf":
            continue
        out.append(extract_features(doc))
    return out
