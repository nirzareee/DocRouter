"""Docling backend: the CPU pipeline tier.

Docling runs a multi-stage pipeline (layout detection, table structure
recognition, reading-order logic) and emits markdown directly. It is the
natural middle tier: meaningfully better than pure-Python parsing on messy
PDFs, no GPU required, permissively licensed.

Install with: pip install docling
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import Backend

SUPPORTED = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".md"}


class DoclingBackend(Backend):
    name = "docling"

    # Not literally zero: this is CPU time on hardware you pay for. Set this
    # from your own measured throughput and machine cost before you publish a
    # cost-vs-accuracy curve, or the curve is fiction.
    cost_per_page_usd = 0.0

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in SUPPORTED

    def is_available(self) -> bool:
        try:
            import docling  # noqa: F401
        except ImportError:
            return False
        return True

    def _parse(self, path: Path) -> tuple[str, int, dict[str, Any]]:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(str(path))
        markdown = result.document.export_to_markdown()
        pages = getattr(result.document, "num_pages", None)
        if callable(pages):
            pages = pages()
        return markdown, int(pages or 1), {"status": str(getattr(result, "status", ""))}
