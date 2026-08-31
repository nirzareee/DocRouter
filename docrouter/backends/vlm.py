"""Vision-language model backend: the expensive tier.

Renders each page to an image and asks a multimodal model to transcribe it as
markdown. Unlike every other backend here, this one has a real per-document
price, which is the point: it puts an actual dollar figure on the Pareto curve
instead of the wall-clock proxies the local backends report.

It also reads pages that have no text layer at all, so it competes with Docling
on the degraded condition rather than scoring zero like the pure-Python tiers.

Cost is computed from reported token usage rather than estimated from page
count, because image token cost depends on resolution and a rendering-DPI
change would silently invalidate a hardcoded per-page figure.

Setup:

    export ANTHROPIC_API_KEY=sk-...
    pip install anthropic

Pricing must be set for your model before any cost figure means anything:

    export VLM_INPUT_USD_PER_MTOK=3.00
    export VLM_OUTPUT_USD_PER_MTOK=15.00

Check current model names and prices at docs.claude.com -- both change, and a
benchmark quoting stale prices is worse than one quoting none.
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any

from .base import Backend

DEFAULT_MODEL = os.environ.get("VLM_MODEL", "claude-sonnet-4-5")

# Resolution of rendered page images. Higher DPI reads small table text more
# reliably and costs proportionally more tokens. 150 is a reasonable middle;
# treat it as a tunable and re-measure cost if you change it.
RENDER_DPI = int(os.environ.get("VLM_DPI", "150"))

# Hard cap so a 562-page filing cannot silently spend a fortune.
MAX_PAGES = int(os.environ.get("VLM_MAX_PAGES", "40"))

PROMPT = """Transcribe this document page into GitHub-flavored Markdown.

Rules:
- Reproduce tables as pipe tables. Preserve every row and column.
- Keep numbers exactly as printed, including parenthesized negatives like (617).
- Use # heading levels matching the visual hierarchy.
- Preserve reading order. On multi-column pages, finish the left column first.
- Do not summarize, explain, or add commentary. Output only the transcription.
- If the page is blank or unreadable, output nothing."""


class VLMBackend(Backend):
    name = "vlm"
    version = "1"

    def __init__(self, model: str | None = None) -> None:
        self.model = model or DEFAULT_MODEL
        self.input_price = float(os.environ.get("VLM_INPUT_USD_PER_MTOK", "0") or 0)
        self.output_price = float(os.environ.get("VLM_OUTPUT_USD_PER_MTOK", "0") or 0)

    def supports(self, path: Path) -> bool:
        # Works on any PDF, including ones with no text layer at all -- which is
        # the whole reason it is here.
        return path.suffix.lower() == ".pdf"

    def is_available(self) -> bool:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return False
        try:
            import anthropic  # noqa: F401
            import pypdfium2  # noqa: F401
        except ImportError:
            return False
        return True

    def _render_pages(self, path: Path) -> tuple[list[str], int]:
        """Render pages to base64 PNGs. Returns (images, total_page_count)."""
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(path))
        total = len(pdf)
        images: list[str] = []
        for i in range(min(total, MAX_PAGES)):
            bitmap = pdf[i].render(scale=RENDER_DPI / 72.0)
            buf = io.BytesIO()
            bitmap.to_pil().convert("RGB").save(buf, format="PNG")
            images.append(base64.b64encode(buf.getvalue()).decode())
        pdf.close()
        return images, total

    def _transcribe(self, client: Any, image_b64: str) -> tuple[str, int, int]:
        """One page. Returns (markdown, input_tokens, output_tokens)."""
        response = client.messages.create(
            model=self.model,
            max_tokens=8192,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": PROMPT},
                ],
            }],
        )
        text = "".join(
            block.text for block in response.content
            if getattr(block, "type", None) == "text"
        )
        usage = response.usage
        return text, usage.input_tokens, usage.output_tokens

    def _parse(self, path: Path) -> tuple[str, int, dict[str, Any]]:
        import anthropic

        client = anthropic.Anthropic()
        images, total_pages = self._render_pages(path)

        pages_md: list[str] = []
        in_tok = out_tok = 0
        for image in images:
            text, i_tok, o_tok = self._transcribe(client, image)
            in_tok += i_tok
            out_tok += o_tok
            if text.strip():
                pages_md.append(text.strip())

        cost = (
            in_tok / 1_000_000 * self.input_price
            + out_tok / 1_000_000 * self.output_price
        )

        meta = {
            "model": self.model,
            "dpi": RENDER_DPI,
            "pages_sent": len(images),
            "pages_total": total_pages,
            "truncated": total_pages > len(images),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "measured_cost_usd": cost,
        }
        if self.input_price == 0:
            # Silently reporting $0.00 would put a false point on the Pareto
            # curve, which is worse than reporting nothing.
            meta["warning"] = "pricing not configured; cost_usd is meaningless"

        return "\n\n".join(pages_md), len(images), meta

    def parse(self, path: Path):
        """Override to report measured cost instead of a per-page estimate."""
        result = super().parse(path)
        if result.ok:
            result.cost_usd = result.meta.get("measured_cost_usd", 0.0)
        return result
