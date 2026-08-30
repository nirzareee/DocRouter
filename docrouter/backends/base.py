"""Common contract every extraction backend implements.

The router's whole job is to pick between implementations of `Backend`, so the
interface has to expose the three things a routing decision needs: what the
backend *can* read, what it *costs*, and what it *produced*.
"""

from __future__ import annotations

import abc
import hashlib
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class ParseResult:
    """The output of one backend on one document."""

    markdown: str
    backend: str
    doc_id: str

    # Routing signals. Every backend must populate these honestly; the cost
    # model is only as good as its worst-reporting backend.
    wall_time_s: float = 0.0
    cost_usd: float = 0.0
    pages: int = 0

    # Populated instead of markdown when the backend fails. A failed parse is a
    # legitimate outcome, not an exception to swallow: the router needs to learn
    # which backends fail on which documents.
    error: str | None = None

    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Backend(abc.ABC):
    """One way of turning a document into markdown."""

    name: str = "unnamed"

    # Bump when a backend's output could change, so cached parses from the old
    # behavior are invalidated without clearing every other backend's entries.
    version: str = "1"

    # Rough per-page cost in USD. Local backends are not free (GPU time, wall
    # clock) but they are cheap enough that the router mostly trades latency.
    cost_per_page_usd: float = 0.0

    @abc.abstractmethod
    def supports(self, path: Path) -> bool:
        """Whether this backend can attempt the file at all."""

    @abc.abstractmethod
    def _parse(self, path: Path) -> tuple[str, int, dict[str, Any]]:
        """Return (markdown, page_count, meta). May raise."""

    def is_available(self) -> bool:
        """Whether the backend's dependencies are importable on this machine.

        Kept separate from `supports` so a missing GPU or unset API key skips
        the backend cleanly instead of scoring it as a failure.
        """
        return True

    def parse(self, path: Path) -> ParseResult:
        """Run the backend, timing it and converting exceptions into results."""
        path = Path(path)
        doc_id = doc_id_for(path)
        start = time.perf_counter()
        try:
            markdown, pages, meta = self._parse(path)
        except Exception as exc:  # noqa: BLE001 - failures are data here
            return ParseResult(
                markdown="",
                backend=self.name,
                doc_id=doc_id,
                wall_time_s=time.perf_counter() - start,
                error=f"{type(exc).__name__}: {exc}",
            )
        elapsed = time.perf_counter() - start
        return ParseResult(
            markdown=markdown,
            backend=self.name,
            doc_id=doc_id,
            wall_time_s=elapsed,
            cost_usd=self.cost_per_page_usd * max(pages, 1),
            pages=pages,
            meta=meta,
        )


def doc_id_for(path: Path) -> str:
    """Content hash, so the same bytes get the same id across machines.

    This doubles as the cache key in the pipeline: reparsing an identical
    document with an identical backend is always wasted money.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]
