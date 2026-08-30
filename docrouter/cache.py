"""Content-addressed cache for parse results.

Docling takes ~180s on a clean 40-page filing and ~320s degraded. At 30
documents across two conditions that is roughly four hours per sweep. During
development this project re-ran the benchmark after every metric change, and
none of those changes affected a single parse.

So the cache stores the *parse*, not the score. Metrics change constantly;
extraction output does not. Re-scoring a cached corpus after a metric fix takes
seconds instead of hours, which is what makes it practical to keep improving
the metrics rather than freezing them to avoid the wait.

The key is (content hash, backend name, backend version). Content hash rather
than filename, so a renamed or moved document hits the same entry and an edited
one misses. Backend version so a backend change invalidates its own entries
without touching anyone else's.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .backends.base import ParseResult

DEFAULT_DIR = Path(".cache/parses")


class ParseCache:
    def __init__(self, root: str | Path = DEFAULT_DIR, enabled: bool = True) -> None:
        self.root = Path(root)
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, doc_id: str, backend: str, version: str) -> Path:
        # Shard by the first two hex chars: thousands of files in one directory
        # is slow to list on most filesystems.
        return self.root / doc_id[:2] / f"{doc_id}_{backend}_{version}.json"

    def get(self, doc_id: str, backend: str, version: str) -> ParseResult | None:
        if not self.enabled:
            return None
        path = self._path(doc_id, backend, version)
        if not path.exists():
            self.misses += 1
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # A truncated entry from an interrupted write is a miss, not a
            # crash. Re-parsing is slow but correct.
            self.misses += 1
            return None
        self.hits += 1
        return ParseResult(**data)

    def put(self, result: ParseResult, backend: str, version: str) -> None:
        if not self.enabled:
            return
        path = self._path(result.doc_id, backend, version)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file and rename: an interrupted run must not leave a
        # half-written entry that looks valid.
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(result)), encoding="utf-8")
        tmp.replace(path)

    def clear(self) -> int:
        if not self.root.exists():
            return 0
        n = sum(1 for _ in self.root.rglob("*.json"))
        shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        return n

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
            "entries": sum(1 for _ in self.root.rglob("*.json"))
            if self.root.exists() else 0,
        }
