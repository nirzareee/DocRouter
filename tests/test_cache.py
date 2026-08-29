"""Tests for the parse cache.

A cache that silently returns stale results is worse than no cache: every
number downstream would be wrong and nothing would look broken.
"""

from __future__ import annotations

from pathlib import Path

from docrouter.backends.base import ParseResult
from docrouter.cache import ParseCache


def _result(doc_id: str = "abc123", markdown: str = "# Hi") -> ParseResult:
    return ParseResult(
        markdown=markdown, backend="pylib", doc_id=doc_id,
        wall_time_s=1.5, cost_usd=0.01, pages=3,
    )


class TestRoundTrip:
    def test_put_then_get_returns_equal_result(self, tmp_path: Path):
        cache = ParseCache(tmp_path)
        cache.put(_result(), "pylib", "1")
        got = cache.get("abc123", "pylib", "1")
        assert got is not None
        assert got.markdown == "# Hi"
        assert got.pages == 3
        assert got.wall_time_s == 1.5

    def test_miss_returns_none(self, tmp_path: Path):
        assert ParseCache(tmp_path).get("nope", "pylib", "1") is None


class TestInvalidation:
    def test_different_backend_is_a_miss(self, tmp_path: Path):
        cache = ParseCache(tmp_path)
        cache.put(_result(), "pylib", "1")
        assert cache.get("abc123", "docling", "1") is None

    def test_different_version_is_a_miss(self, tmp_path: Path):
        # Bumping Backend.version must invalidate that backend's entries.
        cache = ParseCache(tmp_path)
        cache.put(_result(), "pylib", "1")
        assert cache.get("abc123", "pylib", "2") is None

    def test_different_document_is_a_miss(self, tmp_path: Path):
        cache = ParseCache(tmp_path)
        cache.put(_result(doc_id="aaa"), "pylib", "1")
        assert cache.get("bbb", "pylib", "1") is None


class TestRobustness:
    def test_corrupt_entry_is_a_miss_not_a_crash(self, tmp_path: Path):
        cache = ParseCache(tmp_path)
        cache.put(_result(), "pylib", "1")
        path = next(tmp_path.rglob("*.json"))
        path.write_text("{ truncated", encoding="utf-8")
        assert cache.get("abc123", "pylib", "1") is None

    def test_disabled_cache_never_stores_or_returns(self, tmp_path: Path):
        cache = ParseCache(tmp_path, enabled=False)
        cache.put(_result(), "pylib", "1")
        assert cache.get("abc123", "pylib", "1") is None

    def test_failed_parses_are_cached_too(self, tmp_path: Path):
        # Which documents a backend fails on is training signal for the router,
        # so failures must survive a round trip rather than being retried.
        cache = ParseCache(tmp_path)
        failed = ParseResult(markdown="", backend="pylib", doc_id="f1",
                             error="ValueError: no text layer")
        cache.put(failed, "pylib", "1")
        got = cache.get("f1", "pylib", "1")
        assert got is not None and not got.ok
        assert "no text layer" in got.error


class TestStats:
    def test_hit_and_miss_counted(self, tmp_path: Path):
        cache = ParseCache(tmp_path)
        cache.put(_result(), "pylib", "1")
        cache.get("abc123", "pylib", "1")
        cache.get("missing", "pylib", "1")
        st = cache.stats()
        assert st["hits"] == 1 and st["misses"] == 1
        assert st["hit_rate"] == 0.5

    def test_clear_removes_entries(self, tmp_path: Path):
        cache = ParseCache(tmp_path)
        cache.put(_result(doc_id="a1"), "pylib", "1")
        cache.put(_result(doc_id="b2"), "pylib", "1")
        assert cache.clear() == 2
        assert cache.get("a1", "pylib", "1") is None
