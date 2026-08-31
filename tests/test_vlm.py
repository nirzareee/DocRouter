"""Tests for the VLM backend.

No network. The API call is stubbed so the parts that can be wrong without
erroring -- cost arithmetic, page capping, usage accounting -- are checked
directly.

Cost is the reason this backend exists: it is the only tier with a real price,
so a Pareto curve is only as trustworthy as this arithmetic.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from docrouter.backends.vlm import VLMBackend


class _FakeBlock:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _FakeClient:
    """Stands in for anthropic.Anthropic()."""

    def __init__(self, page_text: str = "# Page\n\n| A | B |\n| --- | --- |\n| 1 | 2 |"):
        self.calls = 0
        self.page_text = page_text
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            content=[_FakeBlock(self.page_text)],
            usage=SimpleNamespace(input_tokens=1500, output_tokens=400),
        )


class TestAvailability:
    def test_unavailable_without_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert not VLMBackend().is_available()

    def test_supports_pdf_only(self):
        b = VLMBackend()
        assert b.supports(Path("x.pdf"))
        assert not b.supports(Path("x.docx"))

    def test_supports_pdfs_without_text_layer(self):
        # The reason this tier exists: it does not need a text layer, so it
        # competes on the degraded condition where pylib scores 0.000.
        assert VLMBackend().supports(Path("scanned.pdf"))


class TestCostAccounting:
    @pytest.fixture
    def priced_backend(self, monkeypatch):
        monkeypatch.setenv("VLM_INPUT_USD_PER_MTOK", "3.00")
        monkeypatch.setenv("VLM_OUTPUT_USD_PER_MTOK", "15.00")
        return VLMBackend()

    def test_cost_computed_from_reported_usage(self, priced_backend, monkeypatch):
        client = _FakeClient()
        monkeypatch.setattr(priced_backend, "_render_pages", lambda p: (["img"] * 3, 3))
        monkeypatch.setattr(
            "anthropic.Anthropic", lambda *a, **k: client, raising=False
        )
        md, pages, meta = priced_backend._parse(Path("fake.pdf"))

        # 3 pages x 1500 in / 400 out
        assert meta["input_tokens"] == 4500
        assert meta["output_tokens"] == 1200
        expected = 4500 / 1e6 * 3.00 + 1200 / 1e6 * 15.00
        assert meta["measured_cost_usd"] == pytest.approx(expected)

    def test_unpriced_backend_warns(self, monkeypatch):
        monkeypatch.setenv("VLM_INPUT_USD_PER_MTOK", "0")
        monkeypatch.setenv("VLM_OUTPUT_USD_PER_MTOK", "0")
        b = VLMBackend()
        client = _FakeClient()
        monkeypatch.setattr(b, "_render_pages", lambda p: (["img"], 1))
        monkeypatch.setattr("anthropic.Anthropic", lambda *a, **k: client, raising=False)
        _, _, meta = b._parse(Path("fake.pdf"))
        # Reporting $0.00 silently would put a false point on the Pareto curve.
        assert "warning" in meta


class TestPageHandling:
    def test_one_api_call_per_page(self, monkeypatch):
        b = VLMBackend()
        client = _FakeClient()
        monkeypatch.setattr(b, "_render_pages", lambda p: (["img"] * 5, 5))
        monkeypatch.setattr("anthropic.Anthropic", lambda *a, **k: client, raising=False)
        b._parse(Path("fake.pdf"))
        assert client.calls == 5

    def test_truncation_is_recorded(self, monkeypatch):
        # A 562-page filing must not silently cost a fortune, and a truncated
        # parse must not be scored as if it were complete.
        b = VLMBackend()
        client = _FakeClient()
        monkeypatch.setattr(b, "_render_pages", lambda p: (["img"] * 40, 562))
        monkeypatch.setattr("anthropic.Anthropic", lambda *a, **k: client, raising=False)
        _, _, meta = b._parse(Path("fake.pdf"))
        assert meta["truncated"]
        assert meta["pages_total"] == 562
        assert meta["pages_sent"] == 40

    def test_blank_pages_dropped(self, monkeypatch):
        b = VLMBackend()
        client = _FakeClient(page_text="   ")
        monkeypatch.setattr(b, "_render_pages", lambda p: (["img"] * 3, 3))
        monkeypatch.setattr("anthropic.Anthropic", lambda *a, **k: client, raising=False)
        md, _, _ = b._parse(Path("fake.pdf"))
        assert md == ""


class TestRegistration:
    def test_registered_in_backend_registry(self):
        from docrouter.backends import REGISTRY

        assert "vlm" in REGISTRY
