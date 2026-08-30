"""Tests for the routing feature extractor.

Two properties matter and they pull against each other:

1. Features must actually separate the conditions the router decides between.
2. They must be cheap enough that computing them is not itself the cost the
   router exists to avoid.

A feature that needs a backend to compute is not a routing feature.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from docrouter.features import extract_features

REPO = Path(__file__).resolve().parent.parent
CLEAN = REPO / "data" / "sample_corpus" / "docs" / "0002_twocol.pdf"


@pytest.fixture(scope="module")
def clean_pdf() -> Path:
    if not CLEAN.exists():
        subprocess.run(
            [sys.executable, str(REPO / "scripts" / "make_samples.py")],
            check=True, capture_output=True, cwd=REPO,
        )
    return CLEAN


@pytest.fixture(scope="module")
def degraded_pdf(clean_pdf: Path, tmp_path_factory) -> Path:
    sys.path.insert(0, str(REPO))
    from scripts.degrade import degrade_pdf

    out = tmp_path_factory.mktemp("deg") / "degraded.pdf"
    degrade_pdf(clean_pdf, out, "medium", seed=0)
    return out


class TestTextLayerDetection:
    """The single most useful routing feature."""

    def test_clean_pdf_has_text_layer(self, clean_pdf):
        assert extract_features(clean_pdf).has_text_layer

    def test_degraded_pdf_has_no_text_layer(self, degraded_pdf):
        # Every cheap backend scores exactly 0.000 on these. If this check ever
        # returns True for a rasterized page, the router will send scanned
        # documents to a backend that cannot read them.
        assert not extract_features(degraded_pdf).has_text_layer

    def test_conditions_are_separable(self, clean_pdf, degraded_pdf):
        assert not extract_features(clean_pdf).is_scanned
        assert extract_features(degraded_pdf).is_scanned


class TestLayoutFeatures:
    def test_two_column_document_detected(self, clean_pdf):
        assert extract_features(clean_pdf).est_columns == 2

    def test_image_ratio_near_zero_for_digital(self, clean_pdf):
        assert extract_features(clean_pdf).image_area_ratio < 0.1

    def test_image_ratio_near_one_for_rasterized(self, degraded_pdf):
        # A scanned page is one full-page image and nothing else.
        assert extract_features(degraded_pdf).image_area_ratio > 0.8


class TestCost:
    def test_feature_extraction_is_fast(self, clean_pdf):
        # Docling takes ~180s on a 40-page filing. Features must be orders of
        # magnitude cheaper or the router cannot pay for itself.
        assert extract_features(clean_pdf).feature_time_ms < 2000

    def test_page_sampling_bounds_cost(self, clean_pdf):
        from docrouter.features import SAMPLE_PAGES

        # Cost must be flat in document length, not linear, or feature
        # extraction becomes expensive on exactly the documents where routing
        # matters most.
        assert SAMPLE_PAGES <= 16


class TestSchema:
    def test_serializes_for_jsonl(self, clean_pdf):
        import json

        d = extract_features(clean_pdf).to_dict()
        json.loads(json.dumps(d))
        assert d["doc_key"] == "0002_twocol"
        assert d["page_count"] == 1
