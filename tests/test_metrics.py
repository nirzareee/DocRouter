"""Tests for the scoring functions.

A benchmark whose metrics are untested is a random number generator with a
results table. Each test below encodes a property the metric must have for the
eventual claims in the writeup to mean anything.
"""

from __future__ import annotations

import pytest

from docrouter.metrics import score_document
from docrouter.metrics.teds import (
    extract_markdown_tables,
    rows_to_tree,
    teds,
    teds_from_markdown,
)
from docrouter.metrics.text import reading_order_score, text_similarity

GOLD_TABLE = """# Q3 Financial Summary

Revenue grew in three of four regions.

| Region | Revenue ($M) | Change vs Q2 |
| --- | --- | --- |
| North | 128.4 | +6.1% |
| South | 95.2 | +2.8% |
| East | 143.9 | +11.4% |
| West | 61.7 | -4.3% |
"""

FLATTENED = (
    "Q3 Financial Summary Revenue grew in three of four regions. "
    "Region Revenue ($M) Change vs Q2 North 128.4 +6.1% South 95.2 +2.8% "
    "East 143.9 +11.4% West 61.7 -4.3%"
)


class TestTeds:
    def test_identical_tables_score_one(self):
        assert teds_from_markdown(GOLD_TABLE, GOLD_TABLE) == pytest.approx(1.0)

    def test_flattened_table_scores_zero(self):
        # The core claim of the whole project: text survives, structure does not.
        assert teds_from_markdown(FLATTENED, GOLD_TABLE) == 0.0

    def test_flattened_table_keeps_high_text_similarity(self):
        # Same input, opposite verdict. This is why text similarity alone is not
        # a sufficient benchmark metric.
        assert text_similarity(FLATTENED, GOLD_TABLE) > 0.9

    def test_missing_row_lowers_but_does_not_zero_score(self):
        truncated = "\n".join(GOLD_TABLE.splitlines()[:-1])
        score = teds_from_markdown(truncated, GOLD_TABLE)
        assert 0.5 < score < 1.0

    def test_swapped_cell_content_penalized_by_teds_not_teds_s(self):
        swapped = GOLD_TABLE.replace("128.4", "999.9")
        assert teds_from_markdown(swapped, GOLD_TABLE) < 1.0
        assert teds_from_markdown(swapped, GOLD_TABLE, structure_only=True) == pytest.approx(1.0)

    def test_spurious_table_is_penalized(self):
        extra = GOLD_TABLE + "\n| A | B |\n| --- | --- |\n| 1 | 2 |\n"
        assert teds_from_markdown(extra, GOLD_TABLE) < 1.0

    def test_transposed_table_is_not_a_free_pass(self):
        rows = [["Region", "North", "South"], ["Revenue", "128.4", "95.2"]]
        original = [["Region", "Revenue"], ["North", "128.4"], ["South", "95.2"]]
        assert teds(rows_to_tree(rows), rows_to_tree(original)) < 0.9


class TestMarkdownTableExtraction:
    def test_finds_table_and_drops_separator_row(self):
        tables = extract_markdown_tables(GOLD_TABLE)
        assert len(tables) == 1
        assert len(tables[0]) == 5  # header + 4 data rows, no |---| row
        assert tables[0][0][0] == "Region"

    def test_prose_only_document_has_no_tables(self):
        assert extract_markdown_tables("# Title\n\nJust prose.") == []


class TestReadingOrder:
    GOLD = (
        "The committee reviewed operating expenses. "
        "Headcount growth drove the increase. "
        "Facilities costs remained flat. "
        "Marketing spend rose sharply in the period."
    )

    def test_correct_order_scores_one(self):
        assert reading_order_score(self.GOLD, self.GOLD) == pytest.approx(1.0)

    def test_reversed_order_scores_zero(self):
        sentences = [s.strip() + "." for s in self.GOLD.split(". ") if s.strip()]
        reversed_doc = " ".join(reversed(sentences))
        assert reading_order_score(reversed_doc, self.GOLD) < 0.1

    def test_survives_line_wrapping_differences(self):
        # A backend that emits one line per sentence must not be punished for it.
        wrapped = self.GOLD.replace(". ", ".\n")
        assert reading_order_score(wrapped, self.GOLD) == pytest.approx(1.0)

    def test_partial_recovery_is_penalized_by_coverage(self):
        half = "The committee reviewed operating expenses."
        assert reading_order_score(half, self.GOLD) < 0.5


class TestComposite:
    def test_table_free_document_excludes_teds(self):
        gold = "# Title\n\nA sentence long enough to count as a unit here."
        score = score_document(gold, gold, "d1", "b1")
        assert not score.has_gold_tables
        # Composite must be a clean 1.0 from two metrics, not diluted by a
        # meaningless table score.
        assert score.overall == pytest.approx(1.0)

    def test_failed_parse_scores_zero(self):
        score = score_document("", GOLD_TABLE, "d1", "b1", failed=True)
        assert score.overall == 0.0
        assert score.failed
