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


class TestPerformanceContracts:
    """These pin scaling behavior, not just correctness.

    The original pure-Python metrics were correct and unusably slow: a 130 KB
    filing took roughly an hour per comparison. Correct-but-quadratic is a real
    failure mode for a benchmark harness, so it gets tests.
    """

    @staticmethod
    def _doc(n_sentences: int) -> str:
        words = "revenue segment quarter increased across reportable segments".split()
        return " ".join(
            " ".join(words[i % len(words)] for i in range(s, s + 12)) + "."
            for s in range(n_sentences)
        )

    def test_large_document_scores_quickly(self):
        import time

        from docrouter.metrics.text import reading_order_score, text_similarity

        doc = self._doc(800)
        assert len(doc) > 50_000

        start = time.perf_counter()
        text_similarity(doc, doc)
        reading_order_score(doc, doc)
        elapsed = time.perf_counter() - start
        assert elapsed < 20, f"metrics took {elapsed:.1f}s on a {len(doc)//1024}KB doc"

    def test_teds_variants_share_work(self):
        # teds_both must agree with computing each separately.
        from docrouter.metrics.teds import teds_both_from_markdown, teds_from_markdown

        both = teds_both_from_markdown(GOLD_TABLE, GOLD_TABLE)
        separate = (
            teds_from_markdown(GOLD_TABLE, GOLD_TABLE),
            teds_from_markdown(GOLD_TABLE, GOLD_TABLE, structure_only=True),
        )
        assert both == pytest.approx(separate)

    def test_table_matching_is_content_based(self):
        # Two tables in swapped order must still pair with their counterparts.
        from docrouter.metrics.teds import teds_from_markdown

        t1 = "| A | B |\n| --- | --- |\n| 1 | 2 |"
        t2 = "| X | Y |\n| --- | --- |\n| 9 | 8 |"
        assert teds_from_markdown(f"{t2}\n\n{t1}", f"{t1}\n\n{t2}") == pytest.approx(1.0)


class TestCanonicalization:
    """Predictions and gold are normalized identically before scoring.

    Filings put '$' in its own column. The gold generator merges it; pdfplumber
    does not. Scored raw, a backend that extracted every number correctly lost
    ~0.4 TEDS purely on that convention difference, which means the benchmark
    was ranking house style rather than extraction quality.

    Canonicalization must remove that penalty WITHOUT hiding real errors.
    """

    GOLD = (
        "| | Q3 | Q2 |\n| --- | --- | --- |\n"
        "| Products | $78,678 | $66,613 |\n"
        "| Services | 27,421 | 24,213 |"
    )
    SPLIT_SYMBOL = (
        "| | Q3 | | Q2 | |\n| --- | --- | --- | --- | --- |\n"
        "| Products | $ | 78,678 | $ | 66,613 |\n"
        "| Services | | 27,421 | | 24,213 |"
    )

    def test_currency_convention_does_not_penalize(self):
        assert teds_from_markdown(self.SPLIT_SYMBOL, self.GOLD) == pytest.approx(1.0)

    def test_missing_row_still_penalized(self):
        truncated = "\n".join(self.SPLIT_SYMBOL.splitlines()[:-1])
        assert teds_from_markdown(truncated, self.GOLD) < 1.0

    def test_wrong_number_still_penalized(self):
        wrong = self.SPLIT_SYMBOL.replace("78,678", "99,999")
        assert teds_from_markdown(wrong, self.GOLD) < 1.0

    def test_structure_only_variant_also_canonical(self):
        score = teds_from_markdown(self.SPLIT_SYMBOL, self.GOLD, structure_only=True)
        assert score == pytest.approx(1.0)

    def test_empty_padding_columns_ignored(self):
        padded = (
            "| | | Q3 | | Q2 |\n| --- | --- | --- | --- | --- |\n"
            "| Products | | $78,678 | | $66,613 |\n"
            "| Services | | 27,421 | | 24,213 |"
        )
        assert teds_from_markdown(padded, self.GOLD) == pytest.approx(1.0)

    def test_raw_extraction_still_available(self):
        from docrouter.metrics.teds import extract_markdown_tables

        raw = extract_markdown_tables(self.SPLIT_SYMBOL, canonical=False)
        canon = extract_markdown_tables(self.SPLIT_SYMBOL, canonical=True)
        assert max(len(r) for r in raw[0]) == 5
        assert max(len(r) for r in canon[0]) == 3


class TestReadingOrderAtScale:
    """The blocking path must agree with exact matching, not just be fast.

    Large filings (a bank 10-K has ~5,600 paragraphs) made full pred x gold
    matching quadratic: minutes per backend per document, slower than some
    backends being measured. Candidates are now narrowed by shared rare words
    before scoring.

    An earlier attempt narrowed by *position*, assuming a predicted block sits
    near its gold counterpart. It was 33x faster and wrong: a backend that
    dropped half the document scored 0.030 instead of 0.500, because every
    surviving block was displaced past the window. These tests exist so that
    class of optimization cannot land silently again.
    """

    @staticmethod
    def _sentences(n: int) -> list[str]:
        import random

        rng = random.Random(0)
        words = "revenue segment quarter allowance deferred operating expense".split()
        return [
            " ".join(rng.choice(words) for _ in range(18)) + f" marker{i}."
            for i in range(n)
        ]

    @pytest.mark.parametrize("n", [400, 5000])
    def test_identical_scores_one(self, n):
        from docrouter.metrics.text import reading_order_score

        doc = " ".join(self._sentences(n))
        assert reading_order_score(doc, doc) == pytest.approx(1.0)

    @pytest.mark.parametrize("n", [400, 5000])
    def test_reversed_scores_zero(self, n):
        from docrouter.metrics.text import reading_order_score

        s = self._sentences(n)
        assert reading_order_score(" ".join(reversed(s)), " ".join(s)) < 0.05

    @pytest.mark.parametrize("n", [400, 5000])
    def test_half_the_document_missing_scores_about_half(self, n):
        # The regression that killed positional windowing. Dropped content must
        # cost coverage, not collapse the score to zero.
        from docrouter.metrics.text import reading_order_score

        s = self._sentences(n)
        score = reading_order_score(" ".join(s[: n // 2]), " ".join(s))
        assert 0.40 < score < 0.60

    @pytest.mark.parametrize("n", [400, 5000])
    def test_halves_swapped_scores_about_half(self, n):
        # A large-displacement error: correct locally, wrong globally.
        from docrouter.metrics.text import reading_order_score

        s = self._sentences(n)
        swapped = s[n // 2:] + s[: n // 2]
        score = reading_order_score(" ".join(swapped), " ".join(s))
        assert 0.40 < score < 0.60

    def test_large_document_scores_quickly(self):
        import time

        from docrouter.metrics.text import reading_order_score

        doc = " ".join(self._sentences(8000))
        start = time.perf_counter()
        reading_order_score(doc, doc)
        elapsed = time.perf_counter() - start
        assert elapsed < 30, f"took {elapsed:.1f}s on 8000 sentences"
