"""Tests for HTML -> gold conversion.

The gold generator is the most dangerous component in the project: if it is
wrong, every number downstream is wrong in a way no amount of careful
benchmarking will reveal.
"""

from __future__ import annotations

from docrouter.edgar.html_to_gold import (
    _is_layout_table,
    clean_cell,
    html_to_gold,
    table_to_rows,
)
from lxml import html as lxml_html


def _table(html: str):
    return lxml_html.fromstring(html)


class TestCellCleaning:
    def test_strips_nbsp(self):
        assert clean_cell("1,284\xa0") == "1,284"

    def test_escapes_pipes(self):
        assert clean_cell("a|b") == r"a\|b"


class TestSpans:
    def test_colspan_expands(self):
        rows = table_to_rows(_table(
            "<table><tr><th colspan='2'>Three Months</th></tr>"
            "<tr><td>2026</td><td>2025</td></tr></table>"
        ))
        assert rows[0] == ["Three Months", "Three Months"]

    def test_rowspan_expands(self):
        rows = table_to_rows(_table(
            "<table><tr><td rowspan='2'>Cat</td><td>A</td></tr>"
            "<tr><td>B</td></tr></table>"
        ))
        assert rows[0][0] == "Cat" and rows[1][0] == "Cat"
        assert rows[1][1] == "B"


class TestLayoutTableRejection:
    def test_single_row_is_layout(self):
        assert _is_layout_table([["a", "b"]])

    def test_mostly_empty_is_layout(self):
        assert _is_layout_table([["", ""], ["", ""], ["x", ""]])

    def test_real_data_table_is_kept(self):
        rows = [["Segment", "2026"], ["North", "1,284"], ["Europe", "952"]]
        assert not _is_layout_table(rows)


class TestEndToEnd:
    HTML = """<html><body>
    <table><tr><td></td><td></td></tr></table>
    <h2>Item 2. Discussion</h2>
    <p>Revenue increased in three of four reportable segments this quarter.</p>
    <table>
      <tr><th>Segment</th><th>2026</th></tr>
      <tr><td>North America</td><td>1,284</td></tr>
      <tr><td>Europe</td><td>952</td></tr>
    </table>
    </body></html>"""

    def test_drops_spacer_table_keeps_data_table(self):
        gold = html_to_gold(self.HTML)
        assert gold.n_tables == 1

    def test_emits_pipe_table(self):
        gold = html_to_gold(self.HTML)
        assert "| Segment | 2026 |" in gold.markdown
        assert "| North America | 1,284 |" in gold.markdown

    def test_output_round_trips_through_the_metric(self):
        # Gold must score 1.0 against itself, or the annotation format and the
        # metric disagree about what a table is.
        from docrouter.metrics import score_document

        gold = html_to_gold(self.HTML).markdown
        score = score_document(gold, gold, "d", "b")
        assert score.overall == 1.0
        assert score.has_gold_tables


class TestRealFilingQuirks:
    """Regressions from actual EDGAR filings, not synthetic HTML."""

    IXBRL = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<html xmlns:ix="http://www.xbrl.org/2013/inlineXBRL"><body>'
        "<ix:header><ix:hidden>tagged fact junk</ix:hidden></ix:header>"
        "<h2>Item 2. Discussion</h2>"
        "<p>Revenue increased across three of four reportable segments.</p>"
        "<table><tr><th>Segment</th><th>2026</th></tr>"
        "<tr><td>Americas</td><td>1,284</td></tr>"
        "<tr><td>Europe</td><td>952</td></tr></table>"
        "</body></html>"
    )

    def test_xml_declaration_does_not_crash(self):
        # lxml rejects str input carrying an encoding declaration; modern
        # filings are inline XBRL and always have one.
        gold = html_to_gold(self.IXBRL)
        assert gold.n_tables == 1

    def test_bytes_input_works(self):
        gold = html_to_gold(self.IXBRL.encode("utf-8"))
        assert gold.n_tables == 1

    def test_xbrl_hidden_header_excluded(self):
        # Hidden tagged facts never render in the PDF, so they must not appear
        # in ground truth.
        gold = html_to_gold(self.IXBRL)
        assert "junk" not in gold.markdown
