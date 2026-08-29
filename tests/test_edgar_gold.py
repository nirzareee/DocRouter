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
    """Spanned cells are placed once, at the span origin.

    Repeating the text across the span looks reasonable but breaks on real
    filings, which use colspan for alignment: `colspan=3` on every cell
    triples the table width and fills it with duplicated text. Placing once
    and leaving continuations empty lets pure-layout padding be dropped while
    genuine spans keep their columns.
    """

    def test_colspan_places_text_once(self):
        rows = table_to_rows(_table(
            "<table><tr><th colspan='2'>Three Months</th></tr>"
            "<tr><td>2026</td><td>2025</td></tr></table>"
        ))
        assert rows[0] == ["Three Months", ""]
        assert rows[1] == ["2026", "2025"]

    def test_rowspan_places_text_once(self):
        rows = table_to_rows(_table(
            "<table><tr><td rowspan='2'>Cat</td><td>A</td></tr>"
            "<tr><td>B</td></tr></table>"
        ))
        assert rows[0][0] == "Cat"
        assert rows[1][0] == ""
        assert rows[1][1] == "B"

    def test_layout_colspan_collapses(self):
        # Apple-style: colspan=3 used purely for alignment. The padding columns
        # must not survive into ground truth.
        rows = table_to_rows(_table(
            "<table>"
            "<tr><td colspan='3'>Americas</td><td colspan='3'>1,284</td></tr>"
            "<tr><td colspan='3'>Europe</td><td colspan='3'>952</td></tr>"
            "</table>"
        ))
        from docrouter.edgar.html_to_gold import _drop_empty_columns
        collapsed = _drop_empty_columns(rows)
        assert [len(r) for r in collapsed] == [2, 2]
        assert collapsed[0] == ["Americas", "1,284"]


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


class TestSymbolColumnMerging:
    """Filings put '$' in its own column for alignment.

    Left alone this doubles the column count and shifts colspan headers out of
    register, so a header lands above the currency symbol instead of above the
    number it labels. Merging is an annotation decision, recorded in
    docs/annotation-guide.md.
    """

    APPLE_STYLE = (
        "<html><body><h2>Item 1. Financial Statements</h2><table>"
        "<tr><th colspan='3'>Segment</th><th colspan='3'>2026</th>"
        "<th colspan='3'>2025</th></tr>"
        "<tr><td colspan='3'>Americas</td><td colspan='2'>$</td><td>1,284</td>"
        "<td colspan='2'>$</td><td>1,210</td></tr>"
        "<tr><td colspan='3'>Other</td><td colspan='2'>$</td><td>(617)</td>"
        "<td colspan='2'>$</td><td>(591)</td></tr>"
        "</table></body></html>"
    )

    def _table(self):
        from docrouter.metrics.teds import extract_markdown_tables

        return extract_markdown_tables(html_to_gold(self.APPLE_STYLE).markdown)[0]

    def test_layout_colspan_does_not_inflate_width(self):
        # Nine HTML columns, three real ones.
        assert max(len(r) for r in self._table()) == 3

    def test_headers_align_with_their_values(self):
        t = self._table()
        assert t[0] == ["Segment", "2026", "2025"]

    def test_currency_symbol_merged_into_value(self):
        assert self._table()[1] == ["Americas", "$1,284", "$1,210"]

    def test_parenthesized_negatives_preserved(self):
        # Filings write -617 as (617). Preserve source notation: this is an
        # extraction benchmark, not an interpretation one.
        assert self._table()[2] == ["Other", "$(617)", "$(591)"]


class TestLayoutTableRejectionOnRealPatterns:
    def test_checkbox_block_rejected(self):
        html = (
            "<html><body><table>"
            "<tr><td>Large accelerated filer</td><td>\u2612</td>"
            "<td>Accelerated filer</td><td>\u2610</td></tr>"
            "<tr><td>Non-accelerated filer</td><td>\u2610</td>"
            "<td>Smaller reporting</td><td>\u2610</td></tr>"
            "</table></body></html>"
        )
        assert html_to_gold(html).n_tables == 0

    def test_table_of_contents_rejected(self):
        rows = [
            ["Item 1.", "Financial Statements", "1"],
            ["Item 2.", "Management's Discussion", "13"],
            ["Item 3.", "Market Risk", "24"],
            ["Item 4.", "Controls and Procedures", "25"],
        ]
        assert _is_layout_table(rows)

    def test_text_only_table_rejected(self):
        rows = [["Name", "Title"], ["A Person", "Chief Executive"],
                ["Another", "Chief Financial"]]
        assert _is_layout_table(rows)


class TestSparseCurrencyColumns:
    """Regression from Apple's actual income statement.

    An earlier rule classified a currency column by asking whether most of its
    values were symbols. That fails on real filings: companies print '$' only
    on the first row of a section and on totals, so a genuine currency column
    can be one-third symbols. Colspan header origins land in the same column,
    adding date labels to it.

    The working rule looks at adjacent column pairs: in every row where both
    the symbol column and the value column are populated, the left one is
    always just a symbol.
    """

    ROWS = [
        ["", "Three Months Ended", "", "", "", "Nine Months Ended", "", "", ""],
        ["", "June 27, 2026", "", "June 28, 2025", "", "June 27, 2026", "",
         "June 28, 2025", ""],
        ["Net sales:", "", "", "", "", "", "", "", ""],
        ["Products", "$", "78,678", "$", "66,613", "$", "272,629", "$", "233,287"],
        ["Services", "", "27,421", "", "24,213", "", "80,110", "", "71,224"],
        ["Total net sales", "$", "106,099", "$", "90,826", "$", "352,739",
         "$", "304,511"],
    ]

    def _collapsed(self):
        from docrouter.edgar.html_to_gold import (
            _drop_empty_columns,
            _merge_symbol_columns,
        )

        return _drop_empty_columns(_merge_symbol_columns(self.ROWS))

    def test_sparse_symbol_column_still_detected(self):
        # Only 2 of 6 rows carry '$' in the currency column.
        assert max(len(r) for r in self._collapsed()) == 5

    def test_values_keep_their_currency(self):
        rows = self._collapsed()
        assert rows[3] == ["Products", "$78,678", "$66,613", "$272,629", "$233,287"]

    def test_rows_without_symbol_are_not_shifted(self):
        # The Services row has no '$' at all and must stay in register with the
        # rows that do.
        rows = self._collapsed()
        assert rows[4] == ["Services", "27,421", "24,213", "80,110", "71,224"]

    def test_date_headers_land_over_their_columns(self):
        rows = self._collapsed()
        assert rows[1] == ["", "June 27, 2026", "June 28, 2025",
                           "June 27, 2026", "June 28, 2025"]

    def test_label_column_is_not_merged(self):
        # Column 0 holds row labels, not symbols, and must survive.
        assert self._collapsed()[3][0] == "Products"
