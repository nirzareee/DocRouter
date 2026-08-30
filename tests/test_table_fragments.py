"""Stitching pdfplumber's single-row table fragments back together.

pdfplumber's line-based detection returns one Table per ruled row on some
filings. A 97-page Microsoft 10-Q produced 235 "tables", most of them a single
row like ['Revenue:', '', '', ...]. Every one was then discarded downstream,
because a table needs at least two rows to be a table.

The document scored TEDS 0.011 against 41 real tables. That looked like a hard
document. It was an unassembled one, and the same failure was deflating scores
across the corpus.

The geometry is unambiguous: fragments of one table share left and right edges
to within a fraction of a point and stack with about one line of vertical gap.
"""

from __future__ import annotations

from types import SimpleNamespace

from docrouter.backends.local import _group_table_fragments


def _table(x0: float, top: float, x1: float, bottom: float):
    return SimpleNamespace(bbox=(x0, top, x1, bottom))


class TestFragmentGrouping:
    # Exact bboxes from MSFT_10Q_2026-01-28, page 5.
    MSFT_PAGE_5 = [
        _table(42.0, 154.5, 570.0, 165.7),
        _table(42.0, 177.0, 570.0, 188.2),
        _table(42.0, 203.4, 570.0, 216.0),
        _table(42.0, 227.2, 570.0, 238.5),
        _table(42.0, 253.6, 570.0, 266.2),
        _table(42.0, 277.5, 570.0, 288.7),
        _table(42.0, 300.0, 570.0, 311.2),
        _table(42.0, 322.5, 570.0, 335.1),
    ]

    def test_real_fragments_become_one_table(self):
        groups = _group_table_fragments(self.MSFT_PAGE_5)
        assert len(groups) == 1
        assert len(groups[0]) == 8

    def test_already_whole_table_is_untouched(self):
        # Must be a no-op on documents that never had the problem.
        groups = _group_table_fragments([_table(42.0, 100.0, 570.0, 300.0)])
        assert len(groups) == 1 and len(groups[0]) == 1

    def test_empty_input(self):
        assert _group_table_fragments([]) == []

    def test_vertically_distant_tables_stay_separate(self):
        groups = _group_table_fragments([
            _table(42.0, 100.0, 570.0, 130.0),
            _table(42.0, 400.0, 570.0, 430.0),
        ])
        assert len(groups) == 2

    def test_different_column_extents_stay_separate(self):
        # A narrow table above a wide one is not one table.
        groups = _group_table_fragments([
            _table(42.0, 100.0, 300.0, 112.0),
            _table(42.0, 120.0, 570.0, 132.0),
        ])
        assert len(groups) == 2

    def test_side_by_side_tables_stay_separate(self):
        groups = _group_table_fragments([
            _table(42.0, 100.0, 280.0, 200.0),
            _table(300.0, 100.0, 570.0, 200.0),
        ])
        assert len(groups) == 2

    def test_grouping_is_order_independent(self):
        shuffled = list(reversed(self.MSFT_PAGE_5))
        assert len(_group_table_fragments(shuffled)) == 1

    def test_x_tolerance_absorbs_subpoint_jitter(self):
        # Real edges differ by fractions of a point; that must not split them.
        groups = _group_table_fragments([
            _table(42.0, 154.5, 570.0, 165.7),
            _table(42.4, 177.0, 569.6, 188.2),
        ])
        assert len(groups) == 1
