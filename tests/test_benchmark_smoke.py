"""End-to-end check that the benchmark still discriminates.

Unit tests pin each metric in isolation. This pins the property that actually
matters: run the whole pipeline on the sample corpus and confirm the strawman
backend is still separable from the real one.

A metric can pass every unit test and stop discriminating anyway. Bug 3 in
docs/findings.md was exactly that -- TEDS returned 1.0 for table-free documents,
so every backend got a free point and the differences compressed toward the
mean. Nothing failed; the benchmark just stopped measuring.

Kept as a test rather than inline in the CI workflow so it runs locally with
`pytest` too.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from docrouter.backends import get_backend
from docrouter.corpus import Corpus
from docrouter.evaluate import evaluate

REPO = Path(__file__).resolve().parent.parent
CORPUS = REPO / "data" / "sample_corpus"


@pytest.fixture(scope="module")
def rows() -> dict[tuple[str, str], dict]:
    """Build the sample corpus if needed, then score it."""
    if not (CORPUS / "docs").is_dir():
        subprocess.run(
            [sys.executable, str(REPO / "scripts" / "make_samples.py")],
            check=True, capture_output=True, cwd=REPO,
        )
    run = evaluate(
        Corpus(CORPUS),
        [get_backend("naive"), get_backend("pylib")],
        verbose=False,
    )
    return {(r["doc_id"], r["backend"]): r for r in run.rows}


class TestBenchmarkDiscriminates:
    def test_structure_aware_backend_recovers_tables(self, rows):
        assert rows[("0001_table", "pylib")]["teds"] > 0.9

    def test_strawman_destroys_tables(self, rows):
        # The headline claim of the project: naive keeps the characters and
        # loses the structure entirely.
        assert rows[("0001_table", "naive")]["teds"] == 0.0

    def test_strawman_still_keeps_the_text(self, rows):
        # If this ever drops, the strawman is failing for the wrong reason and
        # the table result above stops being interesting.
        assert rows[("0001_table", "naive")]["text_sim"] > 0.9

    def test_column_detection_beats_glyph_dump(self, rows):
        pylib = rows[("0002_twocol", "pylib")]["reading_order"]
        naive = rows[("0002_twocol", "naive")]["reading_order"]
        assert pylib > naive

    def test_composite_separates_the_backends(self, rows):
        for doc in ("0001_table", "0002_twocol"):
            assert rows[(doc, "pylib")]["overall"] > rows[(doc, "naive")]["overall"]

    def test_no_backend_failed(self, rows):
        for key, row in rows.items():
            assert not row["failed"], f"{key} failed: {row['error']}"
