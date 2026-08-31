"""Results must reach disk as they are produced, not at the end of the run.

A Docling sweep over 28 filings takes ~5 hours. One died on document 8 of 28
and produced no output file at all, because rows were held in memory until the
final write. Four hours of compute, nothing on disk.

For a runner this long, a crash is a normal outcome, not an exceptional one.
"""

from __future__ import annotations

import json
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
def corpus() -> Corpus:
    if not (CORPUS / "docs").is_dir():
        subprocess.run(
            [sys.executable, str(REPO / "scripts" / "make_samples.py")],
            check=True, capture_output=True, cwd=REPO,
        )
    return Corpus(CORPUS)


class TestIncrementalWrite:
    def test_rows_written_to_disk(self, corpus, tmp_path):
        out = tmp_path / "run.jsonl"
        run = evaluate(corpus, [get_backend("naive")], verbose=False, out_path=out)
        on_disk = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        assert len(on_disk) == len(run.rows)

    def test_partial_output_is_valid_jsonl(self, corpus, tmp_path):
        """A crash mid-run must leave a parseable file, not a truncated one."""
        out = tmp_path / "partial.jsonl"

        class Boom(Exception):
            pass

        class ExplodingBackend:
            name = "boom"
            version = "1"
            calls = 0

            def supports(self, path):
                return True

            def parse(self, path):
                type(self).calls += 1
                if type(self).calls > 1:
                    raise Boom("simulated crash")
                return get_backend("naive").parse(path)

        with pytest.raises(Boom):
            evaluate(corpus, [ExplodingBackend()], verbose=False, out_path=out)

        # The row scored before the crash must survive and still parse.
        lines = [l for l in out.read_text().splitlines() if l.strip()]
        assert len(lines) == 1
        json.loads(lines[0])

    def test_no_out_path_still_returns_rows(self, corpus):
        run = evaluate(corpus, [get_backend("naive")], verbose=False)
        assert len(run.rows) > 0
