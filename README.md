# docrouter

Cost-aware routing for document extraction, with a reproducible benchmark
underneath it.

Most ingestion pipelines pick one PDF/Office parser and use it for everything.
That is either wasteful or wrong: expensive vision-language parsers are overkill
for a clean digital docx, and pure-Python extraction collapses on a scanned
two-column page. This project measures where each backend actually wins, then
routes each document to the cheapest backend that will still clear a quality
bar.

The benchmark comes first. The router is only interesting if the numbers behind
it are trustworthy.

## Status

Working: the evaluation harness, three metrics, two baseline backends, a CLI,
and a test suite that pins the metric behavior.

Not built yet: the model-based backends, the router itself, and the LangGraph
serving pipeline. See [Roadmap](#roadmap).

## Quickstart

```bash
pip install -r requirements.txt
python scripts/make_samples.py          # synthetic smoke-test corpus
python -m docrouter.cli bench data/sample_corpus
```

```
| backend | n | text_sim | TEDS  | TEDS-S | read_order | overall | sec/doc | fail% |
| ------- | - | -------- | ----- | ------ | ---------- | ------- | ------- | ----- |
| pylib   | 2 | 1.000    | 1.000 | 1.000  | 1.000      | 1.000   | 0.01    | 0%    |
| naive   | 2 | 0.697    | 0.500 | 0.500  | 0.000      | 0.265   | 0.01    | 0%    |
```

**These numbers are a smoke test, not a result.** The corpus is two synthetic
documents. `pylib` scoring a perfect 1.000 means the corpus is too easy, not
that the backend is solved. Real numbers require real documents (see
[Corpus](#corpus)).

What the smoke test does prove is that the metrics discriminate in the right
direction: the naive backend retains 70% of the characters while scoring 0.0 on
reading order and 0.0 on the table document's TEDS.

## Why three metrics

| Metric | Question | Fails to catch |
|---|---|---|
| `text_sim` | Did the characters survive? | Everything structural |
| `TEDS` / `TEDS-S` | Did the table grid survive? | Prose ordering |
| `reading_order` | Did the sequence survive? | Content accuracy |

The single most important row in the smoke test: on the table document, the
naive backend scores **1.000 text similarity and 0.000 TEDS**. Every character
is present and the table is destroyed. A benchmark using only text-based metrics
would call that backend perfect.

- **TEDS** (Zhong et al., 2019; the table metric in OmniDocBench) renders both
  tables as trees and computes normalized tree edit distance via APTED.
  `TEDS-S` ignores cell text to isolate grid recovery from OCR accuracy.
- **reading_order** matches sentences between prediction and ground truth, then
  scores the resulting index sequence with Kendall tau scaled to [0,1],
  multiplied by coverage so a backend cannot win by emitting two correct
  sentences and stopping.

## Backends

| Name | Tier | Deps | Notes |
|---|---|---|---|
| `naive` | strawman | none | Strips XML / dumps glyphs. Exists to be beaten. |
| `pylib` | free CPU | python-docx, pdfplumber | Emits real GFM tables; geometric column detection in `layout.py`. |
| `docling` | CPU pipeline | `pip install docling` | Registered, not yet benchmarked. |

Adding one is a subclass with three methods:

```python
class MyBackend(Backend):
    name = "mine"
    cost_per_page_usd = 0.002

    def supports(self, path): return path.suffix.lower() == ".pdf"
    def is_available(self): ...   # deps importable?
    def _parse(self, path): return markdown, page_count, meta
```

Register it in `docrouter/backends/__init__.py` and it appears in every
benchmark run automatically. Failures are captured as `ParseResult.error` rather
than raised, because which documents a backend *fails* on is training signal for
the router.

## Corpus

```
corpus/
  docs/       0001.pdf, 0002.docx, ...
  gold/       0001.md,  0002.md
  meta.jsonl  {"doc_key": "0001", "columns": 2, "scanned": false, ...}
```

`meta.jsonl` carries the document attributes you will slice results by and, later,
the features the router learns from. "Backend X wins overall" is a weak finding.
"Backend X wins on scanned multi-column documents and loses everywhere else" is
the finding that justifies building a router.

Two real corpora to layer in:

1. **OmniDocBench** (CVPR 2025) — 1,355 annotated pages across 9 document types.
   The standard comparison set. Needs an adapter from its annotation format to
   `gold/*.md`.
2. **Your own domain set** — SEC filings from EDGAR are free, table-dense, and
   legible in an interview. Annotate 30–50 documents by hand. Painful, and it is
   the part nobody else will have done.

## Design notes

**Failures are data, not exceptions.** `Backend.parse` never raises. A backend
that crashes on scanned PDFs is a backend the router should avoid for scanned
PDFs, which only works if the crash is recorded.

**Per-pair rows, not aggregates.** `results/run.jsonl` holds one row per
(document, backend). Means hide the distribution, and the distribution is where
the routing decisions live.

**TEDS is excluded from the composite when the ground truth has no tables.**
Otherwise every prose document hands every backend a free 1.0 and compresses the
whole benchmark toward the mean. This was a real bug caught during development;
the tests now pin it.

**Reading order is measured on sentences, not paragraphs.** Paragraph splitting
depends on blank lines surviving extraction, which they often do not. The first
version scored a correct parse at 0.0 because the backend emitted one line per
row. The metric was measuring newline style, not reading order.

## Roadmap

- [x] Metrics: TEDS, TEDS-S, text similarity, reading order
- [x] Harness: corpus loader, runner, JSONL output, results table
- [x] Baseline backends + geometric column detection
- [ ] OmniDocBench adapter
- [ ] Docling / Marker / MinerU backends benchmarked
- [ ] VLM backend (page images to a frontier model)
- [ ] Feature extractor: cheap pre-parse signals (text layer present, column
      count, table density, page count)
- [ ] Router: rules baseline, then a learned classifier; report the cost/accuracy
      Pareto curve against always-cheapest and always-best
- [ ] LangGraph pipeline: grounding check, escalate to a stronger backend on
      failure, checkpointing, content-hash cache
- [ ] FastAPI service, Docker, CI

## References

- Ouyang et al., *OmniDocBench* (CVPR 2025) — arXiv:2412.07626
- Zhong et al., *Image-based table recognition* (PubTabNet, TEDS) — arXiv:1911.10683
- Pawlik & Augsten, *APTED* — the tree edit distance implementation used here
