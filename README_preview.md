# docrouter

Cost-aware routing for document extraction, with a reproducible benchmark
underneath it.

Most ingestion pipelines pick one PDF parser and use it for everything. That is
either wasteful or wrong: a neural document pipeline is overkill for a clean
digital filing, and pure-Python extraction returns literally nothing on a
scanned page. **docrouter** measures where each backend actually wins, then
routes each document to the cheapest one that still clears a quality bar.

The benchmark comes first. The router is only interesting if the numbers behind
it are trustworthy.

---

## Headline result

One SEC 10-Q under two conditions sharing a single ground-truth file: a clean
PDF rendered from the source HTML, and a degraded copy rasterized with skew,
blur, noise, and JPEG artifacts. Same content, same gold, one variable.

| backend | clean | degraded | sec/doc (clean) |
|---|---|---|---|
| docling | **0.675** | **0.654** | 179.2 |
| pylib | **0.658** | **0.000** | 2.3 |
| naive | **0.452** | **0.000** | 1.9 |

On the clean filing, a neural pipeline buys **0.017 overall (2.6% relative) for
78x the compute**. On the degraded one it is the difference between a usable
document and nothing at all.

The same backend choice is nearly worthless and completely essential depending
on one property of the input - and that property is detectable in milliseconds,
before any expensive parse begins. That asymmetry is the case for routing.

**n=1.** One company, one form type, one degradation severity. This is a
hypothesis with a mechanism, not an established result. Full caveats, the
component breakdown, and the bug log are in [docs/findings.md](docs/findings.md).

### The composite hides the interesting part

| metric | pylib | docling | delta |
|---|---|---|---|
| TEDS (tables) | 0.398 | 0.490 | **+0.092 (+23%)** |
| reading order | 0.861 | 0.784 | **-0.077 (-9%)** |
| text similarity | 0.716 | 0.753 | +0.037 |

Docling recovers table structure substantially better and reconstructs reading
order **worse**. They nearly cancel in the mean. A benchmark reporting one
composite number would have said "no meaningful difference" and been wrong
twice.

---

## Quickstart

```bash
pip install -r requirements.txt
python scripts/make_samples.py               # synthetic smoke-test corpus
python -m docrouter.cli bench data/sample_corpus
python -m pytest -q                          # 50 tests
```

### Reproducing the EDGAR corpus

Document binaries are not tracked. Regenerate them from tracked provenance:

```bash
export EDGAR_UA="Your Name your@email.com"   # SEC requires this
python scripts/fetch_edgar.py --tickers AAPL --forms 10-Q --limit 1
python scripts/build_corpus.py --degrade medium

python -m docrouter.cli bench data/edgar/clean    --out results/clean.jsonl
python -m docrouter.cli bench data/edgar/degraded --out results/degraded.jsonl
```

Ground truth (`data/*/gold/`), provenance (`data/edgar/sources.jsonl`), and
results (`results/*.jsonl`) **are** tracked - those are annotation and evidence.
PDFs are not; they are large and deterministically regenerable.

Before trusting generated ground truth:

```bash
python scripts/inspect_gold.py data/edgar/clean/gold/*.md
```

---

## Why three metrics

| Metric | Question | Blind to |
|---|---|---|
| `text_sim` | Did the characters survive? | Everything structural |
| `TEDS` / `TEDS-S` | Did the table grid survive? | Prose ordering |
| `reading_order` | Did the sequence survive? | Content accuracy |

The clearest evidence for keeping these separate: on the clean filing, `naive`
scores the **highest text similarity of any backend (0.764)** and **0.000
TEDS**. Every character present, every table destroyed. A text-only benchmark
would rank the strawman first.

- **TEDS** (Zhong et al., 2019; the table metric in OmniDocBench) renders both
  tables as trees and computes normalized tree edit distance via APTED.
  `TEDS-S` ignores cell text to separate grid recovery from OCR accuracy.
- **reading_order** matches sentences between prediction and gold, scores the
  resulting index sequence with Kendall tau scaled to [0,1], and multiplies by
  coverage so a backend cannot win by emitting two correct sentences and
  stopping.

Predictions and ground truth pass through identical structural normalization
(`docrouter/metrics/canonical.py`) before comparison, so no backend is rewarded
for sharing a formatting convention with the annotator. This was worth 0.19
TEDS - see bug 9 in the findings.

---

## Backends

| Name | Tier | Deps | Clean / Degraded |
|---|---|---|---|
| `naive` | strawman | none | 0.452 / 0.000 |
| `pylib` | free CPU | python-docx, pdfplumber | 0.658 / 0.000 |
| `docling` | neural CPU | `pip install docling` | 0.675 / 0.654 |

`pylib` includes geometric column detection (`docrouter/layout.py`): gutter
detection by histogram, full-width band segmentation, then reading-order
reconstruction. It is roughly what a layout model does, done with arithmetic.

Adding a backend is a subclass with three methods:

```python
class MyBackend(Backend):
    name = "mine"
    cost_per_page_usd = 0.002

    def supports(self, path): return path.suffix.lower() == ".pdf"
    def is_available(self): ...          # deps importable?
    def _parse(self, path): return markdown, page_count, meta
```

Register it in `docrouter/backends/__init__.py` and it joins every benchmark run
automatically. Failures are captured as `ParseResult.error` rather than raised -
which documents a backend fails on is training signal for the router.

---

## Corpus layout

```
corpus/
  docs/       0001.pdf, 0002.docx, ...
  gold/       0001.md,  0002.md
  meta.jsonl  {"doc_key": "0001", "columns": 2, "scanned": false, ...}
```

`meta.jsonl` carries the attributes results are sliced by and, later, the
features the router learns from. "Backend X wins overall" is a weak finding.
"Backend X wins on scanned documents and is 78x overpriced everywhere else" is
the finding that justifies building a router.

Ground truth for EDGAR filings is generated from the source HTML
(`docrouter/edgar/html_to_gold.py`) and then corrected by hand. Conventions and
open decisions live in [docs/annotation-guide.md](docs/annotation-guide.md).
**Generated gold is a draft, not ground truth** - four separate components were
invalidated by the first real filing.

---

## Design notes

**Failures are data, not exceptions.** `Backend.parse` never raises. A backend
that crashes on scanned PDFs is one the router should avoid for scanned PDFs,
which only works if the crash is recorded.

**Per-pair rows, not aggregates.** `results/*.jsonl` holds one row per
(document, backend). Means hide the distribution, and the distribution is where
routing decisions live.

**Both sides normalized before scoring.** Structural normalization only -
dropping empty columns, merging currency-symbol columns. Cell content is never
touched, so missing rows and wrong numbers stay penalized.

**TEDS excluded from the composite when the gold has no tables.** Otherwise
every prose document hands every backend a free 1.0 and compresses the whole
benchmark toward the mean.

**Reading order measured on sentences, not paragraphs.** Paragraph splitting
depends on blank lines surviving extraction, which they often do not.

**Metrics must be fast enough to run.** The original pure-Python Levenshtein was
correct and quadratic: roughly an hour per 130 KB document. Correct-but-unusable
is a real failure mode for a benchmark harness, so it has performance
regression tests.

---

## Roadmap

- [x] Metrics: TEDS, TEDS-S, text similarity, reading order
- [x] Harness: corpus loader, runner, JSONL output, results table
- [x] Baseline backends + geometric column detection
- [x] EDGAR pipeline: fetcher, HTML-to-gold, PDF render, degradation
- [x] Matched clean/degraded benchmark across three backends
- [ ] Scale to ~30 filings across companies, forms, and filing agents
- [ ] Content-hash cache (a 30-doc sweep is currently ~5 hours)
- [ ] Feature extractor: text-layer presence, page count, table density, columns
- [ ] Router: rules baseline, then learned; report the cost/accuracy Pareto
      curve against always-cheapest and always-best
- [ ] Replace `cost_per_page_usd` placeholders with measured hardware cost
- [ ] LangGraph pipeline: grounding check, escalation, checkpointing
- [ ] FastAPI service, Docker, CI

---

## References

- Ouyang et al., *OmniDocBench* (CVPR 2025) - arXiv:2412.07626
- Zhong et al., *Image-based table recognition* (PubTabNet, TEDS) - arXiv:1911.10683
- Pawlik & Augsten, *APTED* - the tree edit distance implementation used here
- Docling - IBM's document conversion pipeline
