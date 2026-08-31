# docrouter

Cost-aware routing for document extraction, with a reproducible benchmark
underneath it.

Most ingestion pipelines pick one PDF parser and use it for everything. That is
either wasteful or wrong: a neural document pipeline is 150x slower than
pure-Python parsing and buys almost nothing on a clean digital filing, while
pure-Python parsing returns **literally nothing** on a scanned page.

**docrouter** measures where each backend actually wins, then routes each
document to the cheapest one that clears a quality bar.

---

## Result

28 SEC filings, 3,433 pages, in matched clean and degraded conditions sharing
one ground-truth file per document. Same content, one variable: whether a text
layer exists.

| backend | clean | degraded | sec/doc |
|---|---|---|---|
| docling | **0.675** (n=3) | **0.586** (n=14) | 659 / 2899 |
| pylib | **0.571** (n=28) | **0.000** (n=28) | 12.5 |
| naive | **0.414** (n=28) | **0.000** (n=28) | 13.4 |

**Pure-Python extraction scores exactly 0.000 on all 56 scanned
document-backend pairs.** Not low: zero, without exception, across 10
companies. On clean filings the same backend comes within 0.008 of the neural
pipeline at 1/153rd the compute.

### The router

```python
if not features.has_text_layer:
    return "docling"
return "pylib"
```

| policy | quality | compute |
|---|---|---|
| always(docling) | 0.665 | 2,744 s |
| **rules router** | **0.661** | **782 s** |
| always(pylib) | 0.333 | 14 s |

**99.4% of always-Docling quality at 28.5% of its compute.** An oracle that
picks the best backend per document with hindsight scores only **+0.001** above
the rules router — so on this corpus, no possible router does meaningfully
better, and a learned classifier would have nothing left to learn.

The routing decision itself costs **0.01%** of the parse it replaces.

Full results, per-metric breakdowns, sample-size caveats, and the bug log are
in [docs/findings.md](docs/findings.md).

---

## Why this was hard

Twelve documented measurement bugs, each caught by a specific diagnostic and
pinned by a regression test. Most were invisible until real data arrived,
because the synthetic fixtures encoded the same assumptions as the code they
tested.

The one worth reading: TEDS moved **0.211 → 0.398 with no change to any
backend, any document, or any ground truth** — only to how predictions and gold
were normalized before comparison. The gold generator merged currency symbols
into values; pdfplumber did not. The metric was scoring which formatting
convention a backend happened to share with the annotator.

A benchmark whose score can move 88% through a scoring decision is one whose
scoring decisions need documenting and testing.

---

## Quickstart

```bash
pip install -r requirements.txt
python scripts/make_samples.py
python -m docrouter.cli bench data/sample_corpus
python -m pytest -q                          # 116 tests
```

### Reproducing the EDGAR corpus

Document binaries are not tracked; regenerate them from tracked provenance.

```bash
export EDGAR_UA="Your Name your@email.com"   # SEC requires this
python scripts/fetch_edgar.py --tickers AAPL MSFT JPM --forms 10-Q --limit 1
python scripts/build_corpus.py --degrade medium

python -m docrouter.cli features data/edgar/clean --out results/features_clean.jsonl
python -m docrouter.cli bench data/edgar/clean --out results/clean.jsonl
python scripts/finalize.py
```

Ground truth (`data/*/gold/`), provenance (`sources.jsonl`), and results
(`results/*.jsonl`) **are** tracked — those are annotation and evidence.

Docling is slow on CPU (~4.5 s/page clean, ~8 s/page scanned). The parse cache
makes re-scoring after a metric change instant; only first parses are expensive.

---

## Metrics

| metric | question | blind to |
|---|---|---|
| `text_sim` | Did the characters survive? | Everything structural |
| `TEDS` / `TEDS-S` | Did the table grid survive? | Prose ordering |
| `reading_order` | Did the sequence survive? | Content accuracy |

The clearest case for keeping them separate: on clean filings `naive` scores
the **highest text similarity of any backend (0.687)** and **0.000 TEDS**.
Every character present, every table destroyed. A text-only benchmark would
rank the strawman first.

- **TEDS** (Zhong et al., 2019; the table metric in OmniDocBench) renders both
  tables as trees and computes normalized tree edit distance via APTED.
  `TEDS-S` ignores cell text to separate grid recovery from OCR accuracy.
- **reading_order** matches sentences between prediction and gold, scores the
  index sequence with Kendall tau scaled to [0,1], multiplied by coverage.

Predictions and ground truth pass through identical structural normalization
(`docrouter/metrics/canonical.py`) before comparison, so no backend is rewarded
for sharing a formatting convention with the annotator.

---

## Architecture

```
docrouter/
  backends/     naive, pylib (pdfplumber + geometric column detection),
                docling, vlm (implemented, not benchmarked)
  metrics/      teds, text, canonical
  features.py   cheap pre-parse signals: text layer, columns, table density
  router.py     policies, oracle baseline, policy comparison
  cache.py      content-addressed parse cache
  evaluate.py   benchmark runner, streams results to disk
  layout.py     gutter detection, reading-order reconstruction
```

Adding a backend is a subclass with three methods; register it and it joins
every benchmark run. Failures are captured as `ParseResult.error` rather than
raised — which documents a backend fails on is routing signal.

---

## Design notes

**Failures are data, not exceptions.** A backend that crashes on scanned PDFs
is one the router should avoid for scanned PDFs, which only works if the crash
is recorded.

**Per-pair rows, not aggregates.** One row per (document, backend). Means hide
the distribution, and the distribution is where routing decisions live.

**Cache the parse, not the score.** Metrics changed a dozen times during
development; not one of those changes affected a parse. Re-scoring a cached
corpus takes seconds instead of hours.

**Policies never see quality scores.** A router runs before parsing, so a policy
that could read scores would be leaking. There is a test asserting this.

**Stream results to disk.** A five-hour sweep that dies on document 8 must not
lose the first seven.

---

## Roadmap

**Done**

- [x] Metrics: TEDS, TEDS-S, text similarity, reading order, shared canonicalization
- [x] Harness: corpus loader, streaming runner, JSONL output, parse cache
- [x] Backends: naive, pylib with column detection and table stitching, docling
- [x] EDGAR pipeline: fetcher, HTML-to-gold, PDF render, degradation
- [x] 28-filing matched corpus, 3,433 pages
- [x] Feature extractor with perfect condition separation
- [x] Rules router, oracle baseline, policy comparison
- [x] CI on 3.11/3.12 with a benchmark discrimination check

**Next**

- [ ] Complete the Docling sweeps (needs a GPU or more patience)
- [ ] Diagnose the naive-beats-pylib text similarity anomaly
- [ ] Replace `cost_per_page_usd` placeholders with measured hardware cost
- [ ] Downstream QA evaluation: does better extraction improve answer accuracy?
- [ ] VLM backend benchmarked (implemented, unregistered to avoid API cost)

---

## References

- Ouyang et al., *OmniDocBench* (CVPR 2025) — arXiv:2412.07626
- Zhong et al., *Image-based table recognition* (PubTabNet, TEDS) — arXiv:1911.10683
- Pawlik & Augsten, *APTED* — the tree edit distance implementation used here
- Docling — IBM's document conversion pipeline
