# Findings

Running log of results and of what went wrong producing them. Written as the
work happened rather than reconstructed afterwards.

---

## F1 — Extraction backend choice is nearly irrelevant on clean filings and decisive on scanned ones

**Status:** preliminary, n=1. A hypothesis with a mechanism, not an established
result.

One SEC 10-Q (Apple, filed 2026-07-31, 40 pages) rendered under two conditions
that share a single ground-truth file: a clean PDF rendered from the source
HTML, and a degraded copy rasterized at 150 DPI with skew, blur, contrast loss,
sensor noise, and JPEG artifacts (`scripts/degrade.py`, severity `medium`,
seed 0). Same content, same gold, one variable: whether a text layer exists.

### Clean condition

| backend | text_sim | TEDS | TEDS-S | read_order | overall | sec/doc |
|---|---|---|---|---|---|---|
| docling | 0.753 | 0.490 | 0.550 | 0.784 | **0.675** | 179.17 |
| pylib | 0.716 | 0.398 | 0.436 | 0.861 | **0.658** | 2.31 |
| naive | 0.764 | 0.000 | 0.000 | 0.593 | **0.452** | 1.89 |

### Degraded condition

| backend | text_sim | TEDS | TEDS-S | read_order | overall | sec/doc |
|---|---|---|---|---|---|---|
| docling | 0.740 | 0.482 | 0.547 | **0.654** | 0.654 | 319.49 |
| pylib | 0.000 | 0.000 | 0.000 | 0.000 | **0.000** | 0.02 |
| naive | 0.000 | 0.000 | 0.000 | 0.000 | **0.000** | 0.01 |

### What this says

On the clean filing, Docling beats pure-Python extraction by **0.017 overall
(2.6% relative) for 78× the wall-clock time**. If a corpus were entirely clean
digital filings, that is difficult to justify.

On the degraded filing, pure-Python extraction returns **nothing at all**.
Docling loses only 0.021 (3.1%). The same backend choice is close to worthless
and completely essential depending on one property of the input.

That asymmetry is the case for routing. No fixed backend is correct for both
conditions, and the property that decides it — presence of a text layer — is
detectable in milliseconds, long before the expensive parse begins.

### The composite hides the more interesting result

On the clean filing, Docling and pylib differ by 0.017 overall. The components
do not agree on why:

| metric | pylib | docling | delta |
|---|---|---|---|
| TEDS | 0.398 | 0.490 | **+0.092 (+23.1%)** |
| reading order | 0.861 | 0.784 | **−0.077 (−8.9%)** |
| text similarity | 0.716 | 0.753 | +0.037 |

Docling recovers table structure substantially better and reconstructs reading
order **worse**. Those move in opposite directions and nearly cancel in the
mean. A benchmark reporting only a composite score would have reported "no
meaningful difference" and been wrong twice.

This also reframes the routing question. It is not *which backend is better*
but *which backend for which property of the document*: a table-dense filing
and a prose-heavy one should route differently even within the clean condition.

### Cost

Docling on this hardware (M-series Mac, CPU only):

- clean: 4.48 s/page
- degraded: 7.99 s/page

`cost_per_page_usd` is still 0.0 for every backend. Every cost figure in this
document is wall-clock, not money. **No cost/accuracy curve should be published
until those placeholders are replaced with measured hardware cost.**

### What would falsify or qualify this

- n=1. One company, one form type, one filing agent. Different agents emit
  structurally different HTML.
- One degradation severity. `light` and `heavy` are untested, and the cliff may
  be a gradient.
- Docling on CPU. A GPU changes the cost side of the tradeoff completely.
- The clean condition is a browser-rendered PDF, not a natively-produced one.

---

## F2 — Text similarity alone cannot evaluate a document parser

`naive` — which strips markup and dumps glyphs — scores **0.764 text similarity
on the clean filing, the highest of the three backends**, while scoring **0.000
TEDS**. Every character survives; every table is destroyed.

A benchmark built on text-based metrics alone would rank the strawman first.
This is the concrete argument for TEDS and reading order being separate,
reported columns rather than being folded into one number.

**Open anomaly:** `naive` also beats `pylib` on text similarity (0.764 vs
0.716), which should not happen. `pylib` filters table regions out of the body
text to avoid double-counting, and is likely losing text when table detection
misses. This is an unexplained bug in this repository, not a property of the
strawman.

---

## F3 — The routing signal costs 0.5% of the decision it makes

**Status:** established on this corpus, n=28 per condition.

A router is only worth building if deciding is much cheaper than doing. Feature
extraction (`docrouter/features.py`) samples 8 pages and reads character counts,
ruling lines, image coverage, and word geometry directly from the PDF, without
running any backend.

| condition | documents | classified scanned | ms/doc |
|---|---|---|---|
| clean | 28 | **0** | 905 |
| degraded | 28 | **28** | 76 |

Text-layer presence separates the two conditions **perfectly**: no false
positives, no false negatives, 56/56 correct.

### Why this makes routing viable

| | per document |
|---|---|
| Docling parse | ~180,000 ms |
| Routing decision | 905 ms |
| **Overhead** | **0.5%** |

The decision is ~200x cheaper than the cheapest thing it can decide to do, and
~2,400x cheaper on the degraded arm where the answer arrives in 76 ms because
there is no text layer to sample. Across the whole corpus, feature extraction
costs 27 seconds against roughly 5 hours of Docling.

Note the inversion: classification is *slower* on the documents that need the
*cheaper* backend. Clean PDFs have text to sample; rasterized ones exit almost
immediately. This is the right direction — the expensive branch is identified
fastest.

### The resulting rules baseline

```python
if not features.has_text_layer:
    return "docling"
return "pylib"
```

One condition, no training. Any learned router has to beat this, and on a
corpus with a binary degradation condition it may not. That would itself be a
result worth reporting: the feature that matters was identifiable by inspection,
and the model adds nothing.

### Limitations

- **`est_columns` was 1 for all 28 documents.** Chromium renders SEC HTML
  single-column, so the corpus never exercises multi-column layout. The
  geometric column detection in `pylib` — one of its selling points — is
  therefore validated only on synthetic documents. Any claim about
  multi-column handling is currently unsupported by real data.
- The clean/degraded split is synthetic and binary. Real corpora contain
  partially-degraded documents, mixed scanned and digital pages within one
  file, and PDFs with sparse or wrong text layers. Perfect separation here says
  the feature works on a clean dichotomy, not that it is robust.
- Table-density proxies (`lines_per_page`, `rects_per_page`) are computed but
  not yet validated against anything.

---

## Bug log

Each of these was invisible until real data hit it, and most were hidden by a
synthetic fixture that encoded the same assumption as the code it tested.

| # | Bug | How it hid |
|---|---|---|
| 1 | Synthetic two-column PDF generator wrapped by character count, so columns physically overlapped | The "two-column" test document was never two columns |
| 2 | Reading-order metric split on blank lines, scoring a *correct* parse 0.0 | Backends that emit one line per row became a single block |
| 3 | TEDS returned 1.0 for documents with no tables, inflating every prose document | Free point for every backend, compressing all differences |
| 4 | Metrics were pure-Python and quadratic: ~1 hour per 130 KB document | Correct on two-sentence fixtures, unusable on real filings |
| 5 | lxml rejected inline-XBRL filings carrying an XML declaration | Test HTML had no declaration; every real filing does |
| 6 | Spanned cells repeated their text across the span, tripling table width | Filings use `colspan` for alignment, test HTML did not |
| 7 | Cover pages, checkbox blocks, and tables of contents scored as data tables | Structurally similar to small data tables |
| 8 | Currency-column detection required most values to be `$` | Filings print `$` only on section-leading and total rows |
| 9 | Gold merged `$` into values, pdfplumber did not; TEDS charged 0.4 for the difference | Both conventions defensible; neither is wrong |

### The one worth talking about

Bug 9 is not a coding error. The code did exactly what it was written to do.
The error was in the *measurement design*: predictions and ground truth were
being compared without shared normalization, so the metric was scoring which
formatting convention a backend happened to share with the annotator.

`pylib` TEDS moved **0.211 → 0.398 with no change to any backend, any document,
or any ground truth** — only to how the comparison was performed. A benchmark
that can move a score by 88% through a scoring decision is a benchmark that
needs its scoring decisions documented and tested. Both sides now pass through
`docrouter/metrics/canonical.py`, and six tests pin that structural
normalization does not also hide missing rows or wrong numbers.

### The pattern

Bugs 1, 2, 3, 4, 5, 6, and 8 share a shape: a synthetic fixture written by the
same person, at the same time, with the same mental model as the code it tested.
The fixtures passed because they encoded the assumption rather than challenging
it. Every one surfaced within minutes of real data arriving.

The practical consequence for this project: **annotate one real document before
building anything on top of the pipeline.** The first Apple filing invalidated
four separate components.

---

## Next

1. Scale to ~30 filings across companies, form types, and filing agents. Every
   claim above rests on n=1.
2. Content-hash cache before scaling — a 30-document sweep at Docling's current
   speed is roughly 5 hours, and it will be re-run many times.
3. Feature extractor: text-layer presence, page count, table density, column
   count. Text-layer presence alone perfectly separates the two conditions
   here, so a rules baseline should beat both fixed strategies. Establish that
   number before training anything.
4. Replace `cost_per_page_usd` placeholders with measured values.
5. Record empty output as a distinct outcome. Both cheap backends currently
   report `fail% = 0` while returning empty strings, which is misleading in a
   results table and is a signal a router needs.
