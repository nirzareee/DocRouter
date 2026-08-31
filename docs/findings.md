# Findings

Results and the measurement bugs found producing them. Written as the work
happened rather than reconstructed afterwards.

**Corpus:** 28 SEC filings (10-K and 10-Q) from 10 companies, 3,433 pages, in
matched clean and degraded conditions sharing a single ground-truth file per
document. The degraded condition rasterizes each page at 150 DPI with skew,
blur, contrast loss, sensor noise, and JPEG artifacts (`scripts/degrade.py`,
severity `medium`, seed 0). Same content, same gold, one variable: whether a
text layer exists.

**Sample sizes differ by backend.** The pure-Python backends ran over all 28
documents in both conditions. The Docling sweeps were truncated by compute
limits (see [Limitations](#limitations)). Every claim below is reported at its
own n.

---

## Headline

| backend | condition | n | text_sim | TEDS | read_order | overall | sec/doc |
|---|---|---|---|---|---|---|---|
| naive | clean | 28 | 0.687 | 0.000 | 0.555 | **0.414** | 13.4 |
| naive | degraded | 28 | 0.000 | 0.000 | 0.000 | **0.000** | 0.1 |
| pylib | clean | 28 | 0.653 | 0.356 | 0.703 | **0.571** | 12.5 |
| pylib | degraded | 28 | 0.000 | 0.000 | 0.000 | **0.000** | 0.1 |
| docling | clean | 3 | 0.748 | 0.515 | 0.762 | **0.675** | 658.5 |
| docling | degraded | 14 | 0.633 | 0.475 | 0.649 | **0.586** | 2898.9 |

---

## F1 - Pure-Python extraction on scanned documents is categorically zero

**n = 56 (document, backend) pairs. Fully powered.**

Across all 28 degraded filings and both pure-Python backends:

- pairs scoring above 0.000: **0**
- maximum score observed: **0.0000**

Not "low." Zero, without exception, across 10 companies and two filing form
types. Both backends read the PDF text layer; a rasterized page has none, so
there is nothing to read. Docling scores 0.586 on the same documents.

This is the strongest claim in the project because it is categorical rather
than statistical. No mean, no variance, no sample-size argument: the cheap tier
does not degrade on scanned documents, it fails completely.

---

## F2 - On clean filings, the neural pipeline buys almost nothing

**n = 3 documents. Preliminary.**

| pylib | docling | gain | slowdown | document |
|---|---|---|---|---|
| 0.681 | 0.694 | +0.013 | 178x | AAPL 10-Q 2026-01-30 |
| 0.656 | 0.665 | +0.010 | 111x | AAPL 10-Q 2026-07-31 |
| 0.664 | 0.665 | +0.002 | 169x | AAPL 10-Q 2026-05-01 |

Mean gain **+0.008** for a mean **153x** slowdown.

Taken with F1, this is the asymmetry that makes routing worthwhile. The same
backend choice is nearly irrelevant on clean documents and decisive on scanned
ones - and the property that separates them is detectable before any parsing
begins.

The n=3 caveat is real. Three filings from one company is not a basis for a
general claim about clean documents. What it does establish is that the gain is
*small* on at least some clean filings, which is enough to motivate routing.

---

## F3 - The routing signal costs 0.01% of the decision it makes

**n = 56 documents (28 per condition). Fully powered.**

Feature extraction samples 8 pages and reads character counts, ruling lines,
image coverage, and word geometry directly from the PDF, without running any
backend.

| condition | documents | classified scanned | ms/doc |
|---|---|---|---|
| clean | 28 | **0** | 905 |
| degraded | 28 | **28** | 76 |

Text-layer presence separates the conditions **perfectly**: 56/56 correct, no
false positives or negatives.

Against Docling's ~2,899 s/doc on degraded pages, a 0.9 s decision is 0.03%
overhead; measured across the policy comparison it came to **0.01%**. The
decision is three to four orders of magnitude cheaper than the cheapest thing
it can decide to do.

Note the inversion: classification is *slower* on documents that need the
*cheaper* backend, because clean PDFs have text to sample and rasterized ones
exit immediately. The expensive branch is identified fastest.

---

## F4 - A one-line router is optimal on this corpus

**n = 17 document-conditions with all three backends scored.**

```python
if not features.has_text_layer:
    return "docling"
return "pylib"
```

| policy | n | quality | total_s | s/doc | routing_s |
|---|---|---|---|---|---|
| always(docling) | 17 | 0.601 | 42,560 | 2,503.5 | 2.73 |
| **oracle** | 17 | **0.601** | 42,560 | 2,503.5 | 0.00 |
| **rules(text_layer)** | 17 | **0.600** | 40,598 | 2,388.1 | 2.73 |
| always(pylib) | 17 | 0.118 | 15 | 0.9 | 2.73 |
| always(naive) | 17 | 0.078 | 52 | 3.0 | 2.73 |

The rules router matches always-Docling within **0.001** quality. The oracle -
which picks the best backend per document with hindsight, and is therefore an
upper bound no router can exceed - sits **+0.001** above it.

**No possible router does meaningfully better on this corpus.** A learned
classifier would have nothing left to learn: the signal was identifiable by
inspection, and one condition captures it.

That is a result, not a shortcut. It is worth more than a trained model that
matches the same number, because it says something about the problem rather
than about the model.

### Compute saved, measured on matched pairs

On the 3 documents with both conditions fully scored:

| policy | quality | compute |
|---|---|---|
| always(docling) | 0.665 | 2,744 s |
| **rules router** | **0.661** | **782 s** |
| always(pylib) | 0.333 | 14 s |

**The router keeps 99.4% of always-Docling quality at 28.5% of its compute.**

The 5% saving in the 17-document table is lower only because 14 of those 17 are
degraded, where the router correctly chooses Docling anyway. Savings scale with
the clean fraction of a corpus.

---

## F5 - Text similarity alone cannot evaluate a document parser

On clean filings, `naive` - which strips markup and dumps glyphs - scores the
**highest text similarity of any backend (0.687)** and **0.000 TEDS**. Every
character present, every table destroyed.

A benchmark using only text-based metrics would rank the strawman first. This
is the concrete argument for reporting TEDS and reading order as separate
columns rather than folding everything into one score.

The same lesson appeared again in the Docling comparison: on the AAPL filings
Docling won TEDS by 23% relative while *losing* reading order by 9%, and the
two nearly cancelled in the composite. A single number would have reported "no
meaningful difference" and been wrong twice.

---

## Limitations

- **Docling sample sizes are small** (n=3 clean, n=14 degraded). Full sweeps
  over 3,433 pages required ~12 hours of CPU-only inference on a MacBook Air
  and were truncated: one run segfaulted after twelve hours, likely memory
  exhaustion in the OCR workers. F2 and F4 are preliminary; F1 and F3 are not.
- **`est_columns` was 1 for all 28 documents.** Chromium renders SEC HTML
  single-column, so the corpus never exercises multi-column layout. The
  geometric column detection in `pylib` is therefore validated only on
  synthetic documents.
- **The degradation is synthetic and binary.** Real corpora contain
  partially-degraded documents, mixed scanned and digital pages within one
  file, and PDFs with sparse or wrong text layers. Perfect separation here
  shows the feature works on a clean dichotomy, not that it is robust.
- **Ground truth is generated from filing HTML and only partially hand-checked.**
  The generator was corrected four times against real filings; residual errors
  are likely, especially on filing agents outside the sampled ten companies.
- **`cost_per_page_usd` is 0.0 for every backend.** All cost figures are
  wall-clock, not money.
- **Unexplained:** `naive` beats `pylib` on text similarity (0.687 vs 0.653)
  across all 28 clean filings. A glyph dump should not retain more text than
  the structure-aware backend. Likely `pylib`'s table-region filtering removes
  body text it then fails to re-emit. Reproducible, not diagnosed.

---

## Bug log

Each of these was invisible until real data hit it. Most were hidden by a
synthetic fixture that encoded the same assumption as the code it tested.

| # | Bug | How it hid |
|---|---|---|
| 1 | Synthetic two-column PDF wrapped by character count, so columns physically overlapped | The "two-column" test document was never two columns |
| 2 | Reading-order metric split on blank lines, scoring a *correct* parse 0.0 | Backends emitting one line per row became a single block |
| 3 | TEDS returned 1.0 for table-free documents, inflating every prose document | Free point for every backend, compressing all differences |
| 4 | Metrics were pure-Python and quadratic: ~1 hour per 130 KB document | Correct on two-sentence fixtures, unusable on real filings |
| 5 | lxml rejected inline-XBRL filings carrying an XML declaration | Test HTML had none; every real filing has one |
| 6 | Spanned cells repeated text across the span, tripling table width | Filings use `colspan` for alignment; test HTML did not |
| 7 | Cover pages, checkbox blocks, and tables of contents scored as data tables | Structurally similar to small data tables |
| 8 | Currency-column detection required most values to be `$` | Filings print `$` only on section-leading and total rows |
| 9 | Gold merged `$` into values, pdfplumber did not; TEDS charged 0.19 for the difference | Both conventions defensible; neither is wrong |
| 10 | pdfplumber returned one table per ruled row; 235 single-row fragments all discarded | Scored as a "hard document" (TEDS 0.011) rather than an unassembled one |
| 11 | Reading-order optimization narrowed candidates by *position* | 33x faster and wrong: a backend dropping half a document scored 0.030 instead of 0.500 |
| 12 | Benchmark rows held in memory until the run ended | A 12-hour sweep crashed at document 8 and wrote nothing |

### The two worth talking about

**Bug 9 was not a coding error.** The code did exactly what it was written to
do. The error was in the *measurement design*: predictions and ground truth
were compared without shared normalization, so the metric scored which
formatting convention a backend happened to share with the annotator. TEDS
moved **0.211 -> 0.398 with no change to any backend, any document, or any
ground truth** - only to how the comparison was performed.

A benchmark whose score can move 88% through a scoring decision is one whose
scoring decisions need documenting and testing. Both sides now pass through
`docrouter/metrics/canonical.py`, with six tests pinning that structural
normalization does not also hide missing rows or wrong numbers.

**Bug 11 is the same failure in a different costume.** A positional-window
optimization made reading-order scoring 33x faster and passed every existing
test. It was wrong precisely where it mattered: a backend that dropped half a
document had every surviving block displaced past the window and scored 0.030
instead of 0.500. The fast version was measuring something other than what it
claimed. Content-based blocking replaced it, and ten tests now cover that class
of failure.

### The pattern

Bugs 1-8 and 11 share a shape: a synthetic fixture written by the same person,
at the same time, with the same mental model as the code it tested. They passed
because they encoded the assumption rather than challenging it. Every one
surfaced within minutes of real data arriving.

The practical consequence: **annotate one real document before building
anything on top of the pipeline.** The first Apple filing invalidated four
separate components.

Bug 12 is a different species - operational, not measurement. A five-hour job
was written as if completion were guaranteed. For a run that long, a crash is a
normal outcome, and results that exist only in memory are results you can lose.
