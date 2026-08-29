# Annotation guide

Conventions for ground-truth markdown in `data/edgar/*/gold/`. Every decision
here is a choice, and the choices must be consistent across the corpus or the
benchmark measures annotation noise instead of backend quality.

Generated gold from `html_to_gold.py` is a **draft**. This file records what to
check and how to resolve what the generator cannot.

## Open decisions (resolve these on document 1, then never revisit)

- [ ] **Currency symbol columns.** Filings put `$` in its own column. Keep as a
      separate column, or merge into the adjacent number? Merging is more
      readable; keeping is closer to what a parser sees. Pick one.
- [ ] **Parenthesized negatives.** `(617)` means -617. Preserve the source
      notation verbatim (recommended: you are scoring extraction, not
      interpretation).
- [ ] **Merged header cells.** Markdown cannot express spans, so the generator
      repeats the value across the spanned columns. Confirm this reads sensibly.
- [ ] **Tables split across pages.** In HTML they are one table; in the rendered
      PDF they break. Gold should reflect the HTML (one table), which means
      backends that fail to merge across a page break will be penalized. This is
      intentional but must be stated in the writeup.
- [ ] **Footnote markers.** `(1)`, `*`, superscripts. Keep inline or strip?
- [ ] **Empty spacer columns.** Dropped by default. Verify this does not remove
      a column that carries meaning.

## Fixed conventions

- Tables are GFM pipe tables. First row is the header.
- Headings use `#` levels matching document hierarchy. `Item 2.` style section
  headers are `##`.
- Paragraphs separated by blank lines. No manual line wrapping.
- Cell text is whitespace-normalized; `\xa0` becomes a regular space.
- Literal `|` inside a cell is escaped as `\|`.

## Review checklist per document

1. Open the generated `.md` next to the rendered `clean/docs/*.pdf`.
2. Count tables. Does the gold have every data table, and no layout tables?
3. Check the largest financial table cell by cell against the PDF.
4. Confirm reading order matches the visual document.
5. Note anything the generator got wrong in the log below.

## Correction log

Record generator failures here. Patterns that recur are bugs to fix in
`html_to_gold.py`; one-offs are hand corrections.

| Document | Issue | Resolution |
|---|---|---|
| | | |
