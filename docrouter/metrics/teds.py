"""TEDS: Tree-Edit-Distance-based Similarity for tables.

Introduced with PubTabNet (Zhong et al., 2019) and used as the table metric in
OmniDocBench. The idea: render both tables as trees, compute the tree edit
distance between them, and normalize by tree size.

    TEDS(a, b) = 1 - EditDistance(a, b) / max(|a|, |b|)

Two variants matter and the benchmark should report both:

- **TEDS** scores structure *and* cell text. Penalizes OCR errors.
- **TEDS-S** (structure-only) ignores cell content. Isolates whether the
  backend recovered the right grid, which is the question a router actually
  cares about: a backend that reads every character correctly but merges two
  columns has failed at the thing markdown conversion exists to do.

A backend can score well on raw text similarity while scoring near zero here,
which is the entire argument for converting rather than scraping.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from apted import APTED, Config


@dataclass
class TableNode:
    tag: str
    colspan: int = 1
    rowspan: int = 1
    content: list[str] | None = None
    children: list["TableNode"] = field(default_factory=list)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)


class _TedsConfig(Config):
    """Edit costs. Insert/delete are unit cost; renames depend on the variant."""

    def __init__(self, structure_only: bool = False) -> None:
        self.structure_only = structure_only

    def delete(self, node: TableNode) -> float:
        return 1.0

    def insert(self, node: TableNode) -> float:
        return 1.0

    def children(self, node: TableNode) -> list[TableNode]:
        return node.children

    def rename(self, a: TableNode, b: TableNode) -> float:
        # Grid position is non-negotiable: a cell spanning a different number of
        # rows or columns is a different cell, whatever text is in it.
        if (a.tag, a.colspan, a.rowspan) != (b.tag, b.colspan, b.rowspan):
            return 1.0
        if self.structure_only:
            return 0.0
        if a.content is None and b.content is None:
            return 0.0
        ca = a.content or []
        cb = b.content or []
        if not ca and not cb:
            return 0.0
        return _normalized_edit_distance(ca, cb)


def _normalized_edit_distance(a: list[str], b: list[str]) -> float:
    """Levenshtein over token lists, scaled to [0, 1]."""
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, ta in enumerate(a, 1):
        curr = [i]
        for j, tb in enumerate(b, 1):
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + (ta != tb),
                )
            )
        prev = curr
    return prev[-1] / max(len(a), len(b))


def rows_to_tree(rows: list[list[str]]) -> TableNode:
    """Build a TEDS tree from a plain row matrix.

    Markdown pipe tables cannot express spans, so everything is 1x1. Real
    span handling arrives with the HTML path below, which is what OmniDocBench
    ground truth uses.
    """
    root = TableNode("table")
    body = TableNode("tbody")
    root.children.append(body)
    for row in rows:
        tr = TableNode("tr")
        for cell in row:
            tr.children.append(TableNode("td", content=_tokenize(cell)))
        body.children.append(tr)
    return root


def html_to_tree(html: str) -> TableNode:
    """Build a TEDS tree from an HTML table, honoring colspan/rowspan."""
    from lxml import html as lxml_html

    fragment = lxml_html.fragment_fromstring(html, create_parent="div")
    table_el = fragment if fragment.tag == "table" else fragment.find(".//table")
    if table_el is None:
        return TableNode("table")

    def build(el) -> TableNode:
        tag = str(el.tag)
        node = TableNode(
            tag=tag,
            colspan=int(el.get("colspan", 1) or 1),
            rowspan=int(el.get("rowspan", 1) or 1),
        )
        if tag in {"td", "th"}:
            node.tag = "td"  # th and td are the same slot for structure purposes
            node.content = _tokenize(" ".join(el.itertext()))
        else:
            for child in el:
                if str(child.tag) in {"thead", "tbody", "tr", "td", "th"}:
                    node.children.append(build(child))
        return node

    return build(table_el)


_MD_ROW = re.compile(r"^\s*\|.*\|\s*$")
_MD_SEP = re.compile(r"^\s*\|[\s:|-]+\|\s*$")


def extract_markdown_tables(markdown: str) -> list[list[list[str]]]:
    """Pull every GFM pipe table out of a markdown document as row matrices."""
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in markdown.splitlines():
        if _MD_ROW.match(line):
            if _MD_SEP.match(line):
                continue  # the |---|---| alignment row is not data
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            current.append(cells)
        else:
            if len(current) >= 2:
                tables.append(current)
            current = []
    if len(current) >= 2:
        tables.append(current)
    return tables


def _tokenize(text: str) -> list[str]:
    return list(" ".join(text.split()))


def teds(
    pred: TableNode,
    gold: TableNode,
    structure_only: bool = False,
) -> float:
    """Similarity in [0, 1]. 1.0 is an exact match."""
    n = max(pred.size(), gold.size())
    if n == 0:
        return 1.0
    distance = APTED(pred, gold, _TedsConfig(structure_only)).compute_edit_distance()
    return max(0.0, 1.0 - distance / n)


def _flatten(rows: list[list[str]]) -> str:
    """Cheap text signature of a table, for candidate matching."""
    return " ".join(" ".join(r) for r in rows).lower()


def _match_tables(
    pred_tables: list[list[list[str]]],
    gold_tables: list[list[list[str]]],
) -> list[int | None]:
    """Greedily pair predicted tables to gold tables by text similarity.

    Matching directly on TEDS would need |pred| x |gold| tree edit distances.
    On a real filing with ~20 tables that is 400 APTED runs and takes minutes.
    Instead, pair on a cheap text proxy and spend exact TEDS only on the pairs
    that actually matched: O(n) tree comparisons instead of O(n^2).

    The proxy only decides *which* prediction corresponds to which ground-truth
    table. The score itself is still exact TEDS, so this is a search
    optimization, not an approximation of the metric.
    """
    if not pred_tables:
        return [None] * len(gold_tables)

    pred_sigs = [_flatten(t) for t in pred_tables]
    gold_sigs = [_flatten(t) for t in gold_tables]

    try:
        from rapidfuzz import process as _process
        from rapidfuzz.distance import Levenshtein as _L

        matrix = _process.cdist(
            gold_sigs, pred_sigs, scorer=_L.normalized_similarity, workers=-1
        )
        sim = [[float(matrix[g][p]) for p in range(len(pred_sigs))]
               for g in range(len(gold_sigs))]
    except ImportError:  # pragma: no cover
        from .text import normalized_edit_distance

        sim = [
            [1.0 - normalized_edit_distance(g, p) for p in pred_sigs]
            for g in gold_sigs
        ]

    # Global greedy: take the best available pair overall, then the next best.
    # Row-by-row greedy would let an early gold table claim a prediction that
    # matches a later one far better.
    pairs = sorted(
        ((sim[g][p], g, p) for g in range(len(gold_sigs))
         for p in range(len(pred_sigs))),
        reverse=True,
    )
    assignment: list[int | None] = [None] * len(gold_sigs)
    used_pred: set[int] = set()
    for _, g, p in pairs:
        if assignment[g] is None and p not in used_pred:
            assignment[g] = p
            used_pred.add(p)
    return assignment


def teds_from_markdown(
    pred_md: str,
    gold_md: str,
    structure_only: bool = False,
) -> float:
    """Document-level TEDS: mean over matched table pairs.

    Missing tables score 0 and spurious tables drag the mean down, so a backend
    cannot win by simply not emitting tables it is unsure about.
    """
    pred_tables = extract_markdown_tables(pred_md)
    gold_tables = extract_markdown_tables(gold_md)
    if not gold_tables:
        return 1.0 if not pred_tables else 0.0

    assignment = _match_tables(pred_tables, gold_tables)

    scores: list[float] = []
    for gi, gold_rows in enumerate(gold_tables):
        pi = assignment[gi]
        if pi is None:
            scores.append(0.0)
            continue
        scores.append(
            teds(rows_to_tree(pred_tables[pi]), rows_to_tree(gold_rows), structure_only)
        )

    # Spurious extra tables are penalized as zero-scoring predictions.
    n_unmatched = len(pred_tables) - sum(1 for a in assignment if a is not None)
    scores.extend([0.0] * n_unmatched)
    return sum(scores) / len(scores)


def teds_both_from_markdown(pred_md: str, gold_md: str) -> tuple[float, float]:
    """Return (TEDS, TEDS-S) sharing one extraction and one table matching.

    Calling `teds_from_markdown` twice re-extracts and re-matches every table
    for a result that is identical either way. On a 20-table filing that is
    half the metric's total cost thrown away.
    """
    pred_tables = extract_markdown_tables(pred_md)
    gold_tables = extract_markdown_tables(gold_md)
    if not gold_tables:
        v = 1.0 if not pred_tables else 0.0
        return v, v

    assignment = _match_tables(pred_tables, gold_tables)

    full: list[float] = []
    struct: list[float] = []
    for gi, gold_rows in enumerate(gold_tables):
        pi = assignment[gi]
        if pi is None:
            full.append(0.0)
            struct.append(0.0)
            continue
        # Build each tree once and score it under both cost functions.
        pred_tree = rows_to_tree(pred_tables[pi])
        gold_tree = rows_to_tree(gold_rows)
        full.append(teds(pred_tree, gold_tree, structure_only=False))
        struct.append(teds(pred_tree, gold_tree, structure_only=True))

    n_unmatched = len(pred_tables) - sum(1 for a in assignment if a is not None)
    full.extend([0.0] * n_unmatched)
    struct.extend([0.0] * n_unmatched)
    return sum(full) / len(full), sum(struct) / len(struct)
