"""Scoring surface for the benchmark."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .teds import (
    teds_from_markdown,
    teds_both_from_markdown,
    extract_markdown_tables,
)
from .text import text_similarity, reading_order_score, normalized_edit_distance

__all__ = [
    "DocumentScore",
    "score_document",
    "teds_from_markdown",
    "teds_both_from_markdown",
    "text_similarity",
    "reading_order_score",
    "normalized_edit_distance",
    "extract_markdown_tables",
]


@dataclass
class DocumentScore:
    doc_id: str
    backend: str
    text_sim: float
    teds: float
    teds_struct: float
    reading_order: float
    n_tables_pred: int
    n_tables_gold: int
    has_gold_tables: bool = False
    wall_time_s: float = 0.0
    cost_usd: float = 0.0
    failed: bool = False

    @property
    def overall(self) -> float:
        """Equal-weight composite over the metrics that apply to this document.

        TEDS is excluded when the ground truth contains no tables. Otherwise
        every table-free document hands each backend a free 1.0 and drags the
        whole benchmark toward the mean, hiding exactly the differences the
        benchmark exists to show. Excluding it means the composite is averaged
        over a different number of terms per document, so the per-metric columns
        are the ones to reason about; the composite is a summary, not evidence.

        Weighting is a judgment call, not a fact. Say in the writeup why you
        weighted them the way you did.
        """
        if self.failed:
            return 0.0
        parts = [self.text_sim, self.reading_order]
        if self.has_gold_tables:
            parts.append(self.teds)
        return sum(parts) / len(parts)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["overall"] = self.overall
        return d


def score_document(
    pred_markdown: str,
    gold_markdown: str,
    doc_id: str,
    backend: str,
    failed: bool = False,
    wall_time_s: float = 0.0,
    cost_usd: float = 0.0,
) -> DocumentScore:
    """Score one backend's output against ground truth."""
    gold_tables = extract_markdown_tables(gold_markdown)

    if failed:
        return DocumentScore(
            doc_id=doc_id,
            backend=backend,
            text_sim=0.0,
            teds=0.0,
            teds_struct=0.0,
            reading_order=0.0,
            n_tables_pred=0,
            n_tables_gold=len(gold_tables),
            has_gold_tables=bool(gold_tables),
            wall_time_s=wall_time_s,
            cost_usd=cost_usd,
            failed=True,
        )

    teds_full, teds_s = teds_both_from_markdown(pred_markdown, gold_markdown)

    return DocumentScore(
        doc_id=doc_id,
        backend=backend,
        text_sim=text_similarity(pred_markdown, gold_markdown),
        teds=teds_full,
        teds_struct=teds_s,
        reading_order=reading_order_score(pred_markdown, gold_markdown),
        n_tables_pred=len(extract_markdown_tables(pred_markdown)),
        n_tables_gold=len(gold_tables),
        has_gold_tables=bool(gold_tables),
        wall_time_s=wall_time_s,
        cost_usd=cost_usd,
    )
