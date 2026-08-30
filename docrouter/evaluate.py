"""The benchmark runner.

Runs every backend over every document, scores each output against ground
truth, and writes one row per (document, backend) pair to JSONL. Everything
downstream (the results table, the Pareto curve, the router's training data)
is a groupby on that file.

Writing per-pair rows rather than pre-aggregated means is deliberate. Means
hide the distribution, and the interesting claims in this project are about the
distribution: which documents does the cheap backend already handle, and what
do the expensive ones actually buy you.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .backends import Backend
from .cache import ParseCache
from .corpus import Corpus, Sample
from .metrics import DocumentScore, score_document


@dataclass
class EvalRun:
    rows: list[dict[str, Any]]

    def write_jsonl(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for row in self.rows:
                fh.write(json.dumps(row) + "\n")
        return path

    def summary(self) -> dict[str, dict[str, float]]:
        """Per-backend means across the corpus."""
        by_backend: dict[str, list[dict[str, Any]]] = {}
        for row in self.rows:
            by_backend.setdefault(row["backend"], []).append(row)

        out: dict[str, dict[str, float]] = {}
        for backend, rows in by_backend.items():
            out[backend] = {
                "n": len(rows),
                "text_sim": _mean(rows, "text_sim"),
                "teds": _mean(rows, "teds"),
                "teds_struct": _mean(rows, "teds_struct"),
                "reading_order": _mean(rows, "reading_order"),
                "overall": _mean(rows, "overall"),
                "wall_time_s": _mean(rows, "wall_time_s"),
                "cost_usd": sum(r["cost_usd"] for r in rows),
                "failure_rate": sum(1 for r in rows if r["failed"]) / len(rows),
            }
        return out

    def format_table(self) -> str:
        """The results table, ready to paste into the README."""
        summary = self.summary()
        if not summary:
            return "(no results)"
        headers = [
            "backend", "n", "text_sim", "TEDS", "TEDS-S",
            "read_order", "overall", "sec/doc", "fail%",
        ]
        rows = []
        for backend, s in sorted(
            summary.items(), key=lambda kv: kv[1]["overall"], reverse=True
        ):
            rows.append([
                backend,
                f"{s['n']:.0f}",
                f"{s['text_sim']:.3f}",
                f"{s['teds']:.3f}",
                f"{s['teds_struct']:.3f}",
                f"{s['reading_order']:.3f}",
                f"{s['overall']:.3f}",
                f"{s['wall_time_s']:.2f}",
                f"{s['failure_rate'] * 100:.0f}%",
            ])
        widths = [
            max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)
        ]
        def fmt(cells: list[str]) -> str:
            return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"
        lines = [fmt(headers), "| " + " | ".join("-" * w for w in widths) + " |"]
        lines.extend(fmt(r) for r in rows)
        return "\n".join(lines)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [r[key] for r in rows]
    return statistics.fmean(values) if values else 0.0


def evaluate(
    corpus: Corpus,
    backends: Iterable[Backend],
    verbose: bool = True,
    cache: ParseCache | None = None,
) -> EvalRun:
    backends = list(backends)
    rows: list[dict[str, Any]] = []

    for sample in corpus:
        for backend in backends:
            if not backend.supports(sample.doc_path):
                continue

            result = None
            doc_id = None
            if cache is not None:
                from .backends.base import doc_id_for

                doc_id = doc_id_for(sample.doc_path)
                result = cache.get(doc_id, backend.name, backend.version)

            cached = result is not None
            if result is None:
                result = backend.parse(sample.doc_path)
                if cache is not None:
                    cache.put(result, backend.name, backend.version)
            score = score_document(
                pred_markdown=result.markdown,
                gold_markdown=sample.gold_markdown,
                doc_id=sample.doc_key,
                backend=backend.name,
                failed=not result.ok,
                wall_time_s=result.wall_time_s,
                cost_usd=result.cost_usd,
            )
            row = score.to_dict()
            row["suffix"] = sample.suffix
            row["attrs"] = sample.attrs
            row["error"] = result.error
            rows.append(row)

            if verbose:
                status = "FAIL" if result.error else f"{score.overall:.3f}"
                mark = " (cached)" if cached else ""
                print(f"  {sample.doc_key:<20} {backend.name:<12} {status}{mark}")

    return EvalRun(rows=rows)
