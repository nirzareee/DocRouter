"""Routing policies and their evaluation.

A policy maps document features to a backend name. Evaluation replays measured
benchmark results: for each document, look up what the chosen backend actually
scored and what it actually cost. Nothing is re-parsed, so comparing twenty
policies costs nothing beyond reading two JSONL files.

That separation matters. Parsing is expensive and deterministic; policy design
is cheap and iterative. Coupling them would mean a five-hour wait to test a
one-line rule change.

Four reference policies frame every result:

- `always(cheapest)` -- the lower bound on cost and, usually, on quality
- `always(best)` -- the upper bound on cost; the quality target to match
- `oracle` -- picks the best backend per document with hindsight. Not
  achievable, but it bounds what any router could possibly do, and the gap
  between oracle and always-best says whether routing has anything to exploit
  at all
- the rules router -- what you actually ship

A router that matches always-best quality at less than always-best cost is the
result the project exists to produce. A router that cannot beat always-best is
also a result, and reporting it honestly is worth more than tuning until the
number flatters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

# A policy sees only features -- never the gold, never the scores. Anything
# else would be leakage: the router runs before parsing, when quality is
# unknown by definition.
Policy = Callable[[dict[str, Any]], str]


@dataclass
class PolicyResult:
    name: str
    n_documents: int
    mean_quality: float
    total_seconds: float
    total_cost_usd: float
    routing_overhead_s: float
    choices: dict[str, int]

    @property
    def seconds_per_doc(self) -> float:
        return self.total_seconds / max(self.n_documents, 1)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "policy": self.name,
            "n": self.n_documents,
            "mean_quality": self.mean_quality,
            "total_seconds": self.total_seconds,
            "seconds_per_doc": self.seconds_per_doc,
            "total_cost_usd": self.total_cost_usd,
            "routing_overhead_s": self.routing_overhead_s,
        }
        d.update({f"chose_{k}": v for k, v in sorted(self.choices.items())})
        return d


def load_results(*paths: str | Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Index benchmark rows by (doc_id, backend)."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                out[(row["doc_id"], row["backend"])] = row
    return out


def load_features(*paths: str | Path) -> dict[str, dict[str, Any]]:
    """Index feature rows by doc_key."""
    out: dict[str, dict[str, Any]] = {}
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                out[row["doc_key"]] = row
    return out


# --- Policies ---------------------------------------------------------------

def always(backend: str) -> Policy:
    """Send everything to one backend. The fixed-strategy baselines."""
    def policy(features: dict[str, Any]) -> str:
        return backend
    policy.__name__ = f"always_{backend}"
    return policy


def text_layer_rule(
    with_text: str = "pylib", without_text: str = "docling"
) -> Policy:
    """Route on text-layer presence alone.

    On this corpus that single feature separates the clean and degraded
    conditions perfectly, and the cheap backends score exactly 0.000 wherever
    it is False. One condition, no training, no tuning.

    A learned router has to beat this. On a corpus whose degradation is binary
    it may well not -- which would itself be worth reporting, since it says the
    signal was identifiable by inspection.
    """
    def policy(features: dict[str, Any]) -> str:
        return with_text if features.get("has_text_layer") else without_text
    policy.__name__ = "text_layer_rule"
    return policy


# --- Evaluation -------------------------------------------------------------

def evaluate_policy(
    policy: Policy,
    features: dict[str, dict[str, Any]],
    results: dict[tuple[str, str], dict[str, Any]],
    name: str | None = None,
    quality_key: str = "overall",
) -> PolicyResult:
    """Replay a policy over measured results."""
    qualities: list[float] = []
    seconds = cost = overhead = 0.0
    choices: dict[str, int] = {}

    for doc_key, feat in sorted(features.items()):
        backend = policy(feat)
        row = results.get((doc_key, backend))
        if row is None:
            # Silently skipping would make a policy look good by evaluating it
            # on fewer documents than its competitors.
            raise KeyError(
                f"No benchmark result for ({doc_key}, {backend}). "
                f"Run the benchmark with that backend before evaluating this policy."
            )
        qualities.append(row[quality_key])
        seconds += row["wall_time_s"]
        cost += row["cost_usd"]
        overhead += feat.get("feature_time_ms", 0.0) / 1000.0
        choices[backend] = choices.get(backend, 0) + 1

    return PolicyResult(
        name=name or getattr(policy, "__name__", "policy"),
        n_documents=len(qualities),
        mean_quality=sum(qualities) / max(len(qualities), 1),
        total_seconds=seconds,
        total_cost_usd=cost,
        routing_overhead_s=overhead,
        choices=choices,
    )


def evaluate_oracle(
    features: dict[str, dict[str, Any]],
    results: dict[tuple[str, str], dict[str, Any]],
    backends: Iterable[str],
    quality_key: str = "overall",
) -> PolicyResult:
    """Upper bound: best backend per document, chosen with hindsight.

    Not achievable -- it reads the scores a router cannot see. Its value is
    diagnostic: if the oracle barely beats always-best, there is nothing for a
    router to exploit and the honest conclusion is to skip routing entirely.
    """
    backends = list(backends)
    qualities: list[float] = []
    seconds = cost = 0.0
    choices: dict[str, int] = {}

    for doc_key in sorted(features):
        rows = [
            results[(doc_key, b)] for b in backends
            if (doc_key, b) in results
        ]
        if not rows:
            continue
        best = max(rows, key=lambda r: r[quality_key])
        qualities.append(best[quality_key])
        seconds += best["wall_time_s"]
        cost += best["cost_usd"]
        choices[best["backend"]] = choices.get(best["backend"], 0) + 1

    return PolicyResult(
        name="oracle",
        n_documents=len(qualities),
        mean_quality=sum(qualities) / max(len(qualities), 1),
        total_seconds=seconds,
        total_cost_usd=cost,
        routing_overhead_s=0.0,
        choices=choices,
    )


def format_comparison(results: list[PolicyResult]) -> str:
    """Render the policy comparison table."""
    headers = ["policy", "n", "quality", "total_s", "s/doc", "cost_$", "routing_s"]
    rows = []
    for r in sorted(results, key=lambda x: x.mean_quality, reverse=True):
        rows.append([
            r.name,
            f"{r.n_documents}",
            f"{r.mean_quality:.3f}",
            f"{r.total_seconds:.1f}",
            f"{r.seconds_per_doc:.2f}",
            f"{r.total_cost_usd:.4f}",
            f"{r.routing_overhead_s:.2f}",
        ])
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def line(cells):
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

    out = [line(headers), "| " + " | ".join("-" * w for w in widths) + " |"]
    out.extend(line(r) for r in rows)
    return "\n".join(out)
