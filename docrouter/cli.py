"""Command-line entry point.

    python -m docrouter.cli bench data/sample_corpus
    python -m docrouter.cli bench data/sample_corpus --backends naive pylib
    python -m docrouter.cli parse report.pdf --backend pylib
    python -m docrouter.cli backends
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .backends import REGISTRY, available_backends, get_backend
from .cache import ParseCache
from .corpus import Corpus
from .evaluate import evaluate
from .features import extract_corpus_features
from .router import (
    always,
    complete_documents,
    evaluate_oracle,
    evaluate_policy,
    format_comparison,
    load_features,
    load_results,
    text_layer_rule,
)


def cmd_backends(args: argparse.Namespace) -> int:
    available = {b.name for b in available_backends()}
    for name in sorted(REGISTRY):
        mark = "available" if name in available else "not installed"
        print(f"  {name:<12} {mark}")
    return 0


def cmd_parse(args: argparse.Namespace) -> int:
    backend = get_backend(args.backend)
    if not backend.is_available():
        print(f"Backend {args.backend!r} is not installed.", file=sys.stderr)
        return 1
    result = backend.parse(Path(args.path))
    if not result.ok:
        print(f"Parse failed: {result.error}", file=sys.stderr)
        return 1
    print(result.markdown)
    print(
        f"\n<!-- {backend.name}: {result.pages}p "
        f"{result.wall_time_s:.2f}s ${result.cost_usd:.4f} -->",
        file=sys.stderr,
    )
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    """Compute routing features for a corpus and write them to JSONL."""
    import json

    feats = extract_corpus_features(args.corpus)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for f in feats:
            fh.write(json.dumps(f.to_dict()) + "\n")

    total_ms = sum(f.feature_time_ms for f in feats)
    scanned = sum(1 for f in feats if f.is_scanned)
    print(f"{len(feats)} documents")
    print(f"  scanned (no text layer): {scanned}")
    print(f"  multi-column:            {sum(1 for f in feats if f.est_columns > 1)}")
    print(f"  total feature time:      {total_ms:.0f} ms "
          f"({total_ms / max(len(feats), 1):.0f} ms/doc)")
    print(f"\nWritten to {out}")
    return 0


def cmd_route(args: argparse.Namespace) -> int:
    """Compare routing policies against measured benchmark results."""
    results = load_results(*args.results)
    features = load_features(*args.features)

    backends = sorted({b for _, b in results})
    n_all = len(features)

    if args.complete_only:
        features = complete_documents(features, results, backends)
        dropped = n_all - len(features)
        if dropped:
            print(f"Restricted to {len(features)} of {n_all} documents scored by "
                  f"every backend ({dropped} incomplete, excluded).")
    print(f"{len(features)} documents, backends: {', '.join(backends)}\n")

    if not features:
        print("No document has results from every backend. "
              "Finish the sweep, or pass fewer backends.")
        return 1

    evaluated = [
        evaluate_policy(always(b), features, results, name=f"always({b})")
        for b in backends
    ]
    evaluated.append(
        evaluate_policy(
            text_layer_rule(args.cheap, args.expensive),
            features, results, name="rules(text_layer)",
        )
    )
    evaluated.append(evaluate_oracle(features, results, backends))

    print(format_comparison(evaluated))

    rules = next(r for r in evaluated if r.name == "rules(text_layer)")
    best = max(
        (r for r in evaluated if r.name.startswith("always(")),
        key=lambda r: r.mean_quality,
    )
    oracle = next(r for r in evaluated if r.name == "oracle")

    print(f"\nrules vs {best.name}:")
    print(f"  quality {rules.mean_quality:.3f} vs {best.mean_quality:.3f}"
          f"  ({rules.mean_quality - best.mean_quality:+.3f})")
    if best.total_seconds:
        saved = 1 - rules.total_seconds / best.total_seconds
        verb = "less" if saved >= 0 else "more"
        print(f"  time    {rules.total_seconds:.0f}s vs {best.total_seconds:.0f}s"
              f"  ({abs(saved) * 100:.0f}% {verb})")
    print(f"  routing overhead: {rules.routing_overhead_s:.1f}s "
          f"({rules.routing_overhead_s / max(rules.total_seconds, 1e-9) * 100:.2f}% "
          f"of its own parse time)")
    print(f"\noracle ceiling: {oracle.mean_quality:.3f} "
          f"({oracle.mean_quality - rules.mean_quality:+.3f} above rules)")

    if args.out:
        import json

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            for r in evaluated:
                fh.write(json.dumps(r.to_dict()) + "\n")
        print(f"\nWritten to {out}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    corpus = Corpus(args.corpus)
    if args.backends:
        backends = [get_backend(name) for name in args.backends]
        missing = [b.name for b in backends if not b.is_available()]
        if missing:
            print(f"Not installed: {', '.join(missing)}", file=sys.stderr)
            return 1
    else:
        backends = available_backends()

    print(f"Corpus: {args.corpus}")
    print(f"Backends: {', '.join(b.name for b in backends)}\n")

    cache = ParseCache(enabled=not args.no_cache)
    if args.clear_cache:
        print(f"Cleared {cache.clear()} cache entries\n")

    # Stream rows to disk as they are produced so a crash mid-sweep keeps
    # everything scored so far.
    run = evaluate(
        corpus, backends, verbose=not args.quiet, cache=cache, out_path=args.out
    )
    out = Path(args.out)
    run = evaluate(corpus, backends, verbose=not args.quiet, cache=cache)
    out = run.write_jsonl(args.out)

    print(f"\n{run.format_table()}")
    if not args.no_cache:
        st = cache.stats()
        print(f"\ncache: {st['hits']} hits, {st['misses']} misses "
              f"({st['hit_rate']*100:.0f}% hit rate, {st['entries']} entries)")
    print(f"\nPer-document rows written to {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="docrouter")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("backends", help="list registered backends")
    p.set_defaults(func=cmd_backends)

    p = sub.add_parser("parse", help="parse one document to markdown")
    p.add_argument("path")
    p.add_argument("--backend", default="pylib")
    p.set_defaults(func=cmd_parse)

    p = sub.add_parser("features", help="compute routing features for a corpus")
    p.add_argument("corpus")
    p.add_argument("--out", default="results/features.jsonl")
    p.set_defaults(func=cmd_features)

    p = sub.add_parser("route", help="compare routing policies")
    p.add_argument("--results", nargs="+", required=True,
                   help="benchmark JSONL files (clean and degraded)")
    p.add_argument("--features", nargs="+", required=True,
                   help="feature JSONL files")
    p.add_argument("--cheap", default="pylib",
                   help="backend for documents with a text layer")
    p.add_argument("--expensive", default="docling",
                   help="backend for documents without one")
    p.add_argument("--out", default="results/routing.jsonl")
    p.add_argument("--complete-only", action="store_true",
                   help="evaluate only documents scored by every backend")
    p.set_defaults(func=cmd_route)

    p = sub.add_parser("bench", help="run the benchmark over a corpus")
    p.add_argument("corpus")
    p.add_argument("--backends", nargs="*", default=None)
    p.add_argument("--out", default="results/run.jsonl")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no-cache", action="store_true",
                   help="re-parse everything, ignoring cached results")
    p.add_argument("--clear-cache", action="store_true",
                   help="delete all cached parses before running")
    p.set_defaults(func=cmd_bench)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
