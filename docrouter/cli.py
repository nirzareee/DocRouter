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
