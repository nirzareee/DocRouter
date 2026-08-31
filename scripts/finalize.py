"""Produce every final number from whatever results exist.

The Docling sweeps were truncated by compute limits, so the corpus has three
different sample sizes: 28 documents for the pure-Python backends in both
conditions, and fewer for Docling. This script reports each at its true n
rather than silently intersecting everything down to the smallest.

Silently intersecting is the tempting move and the wrong one: it would throw
away 28-document evidence for the headline claim in order to make one table
look tidy.

    python scripts/finalize.py
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

RESULTS = Path("results")


def load(*names: str) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for name in names:
        path = RESULTS / name
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[(r["doc_id"], r["backend"])] = r
    return out


def summarize(rows: list[dict], label: str) -> dict | None:
    if not rows:
        return None
    return {
        "arm": label,
        "n": len(rows),
        "text_sim": st.fmean(r["text_sim"] for r in rows),
        "teds": st.fmean(r["teds"] for r in rows),
        "reading_order": st.fmean(r["reading_order"] for r in rows),
        "overall": st.fmean(r["overall"] for r in rows),
        "sec": st.fmean(r["wall_time_s"] for r in rows),
    }


def table(rows: list[dict]) -> str:
    if not rows:
        return "(no data)"
    cols = ["backend", "arm", "n", "text_sim", "TEDS", "read_order", "overall", "sec/doc"]
    body = [[
        r["backend"], r["arm"], f"{r['n']}",
        f"{r['text_sim']:.3f}", f"{r['teds']:.3f}",
        f"{r['reading_order']:.3f}", f"{r['overall']:.3f}", f"{r['sec']:.1f}",
    ] for r in rows]
    w = [max(len(c), *(len(b[i]) for b in body)) for i, c in enumerate(cols)]
    line = lambda cs: "| " + " | ".join(c.ljust(x) for c, x in zip(cs, w)) + " |"
    return "\n".join([line(cols), "| " + " | ".join("-" * x for x in w) + " |"]
                     + [line(b) for b in body])


def main() -> int:
    clean = load("clean_cheap.jsonl", "clean_all.jsonl")
    degraded = load("degraded_cheap.jsonl", "degraded_all.jsonl")
    if not clean and not degraded:
        print("No results found under results/.")
        return 1

    summary = []
    for backend in ("naive", "pylib", "docling"):
        for label, data in (("clean", clean), ("degraded", degraded)):
            rows = [r for (_, b), r in data.items() if b == backend]
            s = summarize(rows, label)
            if s:
                s["backend"] = backend
                summary.append(s)

    print("=" * 78)
    print("HEADLINE: extraction quality by backend and condition")
    print("=" * 78)
    print(table(summary))
    print("\nSample sizes differ: the Docling sweeps were truncated by compute")
    print("limits. Each row reports its own n.\n")

    # The categorical claim, on the full 28-document sample.
    deg_cheap = [r for (_, b), r in degraded.items() if b in ("naive", "pylib")]
    if deg_cheap:
        nonzero = sum(1 for r in deg_cheap if r["overall"] > 0)
        print("=" * 78)
        print("F1: pure-Python extraction on scanned documents")
        print("=" * 78)
        print(f"  {len(deg_cheap)} (document, backend) pairs")
        print(f"  scoring above 0.000: {nonzero}")
        print(f"  max score observed:  {max(r['overall'] for r in deg_cheap):.4f}")

    # Per-document clean comparison where both backends ran.
    pairs = []
    for (doc, b), r in clean.items():
        if b == "docling" and (doc, "pylib") in clean:
            p = clean[(doc, "pylib")]
            pairs.append((doc, p["overall"], r["overall"],
                          r["overall"] - p["overall"],
                          r["wall_time_s"] / max(p["wall_time_s"], 1e-9)))
    if pairs:
        print("\n" + "=" * 78)
        print("F2: what Docling buys on CLEAN filings")
        print("=" * 78)
        print(f"{'pylib':>7} {'docling':>8} {'gain':>7} {'slowdown':>9}  doc")
        for doc, pq, dq, gain, ratio in sorted(pairs, key=lambda x: -x[3]):
            print(f"{pq:7.3f} {dq:8.3f} {gain:+7.3f} {ratio:8.0f}x  {doc}")
        print(f"\n  mean gain:     {st.fmean(p[3] for p in pairs):+.3f}")
        print(f"  mean slowdown: {st.fmean(p[4] for p in pairs):.0f}x")
        print(f"  n = {len(pairs)} documents")

    # Routing arithmetic, computed directly rather than via the policy engine
    # so it works with whatever subset completed.
    routable = [d for d in {k[0] for k in clean}
                if (d, "pylib") in clean and (d, "docling") in clean
                and (d, "docling") in degraded]
    if routable:
        print("\n" + "=" * 78)
        print("F3: routing arithmetic on matched pairs")
        print("=" * 78)
        r_q = r_t = a_q = a_t = 0.0
        for d in routable:
            # Clean half -> cheap backend; degraded half -> docling.
            r_q += clean[(d, "pylib")]["overall"] + degraded[(d, "docling")]["overall"]
            r_t += clean[(d, "pylib")]["wall_time_s"] + degraded[(d, "docling")]["wall_time_s"]
            a_q += clean[(d, "docling")]["overall"] + degraded[(d, "docling")]["overall"]
            a_t += clean[(d, "docling")]["wall_time_s"] + degraded[(d, "docling")]["wall_time_s"]
        n = len(routable) * 2
        print(f"  matched documents: {len(routable)} ({n} document-conditions)")
        print(f"  always(docling):   quality {a_q/n:.3f}  time {a_t:8.0f}s")
        print(f"  rules router:      quality {r_q/n:.3f}  time {r_t:8.0f}s")
        print(f"  always(pylib):     quality {sum(clean[(d,'pylib')]['overall'] for d in routable)/n:.3f}"
              f"  time {sum(clean[(d,'pylib')]['wall_time_s'] for d in routable):8.0f}s")
        print(f"\n  router keeps {r_q/a_q*100:.1f}% of always-docling quality "
              f"at {r_t/a_t*100:.1f}% of its compute")

    return 0


if __name__ == "__main__":
    sys.exit(main())
