"""Sanity-check a generated gold file before trusting it.

Reading 130 KB of markdown by eye does not scale and does not reliably catch
the failure modes that matter. These checks do.

    python scripts/inspect_gold.py data/edgar/clean/gold/AAPL_10Q_2026-07-31.md
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docrouter.metrics.teds import extract_markdown_tables  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path")
    ap.add_argument("--show", type=int, default=3, help="sample tables to print")
    args = ap.parse_args()

    md = Path(args.path).read_text(encoding="utf-8")
    blocks = [b.strip() for b in md.split("\n\n") if b.strip()]
    tables = extract_markdown_tables(md)
    headings = [b for b in blocks if b.startswith("#")]
    prose = [b for b in blocks if not b.startswith("#") and not b.startswith("|")]

    print(f"size        {len(md) / 1024:.0f} KB")
    print(f"blocks      {len(blocks)}")
    print(f"headings    {len(headings)}")
    print(f"tables      {len(tables)}")
    print(f"prose       {len(prose)}")

    # Duplicated prose is the nested-div failure: the same text emitted once per
    # ancestor element. It inflates the file and corrupts reading-order scoring.
    counts = Counter(p for p in prose if len(p) > 60)
    dupes = [(n, p) for p, n in counts.items() if n > 1]
    dup_chars = sum((n - 1) * len(p) for n, p in dupes)
    print(f"\nduplicated prose blocks: {len(dupes)}"
          f" ({dup_chars / 1024:.0f} KB wasted, "
          f"{dup_chars / max(len(md), 1) * 100:.0f}% of file)")
    for n, p in sorted(dupes, reverse=True)[:3]:
        print(f"  x{n}: {p[:90]}...")

    # Tables with one data row are usually layout furniture that slipped past
    # the filter; very wide tables usually mean spacer columns survived.
    if tables:
        widths = [max(len(r) for r in t) for t in tables]
        rows = [len(t) for t in tables]
        print(f"\ntable widths  min={min(widths)} median={sorted(widths)[len(widths)//2]} max={max(widths)}")
        print(f"table rows    min={min(rows)} median={sorted(rows)[len(rows)//2]} max={max(rows)}")
        thin = sum(1 for t in tables if len(t) <= 2)
        print(f"tables with <=2 rows: {thin}  (candidates for layout tables)")

        empties = sum(
            1 for t in tables
            for row in t for c in row if not c.strip()
        )
        total_cells = sum(len(row) for t in tables for row in t)
        print(f"empty cells: {empties}/{total_cells} "
              f"({empties / max(total_cells, 1) * 100:.0f}%)")

    print(f"\n--- first {args.show} tables ---")
    for t in tables[:args.show]:
        for row in t[:4]:
            print("  | " + " | ".join(c[:18] for c in row) + " |")
        if len(t) > 4:
            print(f"  ... {len(t) - 4} more rows")
        print()

    # Parenthesized negatives and currency columns: the annotation decisions.
    negs = len(re.findall(r"\(\s*[\d,]+\s*\)", md))
    dollar_cells = len(re.findall(r"\|\s*\$\s*\|", md))
    print(f"parenthesized negatives: {negs}")
    print(f"standalone '$' cells:    {dollar_cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
