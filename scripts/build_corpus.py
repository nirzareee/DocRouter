"""Turn downloaded filing HTML into a benchmark corpus.

For each filing:
  1. HTML -> markdown gold        (docrouter/edgar/html_to_gold.py)
  2. HTML -> clean PDF            (headless Chromium via Playwright)
  3. clean PDF -> degraded PDF    (scripts/degrade.py)

Produces the matched-pair corpus the whole experiment rests on: the same
document under two conditions, sharing one ground truth file.

    python scripts/build_corpus.py --degrade medium

Requires Playwright's browser once:  python -m playwright install chromium
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# Running `python scripts/build_corpus.py` puts scripts/ on sys.path, not the
# repo root, so `import docrouter` fails. Add the project root explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docrouter.edgar.html_to_gold import convert_file

OUT = Path("data/edgar")


def render_pdf(html_path: Path, pdf_path: Path) -> None:
    from playwright.sync_api import sync_playwright

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        # file:// so relative assets resolve; filings are largely self-contained.
        page.goto(html_path.resolve().as_uri(), wait_until="load", timeout=60_000)
        page.pdf(
            path=str(pdf_path),
            format="Letter",
            print_background=True,
            margin={"top": "0.5in", "bottom": "0.5in",
                    "left": "0.5in", "right": "0.5in"},
        )
        browser.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--degrade", choices=["light", "medium", "heavy"], default="medium")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-degraded", action="store_true")
    args = ap.parse_args()

    html_dir = OUT / "html"
    if not html_dir.is_dir():
        raise SystemExit("No data/edgar/html/. Run scripts/fetch_edgar.py first.")

    clean = OUT / "clean"
    degraded = OUT / "degraded"
    for d in (clean / "docs", clean / "gold", degraded / "docs", degraded / "gold"):
        d.mkdir(parents=True, exist_ok=True)

    meta_clean, meta_degraded = [], []

    for html_path in sorted(html_dir.glob("*.html")):
        key = html_path.stem
        print(f"{key}:")

        gold = convert_file(html_path, clean / "gold" / f"{key}.md")
        print(f"  gold: {gold.n_tables} tables, {gold.n_headings} headings,"
              f" {gold.n_paragraphs} paragraphs")

        pdf_path = clean / "docs" / f"{key}.pdf"
        if not pdf_path.exists():
            render_pdf(html_path, pdf_path)
        print(f"  clean pdf: {pdf_path.stat().st_size // 1024} KB")

        meta_clean.append({
            "doc_key": key, "condition": "clean", "scanned": False,
            "n_tables": gold.n_tables, "source": "edgar",
        })

        if not args.skip_degraded:
            from scripts.degrade import degrade_pdf  # noqa: PLC0415

            deg_path = degraded / "docs" / f"{key}.pdf"
            n = degrade_pdf(pdf_path, deg_path, args.degrade, args.seed)
            # Same ground truth, by construction. This is the whole design.
            shutil.copy(clean / "gold" / f"{key}.md", degraded / "gold" / f"{key}.md")
            print(f"  degraded pdf: {n} pages ({args.degrade})")

            meta_degraded.append({
                "doc_key": key, "condition": "degraded", "scanned": True,
                "severity": args.degrade, "n_tables": gold.n_tables,
                "source": "edgar",
            })

    for path, records in ((clean / "meta.jsonl", meta_clean),
                          (degraded / "meta.jsonl", meta_degraded)):
        with open(path, "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")

    print(f"\nclean:    {len(meta_clean)} docs -> {clean}")
    print(f"degraded: {len(meta_degraded)} docs -> {degraded}")
    print("\nNext: review the generated gold by hand, then")
    print("  python -m docrouter.cli bench data/edgar/clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
