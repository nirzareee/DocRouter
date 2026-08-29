"""Download filings from SEC EDGAR.

SEC requires a descriptive User-Agent identifying you, and rate-limits to 10
requests/second. Both are in their published access policy; ignoring either
gets your IP blocked. Set your address once:

    export EDGAR_UA="Your Name your@email.com"

    python scripts/fetch_edgar.py --tickers AAPL MSFT JPM --forms 10-Q --limit 2

Downloads the primary HTML document of each filing to data/edgar/html/ and
records provenance in data/edgar/sources.jsonl. Provenance matters: a benchmark
whose inputs cannot be traced back to a specific accession number is not
reproducible, and "reproducible" is most of what this project is claiming.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import requests

SEC = "https://www.sec.gov"
DATA = "https://data.sec.gov"
OUT = Path("data/edgar")
RATE_LIMIT_S = 0.15  # ~7 req/s, comfortably under SEC's 10/s ceiling


def headers() -> dict[str, str]:
    ua = os.environ.get("EDGAR_UA")
    if not ua:
        raise SystemExit(
            "Set EDGAR_UA first, e.g.\n"
            '  export EDGAR_UA="Your Name your@email.com"\n'
            "SEC requires a descriptive User-Agent and will block requests without one."
        )
    return {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}


def get(url: str) -> requests.Response:
    time.sleep(RATE_LIMIT_S)
    r = requests.get(url, headers=headers(), timeout=30)
    r.raise_for_status()
    return r


def ticker_to_cik() -> dict[str, str]:
    data = get(f"{SEC}/files/company_tickers.json").json()
    return {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in data.values()}


def recent_filings(cik: str, forms: set[str], limit: int) -> list[dict]:
    data = get(f"{DATA}/submissions/CIK{cik}.json").json()
    recent = data["filings"]["recent"]
    out = []
    for i, form in enumerate(recent["form"]):
        if form not in forms:
            continue
        out.append({
            "form": form,
            "accession": recent["accessionNumber"][i].replace("-", ""),
            "primary_doc": recent["primaryDocument"][i],
            "filing_date": recent["filingDate"][i],
            "company": data.get("name", ""),
        })
        if len(out) >= limit:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tickers", nargs="+", required=True)
    ap.add_argument("--forms", nargs="+", default=["10-Q"])
    ap.add_argument("--limit", type=int, default=2, help="filings per ticker")
    args = ap.parse_args()

    html_dir = OUT / "html"
    html_dir.mkdir(parents=True, exist_ok=True)

    print("Resolving tickers...")
    lookup = ticker_to_cik()

    records = []
    for ticker in args.tickers:
        cik = lookup.get(ticker.upper())
        if not cik:
            print(f"  {ticker}: not found, skipping")
            continue

        for f in recent_filings(cik, set(args.forms), args.limit):
            url = (
                f"{SEC}/Archives/edgar/data/{int(cik)}/"
                f"{f['accession']}/{f['primary_doc']}"
            )
            key = f"{ticker.upper()}_{f['form'].replace('-', '')}_{f['filing_date']}"
            dest = html_dir / f"{key}.html"

            if dest.exists():
                print(f"  {key}: cached")
            else:
                try:
                    dest.write_bytes(get(url).content)
                    print(f"  {key}: {dest.stat().st_size // 1024} KB")
                except requests.HTTPError as exc:
                    print(f"  {key}: failed ({exc})")
                    continue

            records.append({
                "doc_key": key, "ticker": ticker.upper(), "cik": cik,
                "url": url, **f,
            })

    with open(OUT / "sources.jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    print(f"\n{len(records)} filings in {html_dir}")
    print(f"Provenance written to {OUT / 'sources.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
