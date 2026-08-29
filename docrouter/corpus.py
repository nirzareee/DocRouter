"""Corpus loading.

A corpus is a directory of documents plus ground-truth markdown. The layout is
deliberately boring so you can point it at OmniDocBench, at your own annotated
set, or at synthetic samples without changing any code:

    corpus/
      docs/      0001.pdf, 0002.docx, ...
      gold/      0001.md,  0002.md,  ...
      meta.jsonl (optional) one JSON object per doc with extra attributes

`meta.jsonl` is where document attributes live (scanned vs digital, column
count, language, table density). Those attributes are the features your router
will eventually learn from, and they are also how you slice results in the
writeup. "Backend X wins overall" is a weak finding; "backend X wins on scanned
multi-column documents and loses everywhere else" is the finding that justifies
building a router at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass
class Sample:
    doc_path: Path
    gold_markdown: str
    doc_key: str
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def suffix(self) -> str:
        return self.doc_path.suffix.lower()


class Corpus:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.docs_dir = self.root / "docs"
        self.gold_dir = self.root / "gold"
        if not self.docs_dir.is_dir():
            raise FileNotFoundError(f"No docs/ directory under {self.root}")
        self._meta = self._load_meta()

    def _load_meta(self) -> dict[str, dict[str, Any]]:
        path = self.root / "meta.jsonl"
        if not path.exists():
            return {}
        meta: dict[str, dict[str, Any]] = {}
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                meta[record["doc_key"]] = record
        return meta

    def __iter__(self) -> Iterator[Sample]:
        for doc_path in sorted(self.docs_dir.iterdir()):
            if doc_path.name.startswith("."):
                continue
            key = doc_path.stem
            gold_path = self.gold_dir / f"{key}.md"
            if not gold_path.exists():
                # Skipping silently would quietly shrink the benchmark, which is
                # the kind of thing that invalidates a results table.
                raise FileNotFoundError(
                    f"Missing ground truth for {doc_path.name}: expected {gold_path}"
                )
            yield Sample(
                doc_path=doc_path,
                gold_markdown=gold_path.read_text(encoding="utf-8"),
                doc_key=key,
                attrs=self._meta.get(key, {}),
            )

    def __len__(self) -> int:
        return sum(1 for _ in self)
