"""Text fidelity and reading-order metrics.

Text similarity answers "did the characters survive?". Reading order answers
"did they survive *in the right sequence*?". These come apart badly on
multi-column PDFs: a glyph dump can score high on text similarity while
interleaving two columns into nonsense, which is precisely the failure that
makes downstream answers wrong.
"""

from __future__ import annotations

import re
import unicodedata


def normalize(text: str) -> str:
    """Collapse the differences no one should be scored on."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    return " ".join(text.split()).lower()


def strip_markdown(text: str) -> str:
    """Remove markup so text metrics compare content, not syntax choices."""
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\|[\s:|-]+\|\s*$", "", text, flags=re.MULTILINE)
    text = text.replace("|", " ")
    text = re.sub(r"[*_`]", "", text)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    return text


def normalized_edit_distance(a: str, b: str) -> float:
    """Character-level Levenshtein scaled to [0, 1]. Lower is better."""
    a, b = normalize(a), normalize(b)
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1] / max(len(a), len(b))


def text_similarity(pred: str, gold: str) -> float:
    """1 - NED over markdown-stripped text. Higher is better."""
    return 1.0 - normalized_edit_distance(strip_markdown(pred), strip_markdown(gold))


def _blocks(text: str, min_chars: int = 20) -> list[str]:
    """Split into ordered comparison units.

    Sentences, not paragraphs. Paragraph splitting depends on blank lines
    surviving extraction, and they frequently do not: a backend that emits one
    line per row produces a single giant "paragraph" and scores 0 on reading
    order despite having gotten the order exactly right. Sentence boundaries
    survive whatever the backend does to whitespace, so the metric measures the
    backend instead of measuring its newline style.
    """
    stripped = strip_markdown(text)
    parts = re.split(r"(?<=[.!?])\s+|\n\s*\n", stripped)
    units = [normalize(p) for p in parts if p and p.strip()]
    kept = [u for u in units if len(u) >= min_chars]
    # Very short documents would otherwise have nothing left to compare.
    return kept if kept else units


def _similar(a: str, b: str) -> float:
    return 1.0 - normalized_edit_distance(a, b)


def reading_order_score(pred: str, gold: str, threshold: float = 0.7) -> float:
    """Normalized Kendall tau over blocks matched between pred and gold.

    Each predicted block is matched to its most similar gold block (above a
    similarity threshold, so hallucinated blocks are dropped rather than
    scored). The result is a sequence of gold indices in predicted order; the
    metric is how well-sorted that sequence is.

    Returns 1.0 for perfect order, 0.0 for fully reversed, and 0.5 for the
    order a coin flip would produce. Documents with fewer than two matched
    blocks return 1.0, since order is undefined.
    """
    pred_blocks = _blocks(pred)
    gold_blocks = _blocks(gold)
    if len(gold_blocks) < 2:
        return 1.0

    sequence: list[int] = []
    used: set[int] = set()
    for pb in pred_blocks:
        best_idx, best_score = -1, threshold
        for gi, gb in enumerate(gold_blocks):
            if gi in used:
                continue
            score = _similar(pb, gb)
            if score > best_score:
                best_idx, best_score = gi, score
        if best_idx >= 0:
            sequence.append(best_idx)
            used.add(best_idx)

    if len(sequence) < 2:
        return 0.0 if gold_blocks else 1.0

    concordant = discordant = 0
    for i in range(len(sequence)):
        for j in range(i + 1, len(sequence)):
            if sequence[i] < sequence[j]:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    tau = (concordant - discordant) / total if total else 1.0

    # Scale Kendall tau from [-1, 1] to [0, 1], then penalize for blocks the
    # backend never recovered at all. Without this, a backend that emits two
    # correctly-ordered blocks out of fifty would score a perfect 1.0.
    order_score = (tau + 1) / 2
    coverage = len(sequence) / len(gold_blocks)
    return order_score * coverage
