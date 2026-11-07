"""Corpus token loading for coverage and conflict-frequency checks."""

import json
from collections import Counter
from pathlib import Path


def _is_word_token(token: str) -> bool:
    return any(ch.isalnum() for ch in token)


def load_corpus_token_counts(path: Path) -> tuple[Counter[str], Counter[str]]:
    raw_counts: Counter[str] = Counter()
    word_counts: Counter[str] = Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            for token in row["text"].split():
                raw_counts[token] += 1
                if _is_word_token(token):
                    word_counts[token] += 1
    return raw_counts, word_counts