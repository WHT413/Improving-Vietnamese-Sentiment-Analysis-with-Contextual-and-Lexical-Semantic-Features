"""Coverage computation against train/dev word and raw tokens."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .config import FEATURES


def compute_split_coverage(
    raw_counts: Counter[str],
    word_counts: Counter[str],
    entries: dict[str, dict[str, list[Any]]],
) -> dict[str, Any]:
    return {
        "denominators": _denominators(raw_counts, word_counts),
        "word_token_coverage": _coverage_block(word_counts, entries),
        "raw_token_coverage": _coverage_block(raw_counts, entries),
        "per_dimension": _per_dimension(raw_counts, word_counts, entries),
    }


def _denominators(raw_counts: Counter[str], word_counts: Counter[str]) -> dict[str, int]:
    total_raw_tokens = sum(raw_counts.values())
    total_word_tokens = sum(word_counts.values())
    return {
        "word_tokens": total_word_tokens,
        "word_types": len(word_counts),
        "raw_tokens": total_raw_tokens,
        "raw_types": len(raw_counts),
        "excluded_punctuation_tokens": total_raw_tokens - total_word_tokens,
        "excluded_punctuation_types": len(raw_counts) - len(word_counts),
    }


def _coverage_block(
    counts: Counter[str], entries: dict[str, dict[str, list[Any]]]
) -> dict[str, Any]:
    total_types = len(counts)
    total_tokens = sum(counts.values())
    covered_types = sum(1 for key in counts if key in entries)
    covered_tokens = sum(count for key, count in counts.items() if key in entries)
    return {
        "types_covered": covered_types,
        "types_total": total_types,
        "type_coverage_pct": round(covered_types / total_types * 100, 2),
        "tokens_covered": covered_tokens,
        "tokens_total": total_tokens,
        "token_coverage_pct": round(covered_tokens / total_tokens * 100, 2),
    }


def _per_dimension(
    raw_counts: Counter[str],
    word_counts: Counter[str],
    entries: dict[str, dict[str, list[Any]]],
) -> dict[str, dict[str, Any]]:
    total_raw_tokens = sum(raw_counts.values())
    total_raw_types = len(raw_counts)
    total_word_tokens = sum(word_counts.values())
    total_word_types = len(word_counts)
    result: dict[str, dict[str, Any]] = {}
    for index, feature in enumerate(FEATURES):
        dim_word_types = sum(
            1
            for key in word_counts
            if key in entries and entries[key]["mask"][index] == 1
        )
        dim_word_tokens = sum(
            count
            for key, count in word_counts.items()
            if key in entries and entries[key]["mask"][index] == 1
        )
        dim_raw_types = sum(
            1
            for key in raw_counts
            if key in entries and entries[key]["mask"][index] == 1
        )
        dim_raw_tokens = sum(
            count
            for key, count in raw_counts.items()
            if key in entries and entries[key]["mask"][index] == 1
        )
        result[feature] = {
            "word_types_covered": dim_word_types,
            "word_type_coverage_pct": round(dim_word_types / total_word_types * 100, 2),
            "word_tokens_covered": dim_word_tokens,
            "word_token_coverage_pct": round(dim_word_tokens / total_word_tokens * 100, 2),
            "raw_types_covered": dim_raw_types,
            "raw_type_coverage_pct": round(dim_raw_types / total_raw_types * 100, 2),
            "raw_tokens_covered": dim_raw_tokens,
            "raw_token_coverage_pct": round(dim_raw_tokens / total_raw_tokens * 100, 2),
        }
    return result