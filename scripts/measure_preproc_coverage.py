#!/usr/bin/env python3
"""Measure variant-map and segmenter impact on UIT-VSFC train/dev.

Metrics:
- segmenter_exact_match: adapter output equals frozen processed JSONL text.
- compound_token_rate / sentences_with_compound: how often RDR joins words.
- collapse_changed / variant_changed: normalization impact before segmentation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vnsentiment.preprocessing.collapse import normalize_basic, normalize_text  # noqa: E402
from vnsentiment.preprocessing.segmenter import VnCoreNLPSegmenter  # noqa: E402
from vnsentiment.preprocessing.variants import VariantMap, VariantMatcher  # noqa: E402

RAW_DIR = ROOT / "data" / "raw" / "uit-vsfc"
PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_MAP = ROOT / "data" / "lexicons" / "teencode_variant_map" / "runtime" / "variant_map.json"


def load_split(split: str, limit: int | None) -> tuple[list[str], list[str]]:
    raw_path = RAW_DIR / f"{split}_sents.txt"
    processed_path = PROCESSED_DIR / f"{split}.jsonl"
    raw_lines = raw_path.read_text(encoding="utf-8").splitlines()
    processed_lines = [
        json.loads(line)["text"]
        for line in processed_path.read_text(encoding="utf-8").splitlines()
    ]
    if len(raw_lines) != len(processed_lines):
        raise ValueError(
            f"{split}: raw={len(raw_lines)} processed={len(processed_lines)}"
        )
    if limit is not None:
        raw_lines = raw_lines[:limit]
        processed_lines = processed_lines[:limit]
    return raw_lines, processed_lines


def measure_split(
    split: str,
    segmenter: VnCoreNLPSegmenter,
    matcher: VariantMatcher,
    limit: int | None,
) -> dict:
    raw_lines, processed_lines = load_split(split, limit)
    exact_match = 0
    compound_tokens = 0
    total_tokens = 0
    sentences_with_compound = 0
    collapse_changed = 0
    variant_changed = 0
    variant_token_delta = 0
    examples = []

    for raw_text, processed_text in zip(raw_lines, processed_lines):
        basic = normalize_basic(raw_text)
        segmented = segmenter.segment(basic)
        if segmented == processed_text:
            exact_match += 1

        tokens = segmented.split()
        total_tokens += len(tokens)
        compounds = sum(1 for token in tokens if "_" in token)
        compound_tokens += compounds
        if compounds:
            sentences_with_compound += 1

        collapsed = normalize_text(raw_text)
        mapped = matcher.replace(collapsed)
        if collapsed != basic:
            collapse_changed += 1
        if mapped != collapsed:
            variant_changed += 1
            variant_token_delta += len(mapped.split()) - len(collapsed.split())
            if len(examples) < 5:
                examples.append({"raw": raw_text, "mapped": mapped})

    rows = len(raw_lines)
    return {
        "split": split,
        "rows": rows,
        "segmenter_exact_match": exact_match,
        "segmenter_exact_match_pct": round(100 * exact_match / rows, 2),
        "compound_tokens": compound_tokens,
        "total_tokens": total_tokens,
        "compound_token_rate_pct": round(100 * compound_tokens / total_tokens, 2),
        "sentences_with_compound": sentences_with_compound,
        "sentences_with_compound_pct": round(100 * sentences_with_compound / rows, 2),
        "mean_tokens": round(total_tokens / rows, 3),
        "collapse_changed": collapse_changed,
        "collapse_changed_pct": round(100 * collapse_changed / rows, 2),
        "variant_changed": variant_changed,
        "variant_changed_pct": round(100 * variant_changed / rows, 2),
        "variant_token_delta": variant_token_delta,
        "examples": examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["train", "dev"])
    parser.add_argument("--limit", type=int, default=None, help="Chỉ đo N dòng đầu mỗi split.")
    parser.add_argument("--map", default=str(DEFAULT_MAP))
    parser.add_argument("--json", action="store_true", help="In JSON thay vì bảng.")
    args = parser.parse_args()

    variant_map = VariantMap.load(Path(args.map))
    if variant_map.review_status != "approved":
        print(
            f"Lỗi: variant map chưa approved ({variant_map.review_status!r})",
            file=sys.stderr,
        )
        return 1
    matcher = VariantMatcher(variant_map)
    segmenter = VnCoreNLPSegmenter()

    results = [
        measure_split(split, segmenter, matcher, args.limit)
        for split in args.splits
    ]
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    print(f"Variant map: v{variant_map.version} | {len(variant_map.entries)} entries")
    for result in results:
        print(f"\n## {result['split']} ({result['rows']} câu)")
        print(f"- Segmenter exact match: {result['segmenter_exact_match']} "
              f"({result['segmenter_exact_match_pct']}%)")
        print(f"- Compound tokens: {result['compound_tokens']}/"
              f"{result['total_tokens']} ({result['compound_token_rate_pct']}%)")
        print(f"- Câu có compound: {result['sentences_with_compound']} "
              f"({result['sentences_with_compound_pct']}%)")
        print(f"- Token/câu: {result['mean_tokens']}")
        print(f"- Collapse làm đổi câu: {result['collapse_changed']} "
              f"({result['collapse_changed_pct']}%)")
        print(f"- Variant map làm đổi câu: {result['variant_changed']} "
              f"({result['variant_changed_pct']}%), token delta "
              f"{result['variant_token_delta']:+d}")
        for example in result["examples"]:
            print(f"  - {example['raw']!r} → {example['mapped']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
