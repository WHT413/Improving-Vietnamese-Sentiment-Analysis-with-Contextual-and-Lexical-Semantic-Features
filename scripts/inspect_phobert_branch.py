#!/usr/bin/env python3
"""In input tensors của PhoBERT branch cho câu đã word-segmented."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vnsentiment.phobert_branch.batch import build_batch  # noqa: E402
from vnsentiment.phobert_branch.tokenizer import load_phobert_tokenizer  # noqa: E402
from vnsentiment.preprocessing.pipeline import (  # noqa: E402
    PreprocessingPipeline,
    RawSample,
)
from vnsentiment.preprocessing.segmenter import VnCoreNLPSegmenter  # noqa: E402
from vnsentiment.preprocessing.variants import VariantMap  # noqa: E402

DEFAULT_TEXT = "sản_phẩm này không tệ"
DEFAULT_RAW_TEXT = "SP này đẹpppp nhưng pin ko trâu"
DEFAULT_MAP = (
    ROOT / "data" / "lexicons" / "teencode_variant_map" / "runtime" / "variant_map.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "text",
        nargs="*",
        help=f"Câu đã segment; mặc định: {DEFAULT_TEXT!r}",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--from-raw",
        action="store_true",
        help="Chạy preprocessing (NFC/collapse/biến thể/VnCoreNLP) trước khi tokenize.",
    )
    args = parser.parse_args()

    raw_text = " ".join(args.text) if args.text else (
        DEFAULT_RAW_TEXT if args.from_raw else DEFAULT_TEXT
    )
    if args.from_raw:
        variant_map = VariantMap.load(DEFAULT_MAP)
        pipeline = PreprocessingPipeline(
            variant_map, segmenter=VnCoreNLPSegmenter()
        )
        text = pipeline.process(RawSample("cli", raw_text)).normalized_text
        print(f"raw             : {raw_text!r}")
        print(f"segmented       : {text!r}")
    else:
        text = raw_text

    tokenizer = load_phobert_tokenizer()
    batch = build_batch(tokenizer, [text], max_length=args.max_length)

    print(f"text            : {text!r}")
    print(f"input_ids       : {batch.input_ids.tolist()}")
    print(f"attention_mask  : {batch.attention_mask.tolist()}")
    print(f"content_mask    : {batch.content_mask.tolist()}")
    print(f"word_ids        : {batch.word_ids.tolist()}")
    for row in batch.input_ids.tolist():
        print(f"tokens          : {tokenizer.convert_ids_to_tokens(row)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
