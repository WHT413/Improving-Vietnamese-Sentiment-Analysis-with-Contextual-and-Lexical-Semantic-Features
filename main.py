#!/usr/bin/env python3
"""Interactive checker for normalization, lexicon profile, and PhoBERT hidden states."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from vnsentiment.inspection import show_sample  # noqa: E402
from vnsentiment.preprocessing.pipeline import (  # noqa: E402
    IdentitySegmenter,
    PreprocessingPipeline,
)
from vnsentiment.preprocessing.segmenter import VnCoreNLPSegmenter  # noqa: E402
from vnsentiment.preprocessing.variants import (  # noqa: E402
    VariantMap,
    VariantMatcher,
)

DEFAULT_MAP = ROOT / "data" / "lexicons" / "teencode_variant_map" / "runtime" / "variant_map.json"


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def build_phobert_branch(device: str):
    from vnsentiment.phobert_branch.branch import PhoBertBranch
    from vnsentiment.phobert_branch.encoder import PhoBertEncoder
    from vnsentiment.phobert_branch.tokenizer import load_phobert_tokenizer

    tokenizer = load_phobert_tokenizer()
    encoder = PhoBertEncoder(device=device)
    encoder.model.eval()
    return PhoBertBranch(tokenizer, encoder)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Kiểm tra chuẩn hóa, lexicon 12 chiều và PhoBERT hidden states."
    )
    parser.add_argument("text", nargs="*", help="Câu cần kiểm tra; bỏ trống để nhập tương tác.")
    parser.add_argument("--map", default=str(DEFAULT_MAP), help="Đường dẫn variant_map.json.")
    parser.add_argument(
        "--segment",
        choices=("vncorenlp", "identity"),
        default="vncorenlp",
        help="Bộ tách từ: vncorenlp (mặc định, RDRSegmenter thật) hoặc identity (offline/debug).",
    )
    parser.add_argument(
        "--vncorenlp-dir",
        default=None,
        help="Thư mục chứa VnCoreNLP-1.2.jar và models/wordsegmenter/.",
    )
    parser.add_argument(
        "--phobert",
        action="store_true",
        help="Chạy PhoBERT-base forward và in hidden states/pooled vector.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Thiết bị cho PhoBERT: auto (mặc định), cpu hoặc cuda.",
    )
    parser.add_argument(
        "--lexicon",
        action="store_true",
        help="Tra composite 12 chiều và in profile từ vựng + projection 768.",
    )
    parser.add_argument(
        "--lexicon-path",
        default=None,
        help="Đường dẫn composite_lexicon.json (mặc định v0.2.0).",
    )
    args = parser.parse_args()

    map_path = Path(args.map)
    if not map_path.is_file():
        print(f"Không tìm thấy bảng ánh xạ: {map_path}", file=sys.stderr)
        print("Chạy lại với --map <path> hoặc kiểm tra lại thư mục data/.", file=sys.stderr)
        return 1

    variant_map = VariantMap.load(map_path)
    matcher = VariantMatcher(variant_map)
    segmenter = (
        VnCoreNLPSegmenter(args.vncorenlp_dir)
        if args.segment == "vncorenlp"
        else IdentitySegmenter()
    )
    segmenter_label = (
        "VnCoreNLP/RDRSegmenter (wseg)"
        if args.segment == "vncorenlp"
        else "identity seam (chưa chạy RDRSegmenter)"
    )
    try:
        pipeline = PreprocessingPipeline(variant_map, segmenter=segmenter)
    except ValueError as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1
    print(
        f"Bảng ánh xạ: v{variant_map.version} | {variant_map.review_status} | "
        f"{len(variant_map.entries)} entries | {variant_map.source}"
    )
    print(f"Segmenter: {segmenter_label}.")

    phobert_branch = None
    if args.phobert:
        device = _resolve_device(args.device)
        try:
            phobert_branch = build_phobert_branch(device)
        except Exception as exc:
            print(f"Lỗi khởi tạo PhoBERT: {exc}", file=sys.stderr)
            return 1
        print(f"PhoBERT: {device} | hidden_size={phobert_branch.encoder.hidden_size}")

    lexicon_cli = None
    if args.lexicon:
        from vnsentiment.lexicon.cli import LexiconCli

        try:
            lexicon_cli = LexiconCli.load(
                args.lexicon_path,
                phobert_branch.tokenizer if phobert_branch is not None else None,
            )
        except Exception as exc:
            print(f"Lỗi khởi tạo lexicon: {exc}", file=sys.stderr)
            return 1
        print(
            f"Lexicon: {len(lexicon_cli.store)} entries | "
            f"{len(lexicon_cli.store.features)} dims."
        )

    if args.text:
        try:
            show_sample(" ".join(args.text), matcher, pipeline, phobert_branch, lexicon_cli)
        except Exception as exc:
            print(f"Lỗi: {exc}", file=sys.stderr)
            return 1
        return 0

    print("\nNhập câu cần kiểm tra (q để thoát).")
    while True:
        try:
            text = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if text.strip().lower() in {"q", "quit", "exit"}:
            return 0
        if not text.strip():
            continue
        try:
            show_sample(text, matcher, pipeline, phobert_branch, lexicon_cli)
        except Exception as exc:  # boundary errors are user-facing here
            print(f"  Lỗi: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
