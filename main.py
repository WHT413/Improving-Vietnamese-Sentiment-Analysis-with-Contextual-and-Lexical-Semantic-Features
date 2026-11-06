#!/usr/bin/env python3
"""Interactive checker for normalization and optional PhoBERT hidden states."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from vnsentiment.preprocessing.collapse import (  # noqa: E402
    collapse_elongations,
    normalize_basic,
)
from vnsentiment.preprocessing.pipeline import (  # noqa: E402
    IdentitySegmenter,
    PreprocessingPipeline,
    RawSample,
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


def build_phobert(device: str):
    from vnsentiment.backbones.backbone import PhoBertBackbone
    from vnsentiment.backbones.encoder import PhoBertEncoder
    from vnsentiment.backbones.tokenizer import load_phobert_tokenizer

    tokenizer = load_phobert_tokenizer()
    encoder = PhoBertEncoder(device=device)
    encoder.model.eval()
    return PhoBertBackbone(tokenizer, encoder)


def print_hidden(backbone, batch, hidden) -> None:
    content = batch.content_mask[0].bool()
    tokens = backbone.tokenizer.convert_ids_to_tokens(batch.input_ids[0].tolist())
    norms = hidden[0].norm(dim=-1)
    print(f"  H          : shape={tuple(hidden.shape)} | device={hidden.device}")
    print("  token norms:")
    for token, is_content, norm in zip(tokens, content.tolist(), norms.tolist()):
        if is_content:
            print(f"    {token:<16} {norm:.4f}")
    if not bool(content.any()):
        print("  pooled     : (không có content token)")
        return
    pooled = hidden[0][content.to(hidden.device)].mean(dim=0)
    first8 = ", ".join(f"{value:.4f}" for value in pooled[:8].tolist())
    print(
        f"  pooled     : norm={pooled.norm().item():.4f} "
        f"mean={pooled.mean().item():+.4f} first8=[{first8}]"
    )
    print("  (chưa có classifier 3 lớp; đây là hidden states PhoBERT)")


def show(
    text: str,
    matcher: VariantMatcher,
    pipeline: PreprocessingPipeline,
    backbone=None,
) -> None:
    basic = normalize_basic(text)
    collapsed = collapse_elongations(basic)
    mapped = matcher.replace(collapsed)
    result = pipeline.process(RawSample("cli", text))
    print(f"  Raw       : {text}")
    print(f"  NFC/lower : {basic}")
    print(f"  Collapse  : {collapsed}")
    print(f"  Biến thể  : {mapped}")
    print(f"  Segmented : {result.normalized_text}")
    if mapped == collapsed:
        print("  (không có biến thể nào được thay)")
    if backbone is not None:
        import torch

        with torch.no_grad():
            batch, hidden = backbone.encode([result.normalized_text])
        print_hidden(backbone, batch, hidden)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Kiểm tra chuẩn hóa đầu vào: NFC/lowercase, collapse, biến thể, segment."
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

    backbone = None
    if args.phobert:
        device = _resolve_device(args.device)
        try:
            backbone = build_phobert(device)
        except Exception as exc:
            print(f"Lỗi khởi tạo PhoBERT: {exc}", file=sys.stderr)
            return 1
        print(f"PhoBERT: {device} | hidden_size={backbone.encoder.hidden_size}")

    if args.text:
        try:
            show(" ".join(args.text), matcher, pipeline, backbone)
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
            show(text, matcher, pipeline, backbone)
        except Exception as exc:  # boundary errors are user-facing here
            print(f"  Lỗi: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
