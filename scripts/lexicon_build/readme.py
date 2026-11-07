"""Generate the human-readable provenance and coverage README."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import FEATURES, VERSION


def write_readme(path: Path, payload: dict[str, Any]) -> None:
    lines = (
        _identity()
        + _key_pipeline()
        + _schema()
        + _conflicts(payload)
        + _coverage(payload)
        + _spot_checks()
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _identity() -> list[str]:
    return [
        f"# Composite 12D Vietnamese Lexicon — `v{VERSION}`",
        "",
        "## 1. Nguồn dữ liệu đầu vào (Read-Only)",
        "",
        "- **VietSentiWordNet 1.3.5:** `data/lexicons/vietsentiwordnet/VietSentiWordnet_Ver1.3.5.txt` (1.016 dòng dữ liệu, 6 cột TSV).",
        "- **Vietnamese NRC-VAD:** `data/lexicons/nrc-vad/Vietnamese-NRC-VAD-Lexicon.txt` (19.971 dòng dữ liệu, 5 cột TSV).",
        "- **VnEmoLex:** `data/lexicons/vnemolex/VnEmoLex.xlsx` (10.627 dòng dữ liệu, giữ 8 cột cảm xúc, bỏ `Positive`/`Negative`).",
        "- **Tập đo coverage:** Chỉ đo trên `data/processed/train.jsonl` và `data/processed/dev.jsonl` (tuyệt đối không đọc `test.jsonl`).",
        "",
    ]


def _key_pipeline() -> list[str]:
    return [
        "## 2. Chuẩn hoá khóa (Canonical Key Pipeline)",
        "",
        "Khóa từ vựng dùng chung chuẩn tách từ với `data/processed/`:",
        "1. Đổi `_` sang khoảng trắng trước khi đưa qua bộ tách từ.",
        "2. Chuẩn hoá Unicode `NFC` -> `lowercase` -> gộp khoảng trắng (`normalize_basic`).",
        "3. Tách từ bằng `VnCoreNLPSegmenter` (RDRSegmenter) và strip ký tự `_` ở đầu/cuối token.",
        "4. Chỉ giữ khóa **single-token** (không chứa khoảng trắng); các mục multi-token sau RDR được loại bỏ và ghi đầy đủ vào `dropped_multitoken.tsv`.",
        "",
    ]


def _schema() -> list[str]:
    return [
        "## 3. Scale Contract & Schema",
        "",
        "- **Thứ tự 12 chiều (`features`):**",
        "  `polarity, valence, arousal, dominance, joy, sadness, anger, fear, disgust, surprise, trust, anticipation`",
        "- **Thang gốc lưu trong composite:**",
        "  - `polarity` (chiều 0): `PosScore - NegScore` $\\in [-1, 1]$ (thực tế $\\in [-1.0, 0.875]$).",
        "  - `valence, arousal, dominance` (chiều 1..3): giữ nguyên thang gốc $[0, 1]$ của file tiếng Việt (**không** áp `(x+1)/2`).",
        "  - 8 chiều emotion (chiều 4..11): nhị phân $\\{0.0, 1.0\\}$.",
        "- **Quy ước `mask`:**",
        "  - `mask = 1, value = 0.0`: dữ liệu thật có giá trị 0 (neutral polarity hoặc không mang cảm xúc đó).",
        "  - `mask = 0, value = 0.0`: chiều thiếu dữ liệu, ô trống, hoặc bị che do xung đột chưa chốt.",
        "- **Hợp đồng với Loader B1 (`store.py`):**",
        "  Loader B1 chỉ đọc `composite_lexicon.json`, ánh xạ `polarity' = (polarity + 1) / 2` về $[0, 1]$ đối với chiều có `mask[0] = 1`, không tính lại điểm nguồn.",
        "",
    ]


def _conflicts(payload: dict[str, Any]) -> list[str]:
    src_stats = payload["stats"]["sources"]
    vswn = src_stats["vietsentiwordnet"]
    vad = src_stats["nrc_vad_vi"]
    selected_outside = vswn["conflict_selected_keys"] - vswn["conflict_selected_in_train"]
    conflicts_outside = vswn["conflict_keys"] - vswn["conflict_keys_in_train"]
    partial_valence = vad["active_keys_by_dim"]["valence"] - vad["unambiguous_keys"]
    partial_arousal = vad["active_keys_by_dim"]["arousal"] - vad["unambiguous_keys"]
    partial_dominance = vad["active_keys_by_dim"]["dominance"] - vad["unambiguous_keys"]
    return [
        "## 4. Chính sách xử lý xung đột (Conflict Policy)",
        "",
        "### 4.1 VietSentiWordNet (`conflicts_vswn.tsv`)",
        f"- Tổng khóa single-token: **{src_stats['vietsentiwordnet']['usable_single_token_keys']}** ({src_stats['vietsentiwordnet']['unambiguous_keys']} đồng nhất, **{src_stats['vietsentiwordnet']['conflict_keys']}** xung đột; trong đó **{src_stats['vietsentiwordnet']['conflict_keys_in_train']}** khóa xuất hiện trong `train.jsonl`).",
        "- Nguyên nhân chính gây xung đột: cột `SynsetTerms` chứa các cụm phủ định/bổ nghĩa viết cách không nối `_` (ví dụ `không tốt#10`, `không thân_thiện#2`, `không hài_lòng#1`, `kém hấp_dẫn#1`, `chưa hoàn_thành#1`, `không tệ#1`), khi tách theo khoảng trắng làm sinh ra mảnh `không` và gán nhầm điểm phủ định vào từ gốc (`tốt`, `thân_thiện`, `hài_lòng`...).",
        f"- Đã kiểm tra và chọn nghĩa standalone có căn cứ cho **{vswn['conflict_selected_keys']}** khóa mang sắc thái cảm xúc rõ ràng (**{vswn['conflict_selected_in_train']}/{vswn['conflict_keys_in_train']}** khóa trong `train.jsonl` + **{selected_outside}/{conflicts_outside}** khóa ngoài `train.jsonl`).",
        f"- Giữ `mask = 0, value = 0.0` cho **{src_stats['vietsentiwordnet']['conflict_masked_keys']}** khóa còn lại (từ hư `không`, `rất`, `có`, `được`, `hơn`, `quá`, `chưa`, `bị`... hoặc mảnh từ đa nghĩa).",
        "",
        "### 4.2 Vietnamese NRC-VAD (`conflicts_vad.tsv`)",
        f"- Tổng khóa single-token: **{src_stats['nrc_vad_vi']['usable_single_token_keys']}** ({src_stats['nrc_vad_vi']['unambiguous_keys']} đồng nhất cả 3 chiều, **{src_stats['nrc_vad_vi']['conflict_keys']}** có ít nhất 1 chiều xung đột).",
        f"- So sánh độc lập từng chiều (`valence`, `arousal`, `dominance`): giữ lại chiều đồng nhất (`mask = 1`) cho **{vad['conflict_partial_kept_keys']}** khóa (thêm {partial_valence} `valence`, {partial_arousal} `arousal`, {partial_dominance} `dominance`), và chỉ đặt `mask = 0, value = 0.0` ở những chiều xung đột (**{vad['conflict_all_masked_keys']}** khóa xung đột cả 3 chiều).",
        "",
        "### 4.3 VnEmoLex (`conflicts_vnemolex.tsv`)",
        f"- Tổng khóa single-token: **{src_stats['vnemolex']['usable_single_token_keys']}**, **{src_stats['vnemolex']['blank_cells_masked']}** ô trống chuyển thành `mask = 0, value = 0.0`.",
        f"- Rà soát toàn bộ **{src_stats['vnemolex']['duplicate_single_token_keys']}** khóa trùng sau chuẩn hoá: **{src_stats['vnemolex']['duplicate_identical_vector_keys']}** cặp biến thể dấu thanh có vector cảm xúc giống hệt nhau; **{src_stats['vnemolex']['duplicate_superset_or_merged_keys']}** cặp còn lại gồm `nhọt` (dòng 235 là tập cha của dòng 312) và 10 cặp biến thể chính tả (`hoà/hòa`, `huỷ/hủy`, `loè/lòe`). Với các cặp này, OR-merge lấy hợp nhãn trên tất cả biến thể; dòng gắn từ gốc tiếng Anh **không nhất thiết** là superset (có trường hợp dòng English toàn 0 và biến thể không-English mới mang nhãn, ví dụ `biến_hoá`, `hoà_đồng`).",
        "",
    ]


def _coverage(payload: dict[str, Any]) -> list[str]:
    stats = payload["stats"]
    cov_train = stats["coverage"]["train"]
    cov_dev = stats["coverage"]["dev"]
    base_train = stats["baseline_exclude_all_conflicts_coverage"]["train"]
    base_dev = stats["baseline_exclude_all_conflicts_coverage"]["dev"]
    train_den = cov_train["denominators"]
    dev_den = cov_dev["denominators"]
    lines = [
        "## 5. Thống kê tổng hợp & Coverage (`train.jsonl` / `dev.jsonl`)",
        "",
        f"- **Tổng số entries trong composite (`v{VERSION}`):** **{stats['total_entries']}** (so với baseline loại toàn bộ conflict: **{stats['baseline_total_entries']}** entries).",
        "- **Định nghĩa mẫu số word-token:** Loại bỏ các token chỉ gồm dấu câu/ký hiệu (`not any(ch.isalnum() for ch in token)`).",
        f"  - `train.jsonl`: `{_format_count(train_den['word_tokens'])}` word-tokens / `{_format_count(train_den['word_types'])}` word-types (tổng raw: `{_format_count(train_den['raw_tokens'])}` tokens / `{_format_count(train_den['raw_types'])}` types; loại `{_format_count(train_den['excluded_punctuation_tokens'])}` token dấu câu / `{_format_count(train_den['excluded_punctuation_types'])}` types).",
        f"  - `dev.jsonl`: `{_format_count(dev_den['word_tokens'])}` word-tokens / `{_format_count(dev_den['word_types'])}` word-types (tổng raw: `{_format_count(dev_den['raw_tokens'])}` tokens / `{_format_count(dev_den['raw_types'])}` types; loại `{_format_count(dev_den['excluded_punctuation_tokens'])}` token dấu câu / `{_format_count(dev_den['excluded_punctuation_types'])}` types).",
        "",
        "### 5.1 Coverage tổng hợp (Overall Coverage — có $\\ge 1$ chiều `mask = 1`)",
        "",
        "| Split | Chính sách | Word-Token Coverage | Word-Type Coverage | Raw-Token Coverage | Raw-Type Coverage |",
        "|---|---|---:|---:|---:|---:|",
        f"| `train` | Baseline (loại toàn bộ conflict, {stats['baseline_total_entries']} entries) | **{base_train['word_token_coverage']['token_coverage_pct']:.2f}%** ({base_train['word_token_coverage']['tokens_covered']}/{base_train['word_token_coverage']['tokens_total']}) | **{base_train['word_token_coverage']['type_coverage_pct']:.2f}%** ({base_train['word_token_coverage']['types_covered']}/{base_train['word_token_coverage']['types_total']}) | {base_train['raw_token_coverage']['token_coverage_pct']:.2f}% ({base_train['raw_token_coverage']['tokens_covered']}/{base_train['raw_token_coverage']['tokens_total']}) | {base_train['raw_token_coverage']['type_coverage_pct']:.2f}% ({base_train['raw_token_coverage']['types_covered']}/{base_train['raw_token_coverage']['types_total']}) |",
        f"| `train` | **Rebuild v{VERSION}** ({stats['total_entries']} entries) | **{cov_train['word_token_coverage']['token_coverage_pct']:.2f}%** ({cov_train['word_token_coverage']['tokens_covered']}/{cov_train['word_token_coverage']['tokens_total']}) | **{cov_train['word_token_coverage']['type_coverage_pct']:.2f}%** ({cov_train['word_token_coverage']['types_covered']}/{cov_train['word_token_coverage']['types_total']}) | {cov_train['raw_token_coverage']['token_coverage_pct']:.2f}% ({cov_train['raw_token_coverage']['tokens_covered']}/{cov_train['raw_token_coverage']['tokens_total']}) | {cov_train['raw_token_coverage']['type_coverage_pct']:.2f}% ({cov_train['raw_token_coverage']['types_covered']}/{cov_train['raw_token_coverage']['types_total']}) |",
        f"| `dev` | Baseline (loại toàn bộ conflict, {stats['baseline_total_entries']} entries) | **{base_dev['word_token_coverage']['token_coverage_pct']:.2f}%** ({base_dev['word_token_coverage']['tokens_covered']}/{base_dev['word_token_coverage']['tokens_total']}) | **{base_dev['word_token_coverage']['type_coverage_pct']:.2f}%** ({base_dev['word_token_coverage']['types_covered']}/{base_dev['word_token_coverage']['types_total']}) | {base_dev['raw_token_coverage']['token_coverage_pct']:.2f}% ({base_dev['raw_token_coverage']['tokens_covered']}/{base_dev['raw_token_coverage']['tokens_total']}) | {base_dev['raw_token_coverage']['type_coverage_pct']:.2f}% ({base_dev['raw_token_coverage']['types_covered']}/{base_dev['raw_token_coverage']['types_total']}) |",
        f"| `dev` | **Rebuild v{VERSION}** ({stats['total_entries']} entries) | **{cov_dev['word_token_coverage']['token_coverage_pct']:.2f}%** ({cov_dev['word_token_coverage']['tokens_covered']}/{cov_dev['word_token_coverage']['tokens_total']}) | **{cov_dev['word_token_coverage']['type_coverage_pct']:.2f}%** ({cov_dev['word_token_coverage']['types_covered']}/{cov_dev['word_token_coverage']['types_total']}) | {cov_dev['raw_token_coverage']['token_coverage_pct']:.2f}% ({cov_dev['raw_token_coverage']['tokens_covered']}/{cov_dev['raw_token_coverage']['tokens_total']}) | {cov_dev['raw_token_coverage']['type_coverage_pct']:.2f}% ({cov_dev['raw_token_coverage']['types_covered']}/{cov_dev['raw_token_coverage']['types_total']}) |",
        "",
        "### 5.2 Coverage theo từng chiều (Per-Dimension Coverage — Rebuild `v" + VERSION + "`)",
        "",
        "| Chiều (`feature`) | Active Entries | Train Word-Token % | Train Word-Type % | Dev Word-Token % | Dev Word-Type % |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for feature in FEATURES:
        active = stats["active_entries_per_dimension"][feature]
        train_dim = cov_train["per_dimension"][feature]
        dev_dim = cov_dev["per_dimension"][feature]
        lines.append(
            f"| `{feature}` | {active} | {train_dim['word_token_coverage_pct']:.2f}% ({train_dim['word_tokens_covered']}) | "
            f"{train_dim['word_type_coverage_pct']:.2f}% ({train_dim['word_types_covered']}) | "
            f"{dev_dim['word_token_coverage_pct']:.2f}% ({dev_dim['word_tokens_covered']}) | "
            f"{dev_dim['word_type_coverage_pct']:.2f}% ({dev_dim['word_types_covered']}) |"
        )
    return lines


def _spot_checks() -> list[str]:
    return [
        "",
        "## 6. Spot Checks đã xác minh",
        "",
        "- `an_ủi`: `polarity = 0.625`, `mask[0] = 1`.",
        "- `tốt`: `polarity = 0.875` (`v#02748627`, `mask[0] = 1`), 8 chiều emotion `mask = 1` (`joy=1, trust=1, anticipation=1`).",
        "- `này`: 8 chiều emotion `values = 0.0` đi kèm `mask = 1` (dữ liệu thật = không có cảm xúc, phân biệt với `mask = 0`).",
        "- `xyz_oov_test` (OOV): không tồn tại trong `entries` (runtime loader trả về `values = 0.0, mask = 0`).",
        "",
    ]


def _format_count(value: int) -> str:
    return f"{value:,}".replace(",", ".")