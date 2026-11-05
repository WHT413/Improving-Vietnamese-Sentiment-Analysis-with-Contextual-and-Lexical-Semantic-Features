# Hướng dẫn tải nhanh PhoBERT-base

Tài liệu này hướng dẫn cách tải **nhanh nhất** model [`vinai/phobert-base`](https://huggingface.co/vinai/phobert-base) từ Hugging Face cho dự án, chỉ lấy các file PyTorch cần thiết (không tải Flax/TensorFlow).

## Thông tin model

| Thuộc tính | Giá trị |
|---|---|
| Model ID | `vinai/phobert-base` |
| Revision nên pin | `01daacda68afe13d83023d16ec647239e344a1e6` |
| License | MIT |
| Kiến trúc | RoBERTa-base, `hidden_size=768`, `max_position_embeddings=258` |
| Token đặc biệt | `<s>`=0, `</s>`=2, `<pad>`=1 |
| Dung lượng cần tải | ≈ 521 MB (`pytorch_model.bin` + tokenizer + config) |

> Không dùng fast tokenizer: `AutoTokenizer` mặc định có thể chọn fast tokenizer vì repo có `tokenizer.json`, nhưng fast tokenizer làm hỏng ánh xạ word→subword của dự án. Luôn truyền `use_fast=False`.

## Cách nhanh nhất (khuyến nghị)

Cài `hf_transfer` để tăng tốc tải, sau đó chỉ định đúng 4 file cần thiết.

### PowerShell (Windows)

```powershell
conda activate nlpdevelop
pip install -U "huggingface_hub[cli]" hf_transfer

$env:HF_HUB_ENABLE_HF_TRANSFER = "1"
$env:HF_HOME = "$PWD\artifacts\hf-cache"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

$REV = "01daacda68afe13d83023d16ec647239e344a1e6"
hf download vinai/phobert-base `
  --revision $REV `
  --local-dir "$env:HF_HOME\phobert-base" `
  --include "config.json" "vocab.txt" "bpe.codes" "pytorch_model.bin"
```

### Git Bash

```bash
conda activate nlpdevelop
pip install -U "huggingface_hub[cli]" hf_transfer

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME="$PWD/artifacts/hf-cache"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

REV=01daacda68afe13d83023d16ec647239e344a1e6
hf download vinai/phobert-base \
  --revision "$REV" \
  --local-dir "$HF_HOME/phobert-base" \
  --include "config.json" "vocab.txt" "bpe.codes" "pytorch_model.bin"
```

Kết quả mong đợi:

```text
artifacts/hf-cache/phobert-base/
├── config.json
├── vocab.txt
├── bpe.codes
└── pytorch_model.bin
```

Ghi chú:

- `hf_transfer` dùng tải song song bằng Rust, nhanh hơn đáng kể so với HTTP thường.
- Nếu CLI cũ chưa có lệnh `hf`, dùng `huggingface-cli download ...` với cùng tham số.
- Có thể đặt `HF_TOKEN` (token read-only) để tránh giới hạn tốc độ khi tải nhiều lần:
  ```powershell
  $env:HF_TOKEN = "hf_xxxxxxxx"
  ```
- Nếu mạng bị ngắt, chạy lại đúng lệnh trên; Hugging Face sẽ tải tiếp file dở dang.

## Cách 2 — Python `snapshot_download`

```python
import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HOME"] = "artifacts/hf-cache"

from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="vinai/phobert-base",
    revision="01daacda68afe13d83023d16ec647239e344a1e6",
    local_dir="artifacts/hf-cache/phobert-base",
    allow_patterns=[
        "config.json",
        "vocab.txt",
        "bpe.codes",
        "pytorch_model.bin",
    ],
    max_workers=8,
)
print(path)
```

`max_workers` mặc định là 8; chỉ tăng nếu đường truyền đủ tốt.

## Cách 3 — Để `transformers` tự tải khi load

```python
from transformers import AutoTokenizer, AutoModel

repo = "vinai/phobert-base"
rev = "01daacda68afe13d83023d16ec647239e344a1e6"

tokenizer = AutoTokenizer.from_pretrained(repo, revision=rev, use_fast=False)
model = AutoModel.from_pretrained(repo, revision=rev)
```

Cách này tự tải đúng file cần thiết vào cache, nhưng khó kiểm soát tiến độ và không nhanh bằng `hf_transfer` + `hf download`.

## Kiểm tra sau khi tải

### 1. Kiểm tra checksum

| File | SHA-256 |
|---|---|
| `pytorch_model.bin` | `a0b0f0912c710147fbaac015b0a4011216a0061a56c03b840b639e40d3bb49cc` |
| `config.json` | `71df79a67a65b1ee472938d0dff77297c0d77e20fef1af41e0013b1d384efb0d` |
| `vocab.txt` | `dffd7543c401fb6567253ae7c3313004532c33d062525250049c616544bad777` |
| `bpe.codes` | `9dfe53fb8abbddd3387da99641c36e00bee757a3679aad9cf0a7cdc4b30f6ef8` |

PowerShell:

```powershell
Get-FileHash .\artifacts\hf-cache\phobert-base\pytorch_model.bin -Algorithm SHA256
```

Git Bash:

```bash
sha256sum artifacts/hf-cache/phobert-base/pytorch_model.bin
```

### 2. Load offline và kiểm tra token đặc biệt

```bash
python -c "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained(r'artifacts\hf-cache\phobert-base', use_fast=False, local_files_only=True); print(type(t).__name__); print(t.convert_tokens_to_ids(['<s>', '</s>', '<pad>']))"
```

Kết quả đúng:

```text
PhobertTokenizer
[0, 2, 1]
```

Kiểm tra model:

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "artifacts/hf-cache/phobert-base",
    local_files_only=True,
)
print(model.config.hidden_size)  # 768
```

## Lưu ý cho dự án

- Văn bản đầu vào phải được tách từ trước, ví dụ `sản_phẩm này không tệ`; UIT-VSFC tải từ nguồn gốc là raw, chưa tách từ.
- Giới hạn 256 token nội dung + 2 token đặc biệt (`<s>`, `</s>`).
- Không commit weights vào git. Cache nằm trong `artifacts/` và đã được `.gitignore` bỏ qua.
- Khi chia sẻ kết quả, ghi rõ revision `01daacd...` và checksum trong `README.source.md` để tái lập.
- Model public, không cần đăng nhập; token chỉ cần khi bị giới hạn tốc độ.

## Xử lý sự cố nhanh

| Hiện tượng | Cách xử lý |
|---|---|
| `hf: command not found` | Nâng cấp `huggingface_hub[cli]`, hoặc dùng `huggingface-cli download` |
| Cảnh báo symlink trên Windows | Đặt `HF_HUB_DISABLE_SYMLINKS_WARNING=1` |
| Tải chậm | Cài `hf_transfer`, đặt `HF_HUB_ENABLE_HF_TRANSFER=1`, dùng `HF_TOKEN` |
| Lỗi 401/403 | Chạy `hf auth login` hoặc đặt `HF_TOKEN` |
| Tokenizer không nhận diện | Nâng cấp `transformers` lên bản mới; không dùng `trust_remote_code` nếu không cần |
| Muốn tải lại sạch | Xoá `artifacts/hf-cache/phobert-base` rồi chạy lại lệnh |

## Citation

```bibtex
@inproceedings{nguyen2020phobert,
  title     = {PhoBERT: Pre-trained language models for Vietnamese},
  author    = {Dat Quoc Nguyen and Anh Tuan Nguyen},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2020},
  year      = {2020}
}
```
