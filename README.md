# Vietnamese sentiment foundation

## Trạng thái hiện tại

Đã reset bản nháp source/test cũ và dựng foundation theo từng miền trách nhiệm (Giai đoạn A). Package hiện chỉ có các ranh giới thư mục và bộ kiểm thử kiến trúc; chưa có model, lexicon loader, alignment, training hoặc inference mới.

Các API cũ như `SentimentModel`, `load_lexicon`, `normalize_vietnamese` và `train_one_epoch` không còn khả dụng. Backup source/test trước reset nằm trong `artifacts/backups/`, có SHA-256 manifest được xác minh trước khi xóa. Không import code từ backup vào package mới.

## Môi trường và kiểm thử

Python >= 3.12, dùng Conda `nlpdevelop`; không tạo `.venv` trong workspace. Với máy hiện tại:

```bash
"D:/anaconda/envs/nlpdevelop/python.exe" -m pip install "pytest>=8.0"
"D:/anaconda/envs/nlpdevelop/python.exe" -m pytest tests/ -q
"D:/anaconda/envs/nlpdevelop/python.exe" -m pytest tests/architecture/ -q
"D:/anaconda/envs/nlpdevelop/python.exe" -m pytest tests/architecture/test_dependency_direction.py -q
```

Sau khi `conda activate nlpdevelop`, có thể dùng `python -m pytest tests/ -q`. Pytest thêm `src` vào import path qua `pyproject.toml`; không cần editable install. Foundation tests không cần torch, transformers, mạng hoặc pretrained weights. Runtime dependencies vẫn được khai báo để phục vụ Giai đoạn B; chưa cài hoặc tải mô hình trong đợt reset.

## Cấu trúc đã tạo

| Thư mục | Trách nhiệm dành cho các tính năng tiếp theo |
| --- | --- |
| `src/vnsentiment/contracts/` | Mỗi data contract một file |
| `src/vnsentiment/lexicon/` | Tách validation, loader và lookup store |
| `src/vnsentiment/alignment/` | Tách word alignment, truncation và lexical broadcast |
| `src/vnsentiment/data/` | Tách JSONL reader, preparation, padding và collator |
| `src/vnsentiment/backbones/` | Tách loader tokenizer và encoder |
| `src/vnsentiment/models/fusion/` | Mỗi projection, gate và fusion mode một file |
| `src/vnsentiment/models/` | Classifier điều phối; pooling độc lập |
| `src/vnsentiment/training/` | Tách move batch, train step và train epoch |
| `src/vnsentiment/evaluation/` | Mỗi metric và evaluate loop một file |
| `src/vnsentiment/diagnostics/` | Inspect alignment và metadata batch |
| `tests/architecture/` | Kiểm tra import, trách nhiệm, kích thước và dependency |
| `tests/helpers/` | Vị trí cho fake encoder/tokenizer và fixture builders |
| `tests/unit/` | Test phản chiếu từng module sản phẩm |
| `tests/integration/` | Vị trí cho pipeline offline và smoke opt-in về sau |

Các thư mục nghiệp vụ hiện chỉ chứa `__init__.py` có docstring, không có implementation giả. Danh sách tên file/API dự kiến và roadmap Giai đoạn B nằm trong [plan đã duyệt](.hermes/plans/2026-10-12_094933-reset-source-modular-debuggable-structure.md).

## Quy tắc giữ source dễ debug

- Một file có tối đa một public function/class chính và hai top-level private helpers; methods cùng trách nhiệm ở cùng class.
- Guardrail 200 dòng/file, 50 dòng/function hoặc method. Không chia vụn wrapper vô nghĩa hoặc nhồi statements để lách giới hạn.
- Không tạo file gom chung `utils.py`, `common.py`, `types.py`, `errors.py`, `validator.py`, `trainer.py` hoặc tương đương.
- `__init__.py` chỉ chứa docstring: không eager imports, I/O, tải tokenizer/model.
- Model không import data/training/evaluation. Dependency direction được kiểm tra bằng AST; dynamic imports và semantics vẫn cần review thủ công.
- Lỗi architecture test chỉ rõ file hoặc cạnh import vi phạm; chạy riêng test module liên quan trước khi chạy toàn bộ suite.

## Ràng buộc nghiên cứu cho Giai đoạn B

Input phải là tiếng Việt đã word-segmented; normalization không tự segmentation. Lexicon ngoài có polarity, VAD và tám chiều emotion, phân biệt `null` với true zero. Thiết kế đích là token-level fusion trước content-only pooling, bốn chế độ contextual/concat/residual/gated; chưa được triển khai.

Không có kết quả accuracy/F1 mới, dataset/lexicon giả hoặc cam kết gate phát hiện ca khó. Fixtures tương lai chỉ kiểm thử phần mềm. Không mở rộng lexicon trên dev/test; hard-case tags chỉ dùng evaluation.

## Tài liệu và tài nguyên được giữ

- `docs/`, bao gồm báo cáo Word và `docs/Part2/figures/draw_architecture.py`, không bị reset.
- `data/`, `configs/`, cache/run cũ và `.gitignore` được giữ nguyên.
- Dashboard và báo cáo bàn giao: [docs_obsidian/00-INDEX.md](docs_obsidian/00-INDEX.md).
- Không có Git repository; không tự init, commit hoặc push.
- AGENTS.md và CLAUDE.md chưa được đồng bộ với reset: quyền ghi AGENTS.md hết hạn, lượt ghi CLAUDE.md bị từ chối bởi partial-read guard. Các mô tả API/test cũ trong hai file đó là lịch sử, không phải tính năng hiện có. Xem dashboard/session mới để biết trạng thái thực tế.
