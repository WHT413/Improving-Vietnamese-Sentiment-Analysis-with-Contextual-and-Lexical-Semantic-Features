# Rà soát sơ đồ kiến trúc Part 2

Đã đọc toàn bộ nội dung ba báo cáo hiện có: Part 1, Part 2 `need_to_fix` và Part 2 `REVISED`; đối chiếu đoạn Python đính kèm và ảnh kiến trúc nhúng trong hai bản Part 2. Workspace hiện chỉ có tài liệu, chưa có mã mô hình, dữ liệu hay kết quả huấn luyện. Vì vậy, các nhận xét về mô hình bên dưới là kiểm tra tính nhất quán của đề xuất, không phải xác nhận hiệu quả thực nghiệm. Chưa kiểm tra bố cục toàn trang Word vì renderer thiếu LibreOffice.

## Mạch dự án

Part 1 xác định nhu cầu kết hợp PhoBERT và tri thức từ vựng, với ba nội dung chính: fusion có chọn lọc; giảm OOV bằng chuẩn hóa và mở rộng lexicon có đo token/type coverage; bổ sung tín hiệu cảm xúc đa chiều và kiểm chứng nhiều miền dữ liệu. Part 2 hệ thống hóa thành GAP 1–4, đề xuất hai nhánh song song cùng cổng học được. Report 3 sẽ chốt cách triển khai; Report 4 sẽ kiểm chứng bằng thực nghiệm.

Cốt lõi kiến trúc là phép cộng dư:

```text
lexical projection: l_tilde_i = W_proj l_i + b_proj
gate:               g_i = sigmoid(W_g [h_i ; l_tilde_i] + b_g)
fusion:             f_i = h_i + g_i * l_tilde_i
classification:     z = Pool(f_1, ..., f_n)
                    P(y|z) = softmax(W_c z + b_c)
```

PhoBERT cung cấp đường biểu diễn ngữ cảnh trực tiếp. Gate điều tiết phần bổ sung từ lexicon; công thức không phải phép trung bình có trọng số giữa hai nhánh.

## Vấn đề của hình cũ

- Chữ phần lớn 5–7 pt trên canvas 9 × 11 inch. Trong Word, ảnh được đặt rộng khoảng 5,4 inch, khiến nhiều nhãn chỉ còn tương đương 3–4 pt. DPI cao không khắc phục được chữ nhỏ.
- Nhánh PhoBERT có khối tô đậm lớn, nhiều khoảng trống; nhánh lexicon có nhiều khối nhỏ và viền đứt. Độ tương phản và mật độ chữ mất cân đối.
- Đường chia nhánh từ Preprocessing bị lệch; đường h_i có đoạn dừng rồi tiếp tục, gây cảm giác luồng bị đứt.
- Hình nhúng trong REVISED thiếu đường l_tilde_i đi trực tiếp vào Fusion. Fusion cần ba đầu vào h_i, g_i và l_tilde_i.
- Các badge BACKBONE, PHỤ TRỢ, chú giải GAP và ghi chú gate nằm rải rác, làm hình giống slide thuyết trình hơn một figure học thuật.
- Các khẳng định như gate đóng khi OOV hoặc mở 0,7–1 tại ca khó khiến người đọc dễ hiểu đó là kết quả đã được chứng minh.

## Lỗi trong đoạn Python được gửi

Đoạn đính kèm không chạy nguyên trạng. Có thể một phần lỗi xuất hiện trong quá trình sao chép; ảnh trong báo cáo rõ ràng đã được tạo bằng một phiên bản khác.

| Vị trí | Vấn đề |
| --- | --- |
| `NAVY = "#00366"`, `TEAL = "#06B6B"`, `"#FF3E0"` | Mã màu hex 5 ký tự không hợp lệ. |
| `text_color = ... else '` | Chuỗi chưa đóng. |
| `arowstyle='->'` | Sai tên tham số; phải là `arrowstyle`. |
| `color=GRAY style='italic'` | Thiếu dấu phẩy. |
| `box(..., NAVY, '', 7, ...)` | Thiếu/đảo tham số `edgecolor`, `text`, `fontsize`. |
| `..., 1.4 shrink_a=3, ...` | Thiếu dấu phẩy. |
| `arrow_straight(5.1, 5.3.1, 4.7, ...)` | Literal số sai và thiếu tọa độ. |
| Lệnh vẽ Classifier | Thiếu `edgecolor` trong chữ ký hàm `box`. |
| `/tmp/diagram_final.jpg` | Đường dẫn gắn cứng, không phù hợp để chạy trực tiếp trên Windows. |

## Những điểm kiến trúc cần chốt

### 1. Kích thước vector lexicon

Mục 3.3 vừa mô tả polarity, ba chiều VAD và tám emotion cues, vừa ghi k ≈ 8. Nếu sử dụng tất cả các chiều đó, số chiều là 1 + 3 + 8 = 12, chưa kể chỉ báo thiếu dữ liệu hay confidence. Nếu chỉ chọn tám chiều, cần liệt kê chính xác tám chiều nào. Figure mới dùng k để tránh chốt thay phần triển khai.

### 2. Căn chỉnh vị trí token

Ngoài chiếu k → d, cần căn chỉnh vị trí từ tra lexicon và vị trí subword đầu vào PhoBERT. Hai loại căn chỉnh này khác nhau. Có thể đưa lexical features của một từ về các subword của từ đó, hoặc gộp contextual subwords về cấp từ trước fusion. Cần chốt cách làm, xử lý special tokens, padding và truncation trong Report 3. Figure mới thêm nhãn “Căn chỉnh vị trí từ / subword” như một bước thiết kế cần hoàn thiện, chưa chọn thuật toán cụ thể.

### 3. OOV không tự động đồng nghĩa với fallback

Nếu l_i = 0 nhưng tầng chiếu có bias, thì l_tilde_i = b_proj, không nhất thiết bằng 0. Sigmoid cũng không bảo đảm học được g_i ≈ 0 cho mọi OOV. Do đó, không nên khẳng định công thức hiện tại bảo đảm f_i = h_i khi thiếu lexicon.

Nếu cần bảo đảm này bằng thiết kế, có thể bổ sung chỉ báo m_i ∈ {0,1}:

```text
l_tilde_i = m_i (W_proj l_i + b_proj)
```

Khi không có mục từ, m_i = 0 nên phần bổ sung bằng 0. Đây là đề xuất sửa kiến trúc, CHƯA đưa vào công thức figure mới. Với dữ liệu thiếu từng nhóm VAD/emotion, nên cân nhắc mask ở cấp đặc trưng thay vì một mask chung.

### 4. Gate điều tiết, không tự chứng minh đảo cực

g_i nằm trong (0,1) nên phép nhân gate không tự đổi dấu l_tilde_i. Tầng chiếu có thể học các giá trị có dấu và toàn mô hình có thể học tương tác ngữ cảnh, nhưng “mở gate” không đồng nghĩa với “đảo cực đúng khi phủ định”. Cần mô tả đây là giả thuyết và kiểm chứng qua ablation, phân tích lỗi phủ định, mỉa mai và OOV.

Gate trong công thức là vector d chiều cho mỗi token, không phải một phần trăm duy nhất. Nếu trình bày một giá trị gate/token, phải định nghĩa cách tổng hợp, ví dụ mean(g_i). Giá trị gate lớn cũng chưa đủ chứng minh đóng góp lexical lớn, vì đóng góp còn phụ thuộc độ lớn và hướng của l_tilde_i.

### 5. Giá trị minh họa không phải kết quả mô hình

Các giá trị 0,85; 0,95; P(Neg) ≈ 0,62 và điểm arousal ví dụ trong Mục 4 chưa có kết quả chạy hay nguồn từng giá trị trong workspace. Cần đánh dấu là giả định minh họa, hoặc thay bằng mô tả định tính. Đặc biệt, gate tại token “không” chỉ tác động lên lexical feature của chính vị trí đó; không trực tiếp nhân với vector từ “tệ” ở vị trí bên cạnh.

### 6. So sánh với concat cần trung tính hơn

Concat không yêu cầu hai vector có cùng số chiều. Tầng chiếu cần thiết ở đây để thực hiện phép cộng dư và Hadamard product; không nên nói concat không thể thực hiện vì 768 và k khác nhau. Số chiều lexical nhỏ cũng không tự chứng minh tín hiệu bị chìm.

FFN có tham số dùng chung nhưng đầu ra vẫn phụ thuộc đầu vào. Do đó, “concat xử lý mọi token y hệt” hoặc “nối zero luôn gây nhiễu” là nhận định quá mạnh. Điểm khác biệt rõ của đề xuất là có cơ chế điều tiết lexical tường minh theo vị trí và chiều; mức cải thiện phải được đo.

### 7. Nội dung cần đồng bộ giữa các mục

- Mục 3.1 ghi Segmentation → Normalize → Regex, còn ví dụ Mục 4 ghi Normalize → Tokenize. Nên thống nhất thứ tự. Figure mới đề xuất chuẩn hóa/rút gọn trước tách từ để bộ tách từ nhận văn bản ít nhiễu hơn.
- Danh sách tham khảo CombViSA giữa Part 1 và Part 2 không khớp: Part 1 ghi 2025/arXiv:2501.08758, Part 2 ghi Vo & Collier 2023/VLSP. Cần xác minh nguồn gốc trước nộp; lần rà soát này chưa xác minh bibliographic metadata bên ngoài.
- Nếu kết hợp VAD và emotion cues có thang đo khác nhau, cần xác định cách scale và biểu diễn thiếu dữ liệu trong Report 3.
- Lexicon Expansion là bước xây dựng tài nguyên trước Lookup. Cross-domain evaluation và coverage là luồng đánh giá, không phải các tầng suy luận sau Softmax. Nên trình bày chúng bằng figure/bảng riêng.
- Mở rộng lexicon, chọn ngưỡng cosine và chuẩn hóa thủ công cần có quy trình tách train/dev/test để kết quả đánh giá không bị rò rỉ.

## Figure mới

`draw_architecture.py` xuất `two_branch_adaptive_fusion.png` ở 300 dpi và `two_branch_adaptive_fusion.svg` dạng vector. Code có kiểm tra kích thước chữ sau khi render và đã chạy thành công. Hình dùng hai nhánh đều nhau, màu nhạt, đường nối vuông góc và đầy đủ các đầu vào Fusion; bỏ các ghi chú gate chưa kiểm chứng. Aggregation và classifier được nhóm vào một khối cuối để figure gọn; vector câu z vẫn được ghi rõ.

Ảnh được thiết kế rộng 7,2 inch, với chữ chủ yếu 9–13 pt. Khi chèn rộng khoảng 5,4 inch, phần chữ chính còn khoảng 7–10 pt. Có thể dùng SVG nếu Word hỗ trợ để giữ nét khi phóng to; PNG là bản tương thích thuận tiện.

Caption gợi ý:

> Hình 1. Kiến trúc Two-Branch Adaptive Fusion đề xuất cho phân tích cảm xúc tiếng Việt. Nhánh PhoBERT trích xuất biểu diễn ngữ cảnh; nhánh lexicon tạo đặc trưng cảm xúc và chiếu về cùng số chiều. Sau khi thống nhất vị trí token giữa hai nhánh, cổng học được điều tiết phần bổ sung từ vựng vào biểu diễn ngữ cảnh thông qua phép cộng dư. Biểu diễn dung hợp được tổng hợp ở cấp câu và đưa qua bộ phân loại ba nhãn.

Các tài liệu Word gốc được giữ nguyên. Hình mới là bản đề xuất để xem và chèn vào báo cáo, không phải xác nhận mô hình đã triển khai hay đạt kết quả thực nghiệm.
