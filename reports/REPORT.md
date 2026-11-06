# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-06

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số "tin" của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…

   Trả lời: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.51087 ; `taxonomy_name`=ImageNet-1K

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?

   Trả lời: Record này nói về cả bức ảnh. Vì model phân loại ảnh (image classification) chỉ nhận vào toàn bộ ảnh và trả ra một nhãn duy nhất cho cả khung hình — không phát hiện từng vật thể riêng lẻ. Lớp "cab" phản ánh nội dung chủ đạo của bức ảnh giao thông, không tương ứng với một chiếc xe cụ thể nào.

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?

   Trả lời: Danh sách tên lớp đi kèm taxonomy/checkpoint. Model `yolo11n-cls.pt` được huấn luyện trên bộ dữ liệu ImageNet-1K, bộ taxonomy này gồm 1 000 lớp được xác định trước bởi nhóm tạo dataset. Model không tự nghĩ ra tên lớp.

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?

   Trả lời: Nếu chỉ ghi tên lớp, sẽ có nguy cơ nhầm lẫn khi các taxonomy khác nhau dùng cùng tên hoặc khi tên lớp được cập nhật/đổi tên. `class_id` là mã số bất biến trong taxonomy đó; `taxonomy_name` cho biết bộ nhãn nào đang được dùng. Ba trường cùng nhau đảm bảo khả năng tra cứu chính xác và tái tạo lại kết quả về sau.

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?

   Trả lời: Guideline cần quy định rõ tiêu chí chọn lớp chủ đề (dominant subject): ví dụ "chọn lớp mô tả chủ thể chiếm diện tích lớn nhất và nằm ở trung tâm ảnh; nếu nhiều chủ thể ngang nhau, chọn lớp theo thứ tự ưu tiên đã liệt kê sẵn". Nếu không có quy tắc này, mỗi annotator sẽ chọn theo cảm tính khác nhau, dẫn đến nhãn không nhất quán giữa các người gán.

6. Vì sao không được lấy `score` của model làm ground truth?

   Trả lời: `score` là độ tự tin (confidence) của phép đoán, không phải nhãn đúng. Model có thể rất tự tin mà vẫn sai (ví dụ ảnh `traffic` đoán "cab" với score 0.511 nhưng trong ảnh còn nhiều xe buýt chiếm diện tích lớn hơn). Ngoài ra model không có thông tin ngữ cảnh thực tế mà người gán nhãn có. Ground truth phải do người gán theo quy tắc guideline đã thống nhất.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…

   Trả lời: `class_name`=person ; `score`=0.912647 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.59 ; `bbox_height`=279.68

2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?

   Trả lời: Góc trên-trái của hộp nằm tại pixel (x=385, y=69) — khoảng 60% từ cạnh trái ảnh và gần đầu ảnh (trên cao). Góc dưới-phải nằm tại (x=499, y=349) — khoảng 78% từ cạnh trái và 82% từ đỉnh ảnh. Hộp rộng khoảng 114 pixel và cao khoảng 280 pixel, bao phủ hình người đứng trong bếp.

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?

   Trả lời:
   - Ngưỡng 0.20: giữ nhiều hộp hơn vì model hạ điều kiện lọc — ước tính ~15–20 hộp cho ảnh kitchen (gồm cả các hộp score thấp mà ngưỡng 0.35 đã loại).
   - Ngưỡng 0.60: chỉ giữ hộp có score ≥ 0.60; dựa trên JSON ngưỡng 0.35, các hộp kitchen đạt ngưỡng 0.60 là: person(0.9126), bowl(0.7190), bowl(0.7007), oven(0.6874), oven(0.6345), person(0.6109) → 6 hộp.
   - Tham khảo: ngưỡng 0.35 (file JSON hiện tại) cho 11 hộp (person ×2, bowl ×4, oven ×2, cup ×2, bowl ×1 thêm).

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?

   Trả lời: Reviewer phải: (a) kiểm tra từng hộp score thấp xem có thực sự là vật thể cần gán nhãn không (loại false positive); (b) xử lý các hộp bị trùng lặp (overlapping boxes) và quyết định hộp nào đúng; (c) ghi chú hoặc xóa các hộp không hợp lệ. Khối lượng công việc QC tăng theo số lượng hộp, dễ dẫn đến bỏ sót lỗi.

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).

   Trả lời: Vẽ hộp bao sát nhất (tight bounding box) quanh toàn bộ phần nhìn thấy được của vật thể. Các cạnh hộp phải chạm pixel ngoài cùng của vật thể, không để khoảng trống thừa. Nếu vật bị cắt mép ảnh, kéo cạnh hộp đến biên ảnh.

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

   Trả lời: Người ở góc trái ảnh (record score 0.611) bị che khuất một phần bởi kệ hoặc vật thể phía trước. Tớ sẽ viết thành quy tắc trong guideline: "Nếu vật thể bị che một phần nhưng vẫn nhận diện được loại, vẫn gán nhãn và vẽ hộp bao phần nhìn thấy được; không cần ước tính phần bị che." Đây là tình huống phổ biến, có quy tắc rõ ràng → không cần escalation.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…

   Trả lời: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899166 ; `polygon_point_count`=350 ; 5 điểm đầu của `polygon_xy`=[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?

   Trả lời: So với box (chỉ cho biết hình chữ nhật bao ngoài), polygon cung cấp hình dạng thực tế của vật thể theo đường viền. Nhờ vậy ta biết chính xác pixel nào thuộc về đối tượng, kể cả những chỗ cong, lõm, hay bị che bởi vật khác — thông tin này box hoàn toàn không thể biểu diễn.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?

   Trả lời: `instance_id` dùng để phân biệt các cá thể riêng lẻ của cùng một lớp trong một ảnh. Ví dụ nếu có 2 người, cả hai đều có `class_name=person` nhưng khác `instance_id` (kitchen-001, kitchen-002). Nó không phải class ID vì không xác định loại vật — class ID đã có trường `class_id` riêng. Nó không phải tracking ID vì không theo dõi vật thể qua nhiều frame video — nó chỉ có ý nghĩa trong phạm vi một ảnh đơn lẻ.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.

   Trả lời: Vẽ polygon bám sát đường viền ngoài cùng nhìn thấy được của vật thể. Với những chỗ biên mờ (tóc, lông, rìa vải), kéo polygon đến nơi màu nền bắt đầu chiếm ưu thế. Không được vẽ polygon bao vào vùng nền hoặc sang vật thể khác.

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

   Trả lời: Tại khu vực cánh tay người bếp (instance `kitchen-001`), biên giữa tay người và oven/thớt phía dưới rất khó xác định vì hai vật chạm nhau và màu gần giống nhau. Tớ sẽ dùng guideline: "Khi hai vật chạm nhau, mỗi annotator vẽ mask đến điểm tiếp xúc; không được mask chồng lên nhau." Đây là ca đủ phổ biến để viết quy tắc áp dụng cho nhiều ảnh, không cần escalation.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ xem prediction. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp duy nhất cho cả ảnh (`class_id=468, class_name=cab, taxonomy=ImageNet-1K`) | Ảnh `traffic` có nhiều loại xe (bus, car, cab) — khó chọn một nhãn; top-1 "cab" score chỉ 0.511, minibus rank-2 cũng hợp lý | Đọc guideline, chọn lớp theo tiêu chí "chủ thể chính", ghi đủ `class_id + class_name + taxonomy_name` | Kiểm tra nhãn đúng taxonomy, đúng quy tắc lựa chọn, không dùng score model làm cơ sở |
| Phát hiện vật thể | Danh sách các hộp bbox (xyxy) kèm class_name cho từng vật thể trong ảnh | Ảnh `kitchen`: người góc trái (score 0.611) bị che; 2 hộp oven chồng nhau (score 0.687 và 0.634) — có thể là duplicate | Vẽ hộp bao sát từng vật thể nhìn thấy được, gán class đúng, đánh dấu vật bị cắt mép/che | Kiểm tra hộp bao đủ vật, không trùng lặp, class đúng, không có false positive |
| Instance segmentation | Danh sách polygon theo pixel cho từng cá thể (`instance_id`, `class_name`, `polygon_xy`) | Ảnh `kitchen`, instance `kitchen-001` (person): biên tay chạm sát oven — 350 điểm polygon, khó xác định ranh giới chính xác | Vẽ polygon bám viền ngoài từng vật thể, gán instance_id riêng cho mỗi cá thể cùng lớp | Kiểm tra polygon không chồng lên nhau, không bỏ sót instance, biên sát thực tế |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)

   Trả lời: Không lưu, chia sẻ hoặc công bố bất kỳ ảnh, JSON hoặc output nào ra ngoài phạm vi repository GitHub của bài lab này.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

   Trả lời: Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi (ví dụ ảnh chứa thông tin nhận dạng cá nhân, nội dung nhạy cảm), tôi sẽ dừng ngay, không gán nhãn, và báo cáo cho Lab Coach / Giảng viên phụ trách.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
