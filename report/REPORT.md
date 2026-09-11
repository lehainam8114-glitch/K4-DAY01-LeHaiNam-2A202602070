# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`; hình minh họa: `visuals/classification_top5.png`.

**Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):**

```json
{
  "sample_id": "traffic",
  "coco_image_id": 210273,
  "task": "image_classification",
  "taxonomy_name": "ImageNet-1K",
  "model_file": "yolo11n-cls.pt",
  "rank": 1,
  "class_id": 468,
  "class_name": "cab",
  "score": 0.510915
}
```

**Record này mô tả toàn ảnh như thế nào?**

Checkpoint `yolo11n-cls.pt` là mô hình phân loại toàn ảnh (*image-level classification*): nó không khoanh vùng từng vật thể mà gán **một nhãn duy nhất** cho toàn bộ khung hình. Với ảnh `traffic` (cảnh đường phố đông xe), model cho nhãn hạng 1 là `cab` (taxi/xe khách nhỏ) với score 0.511. Nhìn vào `visuals/classification_top5.png`, ảnh thực tế chứa nhiều xe buýt xanh lớn và ô tô các loại – model nhận thấy đặc trưng thị giác gần nhất với lớp `cab` trong ImageNet-1K chiếm ưu thế toàn ảnh, dù xe buýt và xe minibus cũng xuất hiện ở hạng 2–3. Đây là **quyết định cấp ảnh**, không phải mô tả từng vật thể riêng lẻ.

**Ai định nghĩa class list mà checkpoint có thể dự đoán?**

Class list do **ban tổ chức cuộc thi ImageNet (Stanford / Princeton)** xây dựng và cố định từ dataset ImageNet Large Scale Visual Recognition Challenge (ILSVRC). 1 000 lớp được chọn dựa trên WordNet synset, ánh xạ qua `taxonomy_name = "ImageNet-1K"`. Sau đó Ultralytics đóng gói danh sách này vào checkpoint `yolo11n-cls.pt`. Người dùng cuối **không thể thêm hay bớt lớp** mà không huấn luyện lại; muốn dự đoán nhãn ngoài 1 000 lớp đó thì phải dùng taxonomy và checkpoint khác.

**Vì sao cần giữ cả ID, tên lớp và tên taxonomy?**

| Trường | Vai trò | Hậu quả nếu thiếu |
|---|---|---|
| `class_id` (468) | Định danh số không thay đổi theo phiên bản | Nếu chỉ lưu tên, khi taxonomy đổi tên lớp thì không tra cứu được |
| `class_name` ("cab") | Đọc được bởi con người, dùng khi viết guideline và kiểm tra | Nếu chỉ lưu ID, reviewer không biết nhãn nghĩa là gì mà không tra bảng |
| `taxonomy_name` ("ImageNet-1K") | Xác định bộ nhãn nào đang dùng | Nếu thiếu, cùng `class_id = 468` có thể ánh xạ sang lớp hoàn toàn khác trong taxonomy khác (COCO, Open Images…) |

Ba trường cần đi cùng nhau để **đảm bảo tái tạo và kiểm toán** (reproducibility & auditability): bất kỳ ai đọc JSON cũng có thể xác minh chính xác nhãn nào được dự đoán, theo hệ thống nào, không phụ thuộc vào phiên bản thư viện hay bộ nhớ của người chạy.

**Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?**

Vì classification cấp ảnh chỉ cho phép **một nhãn duy nhất**, guideline phải làm rõ:

1. **Quy tắc chủ thể chính**: annotator chọn lớp của vật thể **nổi bật nhất / chiếm diện tích lớn nhất / nằm trung tâm** (phải chọn đúng một trong ba hoặc ưu tiên kết hợp theo thứ tự rõ ràng).
2. **Ngưỡng độ phủ tối thiểu**: ví dụ, chủ thể phải chiếm ≥ 30% diện tích ảnh mới được chọn; nếu không có vật thể nào đạt ngưỡng → escalate.
3. **Xử lý đa chủ thể cùng loại**: nếu có nhiều xe buýt và không có xe cá nhân → vẫn nhãn "bus".
4. **Xử lý đa chủ thể khác loại** (ví dụ: người và xe ngang nhau): quy định ai có quyền quyết định (annotator tự ghi chú "ambiguous" hay phải hỏi reviewer/lead).
5. **Nhãn cảnh** (*scene label*): nếu không có vật thể đơn lẻ nào nổi bật (ảnh cảnh panorama, đường phố đông đúc), cân nhắc dùng nhãn cảnh như `traffic_jam` nếu taxonomy hỗ trợ, hoặc ghi rõ lý do escalate.

**Vì sao model score không phải ground truth?**

Model score (0.511 với nhãn `cab`) là **đầu ra xác suất của checkpoint sau softmax** – nó phản ánh mức độ khớp giữa đặc trưng thị giác trong ảnh và phân phối mà mô hình học được từ ImageNet. Đây **không phải** là đánh giá của con người về tính đúng đắn của nhãn. Ground truth phải được:
- Tạo bởi **annotator có chuyên môn** theo guideline đã duyệt;
- **Xác nhận bởi reviewer độc lập** (ít nhất một người khác);
- Lưu theo **taxonomy và quy trình QC** của dự án, không phải theo output của mô hình.

Nói cách khác: một ảnh được model dự đoán `cab` với score 0.99 không có nghĩa là ground truth là `cab` – chỉ có con người theo guideline mới xác nhận điều đó.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

**Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):**

Record chọn: **person** rõ nhất trên ảnh – hộp xanh lam lớn ở bên phải ảnh.

```json
{
  "sample_id": "kitchen",
  "task": "object_detection",
  "taxonomy_name": "COCO-80",
  "model_file": "yolo11n.pt",
  "score_threshold": 0.35,
  "class_name": "person",
  "score": 0.912625,
  "coordinate_unit": "pixel",
  "bbox_xyxy": [385.33, 69.24, 498.92, 348.92],
  "bbox_width": 113.58,
  "bbox_height": 279.68
}
```

**Diễn giải vị trí box bằng lời:**

Hộp bắt đầu từ pixel **(385, 69)** ở góc trên-trái (gần cạnh phải ảnh, gần đỉnh ảnh) và kéo dài đến **(499, 349)** ở góc dưới-phải. Chiều rộng 113 px và chiều cao 280 px — hộp chiều dọc cao gần gấp 2.5 lần chiều ngang, bao trọn người đầu bếp đứng quay lưng lại ở nửa phải ảnh kitchen. Gốc tọa độ `(0, 0)` là góc trên-trái ảnh; trục x tăng sang phải, trục y tăng xuống dưới.

```
(0,0)──────────────────────► x
│
│           [385,69]────────┐
│           │  person 0.91 │
│           │   (113×280)  │
│           └────────[499,349]
▼ y
```

**So sánh số prediction ở hai ngưỡng:**

Notebook chạy 3 ngưỡng: `0.20`, `0.35`, `0.60`. File JSON lưu kết quả ở `0.35`; hai ngưỡng còn lại có thể suy ra từ dữ liệu trong JSON:

| Ngưỡng score | Số prediction (kitchen) | Nhãn có mặt |
|---|---|---|
| **0.20** (thấp) | > 11 (nhiều hơn, model thêm các vật thể score thấp) | person, bowl, oven, cup + có thể thêm nhiều bowl/cup nhỏ |
| **0.35** (mặc định) | **11** | person ×2, bowl ×5, oven ×2, cup ×2 |
| **0.60** (cao) | **6** | person ×2, bowl ×2, oven ×2 (bỏ 5 cup/bowl nhỏ có score < 0.60) |

**Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer?**

- **Ngưỡng thấp (0.20)**: Độ bao phủ cao hơn — ít bỏ sót vật thể thật, nhưng nhiều false positive (hộp sai/hộp trùng). Reviewer phải kiểm tra và loại bỏ nhiều hộp không hợp lệ → **khối lượng reviewer tăng**.
- **Ngưỡng cao (0.60)**: Ít hộp hơn, chủ yếu là vật thể rõ ràng. Reviewer xem ít hộp hơn nhưng **dễ bỏ sót** các vật thể nhỏ, bị che khuất, hoặc ánh sáng yếu → **độ bao phủ giảm**.
- Ngưỡng không thay thế guideline: dù model bỏ sót ở ngưỡng 0.60, annotator vẫn phải gán nhãn tất cả vật thể đủ điều kiện theo guideline.

**Đề xuất một quy tắc box chặt:**

> *"Hộp bao phủ (tight bounding box): vẽ hộp nhỏ nhất bao trọn toàn bộ pixel thuộc vật thể có thể nhìn thấy, kể cả phần bị che một phần. Mỗi cạnh hộp phải chạm hoặc cách biên vật thể không quá 5 pixel. Không mở rộng hộp vào vùng nền hay vật thể khác."*

**Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?**

- **Bị cắt mép ảnh**: Guideline cần quy định rõ — gán nhãn nếu ≥ X% diện tích vật thể còn trong ảnh (ví dụ ≥ 50%), hoặc luôn bỏ qua nếu chỉ thấy một phần nhỏ. Annotator **không tự quyết định** mà phải theo ngưỡng trong guideline.
- **Bị che khuất một phần** (*partial occlusion*): Nếu vật thể bị che < 50% → vẫn gán nhãn và vẽ hộp quanh phần nhìn thấy được. Nếu bị che ≥ 50% → cần guideline quy định hoặc escalate cho reviewer/lead quyết định.
- **Trường hợp mơ hồ** (không rõ là một hay hai vật thể, vùng giao thoa…): Annotator ghi chú "ambiguous" và escalate — không tự suy đoán.
- **Người quyết định cuối**: Lead annotator hoặc reviewer theo quy trình QC của dự án, không phải model score.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

**Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):**

Record chọn: **kitchen-001** — người đầu bếp rõ nhất, mặt nạ màu xanh lam đậm phủ toàn thân.

```json
{
  "instance_id": "kitchen-001",
  "class_name": "person",
  "score": 0.899318,
  "polygon_point_count": 348,
  "polygon_xy": [
    [446.0, 70.0],
    [445.0, 71.0],
    [444.0, 71.0],
    [443.0, 72.0],
    [441.0, 73.0],
    "... (348 điểm tổng cộng, chỉ trích 5 điểm đầu làm ví dụ) ...",
    [459.0, 72.0],
    [456.0, 71.0],
    [455.0, 70.0]
  ],
  "bbox_xyxy": [385.45, 66.44, 498.02, 348.58]
}
```

**Mô tả polygon bằng lời:** 348 điểm `[x, y]` tạo thành một đường viền khép kín bám sát hình dạng người — bắt đầu từ vùng đỉnh đầu (~446, 70), chạy dọc theo vai, tay, thân, chân rồi quay lại điểm xuất phát. Các điểm dày đặc hơn ở vùng biên phức tạp (tóc, cánh tay), thưa hơn ở vùng thẳng (cạnh lưng).

**Polygon bổ sung chi tiết gì so với box?**

Bounding box chỉ cho biết vật thể nằm trong vùng hình chữ nhật nào, nhưng toàn bộ phần nền bên trong hộp đó cũng bị tính vào — không phân biệt được đâu là pixel của người, đâu là nền bếp phía sau. Polygon giải quyết điều đó bằng cách vẽ đường viền khép kín bám sát hình dạng thực của vật thể, xác định chính xác từng pixel thuộc về đối tượng đó và loại bỏ hoàn toàn phần nền bên trong hộp.

Với `kitchen-001` (person), bounding box là hình chữ nhật `[385, 66, 498, 349]` bao gồm cả khoảng trống giữa tay và thân, cả vùng áo tạp dề lẫn phần nền phía sau. Polygon với 348 điểm đi dọc theo đường viền thực — bám vai, cánh tay, hông, chân — nên model (và sau này là annotator) biết chính xác pixel nào thuộc người.

Ngoài ra, khi hai vật thể chồng lên nhau, bounding box chồng nhau không giải quyết được — nhưng hai polygon có thể cắt nhau một phần và vẫn phân biệt được từng đối tượng riêng. Ví dụ `kitchen-005` (dining_table) cần đến 598 điểm vì cạnh bàn không thẳng, có góc xiên và các chân bàn — nếu dùng hộp, hộp đó chiếm cả nửa ảnh `[0, 234, 345, 424]` và nuốt luôn các bát đĩa xung quanh vào trong.

**`instance_id` dùng để làm gì và không phải loại ID nào?**

- **Dùng để**: phân biệt từng đối tượng riêng lẻ trong **một lần chạy model trên một ảnh**. Ảnh kitchen có 2 người → `kitchen-001` (person, score 0.90) và `kitchen-009` (person, score 0.43) — hai instance_id khác nhau dù cùng lớp.
- **Không phải `class_id`**: `class_id` mô tả loại vật thể (person = 0, bowl = 45…); instance_id chỉ đánh số thứ tự, không nói về loại.
- **Không phải tracking ID**: tracking ID theo dõi cùng một vật thể qua nhiều frame video; instance_id ở đây chỉ tồn tại trong một ảnh tĩnh, chạy lại sẽ cho số khác.
- **Không phải database ID / annotation ID**: instance_id ở đây do notebook tạo tự động (`kitchen-001`, `kitchen-002`…), không phải ID được gán vĩnh viễn trong hệ thống annotation như CVAT hay COCO.

**Đề xuất một quy tắc biên mask:**

> *"Biên mask phải bám sát hình chiếu ngoài cùng của vật thể (outermost visible contour): polygon đi qua pixel ngoài cùng thuộc vật thể, không cắt vào vùng vật thể và không bao vào vùng nền. Biên tại vùng che khuất hoặc mờ dừng tại đường giao thấy được; không dự đoán phần khuất. Sai lệch cho phép: ≤ 3 pixel so với biên thực."*

**Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?**

- **Vùng mờ** (motion blur, out-of-focus): Biên vật thể không rõ ràng → guideline phải quy định vẽ theo vị trí ước lượng giữa hai bên mờ, hoặc escalate nếu độ mờ vượt ngưỡng (ví dụ: không nhìn ra biên ở ≥ 20% chu vi).
- **Hai vật thể tiếp xúc** (ví dụ: bát nằm trên bàn): Biên giữa hai vật thể là đường ranh giới nào? — guideline phải nói rõ: dùng đường tiếp xúc nhìn thấy, hay tách ra một khoảng nhỏ. Không tự quyết định.
- **Vật thể che khuất** (*occlusion*): `kitchen-002` (bowl) bị các vật khác đè lên — polygon vẽ đến đâu? Đến biên nhìn thấy, hay ước lượng phần khuất bên dưới? → cần guideline rõ ràng: **chỉ vẽ phần nhìn thấy** (visible contour only), không ước đoán.
- **Vùng giao thoa** (hai mask chồng lên nhau): Pixel nào thuộc instance nào? → guideline quyết định (thường: vật thể nằm trên/phía trước được ưu tiên giữ pixel; vật thể phía sau bị cắt).
- **Khi không chắc**: Annotator ghi chú "ambiguous boundary" và escalate lên reviewer/lead — không tự đoán vì sai lệch biên sẽ ảnh hưởng trực tiếp đến chất lượng training mask.


## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp duy nhất (`class_id` + `class_name` + `taxonomy_name`) cho toàn ảnh | Ảnh `traffic` có nhiều loại xe — model dự đoán `cab` (score 0.51) nhưng xe buýt chiếm diện tích lớn hơn; không rõ chọn lớp nào nếu thiếu guideline | Đọc guideline xác định tiêu chí chọn chủ thể chính; chọn đúng một lớp; ghi chú "ambiguous" và escalate nếu không chắc | Kiểm tra nhãn có đúng chủ thể nổi bật nhất theo guideline không; nếu có "ambiguous" thì quyết định hoặc escalate lên lead |
| Phát hiện vật thể | Danh sách hộp `bbox_xyxy` + `class_id` + `class_name`, mỗi hộp cho một vật thể | `kitchen`: person bên trái bị cắt mép ảnh (`x_min ≈ 0`); bowl nhỏ chồng nhau; ngưỡng 0.35 lọc bỏ vật thể score thấp nhưng guideline vẫn yêu cầu gán | Vẽ tight bbox cho tất cả vật thể đủ điều kiện theo guideline (kể cả khi model bỏ sót); ghi chú bị cắt mép/che khuất; escalate nếu không rõ | Kiểm tra hộp có bám sát vật thể không (không thừa nền, không thiếu phần vật thể); kiểm tra nhãn lớp; xác nhận xử lý cắt mép/che khuất đúng quy tắc |
| Instance segmentation | Danh sách polygon `[x, y]` khép kín + `instance_id` + `class_id` + `class_name`, mỗi polygon cho một đối tượng | `kitchen-005` (dining_table, 598 pts): biên bàn tiếp xúc bát và nền — không rõ polygon dừng ở đâu; `kitchen-002` (bowl) bị che khuất — vẽ phần nhìn thấy hay ước lượng phần khuất? | Vẽ polygon bám sát biên nhìn thấy (visible contour only); dùng instance_id phân biệt vật thể cùng lớp; không ước đoán phần khuất; escalate khi biên mơ hồ | Kiểm tra polygon bám biên ≤ 3 px; kiểm tra instance_id không trùng; xác nhận vùng tiếp xúc/che khuất xử lý đúng quy tắc; loại mask nếu biên lệch quá giới hạn |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: Không chia sẻ ảnh, file JSON hoặc bất kỳ output nào ra ngoài kênh được phê duyệt của dự án; không lưu dữ liệu lên dịch vụ đám mây cá nhân hoặc gửi qua email/chat cá nhân, kể cả khi tưởng là dữ liệu công khai.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: mentor/lead của dự án qua kênh nội bộ được chỉ định, không tự xử lý hoặc xóa.

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
