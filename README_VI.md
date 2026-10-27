# Project phân loại ảnh thời tiết — hướng dẫn chạy và hiểu logic

## 1. Bạn nhận được gì?

Project sạch dùng đúng 8 model học máy truyền thống, ảnh 64×64 RGB, 12.288 raw pixel features và Streamlit. Đọc `BAO_CAO_KIEM_TRA.md` để xem lỗi gốc và số đo thực tế. Không dùng CNN, không dùng model PKL cũ.

- `dataset/`: ảnh đầu vào; vẫn giữ các bản trùng để có thể tái lập báo cáo loại trùng.
- `quarantine/`: 2 file đuôi JPG nhưng nội dung GIF; giữ nguyên bản gốc, không đưa vào training.
- `alien_test/`: 30 ảnh bên ngoài đã có trong ZIP gốc.
- `processed/`: dữ liệu mới, mapping và metadata chứng minh nguồn dữ liệu.
- `models/`: model mới và metadata: tên model, lúc train, phiên bản thư viện, dataset ID.
- `reports/`: các bảng so sánh, ảnh biểu đồ và kết quả kiểm tra.
- `external_labels.csv`: nhãn external lấy từ tên file gốc, ghi rõ nguồn. Bạn cần xác nhận nhãn trước khi coi đây là ground truth đã kiểm duyệt.

Model trong gói đã được chạy tại môi trường Python 3.12 Linux. Mã dùng pathlib và tương thích đường dẫn Windows; chưa có phiên Windows thật để xác minh toàn bộ quá trình trên máy của bạn. Các kiểm tra có thực sự chạy đều được ghi rõ trong báo cáo, không coi kiểm tra wheel là thay thế cho chạy trên Windows.

## 2. Chạy demo bản đã train (ngắn nhất)

Giải nén vào `C:\ML\weather_ml_project`. Mở chính folder này bằng VS Code, chọn Terminal → New Terminal (PowerShell). Chạy từng lệnh, không gõ dấu `>` ở đầu:

```powershell
py -3.12 --version
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe check_setup.py
.\.venv\Scripts\python.exe sanity_check.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

VS Code: Ctrl+Shift+P → Python: Select Interpreter → chọn `.venv\Scripts\python.exe` trong project. Không cần activate môi trường, tránh lỗi Execution Policy. Nếu bước nào báo lỗi, dừng ở đó và gửi toàn bộ traceback, không tiếp tục qua bước training.

`sanity_check.py` chọn trước 3 ảnh mỗi class trong Train, in tên ảnh/nhãn thật/nhãn dự đoán/xác suất/đúng sai. Đây là bước thử máy, KHÔNG phải chứng minh khả năng dự đoán ảnh mới.

Dự đoán một ảnh trực tiếp bằng Python, không qua Streamlit:

```powershell
.\.venv\Scripts\python.exe sanity_check.py --image "C:\ML\my_picture.jpg"
```

Đánh giá bộ external và kiểm tra kỹ thuật lại:

```powershell
.\.venv\Scripts\python.exe verify_pipeline.py
.\.venv\Scripts\python.exe evaluate_external.py
```

## 3. Muốn tự train lại từ đầu để học cách chạy?

Gói đầy đủ đã có kết quả Test. `train_models.py` cố ý không ghi đè/chạy lại Test liên tục. Đây là chốt chống vô tình chọn tham số theo điểm Test, không phải lỗi.

Để tái lập đúng thí nghiệm cũ: tạo một BẢN SAO project; lưu riêng kết quả gốc; trong bản sao chỉ bỏ các folder kết quả `processed/`, `models/`, `reports/` rồi giữ nguyên code/tham số/dataset. Việc này là tái lập thí nghiệm, KHÔNG tạo ra một Test mới độc lập. Không thay tham số dựa trên Test đã xem.

Sau khi tạo môi trường theo mục 2, chạy theo thứ tự:

```powershell
.\.venv\Scripts\python.exe check_setup.py
.\.venv\Scripts\python.exe preprocessing.py
.\.venv\Scripts\python.exe verify_pipeline.py
.\.venv\Scripts\python.exe eda.py
.\.venv\Scripts\python.exe train_models.py
.\.venv\Scripts\python.exe sanity_check.py
.\.venv\Scripts\python.exe evaluate_external.py
.\.venv\Scripts\python.exe verify_pipeline.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Training có 16 cấu hình × 3 fold = 48 lần fit, cộng fit cuối; SVM còn có tính xác suất nội bộ. Có thể mất nhiều phút hoặc lâu hơn tùy CPU. Đợi dòng kết quả tiếp theo; không mở nhiều lượt training cùng lúc. Giới hạn 2 luồng CPU để giảm áp lực RAM và tránh các tầng song song chồng nhau.

Nếu cần nghiên cứu cải thiện mới sau khi đã xem Test: dùng Train/CV cho việc chọn tham số và báo rõ Test cũ đã được xem. Chỉ dữ liệu mới được giữ kín mới là Test cuối độc lập cho nghiên cứu tiếp theo.

## 4. Dây chuyền xử lý ảnh và mapping

`utils.py` chứa hàm DUY NHẤT dùng ở preprocessing, sanity, external và app:

bytes → OpenCV decode BGR → RGB → resize INTER_LINEAR 64×64 → float32 → chia 255 → flatten C-order → vector 12.288 số.

Feature là một manh mối; ở đây mỗi feature chỉ là cường độ một kênh màu tại một vị trí. Flatten giống như trải khay 64×64×3 ô thành một hàng 12.288 ô, không đổi thứ tự.

| ID | Class |
|---|---|
| 0 | cloudy |
| 1 | foggy |
| 2 | rainy |
| 3 | shine |
| 4 | sunrise |

Model bundle chứa cả estimator, mapping, cấu hình preprocessing, dataset ID và phiên bản thư viện. App không lấy mapping từ một file rời không rõ nguồn. Khi load sẽ kiểm tra số features, model.classes_, mapping và cấu hình. App hiển thị dấu vân tay SHA-256 của model; cache thay đổi khi bytes model đổi. Chỉ dùng model bundle của chính project, không nạp PKL không rõ nguồn.

## 5. Chống lộ đề thi — data leakage

1. Hash vector sau preprocessing phát hiện các ảnh mà model nhìn thấy y hệt nhau. Bỏ bản dư trước split; ảnh trùng khác nhãn làm chương trình dừng.
2. pHash so sánh cấu trúc sáng tối để tìm **ứng viên gần trùng** (ngưỡng Hamming ≤6). Đây là bộ dò gần đúng: có thể bỏ sót hoặc báo nhầm, không phải chứng nhận mọi ảnh đều độc lập.
3. Giữ các ứng viên gần trùng cùng nhóm, không xóa hay đổi nhãn theo pHash. Một cặp foggy/sunrise đã được xem trực tiếp là báo nhầm; vẫn nhóm bảo thủ để không tùy tiện điều chỉnh phép chia theo kết quả.
4. Dùng fold đầu của StratifiedGroupKFold(5, shuffle=True, random_state=42) làm Test, xấp xỉ 20%. Khác split 80/20 theo từng ảnh vì cần giữ nguyên nhóm. Không thử nhiều split rồi chọn split cho điểm đẹp.
5. CV trên Train dùng StratifiedGroupKFold 3 fold với chính nhóm đó. Ba fold là ba lượt thi thử; mỗi ảnh Train làm validation một lần. Ba fold giảm thời gian cho 16 cấu hình raw-pixel, không phải hạ chuẩn để chọn model có lợi.
6. Scaler của Logistic Regression và SVM nằm trong sklearn Pipeline, chỉ học mean/std từ phần Train của từng fold, không fit trên toàn bộ X trước CV.
7. Chọn model bằng mean Macro F1 trên CV; hòa thì dùng CV Accuracy, rồi giữ thứ tự ổn định. Không có nhánh ép SVM Polynomial thắng.
8. Fit model thắng trên toàn Train; sau đó mới dự đoán Test. Lưu dấu `final_test_started.json` trước khi mở Test để tránh vô tình chạy lại sau lỗi.
9. External được kiểm tra trùng với TOÀN dataset, không chỉ Train. Các ảnh trùng/gần trùng được ghi riêng và không tính vào điểm external độc lập.

Đã khắc phục leakage phát hiện được; không khẳng định chỉ bằng hash mà loại hết mọi khả năng liên quan nguồn ảnh, ảnh crop hoặc ảnh cùng một cảnh chụp.

## 6. Ý nghĩa từng file code

| File | Làm gì? |
|---|---|
| config.py | Địa chỉ folder, mapping, số features, seed và ngưỡng hash |
| utils.py | Máy xử lý ảnh chung, hash, kiểm tra model, dự đoán |
| check_setup.py | Kiểm tra Python, import thư viện, cấu trúc dataset |
| preprocessing.py | Đọc ảnh, loại trùng, nhóm gần trùng, chia và lưu dữ liệu |
| eda.py | Bảng/biểu đồ phân bố class trước và sau chia |
| train_models.py | Chọn 8 model bằng Train CV, fit cuối, Test và lưu bundle |
| evaluation.py | Hàm xuất metric, classification report, confusion matrix |
| sanity_check.py | Thử vài ảnh có nhãn hoặc một file bất kỳ bằng Python |
| evaluate_external.py | Đánh giá ảnh ngoài dataset, loại chồng lặp khỏi metric |
| verify_pipeline.py | Kiểm tra RGB, flatten, mapping, toàn bộ features, split và model |
| app.py | Nhận ảnh upload và gọi đúng hàm dự đoán chung |

Không có nhánh tự bỏ model thất bại. Nếu một trong tám model lỗi hoặc Logistic Regression chưa hội tụ, training dừng để sửa, không xuất bảng giả như thể đã so sánh đủ.

## 7. Cách đọc kết quả

- Accuracy: tỷ lệ ảnh đoán đúng.
- Macro F1: tính F1 riêng cho mỗi class rồi lấy trung bình; mỗi class có quyền ngang nhau.
- Confusion matrix: hàng là nhãn thật, cột là nhãn đoán. Ô cloudy hàng → foggy cột là số cloudy bị đoán foggy.
- Train–Test gap: chênh lệch điểm luyện tập và điểm thi; chênh lớn là dấu hiệu cần xem xét overfitting (học thuộc), chưa phải tự nó chứng minh nguyên nhân.
- CV std: độ dao động giữa ba lượt thi thử; mean cao hơn rất ít so với std không đủ chứng minh model đó vượt trội chắc chắn.
- Confidence: xác suất ước lượng model gán cho nhãn đang dự đoán; có thể cao mà sai. Không phải “xác suất chắc chắn ngoài đời”.

SVC có thể cho nhãn `predict()` khác nhãn có `predict_proba()` lớn nhất. Code cũ lấy max xác suất gắn vào nhãn predict. Bản mới lấy đúng cột của nhãn predict và vẫn hiển thị toàn bộ xác suất; không tự biến decision score thành phần trăm.

## 8. Phân biệt ba loại sai

A — Bug code: ảnh cùng bytes qua script và app mà vector/nhãn khác nhau; mapping/feature count/path/model ID sai. Sửa chương trình.

B — Sai quy trình ML: ảnh trùng hai phía, dùng Test chọn model, scaler học trên validation hoặc nhãn mâu thuẫn. Sửa quy trình/dữ liệu trước khi kết luận model yếu.

C — Hạn chế model: sau khi A/B đã được kiểm tra, vẫn nhầm vì khác cảnh, độ sáng, bối cảnh, loại mây/sương/mưa; raw pixel nhạy với vị trí và không biểu diễn tốt texture/hình dạng. Ghi rõ kết quả thay vì chỉ nói “ML không đạt 100%”. Không thể xác định nguyên nhân nhân quả của từng ảnh chỉ bằng một con số.

Cloudy và shine có thể cùng xuất hiện mây và nắng; rainy có thể bầu trời âm u nhưng không thấy hạt mưa. Sunrise và sunset thường không phân biệt được chỉ từ ảnh; class sunrise cần được hiểu theo quy ước dataset. Rà soát nhãn là quyết định chuyên môn, không tự đổi nhãn chỉ để model được điểm cao.

Nếu nghiên cứu tiếp, vẫn có thể dùng đặc trưng thủ công histogram màu/HOG/LBP với 8 model trên Train CV, hoặc thu thập nhãn/ảnh mới đa dạng. Chưa tích hợp những hướng này vào bản raw-pixel hiện tại. CNN chỉ là hướng phát triển sau, không nằm trong project này.

## 9. Tài liệu kỹ thuật đối chiếu

- https://scikit-learn.org/stable/common_pitfalls.html
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
- https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

Model file phụ thuộc phiên bản thư viện; dùng requirements đã chốt hoặc huấn luyện lại trong môi trường mới. File `windows_wheel_check.txt` ghi kiểm tra gói Windows khi có; đọc trạng thái cuối, không suy từ việc tải metadata.

## 10. Khi một ảnh demo bị sai

Ví dụ bạn biết ảnh thật là cloudy:

```powershell
.\.venv\Scripts\python.exe debug_prediction.py --image "C:\ML\cloudy.jpg" --true-label cloudy
```

Script in mapping, dấu vân tay file ảnh/model/features, shape, min/max, nhãn dự đoán, xác suất và kết quả đối chiếu đường đọc file với đường upload của app. Nếu cả hai cùng sai thì chưa có bằng chứng lỗi riêng Streamlit; xem báo cáo dữ liệu/model tiếp. Không tự đổi ground truth theo dự đoán.

## 11. Bản nghiên cứu cải thiện, vẫn là Machine Learning truyền thống

Vì bản raw-pixel còn sai cả 4 ảnh cloudy external, đã thực hiện thêm một thí nghiệm CHỈ trên Train/CV:

- `features.py`: nhận đúng vector RGB 12.288 số, trích histogram màu HSV, trung bình/độ lệch chuẩn màu theo vùng và HOG (hướng thay đổi sáng tối). Tổng cộng 1.908 đặc trưng thủ công; không phải Deep Learning.
- `research_features.py`: so sánh lại đủ 8 họ model, cùng fold Train và hai cấu hình mỗi họ. Không đọc X_test/y_test hoặc ảnh external.
- Model thắng nghiên cứu nằm riêng ở `models/experimental_feature_model.pkl`, không ghi đè baseline đã có Final Test.
- Kết quả CV nằm ở `reports/feature_research_comparison.csv`.

Mở app với ứng viên nghiên cứu:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py -- --experimental
```

App sẽ ghi rõ đây là bản nghiên cứu chưa có Test độc lập mới. Không lấy điểm Test 72,79% của baseline gắn cho model này. Không mặc định nó đã sửa được 4 ảnh cloudy: chưa dùng các ảnh external cũ để chấm/chọn ứng viên mới.

Để chốt chất lượng bản cải thiện, cần bộ ảnh có nhãn mới chưa dùng cho các quyết định trước đây. Không đơn giản chia lại 1.500 ảnh cũ rồi gọi đó là một Test hoàn toàn mới.

Muốn tái lập thí nghiệm CV này, chạy `python research_features.py` trong đúng môi trường đã cài. Đây là bước nghiên cứu bổ sung, không thuộc chuỗi chạy baseline ở mục 3. File tham số và Test baseline được giữ nguyên.
