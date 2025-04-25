# Giới thiệu về Fast R-CNN và Faster R-CNN

## 1. Fast R-CNN

### Tổng quan

- **Mục tiêu**: Fast R-CNN cải thiện tốc độ và độ chính xác của R-CNN bằng cách tích hợp các bước trích xuất đặc trưng, phân loại, và hồi quy bounding box vào một mạng **end-to-end**.
- **Thành tựu**:
  - Nhanh hơn R-CNN (~0.3 giây/ảnh so với ~10-45 giây).
  - Độ chính xác cao hơn: mAP ~66% trên PASCAL VOC 2012 (so với ~53% của R-CNN).
  - Huấn luyện đơn giản hơn, không cần SVM riêng.
- **Đặc điểm nổi bật**:
  - Dùng **RoI Pooling** để xử lý vùng đề xuất hiệu quả.
  - Tích hợp phân loại và hồi quy vào CNN.
  - Vẫn dùng **Selective Search** cho vùng đề xuất.

### Ý tưởng cốt lõi

- **Hạn chế của R-CNN**:
  - Chậm: Xử lý từng vùng đề xuất riêng qua CNN (~2000 lần/ảnh).
  - Không end-to-end: Huấn luyện CNN, SVM, hồi quy riêng.
  - Tốn bộ nhớ: Lưu đặc trưng cho 2000 vùng.
- **Giải pháp**: **Fast R-CNN**:
  - Trích xuất đặc trưng cho **toàn bộ ảnh** một lần bằng CNN.
  - Ánh xạ vùng đề xuất lên feature map bằng **RoI Pooling**.
  - Tích hợp phân loại (softmax) và hồi quy bounding box trong cùng mạng.
  - Huấn luyện end-to-end với hàm mất mát kết hợp (classification + regression).

### Kiến trúc Fast R-CNN

#### Đầu vào

- Ảnh RGB (bất kỳ kích thước).
- Danh sách vùng đề xuất (~2000) từ Selective Search.

#### Quy trình

1. **Trích xuất đặc trưng toàn ảnh**:
   - Đưa toàn bộ ảnh qua CNN (như VGG16, ResNet) để tạo feature map.
   - Ví dụ: VGG16 tạo feature map (512 \(\times\) H/16 \(\times\) W/16).

2. **RoI Pooling**:
   - Ánh xạ vùng đề xuất (x, y, w, h) từ không gian ảnh gốc sang feature map.
   - Chia vùng trên feature map thành lưới cố định (ví dụ: 7x7).
   - Áp dụng **Max Pooling** để tạo đặc trưng kích thước cố định (ví dụ: 512 \(\times\) 7 \(\times\) 7).
   - **Đầu ra**: Vector đặc trưng cho mỗi vùng (ví dụ: 512 \(\times\) 49).

3. **Phân loại và hồi quy**:
   - Đặc trưng qua các tầng fully connected (FC).
   - Hai nhánh đầu ra:
     - **Phân loại**: Softmax dự đoán xác suất cho \(K+1\) lớp (K lớp đối tượng + nền).
     - **Hồi quy**: Dự đoán 4 giá trị điều chỉnh (\(\Delta x, \Delta y, \Delta w, \Delta h\)) cho mỗi lớp.
   - **Hàm mất mát**:
     \[
     L = L_{\text{cls}}(p, u) + \lambda [u \geq 1] L_{\text{reg}}(t, t^*)
     \]
     - \(L_{\text{cls}}\): Cross-Entropy Loss cho phân loại.
     - \(L_{\text{reg}}\): Smooth L1 Loss cho hồi quy.
     - \(u\): Nhãn thực (0: nền, 1-K: lớp đối tượng).
     - \(t, t^*\): Tọa độ dự đoán và ground truth.
     - \(\lambda\): Hệ số cân bằng (thường = 1).

4. **Non-Maximum Suppression (NMS)**:
   - Loại bỏ các bounding box chồng lấn dựa trên điểm số softmax.

#### Đầu ra

- Danh sách bounding box: (x, y, w, h, class, score).

#### Số tham số và FLOPs

- **Tham số**:
  - CNN (VGG16): ~138M.
  - FC layers: ~10-20M (tùy cấu hình).
  - Tổng: ~150M.
- **FLOPs**:
  - CNN: ~15B FLOPs/ảnh (VGG16).
  - RoI Pooling + FC: ~1-2B FLOPs.
  - Tổng: ~17B FLOPs, nhanh hơn R-CNN (~1.4T FLOPs).

### Ưu điểm

1. **Nhanh hơn R-CNN**:
   - Xử lý toàn ảnh một lần, giảm thời gian từ ~10-45 giây xuống ~0.3 giây.
2. **End-to-end**:
   - Huấn luyện tích hợp, loại bỏ SVM và lưu đặc trưng riêng.
3. **Độ chính xác cao**:
   - mAP ~66% (PASCAL VOC), nhờ softmax và hồi quy tích hợp.
4. **Linh hoạt**:
   - Có thể dùng bất kỳ CNN backbone (VGG16, ResNet).

### Nhược điểm

1. **Vẫn chậm cho thời gian thực**:
   - ~0.3 giây/ảnh, chậm hơn YOLO (~0.01 giây).
2. **Phụ thuộc Selective Search**:
   - V