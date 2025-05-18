# 💡 Cải Thiện Hình Ảnh Sử Dụng SVR (Support Vector Regression)

## 📌 Mô Tả Dự Án

Dự án này triển khai một hệ thống cải thiện hình ảnh sử dụng **Hồi Quy Vector Hỗ Trợ (SVR)** để dự đoán giá trị gamma tối ưu cho hiệu chỉnh ảnh. Ảnh được xử lý thông qua gamma correction dựa trên đầu ra của mô hình SVR, giúp tăng độ rõ nét và cải thiện chất lượng hiển thị.

---

## 🚀 Tính Năng

- ✅ Tự động trích xuất đặc trưng ảnh (mean, std, entropy)
- ✅ Huấn luyện mô hình SVR để dự đoán giá trị gamma
- ✅ Áp dụng gamma correction lên ảnh mờ
- ✅ Hỗ trợ nhiều định dạng ảnh phổ biến (JPG, PNG, ...)
- ✅ Giao diện web đơn giản (Flask) để tải ảnh và xem kết quả theo thời gian thực

---

## 👨‍💻 Thành Viên Thực Hiện

- **Hồ Minh Luân** – MSSV: 22644751  
- **Trương Công Đạt** – MSSV: 22685561

---

## 🛠️ Công Nghệ Sử Dụng

- Python 3.11
- Flask (Web Framework)
- OpenCV (Xử lý ảnh)
- Scikit-learn (SVR, chuẩn hóa, mô hình)
- Joblib (Lưu mô hình)
- HTML/CSS/JS (Giao diện)

---

## 📁 Cấu Trúc Dự Án

```
dehazing_data/
├── hazy_images/       # Ảnh đầu vào bị mờ do sương
├── clear_images/      # Ảnh rõ nét tương ứng

models/                # Chứa mô hình SVR (.pkl) và scaler

static/
├── uploads/           # Ảnh người dùng tải lên
├── processed/         # Ảnh đã xử lý bằng gamma correction

templates/
├── index.html         # Giao diện người dùng

train_svr.py           # Script huấn luyện SVR
app.py                 # Flask backend
requirements.txt       # Danh sách thư viện cần cài
```

---

## ⚙️ Hướng Dẫn Sử Dụng

1. **Tải dataset** từ:  
   👉 https://www.kaggle.com/datasets/aneeshkaleru/foggy-images-dataset

2. **Tổ chức lại dữ liệu như sau:**
   ```
   dehazing_data/
   ├── hazy_images/
   └── clear_images/
   ```

3. **Cài đặt thư viện:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Huấn luyện mô hình SVR:**
   ```bash
   python train_svr.py
   ```

5. **Chạy ứng dụng Flask:**
   ```bash
   python app.py
   ```

6. **Truy cập trình duyệt tại:**  
   👉 http://localhost:5000

---

## 📦 Yêu Cầu Hệ Thống

- Python ≥ 3.11  
- Máy tính có thể chạy mô hình học máy cơ bản  
- Kết nối Internet để tải dữ liệu và thư viện (lần đầu)

---

## 📄 Giấy Phép

Dự án được thực hiện cho mục đích học tập và nghiên cứu. Không sử dụng cho mục đích thương mại khi chưa có sự cho phép.

---

✅ *Cần hỗ trợ? Hãy mở issue trên GitHub hoặc liên hệ trực tiếp nhóm phát triển.*
