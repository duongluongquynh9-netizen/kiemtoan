# 🛡️ Hệ thống Phát hiện Giao dịch Bất thường (Anomalous Transaction Detection System)

Ứng dụng Web Dashboard phân tích và phát hiện các giao dịch tài chính bất thường bằng thuật toán Học máy **Isolation Forest**, được phát triển dựa trên thư viện **Streamlit** (Python). Hệ thống tự động tiền xử lý dữ liệu giao dịch, chấm điểm rủi ro, phân loại cảnh báo thành **4 mức độ**, và trực quan hóa không gian dữ liệu bằng biểu đồ tương tác 3D/2D Plotly.

---

## ✨ Các tính năng chính

*   **📊 Dashboard tổng quan & EDA**: Hiển thị tổng số giao dịch, tỷ lệ rủi ro, số lượng giao dịch bất thường/khẩn cấp, và **giá trị trung bình số tiền giao dịch** đi kèm biểu đồ thống kê giao dịch theo giờ.
*   **⚠️ Phân loại rủi ro 4 mức độ**:
    1.  **Cấp 1: Bình thường (Low Risk)**: Giao dịch an toàn, không bị phát hiện bất thường bởi Isolation Forest.
    2.  **Cấp 2: Chú ý (Medium Risk)**: Giao dịch bất thường nhẹ (điểm rủi ro nằm ở nửa trên của các giao dịch bị gắn cờ).
    3.  **Cấp 3: Nguy hiểm (High Risk)**: Giao dịch có mức độ bất thường cao.
    4.  **Cấp 4: Khẩn cấp (Critical Risk)**: Giao dịch bất thường đặc biệt nghiêm trọng (thuộc phân vị 25% điểm rủi ro thấp nhất của nhóm bất thường) cần xử lý ngay lập tức.
*   **🛠️ Tùy chỉnh mô hình trực tiếp**: Sidebar hỗ trợ thay đổi tham số của Isolation Forest (`n_estimators`, `contamination`) và điều chỉnh ngưỡng xác định mức "Khẩn cấp".
*   **📥 Xuất báo cáo**: Tải xuống dữ liệu giao dịch bất thường và khẩn cấp đã lọc dưới dạng file CSV/Excel chỉ với một cú nhấp chuột.
*   **🧠 Trình giả lập giao dịch đơn lẻ**: Nhập các thông số giao dịch cụ thể để kiểm tra mức độ rủi ro tức thời mà không cần tải lên file lớn.
*   **🎨 Thiết kế Premium**: Tích hợp giao diện tối màu chuyên nghiệp (Sleek Dark Theme) và cấu trúc hiển thị dạng lưới kính mờ (Glassmorphism) sang trọng.

---

## 📂 Cấu trúc thư mục dự án

```text
├── app.py                      # Mã nguồn chính của ứng dụng Streamlit
├── requirements.txt            # Danh sách các thư viện Python phụ thuộc
├── README.md                   # Hướng dẫn sử dụng và triển khai dự án (File này)
└── transactions_Q1_demo.csv   # File dữ liệu giao dịch mẫu (Q1)
```

---

## 🚀 Hướng dẫn chạy ứng dụng cục bộ (Local Setup)

### 1. Chuẩn bị môi trường Python
Khuyên dùng Python phiên bản **3.9 đến 3.11**. Tạo môi trường ảo để tránh xung đột thư viện:

```bash
# Tạo môi trường ảo (tên là venv)
python -m venv venv

# Kích hoạt môi trường ảo
# Trên Windows (Command Prompt):
venv\Scripts\activate
# Trên Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Trên macOS/Linux:
source venv/bin/activate
```

### 2. Cài đặt các thư viện cần thiết
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Chạy Web App Streamlit
```bash
streamlit run app.py
```
Sau khi khởi chạy thành công, ứng dụng sẽ tự động mở trên trình duyệt tại địa chỉ mặc định: `http://localhost:8501`.

---

## ☁️ Hướng dẫn triển khai lên Streamlit Community Cloud (Deploy)

Streamlit Community Cloud cho phép triển khai trực tuyến ứng dụng từ kho lưu trữ GitHub hoàn toàn miễn phí.

1.  **Đưa dự án lên GitHub**:
    *   Tạo một kho lưu trữ (repository) mới trên GitHub (ở chế độ Public).
    *   Thêm các file `app.py`, `requirements.txt`, `README.md` và `transactions_Q1_demo.csv` vào repository này.
    *   Tiến hành commit và push mã nguồn lên GitHub.

2.  **Triển khai trên Streamlit Cloud**:
    *   Truy cập trang [share.streamlit.io](https://share.streamlit.io) và đăng nhập bằng tài khoản GitHub của bạn.
    *   Nhấn vào nút **"New app"**.
    *   Chọn repository của bạn, chọn nhánh chính (`main` hoặc `master`), và chỉ định file chạy chính là `app.py`.
    *   Nhấn nút **"Deploy"**. Streamlit Cloud sẽ tự động đọc `requirements.txt`, cài đặt các thư viện cần thiết và thiết lập ứng dụng chạy trực tuyến sau vài phút.

---

## 🛠️ Công nghệ sử dụng

*   **Streamlit**: UI & Framework chính.
*   **Scikit-Learn**: Tiền xử lý dữ liệu (`StandardScaler`) và mô hình học máy (`IsolationForest`).
*   **Pandas & Numpy**: Xử lý, tính toán và thao tác dữ liệu bảng.
*   **Plotly**: Thư viện trực quan hóa dữ liệu tương tác 2D và 3D.
*   **Openpyxl**: Xuất file định dạng Excel.
