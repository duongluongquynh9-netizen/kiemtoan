import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import io
import os

# 1. CẤU HÌNH TRANG WEB APP
st.set_page_config(
    page_title="HÊ THỐNG PHÁT HIện GIAO DỊCH BẤT THƯờnG TRONG KIỂM TOÁN NỘI BỘ",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Nhúng phông chữ Inter và các định dạng CSS tùy chỉnh để tối ưu hóa UI/UX
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Kiểu dáng cho các Metric Cards */
.metric-container {
    display: flex;
    justify-content: space-between;
    gap: 15px;
    margin-bottom: 25px;
}
.metric-card {
    flex: 1;
    background: linear-gradient(135deg, rgba(20, 30, 48, 0.9) 0%, rgba(36, 59, 85, 0.9) 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    color: white;
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.25);
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 20px rgba(0, 0, 0, 0.35);
}
.metric-title {
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #BDC3C7;
    margin-bottom: 10px;
}
.metric-value {
    font-size: 26px;
    font-weight: 700;
}

/* Thẻ cảnh báo rủi ro trong phần Giả lập */
.risk-card {
    padding: 20px;
    border-radius: 12px;
    color: white;
    text-align: center;
    font-weight: bold;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
    margin-top: 15px;
}
.risk-low {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    border: 1px solid #38ef7d;
}
.risk-medium {
    background: linear-gradient(135deg, #f1c40f 0%, #f39c12 100%);
    border: 1px solid #f1c40f;
    color: #1a1a1a;
}
.risk-high {
    background: linear-gradient(135deg, #e67e22 0%, #d35400 100%);
    border: 1px solid #e67e22;
}
.risk-critical {
    background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
    border: 1px solid #ff416c;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.02); }
    100% { transform: scale(1); }
}
</style>
""", unsafe_allow_html=True)

# 2. HÀM HELPER HỖ TRỢ XỬ LÝ DỮ LIỆU & CACHING

def metric_card(title, value, color_gradient):
    """Tạo chuỗi HTML cho thẻ chỉ số (metric card) với gradient màu sắc tùy chọn."""
    card_html = f"""
    <div class="metric-card" style="background: {color_gradient};">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
    </div>
    """
    return card_html

def find_column(df, patterns):
    """Tìm tên cột tương thích dựa trên các mẫu gợi ý."""
    for col in df.columns:
        for p in patterns:
            if p.lower() in col.lower():
                return col
    return None

@st.cache_data(show_spinner="Đang đọc dữ liệu giao dịch...")
def load_data(file_source):
    """Đọc dữ liệu từ file CSV và chuẩn hóa các cột cần thiết."""
    df = pd.read_csv(file_source)
    
    # Chuẩn hóa cột ngày giao dịch
    date_col = find_column(df, ["date", "time", "ngay", "gio"])
    if date_col:
        df = df.rename(columns={date_col: "transaction_date"})
        df["transaction_date"] = pd.to_datetime(df["transaction_date"], format="mixed", errors="coerce")
    else:
        # Fallback nếu không có cột thời gian
        df["transaction_date"] = pd.to_datetime("2026-01-01 12:00:00")
        
    # Chuẩn hóa cột số tiền
    amount_col = find_column(df, ["amount", "so_tien", "sotien", "tien"])
    if amount_col:
        df = df.rename(columns={amount_col: "amount"})
    else:
        # Lấy cột số đầu tiên làm amount nếu không tìm thấy
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            df = df.rename(columns={num_cols[0]: "amount"})
        else:
            df["amount"] = 0.0

    # Chuẩn hóa cột nhân viên ngân hàng
    emp_col = find_column(df, ["employee", "nhan_vien", "nhanvien", "co_nhan_vien"])
    if emp_col:
        df = df.rename(columns={emp_col: "is_employee"})
    else:
        df["is_employee"] = False
        
    # Chuyển đổi is_employee sang kiểu bool
    if df["is_employee"].dtype == object:
        df["is_employee"] = df["is_employee"].astype(str).str.upper().str.strip().isin(["TRUE", "1", "YES"])
    else:
        df["is_employee"] = df["is_employee"].astype(bool)
        
    # Tạo thêm trường giờ giao dịch 'hour' nếu chưa có
    df["hour"] = df["transaction_date"].dt.hour
    
    # Tạo trường nhãn nhân viên dạng số 'co_nhan_vien' (0 hoặc 1)
    df["co_nhan_vien"] = df["is_employee"].astype(int)
    
    return df

@st.cache_resource(show_spinner="Đang huấn luyện mô hình Isolation Forest...")
def train_model(X_train, n_estimators, contamination, random_state=42):
    """Huấn luyện mô hình Isolation Forest với StandardScaler."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples="auto",
        random_state=random_state,
        n_jobs=-1
    )
    iso.fit(X_scaled)
    return scaler, iso

# 3. SIDEBAR: THAY ĐỔI THAM SỐ VÀ TẢI FILE DỮ LIỆU
st.sidebar.markdown("### 🛠️ CẤU HÌNH THAM SỐ")

# Tải lên file dữ liệu mới
uploaded_file = st.sidebar.file_uploader("Tải file dữ liệu giao dịch (.csv)", type=["csv"])

# Lựa chọn nguồn file dữ liệu
default_file_path = "transactions_Q1_demo.csv"
if uploaded_file is not None:
    file_source = uploaded_file
    st.sidebar.success("Đã tải file thành công!")
else:
    file_source = default_file_path
    if os.path.exists(default_file_path):
        st.sidebar.info("Đang sử dụng dữ liệu mẫu mặc định.")
    else:
        st.sidebar.error(f"Không tìm thấy file mặc định {default_file_path}. Vui lòng tải file CSV lên!")

# Các tham số mô hình Isolation Forest
st.sidebar.markdown("---")
st.sidebar.markdown("#### 🌲 Tham số Isolation Forest")
n_estimators = st.sidebar.slider("Số lượng cây quyết định (n_estimators)", min_value=50, max_value=500, value=200, step=50)
contamination = st.sidebar.slider("Tỷ lệ bất thường giả định (contamination)", min_value=0.001, max_value=0.05, value=0.01, step=0.001, format="%.3f")
emergency_percentile = st.sidebar.slider("Phần trăm ngưỡng Khẩn cấp (%)", min_value=5, max_value=50, value=25, step=5)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Thông tin thuật toán:**
Mô hình sử dụng các biến:
*   `amount` (Số tiền giao dịch)
*   `hour` (Giờ thực hiện giao dịch)
*   `co_nhan_vien` (Là nhân viên: 1, Không là nhân viên: 0)
""")

# 4. THỰC THI LOAD DATA & TRAIN MODEL
try:
    df_raw = load_data(file_source)
    # Sao chép để tránh ghi đè cache streamlit
    df = df_raw.copy()
    
    # Lấy các cột đặc trưng để huấn luyện
    X = df[['amount', 'hour', 'co_nhan_vien']]
    
    # Huấn luyện mô hình
    scaler, iso = train_model(X, n_estimators, contamination)
    
    # Dự đoán & Tính điểm cho toàn bộ dataset
    X_scaled = scaler.transform(X)
    df["anomaly_score"] = iso.decision_function(X_scaled)
    df["is_anomaly"] = iso.predict(X_scaled) == -1
    
    # PHÂN LOẠI RỦI RO THÀNH 4 CẤP ĐỘ
    df["risk_level"] = "Bình thường"
    df["risk_code"] = 1
    
    anomaly_mask = df["is_anomaly"] == True
    if anomaly_mask.any():
        anomaly_scores = df.loc[anomaly_mask, "anomaly_score"]
        
        # Ngưỡng phân vị động dựa trên cài đặt của người dùng
        threshold_q_emergency = anomaly_scores.quantile(emergency_percentile / 100.0)
        threshold_q_high = anomaly_scores.quantile(0.50)
        
        # Cấp 4: Khẩn cấp (Critical)
        df.loc[anomaly_mask & (df["anomaly_score"] < threshold_q_emergency), "risk_level"] = "Khẩn cấp"
        df.loc[anomaly_mask & (df["anomaly_score"] < threshold_q_emergency), "risk_code"] = 4
        
        # Cấp 3: Nguy hiểm (High)
        df.loc[anomaly_mask & (df["anomaly_score"] >= threshold_q_emergency) & (df["anomaly_score"] < threshold_q_high), "risk_level"] = "Nguy hiểm"
        df.loc[anomaly_mask & (df["anomaly_score"] >= threshold_q_emergency) & (df["anomaly_score"] < threshold_q_high), "risk_code"] = 3
        
        # Cấp 2: Chú ý (Medium)
        df.loc[anomaly_mask & (df["anomaly_score"] >= threshold_q_high), "risk_level"] = "Chú ý"
        df.loc[anomaly_mask & (df["anomaly_score"] >= threshold_q_high), "risk_code"] = 2

except Exception as e:
    st.error(f"Đã xảy ra lỗi khi tải dữ liệu hoặc huấn luyện mô hình: {str(e)}")
    st.stop()

# 5. HIỂN THỊ TIÊU ĐỀ CHÍNH
st.title("🛡️ HỆ THỐNG PHÁT HIỆN GIAO DỊCH BẤT THƯỜNG")
st.markdown("Hệ thống phân tích thông minh sử dụng thuật toán Học máy **Isolation Forest** để nhận diện rủi ro giao dịch.")

# 6. HIỂN THỊ CÁC THẺ METRICS (HTML & CSS TÙY CHỈNH)
total_txns = len(df)
avg_amount = df["amount"].mean() if total_txns > 0 else 0.0
total_anomalies = df["is_anomaly"].sum()
anomaly_rate = (total_anomalies / total_txns) * 100 if total_txns > 0 else 0.0
total_emergency = (df["risk_level"] == "Khẩn cấp").sum()

# Định dạng tiền tệ VND
def format_vnd(amount_val):
    return f"{amount_val:,.0f} VND"

metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
with metric_col1:
    st.markdown(metric_card("Tổng giao dịch", f"{total_txns:,}", "linear-gradient(135deg, #1e3c72 0%, #2a5298 100%)"), unsafe_allow_html=True)
with metric_col2:
    st.markdown(metric_card("Giá trị trung bình", format_vnd(avg_amount), "linear-gradient(135deg, #3a7bd5 0%, #3a6073 100%)"), unsafe_allow_html=True)
with metric_col3:
    st.markdown(metric_card("Tỷ lệ bất thường", f"{anomaly_rate:.2f}%", "linear-gradient(135deg, #F39C12 0%, #D35400 100%)"), unsafe_allow_html=True)
with metric_col4:
    st.markdown(metric_card("Số GD Bất thường", f"{total_anomalies:,}", "linear-gradient(135deg, #e74c3c 0%, #c0392b 100%)"), unsafe_allow_html=True)
with metric_col5:
    st.markdown(metric_card("Số GD Khẩn cấp", f"{total_emergency:,}", "linear-gradient(135deg, #8e44ad 0%, #2c3e50 100%)"), unsafe_allow_html=True)

st.markdown("---")

# 7. CHIA CÁC TAB CHỨC NĂNG CHÍNH
tab_eda, tab_anomalies, tab_plotly, tab_simulator = st.tabs([
    "📊 Tổng quan & EDA",
    "⚠️ Danh sách Cảnh báo",
    "🧠 Không gian Trực quan AI",
    "🚀 Trình Giả lập Giao dịch"
])

# ================= TAB 1: TỔNG QUAN & EDA =================
with tab_eda:
    st.subheader("📊 Phân tích Khám phá Dữ liệu (EDA)")
    
    eda_col1, eda_col2 = st.columns(2)
    
    with eda_col1:
        st.markdown("#### ⏰ Thống kê số lượng giao dịch theo Giờ")
        hour_counts = df['hour'].value_counts().sort_index().reset_index()
        hour_counts.columns = ['Giờ giao dịch', 'Số lượng giao dịch']
        
        fig_hour = px.bar(
            hour_counts, 
            x='Giờ giao dịch', 
            y='Số lượng giao dịch',
            color='Số lượng giao dịch',
            color_continuous_scale='Blues',
            labels={'Số lượng giao dịch': 'Số lượng giao dịch', 'Giờ giao dịch': 'Giờ giao dịch (0-23)'}
        )
        fig_hour.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(tickmode='linear', tick0=0, dtick=1),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_hour, use_container_width=True)
        
    with eda_col2:
        st.markdown("#### 💰 Phân phối giá trị giao dịch")
        # Sử dụng checkbox lọc outlier để đồ thị đẹp hơn
        exclude_outliers = st.checkbox("Ẩn các giao dịch có số tiền quá cao (> 99% quantile) để tối ưu hiển thị biểu đồ", value=True)
        
        q99_val = df["amount"].quantile(0.99)
        if exclude_outliers:
            df_hist = df[df["amount"] <= q99_val]
            hist_title = f"Phân phối số tiền giao dịch (Dưới 99% quantile: < {format_vnd(q99_val)})"
        else:
            df_hist = df
            hist_title = "Phân phối số tiền giao dịch (Tất cả giao dịch)"
            
        fig_hist = px.histogram(
            df_hist,
            x="amount",
            nbins=50,
            color_discrete_sequence=['#3a7bd5'],
            labels={'amount': 'Số tiền giao dịch (VND)', 'count': 'Tần suất'}
        )
        fig_hist.update_layout(
            title_text=hist_title,
            title_x=0.5,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Thêm thông tin mô tả tập dữ liệu mẫu ở phía dưới
    st.markdown("---")
    st.markdown("#### 📁 Khám phá cấu trúc dữ liệu giao dịch")
    st.write("Dưới đây là một số dòng dữ liệu xem trước ngẫu nhiên:")
    st.dataframe(df.sample(min(10, len(df))), use_container_width=True)


# ================= TAB 2: DANH SÁCH CẢNH BÁO (THEO YÊU CẦU 4 MỨC ĐỘ) =================
with tab_anomalies:
    st.subheader("⚠️ Danh sách Giao dịch được phân loại rủi ro")
    
    st.markdown("""
    Hệ thống đã phân tích và tự động chấm điểm cho toàn bộ tập dữ liệu. Dưới đây là danh sách bộ lọc giúp bạn nhanh chóng trích xuất các thông tin giao dịch bất thường.
    """)
    
    # Cấu hình bộ lọc tương tác
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        # Nhóm mặc định chỉ hiển thị các mức rủi ro cần chú ý để thu hút sự tập trung
        risk_filter = st.multiselect(
            "Chọn mức độ rủi ro hiển thị:",
            options=["Bình thường", "Chú ý", "Nguy hiểm", "Khẩn cấp"],
            default=["Chú ý", "Nguy hiểm", "Khẩn cấp"]
        )
        
    with filter_col2:
        # Lọc theo số tiền giao dịch
        min_amt, max_amt = float(df["amount"].min()), float(df["amount"].max())
        amount_range = st.slider(
            "Chọn phạm vi số tiền giao dịch (VND):",
            min_value=min_amt,
            max_value=max_amt,
            value=(min_amt, max_amt),
            format="%d"
        )
        
    with filter_col3:
        # Tìm kiếm theo Transaction ID hoặc mã tài khoản
        search_query = st.text_input("Tìm kiếm theo ID giao dịch hoặc Mã tài khoản:")
        
    # Áp dụng bộ lọc
    df_filtered = df[df["risk_level"].isin(risk_filter)]
    df_filtered = df_filtered[(df_filtered["amount"] >= amount_range[0]) & (df_filtered["amount"] <= amount_range[1])]
    
    if search_query:
        # Tìm kiếm không phân biệt chữ hoa thường
        search_query_lower = search_query.lower()
        search_mask = (
            df_filtered["transaction_id"].astype(str).str.lower().str.contains(search_query_lower) |
            df_filtered["account_no_hash"].astype(str).str.lower().str.contains(search_query_lower) |
            df_filtered["customer_id_hash"].astype(str).str.lower().str.contains(search_query_lower)
        )
        df_filtered = df_filtered[search_mask]
        
    # Sắp xếp theo mức độ rủi ro (giảm dần) và điểm anomaly score (tăng dần - điểm càng thấp càng bất thường)
    df_filtered = df_filtered.sort_values(by=["risk_code", "anomaly_score"], ascending=[False, True])
    
    # Hiển thị số lượng bản ghi thỏa mãn
    st.info(f"Tìm thấy **{len(df_filtered):,}** giao dịch thỏa mãn các tiêu chí lọc.")
    
    # Định dạng các cột hiển thị đẹp mắt hơn
    display_cols = [
        "transaction_id", "transaction_date", "customer_id_hash", 
        "account_no_hash", "amount", "hour", "is_employee", 
        "risk_level", "anomaly_score", "location", "channel", "transaction_type"
    ]
    
    # Render dữ liệu
    df_display = df_filtered[display_cols].copy()
    
    # Định dạng số tiền có phân cách hàng nghìn cho đẹp
    df_display["amount"] = df_display["amount"].apply(lambda x: f"{x:,.0f}")
    df_display["anomaly_score"] = df_display["anomaly_score"].round(4)
    
    st.dataframe(df_display, use_container_width=True)
    
    # Tải báo cáo
    st.markdown("#### 📥 Xuất báo cáo kết quả lọc")
    
    # Chuẩn bị file CSV/Excel cho nút download
    csv_data = df_filtered[display_cols].to_csv(index=False, encoding='utf-8-sig')
    
    # Tạo excel trong memory
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        df_filtered[display_cols].to_excel(writer, index=False, sheet_name='Giao dịch rủi ro')
    excel_data = excel_buffer.getvalue()
    
    dl_col1, dl_col2, _ = st.columns([1, 1, 3])
    with dl_col1:
        st.download_button(
            label="📥 Tải xuống CSV",
            data=csv_data,
            file_name="giao_dich_rui_ro_report.csv",
            mime="text/csv"
        )
    with dl_col2:
        st.download_button(
            label="📥 Tải xuống Excel",
            data=excel_data,
            file_name="giao_dich_rui_ro_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


# ================= TAB 3: BẢN ĐỒ PHÂN TÁN AI =================
with tab_plotly:
    st.subheader("🧠 Bản đồ phân tán dữ liệu được phân tích bởi AI")
    st.markdown("""
    Thuật toán **Isolation Forest** tính toán sự phân lập của các điểm dữ liệu bằng cách sử dụng các cây quyết định ngẫu nhiên.
    Các điểm dữ liệu có **Số tiền giao dịch cực đại**, **Thực hiện đêm muộn (giờ nhỏ)** hoặc **Thực hiện bởi nhân viên ngân hàng** (biến được phân trọng số) sẽ dễ dàng bị phân lập sớm hơn và nhận điểm rủi ro cao (anomaly_score càng thấp càng bất thường).
    """)
    
    # Để tránh việc quá tải hiển thị (lag trình duyệt) do tập dữ liệu có đến 50,000 bản ghi,
    # chúng ta chỉ vẽ toàn bộ giao dịch bất thường + mẫu ngẫu nhiên giao dịch bình thường
    max_normal_points = 2000
    df_normal = df[df["is_anomaly"] == False]
    df_anomalies = df[df["is_anomaly"] == True]
    
    # Lấy mẫu ngẫu nhiên giao dịch bình thường
    if len(df_normal) > max_normal_points:
        df_normal_sample = df_normal.sample(n=max_normal_points, random_state=42)
    else:
        df_normal_sample = df_normal
        
    df_plot = pd.concat([df_normal_sample, df_anomalies]).sort_values(by="risk_code")
    
    # Chọn chế độ vẽ biểu đồ
    plot_dim = st.radio("Chọn loại biểu đồ hiển thị:", ["Biểu đồ 2D (Giờ giao dịch vs Số tiền)", "Biểu đồ 3D (Giờ giao dịch vs Số tiền vs Nhân viên)"], horizontal=True)
    
    # Bảng màu đại diện cho 4 mức độ rủi ro
    color_map = {
        "Bình thường": "#2ecc71", # Xanh lá
        "Chú ý": "#f1c40f",       # Vàng
        "Nguy hiểm": "#e67e22",    # Cam
        "Khẩn cấp": "#e74c3c"      # Đỏ
    }
    
    if "3D" in plot_dim:
        fig_scatter = px.scatter_3d(
            df_plot,
            x='hour',
            y='amount',
            z='co_nhan_vien',
            color='risk_level',
            color_discrete_map=color_map,
            category_orders={"risk_level": ["Bình thường", "Chú ý", "Nguy hiểm", "Khẩn cấp"]},
            labels={'hour': 'Giờ giao dịch', 'amount': 'Số tiền (VND)', 'co_nhan_vien': 'Là nhân viên'},
            hover_data=['transaction_id', 'risk_level', 'anomaly_score'],
            opacity=0.8
        )
        fig_scatter.update_layout(
            margin=dict(l=0, r=0, b=0, t=30),
            scene=dict(
                xaxis_title='Giờ giao dịch (0-23)',
                yaxis_title='Số tiền (VND)',
                zaxis=dict(title='Là nhân viên (0/1)', tickvals=[0, 1])
            )
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        fig_scatter = px.scatter(
            df_plot,
            x='hour',
            y='amount',
            color='risk_level',
            color_discrete_map=color_map,
            category_orders={"risk_level": ["Bình thường", "Chú ý", "Nguy hiểm", "Khẩn cấp"]},
            labels={'hour': 'Giờ giao dịch (0-23)', 'amount': 'Số tiền giao dịch (VND)'},
            hover_data=['transaction_id', 'risk_level', 'anomaly_score'],
            opacity=0.7
        )
        fig_scatter.update_layout(
            plot_bgcolor='rgba(240, 240, 244, 0.1)',
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis=dict(tickmode='linear', tick0=0, dtick=2)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    st.caption(f"Lưu ý: Để tránh hiện tượng giật lag trình duyệt, biểu đồ trên hiển thị toàn bộ {len(df_anomalies):,} giao dịch bị nghi ngờ bất thường và chọn mẫu ngẫu nhiên {len(df_normal_sample):,} giao dịch bình thường.")


# ================= TAB 4: TRÌNH GIẢ LẬP GIAO DỊCH =================
with tab_simulator:
    st.subheader("🚀 Trình Giả lập Kiểm thử Giao dịch Đơn lẻ")
    st.markdown("""
    Điền các thông tin của một giao dịch giả định bên dưới để mô hình học máy tiến hành đánh giá, tính điểm và phân loại rủi ro ngay lập tức theo thang đo 4 cấp độ.
    """)
    
    sim_col1, sim_col2 = st.columns(2)
    
    with sim_col1:
        st.markdown("#### 📝 Nhập thông số giao dịch")
        sim_amount = st.number_input("Số tiền giao dịch (VND):", min_value=0.0, value=5000000.0, step=500000.0, format="%.2f")
        sim_hour = st.slider("Giờ thực hiện giao dịch (0 - 23h):", min_value=0, max_value=23, value=12)
        sim_employee = st.checkbox("Giao dịch được thực hiện bởi nhân viên ngân hàng?")
        
        btn_simulate = st.button("🚀 Chấm điểm & Phân tích rủi ro")
        
    with sim_col2:
        st.markdown("#### 🔍 Kết quả phân tích rủi ro AI")
        if btn_simulate:
            # Tạo vector input
            co_nhan_vien_val = 1 if sim_employee else 0
            x_sim = pd.DataFrame([[sim_amount, sim_hour, co_nhan_vien_val]], columns=['amount', 'hour', 'co_nhan_vien'])
            
            # Chuẩn hóa theo scaler đã train
            x_sim_scaled = scaler.transform(x_sim)
            
            # Chấm điểm và dự đoán bất thường
            score = iso.decision_function(x_sim_scaled)[0]
            is_anom = iso.predict(x_sim_scaled)[0] == -1
            
            # Phân loại mức độ rủi ro dựa trên ngưỡng động đã xác định từ tập dữ liệu train
            if not is_anom:
                risk_lvl = "Bình thường"
                risk_style = "risk-low"
                rec = "Giao dịch có đặc trưng phân phối bình thường. Hệ thống cho phép thực hiện tự động."
            else:
                # Lấy lại ngưỡng động
                anomaly_scores_all = df.loc[df["is_anomaly"] == True, "anomaly_score"]
                q25_all = anomaly_scores_all.quantile(emergency_percentile / 100.0)
                q50_all = anomaly_scores_all.quantile(0.50)
                
                if score < q25_all:
                    risk_lvl = "Khẩn cấp"
                    risk_style = "risk-critical"
                    rec = "CẢNH BÁO NGUY HIỂM CAO! Giao dịch có mức độ lệch chuẩn nghiêm trọng (Số tiền cực lớn/thực hiện ngoài giờ/nhân viên). Khóa tài khoản tạm thời và yêu cầu xác thực OTP/Sinh trắc học thủ công ngay lập tức."
                elif score < q50_all:
                    risk_lvl = "Nguy hiểm"
                    risk_style = "risk-high"
                    rec = "CẢNH BÁO RỦI RO! Số tiền giao dịch hoặc khung giờ có dấu hiệu đáng nghi ngờ. Yêu cầu giao dịch viên kiểm tra hoặc gọi điện xác nhận với chủ tài khoản."
                else:
                    risk_lvl = "Chú ý"
                    risk_style = "risk-medium"
                    rec = "Giao dịch có sai lệch nhẹ so với thông thường. Ghi nhận nhật ký giám sát đặc biệt nhưng không chặn giao dịch."
            
            # Hiển thị thẻ rủi ro có màu sắc sinh động tương ứng
            st.markdown(f"""
            <div class="risk-card {risk_style}">
                <div style="font-size: 14px; opacity: 0.9; text-transform: uppercase;">Mức độ rủi ro đánh giá</div>
                <div style="font-size: 32px; font-weight: 800; margin: 8px 0;">{risk_lvl.upper()}</div>
                <div style="font-size: 14px; opacity: 0.85;">Điểm bất thường: {score:.5f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 💡 Khuyến nghị xử lý hệ thống:")
            st.info(rec)
            
            # Phân tích chi tiết đặc trưng
            st.markdown("#### 🔬 Phân tích đặc trưng giao dịch:")
            features_analysis = []
            
            # Kiểm tra giờ
            if sim_hour < 6 or sim_hour > 18:
                features_analysis.append("⚠️ Thực hiện ngoài giờ hành chính thông thường (6h - 18h).")
            else:
                features_analysis.append("✅ Thực hiện trong giờ hành chính.")
                
            # Kiểm tra số tiền
            q99_amount = df["amount"].quantile(0.99)
            if sim_amount > q99_amount:
                features_analysis.append(f"⚠️ Giá trị giao dịch ({format_vnd(sim_amount)}) thuộc nhóm 1% lớn nhất của hệ thống (> {format_vnd(q99_amount)}).")
            else:
                features_analysis.append(f"✅ Giá trị giao dịch nằm trong phân phối số tiền thông thường.")
                
            # Kiểm tra nhân viên
            if sim_employee:
                features_analysis.append("⚠️ Thực hiện bởi tài khoản đăng ký là nhân viên ngân hàng (đối tượng giám sát chặt chẽ).")
                
            for item in features_analysis:
                st.write(item)
        else:
            st.write("Vui lòng nhập thông tin bên trái và nhấn nút **🚀 Chấm điểm & Phân tích rủi ro** để xem kết quả.")
