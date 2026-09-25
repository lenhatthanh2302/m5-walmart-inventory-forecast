import os
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# =====================================================================
# CẤU HÌNH TRANG
# =====================================================================
st.set_page_config(
    page_title="M5 Walmart — Probabilistic Inventory",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
CLEAN_CSV = os.path.join(BASE_DIR, "Datasets", "m5_ca1_foods_clean.csv")

LEAD_TIME      = 7
Z_SCORE        = 1.645
CU_CO_RATIO    = 3.0
CRITICAL_RATIO = CU_CO_RATIO / (CU_CO_RATIO + 1)   # 0.75

# =====================================================================
# NẠP DỮ LIỆU VÀ MODEL
# =====================================================================
@st.cache_data
def load_data():
    df = pd.read_csv(CLEAN_CSV, parse_dates=["Date"])
    df = df.sort_values(["SKU", "Date"]).reset_index(drop=True)
    df["Lag_1"]        = df.groupby("SKU")["Sales"].shift(1)
    df["Lag_7"]        = df.groupby("SKU")["Sales"].shift(7)
    df["Baseline_30d"] = df.groupby("SKU")["Sales"].transform(
        lambda x: x.shift(1).rolling(window=30, min_periods=7).mean()
    )
    df = df.dropna(subset=["Lag_1", "Lag_7", "Baseline_30d"]).reset_index(drop=True)
    return df


@st.cache_resource
def load_models():
    models   = joblib.load(os.path.join(MODEL_DIR, "quantile_models.pkl"))
    encoder  = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "features.pkl"))
    return models, encoder, features


@st.cache_data
def load_inventory():
    return pd.read_csv(os.path.join(MODEL_DIR, "inventory_table.csv"))


df = load_data()
quantile_models, le, FEATURES = load_models()
df_inv = load_inventory()
df["SKU_encoded"] = le.transform(df["SKU"])

all_skus = sorted(df["SKU"].unique())

# =====================================================================
# SIDEBAR — CHỌN SKU & THAM SỐ
# =====================================================================
st.sidebar.title("🔧 Cài đặt")
st.sidebar.markdown("---")

selected_sku = st.sidebar.selectbox(
    "Chọn mã sản phẩm (SKU):",
    options=all_skus,
    format_func=lambda s: s.replace("FOODS_", ""),
    help="30 SKU bán chạy nhất tại Walmart CA_1 / FOODS"
)

horizon = st.sidebar.select_slider(
    "Độ dài dự báo (ngày):",
    options=[7, 14, 21, 30],
    value=14,
    help="Số ngày tương lai muốn dự báo"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Giả định vận hành:**")
st.sidebar.markdown(f"- Lead Time: **{LEAD_TIME} ngày**")
st.sidebar.markdown(f"- Service Level: **95%** (z = {Z_SCORE})")
st.sidebar.markdown(f"- Cu/Co Ratio: **{CU_CO_RATIO}** → q* = P{CRITICAL_RATIO*100:.0f}")
st.sidebar.markdown("---")
st.sidebar.caption("Pipeline: Ch1 EDA → Ch2 STL → Ch3 GBR → Ch4 Inventory")
st.sidebar.caption("Dataset: M5 Forecasting Accuracy (Walmart)")

# =====================================================================
# HEADER
# =====================================================================
st.title("🛒 Walmart CA_1 — Dự Báo Xác Suất & Tối Ưu Tồn Kho")
st.markdown(
    f"**SKU đang xem:** `{selected_sku}` &nbsp;|&nbsp; "
    f"**Mô hình:** Quantile GBR — P10 / P50 / P90 &nbsp;|&nbsp; "
    f"**Dự báo:** {horizon} ngày tới"
)

# =====================================================================
# DỮ LIỆU & DỰ BÁO CHO SKU ĐƯỢC CHỌN
# =====================================================================
df_sku_hist = df[df["SKU"] == selected_sku].tail(60).copy()
last_date   = df_sku_hist["Date"].max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

last_row   = df_sku_hist.iloc[-1]
price_val  = last_row["Price"]
snap_val   = last_row["SNAP"]
sku_enc    = last_row["SKU_encoded"]

# Rolling forecast: dùng P50 làm seed lag cho ngày tiếp theo
lag_buffer = df_sku_hist["Sales"].tolist()
future_rows = []

for i, date in enumerate(future_dates):
    lag1  = lag_buffer[-1]
    lag7  = lag_buffer[-7] if len(lag_buffer) >= 7 else lag_buffer[0]
    b30   = np.mean(lag_buffer[-30:]) if len(lag_buffer) >= 7 else np.mean(lag_buffer)
    dow   = date.dayofweek
    row = {
        "Lag_1": lag1, "Lag_7": lag7, "Baseline_30d": b30,
        "IsWeekend": 1 if dow >= 5 else 0, "IsEvent": 0,
        "Price": price_val, "SNAP": snap_val,
        "DayOfWeek": dow + 1, "Month": date.month,
        "SKU_encoded": sku_enc,
    }
    future_rows.append(row)
    X_tmp = pd.DataFrame([row])[FEATURES]
    seed  = float(np.maximum(quantile_models["P50"].predict(X_tmp), 0)[0])
    lag_buffer.append(seed)

X_future   = pd.DataFrame(future_rows)[FEATURES]
p10_future = np.maximum(quantile_models["P10"].predict(X_future), 0)
p50_future = np.maximum(quantile_models["P50"].predict(X_future), 0)
p90_future = np.maximum(quantile_models["P90"].predict(X_future), 0)

# Dữ liệu tồn kho của SKU đang chọn
sku_short = selected_sku.replace("FOODS_", "")
inv_row   = df_inv[df_inv["SKU"] == sku_short].iloc[0] if not df_inv[df_inv["SKU"] == sku_short].empty else None

# =====================================================================
# 3 TABS CHÍNH
# =====================================================================
tab1, tab2, tab3 = st.tabs([
    "📈 Dự Báo Xác Suất",
    "🗂️ Chiến Lược Tồn Kho",
    "🌐 Toàn Danh Mục (30 SKU)",
])

# ─────────────────────────────────────────────────────────────────────
# TAB 1 — DỰ BÁO XÁC SUẤT
# ─────────────────────────────────────────────────────────────────────
with tab1:
    last_hist_val = df_sku_hist["Sales"].iloc[-1]
    fut_x = [last_date] + list(future_dates)
    f_p10 = [last_hist_val] + list(p10_future)
    f_p50 = [last_hist_val] + list(p50_future)
    f_p90 = [last_hist_val] + list(p90_future)

    fig_fcst = go.Figure()

    fig_fcst.add_trace(go.Scatter(
        x=list(df_sku_hist["Date"]), y=list(df_sku_hist["Sales"]),
        mode="lines+markers", name="Thực tế ($Y_t$)",
        line=dict(color="#2c3e50", width=2), marker=dict(size=4),
    ))
    fig_fcst.add_trace(go.Scatter(
        x=fut_x, y=f_p90, mode="lines",
        line=dict(width=0), showlegend=False,
    ))
    fig_fcst.add_trace(go.Scatter(
        x=fut_x, y=f_p10, mode="lines", fill="tonexty",
        fillcolor="rgba(231, 76, 60, 0.15)",
        line=dict(width=0), name=f"Dải băng rủi ro ({horizon} ngày)",
    ))
    fig_fcst.add_trace(go.Scatter(
        x=fut_x, y=f_p90, mode="lines",
        name="P90 — Kịch bản lạc quan",
        line=dict(color="#e74c3c", dash="dot", width=1.5),
    ))
    fig_fcst.add_trace(go.Scatter(
        x=fut_x, y=f_p50, mode="lines+markers",
        name="P50 — Kỳ vọng trung lập",
        line=dict(color="#e67e22", dash="dash", width=2.5),
        marker=dict(size=5),
    ))
    fig_fcst.add_trace(go.Scatter(
        x=fut_x, y=f_p10, mode="lines",
        name="P10 — Kịch bản thận trọng",
        line=dict(color="#27ae60", dash="dot", width=1.5),
    ))
    fig_fcst.add_vline(
        x=str(last_date), line_width=1.5,
        line_dash="dash", line_color="#7f8c8d",
        annotation_text="Ranh giới dự báo",
        annotation_position="top left",
    )
    fig_fcst.update_layout(
        title=f"Dải Băng Xác Suất {horizon} Ngày Tới — {selected_sku}",
        xaxis_title="Ngày giao dịch",
        yaxis_title="Doanh số (sản phẩm/ngày)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
        plot_bgcolor="#fafafa",
    )
    st.plotly_chart(fig_fcst, use_container_width=True)

    # Thống kê dự báo nhanh
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("P10 — Ngày thấp nhất (dự báo)", f"{p10_future.min():.0f} đv")
    col_b.metric("P50 — Kỳ vọng TB",              f"{p50_future.mean():.1f} đv")
    col_c.metric("P90 — Ngày cao nhất (dự báo)",  f"{p90_future.max():.0f} đv")
    col_d.metric("Độ rộng dải băng (TB)",         f"{(p90_future - p10_future).mean():.1f} đv",
                 help="P90 - P10 trung bình — độ không chắc chắn của dự báo")

    # [Fix 3] Cảnh báo khi P50 thấp hơn đáng kể so với lịch sử gần đây
    # Bỏ qua SKU intermittent demand (hist_mean < 1) vì P50=0 là dự báo hợp lệ
    hist_mean = df_sku_hist["Sales"].mean()
    p50_mean  = p50_future.mean()
    if hist_mean >= 1.0 and (hist_mean - p50_mean) / hist_mean > 0.15:
        gap_pct = (hist_mean - p50_mean) / hist_mean * 100
        st.info(
            f"ℹ️ **P50 dự báo ({p50_mean:.1f} đv/ngày) thấp hơn ~{gap_pct:.0f}% so với "
            f"doanh số bình quân 60 ngày gần nhất ({hist_mean:.1f} đv/ngày).** "
            f"Điều này có thể phản ánh: (1) xu hướng giảm mùa vụ thực sự, "
            f"hoặc (2) mô hình đang underforecast do thiếu feature sự kiện (SNAP, holiday). "
            f"Nên đối chiếu với lịch sử cùng kỳ năm trước trước khi đặt hàng."
        )

    # Nút xuất dữ liệu
    st.markdown("---")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        hist_exp = df_sku_hist[["Date", "Sales", "Price"]].copy()
        hist_exp.columns = ["Ngày", "Doanh số thực tế", "Giá bán"]
        hist_exp.to_excel(writer, sheet_name="Lịch sử", index=False)

        fcst_exp = pd.DataFrame({
            "Ngày": list(future_dates),
            "P10":  [round(v) for v in p10_future],
            "P50":  [round(v) for v in p50_future],
            "P90":  [round(v) for v in p90_future],
        })
        fcst_exp.to_excel(writer, sheet_name=f"Dự báo {horizon} ngày", index=False)
    buf.seek(0)

    col_x1, col_x2 = st.columns(2)
    with col_x1:
        st.download_button(
            "📥 Tải Excel — Lịch sử & Dự báo",
            data=buf, file_name=f"forecast_{selected_sku}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with col_x2:
        html_bytes = fig_fcst.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")
        st.download_button(
            "🌐 Tải HTML (mở → In PDF)",
            data=html_bytes, file_name=f"forecast_{selected_sku}.html",
            mime="text/html", use_container_width=True,
        )

# ─────────────────────────────────────────────────────────────────────
# TAB 2 — CHIẾN LƯỢC TỒN KHO
# ─────────────────────────────────────────────────────────────────────
with tab2:
    if inv_row is not None:
        avg_s  = round(float(inv_row["TB Sales/ngày"]), 1)
        std_s  = round(float(inv_row["Std Sales"]), 1)
        ss_val = int(inv_row["Safety Stock"])
        rop_val = int(inv_row["Reorder Point"])
        q_star = int(inv_row["Newsvendor q*"])
        cv_val = round(std_s / avg_s, 2) if avg_s > 0 else 0

        # Đánh giá mức độ biến động
        if cv_val >= 0.5:
            cv_label, cv_color = "CAO — Cần Safety Stock lớn", "🔴"
        elif cv_val >= 0.25:
            cv_label, cv_color = "TRUNG BÌNH", "🟡"
        else:
            cv_label, cv_color = "THẤP — Dễ kiểm soát", "🟢"

        st.markdown(f"### Phân tích Rủi ro Tồn kho — `{selected_sku}`")

        m1, m2, m3, m4, m5 = st.columns(5)
        ss_days = round(ss_val / avg_s, 1) if avg_s > 0 else 0
        q_review = round(avg_s * 7)

        m1.metric("TB Bán/ngày",     f"{avg_s} đv")
        m2.metric("Độ lệch chuẩn",   f"{std_s} đv")
        m3.metric("CV (Std/Mean)",    f"{cv_val:.2f}", help=cv_color + " " + cv_label)
        m4.metric("Safety Stock",     f"{ss_val} đv",
                  help=(f"SS = {Z_SCORE} × {std_s} × √{LEAD_TIME} = {ss_val} đv "
                        f"(≈ {ss_days} ngày doanh số)"))
        m5.metric("Reorder Point",    f"{rop_val} đv",
                  help=f"ROP = {avg_s}×{LEAD_TIME} + {ss_val} = {rop_val}")

        # [Fix 2] Cảnh báo CV > 1 — Normal approximation không còn chính xác
        if cv_val > 1.0:
            st.error(
                f"⚠️ **CV = {cv_val:.2f} > 1.0** — Nhu cầu của SKU này có phân phối lệch rất cao "
                f"(có nhiều ngày bằng 0 hoặc spike lớn). Xấp xỉ phân phối chuẩn không còn phù hợp, "
                f"Safety Stock {ss_val} đv **có thể bị ước tính thấp hơn thực tế**. "
                f"Cân nhắc nâng z-score hoặc dùng phân phối Negative Binomial."
            )

        # [Fix 7] Lưu ý về tính mùa vụ trong CV
        st.caption(
            f"📌 CV được tính trên toàn bộ chuỗi lịch sử (2013–2016). "
            f"Trong các tháng cao điểm (Nov–Dec / SNAP week), biến động thực tế có thể cao hơn đáng kể — "
            f"Safety Stock nên được review theo quý."
        )

        st.markdown("---")

        # Biểu đồ waterfall: cấu trúc ROP
        fig_rop = go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative", "relative", "total"],
            x=["Nhu cầu trong<br>Lead Time", "Safety Stock<br>(Dự phòng)", "Reorder Point<br>(Tổng)"],
            y=[avg_s * LEAD_TIME, ss_val, 0],
            text=[f"{avg_s * LEAD_TIME:.0f} đv", f"+{ss_val} đv", f"{rop_val} đv"],
            textposition="outside",
            connector=dict(line=dict(color="#95a5a6")),
            increasing=dict(marker=dict(color="#3498db")),
            totals=dict(marker=dict(color="#e74c3c")),
        ))
        fig_rop.update_layout(
            title=f"Cấu Trúc Reorder Point — {selected_sku}",
            yaxis_title="Số lượng (sản phẩm)",
            height=360,
            plot_bgcolor="#fafafa",
            showlegend=False,
        )
        st.plotly_chart(fig_rop, use_container_width=True)

        # [Fix 4] Lượng đặt hàng định kỳ gợi ý (Review Period = 7 ngày)
        st.markdown("---")
        col_inv1, col_inv2 = st.columns(2)
        with col_inv1:
            st.markdown(f"**Lượng Đặt Hàng Gợi Ý (Continuous Review):**")
            st.markdown(f"`Q = {q_review} đv`")
            st.caption(
                f"Q = TB Bán/ngày × Review Period (7 ngày) = {avg_s} × 7 = {q_review} đv. "
                f"Đặt hàng khi tồn kho chạm ROP = {rop_val} đv."
            )
        with col_inv2:
            st.markdown(f"**Lượng Nhập Hàng Tối Ưu (Newsvendor — Single Period):**")
            st.markdown(f"`q* = {q_star} đv` — Quantile P{CRITICAL_RATIO*100:.0f}")
            st.caption(
                f"Critical Ratio = Cu/(Cu+Co) = {CU_CO_RATIO}/{CU_CO_RATIO+1:.0f} = {CRITICAL_RATIO:.2f}. "
                f"Với giả định chi phí hết hàng (Cu) gấp {CU_CO_RATIO:.0f}× chi phí tồn kho thừa (Co), "
                f"mô hình Newsvendor khuyến nghị {q_star} đv — phù hợp cho **đặt hàng hàng ngày** "
                f"(sản phẩm tươi/perishable). "
                f"⚠️ Không thay thế ROP — hai framework hoạt động độc lập."
            )
    else:
        st.warning("Không tìm thấy dữ liệu tồn kho cho SKU này.")

# ─────────────────────────────────────────────────────────────────────
# TAB 3 — TOÀN DANH MỤC 30 SKU (bản đồ rủi ro)
# ─────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Bản Đồ Rủi Ro Danh Mục — 30 SKU")
    st.caption(
        "Mỗi bong bóng = 1 SKU. "
        "Trục X = doanh số trung bình (volume), "
        "Trục Y = Safety Stock cần dự phòng (risk buffer), "
        "Kích thước = Reorder Point. "
        "SKU ở góc trên phải = bán nhiều nhưng biến động cao → cần buffer lớn nhất."
    )

    df_plot = df_inv.copy()
    df_plot["CV"] = (df_plot["Std Sales"] / df_plot["TB Sales/ngày"]).round(2)
    df_plot["Chiến lược"] = df_plot["CV"].apply(
        lambda v: "Rủi ro cao" if v >= 0.5 else ("Trung bình" if v >= 0.25 else "Ổn định")
    )
    df_plot["SKU_highlight"] = df_plot["SKU"] == sku_short

    fig_scatter = px.scatter(
        df_plot,
        x="TB Sales/ngày", y="Safety Stock",
        size="Reorder Point", color="Chiến lược",
        hover_name="SKU",
        hover_data={
            "TB Sales/ngày": ":.1f",
            "Std Sales": ":.1f",
            "CV": ":.2f",
            "Safety Stock": True,
            "Reorder Point": True,
            "Newsvendor q*": True,
            "Chiến lược": False,
        },
        color_discrete_map={
            "Rủi ro cao": "#e74c3c",
            "Trung bình": "#f39c12",
            "Ổn định":    "#27ae60",
        },
        size_max=45,
        labels={
            "TB Sales/ngày": "Doanh số TB (đv/ngày)",
            "Safety Stock":  "Safety Stock cần dự phòng (đv)",
        },
    )

    # Đánh dấu SKU đang chọn
    sku_row = df_plot[df_plot["SKU"] == sku_short]
    if not sku_row.empty:
        fig_scatter.add_trace(go.Scatter(
            x=sku_row["TB Sales/ngày"], y=sku_row["Safety Stock"],
            mode="markers+text",
            marker=dict(symbol="star", size=18, color="#9b59b6", line=dict(color="white", width=1)),
            text=[f"◀ {sku_short}"],
            textposition="middle right",
            textfont=dict(size=12, color="#9b59b6"),
            name="SKU đang xem",
            showlegend=True,
        ))

    fig_scatter.update_layout(
        height=500,
        plot_bgcolor="#fafafa",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # [Fix 6] Ghi chú nếu không có SKU nào trong nhóm "Ổn định"
    n_stable = (df_plot["Chiến lược"] == "Ổn định").sum()
    if n_stable == 0:
        st.info(
            "ℹ️ **Không có SKU nào trong nhóm 'Ổn định' (CV < 0.25)** — "
            "toàn bộ 30 SKU FOODS tại CA_1 đều có biến động nhu cầu đáng kể. "
            "Đây là đặc trưng của ngành thực phẩm (SNAP weeks, weekday/weekend pattern), "
            "không phải lỗi mô hình."
        )

    # Bảng tổng hợp đầy đủ
    st.markdown("---")
    st.markdown("**Bảng chi tiết — toàn bộ 30 SKU:**")

    df_display = df_inv.copy()
    df_display["CV"] = (df_display["Std Sales"] / df_display["TB Sales/ngày"]).round(2)
    df_display = df_display.sort_values("TB Sales/ngày", ascending=False).reset_index(drop=True)

    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "SKU":            st.column_config.TextColumn("Mã SKU"),
            "TB Sales/ngày":  st.column_config.NumberColumn("TB Sales/ngày", format="%.1f đv"),
            "Std Sales":      st.column_config.NumberColumn("Độ lệch chuẩn", format="%.1f"),
            "CV":             st.column_config.ProgressColumn("CV (biến động)", min_value=0, max_value=1, format="%.2f"),
            "Safety Stock":   st.column_config.NumberColumn("Safety Stock"),
            "Reorder Point":  st.column_config.NumberColumn("Reorder Point"),
            "Newsvendor q*":  st.column_config.NumberColumn("Newsvendor q*"),
        }
    )
