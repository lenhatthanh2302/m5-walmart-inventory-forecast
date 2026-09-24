# =============================================================================
# ĐỒ ÁN NHÓM — MÔN: CÁC MÔ HÌNH DỰ BÁO TRONG KINH DOANH
# Đề tài : Pipeline Dự báo Xác suất & Tối ưu hóa Chiến lược Tồn kho
#          cho Chuỗi Bán lẻ Đa SKU (Dataset: M5 Forecasting – Walmart)
# Dataset : M5 Forecasting Accuracy (Kaggle) — Store CA_1, Category FOODS
# =============================================================================

#%%
# =====================================================================
# 0. KHAI BÁO THƯ VIỆN & CẤU HÌNH TOÀN CỤC
# =====================================================================
import os
import warnings
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.gridspec import GridSpec
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.filters.hp_filter import hpfilter
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

# --- Cấu hình đường dẫn ---
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Datasets")
OUTPUT_FILE = os.path.join(DATASET_DIR, "m5_ca1_foods_clean.csv")

# --- Tham số subset ---
STORE_FILTER  = "CA_1"
CAT_FILTER    = "FOODS"
TOP_N_ITEMS   = 30
START_DATE    = "2013-01-01"

# --- Phong cách biểu đồ thống nhất toàn pipeline ---
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams["font.family"] = "sans-serif"

# Bảng màu chuẩn — dùng xuyên suốt tất cả chương
COLOR_ACTUAL   = "black"        # Dữ liệu thực tế
COLOR_TREND    = "#d62728"      # Xu hướng (đỏ đậm)
COLOR_SEASONAL = "#2ca02c"      # Mùa vụ (xanh lá)
COLOR_CYCLE    = "#ff7f0e"      # Chu kỳ (cam)
COLOR_RESID    = "#9467bd"      # Nhiễu ngẫu nhiên (tím)
COLOR_FORECAST = "#2980b9"      # Dự báo P50 (xanh dương)
COLOR_BAND     = "skyblue"      # Dải băng P10–P90
COLOR_NAIVE    = "#7f7f7f"      # Baseline Naive (xám)
COLOR_GBR      = "#e377c2"      # GBR model (hồng)

print("=" * 70)
print("PIPELINE DỰ BÁO TỒN KHO — M5 WALMART (CA_1 / FOODS / TOP 30 SKU)")
print("=" * 70)
print(f"  Base directory : {BASE_DIR}")
print(f"  Dataset folder : {DATASET_DIR}")


#%%
# =====================================================================
# 1. TIỀN XỬ LÝ DỮ LIỆU — NẠP, LỌC, GẮN NHÃN, XUẤT FILE SẠCH
# =====================================================================
# Mục tiêu: từ 3 file gốc (~30k SKU, ~5 năm) → 1 file sạch tập trung
# vào CA_1 / FOODS / Top 30 SKU từ năm 2013, sẵn sàng cho phân tích.
# =====================================================================

print("\n[1/5] Đang đọc sales_train_evaluation.csv ...")
sales_wide = pd.read_csv(os.path.join(DATASET_DIR, "sales_train_evaluation.csv"))
print(f"      Kích thước gốc: {sales_wide.shape[0]:,} dòng × {sales_wide.shape[1]:,} cột")

# Lọc đúng cửa hàng và danh mục cần phân tích
mask = (sales_wide["store_id"] == STORE_FILTER) & (sales_wide["cat_id"] == CAT_FILTER)
sales_sub = sales_wide[mask].copy()
print(f"      Sau khi lọc {STORE_FILTER} / {CAT_FILTER}: {len(sales_sub):,} mặt hàng")

# Giữ lại Top N SKU bán chạy nhất để tập trung phân tích
day_cols = [c for c in sales_sub.columns if c.startswith("d_")]
sales_sub["total_sales"] = sales_sub[day_cols].sum(axis=1)
top_items = sales_sub.nlargest(TOP_N_ITEMS, "total_sales")["item_id"].tolist()
sales_sub = sales_sub[sales_sub["item_id"].isin(top_items)].copy()
print(f"      Sau khi giữ Top {TOP_N_ITEMS} SKU: {len(sales_sub)} mặt hàng")

# ── Chuyển Wide → Long (mỗi ngày thành 1 dòng) ──────────────────────
print("\n[2/5] Đang chuyển Wide → Long format ...")
id_cols    = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
sales_long = sales_sub.melt(
    id_vars   = id_cols,
    value_vars = day_cols,
    var_name  = "d",
    value_name = "sales"
)
print(f"      Kích thước sau melt: {len(sales_long):,} dòng")

# ── Gắn ngày thực và thông tin sự kiện từ calendar.csv ──────────────
print("\n[3/5] Đang merge với calendar.csv ...")
calendar = pd.read_csv(os.path.join(DATASET_DIR, "calendar.csv"))

# Chỉ giữ những cột cần thiết — bỏ thông tin thừa không dùng đến
cal_cols = ["d", "date", "wm_yr_wk", "wday", "month", "year",
            "event_name_1", "event_type_1", "snap_CA"]
calendar  = calendar[cal_cols].copy()
calendar["date"] = pd.to_datetime(calendar["date"])

sales_long = sales_long.merge(calendar, on="d", how="left")

# Cắt về giai đoạn phân tích — từ 2013 trở đi để đủ 3+ năm dữ liệu
sales_long = sales_long[sales_long["date"] >= START_DATE].copy()
print(f"      Sau khi lọc từ {START_DATE}: {len(sales_long):,} dòng")

# ── Gắn giá bán theo tuần từ sell_prices.csv ────────────────────────
print("\n[4/5] Đang merge với sell_prices.csv ...")
prices = pd.read_csv(os.path.join(DATASET_DIR, "sell_prices.csv"))
prices = prices[prices["store_id"] == STORE_FILTER].copy()

sales_long = sales_long.merge(
    prices[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
    on  = ["store_id", "item_id", "wm_yr_wk"],
    how = "left"
)

# ── Làm sạch và chuẩn hoá tên cột ───────────────────────────────────
print("\n[5/5] Đang làm sạch và xuất file ...")

df = sales_long[[
    "item_id", "date", "sales", "sell_price",
    "wday", "month", "year",
    "event_name_1", "event_type_1", "snap_CA"
]].copy()

df = df.rename(columns={
    "item_id"      : "SKU",
    "date"         : "Date",
    "sales"        : "Sales",
    "sell_price"   : "Price",
    "wday"         : "DayOfWeek",
    "month"        : "Month",
    "year"         : "Year",
    "event_name_1" : "EventName",
    "event_type_1" : "EventType",
    "snap_CA"      : "SNAP"
})

# Sắp xếp theo thứ tự thời gian trong từng SKU
df = df.sort_values(["SKU", "Date"]).reset_index(drop=True)

# Giá bán đôi khi bị khuyết vào đầu kỳ — fill forward trong từng SKU
df["Price"] = df.groupby("SKU")["Price"].ffill().bfill()

# Chuẩn hoá cột sự kiện — ngày bình thường ghi "No_Event" thay vì để trống
# (tránh pandas đọc lại "None" string thành NaN)
df["EventName"] = df["EventName"].fillna("No_Event")
df["EventType"] = df["EventType"].fillna("No_Event")

# Tạo thêm 2 cột nhị phân phục vụ feature engineering ở chương sau
df["IsWeekend"] = df["DayOfWeek"].isin([1, 2]).astype(int)   # 1=T7, 2=CN (M5 convention)
df["IsEvent"]   = (df["EventName"] != "No_Event").astype(int)

# Tính doanh thu để phân tích Unit Economics ở chương 1
df["Revenue"] = df["Sales"] * df["Price"]

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

# ── Báo cáo tóm tắt ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TIỀN XỬ LÝ HOÀN TẤT — TÓM TẮT DỮ LIỆU ĐẦU RA:")
print("=" * 70)
print(f"  Cửa hàng      : {STORE_FILTER}  (Walmart California)")
print(f"  Danh mục      : {CAT_FILTER}  (Thực phẩm)")
print(f"  Số SKU        : {df['SKU'].nunique()} mặt hàng")
print(f"  Giai đoạn     : {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"  Tổng dòng dữ liệu : {len(df):,}")
print(f"  Các cột       : {list(df.columns)}")
print(f"\n  File đã lưu   : {OUTPUT_FILE}")
print(f"  Kích thước    : {os.path.getsize(OUTPUT_FILE)/1024:.1f} KB")
print("=" * 70)
print("\nMẫu dữ liệu (5 dòng đầu):")
print(df.head().to_string(index=False))


#%%
# =====================================================================
# 2. CHƯƠNG 1 — PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA DASHBOARD)
# =====================================================================
# Mục tiêu: hiểu hành vi doanh số trước khi xây mô hình — xu hướng
# tổng thể, hình thái phân phối, mức độ biến động giữa các SKU và mối tương quan giữa các biến số.
# =====================================================================

print("\n" + "=" * 70)
print("CHƯƠNG 1 — PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA)")
print("=" * 70)

# Nạp file sạch vừa tạo ở bước trước
df = pd.read_csv(OUTPUT_FILE, parse_dates=["Date"])

# ── Kiểm tra dữ liệu thiếu trước khi vẽ ─────────────────────────────
print("\n[Kiểm tra Missing Values]")
missing = df.isnull().sum()
missing = missing[missing > 0]
if missing.empty:
    print("  Không có giá trị thiếu — dữ liệu sạch.")
else:
    print(missing)

# ── Chuẩn bị dữ liệu tổng hợp theo ngày (tất cả 30 SKU) ─────────────
daily = df.groupby("Date").agg(
    Sales   = ("Sales",   "sum"),
    Revenue = ("Revenue", "sum")
).reset_index()

daily["Sales_MA7"]   = daily["Sales"].rolling(window=7, min_periods=1).mean()
daily["Revenue_MA7"] = daily["Revenue"].rolling(window=7, min_periods=1).mean()

print(f"\n  Giai đoạn phân tích : {daily['Date'].min().date()} → {daily['Date'].max().date()}")
print(f"  Số ngày quan sát    : {len(daily):,} ngày")
print(f"  Tổng doanh số TB/ngày (30 SKU) : {daily['Sales'].mean():.0f} sản phẩm")
print(f"  Tổng doanh thu TB/ngày (30 SKU): ${daily['Revenue'].mean():.0f}")

# =====================================================================
# FIGURE 1 — PHƯƠNG DIỆN 1: ĐỘNG THÁI CHUỖI THỜI GIAN
# =====================================================================
fig1, ax1 = plt.subplots(figsize=(16, 5))

ax1.plot(daily["Date"], daily["Sales"],
         color="lightgray", linewidth=1, label="Doanh số thực tế ($Y_t$)")
ax1.plot(daily["Date"], daily["Sales_MA7"],
         color=COLOR_TREND, linewidth=2.5, label="Trung bình trượt 7 ngày (MA7)")

# Đánh dấu tự động ngày có doanh số cao nhất — thường là ngày lễ lớn
peak_idx  = daily["Sales"].idxmax()
peak_date = daily["Date"][peak_idx]
peak_val  = daily["Sales"][peak_idx]
ax1.annotate(
    "Đỉnh doanh số\n(Ngày lễ / Sự kiện lớn)",
    xy=(peak_date, peak_val),
    xytext=(peak_date - pd.Timedelta(days=150), peak_val * 0.90),
    arrowprops=dict(facecolor="black", arrowstyle="->"),
    fontsize=10, fontweight="bold"
)

ax1.set_title("PHƯƠNG DIỆN 1: ĐỘNG THÁI CHUỖI THỜI GIAN & XU HƯỚNG CỐT LÕI\n"
              "Tổng 30 SKU — Walmart CA_1 / FOODS ($Y_t$ = Doanh số Ngày)",
              fontsize=13, fontweight="bold")
ax1.set_ylabel("Tổng doanh số 30 SKU (Sản phẩm/ngày)", fontsize=11, fontweight="bold")
ax1.set_xlabel("Thời gian quan sát", fontsize=11, fontweight="bold")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)
ax1.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=10)
ax1.grid(True, linestyle=":", alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "ch1_pd1_timeseries.png"), dpi=150, bbox_inches="tight")
plt.show()
print("\n[✓] Đã lưu: ch1_pd1_timeseries.png")

# =====================================================================
# FIGURE 2 — PHƯƠNG DIỆN 2: PHÂN PHỐI + TƯƠNG QUAN PEARSON
# =====================================================================
fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle("PHƯƠNG DIỆN 2: ĐẶC TÍNH PHÂN PHỐI & TƯƠNG QUAN BIẾN SỐ",
              fontsize=13, fontweight="bold", y=1.01)

# --- Histogram phân phối doanh số ---
# Dùng doanh số cấp SKU-ngày để thấy rõ phân phối thực tế
sns.histplot(df["Sales"], bins=40, kde=True,
             color="steelblue", alpha=0.6, ax=ax2)

mean_val   = df["Sales"].mean()
median_val = df["Sales"].median()
ax2.axvline(mean_val,   color="darkred",    linestyle="--", linewidth=1.8,
            label=f"Mean: {mean_val:.1f}")
ax2.axvline(median_val, color="darkorange", linestyle="-",  linewidth=2,
            label=f"Median: {median_val:.1f}")

ax2.set_title("HÌNH THÁI PHÂN PHỐI\n(Lệch phải — đặc trưng bán lẻ thực phẩm)",
              fontsize=11, fontweight="bold")
ax2.set_xlabel("Doanh số (Sản phẩm/ngày/SKU)", fontsize=10, fontweight="bold")
ax2.set_ylabel("Tần suất quan sát", fontsize=10, fontweight="bold")
ax2.legend(fontsize=10, frameon=True)

# --- Ma trận tương quan Pearson ---
cols_corr   = ["Sales", "Price", "SNAP", "IsWeekend", "IsEvent", "Revenue"]
corr_matrix = df[cols_corr].corr()

sns.heatmap(corr_matrix, annot=True, cmap="Blues",
            vmin=-1, vmax=1, fmt=".2f",
            linewidths=0.5, ax=ax3, annot_kws={"size": 10})

ax3.set_title("MA TRẬN TƯƠNG QUAN PEARSON ($r$)\n(Sales, Price, SNAP, Weekend, Event, Revenue)",
              fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "ch1_pd2_dist_corr.png"), dpi=150, bbox_inches="tight")
plt.show()
print("[✓] Đã lưu: ch1_pd2_dist_corr.png")

# =====================================================================
# FIGURE 3 — PHƯƠNG DIỆN 3: BOXPLOT NGOẠI LAI THEO SKU
# =====================================================================
fig3, ax4 = plt.subplots(figsize=(14, 6))

# Chọn Top 10 SKU bán chạy nhất để boxplot không bị quá chật
top10    = df.groupby("SKU")["Sales"].sum().nlargest(10).index.tolist()
df_top10 = df[df["SKU"].isin(top10)].copy()

# Rút gọn tên SKU cho trục X dễ đọc (bỏ tiền tố "FOODS_")
df_top10["SKU_short"] = df_top10["SKU"].str.replace("FOODS_", "", regex=False)

# Sắp xếp theo median giảm dần để biểu đồ có thứ tự logic
order = (df_top10.groupby("SKU_short")["Sales"]
         .median().sort_values(ascending=False).index.tolist())

sns.boxplot(data=df_top10, x="SKU_short", y="Sales", order=order,
            color="lightsteelblue",
            flierprops=dict(marker=".", color="firebrick", alpha=0.5, markersize=4),
            ax=ax4)

ax4.set_title("PHƯƠNG DIỆN 3: BIẾN ĐỘNG & NGOẠI LAI THEO SKU (TOP 10)\n"
              "Chấm đỏ = Outlier vượt ngưỡng IQR × 1.5  |  Sắp xếp theo Median giảm dần",
              fontsize=13, fontweight="bold")
ax4.set_xlabel("Mã SKU (bỏ tiền tố FOODS_)", fontsize=11, fontweight="bold")
ax4.set_ylabel("Doanh số (Sản phẩm/ngày)", fontsize=11, fontweight="bold")
ax4.tick_params(axis="x", rotation=30, labelsize=9)
ax4.grid(True, linestyle=":", alpha=0.7, axis="y")

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "ch1_pd3_boxplot.png"), dpi=150, bbox_inches="tight")
plt.show()
print("[✓] Đã lưu: ch1_pd3_boxplot.png")

# =====================================================================
# FIGURE 4 — UNIT ECONOMICS: DOANH THU THEO THÁNG
# =====================================================================
monthly = df.copy()
monthly["YearMonth"] = monthly["Date"].dt.to_period("M").dt.to_timestamp()
monthly = monthly.groupby("YearMonth")["Revenue"].sum().reset_index()
monthly["Revenue_MA3"] = monthly["Revenue"].rolling(window=3, min_periods=1).mean()

fig4, ax5 = plt.subplots(figsize=(14, 5))
ax5.bar(monthly["YearMonth"], monthly["Revenue"],
        color="steelblue", alpha=0.5, width=20, label="Doanh thu tháng (USD)")
ax5.plot(monthly["YearMonth"], monthly["Revenue_MA3"],
         color=COLOR_TREND, linewidth=2.5, marker="o", markersize=4,
         label="Trung bình trượt 3 tháng (MA3)")

ax5.set_title(
    "UNIT ECONOMICS — DOANH THU THÁNG CỦA 30 SKU (CA_1 / FOODS)\n"
    "Revenue$_t$ = Sales$_t$ × Price$_t$",
    fontsize=13, fontweight="bold"
)
ax5.set_ylabel("Tổng doanh thu (USD/tháng)", fontsize=11, fontweight="bold")
ax5.set_xlabel("Tháng báo cáo", fontsize=11, fontweight="bold")
ax5.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)
ax5.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=10)
ax5.grid(True, linestyle=":", alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "ch1_pd4_unit_economics.png"), dpi=150, bbox_inches="tight")
plt.show()
print("[✓] Đã lưu: ch1_pd4_unit_economics.png")

print("\n" + "=" * 70)
print("CHƯƠNG 1 HOÀN TẤT — 4 biểu đồ đã xuất.")
print("=" * 70)


#%%
# =====================================================================
# 3. CHƯƠNG 2 — PHÂN TÁCH CHUỖI THỜI GIAN (STL + HP FILTER)
# =====================================================================
# Mục tiêu: tách doanh số thành 4 thành phần riêng biệt để hiểu rõ
# cấu trúc bên trong của chuỗi trước khi xây mô hình dự báo.
#
# Công thức: Y_t = T_t + C_t + S_t + R_t
#   T_t : Xu hướng dài hạn   (HP Filter tách từ TC)
#   C_t : Chu kỳ trung hạn   (HP Filter tách từ TC)
#   S_t : Mùa vụ tuần        (STL, period=7)
#   R_t : Nhiễu ngẫu nhiên   (phần dư còn lại)
# =====================================================================
print("\n" + "=" * 70)
print("CHƯƠNG 2 — PHÂN TÁCH CHUỖI THỜI GIAN (STL + HP FILTER)")
print("=" * 70)

df = pd.read_csv(OUTPUT_FILE, parse_dates=["Date"])

# Chọn SKU đại diện — SKU có tổng doanh số cao nhất trong Top 30
sku_rep  = df.groupby("SKU")["Sales"].sum().idxmax()
df_sku   = df[df["SKU"] == sku_rep].set_index("Date")["Sales"].asfreq("D")

print(f"\n  SKU đại diện    : {sku_rep}")
print(f"  Giai đoạn       : {df_sku.index.min().date()} → {df_sku.index.max().date()}")
print(f"  Tổng quan sát   : {len(df_sku):,} ngày")
print(f"  Doanh số TB     : {df_sku.mean():.1f}  |  Max: {df_sku.max()}  |  Min: {df_sku.min()}")

# ── Bước 1: STL Decomposition (period=7 — chu kỳ tuần) ──────────────
# robust=True giúp mô hình ít bị ảnh hưởng bởi outlier ngày lễ
print("\n[1/2] Đang chạy STL Decomposition (period=7, robust=True) ...")
stl    = STL(df_sku, period=7, robust=True).fit()
S_comp = stl.seasonal   # Thành phần mùa vụ tuần
R_comp = stl.resid      # Nhiễu ngẫu nhiên
TC     = stl.trend      # Xu hướng tổng hợp (Trend + Cycle gộp lại)

# ── Bước 2: HP Filter tách Trend và Cycle từ TC ─────────────────────
# lamb=1600 là giá trị chuẩn — càng lớn thì đường trend càng mượt
print("[2/2] Đang chạy HP Filter (lamb=1600) để tách Trend & Cycle ...")
TC_clean   = TC.dropna()
C_comp_arr, T_comp_arr = hpfilter(TC_clean, lamb=1600)
T_comp = pd.Series(T_comp_arr, index=TC_clean.index)
C_comp = pd.Series(C_comp_arr, index=TC_clean.index)

# Tóm tắt biên độ từng thành phần để đưa vào báo cáo
print("\n  Biên độ các thành phần:")
print(f"    Trend    (T_t): {T_comp.min():.1f} → {T_comp.max():.1f}  (dao động: {T_comp.max()-T_comp.min():.1f})")
print(f"    Cycle    (C_t): {C_comp.min():.1f} → {C_comp.max():.1f}  (dao động: {C_comp.max()-C_comp.min():.1f})")
print(f"    Seasonal (S_t): {S_comp.min():.1f} → {S_comp.max():.1f}  (biên độ tuần: {S_comp.max()-S_comp.min():.1f})")
print(f"    Residual (R_t): {R_comp.min():.1f} → {R_comp.max():.1f}")

# =====================================================================
# FIGURE — BIỂU ĐỒ 4 THÀNH PHẦN PHÂN TÁCH
# =====================================================================
fig, axes = plt.subplots(5, 1, figsize=(15, 14), sharex=True)
fig.suptitle(
    f"CHƯƠNG 2: PHÂN TÁCH CHUỖI THỜI GIAN — STL + HP FILTER\n"
    f"SKU: {sku_rep}  |  $Y_t = T_t + C_t + S_t + R_t$  |  Chu kỳ mùa vụ: 7 ngày (tuần)",
    fontsize=14, fontweight="bold", y=1.01
)

# --- Thành phần gốc Y_t ---
axes[0].plot(df_sku.index, df_sku.values,
             color=COLOR_ACTUAL, linewidth=1, alpha=0.8)
axes[0].set_ylabel("$Y_t$\n(Thực tế)", fontsize=10, fontweight="bold")
axes[0].set_title("DỮ LIỆU GỐC — Doanh số quan sát thực tế",
                  fontsize=11, fontweight="bold")

# --- Xu hướng dài hạn T_t ---
axes[1].plot(T_comp.index, T_comp.values,
             color=COLOR_TREND, linewidth=2.5)
axes[1].set_ylabel("$T_t$\n(Xu hướng)", fontsize=10, fontweight="bold")
axes[1].set_title("XU HƯỚNG DÀI HẠN ($T_t$) — Trích xuất bằng HP Filter",
                  fontsize=11, fontweight="bold")

# --- Chu kỳ trung hạn C_t ---
axes[2].plot(C_comp.index, C_comp.values,
             color=COLOR_CYCLE, linewidth=2)
axes[2].axhline(0, color="gray", linestyle="-", alpha=0.5, linewidth=1)
axes[2].set_ylabel("$C_t$\n(Chu kỳ)", fontsize=10, fontweight="bold")
axes[2].set_title("THÀNH PHẦN CHU KỲ ($C_t$) — Dao động trung hạn quanh xu hướng",
                  fontsize=11, fontweight="bold")

# --- Mùa vụ tuần S_t ---
axes[3].plot(S_comp.index, S_comp.values,
             color=COLOR_SEASONAL, linewidth=1.2, alpha=0.85)
axes[3].axhline(0, color="gray", linestyle="-", alpha=0.5, linewidth=1)
axes[3].set_ylabel("$S_t$\n(Mùa vụ)", fontsize=10, fontweight="bold")
axes[3].set_title("MÙA VỤ TUẦN ($S_t$) — Cuối tuần cao hơn ngày thường",
                  fontsize=11, fontweight="bold")

# --- Nhiễu ngẫu nhiên R_t ---
axes[4].scatter(R_comp.index, R_comp.values,
                color=COLOR_RESID, s=8, alpha=0.6)
axes[4].axhline(0, color="gray", linestyle="-", alpha=0.5, linewidth=1)
axes[4].set_ylabel("$R_t$\n(Nhiễu)", fontsize=10, fontweight="bold")
axes[4].set_title("NHIỄU NGẪU NHIÊN ($R_t$) — Phần không giải thích được",
                  fontsize=11, fontweight="bold")

# Định dạng trục X chung
axes[4].xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
axes[4].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)

for ax in axes:
    ax.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "ch2_stl_decomposition.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("\n[✓] Đã lưu: ch2_stl_decomposition.png")

print("\n" + "=" * 70)
print("CHƯƠNG 2 HOÀN TẤT — 1 biểu đồ đã xuất.")
print("=" * 70)


#%%
# =====================================================================
# 4. CHƯƠNG 3 — MÔ HÌNH DỰ BÁO: NAIVE BASELINE + GLOBAL GBR
#               KIỂM ĐỊNH WALK-FORWARD CROSS-VALIDATION (n=5 FOLDS)
# =====================================================================
# Mục tiêu: xây 2 mô hình dự báo điểm và so sánh độ chính xác:
#   - Naive Baseline : dự báo = doanh số cùng ngày tuần trước (Lag_7)
#                     → mốc so sánh tối thiểu, không cần huấn luyện
#   - Global GBR     : Gradient Boosting train trên tất cả 30 SKU,
#                     features gồm lag, rolling mean, calendar, price
# Phương pháp kiểm định: Walk-Forward CV — mỗi fold mở rộng tập train,
# không để tương lai "rò rỉ" vào quá khứ (no data leakage).
# =====================================================================
print("\n" + "=" * 70)
print("CHƯƠNG 3 — NAIVE BASELINE + GLOBAL GBR + WALK-FORWARD CV")
print("=" * 70)

df = pd.read_csv(OUTPUT_FILE, parse_dates=["Date"])

# ── Feature Engineering ──────────────────────────────────────────────
print("\n[1/4] Đang tạo features ...")

# Sắp xếp đúng thứ tự trước khi tính lag — bắt buộc để tránh data leakage
df = df.sort_values(["SKU", "Date"]).reset_index(drop=True)

# Lag features: tính trong từng SKU riêng để không bị nhảy giữa các SKU
df["Lag_1"]        = df.groupby("SKU")["Sales"].shift(1)
df["Lag_7"]        = df.groupby("SKU")["Sales"].shift(7)
df["Baseline_30d"] = df.groupby("SKU")["Sales"].transform(
    lambda x: x.shift(1).rolling(window=30, min_periods=7).mean()
)

# Encode SKU thành số để GBR có thể học sự khác biệt giữa các mặt hàng
le = LabelEncoder()
df["SKU_encoded"] = le.fit_transform(df["SKU"])

# Bỏ các dòng đầu chưa đủ lag (NaN) sau khi tạo features
df = df.dropna(subset=["Lag_1", "Lag_7", "Baseline_30d"]).reset_index(drop=True)

FEATURES = ["Lag_1", "Lag_7", "Baseline_30d",
            "IsWeekend", "IsEvent", "Price", "SNAP",
            "DayOfWeek", "Month", "SKU_encoded"]

print(f"      Features sử dụng : {FEATURES}")
print(f"      Số dòng sau khi tạo features: {len(df):,}")

# ── Walk-Forward Cross-Validation ────────────────────────────────────
print("\n[2/4] Đang chạy Walk-Forward CV (5 folds, mỗi fold test 60 ngày) ...")

# Lấy danh sách ngày duy nhất để chia folds theo thời gian
all_dates  = sorted(df["Date"].unique())
N_DATES    = len(all_dates)
N_FOLDS    = 5
TEST_DAYS  = 60
MIN_TRAIN  = N_DATES - N_FOLDS * TEST_DAYS   # Tập train tối thiểu ở fold đầu

results_cv = []

print(f"\n  {'Fold':<6} {'Train (ngày)':<16} {'Test (ngày)':<40} {'MAE Naive':>10} {'MAE GBR':>10} {'RMSE Naive':>12} {'RMSE GBR':>10}")
print("  " + "-" * 110)

for fold in range(N_FOLDS):
    # Mỗi fold: train mở rộng thêm, test tiến về phía sau
    train_end_idx = MIN_TRAIN + fold * TEST_DAYS
    test_end_idx  = train_end_idx + TEST_DAYS

    train_dates = all_dates[:train_end_idx]
    test_dates  = all_dates[train_end_idx:test_end_idx]

    df_train = df[df["Date"].isin(train_dates)]
    df_test  = df[df["Date"].isin(test_dates)]

    X_train = df_train[FEATURES]
    y_train = df_train["Sales"]
    X_test  = df_test[FEATURES]
    y_test  = df_test["Sales"]

    # Huấn luyện Global GBR
    gbr = GradientBoostingRegressor(
        n_estimators=200, max_depth=4,
        learning_rate=0.05, subsample=0.8,
        random_state=42
    )
    gbr.fit(X_train, y_train)
    y_pred_gbr = np.maximum(gbr.predict(X_test), 0)   # Doanh số không âm

    # Naive Baseline: dự báo = Lag_7 (doanh số cùng ngày tuần trước)
    y_pred_naive = np.maximum(df_test["Lag_7"].values, 0)

    mae_naive  = mean_absolute_error(y_test, y_pred_naive)
    mae_gbr    = mean_absolute_error(y_test, y_pred_gbr)
    rmse_naive = root_mean_squared_error(y_test, y_pred_naive)
    rmse_gbr   = root_mean_squared_error(y_test, y_pred_gbr)

    results_cv.append({
        "fold": fold + 1,
        "train_days": len(train_dates),
        "test_start": test_dates[0],
        "test_end"  : test_dates[-1],
        "mae_naive" : mae_naive,
        "mae_gbr"   : mae_gbr,
        "rmse_naive": rmse_naive,
        "rmse_gbr"  : rmse_gbr,
        "df_test"   : df_test,
        "y_pred_gbr"  : y_pred_gbr,
        "y_pred_naive": y_pred_naive,
    })

    test_label = f"{pd.Timestamp(test_dates[0]).strftime('%Y-%m-%d')} → {pd.Timestamp(test_dates[-1]).strftime('%Y-%m-%d')}"
    print(f"  Fold {fold+1:<2} {len(train_dates):<16} {test_label:<40} {mae_naive:>10.2f} {mae_gbr:>10.2f} {rmse_naive:>12.2f} {rmse_gbr:>10.2f}")

# Tóm tắt trung bình
mean_mae_naive  = np.mean([r["mae_naive"]  for r in results_cv])
mean_mae_gbr    = np.mean([r["mae_gbr"]    for r in results_cv])
mean_rmse_naive = np.mean([r["rmse_naive"] for r in results_cv])
mean_rmse_gbr   = np.mean([r["rmse_gbr"]   for r in results_cv])
improve_mae     = (mean_mae_naive - mean_mae_gbr) / mean_mae_naive * 100

print("  " + "-" * 110)
print(f"  {'TRUNG BÌNH':<22} {'':<40} {mean_mae_naive:>10.2f} {mean_mae_gbr:>10.2f} {mean_rmse_naive:>12.2f} {mean_rmse_gbr:>10.2f}")
print(f"\n  GBR cải thiện so với Naive Baseline: {improve_mae:.1f}% (theo MAE)")

# ── Train lại trên toàn bộ data — model production ───────────────────
print("\n[3/4] Huấn luyện lại GBR trên toàn bộ dữ liệu (production model) ...")
gbr_final = GradientBoostingRegressor(
    n_estimators=200, max_depth=4,
    learning_rate=0.05, subsample=0.8,
    random_state=42
)
gbr_final.fit(df[FEATURES], df["Sales"])
print("      Hoàn tất.")

# ── Trực quan hóa ────────────────────────────────────────────────────
print("\n[4/4] Đang vẽ biểu đồ ...")

# =====================================================================
# FIGURE 1 — WALK-FORWARD CV: ACTUAL vs GBR vs NAIVE (1 SKU đại diện)
# =====================================================================
sku_rep  = "FOODS_3_090"
fig1, ax = plt.subplots(figsize=(16, 6))

# Vẽ toàn bộ dữ liệu thực tế làm nền
df_sku_all = df[df["SKU"] == sku_rep].copy()
ax.plot(df_sku_all["Date"], df_sku_all["Sales"],
        color="lightgray", linewidth=1, label="Thực tế ($Y_t$)", zorder=1)

# Vẽ từng fold: GBR và Naive chỉ trong vùng test
colors_fold = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]
for i, r in enumerate(results_cv):
    df_t = r["df_test"][r["df_test"]["SKU"] == sku_rep].copy()
    if df_t.empty:
        continue
    dates_t = df_t["Date"].values

    # Lấy prediction tương ứng với SKU đại diện trong fold này
    mask    = r["df_test"]["SKU"] == sku_rep
    pred_g  = np.maximum(r["y_pred_gbr"][mask.values],   0)
    pred_n  = np.maximum(r["y_pred_naive"][mask.values], 0)

    ax.plot(dates_t, pred_g, color=colors_fold[i], linewidth=2,
            linestyle="--", label=f"GBR Fold {i+1}" if i == 0 else f"_Fold {i+1}")
    ax.plot(dates_t, pred_n, color=colors_fold[i], linewidth=1.5,
            linestyle=":", alpha=0.7, label=f"Naive Fold {i+1}" if i == 0 else "_")

    # Đường phân ranh giữa train và test của từng fold
    ax.axvline(x=dates_t[0], color=colors_fold[i],
               linestyle="-.", linewidth=1, alpha=0.5)

# Legend thủ công cho rõ ràng
legend_items = [
    Line2D([0], [0], color="lightgray",  linewidth=2,   label="Thực tế ($Y_t$)"),
    Line2D([0], [0], color="steelblue",  linewidth=2,   linestyle="--", label="GBR Forecast"),
    Line2D([0], [0], color="steelblue",  linewidth=1.5, linestyle=":",  label="Naive Baseline (Lag_7)"),
]
ax.legend(handles=legend_items, loc="upper right", frameon=True, framealpha=0.9, fontsize=10)

ax.set_title(
    f"CHƯƠNG 3: WALK-FORWARD CV — ACTUAL vs GBR vs NAIVE BASELINE\n"
    f"SKU: {sku_rep}  |  5 Folds × 60 ngày  |  "
    f"MAE Naive={mean_mae_naive:.1f}  MAE GBR={mean_mae_gbr:.1f}  (cải thiện {improve_mae:.1f}%)",
    fontsize=13, fontweight="bold"
)
ax.set_ylabel("Doanh số (Sản phẩm/ngày)", fontsize=11, fontweight="bold")
ax.set_xlabel("Thời gian", fontsize=11, fontweight="bold")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)
ax.grid(True, linestyle=":", alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "ch3_walkforward_cv.png"), dpi=150, bbox_inches="tight")
plt.show()
print("[✓] Đã lưu: ch3_walkforward_cv.png")

# =====================================================================
# FIGURE 2 — FEATURE IMPORTANCE CỦA GBR
# =====================================================================
importances = pd.Series(gbr_final.feature_importances_, index=FEATURES)
importances = importances.sort_values(ascending=True)

fig2, ax2 = plt.subplots(figsize=(10, 6))
bars = ax2.barh(importances.index, importances.values,
                color=COLOR_GBR, alpha=0.85, edgecolor="white")

# Ghi số % lên từng thanh
for bar, val in zip(bars, importances.values):
    ax2.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
             f"{val*100:.1f}%", va="center", fontsize=9)

ax2.set_title(
    "CHƯƠNG 3: MỨC ĐỘ ẢNH HƯỞNG CỦA TỪNG FEATURE (FEATURE IMPORTANCE)\n"
    "Global GBR — Train trên toàn bộ 30 SKU × 1,238 ngày",
    fontsize=13, fontweight="bold"
)
ax2.set_xlabel("Importance (tổng = 1.0)", fontsize=11, fontweight="bold")
ax2.set_ylabel("Feature", fontsize=11, fontweight="bold")
ax2.grid(True, linestyle=":", alpha=0.6, axis="x")

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "ch3_feature_importance.png"), dpi=150, bbox_inches="tight")
plt.show()
print("[✓] Đã lưu: ch3_feature_importance.png")

print("\n" + "=" * 70)
print("CHƯƠNG 3 HOÀN TẤT — 2 biểu đồ đã xuất.")
print("=" * 70)


#%%
# =====================================================================
# 5. CHƯƠNG 4 — DỰ BÁO XÁC SUẤT (QUANTILE GBR) & CHIẾN LƯỢC TỒN KHO
# =====================================================================
# Mục tiêu: thay vì dự báo 1 con số (point forecast ở Ch3), chương này
# dự báo 3 kịch bản đồng thời:
#   P10 (bi quan)  — 10% khả năng doanh số thấp hơn con số này
#   P50 (trung lập) — dự báo kỳ vọng
#   P90 (lạc quan) — 10% khả năng doanh số cao hơn con số này
#
# Từ 3 kịch bản đó, tính toán chiến lược tồn kho tối ưu:
#   Safety Stock (SS)    : lượng dự phòng để đảm bảo service level 95%
#   Reorder Point (ROP)  : mức tồn kho cần đặt hàng lại
#   Newsvendor q*        : lượng nhập tối ưu cân bằng chi phí thiếu/thừa
# =====================================================================

print("\n" + "=" * 70)
print("CHƯƠNG 4 — DỰ BÁO XÁC SUẤT & CHIẾN LƯỢC TỒN KHO")
print("=" * 70)

df = pd.read_csv(OUTPUT_FILE, parse_dates=["Date"])
df = df.sort_values(["SKU", "Date"]).reset_index(drop=True)

# Tái tạo features (giống Ch3)
df["Lag_1"]        = df.groupby("SKU")["Sales"].shift(1)
df["Lag_7"]        = df.groupby("SKU")["Sales"].shift(7)
df["Baseline_30d"] = df.groupby("SKU")["Sales"].transform(
    lambda x: x.shift(1).rolling(window=30, min_periods=7).mean()
)
le2 = LabelEncoder()
df["SKU_encoded"] = le2.fit_transform(df["SKU"])
df = df.dropna(subset=["Lag_1", "Lag_7", "Baseline_30d"]).reset_index(drop=True)

FEATURES = ["Lag_1", "Lag_7", "Baseline_30d",
            "IsWeekend", "IsEvent", "Price", "SNAP",
            "DayOfWeek", "Month", "SKU_encoded"]

# Chia train/test: toàn bộ data trừ 90 ngày cuối để backtesting
all_dates  = sorted(df["Date"].unique())
cutoff     = all_dates[-90]
df_train   = df[df["Date"] <  cutoff]
df_test    = df[df["Date"] >= cutoff]

X_train, y_train = df_train[FEATURES], df_train["Sales"]
X_test,  y_test  = df_test[FEATURES],  df_test["Sales"]

print(f"\n  Train: {df_train['Date'].min().date()} → {df_train['Date'].max().date()} ({df_train['Date'].nunique()} ngày)")
print(f"  Test : {df_test['Date'].min().date()} → {df_test['Date'].max().date()} ({df_test['Date'].nunique()} ngày)")

# ── Huấn luyện 3 Quantile GBR (P10, P50, P90) ───────────────────────
print("\n[1/4] Đang huấn luyện 3 Quantile GBR (P10 / P50 / P90) ...")

quantile_models = {}
for q, label in [(0.10, "P10"), (0.50, "P50"), (0.90, "P90")]:
    m = GradientBoostingRegressor(
        loss="quantile", alpha=q,
        n_estimators=200, max_depth=4,
        learning_rate=0.05, subsample=0.8,
        random_state=42
    )
    m.fit(X_train, y_train)
    quantile_models[label] = m
    print(f"      [{label}] Hoàn tất (alpha={q})")

# Dự báo trên tập test
df_test = df_test.copy()
for label, m in quantile_models.items():
    df_test[label] = np.maximum(m.predict(X_test), 0)

# ── Pinball Loss — đánh giá chất lượng dự báo xác suất ──────────────
print("\n[2/4] Tính Pinball Loss (backtesting 90 ngày) ...")

def pinball(q, y_true, y_pred):
    err = y_true - y_pred
    return np.mean(np.maximum(q * err, (q - 1) * err))

pb_p10 = pinball(0.10, y_test, df_test["P10"])
pb_p50 = pinball(0.50, y_test, df_test["P50"])
pb_p90 = pinball(0.90, y_test, df_test["P90"])

print(f"\n  {'Quantile':<10} {'Pinball Loss':>14}")
print(f"  {'-'*26}")
print(f"  {'P10 (10%)':<10} {pb_p10:>14.3f}")
print(f"  {'P50 (50%)':<10} {pb_p50:>14.3f}")
print(f"  {'P90 (90%)':<10} {pb_p90:>14.3f}")
print(f"  {'Trung bình':<10} {np.mean([pb_p10, pb_p50, pb_p90]):>14.3f}")

# ── Tính Safety Stock, ROP, Newsvendor q* cho từng SKU ───────────────
print("\n[3/4] Tính chiến lược tồn kho cho 30 SKU ...")

# Giả định vận hành
LEAD_TIME    = 7       # ngày (thời gian từ lúc đặt hàng đến khi nhận hàng)
SERVICE_LEVEL = 0.95   # mức dịch vụ mục tiêu → z = 1.645
Z_SCORE      = 1.645
CU_CO_RATIO  = 3.0     # chi phí hết hàng gấp 3 lần chi phí tồn kho thừa
CRITICAL_RATIO = CU_CO_RATIO / (CU_CO_RATIO + 1)   # = 0.75 → dùng P75

inventory_table = []
for sku in sorted(df["SKU"].unique()):
    df_sku = df[df["SKU"] == sku]["Sales"]

    avg_daily = df_sku.mean()
    std_daily = df_sku.std()

    # Safety Stock: lượng dự phòng để đảm bảo service level 95%
    ss = Z_SCORE * std_daily * np.sqrt(LEAD_TIME)

    # Reorder Point: mức tồn kho cần bắt đầu đặt hàng lại
    rop = avg_daily * LEAD_TIME + ss

    # Newsvendor q*: lượng nhập tối ưu (dùng quantile critical ratio)
    q_star = df_sku.quantile(CRITICAL_RATIO)

    inventory_table.append({
        "SKU"           : sku.replace("FOODS_", ""),
        "TB Sales/ngày" : round(avg_daily, 1),
        "Std Sales"     : round(std_daily, 1),
        "Safety Stock"  : round(ss, 0),
        "Reorder Point" : round(rop, 0),
        "Newsvendor q*" : round(q_star, 0),
    })

df_inv = pd.DataFrame(inventory_table).sort_values("TB Sales/ngày", ascending=False)

print("\n  BẢNG CHIẾN LƯỢC TỒN KHO (Top 15 SKU theo doanh số):")
print(f"  Giả định: Lead Time={LEAD_TIME} ngày | Service Level={SERVICE_LEVEL*100:.0f}% | Cu/Co={CU_CO_RATIO}")
print()
print(df_inv.head(15).to_string(index=False))

# ── Trực quan hóa ────────────────────────────────────────────────────
print("\n[4/4] Đang vẽ biểu đồ ...")

# =====================================================================
# FIGURE 1 — DẢI BĂNG RỦI RO P10/P50/P90 (1 SKU đại diện)
# =====================================================================
sku_rep   = "FOODS_3_090"
hist_days = 60   # Lấy thêm 60 ngày lịch sử trước vùng test để biểu đồ có đà

df_sku_hist = df[(df["SKU"] == sku_rep) & (df["Date"] < cutoff)].tail(hist_days)
df_sku_test = df_test[df_test["SKU"] == sku_rep].copy()
last_actual = df_sku_hist["Date"].max()

fig1, ax1 = plt.subplots(figsize=(15, 7))

# Vùng lịch sử
ax1.plot(df_sku_hist["Date"], df_sku_hist["Sales"],
         color=COLOR_ACTUAL, linewidth=1.5, marker="o", markersize=3,
         label="Thực tế bán hàng ($Y_t$)")

# Vùng backtesting: actual + dải băng P10–P90
ax1.plot(df_sku_test["Date"], df_sku_test["Sales"],
         color=COLOR_ACTUAL, linewidth=1.5, marker="o", markersize=3)
ax1.plot(df_sku_test["Date"], df_sku_test["P50"],
         color=COLOR_FORECAST, linewidth=2.5, linestyle="--",
         label="Kỳ vọng AI (P50)")
ax1.fill_between(df_sku_test["Date"],
                 df_sku_test["P10"], df_sku_test["P90"],
                 color=COLOR_BAND, alpha=0.35,
                 label="Vùng rủi ro 80% (P10 – P90)")
ax1.plot(df_sku_test["Date"], df_sku_test["P90"],
         color="#1f77b4", linewidth=1, linestyle=":", alpha=0.8, label="P90 (kịch bản lạc quan)")
ax1.plot(df_sku_test["Date"], df_sku_test["P10"],
         color="#d62728", linewidth=1, linestyle=":", alpha=0.8, label="P10 (kịch bản bi quan)")

# Ranh giới train / test
ax1.axvline(x=last_actual, color="#2c3e50", linestyle="-.", linewidth=2)
ax1.text(last_actual - pd.Timedelta(days=3),
         ax1.get_ylim()[1] * 0.93,
         "Vùng Kiểm Định\n(Backtesting)",
         color="#2c3e50", ha="right", fontsize=10, fontweight="bold",
         bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
ax1.text(last_actual + pd.Timedelta(days=3),
         ax1.get_ylim()[1] * 0.93,
         "Vùng Dự Báo\n(Out-of-sample)",
         color="#2c3e50", ha="left", fontsize=10, fontweight="bold",
         bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

ax1.set_title(
    f"CHƯƠNG 4: DẢI BĂNG XÁC SUẤT — QUANTILE GBR (P10 / P50 / P90)\n"
    f"SKU: {sku_rep}  |  Pinball P50={pb_p50:.3f}  |  "
    f"Safety Stock={df_inv[df_inv['SKU']=='3_090']['Safety Stock'].values[0]:.0f} đv",
    fontsize=13, fontweight="bold"
)
ax1.set_ylabel("Doanh số (Sản phẩm/ngày)", fontsize=11, fontweight="bold")
ax1.set_xlabel("Thời gian (Ngày giao dịch)", fontsize=11, fontweight="bold")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=45)
ax1.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=10)
ax1.grid(True, linestyle=":", alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "ch4_risk_band.png"), dpi=150, bbox_inches="tight")
plt.show()
print("[✓] Đã lưu: ch4_risk_band.png")

# =====================================================================
# FIGURE 2 — BẢNG TỒN KHO: SAFETY STOCK & REORDER POINT (TOP 20 SKU)
# =====================================================================
df_plot = df_inv.head(20).set_index("SKU")

fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(16, 7))
fig2.suptitle(
    "CHƯƠNG 4: CHIẾN LƯỢC TỒN KHO TỐI ƯU — TOP 20 SKU\n"
    f"Lead Time={LEAD_TIME} ngày  |  Service Level={SERVICE_LEVEL*100:.0f}%  |  Cu/Co Ratio={CU_CO_RATIO}",
    fontsize=13, fontweight="bold", y=1.01
)

# --- Safety Stock theo SKU ---
ax2.barh(df_plot.index, df_plot["Safety Stock"],
         color=COLOR_FORECAST, alpha=0.85, edgecolor="white")
ax2.barh(df_plot.index, df_plot["TB Sales/ngày"] * LEAD_TIME,
         left=df_plot["Safety Stock"],
         color="lightsteelblue", alpha=0.6, edgecolor="white",
         label="Demand trong Lead Time")

ax2.set_title("SAFETY STOCK & REORDER POINT\n(SS = z × σ × √LT  |  ROP = μ×LT + SS)",
              fontsize=11, fontweight="bold")
ax2.set_xlabel("Số lượng sản phẩm", fontsize=10, fontweight="bold")
ax2.set_ylabel("Mã SKU", fontsize=10, fontweight="bold")

legend_inv = [
    Patch(color=COLOR_FORECAST, alpha=0.85, label="Safety Stock (SS)"),
    Patch(color="lightsteelblue", alpha=0.6, label=f"Demand × Lead Time ({LEAD_TIME} ngày)"),
]
ax2.legend(handles=legend_inv, loc="lower right", fontsize=9, frameon=True)
ax2.grid(True, linestyle=":", alpha=0.6, axis="x")

# --- Newsvendor q* theo SKU ---
ax3.barh(df_plot.index, df_plot["Newsvendor q*"],
         color=COLOR_TREND, alpha=0.8, edgecolor="white")
ax3.barh(df_plot.index, df_plot["TB Sales/ngày"],
         color="lightgray", alpha=0.7, edgecolor="white",
         label="TB Sales/ngày")

ax3.set_title(f"NEWSVENDOR q* (SỐ LƯỢNG NHẬP TỐI ƯU)\n"
              f"q* = Quantile(P{CRITICAL_RATIO*100:.0f})  |  Critical Ratio = Cu/(Cu+Co) = {CRITICAL_RATIO:.2f}",
              fontsize=11, fontweight="bold")
ax3.set_xlabel("Số lượng sản phẩm", fontsize=10, fontweight="bold")
ax3.set_ylabel("Mã SKU", fontsize=10, fontweight="bold")
ax3.legend(loc="lower right", fontsize=9, frameon=True)
ax3.grid(True, linestyle=":", alpha=0.6, axis="x")

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "ch4_inventory_strategy.png"), dpi=150, bbox_inches="tight")
plt.show()
print("[✓] Đã lưu: ch4_inventory_strategy.png")

print("\n" + "=" * 70)
print("CHƯƠNG 4 HOÀN TẤT — 2 biểu đồ đã xuất.")
print("=" * 70)

# ── Lưu model và artifacts cho Streamlit App ─────────────────────────
print("\n[Lưu models] Đang lưu Quantile GBR models, LabelEncoder, và bảng tồn kho ...")

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(quantile_models, os.path.join(MODEL_DIR, "quantile_models.pkl"))
joblib.dump(le2,             os.path.join(MODEL_DIR, "label_encoder.pkl"))
joblib.dump(FEATURES,        os.path.join(MODEL_DIR, "features.pkl"))
df_inv.to_csv(os.path.join(MODEL_DIR, "inventory_table.csv"), index=False, encoding="utf-8-sig")

print(f"  [✓] quantile_models.pkl — 3 mô hình Quantile GBR (P10/P50/P90)")
print(f"  [✓] label_encoder.pkl  — LabelEncoder 30 SKU")
print(f"  [✓] features.pkl       — Danh sách {len(FEATURES)} features")
print(f"  [✓] inventory_table.csv — Bảng Safety Stock / ROP / q* (30 SKU)")

print("\n" + "=" * 70)
print("PIPELINE HOÀN TẤT — TẤT CẢ 4 CHƯƠNG ĐÃ CHẠY THÀNH CÔNG.")
print("=" * 70)
