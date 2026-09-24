# Pipeline Dự Báo Xác Suất & Tối Ưu Hóa Chiến Lược Tồn Kho

> **Đồ án nhóm — Môn: Các mô hình dự báo trong kinh doanh**  
> Dataset: M5 Forecasting Accuracy (Walmart) · Store CA_1 · Category FOODS · Top 30 SKU

---

## Mục lục

1. [Nhóm thực hiện](#nhóm-thực-hiện)
2. [Tổng quan](#tổng-quan)
3. [Kết quả chính](#kết-quả-chính)
4. [Cấu trúc Pipeline](#cấu-trúc-pipeline)
5. [Subset Strategy](#subset-strategy)
6. [Cấu trúc Project](#cấu-trúc-project)
7. [Cài đặt & Chạy](#cài-đặt--chạy)
8. [Streamlit App](#streamlit-app)
9. [Stack kỹ thuật](#stack-kỹ-thuật)
10. [Tham chiếu](#tham-chiếu)

---

## Nhóm thực hiện

| MSHV | Họ và tên |
|---|---|
| C25611254 | Lê Nhật Thanh |
| C25611253 | Lê Ngọc Phú |
| C25611251 | Huỳnh Trúc Ngân |
| C25611258 | Phạm Thị Thúy Kiều |
| C25611256 | Trịnh Quang Tân |

**GVHD:** TS. Trần Duy Thanh

---

## Tổng quan

Đồ án xây dựng một **pipeline dự báo hoàn chỉnh 4 chương** cho bài toán quản trị tồn kho chuỗi bán lẻ đa SKU, áp dụng trên dữ liệu thực từ cuộc thi Kaggle M5 Forecasting Accuracy (Walmart).

Thay vì dừng lại ở point forecast truyền thống, pipeline kết hợp **dự báo xác suất (Quantile GBR)** với **mô hình Newsvendor** để ra quyết định tồn kho tối ưu dưới điều kiện bất định.

---

## Kết quả chính

| Chỉ số | Naive Baseline | Global GBR | Cải thiện |
|---|---|---|---|
| MAE (Walk-Forward CV) | 7.95 đv | **6.10 đv** | ↓ 23.4% |
| RMSE | 12.85 đv | **9.31 đv** | ↓ 27.5% |
| Pinball Loss P50 | — | **2.716** | — |

SKU điển hình — FOODS\_3\_090 (bán chạy nhất, TB 65.1 sp/ngày):

- Safety Stock = **235 đv**, Reorder Point = **690 đv**, Newsvendor q\* = **P75**
- CV = 0.83 → biến động cực lớn, không thể dùng fixed reorder point

---

## Cấu trúc Pipeline

```
Ch1 · EDA              →  Phân tích chuỗi thời gian, phân phối, outlier IQR,
                           unit economics (Revenue = Sales × Price)

Ch2 · Decomposition    →  STL (period=7) + HP Filter: tách Trend / Cycle /
                           Seasonal / Residual; xác nhận additive model

Ch3 · GBR Forecasting  →  Global GBR (30 SKU cùng lúc), Walk-Forward CV 5 folds,
                           features: Lag_1, Lag_7, Baseline_30d, IsWeekend,
                           IsEvent, Price, SNAP_CA, DayOfWeek, Month

Ch4 · Probabilistic    →  Quantile GBR P10/P50/P90, Pinball Loss backtesting,
     Inventory          Safety Stock = z·σ·√LT, Newsvendor q*,
                           Reorder Point table cho 30 SKU
```

---

## Subset Strategy

| Tham số | Giá trị | Lý do |
|---|---|---|
| Store | CA_1 | 1 địa điểm → Lead time & hành vi KH nhất quán |
| Category | FOODS | Perishable → inventory-critical nhất |
| Items | Top 30 SKU | Pareto 80/20; đảm bảo mean > 5 đv/ngày cho Safety Stock |
| Period | 2013-01-01 → 2016 | ~3 năm, đủ seasonality và holiday pattern |

---

## Cấu trúc Project

```
m5-walmart-inventory-forecast/
│
├── app.py                          # Streamlit app (3 tab)
├── pipeline_m5_walmart.py          # Pipeline đầy đủ 4 chương (5 cell #%%)
├── requirements.txt                # Thư viện Python
├── README.md
│
├── Datasets/
│   ├── m5_ca1_foods_clean.csv      # Subset đã xử lý — CA_1/FOODS/Top30 (2.6 MB)
│   └── calendar.csv                # Lịch sự kiện Walmart (103 KB)
│
├── models/
│   ├── quantile_models.pkl         # Quantile GBR: P10 / P50 / P90 (1.4 MB)
│   ├── label_encoder.pkl           # SKU label encoder
│   ├── features.pkl                # Danh sách features theo thứ tự
│   └── inventory_table.csv         # Safety Stock & ROP cho 30 SKU
│
├── ch1_pd1_timeseries.png          # EDA — Chuỗi thời gian Top 5 SKU
├── ch1_pd2_dist_corr.png           # EDA — Phân phối & ma trận tương quan
├── ch1_pd3_boxplot.png             # EDA — Boxplot outlier (IQR)
├── ch1_pd4_unit_economics.png      # EDA — Doanh thu theo SKU
├── ch2_stl_decomposition.png       # STL + HP Filter — 4 thành phần T/C/S/R
├── ch3_feature_importance.png      # GBR — Feature importance
├── ch3_walkforward_cv.png          # GBR — Walk-Forward CV 5 folds
├── ch4_risk_band.png               # Quantile — Dải băng P10/P50/P90
└── ch4_inventory_strategy.png      # Inventory — Safety Stock & ROP table
```

> **Lưu ý:** File raw M5 (`sales_train_*.csv`, `sell_prices.csv`) không đưa lên GitHub do kích thước 115–194 MB.  
> Tải về từ: [Kaggle M5 Forecasting Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)

---

## Cài đặt & Chạy

```bash
# 1. Clone repo
git clone https://github.com/lenhatthanh2302/m5-walmart-inventory-forecast.git
cd m5-walmart-inventory-forecast

# 2. Tải raw M5 data từ Kaggle → bỏ vào Datasets/

# 3. Cài thư viện
pip install -r requirements.txt

# 4. Chạy pipeline (sinh clean CSV + model .pkl)
python pipeline_m5_walmart.py

# 5. Chạy Streamlit app
streamlit run app.py
```

---

## Streamlit App

App gồm 3 tab:

| Tab | Nội dung |
|---|---|
| Dự Báo Xác Suất | Dải băng P10/P50/P90 cho horizon 7–30 ngày; export Excel & HTML |
| Chiến Lược Tồn Kho | Safety Stock, Reorder Point, Newsvendor q\*, biểu đồ waterfall ROP |
| Toàn Danh Mục | Bản đồ rủi ro bubble chart 30 SKU + bảng chi tiết |

---

## Stack kỹ thuật

- **Python 3.14** — pandas, numpy, scikit-learn, statsmodels
- **Mô hình:** Gradient Boosting Regressor (sklearn) — point forecast & quantile
- **Phân tích chuỗi:** STL Decomposition + Hodrick-Prescott Filter
- **Tồn kho:** Newsvendor Model, Safety Stock (Normal approximation), Walk-Forward Cross-Validation
- **Visualization:** matplotlib, seaborn (pipeline) · plotly (app)
- **App:** Streamlit

---

## Tham chiếu

- Makridakis, S. et al. (2022). *M5 accuracy competition: Results, findings, and conclusions.* International Journal of Forecasting.
- Hyndman, R. J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.).
- Silver, E. A., Pyke, D. F., & Thomas, D. J. (2017). *Inventory and Production Management in Supply Chains* (4th ed.).
