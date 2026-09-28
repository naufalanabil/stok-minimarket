from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Prediksi Stok Minimarket",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
REQUIRED_FILES = {
    "summary": "rekomendasi_stok.csv",
    "metrics": "evaluasi_model.csv",
    "forecast": "forecast_30_hari.csv",
    "history": "data_bersih.csv",
}

st.markdown(
    """
    <style>
    :root { --ink:#102a2a; --muted:#647a78; --teal:#087f70; --lime:#b7e36b;
      --canvas:#f4f8f6; --line:#dce8e3; }
    .stApp { background: var(--canvas); color: var(--ink); }
    .block-container {padding: 1.55rem 2.2rem 3rem; max-width: 1500px;}
    h1, h2, h3 {letter-spacing:-.035em; color:var(--ink);}
    [data-testid="stSidebar"] {background:linear-gradient(180deg,#e8f3ee 0%,#f4f8f6 100%); border-right:1px solid #dce8e3;}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] small {color:#183a35;}
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {color:#5c746e;}
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="input"] > div {background:#ffffff; border-color:#cbdcd5;}
    [data-testid="stSidebar"] [data-baseweb="select"] input,
    [data-testid="stSidebar"] [data-baseweb="input"] input {color:#183a35;}
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {background:#ffffff; border:1px solid #dce8e3; border-radius:12px; padding:8px;}
    [data-baseweb="popover"] [role="option"] {color:#183a35 !important; background:#ffffff;}
    [data-baseweb="popover"] [role="option"]:hover {background:#e8f3ee !important;}
    [data-testid="stMetric"] {background:linear-gradient(145deg,#ffffff,#f2f8f5);
      border:1px solid var(--line); padding:17px 19px; border-radius:17px;
      box-shadow:0 7px 20px rgba(22,61,56,.055);}
    [data-testid="stMetricLabel"] {color:var(--muted); font-weight:650;}
    [data-testid="stMetricValue"] {color:var(--ink); font-weight:750;}
    [data-testid="stMetricDelta"] {color:var(--teal);}
    .hero-card {padding:27px 30px; border-radius:22px; margin:0 0 20px;
      background:linear-gradient(115deg,#103c36 0%,#087f70 68%,#36a786 100%);
      color:#fff; box-shadow:0 14px 32px rgba(8,93,78,.16); position:relative; overflow:hidden;}
    .hero-card:after {content:""; position:absolute; width:220px; height:220px; right:4%; top:-110px;
      border:1px solid rgba(255,255,255,.18); border-radius:50%; box-shadow:0 0 0 28px rgba(255,255,255,.04),0 0 0 58px rgba(255,255,255,.035);}
    .hero-kicker {font-size:.76rem; text-transform:uppercase; letter-spacing:.15em; color:#c8f2d3; font-weight:750;}
    .hero-title {font-size:2rem; line-height:1.18; font-weight:800; margin:.45rem 0 .45rem; color:#fff;}
    .hero-sub {color:#d9eee7; font-size:.98rem; margin:0; max-width:760px;}
    .stTabs [data-baseweb="tab-list"] {gap:7px; background:#e8f0ec; padding:6px; border-radius:14px;}
    .stTabs [data-baseweb="tab"] {height:44px; border-radius:10px; padding:0 15px; color:#526964; font-weight:650;}
    .stTabs [aria-selected="true"] {background:#fff !important; color:var(--teal) !important;
      box-shadow:0 2px 8px rgba(16,42,42,.09);}
    [data-testid="stDataFrame"] {border:1px solid var(--line); border-radius:13px; overflow:hidden;}
    .stButton > button, .stDownloadButton > button {border-radius:10px; font-weight:650;}
    .stDownloadButton > button {background:#087f70; color:white; border:0;}
    .stDownloadButton > button:hover {background:#06685d; color:white; border:0;}
    [data-testid="stAlert"] {border-radius:12px;}
    hr {border-color:var(--line);}
    @media (max-width: 800px) {.block-container {padding:1rem 1rem 2rem;} .hero-title {font-size:1.55rem;}}
    </style>
    """,
    unsafe_allow_html=True,
)


def _read_csv(upload: Any | None, filename: str) -> pd.DataFrame:
    """Use an uploaded file if supplied; otherwise load the generated output CSV."""
    if upload is not None:
        upload.seek(0)
        return pd.read_csv(upload)
    path = OUTPUT_DIR / filename
    if not path.exists():
        path = BASE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(filename)
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_tables(file_payloads: dict[str, bytes] | None, modified: tuple[int, ...]):
    del modified  # Included in the cache key so changing uploads refreshes the data.
    tables = {}
    for key, filename in REQUIRED_FILES.items():
        payload = file_payloads.get(key) if file_payloads else None
        if payload is not None:
            tables[key] = pd.read_csv(pd.io.common.BytesIO(payload))
        else:
            path = OUTPUT_DIR / filename
            if not path.exists():
                path = BASE_DIR / filename
            if not path.exists():
                raise FileNotFoundError(filename)
            tables[key] = pd.read_csv(path)
    return tables


def normalize_tables(tables: dict[str, pd.DataFrame]):
    summary = tables["summary"].copy()
    metrics = tables["metrics"].copy()
    forecast = tables["forecast"].copy()
    history = tables["history"].copy()

    for name, frame in (("rekomendasi_stok", summary), ("forecast", forecast), ("data_bersih", history)):
        if "product_id" not in frame.columns:
            raise ValueError(f"File {name} harus memiliki kolom product_id.")
        frame["product_id"] = frame["product_id"].astype(str)
    for name, frame in (("forecast", forecast), ("data_bersih", history)):
        if "date" not in frame.columns:
            raise ValueError(f"File {name} harus memiliki kolom date.")
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame.dropna(subset=["date"], inplace=True)
    for col in ("forecast_demand", "rf_demand", "prophet_demand", "stock", "safety_stock",
                "lead_time_demand", "reorder_point", "recommended_order", "avg_daily_demand", "days_of_stock"):
        if col in summary:
            summary[col] = pd.to_numeric(summary[col], errors="coerce")
    for col in ("sales", "stock"):
        if col in history:
            history[col] = pd.to_numeric(history[col], errors="coerce")
    for col in ("rf_prediction", "prophet_prediction", "ensemble_prediction"):
        if col in forecast:
            forecast[col] = pd.to_numeric(forecast[col], errors="coerce")
    for col in ("MAE", "RMSE"):
        if col in metrics:
            metrics[col] = pd.to_numeric(metrics[col], errors="coerce")

    if "ensemble_prediction" not in forecast:
        if {"rf_prediction", "prophet_prediction"}.issubset(forecast.columns):
            forecast["ensemble_prediction"] = forecast[["rf_prediction", "prophet_prediction"]].mean(axis=1)
        else:
            raise ValueError("forecast_30_hari.csv perlu kolom ensemble_prediction atau prediksi kedua model.")
    if "sales" not in history:
        raise ValueError("data_bersih.csv harus memiliki kolom sales.")
    if "stock" not in summary:
        raise ValueError("rekomendasi_stok.csv harus memiliki kolom stock.")

    return summary, metrics, forecast, history


def build_inventory_view(summary: pd.DataFrame, forecast: pd.DataFrame, safety_multiplier: float, lead_time: int):
    view = summary.copy()
    if "avg_daily_demand" not in view:
        horizon = max(1, int(forecast["date"].nunique()))
        view["avg_daily_demand"] = view.get("forecast_demand", 0) / horizon
    if "forecast_demand" not in view:
        view["forecast_demand"] = view["avg_daily_demand"] * max(1, int(forecast["date"].nunique()))
    base_safety = view.get("safety_stock", pd.Series(0, index=view.index)).fillna(0)
    view["safety_stock_simulated"] = np.ceil(base_safety * safety_multiplier).clip(lower=0)
    if "lead_time_demand" not in view or view["lead_time_demand"].isna().all():
        view["lead_time_demand"] = view["avg_daily_demand"] * lead_time
    view["reorder_point_simulated"] = np.ceil(view["lead_time_demand"].fillna(0) + view["safety_stock_simulated"])
    target_stock = view["forecast_demand"].fillna(0) + view["safety_stock_simulated"]
    view["recommended_order_simulated"] = np.ceil((target_stock - view["stock"].fillna(0)).clip(lower=0))
    if "days_of_stock" not in view:
        view["days_of_stock"] = np.where(view["avg_daily_demand"] > 0,
                                          view["stock"] / view["avg_daily_demand"], np.inf)
    view["status"] = np.select(
        [view["stock"].fillna(0) <= 0,
         view["stock"].fillna(0) <= view["reorder_point_simulated"]],
        ["Kritis / Habis", "Perlu Reorder"], default="Stok Aman"
    )
    return view


st.markdown(
    """
    <section class="hero-card">
      <div class="hero-kicker">Smart inventory · Minimarket</div>
      <div class="hero-title">Prediksi dan Rekomendasi Stok</div>
      <p class="hero-sub">Pantau kebutuhan barang, bandingkan hasil Random Forest dan Prophet, lalu siapkan pengadaan berdasarkan proyeksi permintaan.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Pengaturan")
    st.caption("Unggah hasil dari notebook Colab atau jalankan `main.py` terlebih dahulu.")
    with st.expander("Unggah file hasil CSV", expanded=False):
        uploaded = {
            "summary": st.file_uploader("rekomendasi_stok.csv", type="csv", key="summary_csv"),
            "metrics": st.file_uploader("evaluasi_model.csv", type="csv", key="metrics_csv"),
            "forecast": st.file_uploader("forecast_30_hari.csv", type="csv", key="forecast_csv"),
            "history": st.file_uploader("data_bersih.csv", type="csv", key="history_csv"),
        }
    safety_multiplier = st.slider("Pengali safety stock", 0.5, 3.0, 1.0, 0.1,
                                  help="Simulasi perubahan safety stock dari hasil model.")
    lead_time = st.number_input("Lead time pemasok (hari)", min_value=0, max_value=365, value=7, step=1)
    selected_statuses = st.multiselect(
        "Filter status", ["Kritis / Habis", "Perlu Reorder", "Stok Aman"],
        default=["Kritis / Habis", "Perlu Reorder", "Stok Aman"],
    )

payloads = {key: file.getvalue() if file is not None else None for key, file in uploaded.items()}
provided = [key for key, payload in payloads.items() if payload is not None]
if provided and len(provided) != len(REQUIRED_FILES):
    missing_names = [REQUIRED_FILES[key] for key, payload in payloads.items() if payload is None]
    st.sidebar.warning("Unggah keempat CSV hasil agar data tidak tercampur dengan file lama: " + ", ".join(missing_names))
    st.warning("Unggahan belum lengkap. Dashboard memerlukan `rekomendasi_stok.csv`, `evaluasi_model.csv`, `forecast_30_hari.csv`, dan `data_bersih.csv` dari satu kali proses model.")
    st.stop()
cache_key = tuple(hash(payload) if payload is not None else 0 for payload in payloads.values())
try:
    tables = load_tables(payloads, cache_key)
    summary, metrics, forecast, history = normalize_tables(tables)
    inventory = build_inventory_view(summary, forecast, safety_multiplier, int(lead_time))
except FileNotFoundError as exc:
    st.error(f"File `{exc.args[0]}` belum ditemukan.")
    st.info("Jalankan `python main.py` untuk membuat CSV di folder `outputs/`, atau unggah keempat CSV melalui sidebar.")
    st.stop()
except (ValueError, pd.errors.ParserError) as exc:
    st.error(f"Data tidak dapat dibaca: {exc}")
    st.stop()

if inventory.empty or history.empty or forecast.empty:
    st.warning("Data belum berisi produk, histori, atau hasil prediksi.")
    st.stop()

if provided:
    st.success(f"Data aktif berasal dari {len(provided)}/4 CSV yang diunggah. Dashboard membaca hasil yang Anda unggah; model tidak dilatih ulang di halaman ini.", icon="✅")
else:
    st.info("Data aktif berasal dari CSV di folder outputs. Untuk memperbarui hasil, jalankan ulang notebook Colab atau `main.py`, lalu unggah keempat CSV hasilnya.", icon="ℹ️")

products = sorted(set(inventory["product_id"]) & set(history["product_id"]) & set(forecast["product_id"]))
if not products:
    st.error("ID produk pada file rekomendasi, histori, dan forecast tidak cocok.")
    st.stop()

visible_inventory = inventory[inventory["status"].isin(selected_statuses)]
if not selected_statuses:
    visible_inventory = inventory.iloc[0:0]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Produk terdaftar", f"{inventory['product_id'].nunique():,}")
k2.metric("Prediksi permintaan", f"{inventory['forecast_demand'].fillna(0).sum():,.0f} unit",
          help=f"Total horizon forecast ({forecast['date'].nunique()} hari)")
k3.metric("Perlu tindakan", f"{inventory['status'].isin(['Kritis / Habis','Perlu Reorder']).sum():,} SKU")
k4.metric("Jumlah order simulasi", f"{inventory['recommended_order_simulated'].sum():,.0f} unit")

st.caption(f"Periode histori: {history['date'].min():%d-%m-%Y} — {history['date'].max():%d-%m-%Y} | "
           f"Periode prediksi: {forecast['date'].min():%d-%m-%Y} — {forecast['date'].max():%d-%m-%Y}")

summary_tab, forecast_tab, performance_tab, analysis_tab, data_tab = st.tabs(
    ["📋 Rekomendasi stok", "📈 Forecast produk", "📊 Evaluasi model", "🔎 Analisis data", "🧾 Data"]
)

with summary_tab:
    st.subheader("Daftar tindakan pengadaan")
    display_cols = [c for c in ["product_id", "status", "stock", "recommended_order_simulated",
                                  "reorder_point_simulated", "forecast_demand", "avg_daily_demand",
                                  "safety_stock_simulated", "days_of_stock"] if c in inventory.columns]
    st.dataframe(
        visible_inventory[display_cols].sort_values("recommended_order_simulated", ascending=False),
        use_container_width=True, hide_index=True,
        column_config={
            "product_id": st.column_config.TextColumn("ID Produk"),
            "status": st.column_config.TextColumn("Status"),
            "stock": st.column_config.NumberColumn("Stok saat ini", format="%.0f unit"),
            "recommended_order_simulated": st.column_config.NumberColumn("Saran order", format="%.0f unit"),
            "reorder_point_simulated": st.column_config.NumberColumn("Reorder point", format="%.0f unit"),
            "forecast_demand": st.column_config.NumberColumn("Demand horizon", format="%.1f unit"),
            "avg_daily_demand": st.column_config.NumberColumn("Rata-rata per hari", format="%.2f unit"),
            "safety_stock_simulated": st.column_config.NumberColumn("Safety stock", format="%.0f unit"),
            "days_of_stock": st.column_config.NumberColumn("Ketahanan stok", format="%.1f hari"),
        },
    )
    st.download_button("Unduh rekomendasi terfilter (CSV)",
                       visible_inventory.to_csv(index=False).encode("utf-8-sig"),
                       "rekomendasi_stok_minimarket.csv", "text/csv")
    st.info("Reorder point = kebutuhan selama lead time + safety stock. Saran order memperkirakan kebutuhan selama seluruh horizon forecast + safety stock − stok saat ini.")

with forecast_tab:
    st.subheader("Riwayat dan prediksi penjualan")
    product = st.selectbox("Pilih produk", products)
    hist = history[history["product_id"] == product].sort_values("date")
    pred = forecast[forecast["product_id"] == product].sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["sales"], mode="lines", name="Penjualan aktual"))
    if "rf_prediction" in pred:
        fig.add_trace(go.Scatter(x=pred["date"], y=pred["rf_prediction"], mode="lines", name="Random Forest", line={"dash":"dot"}))
    if "prophet_prediction" in pred:
        fig.add_trace(go.Scatter(x=pred["date"], y=pred["prophet_prediction"], mode="lines", name="Prophet", line={"dash":"dot"}))
    fig.add_trace(go.Scatter(x=pred["date"], y=pred["ensemble_prediction"], mode="lines+markers", name="Ensemble", line={"width":3}))
    if not hist.empty:
        fig.add_vline(x=hist["date"].max(), line_dash="dash", line_color="gray", annotation_text="Mulai forecast")
    fig.update_layout(template="plotly_white", height=480, xaxis_title="Tanggal", yaxis_title="Unit terjual", hovermode="x unified",
                      colorway=["#087f70", "#ec9f45", "#7789d5", "#d96060"],
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff",
                      margin=dict(l=10,r=15,t=40,b=10), legend=dict(orientation="h",y=1.12))
    fig.update_xaxes(showgrid=False, linecolor="#dce8e3")
    fig.update_yaxes(gridcolor="#edf2ef", zerolinecolor="#dce8e3")
    st.plotly_chart(fig, use_container_width=True)
    st.download_button("Unduh forecast produk (CSV)", pred.to_csv(index=False).encode("utf-8-sig"),
                       f"forecast_{product}.csv", "text/csv")

with performance_tab:
    st.subheader("Evaluasi model")
    if not {"model", "MAE"}.issubset(metrics.columns):
        st.warning("Kolom model dan MAE tidak ditemukan pada evaluasi_model.csv.")
    else:
        metric_cols = [c for c in ["model", "product_id", "MAE", "RMSE"] if c in metrics.columns]
        st.dataframe(metrics[metric_cols], use_container_width=True, hide_index=True)
        plot_metrics = metrics.dropna(subset=["MAE"]).copy()
        if not plot_metrics.empty:
            plot_metrics["seri"] = plot_metrics["model"].astype(str)
            if "product_id" in plot_metrics:
                plot_metrics["seri"] += plot_metrics["product_id"].fillna("").astype(str).radd(" — ")
            figm = px.bar(plot_metrics, x="seri", y="MAE", color="model", title="MAE per model / produk",
                          labels={"seri":"Model / Produk", "MAE":"MAE (unit)"}, text_auto=".2f")
            figm.update_layout(template="plotly_white", height=420, showlegend=True,
                               colorway=["#087f70", "#ec9f45", "#7789d5", "#d96060"],
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff",
                               margin=dict(l=10,r=15,t=50,b=10))
            figm.update_yaxes(gridcolor="#edf2ef")
            st.plotly_chart(figm, use_container_width=True)
        st.caption("MAE dan RMSE dihitung dari periode evaluasi notebook. Bandingkan model pada rentang uji yang sama; metrik lebih kecil menunjukkan error lebih rendah.")

with analysis_tab:
    st.subheader("Eksplorasi hubungan dan pola penjualan")
    analysis_products = sorted(history["product_id"].dropna().astype(str).unique())
    analysis_product = st.selectbox("Pilih produk untuk analisis", analysis_products, key="analysis_product")
    data_product = history[history["product_id"] == analysis_product].copy().sort_values("date")
    data_product["hari"] = data_product["date"].dt.day_name()
    data_product["bulan"] = data_product["date"].dt.month_name()
    data_product["hari_ke"] = data_product["date"].dt.dayofweek
    data_product["lag_1"] = data_product["sales"].shift(1)
    data_product["rata_mingguan"] = data_product["sales"].shift(1).rolling(7, min_periods=3).mean()

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown("#### Scatter penjualan vs stok")
        if "stock" in data_product and data_product["stock"].notna().any():
            scatter = px.scatter(data_product, x="stock", y="sales", color="hari",
                                 hover_data=["date"], opacity=.72,
                                 labels={"stock":"Stok tercatat", "sales":"Unit terjual", "hari":"Hari"},
                                 color_discrete_sequence=["#087f70", "#ec9f45", "#7789d5", "#d96060", "#55a9a1", "#9a72b0", "#b4a23a"],
                                 title=f"Apakah stok berkaitan dengan penjualan? — {analysis_product}")
            scatter.update_layout(template="plotly_white", height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fff")
            st.plotly_chart(scatter, use_container_width=True)
        else:
            st.info("Kolom stock tidak tersedia pada data historis.")
    with chart_right:
        st.markdown("#### Distribusi penjualan harian")
        hist_fig = px.histogram(data_product, x="sales", nbins=25, marginal="box",
                                labels={"sales":"Unit terjual", "count":"Jumlah hari"},
                                color_discrete_sequence=["#087f70"], title="Sebaran dan pencilan penjualan")
        hist_fig.update_layout(template="plotly_white", height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fff")
        st.plotly_chart(hist_fig, use_container_width=True)

    st.markdown("#### Correlation matrix")
    corr_cols = [c for c in ["sales", "stock", "hari_ke", "lag_1", "rata_mingguan"] if c in data_product and data_product[c].notna().sum() > 2]
    corr_data = data_product[corr_cols].dropna().corr(numeric_only=True) if len(corr_cols) >= 2 else pd.DataFrame()
    if not corr_data.empty:
        heat = go.Figure(go.Heatmap(z=corr_data.values, x=corr_data.columns, y=corr_data.index,
                                    zmin=-1, zmax=1, colorscale=[[0,"#d96060"],[.5,"#ffffff"],[1,"#087f70"]],
                                    text=np.round(corr_data.values, 2), texttemplate="%{text:.2f}",
                                    colorbar={"title":"Korelasi"}, hovertemplate="%{y} × %{x}<br>r = %{z:.2f}<extra></extra>"))
        heat.update_layout(template="plotly_white", height=420, paper_bgcolor="rgba(0,0,0,0)",
                           margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(heat, use_container_width=True)
        st.caption("Korelasi menunjukkan hubungan linear (−1 sampai +1), bukan sebab-akibat. Hari dalam minggu dikodekan 0–6 sehingga interpretasikan korelasinya dengan hati-hati.")
        st.dataframe(corr_data.style.format("{:.2f}").background_gradient(cmap="RdYlGn", vmin=-1, vmax=1), use_container_width=True)
    else:
        st.info("Belum cukup kolom atau baris numerik untuk menghitung korelasi.")

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily_pattern = data_product.groupby("hari", as_index=False)["sales"].mean()
    daily_pattern["hari"] = pd.Categorical(daily_pattern["hari"], categories=day_order, ordered=True)
    daily_pattern = daily_pattern.sort_values("hari")
    monthly_pattern = data_product.groupby(data_product["date"].dt.to_period("M"))["sales"].mean().reset_index()
    monthly_pattern["date"] = monthly_pattern["date"].dt.to_timestamp()
    pat_left, pat_right = st.columns(2)
    with pat_left:
        weekly_fig = px.bar(daily_pattern, x="hari", y="sales", title="Rata-rata penjualan menurut hari",
                            labels={"hari":"Hari", "sales":"Rata-rata unit terjual"}, color_discrete_sequence=["#087f70"])
        weekly_fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fff")
        st.plotly_chart(weekly_fig, use_container_width=True)
    with pat_right:
        month_fig = px.line(monthly_pattern, x="date", y="sales", markers=True, title="Tren rata-rata penjualan bulanan",
                            labels={"date":"Bulan", "sales":"Rata-rata unit terjual"}, color_discrete_sequence=["#087f70"])
        month_fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fff")
        st.plotly_chart(month_fig, use_container_width=True)

with data_tab:
    st.subheader("Data yang digunakan")
    left, right = st.columns(2)
    with left:
        st.write("Sampel histori penjualan")
        st.dataframe(history.sort_values("date", ascending=False).head(100), use_container_width=True, hide_index=True)
    with right:
        st.write("Sampel prediksi")
        st.dataframe(forecast.head(100), use_container_width=True, hide_index=True)
    st.download_button("Unduh data histori (CSV)", history.to_csv(index=False).encode("utf-8-sig"),
                       "data_bersih.csv", "text/csv")

st.divider()
st.caption("Dashboard membaca hasil model; pelatihan dilakukan di notebook Colab atau main.py. Periksa data stok, lead time, dan kebijakan safety stock sebelum membuat pesanan.")

