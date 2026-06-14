import warnings
warnings.filterwarnings("ignore")

import io
import textwrap
import html
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Simulasi Sampah Kota Bandung",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)


def render_clean_html(html: str) -> None:
    """Render HTML di Streamlit tanpa terbaca sebagai markdown code block."""
    cleaned = "\n".join(line.strip() for line in html.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


FILE_NAME = "jumlah_capaian_penanganan_sampah_di_kota_bandung.xlsx"
BASE_HISTORICAL_ROWS = 96
DEFAULT_FORECAST_MAX_MONTHS = 24
UPLOAD_FORECAST_MAX_MONTHS = 48

BULAN_MAP = {
    "JANUARI": 1,
    "FEBRUARI": 2,
    "MARET": 3,
    "APRIL": 4,
    "MEI": 5,
    "JUNI": 6,
    "JULI": 7,
    "AGUSTUS": 8,
    "SEPTEMBER": 9,
    "OKTOBER": 10,
    "NOVEMBER": 11,
    "DESEMBER": 12,
}

BULAN_INDO = {
    1: "Januari",
    2: "Februari",
    3: "Maret",
    4: "April",
    5: "Mei",
    6: "Juni",
    7: "Juli",
    8: "Agustus",
    9: "September",
    10: "Oktober",
    11: "November",
    12: "Desember",
}

MENU_OPTIONS = ["Simulasi Pengelolaan", "Kelola Data Upload", "Ringkasan Data & Model"]

# Konstanta simulasi operasional
# Densitas digunakan untuk mengonversi prediksi ton menjadi volume m³.
# Kapasitas truk compactor mengacu pada ukuran 6 m³ dan 12 m³; default dipakai 12 m³.
DENSITAS_SAMPAH_KG_PER_M3 = 227.297


# ============================================================
# FORMATTER
# ============================================================

def format_periode(date_value):
    return f"{BULAN_INDO[date_value.month]} {date_value.year}"


def format_rupiah(value):
    return "Rp{:,.0f}".format(value).replace(",", ".")


def format_angka(value):
    return "{:,.2f}".format(value).replace(",", "X").replace(".", ",").replace("X", ".")


def format_integer(value):
    return "{:,.0f}".format(value).replace(",", ".")


def evaluate_model(actual, forecast):
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    r2 = r2_score(actual, forecast)
    return mae, rmse, mape, r2


# ============================================================
# LOAD DATA
# ============================================================

def safe_unique_text(df_raw, column_name, default="Tidak tersedia"):
    if column_name not in df_raw.columns:
        return default

    values = df_raw[column_name].dropna().astype(str).str.strip()
    values = values[values != ""]

    if values.empty:
        return default

    return ", ".join(values.unique())


def normalize_uploaded_columns(df_raw):
    df_raw = df_raw.copy()
    column_lookup = {str(col).strip().lower(): col for col in df_raw.columns}

    required_columns = ["tahun", "bulan", "jumlah_sampah"]
    missing_columns = [col for col in required_columns if col not in column_lookup]

    if missing_columns:
        raise ValueError(
            "Kolom wajib belum lengkap. Pastikan file memiliki kolom: tahun, bulan, dan jumlah_sampah."
        )

    rename_map = {column_lookup[col]: col for col in required_columns}
    df_raw = df_raw.rename(columns=rename_map)

    return df_raw


def preprocess_data(df_raw):
    df_raw = normalize_uploaded_columns(df_raw)

    df = df_raw.copy()

    bulan_numeric = pd.to_numeric(df["bulan"], errors="coerce")
    bulan_text = df["bulan"].astype(str).str.strip().str.upper()
    df["bulan_num"] = bulan_text.map(BULAN_MAP)
    df.loc[bulan_numeric.between(1, 12), "bulan_num"] = bulan_numeric[bulan_numeric.between(1, 12)]

    df["tahun"] = pd.to_numeric(df["tahun"], errors="coerce")
    df["jumlah_sampah"] = pd.to_numeric(df["jumlah_sampah"], errors="coerce")

    df = df.dropna(subset=["tahun", "bulan_num", "jumlah_sampah"])

    if df.empty:
        raise ValueError("Data tidak dapat diproses karena nilai tahun, bulan, atau jumlah_sampah tidak valid.")

    df["tahun"] = df["tahun"].astype(int)
    df["bulan_num"] = df["bulan_num"].astype(int)

    df["tanggal"] = pd.to_datetime({
        "year": df["tahun"],
        "month": df["bulan_num"],
        "day": 1
    })

    df = df.sort_values("tanggal")
    df = df.drop_duplicates(subset=["tanggal"], keep="last")
    df = df.set_index("tanggal")

    ts = df["jumlah_sampah"].asfreq("MS")

    if ts.isna().sum() > 0:
        ts = ts.interpolate(method="time").bfill().ffill()

    if len(ts.dropna()) < 24:
        raise ValueError("Minimal dibutuhkan 24 bulan data historis agar model SARIMA dapat dilatih dengan lebih stabil.")

    return df_raw, df, ts


def read_uploaded_dataframe(uploaded_bytes, uploaded_name):
    file_buffer = io.BytesIO(uploaded_bytes)

    if uploaded_name.lower().endswith(".csv"):
        uploaded_df = pd.read_csv(file_buffer)
    else:
        uploaded_df = pd.read_excel(file_buffer)

    return normalize_uploaded_columns(uploaded_df)


def shorten_file_name(file_name, max_chars=28):
    file_name = str(file_name)
    if len(file_name) <= max_chars:
        return file_name

    if "." in file_name:
        stem, ext = file_name.rsplit(".", 1)
        ext = "." + ext
    else:
        stem, ext = file_name, ""

    keep = max_chars - len(ext) - 3
    if keep < 8:
        keep = max_chars - 3
        ext = ""

    left = max(5, keep // 2)
    right = max(4, keep - left)
    return f"{stem[:left]}...{stem[-right:]}{ext}"


def format_file_size(size_bytes):
    size_bytes = float(size_bytes or 0)
    if size_bytes < 1024:
        return f"{size_bytes:.0f} B"
    if size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 ** 2):.1f} MB"


@st.cache_data
def load_data(upload_payloads=None):
    df_base_raw = pd.read_excel(FILE_NAME)
    df_base_raw = normalize_uploaded_columns(df_base_raw)

    upload_payloads = upload_payloads or tuple()
    uploaded_frames = []
    uploaded_names = []

    for uploaded_name, uploaded_bytes in upload_payloads:
        if uploaded_name is None or uploaded_bytes is None:
            continue

        df_uploaded_raw = read_uploaded_dataframe(uploaded_bytes, uploaded_name)
        uploaded_frames.append(df_uploaded_raw)
        uploaded_names.append(uploaded_name)

    if uploaded_frames:
        # Data upload digabung dengan data bawaan dan seluruh upload sebelumnya.
        # Kalau periode upload sama dengan periode bawaan/upload lama, nilai terakhir menjadi versi terbaru
        # karena preprocess_data memakai drop_duplicates(..., keep="last").
        df_raw = pd.concat(
            [df_base_raw] + uploaded_frames,
            ignore_index=True,
            sort=False
        )

        source_data_name = f"Badan Pusat Statistik + {len(uploaded_frames)} file upload"
        source_data_type = "upload"
    else:
        df_raw = df_base_raw
        source_data_name = "Badan Pusat Statistik"
        source_data_type = "default"

    df_raw, df, ts = preprocess_data(df_raw)

    return df_raw, df, ts, source_data_name, source_data_type


SARIMA_CANDIDATES = [
    ((1, 2, 2), (0, 1, 1, 12)),
    ((1, 1, 1), (0, 1, 1, 12)),
    ((0, 1, 1), (0, 1, 1, 12)),
    ((1, 1, 2), (0, 1, 1, 12)),
    ((2, 1, 1), (0, 1, 1, 12)),
    ((2, 1, 2), (0, 1, 1, 12)),
    ((1, 1, 1), (1, 1, 1, 12)),
    ((2, 1, 2), (1, 1, 1, 12)),
]


def sarima_label(order, seasonal_order):
    return f"SARIMA{order}{seasonal_order}"


@st.cache_data(show_spinner=False)
def select_best_sarima(ts):
    results = []
    best_order = None
    best_seasonal_order = None
    best_aic = np.inf

    for order, seasonal_order in SARIMA_CANDIDATES:
        try:
            model = SARIMAX(
                ts,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fit = model.fit(disp=False)
            aic = fit.aic

            results.append({
                "Model": sarima_label(order, seasonal_order),
                "AIC": aic,
                "Status": "Berhasil"
            })

            if np.isfinite(aic) and aic < best_aic:
                best_aic = aic
                best_order = order
                best_seasonal_order = seasonal_order

        except Exception:
            results.append({
                "Model": sarima_label(order, seasonal_order),
                "AIC": np.nan,
                "Status": "Gagal"
            })

    selection_df = pd.DataFrame(results)

    if best_order is None or best_seasonal_order is None:
        best_order = (1, 2, 2)
        best_seasonal_order = (0, 1, 1, 12)

    model_name = sarima_label(best_order, best_seasonal_order)

    return best_order, best_seasonal_order, model_name, selection_df


@st.cache_data
def make_sarima_forecast(ts, forecast_steps):
    best_order, best_seasonal_order, model_name, selection_df = select_best_sarima(ts)

    model = SARIMAX(
        ts,
        order=best_order,
        seasonal_order=best_seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fit = model.fit(disp=False)

    future_index = pd.date_range(
        start=ts.index.max() + pd.DateOffset(months=1),
        periods=forecast_steps,
        freq="MS"
    )

    forecast = fit.forecast(steps=forecast_steps)
    forecast = pd.Series(forecast.values, index=future_index)

    return forecast, model_name, selection_df


@st.cache_data
def evaluate_sarima(ts):
    train = ts.iloc[:-12]
    test = ts.iloc[-12:]

    best_order, best_seasonal_order, model_name, selection_df = select_best_sarima(train)

    model = SARIMAX(
        train,
        order=best_order,
        seasonal_order=best_seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fit = model.fit(disp=False)
    forecast = fit.forecast(steps=len(test))
    forecast.index = test.index

    mae, rmse, mape, r2 = evaluate_model(test, forecast)

    eval_df = pd.DataFrame({
        "Model": [model_name],
        "MAE": [mae],
        "RMSE": [rmse],
        "MAPE (%)": [mape],
        "R²": [r2]
    })

    comparison_df = pd.DataFrame({
        "Periode": [format_periode(date) for date in test.index],
        "Aktual": test.values,
        "Prediksi": forecast.values
    })

    return eval_df, comparison_df, test, forecast, model_name, selection_df


# ============================================================
# OPERASIONAL
# ============================================================

def build_simulation_table(forecast, biaya_per_ton, kapasitas_truk_compactor_m3, hari_operasional_angkut_per_minggu):
    output = pd.DataFrame({
        "Tanggal": forecast.index,
        "Periode": [format_periode(date) for date in forecast.index],
        "Prediksi Sampah (Ton)": forecast.values
    })

    output["Jumlah Hari"] = output["Tanggal"].dt.days_in_month
    output["Estimasi Anggaran"] = output["Prediksi Sampah (Ton)"] * biaya_per_ton

    output["Estimasi Hari Operasional Angkut"] = np.ceil(
        output["Jumlah Hari"] * (hari_operasional_angkut_per_minggu / 7)
    )

    output["Estimasi Volume Sampah (m³)"] = (
        output["Prediksi Sampah (Ton)"] * 1000 / DENSITAS_SAMPAH_KG_PER_M3
    )

    output["Estimasi Kebutuhan Muatan Truk"] = np.ceil(
        output["Estimasi Volume Sampah (m³)"] / kapasitas_truk_compactor_m3
    )

    output["Muatan Truk per Hari Angkut"] = np.ceil(
        output["Estimasi Kebutuhan Muatan Truk"] / output["Estimasi Hari Operasional Angkut"]
    )

    return output


def prepare_display_table(output):
    display = output.copy()

    display["Prediksi Sampah (Ton)"] = display["Prediksi Sampah (Ton)"].apply(format_angka)
    display["Estimasi Anggaran"] = display["Estimasi Anggaran"].apply(format_rupiah)
    display["Estimasi Volume Sampah (m³)"] = display["Estimasi Volume Sampah (m³)"].apply(format_angka)
    display["Estimasi Hari Operasional Angkut"] = display["Estimasi Hari Operasional Angkut"].astype(int).apply(format_integer)
    display["Estimasi Kebutuhan Muatan Truk"] = display["Estimasi Kebutuhan Muatan Truk"].astype(int).apply(format_integer)
    display["Muatan Truk per Hari Angkut"] = display["Muatan Truk per Hari Angkut"].astype(int).apply(format_integer)

    display = display[
        [
            "Periode",
            "Prediksi Sampah (Ton)",
            "Estimasi Anggaran",
            "Estimasi Volume Sampah (m³)",
            "Estimasi Hari Operasional Angkut",
            "Estimasi Kebutuhan Muatan Truk",
            "Muatan Truk per Hari Angkut"
        ]
    ]

    display = display.rename(columns={
        "Prediksi Sampah (Ton)": "Prediksi<br>Sampah (Ton)",
        "Estimasi Anggaran": "Estimasi<br>Anggaran",
        "Estimasi Volume Sampah (m³)": "Estimasi<br>Volume (m³)",
        "Estimasi Hari Operasional Angkut": "Hari<br>Operasional<br>Angkut",
        "Estimasi Kebutuhan Muatan Truk": "Kebutuhan<br>Muatan<br>Truk",
        "Muatan Truk per Hari Angkut": "Muatan<br>Truk/Hari"
    })

    return display


def prepare_eval_display(eval_df):
    display = eval_df.copy()
    display["MAE"] = display["MAE"].apply(format_angka)
    display["RMSE"] = display["RMSE"].apply(format_angka)
    display["MAPE (%)"] = display["MAPE (%)"].apply(lambda x: f"{x:.2f}%")
    display["R²"] = display["R²"].apply(lambda x: f"{x:.4f}")
    return display


def prepare_comparison_display(comparison_df):
    display = comparison_df.copy()
    display["Aktual"] = display["Aktual"].apply(format_angka)
    display["Prediksi"] = display["Prediksi"].apply(format_angka)
    return display


# ============================================================
# SESSION STATE
# ============================================================

st.session_state.theme_mode = "Gelap"

if "active_menu" not in st.session_state:
    st.session_state.active_menu = "Simulasi Pengelolaan"

if st.session_state.active_menu == "Data Singkat":
    st.session_state.active_menu = "Ringkasan Data & Model"

if st.session_state.active_menu not in MENU_OPTIONS:
    st.session_state.active_menu = "Simulasi Pengelolaan"

if "uploaded_data_payloads" not in st.session_state:
    st.session_state.uploaded_data_payloads = []

if "upload_queue" not in st.session_state:
    st.session_state.upload_queue = []

if "upload_error_messages" not in st.session_state:
    st.session_state.upload_error_messages = []

if "upload_success_message" not in st.session_state:
    st.session_state.upload_success_message = ""

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


def set_theme(mode):
    st.session_state.theme_mode = mode


# ============================================================
# THEME
# ============================================================

def apply_theme(mode):
    if mode == "Terang":
        cfg = {
            "bg": "#EFE9DC",
            "card": "#FFFDF7",
            "card2": "#F5F1E8",
            "text": "#172018",
            "muted": "#3F4B3E",
            "border": "#CFC7B8",
            "accent": "#2E6F4F",
            "accent2": "#B88A3D",
            "accent_soft": "rgba(46, 111, 79, 0.11)",
            "accent_hover": "#2E6F4F",
            "shadow": "rgba(31, 41, 51, 0.06)",
            "hero": "linear-gradient(135deg, #2E6F4F 0%, #6B9A61 56%, #B88A3D 100%)",
            "sidebar_bg": """
                radial-gradient(circle at 14% 8%, rgba(107, 154, 97, 0.22), transparent 24%),
                radial-gradient(circle at 90% 23%, rgba(184, 138, 61, 0.18), transparent 25%),
                linear-gradient(180deg, #EFECDD 0%, #E7E3D3 48%, #EDE3CF 100%)
            """,
            "sidebar_visual": "linear-gradient(135deg, #2E6F4F 0%, #527D52 60%, #8C6B31 100%)",
            "chart_bg": "#FFFDF7",
            "chart_grid": "rgba(23, 32, 24, 0.12)",
            "chart_font": "#172018",
            "chart_axis": "#172018",
            "chart_hist": "#2E6F4F",
            "chart_pred": "#B88A3D",
            "chart_hist_fill": "rgba(46, 111, 79, 0.14)",
            "chart_pred_fill": "rgba(184, 138, 61, 0.16)",
            "chart_divider": "rgba(23, 32, 24, 0.42)",
            "chart_legend_bg": "rgba(255,253,247,0.96)",
            "chart_legend_border": "rgba(63,75,62,0.25)",
            "annotation_bg": "rgba(255,253,247,0.97)",
            "annotation_border": "rgba(63,75,62,0.26)",
            "input_bg": "#FFFDF7",
            "input_btn": "#E6DECE",
            "input_btn_hover": "#2E6F4F",
        }
    else:
        cfg = {
            "bg": "#151A17",
            "card": "#222A24",
            "card2": "#263029",
            "text": "#F5F7F2",
            "muted": "#D8E0D4",
            "border": "#3D4A40",
            "accent": "#8BCB88",
            "accent2": "#E2B15D",
            "accent_soft": "rgba(139, 203, 136, 0.12)",
            "accent_hover": "#2F7D52",
            "shadow": "rgba(0, 0, 0, 0.14)",
            "hero": "linear-gradient(135deg, #1F4D36 0%, #4F8B59 55%, #B78335 100%)",
            "sidebar_bg": """
                radial-gradient(circle at 8% 12%, rgba(139, 203, 136, 0.28), transparent 16%),
                radial-gradient(circle at 92% 28%, rgba(47, 111, 78, 0.18), transparent 17%),
                radial-gradient(circle at 18% 88%, rgba(226, 177, 93, 0.18), transparent 18%),
                linear-gradient(180deg, #1E241F 0%, #1B241C 60%, #222819 100%)
            """,
            "sidebar_visual": "linear-gradient(135deg, #26382C 0%, #2F6F4E 65%, #6A4A1E 100%)",
            "chart_bg": "#111915",
            "chart_grid": "rgba(255,255,255,0.08)",
            "chart_font": "#F2F5F1",
            "chart_axis": "#F2F5F1",
            "chart_hist": "#67F0C1",
            "chart_pred": "#F1BE54",
            "chart_hist_fill": "rgba(103,240,193,0.18)",
            "chart_pred_fill": "rgba(241,190,84,0.18)",
            "chart_divider": "rgba(255,255,255,0.35)",
            "chart_legend_bg": "rgba(20,28,24,0.90)",
            "chart_legend_border": "rgba(255,255,255,0.15)",
            "annotation_bg": "rgba(20,28,24,0.92)",
            "annotation_border": "rgba(255,255,255,0.22)",
            "input_bg": "#222A24",
            "input_btn": "#2B2D3A",
            "input_btn_hover": "#2F7D52",
        }

    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(circle at 2% 4%, rgba(139, 203, 136, 0.06), transparent 20%),
                {cfg["bg"]} !important;
            color: {cfg["text"]} !important;
        }}

        footer, #MainMenu,
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"] {{
            display: none !important;
        }}

        header,
        [data-testid="stHeader"] {{
            visibility: visible !important;
            background: transparent !important;
            height: 0 !important;
            min-height: 0 !important;
        }}

        .mobile-kpi-summary {{
            display: none;
        }}

        h1, h2, h3, h4, h5, h6, p, label, span, div {{
            color: inherit;
        }}

        .block-container {{
            max-width: 1420px !important;
            padding-top: 0rem !important;
            padding-left: 0.72rem !important;
            padding-right: 0.72rem !important;
            padding-bottom: 0rem !important;
        }}

        [data-testid="stMainBlockContainer"] {{
            padding-top: 0rem !important;
        }}

        main .block-container {{
            padding-top: 0rem !important;
        }}

        [data-testid="stSidebar"] {{
            background: {cfg["sidebar_bg"]} !important;
            border-right: 1px solid {cfg["border"]};
            width: 304px !important;
            min-width: 304px !important;
        }}

        [data-testid="stSidebarContent"] {{
            width: 304px !important;
            padding-top: 0rem !important;
        }}

        section[data-testid="stSidebar"] > div {{
            padding-top: 0rem !important;
        }}

        [data-testid="stSidebarUserContent"] {{
            padding-top: 0rem !important;
            margin-top: -2.55rem !important;
            padding-left: 1.05rem !important;
            padding-right: 1.05rem !important;
            padding-bottom: 10px !important;
            overflow-y: hidden !important;
        }}

        [data-testid="stSidebar"],
        [data-testid="stSidebarContent"],
        section[data-testid="stSidebar"],
        [data-testid="stSidebarUserContent"] {{
            overflow-y: hidden !important;
            scrollbar-width: none !important;
        }}

        [data-testid="stSidebar"]::-webkit-scrollbar,
        [data-testid="stSidebarContent"]::-webkit-scrollbar,
        [data-testid="stSidebarUserContent"]::-webkit-scrollbar {{
            display: none !important;
        }}

        [data-testid="stSidebar"] * {{
            color: {cfg["text"]} !important;
        }}

        .theme-label {{
            font-size: 13px;
            font-weight: 800;
            margin-bottom: 8px;
            color: {cfg["text"]} !important;
        }}

        .data-input-title {{
            display: flex;
            align-items: center;
            gap: 10px;
            width: 100%;
            margin-top: 18px;
            margin-bottom: 10px;
            padding: 9px 11px;
            border-radius: 15px;
            border: 1px solid {cfg["border"]};
            background:
                linear-gradient(145deg, rgba(255,255,255,0.030), rgba(255,255,255,0)),
                {cfg["card"]};
            box-shadow: 0 8px 20px {cfg["shadow"]};
            box-sizing: border-box;
        }}

        .data-input-title-text {{
            font-size: 13px;
            font-weight: 900;
            color: {cfg["text"]} !important;
            letter-spacing: 0.01em;
            line-height: 1.1;
        }}

        .modern-section-icon {{
            width: 26px;
            height: 26px;
            min-width: 26px;
            border-radius: 9px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background:
                linear-gradient(135deg, {cfg["accent_soft"]}, rgba(226, 177, 93, 0.11));
            border: 1px solid {cfg["border"]};
            color: {cfg["accent"]} !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
        }}

        .modern-section-icon svg {{
            width: 16px;
            height: 16px;
            display: block;
        }}

        .data-input-note {{
            color: {cfg["muted"]} !important;
            font-size: 11.5px;
            font-weight: 600;
            line-height: 1.45;
            margin-top: -4px;
            margin-bottom: 8px;
        }}

        .data-status {{
            border: 1px solid {cfg["border"]};
            border-radius: 13px;
            background: {cfg["card"]};
            color: {cfg["text"]} !important;
            font-size: 11.2px;
            font-weight: 850;
            line-height: 1.25;
            padding: 8px 10px;
            margin: 6px 0 12px 0;
            box-shadow: 0 7px 16px {cfg["shadow"]};
        }}

        .data-status span {{
            color: {cfg["muted"]} !important;
            font-size: 9.8px;
            font-weight: 650;
        }}

        .modern-status-dot {{
            width: 14px;
            height: 14px;
            border-radius: 5px;
            display: inline-flex;
            vertical-align: -2px;
            margin-right: 6px;
            border: 1px solid {cfg["border"]};
            background: {cfg["accent_soft"]};
            position: relative;
        }}

        .modern-status-dot::after {{
            content: "";
            width: 5px;
            height: 5px;
            border-radius: 50%;
            background: {cfg["accent"]};
            position: absolute;
            left: 4px;
            top: 4px;
        }}

        .data-status.success {{
            border-color: {cfg["accent"]};
            background: {cfg["accent_soft"]};
        }}

        [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] {{
            gap: 0.42rem !important;
        }}

        [data-testid="stSidebar"] .stButton > button {{
            width: 100% !important;
            height: 34px !important;
            min-height: 34px !important;
            border-radius: 12px !important;
            border: 1px solid {cfg["border"]} !important;
            background: {cfg["card"]} !important;
            color: {cfg["text"]} !important;
            font-weight: 850 !important;
            font-size: 12px !important;
            box-shadow: 0 6px 14px rgba(0,0,0,0.08) !important;
            transition: all 0.16s ease-in-out !important;
            cursor: pointer !important;
        }}

        [data-testid="stSidebar"] .stButton > button:hover {{
            background: {cfg["accent_hover"]} !important;
            color: white !important;
            border-color: {cfg["accent_hover"]} !important;
            transform: translateY(-1px);
        }}

        [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] > div:nth-child(1) .stButton > button {{
            background: {cfg["accent_hover"] if mode == "Terang" else cfg["card"]} !important;
            color: {"white" if mode == "Terang" else cfg["text"]} !important;
            border-color: {cfg["accent_hover"] if mode == "Terang" else cfg["border"]} !important;
        }}

        [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] > div:nth-child(2) .stButton > button {{
            background: {cfg["accent_hover"] if mode == "Gelap" else cfg["card"]} !important;
            color: {"white" if mode == "Gelap" else cfg["text"]} !important;
            border-color: {cfg["accent_hover"] if mode == "Gelap" else cfg["border"]} !important;
        }}

        .sidebar-visual {{
            background: {cfg["sidebar_visual"]};
            border: 1px solid rgba(255,255,255,0.18);
            border-radius: 18px;
            padding: 10px 12px;
            margin: 12px 0 0 0 !important;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.15);
            position: relative !important;
            left: auto !important;
            bottom: auto !important;
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
            overflow: hidden;
            z-index: 2;
        }}

        .sidebar-visual::before {{
            content: "";
            position: absolute;
            width: 62px;
            height: 62px;
            right: -18px;
            top: -20px;
            background: rgba(255, 255, 255, 0.14);
            border-radius: 50%;
        }}

        .sidebar-emoji {{
            font-size: 42px;
            line-height: 1;
            margin-bottom: 8px;
            position: relative;
            z-index: 2;
        }}

        .sidebar-icons {{
            display: flex;
            align-items: center;
            gap: 7px;
            margin-bottom: 8px;
            position: relative;
            z-index: 2;
            line-height: 1;
        }}

        .sidebar-icons span {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 30px;
            height: 30px;
            border-radius: 11px;
            background: rgba(255,255,255,0.15);
            border: 1px solid rgba(255,255,255,0.22);
            backdrop-filter: blur(8px);
            color: white !important;
            box-shadow: 0 10px 22px rgba(0,0,0,0.16);
        }}

        .sidebar-icons svg {{
            width: 16px;
            height: 16px;
            display: block;
            stroke: currentColor;
        }}

        .sidebar-visual-title {{
            font-size: 14.3px;
            font-weight: 850;
            position: relative;
            z-index: 2;
            color: white !important;
        }}

        .sidebar-visual-subtitle {{
            font-size: 10.4px;
            color: white !important;
            margin-top: 3px;
            line-height: 1.30;
            position: relative;
            z-index: 2;
            font-weight: 500;
        }}

        .team-name {{
            margin-top: 7px;
            padding: 6px 8px;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.16);
            border: 1px solid rgba(255, 255, 255, 0.24);
            font-size: 10.4px;
            font-weight: 760;
            position: relative;
            z-index: 2;
            color: white !important;
            line-height: 1.45;
            text-align: center !important;
            display: flex;
            justify-content: center;
            align-items: center;
        }}

        .hero {{
            background: {cfg["hero"]};
            color: white !important;
            padding: 24px 31px;
            border-radius: 24px;
            margin-top: -130px !important;
            margin-bottom: 24px;
            box-shadow: 0 18px 42px rgba(31, 41, 51, 0.18);
        }}

        .hero * {{
            color: white !important;
        }}

        .hero-title {{
            font-size: 35px;
            font-weight: 850;
            line-height: 1.12;
            margin-bottom: 8px;
        }}

        .hero-subtitle {{
            font-size: 15px;
            max-width: 1120px;
            opacity: 0.96;
            line-height: 1.6;
        }}

        .section-title {{
            font-size: 25px;
            font-weight: 850;
            color: {cfg["text"]} !important;
            margin-bottom: 7px;
        }}

        .section-desc {{
            color: {cfg["muted"]} !important;
            font-size: 14px;
            font-weight: 500;
            margin-bottom: 22px;
            line-height: 1.65;
        }}

        .small-title {{
            font-size: 16.5px;
            font-weight: 800;
            color: {cfg["text"]} !important;
            margin-bottom: 12px;
        }}

        .info-card {{
            background:
                linear-gradient(145deg, rgba(255,255,255,0.035), rgba(255,255,255,0)),
                {cfg["card"]};
            border: 1px solid {cfg["border"]};
            border-radius: 22px;
            padding: 22px 24px;
            margin-bottom: 22px;
            min-height: auto;
            box-shadow: 0 12px 30px {cfg["shadow"]};
            position: relative;
            overflow: hidden;
        }}

        .info-card::before {{
            content: "";
            position: absolute;
            inset: 0;
            border-radius: 22px;
            background:
                radial-gradient(circle at 6% 12%, {cfg["accent_soft"]}, transparent 32%),
                radial-gradient(circle at 96% 8%, rgba(226, 177, 93, 0.09), transparent 30%);
            pointer-events: none;
        }}

        .info-card > * {{
            position: relative;
            z-index: 2;
        }}

        .text-muted {{
            color: {cfg["muted"]} !important;
            font-size: 14px;
            font-weight: 500;
            line-height: 1.7;
        }}

        .text-muted li {{
            color: {cfg["muted"]} !important;
            margin-bottom: 8px;
        }}

        .kpi-card {{
            background:
                linear-gradient(145deg, rgba(255,255,255,0.035), rgba(255,255,255,0.000)),
                {cfg["card"]};
            border: 1px solid {cfg["border"]};
            border-radius: 20px;
            padding: 12px 15px 15px 15px;
            height: 120px;
            min-height: 120px;
            box-shadow: 0 10px 26px {cfg["shadow"]};
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: stretch;
            gap: 5px;
            transform: translateY(-1px);
            margin-bottom: 14px;
            box-sizing: border-box;
            overflow: hidden;
            position: relative;
            width: 100%;
        }}

        .kpi-card::before {{
            content: "";
            position: absolute;
            inset: 0;
            border-radius: 22px;
            background:
                radial-gradient(circle at 16% 8%, {cfg["accent_soft"]}, transparent 34%),
                radial-gradient(circle at 95% 12%, rgba(226, 177, 93, 0.10), transparent 32%);
            pointer-events: none;
        }}

        .kpi-header {{
            position: relative;
            z-index: 2;
            display: flex;
            align-items: center;
            gap: 9px;
            height: 27px;
            min-height: 27px;
            width: 100%;
        }}

        .kpi-icon {{
            width: 26px;
            height: 26px;
            min-width: 26px;
            border-radius: 9px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: {cfg["accent"]} !important;
            background:
                linear-gradient(135deg, {cfg["accent_soft"]}, rgba(226, 177, 93, 0.12));
            border: 1px solid {cfg["border"]};
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
        }}

        .kpi-icon svg {{
            width: 15.5px;
            height: 15.5px;
            display: block;
        }}

        .kpi-label {{
            color: {cfg["muted"]} !important;
            font-size: 12.4px;
            font-weight: 900;
            line-height: 1.15;
            letter-spacing: -0.1px;
            margin: 0;
            max-width: 100%;
            overflow: hidden;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
        }}

        .kpi-value {{
            position: relative;
            z-index: 2;
            color: {cfg["text"]} !important;
            font-size: clamp(18px, 1.15vw, 27px);
            font-weight: 950;
            line-height: 1.06;
            letter-spacing: -0.45px;
            margin: 0;
            min-height: 28px;
            display: flex;
            align-items: center;
            overflow-wrap: normal;
            word-break: keep-all;
            white-space: nowrap;
        }}

        .kpi-value-long {{
            font-size: clamp(15px, 0.95vw, 22px);
            line-height: 1.06;
            letter-spacing: -0.45px;
        }}

        .kpi-value-period {{
            font-size: clamp(17px, 1.05vw, 24px);
            line-height: 1.10;
            white-space: normal;
            word-break: normal;
            overflow-wrap: normal;
        }}

        .kpi-note {{
            position: relative;
            z-index: 2;
            color: {cfg["muted"]} !important;
            font-size: 10.8px;
            font-weight: 760;
            line-height: 1.12;
            margin: 0;
            opacity: 0.95;
            min-height: 13px;
            overflow: hidden;
            white-space: nowrap;
            text-overflow: ellipsis;
        }}

        body:not(.sidebar-custom-closed) .kpi-card {{
            padding-left: 16px !important;
            padding-right: 16px !important;
        }}

        body:not(.sidebar-custom-closed) .kpi-value {{
            font-size: clamp(17px, 1.05vw, 25px) !important;
        }}

        body:not(.sidebar-custom-closed) .kpi-value-long {{
            font-size: clamp(14px, 0.88vw, 20px) !important;
        }}

        body:not(.sidebar-custom-closed) .kpi-value-period {{
            font-size: clamp(16px, 0.98vw, 22px) !important;
            white-space: normal !important;
        }}

        .mobile-kpi-label-row {{
            display: flex;
            align-items: center;
            gap: 7px;
            min-width: 0;
        }}

        .mobile-kpi-icon {{
            width: 20px;
            height: 20px;
            min-width: 20px;
            border-radius: 8px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: {cfg["accent"]} !important;
            background: {cfg["accent_soft"]};
            border: 1px solid {cfg["border"]};
        }}

        .mobile-kpi-icon svg {{
            width: 12px;
            height: 12px;
            display: block;
        }}
        [data-testid="stFileUploader"] section {{
            background: {cfg["card"]} !important;
            border: 1px dashed {cfg["border"]} !important;
            border-radius: 13px !important;
            padding: 8px !important;
            min-height: 118px !important;
        }}

        [data-testid="stFileUploader"] section:hover {{
            border-color: {cfg["accent"]} !important;
        }}

        [data-testid="stFileUploader"] section {{
            text-align: center !important;
        }}

        [data-testid="stFileUploader"] section small,
        [data-testid="stFileUploader"] section p,
        [data-testid="stFileUploader"] section span {{
            text-align: center !important;
        }}

        [data-testid="stFileUploader"] section small {{
            display: block !important;
            width: 100% !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }}

        [data-testid="stFileUploader"] small {{
            color: {cfg["muted"]} !important;
        }}

        [data-testid="stFileUploader"] section > div {{
            width: 100% !important;
        }}


        [data-testid="stFileUploader"] section div,
        [data-testid="stFileUploader"] section label {{
            text-align: center !important;
            justify-content: center !important;
        }}

        [data-testid="stFileUploader"] section [data-testid="stMarkdownContainer"] {{
            width: 100% !important;
            text-align: center !important;
        }}

        [data-testid="stFileUploader"] section button,
        [data-testid="stFileUploader"] button[data-testid="baseButton-secondary"],
        [data-testid="stFileUploader"] button[kind="secondary"] {{
            width: 100% !important;
            min-height: 38px !important;
            height: 38px !important;
            border-radius: 12px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 10px !important;
            font-weight: 850 !important;
            font-size: 13.5px !important;
            background: {cfg["input_btn"]} !important;
            color: {cfg["text"]} !important;
            border: 1px solid {cfg["border"]} !important;
        }}

        [data-testid="stFileUploader"] section button:hover,
        [data-testid="stFileUploader"] button[data-testid="baseButton-secondary"]:hover,
        [data-testid="stFileUploader"] button[kind="secondary"]:hover {{
            background: {cfg["accent_hover"]} !important;
            color: white !important;
            border-color: {cfg["accent_hover"]} !important;
        }}

        div[data-baseweb="select"] > div {{
            background: {cfg["card"]} !important;
            border: 1px solid {cfg["border"]} !important;
            border-radius: 14px !important;
            color: {cfg["text"]} !important;
            min-height: 42px !important;
            cursor: pointer !important;
            box-shadow: none !important;
        }}

        div[data-baseweb="select"] span {{
            color: {cfg["text"]} !important;
            font-weight: 650 !important;
        }}

        div[data-baseweb="select"] svg {{
            color: {cfg["accent"]} !important;
            fill: {cfg["accent"]} !important;
            opacity: 1 !important;
            transition: transform 0.18s ease-in-out !important;
            transform-origin: center !important;
        }}

        /* Dropdown EDA/select: saat opsi terbuka, chevron kanan berubah arah ke atas */
        div[data-baseweb="select"] [aria-expanded="true"] svg,
        div[data-baseweb="select"]:has([aria-expanded="true"]) svg {{
            transform: rotate(180deg) !important;
        }}

        /* Hilangkan scrollbar ganda pada panel opsi select */
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] > div,
        div[data-baseweb="popover"] > div > div {{
            border: none !important;
            outline: none !important;
            background: transparent !important;
            box-shadow: none !important;
        }}

        div[data-baseweb="popover"] ul,
        div[role="listbox"] {{
            border: none !important;
            outline: none !important;
            scrollbar-width: none !important;
            -ms-overflow-style: none !important;
        }}

        div[data-baseweb="popover"] ul::-webkit-scrollbar,
        div[role="listbox"]::-webkit-scrollbar {{
            display: none !important;
            width: 0 !important;
            height: 0 !important;
        }}

        [data-testid="stNumberInput"] {{
            border-radius: 14px !important;
        }}

        [data-testid="stNumberInput"] > div {{
            border: 1px solid {cfg["border"]} !important;
            border-radius: 14px !important;
            overflow: hidden !important;
            box-shadow: none !important;
            outline: none !important;
            background: {cfg["input_bg"]} !important;
        }}

        [data-testid="stNumberInput"] div[data-baseweb="input"] {{
            background: {cfg["input_bg"]} !important;
            border: none !important;
            box-shadow: none !important;
            outline: none !important;
        }}

        [data-testid="stNumberInput"] div[data-baseweb="input"] > div {{
            border: none !important;
            box-shadow: none !important;
            outline: none !important;
        }}

        [data-testid="stNumberInput"] input {{
            background: {cfg["input_bg"]} !important;
            border: none !important;
            box-shadow: none !important;
            outline: none !important;
            color: {cfg["text"]} !important;
            min-height: 42px !important;
            font-weight: 650 !important;
        }}

        [data-testid="stNumberInput"] input:focus {{
            border: none !important;
            box-shadow: none !important;
            outline: none !important;
        }}

        [data-testid="stNumberInput"] button {{
            background: {cfg["input_btn"]} !important;
            color: {cfg["text"]} !important;
            border: none !important;
            border-left: 1px solid {cfg["border"]} !important;
            min-height: 42px !important;
            height: 42px !important;
            margin-top: -3px !important;
            padding-top: 2px !important;
            padding-bottom: 0px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            box-shadow: none !important;
            outline: none !important;
        }}

        [data-testid="stNumberInput"] button:hover {{
            background: {cfg["input_btn_hover"]} !important;
            color: white !important;
        }}

        .stSlider label, .stSelectbox label, .stRadio label, .stNumberInput label {{
            color: {cfg["text"]} !important;
            font-weight: 700 !important;
        }}

        [data-testid="stSlider"] span {{
            color: {cfg["text"]} !important;
            font-weight: 700 !important;
        }}

        [data-testid="stSidebar"] [role="radiogroup"] label {{
            border-radius: 12px !important;
            padding: 4px 8px !important;
            transition: all 0.16s ease-in-out !important;
            cursor: pointer !important;
        }}

        [data-testid="stSidebar"] [role="radiogroup"] label:hover {{
            background: {cfg["accent_soft"]} !important;
            transform: translateX(2px);
        }}


        [data-testid="stSidebar"] [data-testid="stFileUploader"] label p {{
            font-size: 16px !important;
            line-height: 1.25 !important;
            margin-bottom: 6px !important;
            letter-spacing: 0.01em !important;
        }}

        [data-testid="stSidebar"] [data-testid="stFileUploader"] small,
        [data-testid="stSidebar"] [data-testid="stFileUploader"] section small {{
            font-size: 11px !important;
            line-height: 1.25 !important;
            margin-top: 2px !important;
        }}

        [data-testid="stSidebar"] [data-testid="stFileUploader"] {{
            margin-bottom: 6px !important;
        }}

        [data-testid="stSidebar"] .stRadio > label p {{
            font-size: 15px !important;
            line-height: 1.2 !important;
            margin-bottom: 4px !important;
        }}

        [data-testid="stSidebar"] [role="radiogroup"] {{
            gap: 2px !important;
        }}

        [data-testid="stSidebar"] [role="radiogroup"] label {{
            min-height: 34px !important;
            padding-top: 3px !important;
            padding-bottom: 3px !important;
        }}

        [data-testid="stSidebar"] [role="radiogroup"] label p {{
            font-size: 13.2px !important;
            line-height: 1.2 !important;
        }}

        
        [data-testid="stSidebar"] .theme-label {{
            font-size: 14px !important;
            margin-bottom: 5px !important;
        }}

        [data-testid="stSidebar"] hr {{
            margin: 8px 0 !important;
        }}

        [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {{
            margin-bottom: 0.35rem !important;
        }}

        .custom-table-wrapper {{
            width: 100%;
            overflow-x: hidden !important;
            border: none !important;
            border-radius: 18px;
            background: transparent !important;
            margin-bottom: 24px;
            box-shadow: none !important;
        }}

        table.custom-table {{
            width: 100%;
            table-layout: fixed !important;
            border-collapse: separate;
            border-spacing: 0;
            background: transparent !important;
            color: {cfg["text"]} !important;
            font-size: 11.2px;
            line-height: 1.16;
            margin: 0 !important;
            border: none !important;
            border-radius: 18px;
            overflow: hidden;
        }}

        table.custom-table thead tr th {{
            background: {cfg["card2"]} !important;
            color: {cfg["text"]} !important;
            font-weight: 900;
            padding: 8px 8px;
            height: 38px;
            border-top: 1px solid {cfg["border"]};
            border-bottom: 1px solid {cfg["border"]};
            border-right: 1px solid {cfg["border"]};
            text-align: left;
            white-space: normal;
            overflow-wrap: anywhere;
            word-break: normal;
            vertical-align: middle;
        }}

        table.custom-table thead tr th:nth-child(1) {{ width: 10.5%; }}
        table.custom-table thead tr th:nth-child(2) {{ width: 12%; }}
        table.custom-table thead tr th:nth-child(3) {{ width: 13.5%; }}
        table.custom-table thead tr th:nth-child(4) {{ width: 14%; }}
        table.custom-table thead tr th:nth-child(5) {{ width: 15%; }}
        table.custom-table thead tr th:nth-child(6) {{ width: 16%; }}
        table.custom-table thead tr th:nth-child(7) {{ width: 14%; }}

        table.custom-table thead tr th:first-child {{
            border-left: 1px solid {cfg["border"]};
            border-top-left-radius: 18px;
        }}

        table.custom-table thead tr th:last-child {{
            border-top-right-radius: 18px;
        }}

        table.custom-table tbody tr td,
        table.custom-table tbody tr th {{
            background: {cfg["card"]} !important;
            color: {cfg["text"]} !important;
            padding: 8px 8px;
            height: 36px;
            border-right: 1px solid {cfg["border"]};
            border-bottom: 1px solid {cfg["border"]};
            font-weight: 700;
            white-space: normal;
            overflow-wrap: anywhere;
            word-break: normal;
            vertical-align: middle;
        }}

        table.custom-table tbody tr td:first-child,
        table.custom-table tbody tr th:first-child {{
            border-left: 1px solid {cfg["border"]};
        }}

        table.custom-table tbody tr:last-child td,
        table.custom-table tbody tr:last-child th {{
            height: 36px !important;
            padding-top: 8px !important;
            padding-bottom: 8px !important;
        }}

        table.custom-table tbody tr:last-child td:first-child,
        table.custom-table tbody tr:last-child th:first-child {{
            border-bottom-left-radius: 18px;
        }}

        table.custom-table tbody tr:last-child td:last-child,
        table.custom-table tbody tr:last-child th:last-child {{
            border-bottom-right-radius: 18px;
        }}

        .stPlotlyChart {{
            background: {cfg["chart_bg"]} !important;
            border: 1.5px solid {cfg["border"]} !important;
            border-radius: 20px !important;
            padding: 4px 4px 4px 4px !important;
            margin-bottom: 22px !important;
            box-shadow: 0 8px 26px {cfg["shadow"]} !important;
            overflow: hidden !important;
        }}

        .stPlotlyChart > div {{
            border-radius: 16px !important;
            overflow: hidden !important;
        }}


        .desktop-chart {{
            display: block;
        }}

        .mobile-chart {{
            display: none;
        }}


        div[data-testid="stExpander"] [role="radiogroup"] label:hover::before,
        div[data-testid="stExpander"] [role="radiogroup"] > div:hover label::before,
        div[data-testid="stExpander"] [role="radiogroup"] > div:hover div + div::before {{
            border-color: #FF4B4B !important;
        }}
        @media screen and (max-width: 900px) {{
            .block-container {{
                padding-top: 0.7rem !important;
                padding-left: 0.62rem !important;
                padding-right: 0.62rem !important;
                padding-bottom: 1rem !important;
                max-width: 100% !important;
            }}

            header,
            [data-testid="stHeader"] {{
                visibility: visible !important;
                background: transparent !important;
                height: 46px !important;
                min-height: 46px !important;
            }}

            [data-testid="stToolbar"],
            [data-testid="stDecoration"],
            [data-testid="stStatusWidget"] {{
                display: none !important;
            }}

            [data-testid="stSidebarCollapsedControl"],
            [data-testid="collapsedControl"],
            [data-testid="stSidebarNav"],
            button[title="Open sidebar"],
            button[aria-label="Open sidebar"],
            button[data-testid="stBaseButton-headerNoPadding"],
            button[data-testid="baseButton-headerNoPadding"],
            button[kind="headerNoPadding"] {{
                display: flex !important;
                visibility: visible !important;
                opacity: 1 !important;
                position: fixed !important;
                top: 10px !important;
                left: 10px !important;
                z-index: 999999 !important;
                background: {cfg["card"]} !important;
                border: 1px solid {cfg["border"]} !important;
                border-radius: 13px !important;
                box-shadow: 0 8px 22px {cfg["shadow"]} !important;
                width: 42px !important;
                height: 42px !important;
                min-width: 42px !important;
                min-height: 42px !important;
                align-items: center !important;
                justify-content: center !important;
                color: {cfg["text"]} !important;
            }}

            [data-testid="stSidebarCollapsedControl"]::before,
            [data-testid="collapsedControl"]::before,
            button[title="Open sidebar"]::before,
            button[aria-label="Open sidebar"]::before,
            button[data-testid="stBaseButton-headerNoPadding"]::before,
            button[data-testid="baseButton-headerNoPadding"]::before,
            button[kind="headerNoPadding"]::before {{
                content: "☰";
                font-size: 22px;
                line-height: 1;
                font-weight: 900;
                color: {cfg["text"]} !important;
            }}

            [data-testid="stSidebarCollapsedControl"] svg,
            [data-testid="collapsedControl"] svg,
            button[title="Open sidebar"] svg,
            button[aria-label="Open sidebar"] svg,
            button[data-testid="stBaseButton-headerNoPadding"] svg,
            button[data-testid="baseButton-headerNoPadding"] svg,
            button[kind="headerNoPadding"] svg {{
                width: 22px !important;
                height: 22px !important;
                color: {cfg["text"]} !important;
                fill: {cfg["text"]} !important;
            }}

            [data-testid="stSidebar"] {{
                width: 286px !important;
                min-width: 286px !important;
            }}

            [data-testid="stSidebarContent"] {{
                width: 286px !important;
            }}

            [data-testid="stSidebarUserContent"] {{
                padding-left: 1rem !important;
                padding-right: 1rem !important;
                padding-bottom: 260px !important;
                margin-top: 0rem !important;
            }}

            [data-testid="stSidebar"] .theme-label {{
                margin-top: 48px !important;
                margin-bottom: 8px !important;
            }}

            [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] {{
                display: grid !important;
                grid-template-columns: 1fr 1fr !important;
                gap: 0.55rem !important;
                width: 100% !important;
            }}

            [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {{
                width: 100% !important;
                min-width: 0 !important;
                flex: none !important;
            }}

            [data-testid="stSidebar"] .stButton > button {{
                height: 38px !important;
                border-radius: 13px !important;
            }}

            [data-testid="stAppViewContainer"] {{
                overflow-x: hidden !important;
            }}

            .hero {{
                padding: 16px 14px !important;
                border-radius: 19px !important;
                margin-top: 50px !important;
                margin-bottom: 14px !important;
            }}

            .hero-title {{
                font-size: 18px !important;
                line-height: 1.25 !important;
                margin-bottom: 7px !important;
                letter-spacing: -0.2px !important;
            }}

            .hero-subtitle {{
                font-size: 12px !important;
                line-height: 1.5 !important;
            }}

            .section-title {{
                font-size: 17px !important;
                margin-bottom: 5px !important;
            }}

            .section-desc {{
                font-size: 12.2px !important;
                line-height: 1.5 !important;
                margin-bottom: 10px !important;
            }}

            .small-title {{
                font-size: 14px !important;
                margin-top: 2px !important;
                margin-bottom: 5px !important;
            }}

            div[data-testid="stHorizontalBlock"] {{
                gap: 0.45rem !important;
            }}

            div[data-testid="column"] {{
                min-width: 0 !important;
                padding-left: 0 !important;
                padding-right: 0 !important;
            }}

            /* Mobile khusus: slider full 1 baris, 3 input angka sejajar 1 baris */
            div[data-testid="stHorizontalBlock"]:has([data-testid="stNumberInput"]) {{
                display: grid !important;
                grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
                gap: 0.40rem !important;
                align-items: end !important;
            }}

            div[data-testid="stHorizontalBlock"]:has([data-testid="stNumberInput"]) > div[data-testid="column"] {{
                width: 100% !important;
                min-width: 0 !important;
                max-width: 100% !important;
                flex: none !important;
            }}

            div[data-testid="stHorizontalBlock"]:has([data-testid="stNumberInput"]) > div[data-testid="column"]:first-child {{
                grid-column: 1 / -1 !important;
            }}

            div[data-testid="stHorizontalBlock"]:has([data-testid="stNumberInput"]) > div[data-testid="column"]:nth-child(2) {{
                grid-column: 1 / 2 !important;
            }}

            div[data-testid="stHorizontalBlock"]:has([data-testid="stNumberInput"]) > div[data-testid="column"]:nth-child(3) {{
                grid-column: 2 / 3 !important;
            }}

            div[data-testid="stHorizontalBlock"]:has([data-testid="stNumberInput"]) > div[data-testid="column"]:nth-child(4) {{
                grid-column: 3 / 4 !important;
            }}

            /* Perbaikan khusus HP: slider satu baris penuh, 3 input angka sejajar di bawahnya */
            div[data-testid="stHorizontalBlock"]:has(> div:nth-child(1) [data-testid="stSlider"]):has(> div:nth-child(2) [data-testid="stNumberInput"]) {{
                display: grid !important;
                grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
                gap: 0.42rem !important;
                align-items: end !important;
                width: 100% !important;
            }}

            div[data-testid="stHorizontalBlock"]:has(> div:nth-child(1) [data-testid="stSlider"]):has(> div:nth-child(2) [data-testid="stNumberInput"]) > div:nth-child(1) {{
                grid-column: 1 / -1 !important;
                width: 100% !important;
                max-width: 100% !important;
            }}

            div[data-testid="stHorizontalBlock"]:has(> div:nth-child(1) [data-testid="stSlider"]):has(> div:nth-child(2) [data-testid="stNumberInput"]) > div:nth-child(2) {{
                grid-column: 1 / 2 !important;
            }}

            div[data-testid="stHorizontalBlock"]:has(> div:nth-child(1) [data-testid="stSlider"]):has(> div:nth-child(2) [data-testid="stNumberInput"]) > div:nth-child(3) {{
                grid-column: 2 / 3 !important;
            }}

            div[data-testid="stHorizontalBlock"]:has(> div:nth-child(1) [data-testid="stSlider"]):has(> div:nth-child(2) [data-testid="stNumberInput"]) > div:nth-child(4) {{
                grid-column: 3 / 4 !important;
            }}

            div[data-testid="stHorizontalBlock"]:has(> div:nth-child(1) [data-testid="stSlider"]):has(> div:nth-child(2) [data-testid="stNumberInput"]) > div:nth-child(2),
            div[data-testid="stHorizontalBlock"]:has(> div:nth-child(1) [data-testid="stSlider"]):has(> div:nth-child(2) [data-testid="stNumberInput"]) > div:nth-child(3),
            div[data-testid="stHorizontalBlock"]:has(> div:nth-child(1) [data-testid="stSlider"]):has(> div:nth-child(2) [data-testid="stNumberInput"]) > div:nth-child(4) {{
                width: 100% !important;
                min-width: 0 !important;
                max-width: 100% !important;
                flex: none !important;
            }}

            div[data-testid="stHorizontalBlock"]:has(> div:nth-child(1) [data-testid="stSlider"]):has(> div:nth-child(2) [data-testid="stNumberInput"]) label p {{
                font-size: 9.2px !important;
                line-height: 1.12 !important;
                white-space: normal !important;
            }}

            .stSlider {{
                margin-bottom: 2px !important;
            }}

            [data-testid="stSlider"] {{
                margin-bottom: 2px !important;
            }}

            [data-testid="stSlider"] span {{
                font-size: 11.5px !important;
            }}

            .stNumberInput {{
                margin-bottom: 4px !important;
            }}

            [data-testid="stNumberInput"] {{
                margin-top: 0px !important;
                margin-bottom: 4px !important;
            }}

            [data-testid="stNumberInput"] label {{
                font-size: 9.2px !important;
                margin-bottom: 4px !important;
                line-height: 1.10 !important;
                min-height: 24px !important;
            }}

            [data-testid="stNumberInput"] div[data-baseweb="input"] {{
                min-height: 36px !important;
                overflow: hidden !important;
            }}

            [data-testid="stNumberInput"] input {{
                height: 34px !important;
                min-height: 34px !important;
                font-size: 9.8px !important;
                font-weight: 800 !important;
                padding-top: 0 !important;
                padding-bottom: 0 !important;
                padding-left: 6px !important;
                padding-right: 2px !important;
            }}

            [data-testid="stNumberInput"] > div {{
                width: 100% !important;
                min-width: 0 !important;
                overflow: hidden !important;
            }}

            [data-testid="stNumberInput"] div[data-baseweb="input"] > div {{
                width: 100% !important;
                min-width: 0 !important;
                overflow: hidden !important;
            }}

            [data-testid="stNumberInput"] button {{
                height: 34px !important;
                min-height: 34px !important;
                width: 24px !important;
                min-width: 24px !important;
                max-width: 24px !important;
                margin-top: 0px !important;
                padding: 0 !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                font-size: 11px !important;
                line-height: 1 !important;
                overflow: visible !important;
                box-sizing: border-box !important;
            }}

            .mobile-kpi-summary {{
                display: block !important;
                background: {cfg["card"]};
                border: 1px solid {cfg["border"]};
                border-radius: 18px;
                padding: 14px 14px 12px 14px;
                margin: 10px 0 14px 0;
                box-shadow: 0 8px 20px {cfg["shadow"]};
            }}

            .mobile-kpi-summary-title {{
                font-size: 14px;
                font-weight: 900;
                color: {cfg["text"]} !important;
                margin-bottom: 10px;
            }}

            .mobile-kpi-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px 12px;
            }}

            .mobile-kpi-item {{
                border-bottom: 1px solid {cfg["border"]};
                padding-bottom: 8px;
                min-width: 0;
            }}

            .mobile-kpi-item:nth-last-child(-n+2) {{
                border-bottom: none;
                padding-bottom: 0;
            }}

            .mobile-kpi-label {{
                font-size: 10.5px;
                line-height: 1.25;
                font-weight: 850;
                color: {cfg["muted"]} !important;
                margin-bottom: 3px;
            }}

            .mobile-kpi-value {{
                font-size: 13.5px;
                line-height: 1.25;
                font-weight: 950;
                color: {cfg["text"]} !important;
                overflow-wrap: anywhere;
            }}

            .mobile-kpi-note {{
                font-size: 9.8px;
                line-height: 1.25;
                font-weight: 650;
                color: {cfg["muted"]} !important;
                margin-top: 2px;
            }}

            .kpi-card {{
                display: none !important;
            }}

            .info-card {{
                min-height: auto !important;
                padding: 13px 13px !important;
                border-radius: 16px !important;
                margin-bottom: 10px !important;
            }}

            .text-muted {{
                font-size: 12.2px !important;
                line-height: 1.5 !important;
            }}

            .text-muted li {{
                margin-bottom: 5px !important;
            }}

            .desktop-chart {{
                display: block !important;
            }}

            .mobile-chart {{
                display: none !important;
            }}


            .stPlotlyChart {{
                border-radius: 14px !important;
                padding: 0px !important;
                margin-bottom: 2px !important;
                max-height: 340px !important;
                overflow: hidden !important;
            }}

            .stPlotlyChart > div,
            .stPlotlyChart .js-plotly-plot,
            .stPlotlyChart .plot-container,
            .stPlotlyChart .svg-container {{
                max-height: 340px !important;
            }}

            .stPlotlyChart svg {{
                max-height: 340px !important;
            }}

            .stPlotlyChart svg .gtitle {{
                font-size: 10.5px !important;
            }}

            .stPlotlyChart svg .xtitle,
            .stPlotlyChart svg .ytitle {{
                font-size: 10px !important;
            }}

            .stPlotlyChart svg .legend text {{
                font-size: 9px !important;
            }}

            .stPlotlyChart svg .annotation-text,
            .stPlotlyChart svg .annotation text {{
                font-size: 9px !important;
            }}

            .custom-table-wrapper {{
                border-radius: 15px !important;
                margin-bottom: 10px !important;
                overflow-x: hidden !important;
                width: 100% !important;
            }}

            table.custom-table {{
                width: 100% !important;
                min-width: 0 !important;
                table-layout: fixed !important;
                font-size: 8.3px !important;
            }}

            table.custom-table thead tr th,
            table.custom-table tbody tr td,
            table.custom-table tbody tr th {{
                padding: 5px 3px !important;
                height: auto !important;
                min-height: 28px !important;
                font-size: 8.3px !important;
                line-height: 1.2 !important;
                white-space: normal !important;
                overflow-wrap: anywhere !important;
                word-break: break-word !important;
                vertical-align: middle !important;
            }}

            table.custom-table thead tr th {{
                font-size: 8px !important;
                font-weight: 900 !important;
            }}

            .sidebar-visual {{
                display: block !important;
                position: fixed !important;
                left: 16px !important;
                bottom: 22px !important;
                width: 218px !important;
                max-width: 218px !important;
                padding: 14px 13px !important;
                border-radius: 20px !important;
                z-index: 25 !important;
            }}

            .sidebar-emoji {{
                font-size: 30px !important;
                margin-bottom: 6px !important;
            }}

            .sidebar-icons {{
                gap: 10px !important;
                margin-bottom: 8px !important;
                font-size: 24px !important;
            }}

            .sidebar-icons span {{
                width: 28px !important;
                height: 28px !important;
            }}

            .sidebar-visual-title {{
                font-size: 15px !important;
            }}

            .sidebar-visual-subtitle {{
                font-size: 10.8px !important;
                line-height: 1.35 !important;
            }}

            .team-name {{
                font-size: 10.5px !important;
                padding: 7px 8px !important;
                margin-top: 8px !important;
            }}

            /* FINAL MOBILE ONLY OVERRIDES */
            [data-testid="stSidebar"] .theme-label {{
                margin-top: 0px !important;
                margin-bottom: 7px !important;
                font-size: 13px !important;
                font-weight: 900 !important;
                line-height: 1.15 !important;
            }}

            .data-input-title {{
                margin-top: 12px !important;
                margin-bottom: 6px !important;
                font-size: 12.5px !important;
            }}

            .data-input-note {{
                font-size: 10.5px !important;
                line-height: 1.35 !important;
                margin-bottom: 6px !important;
            }}

            .data-status {{
                font-size: 11px !important;
                padding: 8px 9px !important;
                margin-bottom: 12px !important;
            }}

            /* Mobile: segmented button ☀️ / 🌙 dibuat simetris */
            [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.stButton) {{
                width: 188px !important;
                max-width: 188px !important;
                height: 48px !important;
                min-height: 48px !important;
                display: grid !important;
                grid-template-columns: 94px 94px !important;
                gap: 0 !important;
                column-gap: 0 !important;
                row-gap: 0 !important;
                padding: 0 !important;
                margin: 0 !important;
                border-radius: 21px !important;
                background: {cfg["card"]} !important;
                border: 1px solid {cfg["border"]} !important;
                box-shadow: 0 8px 18px {cfg["shadow"]} !important;
                overflow: hidden !important;
            }}

            [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.stButton) > div,
            [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.stButton) > div[data-testid="column"] {{
                width: 94px !important;
                min-width: 94px !important;
                max-width: 94px !important;
                flex: 0 0 94px !important;
                padding: 0 !important;
                margin: 0 !important;
            }}

            [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.stButton) .stButton {{
                width: 94px !important;
                min-width: 94px !important;
                max-width: 94px !important;
                height: 48px !important;
                padding: 0 !important;
                margin: 0 !important;
            }}

            [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.stButton) .stButton > button {{
                width: 94px !important;
                min-width: 94px !important;
                max-width: 94px !important;
                height: 48px !important;
                min-height: 48px !important;
                padding: 0 !important;
                margin: 0 !important;
                border: none !important;
                box-shadow: none !important;
                outline: none !important;
                font-size: 22px !important;
                line-height: 1 !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                text-align: center !important;
                background: transparent !important;
                transform: none !important;
            }}

            [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.stButton) > div:nth-child(1) .stButton > button {{
                border-radius: 20px 0 0 20px !important;
                background: {cfg["accent_hover"] if mode == "Terang" else "transparent"} !important;
                color: {"white" if mode == "Terang" else cfg["text"]} !important;
            }}

            [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.stButton) > div:nth-child(2) .stButton > button {{
                border-radius: 0 20px 20px 0 !important;
                background: {cfg["accent_hover"] if mode == "Gelap" else "transparent"} !important;
                color: {"white" if mode == "Gelap" else cfg["text"]} !important;
            }}

            [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.stButton) .stButton > button:hover {{
                background: {cfg["accent_hover"]} !important;
                color: white !important;
                transform: none !important;
            }}

            /* mobile number input: rapikan area input dan tombol +/- */
            [data-testid="stNumberInput"] {{
                margin-top: 0 !important;
                margin-bottom: 3px !important;
            }}

            [data-testid="stNumberInput"] > div {{
                border-radius: 14px !important;
                min-height: 36px !important;
                height: 36px !important;
                overflow: hidden !important;
                background: {cfg["input_bg"]} !important;
                border: 1px solid {cfg["border"]} !important;
                box-shadow: none !important;
            }}

            [data-testid="stNumberInput"] div[data-baseweb="input"] {{
                min-height: 36px !important;
                height: 36px !important;
                background: {cfg["input_bg"]} !important;
                overflow: hidden !important;
                border: none !important;
                box-shadow: none !important;
            }}

            [data-testid="stNumberInput"] div[data-baseweb="input"] > div {{
                height: 36px !important;
                min-height: 36px !important;
                background: {cfg["input_bg"]} !important;
                border: none !important;
                box-shadow: none !important;
                overflow: hidden !important;
            }}

            [data-testid="stNumberInput"] input {{
                height: 36px !important;
                min-height: 36px !important;
                background: {cfg["input_bg"]} !important;
                color: {cfg["text"]} !important;
                border: none !important;
                box-shadow: none !important;
                outline: none !important;
                padding-left: 7px !important;
                font-size: 9.8px !important;
            }}

            [data-testid="stNumberInput"] button {{
                height: 36px !important;
                min-height: 36px !important;
                width: 25px !important;
                min-width: 25px !important;
                max-width: 25px !important;
                margin-top: 0 !important;
                padding: 0 !important;
                background: {cfg["input_btn"]} !important;
                border-radius: 0 !important;
                border-top: none !important;
                border-bottom: none !important;
                border-left: 1px solid {cfg["border"]} !important;
                border-right: none !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                box-sizing: border-box !important;
                overflow: hidden !important;
            }}

            [data-testid="stNumberInput"] button:last-child {{
                border-top-right-radius: 14px !important;
                border-bottom-right-radius: 14px !important;
            }}

            [data-testid="stNumberInput"] button svg {{
                width: 12px !important;
                height: 12px !important;
            }}

            /* compact mobile chart-table spacing: rapat, tapi tidak sampai nabrak */
            div[data-testid="stVerticalBlock"] {{
                gap: 0.18rem !important;
            }}

            .stPlotlyChart {{
                margin-bottom: 4px !important;
                padding-bottom: 0 !important;
            }}

            div[data-testid="stElementContainer"]:has(.stPlotlyChart),
            div[data-testid="element-container"]:has(.stPlotlyChart) {{
                margin-bottom: 4px !important;
                padding-bottom: 0 !important;
            }}

            div[data-testid="stElementContainer"]:has(.stPlotlyChart) + div[data-testid="stElementContainer"],
            div[data-testid="element-container"]:has(.stPlotlyChart) + div[data-testid="element-container"] {{
                margin-top: 2px !important;
                padding-top: 0 !important;
            }}

            .custom-table-wrapper {{
                margin-top: 3px !important;
                margin-bottom: 7px !important;
                padding-top: 0 !important;
            }}

            .small-title {{
                margin-top: 8px !important;
                margin-bottom: 4px !important;
                padding-top: 0 !important;
                line-height: 1.12 !important;
            }}

            .desktop-chart {{
                margin-bottom: 0 !important;
                padding-bottom: 0 !important;
            }}

            .table-title-mobile-tight {{
                margin-top: 10px !important;
                margin-bottom: 4px !important;
                padding-top: 0 !important;
                line-height: 1.12 !important;
            }}



        /* FINAL OVERRIDE: native sidebar toggle dibuat seperti tombol web, bukan hamburger */
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="collapsedControl"] {{
            display: block !important;
            visibility: visible !important;
            opacity: 1 !important;
            pointer-events: auto !important;
        }}

        [data-testid="stSidebarCollapsedControl"]:not(:has(button)),
        [data-testid="collapsedControl"]:not(:has(button)),
        [data-testid="stSidebarCollapsedControl"] button,
        [data-testid="collapsedControl"] button,
        button[title="Open sidebar"],
        button[aria-label="Open sidebar"],
        button[title*="sidebar" i],
        button[aria-label*="sidebar" i],
        button[kind="headerNoPadding"],
        button[data-testid="baseButton-headerNoPadding"],
        button[data-testid="stBaseButton-headerNoPadding"],
        button[data-testid*="header" i] {{
            position: fixed !important;
            top: 20px !important;
            left: 22px !important;
            width: 58px !important;
            height: 46px !important;
            min-width: 58px !important;
            min-height: 46px !important;
            max-width: 58px !important;
            max-height: 46px !important;
            border-radius: 16px !important;
            border: 1px solid rgba(139, 203, 136, 0.44) !important;
            background: rgba(18, 30, 23, 0.88) !important;
            box-shadow: 0 12px 30px rgba(0,0,0,.22) !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            padding: 0 !important;
            margin: 0 !important;
            color: transparent !important;
            cursor: pointer !important;
            overflow: hidden !important;
            z-index: 2147483646 !important;
            transition: background .16s ease, border-color .16s ease, transform .16s ease, box-shadow .16s ease !important;
        }}

        [data-testid="stSidebarCollapsedControl"]:not(:has(button)):hover,
        [data-testid="collapsedControl"]:not(:has(button)):hover,
        [data-testid="stSidebarCollapsedControl"] button:hover,
        [data-testid="collapsedControl"] button:hover,
        button[title="Open sidebar"]:hover,
        button[aria-label="Open sidebar"]:hover,
        button[title*="sidebar" i]:hover,
        button[aria-label*="sidebar" i]:hover,
        button[kind="headerNoPadding"]:hover,
        button[data-testid="baseButton-headerNoPadding"]:hover,
        button[data-testid="stBaseButton-headerNoPadding"]:hover,
        button[data-testid*="header" i]:hover {{
            background: rgba(47, 125, 82, 0.96) !important;
            border-color: rgba(139, 203, 136, 0.86) !important;
            transform: translateY(-1px) !important;
        }}

        [data-testid="stSidebarCollapsedControl"]:not(:has(button)) > *,
        [data-testid="collapsedControl"]:not(:has(button)) > *,
        [data-testid="stSidebarCollapsedControl"] button > *,
        [data-testid="collapsedControl"] button > *,
        button[title="Open sidebar"] > *,
        button[aria-label="Open sidebar"] > *,
        button[title*="sidebar" i] > *,
        button[aria-label*="sidebar" i] > *,
        button[kind="headerNoPadding"] > *,
        button[data-testid="baseButton-headerNoPadding"] > *,
        button[data-testid="stBaseButton-headerNoPadding"] > *,
        button[data-testid*="header" i] > * {{
            display: none !important;
            opacity: 0 !important;
            visibility: hidden !important;
        }}

        [data-testid="stSidebarCollapsedControl"]:not(:has(button))::before,
        [data-testid="collapsedControl"]:not(:has(button))::before,
        [data-testid="stSidebarCollapsedControl"] button::before,
        [data-testid="collapsedControl"] button::before,
        button[title="Open sidebar"]::before,
        button[aria-label="Open sidebar"]::before,
        button[title*="sidebar" i]::before,
        button[aria-label*="sidebar" i]::before,
        button[kind="headerNoPadding"]::before,
        button[data-testid="baseButton-headerNoPadding"]::before,
        button[data-testid="stBaseButton-headerNoPadding"]::before,
        button[data-testid*="header" i]::before {{
            content: ">>";
            color: #F5F7F2 !important;
            font-size: 20px !important;
            font-weight: 650 !important;
            letter-spacing: -2px !important;
            line-height: 1 !important;
            transform: translateX(-1px) !important;
        }}

        button[title="Close sidebar"],
        button[aria-label="Close sidebar"] {{
            position: fixed !important;
            top: 22px !important;
            left: 246px !important;
            width: 44px !important;
            height: 40px !important;
            min-width: 44px !important;
            min-height: 40px !important;
            max-width: 44px !important;
            max-height: 40px !important;
            border-radius: 14px !important;
            border: 1px solid rgba(139, 203, 136, 0.36) !important;
            background: rgba(18, 30, 23, 0.55) !important;
            box-shadow: 0 10px 24px rgba(0,0,0,.18) !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            padding: 0 !important;
            margin: 0 !important;
            color: transparent !important;
            cursor: pointer !important;
            overflow: hidden !important;
            z-index: 2147483646 !important;
        }}

        button[title="Close sidebar"]:hover,
        button[aria-label="Close sidebar"]:hover {{
            background: rgba(47, 125, 82, 0.92) !important;
            border-color: rgba(139, 203, 136, 0.82) !important;
            transform: translateY(-1px) !important;
        }}

        button[title="Close sidebar"] > *,
        button[aria-label="Close sidebar"] > * {{
            display: none !important;
            opacity: 0 !important;
            visibility: hidden !important;
        }}

        button[title="Close sidebar"]::before,
        button[aria-label="Close sidebar"]::before {{
            content: "<<";
            color: #F5F7F2 !important;
            font-size: 20px !important;
            font-weight: 650 !important;
            letter-spacing: -2px !important;
            line-height: 1 !important;
            transform: translateX(-1px) !important;
        }}


        body.sidebar-is-open button[data-testid*="header" i],
        body.sidebar-is-open button[kind="headerNoPadding"],
        body.sidebar-is-open button[data-testid="baseButton-headerNoPadding"],
        body.sidebar-is-open button[data-testid="stBaseButton-headerNoPadding"] {{
            top: 22px !important;
            left: 246px !important;
            width: 44px !important;
            height: 40px !important;
            min-width: 44px !important;
            min-height: 40px !important;
            max-width: 44px !important;
            max-height: 40px !important;
            border-radius: 14px !important;
            background: rgba(18, 30, 23, 0.55) !important;
            border-color: rgba(139, 203, 136, 0.36) !important;
        }}

        body.sidebar-is-open button[data-testid*="header" i]::before,
        body.sidebar-is-open button[kind="headerNoPadding"]::before,
        body.sidebar-is-open button[data-testid="baseButton-headerNoPadding"]::before,
        body.sidebar-is-open button[data-testid="stBaseButton-headerNoPadding"]::before {{
            content: "<<" !important;
        }}

        body.sidebar-is-closed [data-testid="stAppViewContainer"],
        body.sidebar-is-closed [data-testid="stMain"],
        body.sidebar-is-closed section.main,
        body.sidebar-is-closed .main {{
            margin-left: 0 !important;
            padding-left: 0 !important;
            width: 100vw !important;
        }}

        body.sidebar-is-closed .block-container,
        body.sidebar-is-closed [data-testid="stMainBlockContainer"] {{
            max-width: min(1640px, calc(100vw - 96px)) !important;
            margin-left: auto !important;
            margin-right: auto !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }}

        @media screen and (max-width: 900px) {{
            [data-testid="stSidebarCollapsedControl"] button,
            [data-testid="collapsedControl"] button,
            button[title="Open sidebar"],
            button[aria-label="Open sidebar"],
            button[kind="headerNoPadding"],
            button[data-testid="baseButton-headerNoPadding"],
            button[data-testid="stBaseButton-headerNoPadding"] {{
                top: 11px !important;
                left: 11px !important;
                width: 48px !important;
                height: 42px !important;
                min-width: 48px !important;
                min-height: 42px !important;
                border-radius: 14px !important;
            }}

            button[title="Close sidebar"],
            button[aria-label="Close sidebar"] {{
                top: 13px !important;
                left: 230px !important;
                width: 42px !important;
                height: 38px !important;
            }}
        }}
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    return cfg


st.session_state.theme_mode = "Gelap"
theme = apply_theme("Gelap")


# ============================================================
# OVERVIEW CARD BACKGROUND FIX — v61
# Menyamakan background card Ringkasan Data & Model dengan KPI card dark theme.
# ============================================================

st.markdown(
    """
    <style>
    .overview-mini-card,
    .overview-panel {
        background:
            linear-gradient(145deg, rgba(255,255,255,0.035), rgba(255,255,255,0.000)),
            #222A24 !important;
        border-color: #3D4A40 !important;
        box-shadow: 0 10px 26px rgba(0,0,0,0.14) !important;
    }

    .overview-mini-card::before,
    .overview-panel::before {
        background:
            radial-gradient(circle at 16% 8%, rgba(139, 203, 136, 0.12), transparent 34%),
            radial-gradient(circle at 95% 12%, rgba(226, 177, 93, 0.10), transparent 32%) !important;
        pointer-events: none !important;
    }

    .overview-mini-card::after {
        display: none !important;
    }

    .overview-mini-card > *,
    .overview-panel > * {
        position: relative !important;
        z-index: 2 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)




# ============================================================
# HERO VISUAL GAP FIX — v60
# Target: hero naik seperti referensi, tetapi section bawah tetap ikut rapat.
# Penyebab gap besar adalah elemen CSS/komponen sebelum hero; jadi yang digeser
# adalah container hero-nya, bukan isi .hero saja.
# ============================================================

st.markdown(
    """
    <style>
    /* Kurangi padding bawaan halaman */
    .block-container,
    [data-testid="stMainBlockContainer"],
    section.main > div.block-container,
    [data-testid="stAppViewContainer"] .main .block-container {
        padding-top: 0.65rem !important;
    }

    /* Ini kunci: parent container hero dinaikkan cukup jauh.
       Kalau masih terlalu turun, ubah -345px jadi -365px.
       Kalau terlalu naik/kepotong, ubah jadi -320px. */
    div[data-testid="stElementContainer"]:has(.hero) {
        margin-top: -230px !important;
        margin-bottom: 18px !important;
        padding-top: 0px !important;
        padding-bottom: 0px !important;
    }

    div[data-testid="stMarkdownContainer"]:has(.hero) {
        margin-top: 0px !important;
        margin-bottom: 0px !important;
        padding-top: 0px !important;
        padding-bottom: 0px !important;
    }

    .hero {
        margin-top: 0px !important;
        margin-bottom: 0px !important;
    }

    .section-title {
        margin-top: 0px !important;
        margin-bottom: 7px !important;
    }

    .section-desc {
        margin-top: 0px !important;
        margin-bottom: 18px !important;
    }

    @media screen and (max-width: 900px) {
        div[data-testid="stElementContainer"]:has(.hero) {
            margin-top: 0px !important;
            margin-bottom: 14px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)




# ============================================================
# PREMIUM READABLE TABLE DESIGN — v56
# Tabel dibuat lebih terang, rapi, center, compact tapi tetap mudah dibaca.
# ============================================================

st.markdown(
    """
    <style>
    .custom-table-wrapper {
        width: 100% !important;
        overflow-x: auto !important;
        border: 1px solid rgba(139, 203, 136, 0.30) !important;
        border-radius: 22px !important;
        background:
            radial-gradient(circle at 5% 0%, rgba(139, 203, 136, 0.16), transparent 28%),
            radial-gradient(circle at 96% 4%, rgba(226, 177, 93, 0.09), transparent 26%),
            linear-gradient(145deg, rgba(255,255,255,0.045), rgba(255,255,255,0.010)),
            #1A211D !important;
        padding: 10px !important;
        margin: 8px 0 22px 0 !important;
        box-shadow: 0 16px 36px rgba(0,0,0,0.18) !important;
        box-sizing: border-box !important;
    }

    table.custom-table {
        width: 100% !important;
        table-layout: fixed !important;
        border-collapse: separate !important;
        border-spacing: 0 6px !important;
        background: transparent !important;
        color: #F5F7F2 !important;
        font-size: 13px !important;
        line-height: 1.22 !important;
        margin: 0 !important;
        border: none !important;
    }

    table.custom-table thead tr th {
        background:
            linear-gradient(135deg, rgba(55, 93, 64, 0.96), rgba(45, 66, 47, 0.98)) !important;
        color: #F5F7F2 !important;
        font-weight: 950 !important;
        font-size: 12.4px !important;
        letter-spacing: 0.01em !important;
        padding: 11px 10px !important;
        height: 42px !important;
        border: none !important;
        text-align: center !important;
        white-space: normal !important;
        overflow-wrap: normal !important;
        word-break: normal !important;
        vertical-align: middle !important;
        box-shadow: inset 0 -1px 0 rgba(139,203,136,0.18) !important;
    }

    table.custom-table thead tr th:first-child {
        border-top-left-radius: 16px !important;
        border-bottom-left-radius: 16px !important;
    }

    table.custom-table thead tr th:last-child {
        border-top-right-radius: 16px !important;
        border-bottom-right-radius: 16px !important;
    }

    table.custom-table tbody tr td,
    table.custom-table tbody tr th {
        background:
            linear-gradient(145deg, rgba(255,255,255,0.030), rgba(255,255,255,0.008)),
            #243126 !important;
        color: rgba(245,247,242,0.92) !important;
        padding: 10px 10px !important;
        height: 38px !important;
        border: none !important;
        border-top: 1px solid rgba(139,203,136,0.13) !important;
        border-bottom: 1px solid rgba(139,203,136,0.13) !important;
        font-weight: 780 !important;
        text-align: center !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        vertical-align: middle !important;
    }

    table.custom-table tbody tr:nth-child(even) td,
    table.custom-table tbody tr:nth-child(even) th {
        background:
            linear-gradient(145deg, rgba(255,255,255,0.035), rgba(255,255,255,0.010)),
            #202A23 !important;
    }

    table.custom-table tbody tr td:first-child,
    table.custom-table tbody tr th:first-child {
        color: #F5F7F2 !important;
        font-weight: 950 !important;
        border-left: 1px solid rgba(139,203,136,0.15) !important;
        border-top-left-radius: 15px !important;
        border-bottom-left-radius: 15px !important;
    }

    table.custom-table tbody tr td:last-child,
    table.custom-table tbody tr th:last-child {
        border-right: 1px solid rgba(139,203,136,0.15) !important;
        border-top-right-radius: 15px !important;
        border-bottom-right-radius: 15px !important;
    }

    table.custom-table tbody tr:hover td,
    table.custom-table tbody tr:hover th {
        background:
            linear-gradient(135deg, rgba(139,203,136,0.18), rgba(226,177,93,0.08)),
            #27382C !important;
        color: #FFFFFF !important;
        transform: translateY(-1px);
        transition: all 0.14s ease-in-out !important;
    }

    /* Lebar kolom tabel simulasi agar tidak padat dan tetap kebaca */
    table.custom-table thead tr th:nth-child(1) { width: 12.2% !important; }
    table.custom-table thead tr th:nth-child(2) { width: 13.0% !important; }
    table.custom-table thead tr th:nth-child(3) { width: 15.2% !important; }
    table.custom-table thead tr th:nth-child(4) { width: 15.2% !important; }
    table.custom-table thead tr th:nth-child(5) { width: 13.0% !important; }
    table.custom-table thead tr th:nth-child(6) { width: 15.0% !important; }
    table.custom-table thead tr th:nth-child(7) { width: 12.4% !important; }

    /* Tabel evaluasi model cuma 5 kolom: lebih seimbang */
    table.custom-table:has(thead tr th:nth-child(5):last-child) thead tr th,
    table.custom-table:has(thead tr th:nth-child(5):last-child) tbody tr td {
        text-align: center !important;
    }

    table.custom-table:has(thead tr th:nth-child(5):last-child) thead tr th:nth-child(1) { width: 26% !important; }
    table.custom-table:has(thead tr th:nth-child(5):last-child) thead tr th:nth-child(2) { width: 18% !important; }
    table.custom-table:has(thead tr th:nth-child(5):last-child) thead tr th:nth-child(3) { width: 18% !important; }
    table.custom-table:has(thead tr th:nth-child(5):last-child) thead tr th:nth-child(4) { width: 18% !important; }
    table.custom-table:has(thead tr th:nth-child(5):last-child) thead tr th:nth-child(5) { width: 20% !important; }

    .custom-table-wrapper table.custom-table tbody:has(tr:only-child) tr td,
    .custom-table-wrapper table.custom-table tbody:has(tr:only-child) tr th {
        height: 42px !important;
        padding-top: 12px !important;
        padding-bottom: 12px !important;
    }

    @media screen and (max-width: 900px) {
        .custom-table-wrapper {
            padding: 7px !important;
            border-radius: 16px !important;
            margin-bottom: 12px !important;
            overflow-x: auto !important;
        }

        table.custom-table {
            min-width: 620px !important;
            font-size: 10.4px !important;
            border-spacing: 0 5px !important;
        }

        table.custom-table thead tr th,
        table.custom-table tbody tr td,
        table.custom-table tbody tr th {
            font-size: 10px !important;
            padding: 8px 7px !important;
            height: 32px !important;
            white-space: nowrap !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)








# ============================================================
# DATA & MODEL OVERVIEW REDESIGN — v52
# Redesign khusus halaman Ringkasan Data & Model.
# ============================================================

st.markdown(
    """
    <style>
    .overview-wrap {
        margin-top: 4px;
        margin-bottom: 24px;
    }

    .overview-mini-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 14px;
        margin: 8px 0 18px 0;
    }

    .overview-mini-card {
        min-height: 112px;
        border-radius: 20px;
        border: 1px solid #3D4A40;
        background:
            linear-gradient(145deg, rgba(255,255,255,0.035), rgba(255,255,255,0.000)),
            #222A24 !important;
        box-shadow: 0 10px 26px rgba(0,0,0,0.14);
        padding: 15px 17px;
        box-sizing: border-box;
        position: relative;
        overflow: hidden;
    }

    .overview-mini-card::before {
        content: "";
        position: absolute;
        inset: 0;
        border-radius: 20px;
        background:
            radial-gradient(circle at 16% 8%, rgba(139, 203, 136, 0.12), transparent 34%),
            radial-gradient(circle at 95% 12%, rgba(226, 177, 93, 0.10), transparent 32%);
        pointer-events: none;
        z-index: 1;
    }

    .overview-mini-card::after {
        display: none !important;
    }

    .overview-mini-card > * {
        position: relative;
        z-index: 2;
    }

    .overview-mini-top {
        display: flex;
        align-items: center;
        gap: 9px;
        margin-bottom: 11px;
        position: relative;
        z-index: 2;
    }

    .overview-mini-icon {
        width: 31px;
        height: 31px;
        min-width: 31px;
        border-radius: 11px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: #8BCB88 !important;
        background: rgba(139, 203, 136, 0.13);
        border: 1px solid rgba(139, 203, 136, 0.23);
    }

    .overview-mini-icon svg {
        width: 17px;
        height: 17px;
        display: block;
    }

    .overview-mini-label {
        color: #D8E0D4 !important;
        font-size: 12.2px;
        font-weight: 900;
        letter-spacing: -0.1px;
        line-height: 1.15;
    }

    .overview-mini-value {
        color: #F5F7F2 !important;
        font-size: clamp(17px, 1.25vw, 25px);
        font-weight: 950;
        line-height: 1.08;
        letter-spacing: -0.35px;
        position: relative;
        z-index: 2;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .overview-mini-note {
        color: #D8E0D4 !important;
        opacity: 0.86;
        font-size: 10.7px;
        font-weight: 700;
        margin-top: 6px;
        line-height: 1.25;
        position: relative;
        z-index: 2;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .overview-panel-grid {
        display: grid;
        grid-template-columns: 0.92fr 1.08fr;
        gap: 16px;
        margin-bottom: 24px;
        align-items: stretch;
    }

    .overview-panel {
        border-radius: 24px;
        border: 1px solid #3D4A40;
        background:
            linear-gradient(145deg, rgba(255,255,255,0.035), rgba(255,255,255,0.000)),
            #222A24 !important;
        box-shadow: 0 12px 30px rgba(0,0,0,0.14);
        padding: 21px 23px;
        min-height: 318px;
        box-sizing: border-box;
        overflow: hidden;
        position: relative;
    }

    .overview-panel::before {
        content: "";
        position: absolute;
        inset: 0;
        border-radius: 24px;
        background:
            radial-gradient(circle at 16% 8%, rgba(139, 203, 136, 0.12), transparent 34%),
            radial-gradient(circle at 95% 12%, rgba(226, 177, 93, 0.10), transparent 32%);
        pointer-events: none;
    }

    .overview-panel > * {
        position: relative;
        z-index: 2;
    }

    .overview-panel-title {
        display: flex;
        align-items: center;
        gap: 10px;
        color: #F5F7F2 !important;
        font-size: 18px;
        font-weight: 950;
        line-height: 1.1;
        margin-bottom: 16px;
    }

    .overview-panel-title .overview-mini-icon {
        width: 34px;
        height: 34px;
        min-width: 34px;
        border-radius: 12px;
    }

    .overview-profile-table {
        display: grid;
        gap: 8px;
    }

    .overview-profile-row {
        display: grid;
        grid-template-columns: 132px minmax(0, 1fr);
        gap: 12px;
        align-items: center;
        border: 1px solid rgba(61, 74, 64, 0.72);
        background: rgba(21, 26, 23, 0.34);
        border-radius: 14px;
        padding: 9px 12px;
    }

    .overview-profile-key {
        color: #D8E0D4 !important;
        font-size: 12px;
        font-weight: 850;
        opacity: 0.85;
    }

    .overview-profile-value {
        color: #F5F7F2 !important;
        font-size: 12.6px;
        font-weight: 900;
        text-align: right;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .overview-model-highlight {
        border-radius: 18px;
        background:
            linear-gradient(145deg, rgba(255,255,255,0.035), rgba(255,255,255,0.000)),
            rgba(38, 48, 41, 0.82) !important;
        border: 1px solid rgba(139, 203, 136, 0.26);
        padding: 14px 16px;
        margin-bottom: 14px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
    }

    .overview-model-kicker {
        color: #8BCB88 !important;
        font-size: 11.2px;
        font-weight: 950;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 7px;
    }

    .overview-model-name {
        color: #F5F7F2 !important;
        font-size: clamp(19px, 1.35vw, 28px);
        font-weight: 950;
        line-height: 1.12;
        letter-spacing: -0.35px;
        word-break: break-word;
    }

    .overview-model-caption {
        color: #D8E0D4 !important;
        font-size: 12.2px;
        font-weight: 720;
        line-height: 1.45;
        margin-top: 8px;
    }

    .overview-step-list {
        display: grid;
        gap: 12px;
        margin-top: 22px;
    }
    
    .overview-step {
        display: grid;
        grid-template-columns: 34px minmax(0, 1fr);
        gap: 10px;
        align-items: start;
    }
    
    .overview-step-number {
        width: 26px;
        height: 26px;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: #151A17 !important;
        background: #8BCB88;
        font-size: 11.5px;
        font-weight: 950;
        margin-top: 4px;
        box-shadow: 0 8px 18px rgba(139,203,136,0.18);
    }
    
    .overview-step-text {
        color: #D8E0D4 !important;
        font-size: 13px;
        font-weight: 720;
        line-height: 1.45;
        margin-top: 4px !important;
    }

    .overview-step-text b {
        color: #F5F7F2 !important;
        font-weight: 950;
    }

    @media screen and (max-width: 900px) {
        .overview-mini-grid {
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }

        .overview-mini-card {
            min-height: 92px;
            padding: 12px 12px;
            border-radius: 16px;
        }

        .overview-mini-value {
            font-size: 14px;
        }

        .overview-panel-grid {
            grid-template-columns: 1fr;
            gap: 10px;
        }

        .overview-panel {
            min-height: auto;
            border-radius: 18px;
            padding: 15px 14px;
        }

        .overview-profile-row {
            grid-template-columns: 1fr;
            gap: 3px;
        }

        .overview-profile-value {
            text-align: left;
            white-space: normal;
        }

        .overview-panel-title {
            font-size: 15px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)




# ============================================================
# DARK ONLY SIDEBAR FIX — v50
# Mode terang/gelap dihapus. Aplikasi selalu memakai mode gelap.
# ============================================================

st.markdown(
    """
    <style>
    /* Karena pilihan mode dihapus, konten sidebar dinaikkan secukupnya */
    [data-testid="stSidebarUserContent"] {
        padding-top: 0rem !important;
        margin-top: -2.35rem !important;
    }

    /* Kalau masih ada style lama untuk theme button, jangan kasih ruang kosong */
    [data-testid="stSidebar"] .theme-label {
        display: none !important;
        margin: 0 !important;
        padding: 0 !important;
        height: 0 !important;
    }

    [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.stButton) {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)




# ============================================================
# SIDEBAR TARGET STYLE — v36
# Target: mengikuti referensi gambar sidebar rapi.
# Sidebar dibuat lebih masuk akal: 304px, inner 252px.
# ============================================================

st.markdown(
    """
    <style>
    /* ---------- SIDEBAR FRAME ---------- */
    [data-testid="stSidebar"],
    [data-testid="stSidebarContent"],
    section[data-testid="stSidebar"] {
        width: 304px !important;
        min-width: 304px !important;
        max-width: 304px !important;
        flex: 0 0 304px !important;
        overflow-x: hidden !important;
        overflow-y: hidden !important;
    }

    [data-testid="stSidebarUserContent"] {
        width: 304px !important;
        max-width: 304px !important;
        padding-left: 26px !important;
        padding-right: 26px !important;
        padding-top: 0 !important;
        padding-bottom: 222px !important;
        margin-top: -36px !important;
        overflow-x: hidden !important;
        overflow-y: hidden !important;
        box-sizing: border-box !important;
    }

    [data-testid="stSidebar"] *,
    [data-testid="stSidebar"] *::before,
    [data-testid="stSidebar"] *::after {
        box-sizing: border-box !important;
    }

    [data-testid="stSidebar"] .element-container {
        width: 252px !important;
        max-width: 252px !important;
        margin-left: auto !important;
        margin-right: auto !important;
        margin-bottom: 4px !important;
    }

    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0 !important;
    }

    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
        margin: 0 0 5px 0 !important;
    }

    /* ---------- TOGGLE BUTTON ---------- */
    #custom-sidebar-toggle-v36 {
        left: 258px !important;
        top: 5px !important;
        width: 36px !important;
        height: 32px !important;
        border-radius: 12px !important;
        z-index: 2147483647 !important;
    }

    body.sidebar-custom-closed #custom-sidebar-toggle-v36 {
        left: 2px !important;
    }

    /* ---------- PILIH TAMPILAN ---------- */
    [data-testid="stSidebar"] .theme-label {
        width: 252px !important;
        max-width: 252px !important;
        margin: 0 auto 9px auto !important;
        font-size: 14px !important;
        line-height: 1.15 !important;
        font-weight: 850 !important;
        white-space: nowrap !important;
    }

    [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] {
        width: 252px !important;
        max-width: 252px !important;
        min-width: 252px !important;
        margin-left: auto !important;
        margin-right: auto !important;
        margin-bottom: 22px !important;
        gap: 10px !important;
        display: flex !important;
        flex-wrap: nowrap !important;
        overflow: hidden !important;
    }

    [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] [data-testid="column"] {
        width: 121px !important;
        min-width: 121px !important;
        max-width: 121px !important;
        flex: 0 0 121px !important;
        padding: 0 !important;
    }

    [data-testid="stSidebar"] .stButton > button {
        width: 121px !important;
        max-width: 121px !important;
        height: 42px !important;
        min-height: 42px !important;
        border-radius: 16px !important;
        font-size: 14px !important;
        padding: 0 !important;
    }

    /* ---------- INPUT DATA HEADER ---------- */
    .data-input-title {
        width: 252px !important;
        max-width: 252px !important;
        margin: 0 auto 10px auto !important;
        padding: 0 !important;
        border-radius: 0 !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        gap: 7px !important;
    }

    .modern-section-icon {
        width: 20px !important;
        height: 20px !important;
        min-width: 20px !important;
        border-radius: 7px !important;
    }

    .modern-section-icon svg {
        width: 13px !important;
        height: 13px !important;
    }

    .data-input-title-text {
        font-size: 12.6px !important;
        line-height: 1.12 !important;
        white-space: nowrap !important;
    }

    /* ---------- FILE UPLOADER ---------- */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        width: 252px !important;
        max-width: 252px !important;
        margin: 0 auto 8px auto !important;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploader"] > label {
        width: 252px !important;
        max-width: 252px !important;
        display: block !important;
        padding-right: 28px !important;
        margin: 0 0 8px 0 !important;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploader"] label p {
        font-size: 13.4px !important;
        line-height: 1.20 !important;
        max-width: 210px !important;
        white-space: nowrap !important;
        letter-spacing: 0 !important;
        margin: 0 !important;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stTooltipIcon"] {
        transform: scale(0.78) !important;
        transform-origin: center !important;
    }

    [data-testid="stFileUploader"] section {
        width: 252px !important;
        max-width: 252px !important;
        min-height: 102px !important;
        padding: 12px !important;
        border-radius: 15px !important;
        overflow: hidden !important;
        text-align: left !important;
    }

    [data-testid="stFileUploader"] section button,
    [data-testid="stFileUploader"] button[data-testid="baseButton-secondary"],
    [data-testid="stFileUploader"] button[kind="secondary"] {
        width: 112px !important;
        min-width: 112px !important;
        max-width: 112px !important;
        height: 38px !important;
        min-height: 38px !important;
        border-radius: 10px !important;
        font-size: 13px !important;
        padding: 0 10px !important;
        margin-left: 0 !important;
        margin-right: auto !important;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploader"] section small,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section p,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section span {
        font-size: 11.1px !important;
        line-height: 1.20 !important;
        white-space: nowrap !important;
        text-align: left !important;
    }

    /* ---------- FORMAT TEXT ---------- */
    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] strong,
    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] b {
        font-size: 11.4px !important;
        line-height: 1.18 !important;
        white-space: normal !important;
    }

    .data-input-note {
        width: 252px !important;
        max-width: 252px !important;
        margin: 8px auto 14px auto !important;
        font-size: 11.4px !important;
        line-height: 1.24 !important;
    }

    /* ---------- DATA STATUS ---------- */
    .data-status {
        width: 252px !important;
        max-width: 252px !important;
        padding: 9px 11px !important;
        margin: 0 auto 20px auto !important;
        border-radius: 14px !important;
        font-size: 12.2px !important;
        line-height: 1.22 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }

    .data-status span {
        font-size: 10.2px !important;
        line-height: 1.12 !important;
        white-space: nowrap !important;
    }

    .modern-status-dot {
        width: 15px !important;
        height: 15px !important;
        min-width: 15px !important;
        border-radius: 5px !important;
        margin-right: 7px !important;
        vertical-align: -3px !important;
        aspect-ratio: 1 / 1 !important;
    }

    .modern-status-dot::after {
        width: 5px !important;
        height: 5px !important;
        border-radius: 999px !important;
        left: 5px !important;
        top: 5px !important;
    }

    /* ---------- RADIO MENU ---------- */
    [data-testid="stSidebar"] .stRadio > label {
        width: 252px !important;
        max-width: 252px !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }

    [data-testid="stSidebar"] .stRadio > label p {
        font-size: 13.5px !important;
        line-height: 1.15 !important;
        margin: 0 0 10px 0 !important;
        white-space: nowrap !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] {
        width: 252px !important;
        max-width: 252px !important;
        margin-left: auto !important;
        margin-right: auto !important;
        gap: 7px !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        width: 252px !important;
        max-width: 252px !important;
        height: 32px !important;
        min-height: 32px !important;
        padding: 0 8px !important;
        margin: 0 !important;
        border-radius: 10px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        gap: 9px !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {
        transform: none !important;
        width: 16px !important;
        min-width: 16px !important;
        max-width: 16px !important;
        height: 16px !important;
        min-height: 16px !important;
        max-height: 16px !important;
        aspect-ratio: 1 / 1 !important;
        margin: 0 3px 0 0 !important;
        padding: 0 !important;
        border-radius: 999px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        overflow: visible !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label > div:first-child * {
        width: 16px !important;
        min-width: 16px !important;
        max-width: 16px !important;
        height: 16px !important;
        min-height: 16px !important;
        max-height: 16px !important;
        aspect-ratio: 1 / 1 !important;
        border-radius: 999px !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label > div:last-child,
    [data-testid="stSidebar"] [role="radiogroup"] label [data-testid="stMarkdownContainer"] {
        height: 32px !important;
        display: flex !important;
        align-items: center !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label p {
        font-size: 13px !important;
        line-height: 1 !important;
        margin: 0 !important;
        padding: 0 !important;
        white-space: nowrap !important;
    }

    /* ---------- BOTTOM DASHBOARD CARD ---------- */
    .sidebar-visual {
        position: fixed !important;
        left: 16px !important;
        bottom: 12px !important;
        width: 272px !important;
        max-width: 272px !important;
        min-height: 214px !important;
        max-height: 228px !important;
        border-radius: 20px !important;
        padding: 18px 17px !important;
        margin: 0 !important;
        overflow: hidden !important;
        z-index: 20 !important;
    }

    .sidebar-visual::before {
        width: 75px !important;
        height: 75px !important;
        right: -22px !important;
        top: -24px !important;
    }

    .sidebar-icons {
        gap: 10px !important;
        margin-bottom: 16px !important;
    }

    .sidebar-icons span {
        width: 43px !important;
        height: 43px !important;
        border-radius: 14px !important;
    }

    .sidebar-icons svg {
        width: 22px !important;
        height: 22px !important;
    }

    .sidebar-visual-title {
        font-size: 20px !important;
        line-height: 1.08 !important;
        margin: 0 !important;
        white-space: nowrap !important;
    }

    .sidebar-visual-subtitle {
        font-size: 13px !important;
        line-height: 1.36 !important;
        margin-top: 10px !important;
        max-width: 230px !important;
    }

    .team-name {
        margin-top: 14px !important;
        padding: 9px 10px !important;
        min-height: 38px !important;
        border-radius: 14px !important;
        font-size: 13px !important;
        line-height: 1.12 !important;
        white-space: nowrap !important;
    }

    @media screen and (max-height: 760px) {
        [data-testid="stSidebarUserContent"] {
            padding-bottom: 192px !important;
        }

        .sidebar-visual {
            min-height: 186px !important;
            max-height: 198px !important;
            padding: 12px 14px !important;
        }

        .sidebar-icons span {
            width: 34px !important;
            height: 34px !important;
        }

        .sidebar-icons svg {
            width: 17px !important;
            height: 17px !important;
        }

        .sidebar-visual-title {
            font-size: 16px !important;
        }

        .sidebar-visual-subtitle {
            font-size: 10.5px !important;
            line-height: 1.24 !important;
            margin-top: 7px !important;
        }

        .team-name {
            font-size: 10.5px !important;
            padding: 6px 8px !important;
            margin-top: 8px !important;
            min-height: 30px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

















# ============================================================
# CUSTOM SIDEBAR TOGGLE — STABLE VERSION
# Tidak memakai tombol native Streamlit. Tombol custom selalu terlihat:
# << untuk menutup sidebar, >> untuk membuka kembali.
# ============================================================

st.markdown(
    """
    <style>
    /* Matikan tombol native Streamlit agar tidak numpuk dengan tombol custom. */
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapseButton"] button,
    button[title="Open sidebar"],
    button[aria-label="Open sidebar"],
    button[title="Close sidebar"],
    button[aria-label="Close sidebar"],
    button[title="Collapse sidebar"],
    button[aria-label="Collapse sidebar"],
    button[title="Expand sidebar"],
    button[aria-label="Expand sidebar"],
    button[data-testid="stBaseButton-headerNoPadding"],
    button[data-testid="baseButton-headerNoPadding"],
    button[kind="headerNoPadding"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }

    /* Layout normal saat sidebar terbuka. */
    section[data-testid="stSidebar"],
    [data-testid="stSidebar"] {
        width: 304px !important;
        min-width: 304px !important;
        flex: 0 0 304px !important;
        transform: translateX(0) !important;
        transition: width .22s ease, min-width .22s ease, flex-basis .22s ease, transform .22s ease, opacity .16s ease !important;
        overflow: hidden !important;
        z-index: 999 !important;
    }

    [data-testid="stSidebarContent"] {
        width: 304px !important;
        min-width: 304px !important;
        transition: opacity .12s ease !important;
    }

    [data-testid="stMain"],
    [data-testid="stAppViewContainer"],
    section.main,
    .main {
        overflow-x: hidden !important;
        transition: margin .22s ease, padding .22s ease, width .22s ease !important;
    }

    .block-container,
    [data-testid="stMainBlockContainer"] {
        margin-left: auto !important;
        margin-right: auto !important;
        transition: max-width .22s ease, padding .22s ease, transform .22s ease !important;
    }

    /* Saat sidebar ditutup: sidebar benar-benar tidak mengambil ruang,
       sehingga konten utama bergerak ke tengah dan tidak ketutup. */
    body.sidebar-custom-closed section[data-testid="stSidebar"],
    body.sidebar-custom-closed [data-testid="stSidebar"] {
        width: 0 !important;
        min-width: 0 !important;
        max-width: 0 !important;
        flex: 0 0 0 !important;
        transform: translateX(-330px) !important;
        opacity: 0 !important;
        pointer-events: none !important;
        border-right: 0 !important;
    }

    body.sidebar-custom-closed [data-testid="stSidebarContent"],
    body.sidebar-custom-closed [data-testid="stSidebarUserContent"] {
        opacity: 0 !important;
        width: 0 !important;
        min-width: 0 !important;
        max-width: 0 !important;
        overflow: hidden !important;
        pointer-events: none !important;
    }

    body.sidebar-custom-closed [data-testid="stMain"],
    body.sidebar-custom-closed [data-testid="stAppViewContainer"],
    body.sidebar-custom-closed section.main,
    body.sidebar-custom-closed .main {
        margin-left: 0 !important;
        padding-left: 0 !important;
        width: 100vw !important;
        max-width: 100vw !important;
        transform: none !important;
    }

    body.sidebar-custom-closed .block-container,
    body.sidebar-custom-closed [data-testid="stMainBlockContainer"] {
        width: min(1500px, calc(100vw - 90px)) !important;
        max-width: min(1500px, calc(100vw - 90px)) !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
        transform: translateX(6px) !important;
    }

    /* Tombol custom: ikon panel-web kecil, lebih halus, dan tidak menabrak hero. */
    #custom-sidebar-toggle-v36 {
        position: fixed !important;
        top: 5px !important;
        left: 258px !important;
        width: 36px !important;
        height: 32px !important;
        border-radius: 11px !important;
        border: 1px solid rgba(139,203,136,.28) !important;
        background: rgba(15,27,20,.54) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        color: #F5F7F2 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        z-index: 2147483647 !important;
        box-shadow: 0 10px 22px rgba(0,0,0,.18) !important;
        cursor: pointer !important;
        user-select: none !important;
        padding: 0 !important;
        margin: 0 !important;
        text-align: center !important;
        transition: left .22s ease, background .15s ease, border-color .15s ease, transform .15s ease, opacity .15s ease, box-shadow .15s ease !important;
    }

    #custom-sidebar-toggle-v36 svg {
        width: 21px !important;
        height: 21px !important;
        display: block !important;
        stroke: currentColor !important;
        fill: none !important;
        stroke-width: 2.35 !important;
        stroke-linecap: round !important;
        stroke-linejoin: round !important;
        margin: 0 !important;
    }

    #custom-sidebar-toggle-v36:hover {
        background: rgba(47,125,82,.92) !important;
        border-color: rgba(139,203,136,.76) !important;
        transform: translateY(-1px) !important;
        opacity: 1 !important;
        box-shadow: 0 14px 28px rgba(0,0,0,.26) !important;
    }

    body.sidebar-custom-closed #custom-sidebar-toggle-v36 {
        left: 2px !important;
        top: 5px !important;
        opacity: 1 !important;
    }

    body:not(.sidebar-custom-closed) #custom-sidebar-toggle-v36 {
        opacity: .48 !important;
    }

    body:not(.sidebar-custom-closed) #custom-sidebar-toggle-v36:hover {
        opacity: 1 !important;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
        height: 0 !important;
        min-height: 0 !important;
    }

    @media screen and (max-width: 900px) {
        #custom-sidebar-toggle-v36 {
            left: 258px !important;
            top: 5px !important;
        }
        body.sidebar-custom-closed #custom-sidebar-toggle-v36 {
            left: 2px !important;
            top: 5px !important;
        }
        body.sidebar-custom-closed .block-container,
        body.sidebar-custom-closed [data-testid="stMainBlockContainer"] {
            width: calc(100vw - 28px) !important;
            max-width: calc(100vw - 28px) !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


def inject_custom_sidebar_toggle():
    components.html(
        """
        <script>
        (function() {
            const doc = window.parent.document;
            const STORAGE_KEY = "bandung_sidebar_custom_closed_v36";
            const BTN_ID = "custom-sidebar-toggle-v36";

            function isClosed() {
                return localStorage.getItem(STORAGE_KEY) === "1";
            }

            function panelSvg(direction) {
                const arrow = direction === "right"
                    ? '<path d="M7.2 6.8l5.2 5.2-5.2 5.2"/><path d="M12.2 6.8l5.2 5.2-5.2 5.2"/>'
                    : '<path d="M16.8 6.8L11.6 12l5.2 5.2"/><path d="M11.8 6.8L6.6 12l5.2 5.2"/>';
                return '<svg viewBox="0 0 24 24" aria-hidden="true">' + arrow + '</svg>';
            }

            function setClosed(closed) {
                localStorage.setItem(STORAGE_KEY, closed ? "1" : "0");
                doc.body.classList.toggle("sidebar-custom-closed", closed);
                const btn = doc.getElementById(BTN_ID);
                if (btn) {
                    btn.innerHTML = panelSvg(closed ? "right" : "left");
                    btn.title = closed ? "Buka sidebar" : "Tutup sidebar";
                    btn.setAttribute("aria-label", closed ? "Buka sidebar" : "Tutup sidebar");
                }
            }

            function removeOldButtons() {
                ["custom-sidebar-toggle-v19", "custom-sidebar-toggle-v20", "custom-sidebar-toggle-v21", "custom-sidebar-toggle-v22", "custom-sidebar-toggle-v23", "custom-sidebar-toggle-v30", "custom-sidebar-toggle-v31", "custom-sidebar-toggle-v32", "custom-sidebar-toggle-v33", "custom-sidebar-toggle-v34", "custom-sidebar-toggle-v35", "app-sidebar-toggle-btn",
                 "custom-mobile-sidebar-button", "custom-sidebar-open-button", "custom-sidebar-close-button"].forEach(function(id) {
                    const old = doc.getElementById(id);
                    if (old) old.remove();
                });
            }

            function ensureButton() {
                let btn = doc.getElementById(BTN_ID);
                if (!btn) {
                    btn = doc.createElement("button");
                    btn.id = BTN_ID;
                    btn.type = "button";
                    btn.setAttribute("data-sidebar-toggle-bound", "1");
                    btn.addEventListener("click", function(e) {
                        e.preventDefault();
                        e.stopPropagation();
                        setClosed(!isClosed());
                    }, true);
                    doc.body.appendChild(btn);
                }
                return btn;
            }

            removeOldButtons();
            ensureButton();
            setClosed(isClosed());

            // Event delegation cadangan supaya klik tetap jalan walaupun DOM Streamlit rerender.
            doc.addEventListener("click", function(e) {
                const btn = e.target.closest && e.target.closest("#" + BTN_ID);
                if (!btn) return;
                e.preventDefault();
                e.stopPropagation();
                setClosed(!isClosed());
            }, true);

            // Pastikan tombol tidak hilang setelah rerun Streamlit.
            setInterval(function() {
                removeOldButtons();
                ensureButton();
                setClosed(isClosed());
            }, 700);
        })();
        </script>
        """,
        height=0,
    )


inject_custom_sidebar_toggle()


# UI COMPONENTS
# ============================================================

def clean_kpi_label(label):
    label = str(label)
    emoji_tokens = [
        "🗓️", "♻️", "💰", "📦", "📈", "📉", "🚛", "🚚", "📆", "📋", "📝", "📂", "✅", "ℹ️",
        "🗓", "☀️", "🌙"
    ]
    for token in emoji_tokens:
        label = label.replace(token, "")
    return " ".join(label.split())


def resolve_kpi_icon(label):
    text = clean_kpi_label(label).lower()
    if "periode" in text or "bulan" in text:
        return "calendar"
    if "sampah" in text:
        return "waste"
    if "anggaran" in text or "biaya" in text or "rupiah" in text:
        return "money"
    if "volume" in text or "densitas" in text:
        return "cube"
    if "tertinggi" in text:
        return "trend_up"
    if "terendah" in text:
        return "trend_down"
    if "hari angkut" in text or "maksimum" in text or "maks/hari" in text:
        return "route"
    if "muatan" in text or "truk" in text:
        return "truck"
    return "dashboard"


def kpi_svg(icon_name):
    if icon_name == "calendar":
        return '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="4" y="5" width="16" height="15" rx="3" fill="none" stroke="currentColor" stroke-width="1.9"/><path d="M8 3.5v4M16 3.5v4M4.8 10h14.4" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round"/><path d="M8 14h2.2M13.8 14H16M8 17h2.2" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round"/></svg>'
    if icon_name == "waste":
        return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7.3 7.4 10 3.8c.9-1.2 2.7-1.2 3.6.1l1.1 1.6" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/><path d="M13.8 5.4h3.8l-1.1-3.5" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/><path d="M17.7 10.1 20 14.2c.7 1.3-.2 2.9-1.7 2.9h-2" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/><path d="M16.4 13.9 14.5 17l3.6.3" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/><path d="M11 18.7H6.4c-1.5 0-2.4-1.6-1.7-2.9l1-1.7" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/><path d="M8.7 14.1H5.1l1.5 3.3" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/></svg>'
    if icon_name == "money":
        return '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="4" y="7" width="16" height="11" rx="2.4" fill="none" stroke="currentColor" stroke-width="1.9"/><path d="M7 10.8h2.2M14.8 14.2H17" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round"/><circle cx="12" cy="12.5" r="2.5" fill="none" stroke="currentColor" stroke-width="1.9"/><path d="M7.3 6.8l9.2-2.3c1.2-.3 2.3.4 2.6 1.5l.2.8" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round"/></svg>'
    if icon_name == "cube":
        return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3.5 20 8v8.2l-8 4.3-8-4.3V8l8-4.5Z" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linejoin="round"/><path d="M4.4 8.2 12 12.6l7.6-4.4M12 20.2v-7.6" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/></svg>'
    if icon_name == "trend_up":
        return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4.5 17.5h15" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round"/><path d="M6 15.5l4.2-4.2 3.2 3.1 5.2-6" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round"/><path d="M15.3 8.3h3.4v3.4" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round"/></svg>'
    if icon_name == "trend_down":
        return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4.5 17.5h15" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round"/><path d="M6 8.2l4.2 4.1 3.2-3.1 5.2 6" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round"/><path d="M15.3 15.3h3.4v-3.4" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round"/></svg>'
    if icon_name == "truck":
        return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3.8 8h10.4v8.2H3.8z" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linejoin="round"/><path d="M14.2 10.2h3.6l2.4 2.8v3.2h-6" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/><circle cx="7.2" cy="17" r="1.7" fill="none" stroke="currentColor" stroke-width="1.9"/><circle cx="17.2" cy="17" r="1.7" fill="none" stroke="currentColor" stroke-width="1.9"/><path d="M5.4 11h5.8" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round"/></svg>'
    if icon_name == "route":
        return '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="7" cy="6.5" r="2.2" fill="none" stroke="currentColor" stroke-width="1.9"/><circle cx="17" cy="17.5" r="2.2" fill="none" stroke="currentColor" stroke-width="1.9"/><path d="M9.2 6.5h4.6c2 0 3.2 1 3.2 2.7s-1.2 2.7-3.2 2.7H10c-2 0-3.2 1-3.2 2.7s1.2 2.9 3.2 2.9h4.8" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/></svg>'
    return '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="4" y="5" width="7" height="6" rx="2" fill="none" stroke="currentColor" stroke-width="1.9"/><rect x="13" y="5" width="7" height="6" rx="2" fill="none" stroke="currentColor" stroke-width="1.9"/><rect x="4" y="13" width="7" height="6" rx="2" fill="none" stroke="currentColor" stroke-width="1.9"/><rect x="13" y="13" width="7" height="6" rx="2" fill="none" stroke="currentColor" stroke-width="1.9"/></svg>'


def kpi_card(label, value, note=None):
    clean_label = clean_kpi_label(label)
    icon_html = kpi_svg(resolve_kpi_icon(label))
    note_html = note if note else "&nbsp;"
    value_text = str(value)
    label_text = clean_label.lower()
    extra_classes = []
    if "periode" in label_text:
        extra_classes.append("kpi-value-period")
    if len(value_text) > 15 or value_text.startswith("Rp"):
        extra_classes.append("kpi-value-long")
    value_class = "kpi-value" + (" " + " ".join(extra_classes) if extra_classes else "")

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <div class="kpi-icon">{icon_html}</div>
                <div class="kpi-label">{clean_label}</div>
            </div>
            <div class="{value_class}">{value}</div>
            <div class="kpi-note">{note_html}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def mobile_kpi_summary(items):
    html_items = ""
    for item in items:
        clean_label = clean_kpi_label(item["label"])
        icon_html = kpi_svg(resolve_kpi_icon(item["label"]))
        note_html = f'<div class="mobile-kpi-note">{item.get("note", "")}</div>' if item.get("note") else ""
        html_items += (
            f'<div class="mobile-kpi-item">'
            f'<div class="mobile-kpi-label-row">'
            f'<div class="mobile-kpi-icon">{icon_html}</div>'
            f'<div class="mobile-kpi-label">{clean_label}</div>'
            f'</div>'
            f'<div class="mobile-kpi-value">{item["value"]}</div>'
            f'{note_html}'
            f'</div>'
        )

    st.markdown(
        f"""
        <div class="mobile-kpi-summary">
            <div class="mobile-kpi-summary-title">Ringkasan Simulasi</div>
            <div class="mobile-kpi-grid">
                {html_items}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def bullet_card(title, items):
    html = "".join([f"<li>{item}</li>" for item in items])
    st.markdown(
        f"""
        <div class="info-card">
            <div class="small-title">{title}</div>
            <ul class="text-muted" style="padding-left:20px; margin-bottom:0;">
                {html}
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_table(data):
    html = data.to_html(classes="custom-table", border=0, index=False, escape=False)
    st.markdown(
        f"""
        <div class="custom-table-wrapper">
            {html}
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# CHART
# ============================================================

def make_forecast_chart(ts, forecast, theme):
    hist = ts.tail(24)

    forecast_start = forecast.index.min()
    max_pred_value = forecast.max()
    max_pred_date = forecast.idxmax()

    y_min = min(hist.min(), forecast.min()) * 0.72
    y_max = max(hist.max(), forecast.max()) * 1.12

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist.values,
        mode="lines+markers",
        name="Data historis",
        line=dict(color=theme["chart_hist"], width=3, shape="linear"),
        marker=dict(
            size=7,
            color=theme["chart_hist"],
            line=dict(color=theme["chart_bg"], width=1.5)
        ),
        fill="tozeroy",
        fillcolor=theme["chart_hist_fill"],
        hovertemplate="<b>%{x|%b %Y}</b><br>Historis: %{y:,.0f} ton<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=forecast.index,
        y=forecast.values,
        mode="lines+markers",
        name="Prediksi",
        line=dict(color=theme["chart_pred"], width=3, shape="linear"),
        marker=dict(
            size=7,
            color=theme["chart_pred"],
            line=dict(color=theme["chart_bg"], width=1.5)
        ),
        fill="tozeroy",
        fillcolor=theme["chart_pred_fill"],
        hovertemplate="<b>%{x|%b %Y}</b><br>Prediksi: %{y:,.0f} ton<extra></extra>"
    ))

    fig.add_vline(
        x=forecast_start,
        line_width=1.6,
        line_dash="dash",
        line_color=theme["chart_divider"]
    )

    fig.add_annotation(
        x=forecast_start,
        y=y_max * 0.98,
        text="<b>Mulai prediksi</b>",
        showarrow=False,
        font=dict(size=12, color=theme["chart_font"], family="Arial"),
        bgcolor=theme["annotation_bg"],
        bordercolor=theme["annotation_border"],
        borderwidth=1,
        borderpad=4
    )

    fig.add_annotation(
        x=max_pred_date,
        y=max_pred_value,
        text=f"<b>Prediksi tertinggi</b><br>{format_integer(max_pred_value)} ton",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1.4,
        arrowcolor=theme["chart_divider"],
        ax=55,
        ay=-40,
        font=dict(size=12, color=theme["chart_font"], family="Arial"),
        bgcolor=theme["annotation_bg"],
        bordercolor=theme["annotation_border"],
        borderwidth=1,
        borderpad=4
    )

    fig.update_layout(
        title=dict(
            text="<b>Prediksi Jumlah Sampah Kota Bandung</b>",
            x=0.5,
            xanchor="center",
            font=dict(size=20, color=theme["chart_font"], family="Arial")
        ),
        paper_bgcolor=theme["chart_bg"],
        plot_bgcolor=theme["chart_bg"],
        margin=dict(l=54, r=8, t=86, b=52),
        height=470,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.07,
            xanchor="right",
            x=0.99,
            bgcolor=theme["chart_legend_bg"],
            bordercolor=theme["chart_legend_border"],
            borderwidth=1,
            font=dict(size=12, color=theme["chart_font"], family="Arial")
        )
    )

    fig.update_xaxes(
        title="<b>Periode</b>",
        showgrid=True,
        gridcolor=theme["chart_grid"],
        tickformat="%b %Y",
        tickfont=dict(size=12, color=theme["chart_axis"], family="Arial"),
        title_font=dict(size=14, color=theme["chart_axis"], family="Arial"),
        zeroline=False,
        automargin=True
    )

    fig.update_yaxes(
        title="<b>Jumlah sampah (ton)</b>",
        showgrid=True,
        gridcolor=theme["chart_grid"],
        tickfont=dict(size=12, color=theme["chart_axis"], family="Arial"),
        title_font=dict(size=14, color=theme["chart_axis"], family="Arial"),
        zeroline=False,
        range=[y_min, y_max],
        automargin=True
    )

    return fig




def make_forecast_chart_mobile(ts, forecast, theme):
    hist = ts.tail(15)

    forecast_start = forecast.index.min()
    max_pred_value = forecast.max()
    max_pred_date = forecast.idxmax()

    y_min = min(hist.min(), forecast.min()) * 0.78
    y_max = max(hist.max(), forecast.max()) * 1.08

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist.values,
        mode="lines+markers",
        name="Historis",
        line=dict(color=theme["chart_hist"], width=1.8, shape="linear"),
        marker=dict(size=3.4, color=theme["chart_hist"], line=dict(color=theme["chart_bg"], width=0.7)),
        fill="tozeroy",
        fillcolor=theme["chart_hist_fill"],
        hovertemplate="<b>%{x|%b %Y}</b><br>Historis: %{y:,.0f} ton<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=forecast.index,
        y=forecast.values,
        mode="lines+markers",
        name="Prediksi",
        line=dict(color=theme["chart_pred"], width=1.8, shape="linear"),
        marker=dict(size=3.4, color=theme["chart_pred"], line=dict(color=theme["chart_bg"], width=0.7)),
        fill="tozeroy",
        fillcolor=theme["chart_pred_fill"],
        hovertemplate="<b>%{x|%b %Y}</b><br>Prediksi: %{y:,.0f} ton<extra></extra>"
    ))

    fig.add_vline(
        x=forecast_start,
        line_width=1,
        line_dash="dash",
        line_color=theme["chart_divider"]
    )

    fig.add_annotation(
        x=forecast_start,
        y=y_max * 0.96,
        text="<b>Mulai</b>",
        showarrow=False,
        font=dict(size=7.2, color=theme["chart_font"], family="Arial"),
        bgcolor=theme["annotation_bg"],
        bordercolor=theme["annotation_border"],
        borderwidth=1,
        borderpad=2
    )

    fig.add_annotation(
        x=max_pred_date,
        y=max_pred_value,
        text=f"<b>Tertinggi</b><br>{format_integer(max_pred_value)} ton",
        showarrow=True,
        arrowhead=2,
        arrowsize=0.8,
        arrowwidth=1,
        arrowcolor=theme["chart_divider"],
        ax=22,
        ay=-24,
        font=dict(size=7.2, color=theme["chart_font"], family="Arial"),
        bgcolor=theme["annotation_bg"],
        bordercolor=theme["annotation_border"],
        borderwidth=1,
        borderpad=2
    )

    fig.update_layout(
        title=dict(
            text="<b>Prediksi Sampah</b>",
            x=0.5,
            xanchor="center",
            y=0.96,
            font=dict(size=8.5, color=theme["chart_font"], family="Arial")
        ),
        paper_bgcolor=theme["chart_bg"],
        plot_bgcolor=theme["chart_bg"],
        margin=dict(l=28, r=4, t=30, b=30),
        height=210,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.00,
            xanchor="right",
            x=0.99,
            bgcolor=theme["chart_legend_bg"],
            bordercolor=theme["chart_legend_border"],
            borderwidth=1,
            font=dict(size=7.2, color=theme["chart_font"], family="Arial")
        )
    )

    fig.update_xaxes(
        title="<b>Periode</b>",
        showgrid=True,
        gridcolor=theme["chart_grid"],
        tickformat="%b %Y",
        nticks=4,
        tickfont=dict(size=7.2, color=theme["chart_axis"], family="Arial"),
        title_font=dict(size=8, color=theme["chart_axis"], family="Arial"),
        zeroline=False,
        automargin=True
    )

    fig.update_yaxes(
        title="<b>Ton</b>",
        showgrid=True,
        gridcolor=theme["chart_grid"],
        tickfont=dict(size=7.2, color=theme["chart_axis"], family="Arial"),
        title_font=dict(size=8, color=theme["chart_axis"], family="Arial"),
        zeroline=False,
        range=[y_min, y_max],
        automargin=True
    )

    return fig


def make_eval_chart(actual, predicted, theme):
    y_max = max(actual.max(), predicted.max()) * 1.15
    y_min = min(actual.min(), predicted.min()) * 0.90

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=actual.index,
        y=actual.values,
        mode="lines+markers",
        name="Aktual",
        line=dict(color=theme["chart_hist"], width=3, shape="linear"),
        marker=dict(
            size=7,
            color=theme["chart_hist"],
            line=dict(color=theme["chart_bg"], width=1.4)
        ),
        hovertemplate="<b>%{x|%b %Y}</b><br>Aktual: %{y:,.0f} ton<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=predicted.index,
        y=predicted.values,
        mode="lines+markers",
        name="Prediksi",
        line=dict(color=theme["chart_pred"], width=3, shape="linear"),
        marker=dict(
            size=7,
            color=theme["chart_pred"],
            line=dict(color=theme["chart_bg"], width=1.4)
        ),
        hovertemplate="<b>%{x|%b %Y}</b><br>Prediksi: %{y:,.0f} ton<extra></extra>"
    ))

    fig.update_layout(
        title=dict(
            text="<b>Aktual vs Prediksi Data Uji</b>",
            x=0.5,
            xanchor="center",
            font=dict(size=20, color=theme["chart_font"], family="Arial")
        ),
        paper_bgcolor=theme["chart_bg"],
        plot_bgcolor=theme["chart_bg"],
        margin=dict(l=54, r=8, t=86, b=52),
        height=430,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.07,
            xanchor="right",
            x=0.99,
            bgcolor=theme["chart_legend_bg"],
            bordercolor=theme["chart_legend_border"],
            borderwidth=1,
            font=dict(size=12, color=theme["chart_font"], family="Arial")
        )
    )

    fig.update_xaxes(
        title="<b>Periode</b>",
        showgrid=True,
        gridcolor=theme["chart_grid"],
        tickformat="%b %Y",
        tickfont=dict(size=12, color=theme["chart_axis"], family="Arial"),
        title_font=dict(size=14, color=theme["chart_axis"], family="Arial"),
        zeroline=False,
        automargin=True
    )

    fig.update_yaxes(
        title="<b>Jumlah sampah (ton)</b>",
        showgrid=True,
        gridcolor=theme["chart_grid"],
        tickfont=dict(size=12, color=theme["chart_axis"], family="Arial"),
        title_font=dict(size=14, color=theme["chart_axis"], family="Arial"),
        zeroline=False,
        range=[y_min, y_max],
        automargin=True
    )

    return fig


# ============================================================
# MODERN EDA VISUALIZATION
# ============================================================

EDA_OPTIONS = [
    "Time Series Plot",
    "Moving Average 12 Bulan",
    "Boxplot per Tahun",
    "Rata-rata per Tahun",
    "Rata-rata per Bulan",
    "Heatmap Tahun-Bulan",
    "Distribusi Jumlah Sampah",
    "Seasonal Decomposition",
]


def build_eda_df(ts):
    eda_df = pd.DataFrame({
        "tanggal": ts.index,
        "jumlah_sampah": ts.values
    })
    eda_df["tahun"] = eda_df["tanggal"].dt.year
    eda_df["bulan_num"] = eda_df["tanggal"].dt.month
    eda_df["bulan"] = eda_df["bulan_num"].map(BULAN_INDO)
    eda_df["periode"] = eda_df["tanggal"].apply(format_periode)
    return eda_df


def hex_to_rgba(hex_color, alpha=0.18):
    """Ubah warna hex menjadi rgba agar grafik tidak terlihat terlalu blok/penuh."""
    if not isinstance(hex_color, str) or not hex_color.startswith("#") or len(hex_color) != 7:
        return f"rgba(139, 203, 136, {alpha})"
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def eda_line_color(theme):
    return theme.get("chart_hist", theme.get("accent", "#67F0C1"))


def eda_fill_color(theme, alpha=0.16):
    base = theme.get("chart_hist", theme.get("accent", "#67F0C1"))
    return hex_to_rgba(base, alpha)


def apply_eda_layout(fig, theme, title, height=430, x_title=None, y_title="Jumlah sampah (ton)"):
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            xanchor="center",
            font=dict(size=20, color=theme["chart_font"], family="Arial")
        ),
        paper_bgcolor=theme["chart_bg"],
        plot_bgcolor=theme["chart_bg"],
        margin=dict(l=54, r=24, t=72, b=52),
        height=height,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="right",
            x=1,
            bgcolor=theme["chart_legend_bg"],
            bordercolor=theme["chart_legend_border"],
            borderwidth=1,
            font=dict(size=12, color=theme["chart_font"], family="Arial")
        ),
        font=dict(color=theme["chart_font"], family="Arial")
    )

    fig.update_xaxes(
        title=f"<b>{x_title}</b>" if x_title else None,
        showgrid=True,
        gridcolor=theme["chart_grid"],
        tickfont=dict(size=12, color=theme["chart_axis"], family="Arial"),
        title_font=dict(size=14, color=theme["chart_axis"], family="Arial"),
        zeroline=False,
        automargin=True
    )

    fig.update_yaxes(
        title=f"<b>{y_title}</b>" if y_title else None,
        showgrid=True,
        gridcolor=theme["chart_grid"],
        tickfont=dict(size=12, color=theme["chart_axis"], family="Arial"),
        title_font=dict(size=14, color=theme["chart_axis"], family="Arial"),
        zeroline=False,
        automargin=True
    )
    return fig


def make_eda_timeseries(ts, theme):
    peak_date = ts.idxmax()
    peak_value = ts.max()
    low_date = ts.idxmin()
    low_value = ts.min()
    line_color = eda_line_color(theme)
    fill_color = eda_fill_color(theme, 0.12)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts.index,
        y=ts.values,
        mode="lines+markers",
        name="Jumlah Sampah",
        line=dict(color=line_color, width=3.0, shape="spline"),
        marker=dict(size=6.4, color=theme["chart_bg"], line=dict(color=line_color, width=1.8)),
        fill="tozeroy",
        fillcolor=fill_color,
        hovertemplate="<b>%{x|%b %Y}</b><br>%{y:,.0f} ton<extra></extra>"
    ))

    for label, date_value, val, ax, ay in [
        ("Tertinggi", peak_date, peak_value, 44, -38),
        ("Terendah", low_date, low_value, -44, 38)
    ]:
        fig.add_annotation(
            x=date_value,
            y=val,
            text=f"<b>{label}</b><br>{format_integer(val)} ton",
            showarrow=True,
            arrowhead=2,
            ax=ax,
            ay=ay,
            font=dict(size=12, color=theme["chart_font"], family="Arial"),
            bgcolor=theme["annotation_bg"],
            bordercolor=theme["annotation_border"],
            borderwidth=1,
            borderpad=4
        )

    fig = apply_eda_layout(fig, theme, "Pola Time Series Jumlah Sampah", x_title="Periode", height=430)
    fig.update_xaxes(tickformat="%Y")
    return fig


def make_eda_moving_average(ts, theme):
    rolling_mean = ts.rolling(window=12).mean()
    line_color = eda_line_color(theme)
    soft_color = eda_fill_color(theme, 0.34)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts.index,
        y=ts.values,
        mode="lines+markers",
        name="Aktual",
        line=dict(color=soft_color, width=2.2),
        marker=dict(size=5, color=theme["chart_bg"], line=dict(color=line_color, width=1.3)),
        hovertemplate="<b>%{x|%b %Y}</b><br>Aktual: %{y:,.0f} ton<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=rolling_mean.index,
        y=rolling_mean.values,
        mode="lines",
        name="Moving Average 12 Bulan",
        line=dict(color=line_color, width=3.8, shape="spline"),
        hovertemplate="<b>%{x|%b %Y}</b><br>MA 12 bulan: %{y:,.0f} ton<extra></extra>"
    ))

    fig = apply_eda_layout(fig, theme, "Moving Average Jumlah Sampah", x_title="Periode", height=430)
    fig.update_xaxes(tickformat="%Y")
    return fig


def make_eda_boxplot_year(ts, theme):
    eda_df = build_eda_df(ts)
    line_color = eda_line_color(theme)
    fill_color = eda_fill_color(theme, 0.18)
    fig = go.Figure()

    for year, group in eda_df.groupby("tahun"):
        fig.add_trace(go.Box(
            y=group["jumlah_sampah"],
            name=str(year),
            boxmean=True,
            fillcolor=fill_color,
            marker=dict(color=line_color, size=5),
            line=dict(color=line_color, width=1.8),
            hovertemplate=f"<b>{year}</b><br>%{{y:,.0f}} ton<extra></extra>"
        ))

    fig = apply_eda_layout(fig, theme, "Sebaran Jumlah Sampah per Tahun", x_title="Tahun", height=430)
    fig.update_layout(showlegend=False)
    return fig


def make_eda_avg_year(ts, theme):
    eda_df = build_eda_df(ts)
    avg_year = eda_df.groupby("tahun", as_index=False)["jumlah_sampah"].mean()
    line_color = eda_line_color(theme)
    fill_color = eda_fill_color(theme, 0.12)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=avg_year["tahun"].astype(str),
        y=avg_year["jumlah_sampah"],
        name="Rata-rata",
        marker=dict(color=fill_color, line=dict(color=line_color, width=2.0)),
        hovertemplate="<b>%{x}</b><br>Rata-rata: %{y:,.0f} ton<extra></extra>",
        text=[format_integer(v) for v in avg_year["jumlah_sampah"]],
        textposition="outside",
        textfont=dict(color=theme["chart_font"], size=12, family="Arial")
    ))

    fig = apply_eda_layout(fig, theme, "Rata-rata Jumlah Sampah per Tahun", x_title="Tahun", height=410)
    fig.update_layout(showlegend=False, bargap=0.38, uniformtext_minsize=9, uniformtext_mode="hide")
    fig.update_yaxes(rangemode="tozero")
    return fig


def make_eda_avg_month(ts, theme):
    eda_df = build_eda_df(ts)
    avg_month = eda_df.groupby(["bulan_num", "bulan"], as_index=False)["jumlah_sampah"].mean()
    avg_month = avg_month.sort_values("bulan_num")
    line_color = eda_line_color(theme)
    fill_color = eda_fill_color(theme, 0.12)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=avg_month["bulan"],
        y=avg_month["jumlah_sampah"],
        name="Rata-rata",
        marker=dict(color=fill_color, line=dict(color=line_color, width=2.0)),
        hovertemplate="<b>%{x}</b><br>Rata-rata: %{y:,.0f} ton<extra></extra>",
        text=[format_integer(v) for v in avg_month["jumlah_sampah"]],
        textposition="outside",
        textfont=dict(color=theme["chart_font"], size=11, family="Arial")
    ))

    fig = apply_eda_layout(fig, theme, "Rata-rata Jumlah Sampah per Bulan", x_title="Bulan", height=420)
    fig.update_layout(showlegend=False, bargap=0.34, uniformtext_minsize=9, uniformtext_mode="hide")
    fig.update_xaxes(tickangle=-35)
    fig.update_yaxes(rangemode="tozero")
    return fig


def make_eda_heatmap(ts, theme):
    eda_df = build_eda_df(ts)
    pivot = eda_df.pivot_table(
        values="jumlah_sampah",
        index="tahun",
        columns="bulan_num",
        aggfunc="mean"
    ).reindex(columns=list(range(1, 13)))

    month_labels = [BULAN_INDO[i] for i in range(1, 13)]
    z_values = pivot.to_numpy(dtype=float)

    line_color = eda_line_color(theme)
    bg_color = theme.get("chart_bg", "#111915")
    card_color = theme.get("card", "#222A24")

    # Heatmap dibuat minimal-valid supaya aman di berbagai versi Plotly/Streamlit.
    # Hindari texttemplate/xgap/ygap/marker.colorbar yang sebelumnya rawan error.
    fig = go.Figure(data=go.Heatmap(
        x=month_labels,
        y=[str(y) for y in pivot.index],
        z=z_values,
        colorscale=[
            [0.00, bg_color],
            [0.50, card_color],
            [1.00, line_color]
        ],
        hoverongaps=False,
        showscale=True,
        colorbar=dict(
            title="Ton",
            tickfont=dict(color=theme["chart_font"], size=11),
            len=0.76,
            thickness=12,
            outlinewidth=0
        ),
        hovertemplate="<b>%{x} %{y}</b><br>Jumlah sampah: %{z:,.0f} ton<extra></extra>"
    ))

    # Angka di dalam cell pakai annotation, bukan texttemplate, agar kompatibel.
    if pivot.size:
        finite_values = z_values[np.isfinite(z_values)]
        threshold = np.nanmedian(finite_values) if finite_values.size else np.nan
        for row_idx, year in enumerate(pivot.index):
            for col_idx, month in enumerate(month_labels):
                value = z_values[row_idx, col_idx]
                if np.isfinite(value):
                    fig.add_annotation(
                        x=month,
                        y=str(year),
                        text=format_integer(value),
                        showarrow=False,
                        font=dict(size=10, color=theme["chart_font"]),
                        opacity=0.92
                    )

    fig = apply_eda_layout(fig, theme, "Heatmap Jumlah Sampah Tahun-Bulan", x_title="Bulan", y_title="Tahun", height=470)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(tickangle=-35, showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def make_eda_distribution(ts, theme):
    line_color = eda_line_color(theme)
    fill_color = eda_fill_color(theme, 0.12)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=ts.values,
        nbinsx=10,
        name="Frekuensi",
        marker=dict(color=fill_color, line=dict(color=line_color, width=2.0)),
        opacity=1,
        hovertemplate="Rentang jumlah sampah: %{x:,.0f} ton<br>Frekuensi: %{y}<extra></extra>"
    ))

    fig = apply_eda_layout(fig, theme, "Distribusi Frekuensi Jumlah Sampah", x_title="Jumlah sampah (ton)", y_title="Frekuensi", height=410)
    fig.update_layout(showlegend=False, bargap=0.08)
    fig.update_yaxes(rangemode="tozero")
    return fig

def make_eda_decomposition(ts, theme):
    decomposition = seasonal_decompose(ts, model="additive", period=12, extrapolate_trend="freq")
    line_color = eda_line_color(theme)
    soft_color = eda_fill_color(theme, 0.36)

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Observed", "Trend", "Seasonal", "Residual")
    )

    components_data = [
        (decomposition.observed, "Observed", line_color, "lines"),
        (decomposition.trend, "Trend", line_color, "lines"),
        (decomposition.seasonal, "Seasonal", soft_color, "lines"),
        (decomposition.resid, "Residual", line_color, "markers"),
    ]

    for row, (series, name, color, mode) in enumerate(components_data, start=1):
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            mode=mode,
            name=name,
            line=dict(color=color, width=2.4),
            marker=dict(size=5, color=theme["chart_bg"], line=dict(color=line_color, width=1.2)),
            hovertemplate=f"<b>%{{x|%b %Y}}</b><br>{name}: %{{y:,.0f}}<extra></extra>"
        ), row=row, col=1)

    fig.update_layout(
        title=dict(
            text="<b>Seasonal Decomposition Jumlah Sampah</b>",
            x=0.5,
            xanchor="center",
            font=dict(size=20, color=theme["chart_font"], family="Arial")
        ),
        paper_bgcolor=theme["chart_bg"],
        plot_bgcolor=theme["chart_bg"],
        margin=dict(l=54, r=24, t=74, b=42),
        height=600,
        hovermode="x unified",
        showlegend=False,
        font=dict(color=theme["chart_font"], family="Arial")
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor=theme["chart_grid"],
        tickformat="%Y",
        tickfont=dict(size=11, color=theme["chart_axis"], family="Arial"),
        zeroline=False,
        automargin=True
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=theme["chart_grid"],
        tickfont=dict(size=11, color=theme["chart_axis"], family="Arial"),
        zeroline=False,
        automargin=True
    )

    for annotation in fig.layout.annotations:
        annotation.font = dict(size=13, color=theme["chart_font"], family="Arial")

    return fig


def eda_metric_card(title, value, note, theme, icon="dashboard"):
    icons = {
        "calendar": '<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="5" width="16" height="15" rx="2"/><path d="M8 3v4M16 3v4M4 10h16"/></svg>',
        "average": '<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 18h16"/><path d="M7 15l4-8 3 6 3-4"/></svg>',
        "up": '<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 17l6-6 4 4 6-8"/><path d="M14 7h6v6"/></svg>',
        "down": '<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 7l6 6 4-4 6 8"/><path d="M14 17h6v-6"/></svg>',
        "dashboard": '<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="7" height="7" rx="1.5"/><rect x="13" y="4" width="7" height="7" rx="1.5"/><rect x="4" y="13" width="7" height="7" rx="1.5"/><rect x="13" y="13" width="7" height="7" rx="1.5"/></svg>',
    }
    icon_svg = icons.get(icon, icons["dashboard"])
    st.markdown(
        f"""
        <div class="eda-metric-card">
            <div class="eda-metric-head">
                <span class="eda-metric-icon">{icon_svg}</span>
                <div class="eda-metric-title">{title}</div>
            </div>
            <div class="eda-metric-value">{value}</div>
            <div class="eda-metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_eda_section(ts, theme):
    st.markdown(
        f"""
        <style>
        .eda-metric-card {{
            background:
                linear-gradient(145deg, rgba(255,255,255,0.030), rgba(255,255,255,0)),
                {theme["card"]};
            border: 1px solid {theme["border"]};
            border-radius: 20px;
            padding: 15px 16px;
            min-height: 98px;
            box-shadow: 0 10px 26px {theme["shadow"]};
            position: relative;
            overflow: hidden;
        }}
        .eda-metric-card::before {{
            content: "";
            position: absolute;
            inset: 0;
            border-radius: 20px;
            background:
                radial-gradient(circle at 12% 10%, {theme["accent_soft"]}, transparent 34%),
                radial-gradient(circle at 95% 8%, rgba(226, 177, 93, 0.08), transparent 30%);
            pointer-events: none;
        }}
        .eda-metric-card > * {{
            position: relative;
            z-index: 2;
        }}
        .eda-metric-head {{
            display: flex;
            align-items: center;
            gap: 9px;
            margin-bottom: 8px;
        }}
        .eda-metric-icon {{
            width: 26px;
            height: 26px;
            min-width: 26px;
            border-radius: 9px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: {theme["accent"]} !important;
            background: {theme["accent_soft"]};
            border: 1px solid {theme["border"]};
        }}
        .eda-metric-icon svg {{
            width: 15px;
            height: 15px;
            display: block;
        }}
        .eda-metric-title {{
            font-size: 12.5px;
            font-weight: 850;
            color: {theme["muted"]} !important;
            margin-bottom: 0;
        }}
        .eda-metric-value {{
            font-size: 21px;
            font-weight: 950;
            color: {theme["text"]} !important;
            line-height: 1.15;
        }}
        .eda-metric-note {{
            font-size: 11.2px;
            font-weight: 650;
            color: {theme["muted"]} !important;
            margin-top: 6px;
        }}
        .eda-select-gap {{
            height: 20px;
        }}

        /* EDA dropdown stabil: klik judul sekali buka, klik lagi tutup */
        div[data-testid="stExpander"] {{
            margin-top: 10px !important;
            margin-bottom: 18px !important;
            border: none !important;
        }}
        div[data-testid="stExpander"] details {{
            border: 1px solid {theme["border"]} !important;
            border-radius: 18px !important;
            background: {theme["card"]} !important;
            overflow: hidden !important;
            box-shadow: none !important;
        }}
        div[data-testid="stExpander"] summary {{
            min-height: 56px !important;
            width: 100% !important;
            box-sizing: border-box !important;
            padding: 0 18px !important;
            font-size: 15.5px !important;
            font-weight: 850 !important;
            color: {theme["text"]} !important;
            display: flex !important;
            align-items: center !important;
            background: linear-gradient(90deg, rgba(139, 203, 136, 0.20) 0%, rgba(139, 203, 136, 0.12) 64%, rgba(139, 203, 136, 0.08) 100%) !important;
            transition: background 0.16s ease-in-out !important;
        }}
        div[data-testid="stExpander"] details[open] > summary {{
            background: linear-gradient(90deg, rgba(139, 203, 136, 0.24) 0%, rgba(139, 203, 136, 0.16) 64%, rgba(139, 203, 136, 0.10) 100%) !important;
        }}
        div[data-testid="stExpander"] summary:hover {{
            background: linear-gradient(90deg, rgba(139, 203, 136, 0.26) 0%, rgba(139, 203, 136, 0.17) 64%, rgba(139, 203, 136, 0.11) 100%) !important;
        }}
        div[data-testid="stExpander"] summary svg {{
            color: {theme["accent"]} !important;
            fill: {theme["accent"]} !important;
            width: 17px !important;
            height: 17px !important;
        }}
        div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {{
            background: #080C12 !important;
            border-top: none !important;
            padding: 8px 0 10px 0 !important;
        }}
        div[data-testid="stExpander"] [role="radiogroup"] {{
            width: 100% !important;
            display: flex !important;
            flex-direction: column !important;
            gap: 6px !important;
            padding: 0 !important;
            margin: 0 !important;
        }}
        div[data-testid="stExpander"] [role="radiogroup"] > div {{
            width: 100% !important;
            min-width: 100% !important;
            box-sizing: border-box !important;
            border-radius: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            background: transparent !important;
            overflow: hidden !important;
        }}
        div[data-testid="stExpander"] [role="radiogroup"] > div:hover {{
            background: rgba(139, 203, 136, 0.08) !important;
        }}
        div[data-testid="stExpander"] [role="radiogroup"] > div:hover label::before {{
            border-color: #FF4B4B !important;
            background: radial-gradient(circle, #FF4B4B 0 42%, rgba(255,255,255,0.96) 43% 64%, transparent 65% 100%) !important;
            box-shadow: 0 0 0 5px rgba(255, 75, 75, 0.16) !important;
            transform: scale(1.08);
        }}
        div[data-testid="stExpander"] [role="radiogroup"] > div:has(input:checked) {{
            background: linear-gradient(90deg, rgba(139, 203, 136, 0.24) 0%, rgba(139, 203, 136, 0.16) 70%, rgba(139, 203, 136, 0.10) 100%) !important;
            border-left: 3px solid rgba(165, 224, 161, 0.88) !important;
        }}
        div[data-testid="stExpander"] [role="radiogroup"] label {{
            width: 100% !important;
            min-width: 100% !important;
            min-height: 50px !important;
            box-sizing: border-box !important;
            border-radius: 0 !important;
            padding: 12px 24px !important;
            margin: 0 !important;
            color: {theme["text"]} !important;
            font-size: 14.2px !important;
            font-weight: 760 !important;
            display: flex !important;
            align-items: center !important;
            gap: 12px !important;
            background: transparent !important;
            cursor: pointer !important;
            transition: background 0.14s ease-in-out !important;
        }}
        /* Hilangkan radio bawaan, ganti dengan bulatan custom yang lebih clean. */
        div[data-testid="stExpander"] [role="radiogroup"] label > div:first-child {{
            display: none !important;
        }}
        div[data-testid="stExpander"] [role="radiogroup"] label::before {{
            content: "";
            width: 14px;
            height: 14px;
            min-width: 14px;
            border-radius: 50%;
            border: 2px solid rgba(232, 239, 228, 0.32);
            background: rgba(255,255,255,0.03);
            box-sizing: border-box;
            transition: all 0.14s ease-in-out;
        }}
        div[data-testid="stExpander"] [role="radiogroup"] > div:has(input:checked) label::before {{
            border-color: {theme["accent"]};
            background: radial-gradient(circle, {theme["accent"]} 0 42%, rgba(255,255,255,0.96) 43% 64%, transparent 65% 100%);
            box-shadow: 0 0 0 3px rgba(139, 203, 136, 0.12);
        }}

        /* Saat mouse menyentuh baris opsi mana pun, bulatan kiri berubah merah. */
        div[data-testid="stExpander"] [role="radiogroup"] > div:hover label::before,
        div[data-testid="stExpander"] [role="radiogroup"] label:hover::before {{
            border-color: #FF4B4B !important;
            background: radial-gradient(circle, #FF4B4B 0 55%, #FF4B4B 56% 100%) !important;
            box-shadow: 0 0 0 5px rgba(255, 75, 75, 0.16) !important;
            transform: scale(1.08) !important;
        }}
        div[data-testid="stExpander"] [role="radiogroup"] p {{
            color: {theme["text"]} !important;
            font-weight: 760 !important;
            line-height: 1.2 !important;
            margin: 0 !important;
        }}
        div[data-testid="stExpander"] [role="radiogroup"] > div:has(input:checked) p {{
            font-weight: 900 !important;
        }}
        @media screen and (max-width: 900px) {{
            .eda-metric-card {{ padding: 11px 12px; min-height: 84px; }}
            .eda-metric-title {{ font-size: 10.5px; }}
            .eda-metric-value {{ font-size: 14.5px; }}
            .eda-metric-note {{ font-size: 9.6px; }}
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    eda_values = ts.dropna()
    metric1, metric2, metric3, metric4 = st.columns(4, gap="small")
    with metric1:
        eda_metric_card("Jumlah Observasi", f"{format_integer(len(eda_values))} bulan", f"{format_periode(eda_values.index.min())} - {format_periode(eda_values.index.max())}", theme, "calendar")
    with metric2:
        eda_metric_card("Rata-rata Bulanan", f"{format_angka(eda_values.mean())} ton", "nilai tengah umum periode data", theme, "average")
    with metric3:
        eda_metric_card("Nilai Tertinggi", f"{format_angka(eda_values.max())} ton", format_periode(eda_values.idxmax()), theme, "up")
    with metric4:
        eda_metric_card("Nilai Terendah", f"{format_angka(eda_values.min())} ton", format_periode(eda_values.idxmin()), theme, "down")

    st.markdown('<div class="eda-select-gap"></div>', unsafe_allow_html=True)
    st.markdown('<div class="small-title">Pilih tampilan EDA</div>', unsafe_allow_html=True)

    if "eda_choice" not in st.session_state or st.session_state.eda_choice not in EDA_OPTIONS:
        st.session_state.eda_choice = EDA_OPTIONS[0]

    current_eda_index = EDA_OPTIONS.index(st.session_state.eda_choice)
    with st.expander(st.session_state.eda_choice, expanded=False):
        selected_eda = st.radio(
            "Pilih visualisasi EDA",
            EDA_OPTIONS,
            index=current_eda_index,
            key="eda_choice_radio",
            label_visibility="collapsed"
        )
        if selected_eda != st.session_state.eda_choice:
            st.session_state.eda_choice = selected_eda
            st.rerun()

    eda_choice = st.session_state.eda_choice

    if eda_choice == "Time Series Plot":
        fig_eda = make_eda_timeseries(ts, theme)
        insight_items = [
            "Grafik ini dipakai untuk melihat perubahan jumlah sampah dari waktu ke waktu.",
            "Puncak dan titik terendah diberi anotasi agar pola ekstrem lebih mudah dibaca.",
            "Visual ini cocok untuk melihat perubahan umum sebelum masuk ke model prediksi."
        ]
    elif eda_choice == "Moving Average 12 Bulan":
        fig_eda = make_eda_moving_average(ts, theme)
        insight_items = [
            "Moving average 12 bulan membantu membaca tren tahunan tanpa terlalu terganggu fluktuasi bulanan.",
            "Garis aktual tetap ditampilkan agar perbedaan antara fluktuasi dan tren tetap terlihat.",
            "Visual ini cocok untuk menjelaskan apakah jumlah sampah cenderung naik, turun, atau stabil."
        ]
    elif eda_choice == "Boxplot per Tahun":
        fig_eda = make_eda_boxplot_year(ts, theme)
        insight_items = [
            "Boxplot membandingkan sebaran jumlah sampah antar tahun.",
            "Visual ini membantu melihat median, variasi, dan kemungkinan nilai ekstrem pada tiap tahun.",
            "Tahun dengan box lebih panjang berarti variasi bulanannya lebih besar."
        ]
    elif eda_choice == "Rata-rata per Tahun":
        fig_eda = make_eda_avg_year(ts, theme)
        insight_items = [
            "Grafik ini merangkum rata-rata jumlah sampah untuk setiap tahun.",
            "Visual ini cocok untuk melihat tahun mana yang secara umum tinggi atau rendah.",
            "Angka di atas bar memudahkan pembacaan tanpa harus membuka tabel tambahan."
        ]
    elif eda_choice == "Rata-rata per Bulan":
        fig_eda = make_eda_avg_month(ts, theme)
        insight_items = [
            "Grafik ini menunjukkan rata-rata jumlah sampah berdasarkan bulan kalender.",
            "Visual ini membantu membaca indikasi pola musiman bulanan.",
            "Bulan dengan rata-rata tinggi dapat menjadi perhatian untuk perencanaan kapasitas operasional."
        ]
    elif eda_choice == "Heatmap Tahun-Bulan":
        fig_eda = make_eda_heatmap(ts, theme)
        insight_items = [
            "Heatmap menunjukkan pola jumlah sampah berdasarkan kombinasi tahun dan bulan.",
            "Warna yang lebih pekat menandakan jumlah sampah yang lebih tinggi pada periode tersebut.",
            "Visual ini membantu menemukan bulan dengan lonjakan, penurunan, atau pola tidak biasa."
        ]
    elif eda_choice == "Distribusi Jumlah Sampah":
        fig_eda = make_eda_distribution(ts, theme)
        insight_items = [
            "Histogram menunjukkan sebaran nilai jumlah sampah dalam seluruh periode data.",
            "Garis vertikal membantu membandingkan posisi rata-rata dan median.",
            "Visual ini berguna untuk melihat apakah data terkonsentrasi pada rentang tertentu atau memiliki nilai ekstrem."
        ]
    else:
        fig_eda = make_eda_decomposition(ts, theme)
        insight_items = [
            "Seasonal decomposition memecah data menjadi observed, trend, seasonal, dan residual.",
            "Bagian trend membantu membaca arah jangka panjang.",
            "Bagian seasonal membantu melihat pola musiman yang berulang setiap 12 bulan."
        ]

    st.plotly_chart(fig_eda, use_container_width=True, config={"displayModeBar": False, "responsive": True})



# ============================================================
# SIDEBAR VISUAL POSITION FIX — v63
# Memastikan card Dashboard Sampah di sidebar tidak geser ke kiri pada desktop.
# ============================================================

st.markdown(
    """
    <style>
    @media screen and (min-width: 901px) {
        [data-testid="stSidebar"] .sidebar-visual {
            position: fixed !important;
            left: 16px !important;
            bottom: 18px !important;
            width: 272px !important;
            max-width: 272px !important;
            min-width: 272px !important;
            transform: none !important;
            margin-left: 0 !important;
            margin-right: 0 !important;
            box-sizing: border-box !important;
            z-index: 25 !important;
        }

        body.sidebar-custom-closed [data-testid="stSidebar"] .sidebar-visual,
        body.sidebar-custom-closed .sidebar-visual {
            display: none !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR KIRI
# ============================================================

# ============================================================
# UPLOAD QUEUE — v70
# File yang dipilih masuk antrian dulu, belum digabung ke data aktif.
# User bisa hapus salah satu file / batalkan semua / gabungkan ke data.
# ============================================================

st.markdown(
    '''
    <style>
    /* Sembunyikan row file bawaan Streamlit agar tidak crop/overflow */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"],
    [data-testid="stSidebar"] [data-testid="stFileUploader"] ul,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] li {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        width: 100% !important;
        max-width: 100% !important;
        overflow: hidden !important;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
        overflow: hidden !important;
    }

    .upload-queue-card,
    .upload-active-card {
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
        margin: 10px 0 12px 0 !important;
        padding: 13px 14px !important;
        border-radius: 18px !important;
        border: 1px solid rgba(139, 203, 136, 0.55) !important;
        background: rgba(34, 42, 36, 0.78) !important;
        box-shadow: 0 8px 20px rgba(0,0,0,0.12) !important;
        text-align: center !important;
        overflow: hidden !important;
    }

    .upload-queue-title,
    .upload-active-title {
        font-size: 13.6px !important;
        line-height: 1.2 !important;
        font-weight: 950 !important;
        color: #F5F7F2 !important;
        margin: 0 0 4px 0 !important;
        text-align: center !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }

    .upload-queue-sub,
    .upload-active-sub {
        font-size: 11.2px !important;
        line-height: 1.35 !important;
        font-weight: 750 !important;
        color: #D8E0D4 !important;
        margin: 0 !important;
        text-align: center !important;
        overflow-wrap: anywhere !important;
    }

    .upload-file-pill {
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        background: rgba(255,255,255,0.055) !important;
        border-radius: 13px !important;
        padding: 8px 9px !important;
        margin: 6px 0 !important;
        overflow: hidden !important;
    }

    .upload-file-name {
        font-size: 11.4px !important;
        line-height: 1.25 !important;
        font-weight: 850 !important;
        color: #F5F7F2 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        text-align: left !important;
        display: block !important;
        max-width: 100% !important;
    }

    .upload-file-meta {
        display: block !important;
        font-size: 9.8px !important;
        line-height: 1.2 !important;
        font-weight: 700 !important;
        color: #D8E0D4 !important;
        text-align: left !important;
        margin-top: 2px !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }

    [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] {
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
    }

    [data-testid="stSidebar"] div[data-testid="column"] {
        min-width: 0 !important;
        box-sizing: border-box !important;
    }

    [data-testid="stSidebar"] .stButton > button {
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    </style>
    ''',
    unsafe_allow_html=True
)

# Upload/antrian dipindahkan ke halaman utama "Kelola Data Upload" agar sidebar tetap rapi.
# Sidebar hanya menampilkan ringkasan data aktif, menu utama, dan kartu dashboard.

try:
    upload_payloads = tuple(
        (item["name"], item["bytes"])
        for item in st.session_state.uploaded_data_payloads
    )
    df_raw, df, ts, source_data_name, source_data_type = load_data(upload_payloads)
except Exception as error:
    st.sidebar.error(f"Data tidak dapat dimuat: {error}")
    st.session_state.uploaded_data_payloads = []
    st.session_state.upload_queue = []
    st.cache_data.clear()
    df_raw, df, ts, source_data_name, source_data_type = load_data()

provinsi = safe_unique_text(df_raw, "nama_provinsi")
kota = safe_unique_text(df_raw, "bps_nama_kabupaten_kota")
satuan = safe_unique_text(df_raw, "satuan", default="Ton")
periode_data = f"{format_periode(ts.index.min())} - {format_periode(ts.index.max())}"

forecast_max_months = DEFAULT_FORECAST_MAX_MONTHS
if source_data_type == "upload":
    tambahan_bulan_data = max(0, len(ts.dropna()) - BASE_HISTORICAL_ROWS)
    forecast_max_months = min(
        UPLOAD_FORECAST_MAX_MONTHS,
        DEFAULT_FORECAST_MAX_MONTHS + tambahan_bulan_data
    )
    forecast_max_months = max(DEFAULT_FORECAST_MAX_MONTHS, forecast_max_months)

# Ringkasan data aktif di sidebar dihapus agar tampilan sidebar lebih bersih.

menu = st.sidebar.radio(
    "Menu Utama",
    MENU_OPTIONS,
    key="active_menu"
)

st.sidebar.markdown(
    """
    <div class="sidebar-visual">
        <div class="sidebar-icons">
            <span><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7 7h10v3l4-5-4-5v3H7a5 5 0 0 0-4.6 3"/><path d="M17 17H7v-3l-4 5 4 5v-3h10a5 5 0 0 0 4.6-3"/></svg></span>
            <span><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 7h16"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M6 7l1 13h10l1-13"/><path d="M9 7V4h6v3"/></svg></span>
            <span><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 4c-7.5 1-12 4.7-13.5 10.8"/><path d="M20 4c.2 7.4-4.5 13.1-11.8 12.8"/><path d="M4 20c2.6-5.9 7.1-9.5 13-11"/></svg></span>
        </div>
        <div class="sidebar-visual-title">dashboard sampah</div>
        <div class="sidebar-visual-subtitle">
            Prediksi jumlah sampah, estimasi anggaran, volume sampah, dan kebutuhan muatan truk compactor.
        </div>
        <div class="team-name">
            Kelompok 5 Capstone
        </div>
    </div>
    """,
    unsafe_allow_html=True
)



# ============================================================
# SIDEBAR FINAL ALIGNMENT FIX — v44
# Final polish: center elements, green radio, remove format note, compact upload spacing.
# ============================================================

st.markdown(
    f"""
    <style>
    :root {{
        --sidebar-inner-w-final: 252px;
        --sidebar-card-w-final: 272px;
    }}

    [data-testid="stSidebarUserContent"] {{
        padding-left: 16px !important;
        padding-right: 16px !important;
        padding-bottom: 214px !important;
        overflow-y: hidden !important;
    }}

    [data-testid="stSidebar"] .element-container {{
        margin-left: auto !important;
        margin-right: auto !important;
    }}

    [data-testid="stSidebar"] .theme-label {{
        width: var(--sidebar-inner-w-final) !important;
        max-width: var(--sidebar-inner-w-final) !important;
        margin: 0 auto 9px auto !important;
        padding: 0 !important;
        line-height: 1.08 !important;
    }}

    [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.stButton) {{
        width: var(--sidebar-inner-w-final) !important;
        max-width: var(--sidebar-inner-w-final) !important;
        min-width: var(--sidebar-inner-w-final) !important;
        height: 42px !important;
        min-height: 42px !important;
        display: grid !important;
        grid-template-columns: 122px 122px !important;
        gap: 8px !important;
        margin: 0 auto 18px auto !important;
        padding: 0 !important;
    }}

    [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.stButton) [data-testid="column"] {{
        width: 122px !important;
        min-width: 122px !important;
        max-width: 122px !important;
        flex: 0 0 122px !important;
        padding: 0 !important;
        margin: 0 !important;
    }}

    [data-testid="stSidebar"] .stButton > button {{
        width: 122px !important;
        min-width: 122px !important;
        max-width: 122px !important;
        height: 42px !important;
        min-height: 42px !important;
        padding: 0 !important;
    }}

    [data-testid="stSidebar"] [data-testid="stFileUploader"] {{
        width: var(--sidebar-inner-w-final) !important;
        max-width: var(--sidebar-inner-w-final) !important;
        margin: 0 auto !important;
    }}

    [data-testid="stSidebar"] [data-testid="stFileUploader"] > label {{
        width: var(--sidebar-inner-w-final) !important;
        max-width: var(--sidebar-inner-w-final) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        gap: 8px !important;
        margin: 0 auto 6px auto !important;
        padding: 0 !important;
    }}

    [data-testid="stSidebar"] [data-testid="stFileUploader"] > label p {{
        margin: 0 !important;
        padding: 0 !important;
        max-width: 222px !important;
        line-height: 1.05 !important;
        font-size: 12.6px !important;
        font-weight: 850 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        display: flex !important;
        align-items: center !important;
    }}

    [data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stTooltipIcon"] {{
        margin: 0 !important;
        padding: 0 !important;
        transform: scale(.78) translateY(0) !important;
        transform-origin: center !important;
        align-self: center !important;
        flex: 0 0 auto !important;
    }}

    [data-testid="stSidebar"] [data-testid="stFileUploader"] section {{
        width: var(--sidebar-inner-w-final) !important;
        max-width: var(--sidebar-inner-w-final) !important;
        min-height: 82px !important;
        padding: 10px 12px !important;
        margin: 0 auto !important;
        border-radius: 15px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
        gap: 7px !important;
    }}

    [data-testid="stSidebar"] [data-testid="stFileUploader"] section button,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button[data-testid="baseButton-secondary"],
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button[kind="secondary"] {{
        width: 100% !important;
        max-width: 100% !important;
        min-width: 100% !important;
        height: 34px !important;
        min-height: 34px !important;
        border-radius: 12px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 8px !important;
        margin: 0 !important;
        padding: 0 10px !important;
        line-height: 1 !important;
        text-align: center !important;
    }}

    [data-testid="stSidebar"] [data-testid="stFileUploader"] section button svg {{
        margin: 0 6px 0 0 !important;
        transform: translateY(0) !important;
        flex: 0 0 auto !important;
    }}

    [data-testid="stSidebar"] [data-testid="stFileUploader"] section small,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section p,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section span {{
        text-align: center !important;
        margin-left: auto !important;
        margin-right: auto !important;
        line-height: 1.08 !important;
    }}

    [data-testid="stSidebar"] .data-input-note {{
        display: none !important;
        height: 0 !important;
        min-height: 0 !important;
        max-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
    }}

    [data-testid="stSidebar"] .data-status {{
        width: var(--sidebar-inner-w-final) !important;
        max-width: var(--sidebar-inner-w-final) !important;
        min-height: 42px !important;
        margin: 10px auto 18px auto !important;
        padding: 8px 10px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        gap: 9px !important;
        border-radius: 13px !important;
        box-sizing: border-box !important;
    }}

    [data-testid="stSidebar"] .modern-status-dot,
    [data-testid="stSidebar"] .data-status-icon {{
        width: 20px !important;
        height: 20px !important;
        min-width: 20px !important;
        max-width: 20px !important;
        min-height: 20px !important;
        max-height: 20px !important;
        margin: 0 !important;
        border-radius: 8px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        flex: 0 0 20px !important;
    }}

    [data-testid="stSidebar"] .data-status span {{
        white-space: nowrap !important;
    }}

    [data-testid="stSidebar"] .stRadio {{
        width: var(--sidebar-inner-w-final) !important;
        max-width: var(--sidebar-inner-w-final) !important;
        margin: 4px auto 0 auto !important;
    }}

    [data-testid="stSidebar"] .stRadio > label {{
        width: var(--sidebar-inner-w-final) !important;
        max-width: var(--sidebar-inner-w-final) !important;
        margin: 0 auto 3px auto !important;
        padding: 0 !important;
    }}

    [data-testid="stSidebar"] .stRadio > label p {{
        margin: 0 !important;
        line-height: 1.05 !important;
    }}

    [data-testid="stSidebar"] [role="radiogroup"] {{
        width: var(--sidebar-inner-w-final) !important;
        max-width: var(--sidebar-inner-w-final) !important;
        margin: 0 auto !important;
        gap: 4px !important;
    }}

    [data-testid="stSidebar"] [role="radiogroup"] label {{
        width: var(--sidebar-inner-w-final) !important;
        max-width: var(--sidebar-inner-w-final) !important;
        height: 32px !important;
        min-height: 32px !important;
        padding: 0 6px !important;
        margin: 0 auto !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        gap: 10px !important;
        box-sizing: border-box !important;
    }}

    [data-testid="stSidebar"] [role="radiogroup"] label p {{
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1 !important;
        white-space: nowrap !important;
    }}

    [data-testid="stSidebar"] input[type="radio"] {{
        appearance: none !important;
        -webkit-appearance: none !important;
        width: 16px !important;
        height: 16px !important;
        min-width: 16px !important;
        max-width: 16px !important;
        min-height: 16px !important;
        max-height: 16px !important;
        border-radius: 999px !important;
        border: 1px solid rgba(139,203,136,.42) !important;
        background: #2B2D3A !important;
        margin: 0 !important;
        padding: 0 !important;
        display: inline-block !important;
        box-shadow: none !important;
        flex: 0 0 16px !important;
    }}

    [data-testid="stSidebar"] input[type="radio"]:checked {{
        background:
            radial-gradient(circle at center, #DFF4DB 0 24%, transparent 26%),
            {theme["accent_hover"]} !important;
        border-color: {theme["accent"]} !important;
        box-shadow: 0 0 0 4px {theme["accent_soft"]} !important;
    }}

    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) > div:first-child,
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) > div:first-child * {{
        background: {theme["accent_hover"]} !important;
        border-color: {theme["accent"]} !important;
        color: {theme["accent"]} !important;
        fill: {theme["accent"]} !important;
    }}

    [data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {{
        width: 16px !important;
        height: 16px !important;
        min-width: 16px !important;
        max-width: 16px !important;
        min-height: 16px !important;
        max-height: 16px !important;
        aspect-ratio: 1 / 1 !important;
        border-radius: 999px !important;
        margin: 0 4px 0 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        flex: 0 0 16px !important;
    }}

    [data-testid="stSidebar"] [role="radiogroup"] label > div:first-child * {{
        width: 16px !important;
        height: 16px !important;
        min-width: 16px !important;
        max-width: 16px !important;
        min-height: 16px !important;
        max-height: 16px !important;
        aspect-ratio: 1 / 1 !important;
        border-radius: 999px !important;
    }}

    .sidebar-visual {{
        left: 50% !important;
        transform: translateX(-50%) !important;
        width: var(--sidebar-card-w-final) !important;
        max-width: var(--sidebar-card-w-final) !important;
        margin-left: 0 !important;
        margin-right: 0 !important;
        box-sizing: border-box !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)




# ============================================================
# SIDEBAR SELECTIVE BALANCE FIX — v46
# Perbaikan dari v45:
# - Card Dashboard Sampah tidak ikut geser.
# - Elemen atas/sidebar hanya digeser sedikit ke kiri.
# - Tombol matahari/bulan diturunkan agar tidak nabrak judul.
# - Upload button dibuat selebar penuh di dalam upload box.
# ============================================================

st.markdown(
    f"""
    <style>
    :root {{
        --sidebar-soft-shift: -10px;
    }}

    /* Jangan geser card dashboard bawah. Card ini sudah pas. */
    .sidebar-visual {{
        left: 50% !important;
        transform: translateX(-50%) !important;
    }}

    /* Geser hanya elemen kontrol atas sedikit ke kiri, bukan card bawah */
    [data-testid="stSidebar"] .theme-label,
    [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.stButton),
    [data-testid="stSidebar"] [data-testid="stFileUploader"],
    [data-testid="stSidebar"] .data-status,
    [data-testid="stSidebar"] .stRadio {{
        transform: translateX(var(--sidebar-soft-shift)) !important;
    }}

    /* Judul Pilih Tampilan diberi napas ke tombol tema */
    [data-testid="stSidebar"] .theme-label {{
        margin-bottom: 13px !important;
        line-height: 1.1 !important;
        position: relative !important;
        z-index: 2 !important;
    }}

    /* Tombol matahari/bulan turun sedikit, tidak nabrak judul */
    [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.stButton) {{
        margin-top: 7px !important;
        margin-bottom: 18px !important;
        position: relative !important;
        z-index: 1 !important;
    }}

    /* Upload label dan ikon ? tetap sejajar, agak dekat dengan box */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] > label {{
        margin-bottom: 7px !important;
        align-items: center !important;
    }}

    [data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stTooltipIcon"] {{
        transform: scale(.78) translateY(0px) !important;
        align-self: center !important;
    }}

    /* Box upload: isi stretch agar tombol bisa selebar penuh */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section {{
        align-items: stretch !important;
        justify-content: center !important;
    }}

    /* Tombol upload selebar penuh, tapi tetap rapi */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section button,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button[data-testid="baseButton-secondary"],
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button[kind="secondary"] {{
        width: 100% !important;
        min-width: 100% !important;
        max-width: 100% !important;
        justify-content: center !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
    }}

    [data-testid="stSidebar"] [data-testid="stFileUploader"] section button svg,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button[data-testid="baseButton-secondary"] svg {{
        margin-right: 8px !important;
        transform: translateY(0px) !important;
    }}

    /* Data status jangan sampai terlalu mepet kanan setelah shift */
    [data-testid="stSidebar"] .data-status {{
        overflow: hidden !important;
    }}

    [data-testid="stSidebar"] .data-status span {{
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)




# ============================================================
# SIDEBAR UPLOAD AND STATUS POLISH — v47
# Fokus:
# - Tombol Upload full-width di dalam dropzone.
# - Icon upload + teks Upload satu baris dan center.
# - Data bawaan aktif dibuat 2 baris agar periode terbaca penuh.
# - Icon data dibuat lebih rapi.
# ============================================================

st.markdown(
    f"""
    <style>
    /* Upload dropzone: isi harus stretch, bukan center button kecil */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section,
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{
        display: flex !important;
        flex-direction: column !important;
        align-items: stretch !important;
        justify-content: center !important;
        gap: 8px !important;
    }}

    /* Semua wrapper di area upload dibuat full width agar button bisa memanjang */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section > div,
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] > div,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section [data-testid="stWidgetLabel"],
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section [data-testid="baseButton-secondary"] {{
        width: 100% !important;
        max-width: 100% !important;
        min-width: 100% !important;
    }}

    /* Tombol Upload selebar bagian dalam dropzone */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section button,
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button[data-testid="baseButton-secondary"],
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button[kind="secondary"] {{
        width: 100% !important;
        min-width: 100% !important;
        max-width: 100% !important;
        height: 34px !important;
        min-height: 34px !important;
        border-radius: 12px !important;
        padding: 0 12px !important;
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 9px !important;
        white-space: nowrap !important;
        line-height: 1 !important;
        text-align: center !important;
    }}

    /* Icon Upload + tulisan satu baris */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section button *,
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button *,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button[data-testid="baseButton-secondary"] * {{
        display: inline-flex !important;
        align-items: center !important;
        vertical-align: middle !important;
        line-height: 1 !important;
        white-space: nowrap !important;
    }}

    [data-testid="stSidebar"] [data-testid="stFileUploader"] section button svg,
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button svg,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button[data-testid="baseButton-secondary"] svg {{
        width: 14px !important;
        height: 14px !important;
        min-width: 14px !important;
        margin: 0 7px 0 0 !important;
        transform: translateY(0) !important;
        flex: 0 0 14px !important;
    }}

    /* Teks format file tetap center di bawah button */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section small,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section p:not(:has(button)),
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section span:not(button span),
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] p,
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] span:not(button span) {{
        text-align: center !important;
        justify-content: center !important;
        line-height: 1.08 !important;
        white-space: nowrap !important;
    }}

    /* Data aktif: 2 baris, tidak kepotong */
    [data-testid="stSidebar"] .data-status {{
        min-height: 54px !important;
        height: auto !important;
        padding: 9px 11px !important;
        gap: 9px !important;
        overflow: visible !important;
        align-items: center !important;
    }}

    [data-testid="stSidebar"] .data-status > div:last-child,
    [data-testid="stSidebar"] .data-status-text {{
        min-width: 0 !important;
        max-width: calc(100% - 31px) !important;
        display: flex !important;
        flex-direction: column !important;
        gap: 3px !important;
        overflow: visible !important;
    }}

    [data-testid="stSidebar"] .data-status b {{
        display: block !important;
        width: 100% !important;
        font-size: 11.8px !important;
        line-height: 1.05 !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }}

    [data-testid="stSidebar"] .data-status span {{
        display: block !important;
        width: 100% !important;
        font-size: 9.6px !important;
        line-height: 1.05 !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }}

    /* Icon data status lebih rapi, bukan kotak kosong besar */
    [data-testid="stSidebar"] .modern-status-dot,
    [data-testid="stSidebar"] .data-status-icon {{
        width: 22px !important;
        height: 22px !important;
        min-width: 22px !important;
        max-width: 22px !important;
        min-height: 22px !important;
        max-height: 22px !important;
        border-radius: 8px !important;
        background:
            radial-gradient(circle at 50% 50%, #8BCB88 0 28%, transparent 30%),
            rgba(139,203,136,.15) !important;
        border: 1px solid rgba(139,203,136,.22) !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.06) !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        flex: 0 0 22px !important;
        color: transparent !important;
        font-size: 0 !important;
        overflow: hidden !important;
    }}

    [data-testid="stSidebar"] .modern-status-dot *,
    [data-testid="stSidebar"] .data-status-icon * {{
        display: none !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)




# ============================================================
# SIDEBAR UPLOAD STATUS FINAL — v48
# Final targeted fix:
# - Upload button full-width sesuai lebar dalam dropzone.
# - Icon upload dan teks Upload selalu satu baris.
# - Data aktif jadi 2 baris dan periode terbaca penuh.
# - Status icon tidak double.
# - Radio menu disejajarkan dan dibuat lebih modern.
# ============================================================

st.markdown(
    """
    <style>
    /* =========================
       Upload box final
       ========================= */

    [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
        align-items: stretch !important;
        justify-content: center !important;
        gap: 8px !important;
        overflow: hidden !important;
    }

    /* Parent button harus full width. Streamlit sering membatasi wrapper-nya, jadi dipaksa di sini. */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section div:has(button),
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section [data-testid="stFileUploaderDropzoneInstructions"],
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] div:has(button),
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] {
        width: 100% !important;
        min-width: 100% !important;
        max-width: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
        margin-left: 0 !important;
        margin-right: 0 !important;
        box-sizing: border-box !important;
    }

    /* Tombol abu-abu Upload memanjang penuh */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section button,
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button[data-testid="baseButton-secondary"],
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button[data-testid="stBaseButton-secondary"],
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button[kind="secondary"] {
        width: 100% !important;
        min-width: 100% !important;
        max-width: 100% !important;
        height: 36px !important;
        min-height: 36px !important;
        border-radius: 12px !important;
        padding: 0 12px !important;
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 8px !important;
        line-height: 1 !important;
        white-space: nowrap !important;
        text-align: center !important;
        box-sizing: border-box !important;
    }

    /* Icon + tulisan Upload satu baris dan vertical-center */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section button *,
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button *,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button[data-testid="baseButton-secondary"] *,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button[data-testid="stBaseButton-secondary"] * {
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        vertical-align: middle !important;
        line-height: 1 !important;
        white-space: nowrap !important;
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploader"] section button svg,
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button svg,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button[data-testid="baseButton-secondary"] svg,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button[data-testid="stBaseButton-secondary"] svg {
        width: 14px !important;
        height: 14px !important;
        min-width: 14px !important;
        max-width: 14px !important;
        margin: 0 8px 0 0 !important;
        transform: translateY(0px) !important;
        flex: 0 0 14px !important;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploader"] section button p,
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button p,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section button span,
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button span {
        font-size: 11px !important;
        font-weight: 700 !important;
        letter-spacing: .01em !important;
        transform: translateY(0px) !important;
    }

    /* Teks format file tetap di tengah */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section small,
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section > div:not(:has(button)),
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] > div:not(:has(button)) {
        width: 100% !important;
        text-align: center !important;
        justify-content: center !important;
    }

    /* =========================
       Data aktif final
       ========================= */

    [data-testid="stSidebar"] .data-status {
        width: 252px !important;
        max-width: 252px !important;
        min-height: 52px !important;
        height: auto !important;
        margin: 10px auto 18px auto !important;
        padding: 8px 10px !important;
        display: grid !important;
        grid-template-columns: 22px minmax(0, 1fr) !important;
        column-gap: 9px !important;
        align-items: center !important;
        overflow: hidden !important;
        box-sizing: border-box !important;
    }

    [data-testid="stSidebar"] .data-status-copy {
        min-width: 0 !important;
        max-width: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        gap: 3px !important;
        overflow: visible !important;
    }

    [data-testid="stSidebar"] .data-status-copy b {
        display: block !important;
        width: 100% !important;
        max-width: 100% !important;
        font-size: 11.2px !important;
        font-weight: 850 !important;
        line-height: 1.05 !important;
        letter-spacing: .01em !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    [data-testid="stSidebar"] .data-status-copy span {
        display: block !important;
        width: 100% !important;
        max-width: 100% !important;
        font-size: 8.65px !important;
        font-weight: 700 !important;
        line-height: 1.05 !important;
        letter-spacing: -0.22px !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Matikan pseudo-dot lama supaya tidak double */
    [data-testid="stSidebar"] .modern-status-dot::after {
        content: none !important;
        display: none !important;
    }

    [data-testid="stSidebar"] .modern-status-dot,
    [data-testid="stSidebar"] .data-status-icon {
        width: 22px !important;
        height: 22px !important;
        min-width: 22px !important;
        max-width: 22px !important;
        min-height: 22px !important;
        max-height: 22px !important;
        margin: 0 !important;
        padding: 0 !important;
        border-radius: 8px !important;
        border: 1px solid rgba(139,203,136,.24) !important;
        background:
            radial-gradient(circle at 50% 50%, #8BCB88 0 26%, transparent 28%),
            linear-gradient(145deg, rgba(139,203,136,.20), rgba(139,203,136,.07)) !important;
        box-shadow:
            inset 0 1px 0 rgba(255,255,255,.09),
            0 6px 12px rgba(0,0,0,.14) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        flex: 0 0 22px !important;
        align-self: center !important;
        color: transparent !important;
        font-size: 0 !important;
        overflow: hidden !important;
    }

    /* =========================
       Menu utama radio final
       ========================= */

    [data-testid="stSidebar"] [role="radiogroup"] label {
        height: 34px !important;
        min-height: 34px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        gap: 10px !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label p {
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1 !important;
        display: flex !important;
        align-items: center !important;
        transform: translateY(-1px) !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label > div:first-child,
    [data-testid="stSidebar"] input[type="radio"] {
        width: 18px !important;
        height: 18px !important;
        min-width: 18px !important;
        max-width: 18px !important;
        min-height: 18px !important;
        max-height: 18px !important;
        border-radius: 999px !important;
        margin: 0 5px 0 0 !important;
        padding: 0 !important;
        flex: 0 0 18px !important;
        align-self: center !important;
        box-sizing: border-box !important;
    }

    [data-testid="stSidebar"] input[type="radio"] {
        appearance: none !important;
        -webkit-appearance: none !important;
        border: 1px solid rgba(139,203,136,.26) !important;
        background:
            linear-gradient(145deg, rgba(43,45,58,.96), rgba(31,34,45,.96)) !important;
        box-shadow:
            inset 0 1px 1px rgba(255,255,255,.06),
            0 4px 10px rgba(0,0,0,.15) !important;
    }

    [data-testid="stSidebar"] input[type="radio"]:checked {
        border-color: rgba(139,203,136,.75) !important;
        background:
            radial-gradient(circle at center, #E7F7E2 0 19%, transparent 21%),
            linear-gradient(145deg, #2E8B57, #257448) !important;
        box-shadow:
            0 0 0 4px rgba(46,139,87,.18),
            0 8px 16px rgba(0,0,0,.20) !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label > div:first-child * {
        border-radius: 999px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)




# ============================================================
# SIDEBAR THEME ICON POSITION FIX — v49
# Turunkan hanya emoji matahari/bulan di dalam tombol tema.
# Posisi dan ukuran tombol tidak diubah.
# ============================================================

st.markdown(
    """
    <style>
    /* Turunkan emoji tema sedikit tanpa menggeser button/elemen luar */
    [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.stButton) .stButton > button p,
    [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.stButton) .stButton > button span {
        transform: translateY(0.3px) !important;
        line-height: 1 !important;
        margin: 0 !important;
        padding: 0 !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.stButton) .stButton > button {
        align-items: center !important;
        justify-content: center !important;
        overflow: hidden !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)




# ============================================================
# SIDEBAR FILE UPLOADER HELP ICON FIX — v51
# Geser ikon ? agar tepat di sebelah teks "Upload data sampah terbaru",
# bukan jauh di kanan.
# ============================================================

st.markdown(
    """
    <style>
    /* Label file uploader: teks dan ikon ? dibuat berdampingan */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] > label,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] label {
        display: inline-flex !important;
        width: auto !important;
        max-width: max-content !important;
        align-items: center !important;
        justify-content: flex-start !important;
        gap: 7px !important;
        margin-bottom: 7px !important;
    }

    /* Teks label jangan mengambil lebar penuh */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] > label p,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] label p,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] > label span,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] label span {
        width: auto !important;
        max-width: max-content !important;
        display: inline-flex !important;
        align-items: center !important;
        margin: 0 !important;
        padding: 0 !important;
        white-space: nowrap !important;
        line-height: 1.1 !important;
    }

    /* Ikon ? ditempel dekat teks, bukan di ujung kanan */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stTooltipIcon"],
    [data-testid="stSidebar"] [data-testid="stFileUploader"] [aria-label="help"],
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button[aria-label="help"] {
        position: static !important;
        margin-left: 2px !important;
        transform: translateY(0px) scale(.78) !important;
        align-self: center !important;
    }

    /* Hindari wrapper label memakai space-between */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] > label > div {
        width: auto !important;
        max-width: max-content !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        gap: 7px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)




# ============================================================
# HALAMAN KELOLA DATA UPLOAD
# ============================================================

def render_upload_management_section(ts, source_data_type, periode_data, forecast_max_months):
    """Halaman Kelola Data Upload.
    Revisi fokus:
    - dropzone hijau upload dilebarkan ke kanan sampai mendekati batas card putih
    - teks 200MB per file ditampilkan penuh tanpa ellipsis
    - card antrian diberi ruang bawah lebih lega tanpa menggeser garis hijau
    - daftar file bawaan Streamlit setelah upload disembunyikan
    """

    active_upload_count = len(st.session_state.uploaded_data_payloads)
    queue_count = len(st.session_state.upload_queue)

    st.markdown(
        """
        <style>
        /* ============================================================
           KELOLA DATA UPLOAD — FINAL PRECISE LAYOUT
           ============================================================ */

        main .block-container {
            padding-top: 0rem !important;
        }

        [data-testid="stMainBlockContainer"] {
            padding-top: 0rem !important;
        }

        /* Tarik hero ke atas agar ruang kosong tidak terlalu lebar */
        div[data-testid="stElementContainer"]:has(.upload-v5-hero) {
            margin-top: -270px !important;
            margin-bottom: 16px !important;
            padding-top: 0 !important;
            padding-bottom: 0 !important;
        }

        .upload-v5-wrap {
            width: 100%;
            max-width: 1320px;
            margin: 0 auto;
        }

        .upload-v5-hero {
            position: relative;
            overflow: hidden;
            border-radius: 28px;
            padding: 34px 36px;
            margin-bottom: 18px;
            border: 1px solid rgba(255,255,255,0.14);
            background: linear-gradient(135deg, #1F4D36 0%, #4F8B59 55%, #B78335 100%);
            box-shadow: 0 18px 42px rgba(31, 41, 51, 0.18);
        }

        .upload-v5-hero::before {
            content: "";
            position: absolute;
            width: 280px;
            height: 280px;
            right: -85px;
            top: -105px;
            border-radius: 50%;
            background: rgba(255,255,255,.12);
        }

        .upload-v5-hero::after {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(90deg, rgba(0,0,0,.06), rgba(255,255,255,.02));
            pointer-events: none;
        }

        .upload-v5-title {
            position: relative;
            z-index: 2;
            color: white !important;
            font-size: clamp(34px, 3.6vw, 52px);
            line-height: 1.08;
            font-weight: 950;
            letter-spacing: -.9px;
            margin: 0 0 12px 0;
        }

        .upload-v5-desc {
            position: relative;
            z-index: 2;
            color: rgba(255,255,255,.92) !important;
            font-size: 15.5px;
            line-height: 1.65;
            font-weight: 650;
            max-width: 1060px;
            margin: 0;
        }

        /* Card putih/luar Streamlit */
        .upload-v5-wrap div[data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid rgba(190,202,191,.42) !important;
            border-radius: 24px !important;
            background:
                linear-gradient(150deg, rgba(255,255,255,.035), rgba(255,255,255,.006)),
                rgba(18,27,22,.96) !important;
            box-shadow: 0 16px 38px rgba(0,0,0,.14) !important;
            overflow: hidden !important;
        }

        .upload-v5-wrap div[data-testid="stVerticalBlockBorderWrapper"] > div {
            padding: 22px !important;
        }

        .upload-v5-card-head {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 14px;
            margin-bottom: 16px;
        }

        .upload-v5-title-row {
            display: flex;
            align-items: center;
            gap: 12px;
            min-width: 0;
        }

        .upload-v5-icon {
            width: 42px;
            height: 42px;
            min-width: 42px;
            border-radius: 15px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, rgba(103,240,193,.15), rgba(241,190,84,.14));
            border: 1px solid rgba(139,203,136,.18);
            color: #DDF4D8 !important;
            font-size: 18px;
            font-weight: 900;
        }

        .upload-v5-card-title {
            color: #F5F7F2 !important;
            font-size: 20px;
            line-height: 1.15;
            font-weight: 950;
            letter-spacing: -.28px;
            margin: 0 0 5px 0;
        }

        .upload-v5-card-subtitle {
            color: #D8E0D4 !important;
            font-size: 12.8px;
            line-height: 1.5;
            font-weight: 650;
            margin: 0;
        }

        .upload-v5-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            white-space: nowrap;
            padding: 8px 12px;
            border-radius: 999px;
            border: 1px solid rgba(139,203,136,.18);
            background: rgba(139,203,136,.08);
            color: #CFE7CC !important;
            font-size: 11.5px;
            font-weight: 850;
        }

        .upload-v5-note {
            color: #D8E0D4 !important;
            font-size: 12.4px;
            line-height: 1.55;
            font-weight: 650;
            margin-top: 13px;
            opacity: .92;
        }

        /* ============================================================
           FILE UPLOADER: hijau dalam card dilebarkan, bukan card luarnya
           ============================================================ */

        [data-testid="stFileUploader"] {
            width: 100% !important;
            max-width: 100% !important;
            margin-top: 0 !important;
            margin-bottom: 0 !important;
        }

        [data-testid="stFileUploader"] > div,
        [data-testid="stFileUploader"] div:has(> section) {
            width: 100% !important;
            max-width: 100% !important;
        }

        [data-testid="stFileUploader"] > label,
        [data-testid="stFileUploader"] label {
            display: none !important;
        }

        [data-testid="stFileUploader"] section,
        [data-testid="stFileUploaderDropzone"] {
            width: 100% !important;
            max-width: 100% !important;
            min-height: 196px !important;
            padding: 26px 30px !important;
            border-radius: 24px !important;
            border: 1.4px dashed rgba(139,203,136,.28) !important;
            background:
                radial-gradient(circle at 10% 12%, rgba(103,240,193,.09), transparent 22%),
                linear-gradient(145deg, rgba(255,255,255,.035), rgba(255,255,255,.012)),
                #1B241E !important;
            box-sizing: border-box !important;
            overflow: visible !important;
            display: flex !important;
            align-items: center !important;
            justify-content: flex-start !important;
            text-align: left !important;
        }

        [data-testid="stFileUploader"] section:hover,
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: rgba(103,240,193,.50) !important;
            background:
                radial-gradient(circle at 10% 12%, rgba(103,240,193,.12), transparent 22%),
                linear-gradient(145deg, rgba(255,255,255,.045), rgba(255,255,255,.018)),
                #1C2720 !important;
        }

        [data-testid="stFileUploader"] section > div,
        [data-testid="stFileUploaderDropzone"] > div {
            width: 100% !important;
            max-width: 100% !important;
            min-width: 0 !important;
            display: grid !important;
            grid-template-columns: 185px minmax(250px, max-content) !important;
            align-items: center !important;
            justify-content: flex-start !important;
            column-gap: 30px !important;
            row-gap: 0 !important;
            overflow: visible !important;
            text-align: left !important;
        }

        [data-testid="stFileUploader"] section button,
        [data-testid="stFileUploaderDropzone"] button,
        [data-testid="stFileUploader"] button[data-testid="baseButton-secondary"],
        [data-testid="stFileUploader"] button[kind="secondary"] {
            width: 185px !important;
            min-width: 185px !important;
            max-width: 185px !important;
            min-height: 58px !important;
            height: 58px !important;
            border-radius: 18px !important;
            border: 1px solid rgba(139,203,136,.28) !important;
            background: linear-gradient(135deg, #26322A 0%, #2D3A30 100%) !important;
            color: #F5F7F2 !important;
            font-size: 16px !important;
            font-weight: 900 !important;
            box-shadow: 0 8px 20px rgba(0,0,0,.14) !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 10px !important;
            overflow: visible !important;
            white-space: nowrap !important;
            grid-column: 1 / 2 !important;
            align-self: center !important;
        }

        [data-testid="stFileUploader"] section button:hover,
        [data-testid="stFileUploaderDropzone"] button:hover,
        [data-testid="stFileUploader"] button[data-testid="baseButton-secondary"]:hover,
        [data-testid="stFileUploader"] button[kind="secondary"]:hover {
            background: linear-gradient(135deg, #2F7D52 0%, #4A8A58 100%) !important;
            color: white !important;
            border-color: #2F7D52 !important;
        }

        /* Teks kanan sejajar vertikal dengan tombol kiri, dan tidak boleh ellipsis */
        [data-testid="stFileUploader"] small {
            grid-column: 2 / 3 !important;
            align-self: center !important;
            min-width: 290px !important;
            width: max-content !important;
            max-width: none !important;
            overflow: visible !important;
            text-overflow: clip !important;
            white-space: nowrap !important;
            color: transparent !important;
            font-size: 0 !important;
            line-height: 1 !important;
            margin: 0 !important;
            padding: 0 !important;
            display: block !important;
        }

        [data-testid="stFileUploader"] small::after {
            content: "200MB per file • XLSX, XLS, CSV";
            display: inline-block !important;
            color: #F5F7F2 !important;
            font-size: 16px !important;
            line-height: 1.2 !important;
            font-weight: 900 !important;
            white-space: nowrap !important;
            overflow: visible !important;
            text-overflow: clip !important;
        }

        [data-testid="stFileUploader"] section p,
        [data-testid="stFileUploader"] section span,
        [data-testid="stFileUploader"] section div,
        [data-testid="stFileUploaderDropzone"] p,
        [data-testid="stFileUploaderDropzone"] span,
        [data-testid="stFileUploaderDropzone"] div {
            overflow: visible !important;
            text-overflow: clip !important;
        }

        /* Setelah file dipilih, card bawaan Streamlit, tombol X, tombol +, dan status proses disembunyikan */
        [data-testid="stFileUploaderFile"],
        [data-testid="stFileUploader"] ul,
        [data-testid="stFileUploader"] li,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"],
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] *,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFileDeleteBtn"],
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"],
        [data-testid="stFileUploader"] [data-testid="stProgress"],
        [data-testid="stFileUploader"] [data-testid="stSpinner"],
        [data-testid="stFileUploader"] [data-testid="stStatusWidget"],
        [data-testid="stFileUploader"] [role="progressbar"],
        [data-testid="stFileUploader"] section ~ div,
        [data-testid="stFileUploader"] section ~ div *,
        [data-testid="stFileUploaderDropzone"] + div,
        [data-testid="stFileUploaderDropzone"] + div *,
        [data-testid="stFileUploader"] button[aria-label*="Remove"],
        [data-testid="stFileUploader"] button[aria-label*="remove"],
        [data-testid="stFileUploader"] button[aria-label*="Delete"],
        [data-testid="stFileUploader"] button[aria-label*="delete"],
        [data-testid="stFileUploader"] button[aria-label*="Clear"],
        [data-testid="stFileUploader"] button[aria-label*="clear"],
        [data-testid="stFileUploader"] button[aria-label*="Add"],
        [data-testid="stFileUploader"] button[aria-label*="add"],
        [data-testid="stFileUploader"] button[title*="Remove"],
        [data-testid="stFileUploader"] button[title*="Delete"],
        [data-testid="stFileUploader"] button[title*="Clear"],
        [data-testid="stFileUploader"] button[title*="Add"] {
            display: none !important;
            visibility: hidden !important;
            opacity: 0 !important;
            height: 0 !important;
            min-height: 0 !important;
            max-height: 0 !important;
            width: 0 !important;
            min-width: 0 !important;
            max-width: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            border: 0 !important;
            overflow: hidden !important;
            pointer-events: none !important;
        }

        /* ============================================================
           EMPTY STATE ANTRIAN
           Garis hijau tetap, garis putih/card luar diturunkan dengan spacer bawah.
           ============================================================ */

        .upload-v5-empty {
            min-height: 196px;
            border: 1.4px dashed rgba(139,203,136,.22);
            border-radius: 22px;
            background:
                radial-gradient(circle at 50% 0%, rgba(139,203,136,.08), transparent 35%),
                rgba(255,255,255,.018);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 34px 22px 38px;
            box-sizing: border-box;
        }

        .upload-v5-empty-icon {
            width: 70px;
            height: 70px;
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, rgba(103,240,193,.13), rgba(241,190,84,.12));
            border: 1px solid rgba(139,203,136,.18);
            margin-bottom: 16px;
        }

        .upload-v5-empty-icon svg {
            width: 34px;
            height: 34px;
            display: block;
        }

        .upload-v5-empty-title {
            color: #F5F7F2 !important;
            font-size: 16px;
            line-height: 1.35;
            font-weight: 900;
            margin-bottom: 8px;
        }

        .upload-v5-empty-desc {
            color: #D8E0D4 !important;
            font-size: 12.4px;
            line-height: 1.6;
            font-weight: 700;
            max-width: 470px;
            margin: 0 auto;
        }

        .upload-v5-right-bottom-spacer {
            height: 18px;
        }

        .upload-v5-file-row {
            border: 1px solid rgba(61,74,64,.85);
            border-radius: 18px;
            padding: 13px 15px;
            background: rgba(255,255,255,.03);
            margin-bottom: 10px;
        }

        .upload-v5-file-title {
            color: #F5F7F2 !important;
            font-size: 13.1px;
            line-height: 1.4;
            font-weight: 850;
            margin-bottom: 4px;
            word-break: break-word;
        }

        .upload-v5-file-meta {
            color: #BFC9BD !important;
            font-size: 11.5px;
            line-height: 1.4;
            font-weight: 700;
        }

        .upload-v5-success {
            display: flex;
            align-items: flex-start;
            gap: 14px;
            padding: 16px 18px;
            margin-top: 16px;
            border-radius: 20px;
            border: 1px solid rgba(103,240,193,.22);
            background: linear-gradient(135deg, rgba(103,240,193,.10), rgba(139,203,136,.06));
            box-shadow: 0 14px 30px rgba(0,0,0,.12);
        }

        .upload-v5-success-icon {
            width: 42px;
            height: 42px;
            min-width: 42px;
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(103,240,193,.14);
            border: 1px solid rgba(103,240,193,.20);
            color: #67F0C1 !important;
            font-size: 22px;
            font-weight: 950;
        }

        .upload-v5-success-title {
            color: #F5F7F2 !important;
            font-size: 15px;
            font-weight: 900;
            margin-bottom: 4px;
        }

        .upload-v5-success-text {
            color: #D8E0D4 !important;
            font-size: 12.8px;
            line-height: 1.55;
            font-weight: 700;
        }

        .upload-v5-wrap .stButton > button {
            width: 100% !important;
            min-height: 46px !important;
            border-radius: 15px !important;
            font-size: 13px !important;
            font-weight: 850 !important;
            border: 1px solid rgba(61,74,64,.95) !important;
            background: linear-gradient(135deg, #212A23 0%, #29332B 100%) !important;
            color: #F5F7F2 !important;
            box-shadow: 0 10px 22px rgba(0,0,0,.14) !important;
        }

        .upload-v5-wrap .stButton > button:hover {
            background: linear-gradient(135deg, #2F7D52 0%, #4A8A58 100%) !important;
            color: white !important;
            border-color: #2F7D52 !important;
        }

        .upload-v5-wrap .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #2F7D52 0%, #4A8A58 58%, #D1A64A 100%) !important;
            border-color: rgba(209,166,74,.58) !important;
            color: white !important;
        }

        .upload-v5-wrap .stButton > button:disabled {
            opacity: .50 !important;
            cursor: not-allowed !important;
        }

        @media screen and (max-width: 960px) {
            div[data-testid="stElementContainer"]:has(.upload-v5-hero) {
                margin-top: 50px !important;
            }

            .upload-v5-hero {
                padding: 20px 18px;
                border-radius: 22px;
            }

            .upload-v5-title {
                font-size: 24px;
            }

            .upload-v5-desc {
                font-size: 13px;
            }

            .upload-v5-wrap div[data-testid="stVerticalBlockBorderWrapper"] > div {
                padding: 16px !important;
            }

            [data-testid="stFileUploader"] section,
            [data-testid="stFileUploaderDropzone"],
            .upload-v5-empty {
                min-height: 165px !important;
            }

            [data-testid="stFileUploader"] section > div,
            [data-testid="stFileUploaderDropzone"] > div {
                grid-template-columns: 1fr !important;
                row-gap: 14px !important;
            }

            [data-testid="stFileUploader"] section button,
            [data-testid="stFileUploaderDropzone"] button,
            [data-testid="stFileUploader"] button[data-testid="baseButton-secondary"],
            [data-testid="stFileUploader"] button[kind="secondary"] {
                width: 100% !important;
                max-width: 100% !important;
            }

            [data-testid="stFileUploader"] small {
                grid-column: 1 / 2 !important;
                min-width: 0 !important;
                width: 100% !important;
            }

            [data-testid="stFileUploader"] small::after {
                white-space: normal !important;
                font-size: 13px !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="upload-v5-wrap">', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="upload-v5-hero">
            <div class="upload-v5-title">Kelola Data Upload</div>
            <div class="upload-v5-desc">
                Tambahkan file baru ke antrian terlebih dahulu. File yang diunggah belum memengaruhi model sampai tombol
                <b>Upload ke Data Aktif</b> diklik.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    left_col, right_col = st.columns([0.95, 1.05], gap="large")

    with left_col:
        with st.container(border=True):
            st.markdown(
                """
                <div class="upload-v5-card-head">
                    <div class="upload-v5-title-row">
                        <div class="upload-v5-icon">⤴</div>
                        <div>
                            <div class="upload-v5-card-title">Upload File ke Antrian</div>
                            <div class="upload-v5-card-subtitle">Pilih file Excel atau CSV. File akan masuk antrian dulu sebelum digabung ke data aktif.</div>
                        </div>
                    </div>
                    <div class="upload-v5-pill">Excel / CSV</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            uploaded_files_main = st.file_uploader(
                "Upload file data sampah",
                type=["xlsx", "xls", "csv"],
                help="File wajib memiliki kolom: tahun, bulan, jumlah_sampah.",
                accept_multiple_files=True,
                key=f"no_empty_upload_panel_{st.session_state.uploader_key}",
                label_visibility="collapsed"
            )

            components.html(
                """
                <script>
                const hideUploadArtifacts = () => {
                    const doc = window.parent.document;
                    const selectors = [
                        '[data-testid="stFileUploaderFile"]',
                        '[data-testid="stFileUploader"] ul',
                        '[data-testid="stFileUploader"] li',
                        '[data-testid="stFileUploader"] [role="progressbar"]',
                        '[data-testid="stFileUploader"] [data-testid="stProgress"]',
                        '[data-testid="stFileUploader"] [data-testid="stSpinner"]',
                        '[data-testid="stFileUploader"] [data-testid="stStatusWidget"]'
                    ];
                    selectors.forEach((selector) => {
                        doc.querySelectorAll(selector).forEach((el) => {
                            el.style.setProperty('display', 'none', 'important');
                            el.style.setProperty('visibility', 'hidden', 'important');
                            el.style.setProperty('height', '0px', 'important');
                            el.style.setProperty('margin', '0px', 'important');
                            el.style.setProperty('padding', '0px', 'important');
                            el.style.setProperty('overflow', 'hidden', 'important');
                        });
                    });
                    doc.querySelectorAll('[data-testid="stFileUploader"] button').forEach((btn) => {
                        const text = (btn.innerText || '').trim();
                        const aria = (btn.getAttribute('aria-label') || '').toLowerCase();
                        const title = (btn.getAttribute('title') || '').toLowerCase();
                        const mustHide = text === '+' || text === '×' || text.toLowerCase() === 'x' ||
                            aria.includes('remove') || aria.includes('delete') || aria.includes('clear') || aria.includes('add') ||
                            title.includes('remove') || title.includes('delete') || title.includes('clear') || title.includes('add');
                        if (mustHide) {
                            btn.style.setProperty('display', 'none', 'important');
                            btn.style.setProperty('visibility', 'hidden', 'important');
                            btn.style.setProperty('pointer-events', 'none', 'important');
                        }
                    });
                };
                hideUploadArtifacts();
                setTimeout(hideUploadArtifacts, 50);
                setTimeout(hideUploadArtifacts, 200);
                setTimeout(hideUploadArtifacts, 600);
                </script>
                """,
                height=0,
                width=0,
            )

    if uploaded_files_main:
        new_errors = []
        existing_keys = {
            (item.get("name"), item.get("size"))
            for item in st.session_state.upload_queue
        }

        for uploaded_file in uploaded_files_main:
            uploaded_bytes = uploaded_file.getvalue()
            uploaded_name = uploaded_file.name
            uploaded_size = len(uploaded_bytes)

            try:
                read_uploaded_dataframe(uploaded_bytes, uploaded_name)
                item_key = (uploaded_name, uploaded_size)
                if item_key not in existing_keys:
                    st.session_state.upload_queue.append({
                        "name": uploaded_name,
                        "bytes": uploaded_bytes,
                        "size": uploaded_size
                    })
                    existing_keys.add(item_key)
            except Exception as error:
                new_errors.append(f"{uploaded_name}: {error}")

        st.session_state.upload_error_messages = new_errors
        st.session_state.upload_success_message = ""
        st.session_state.uploader_key += 1
        st.rerun()

    with right_col:
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="upload-v5-card-head">
                    <div class="upload-v5-title-row">
                        <div class="upload-v5-icon">☷</div>
                        <div>
                            <div class="upload-v5-card-title">Antrian File</div>
                            <div class="upload-v5-card-subtitle">Cek file sebelum digabung ke data aktif.</div>
                        </div>
                    </div>
                    <div class="upload-v5-pill">{queue_count} file</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if queue_count == 0:
                st.markdown(
                    """
                    <div class="upload-v5-empty">
                        <div class="upload-v5-empty-icon">
                            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                                <path d="M5 6.2H11.7L13.5 8H19C20.1 8 21 8.9 21 10V17.4C21 18.5 20.1 19.4 19 19.4H5C3.9 19.4 3 18.5 3 17.4V8.2C3 7.1 3.9 6.2 5 6.2Z" stroke="#F5F7F2" stroke-width="1.7" stroke-linejoin="round"/>
                                <path d="M7.5 12H16.5" stroke="#F5F7F2" stroke-width="1.7" stroke-linecap="round"/>
                                <path d="M7.5 15H14" stroke="#F5F7F2" stroke-width="1.7" stroke-linecap="round"/>
                            </svg>
                        </div>
                        <div class="upload-v5-empty-title">Belum ada file di antrian</div>
                        <div class="upload-v5-empty-desc">
                            Unggah file dari panel kiri. Setelah valid, file akan muncul di sini sebelum digabung ke data aktif.
                        </div>
                    </div>
                    <div class="upload-v5-right-bottom-spacer"></div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                for idx, item in enumerate(list(st.session_state.upload_queue)):
                    qcol1, qcol2 = st.columns([0.80, 0.20], gap="small")
                    with qcol1:
                        st.markdown(
                            f"""
                            <div class="upload-v5-file-row">
                                <div class="upload-v5-file-title">{idx + 1}. {html.escape(item.get('name', 'file upload'))}</div>
                                <div class="upload-v5-file-meta">{format_file_size(item.get('size', 0))} • siap diproses ke data aktif</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with qcol2:
                        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
                        if st.button("Hapus", key=f"upload_remove_{idx}", use_container_width=True):
                            st.session_state.upload_queue.pop(idx)
                            st.session_state.upload_success_message = ""
                            st.session_state.uploader_key += 1
                            st.rerun()

    if st.session_state.upload_error_messages:
        for error_message in st.session_state.upload_error_messages:
            st.error(f"File tidak valid: {error_message}")

    action_col1, action_col2 = st.columns([1.35, 0.75], gap="small")

    with action_col1:
        if st.button("Upload ke Data Aktif", key="upload_commit_queue", type="primary", disabled=queue_count == 0, use_container_width=True):
            jumlah_digabung = len(st.session_state.upload_queue)
            st.session_state.uploaded_data_payloads.extend(st.session_state.upload_queue)
            st.session_state.upload_queue = []
            st.session_state.upload_error_messages = []
            st.session_state.upload_success_message = f"{jumlah_digabung} file berhasil digabungkan ke data aktif."
            st.session_state.uploader_key += 1
            st.cache_data.clear()
            st.rerun()

    with action_col2:
        reset_disabled = active_upload_count == 0 and queue_count == 0
        if st.button("Reset Awal", key="upload_reset_default", disabled=reset_disabled, use_container_width=True):
            st.session_state.uploaded_data_payloads = []
            st.session_state.upload_queue = []
            st.session_state.upload_error_messages = []
            st.session_state.upload_success_message = ""
            st.session_state.uploader_key += 1
            st.cache_data.clear()
            st.rerun()

    if st.session_state.upload_success_message:
        st.markdown(
            f"""
            <div class="upload-v5-success">
                <div class="upload-v5-success-icon">✓</div>
                <div>
                    <div class="upload-v5-success-title">Upload berhasil</div>
                    <div class="upload-v5-success-text">{html.escape(st.session_state.upload_success_message)}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# HERO
# ============================================================

if menu != "Kelola Data Upload":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">Simulasi Pengelolaan Sampah Kota Bandung</div>
            <div class="hero-subtitle">
                Dashboard ini difokuskan untuk membantu staf DLH melakukan simulasi kebutuhan operasional
                berdasarkan prediksi jumlah sampah bulanan.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MENU 1
# ============================================================

if menu == "Simulasi Pengelolaan":
    st.markdown('<div class="section-title">Simulasi Pengelolaan</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Isi operasional, lalu sistem akan menghitung prediksi sampah, anggaran, volume sampah, dan kebutuhan muatan truk compactor.</div>',
        unsafe_allow_html=True
    )

    input1, input2, input3, input4 = st.columns(4, gap="small")

    with input1:
        forecast_steps = st.slider(
            "Simulasi untuk berapa bulan ke depan?",
            min_value=1,
            max_value=int(forecast_max_months),
            value=min(12, int(forecast_max_months)),
            step=1
        )

    with input2:
        biaya_per_ton = st.number_input(
            "Biaya penanganan per ton",
            min_value=0,
            value=308482,
            step=1000
        )

    with input3:
        kapasitas_truk_compactor_m3 = st.selectbox(
            "Kapasitas truk compactor (m³)",
            options=[6, 12],
            index=1
        )

    with input4:
        hari_operasional_angkut_per_minggu = st.number_input(
            "Hari operasional angkut per minggu",
            min_value=1,
            max_value=7,
            value=4,
            step=1
        )

    forecast, forecast_model_label, model_selection_df = make_sarima_forecast(ts, forecast_steps)

    simulation_df = build_simulation_table(
        forecast=forecast,
        biaya_per_ton=biaya_per_ton,
        kapasitas_truk_compactor_m3=kapasitas_truk_compactor_m3,
        hari_operasional_angkut_per_minggu=hari_operasional_angkut_per_minggu
    )

    total_sampah = simulation_df["Prediksi Sampah (Ton)"].sum()
    total_anggaran = simulation_df["Estimasi Anggaran"].sum()
    total_volume_sampah = simulation_df["Estimasi Volume Sampah (m³)"].sum()
    total_muatan_truk = simulation_df["Estimasi Kebutuhan Muatan Truk"].sum()

    highest_row = simulation_df.loc[simulation_df["Prediksi Sampah (Ton)"].idxmax()]
    lowest_row = simulation_df.loc[simulation_df["Prediksi Sampah (Ton)"].idxmin()]

    start_period = format_periode(simulation_df["Tanggal"].min())
    end_period = format_periode(simulation_df["Tanggal"].max())

    mobile_kpi_summary([
        {"label": "🗓️ Periode", "value": f"{start_period} - {end_period}", "note": f"{forecast_steps} bulan ke depan"},
        {"label": "♻️ Total Sampah", "value": f"{format_angka(total_sampah)} ton"},
        {"label": "💰 Total Anggaran", "value": format_rupiah(total_anggaran), "note": f"{format_rupiah(biaya_per_ton)} per ton"},
        {"label": "📦 Total Volume", "value": f"{format_angka(total_volume_sampah)} m³", "note": f"densitas {format_angka(DENSITAS_SAMPAH_KG_PER_M3)} kg/m³"},
        {"label": "📈 Beban Tertinggi", "value": highest_row["Periode"], "note": f"{format_angka(highest_row['Prediksi Sampah (Ton)'])} ton"},
        {"label": "📉 Beban Terendah", "value": lowest_row["Periode"], "note": f"{format_angka(lowest_row['Prediksi Sampah (Ton)'])} ton"},
        {"label": "🚛 Total Muatan Truk", "value": f"{format_integer(total_muatan_truk)} muatan", "note": f"kapasitas {kapasitas_truk_compactor_m3} m³"},
        {"label": "🚚 Muatan Maks/Hari Angkut", "value": f"{format_integer(int(simulation_df['Muatan Truk per Hari Angkut'].max()))} muatan/hari"},
        {"label": "📆 Hari Angkut/Minggu", "value": f"{hari_operasional_angkut_per_minggu} hari/minggu", "note": "parameter jadwal angkut"},
    ])

    row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4, gap="small")

    with row1_col1:
        kpi_card("🗓️ Periode Simulasi", f"{start_period} - {end_period}", f"{forecast_steps} bulan ke depan")

    with row1_col2:
        kpi_card("♻️ Total Prediksi Sampah", f"{format_angka(total_sampah)} ton")

    with row1_col3:
        kpi_card("💰 Total Estimasi Anggaran", format_rupiah(total_anggaran), f" {format_rupiah(biaya_per_ton)} per ton")

    with row1_col4:
        kpi_card("📦 Total Estimasi Volume", f"{format_angka(total_volume_sampah)} m³", f"Densitas {format_angka(DENSITAS_SAMPAH_KG_PER_M3)} kg/m³")

    row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4, gap="small")

    with row2_col1:
        kpi_card("📈 Beban Tertinggi", highest_row["Periode"], f"{format_angka(highest_row['Prediksi Sampah (Ton)'])} ton")

    with row2_col2:
        kpi_card("📉 Beban Terendah", lowest_row["Periode"], f"{format_angka(lowest_row['Prediksi Sampah (Ton)'])} ton")

    with row2_col3:
        kpi_card("🚛 Total Kebutuhan Muatan Truk", f"{format_integer(total_muatan_truk)} muatan", f"Kapasitas {kapasitas_truk_compactor_m3} m³")

    with row2_col4:
        kpi_card("🚚 Muatan Maksimum per Hari Angkut", f"{format_integer(int(simulation_df['Muatan Truk per Hari Angkut'].max()))} muatan/hari", f"{hari_operasional_angkut_per_minggu} hari angkut/minggu")

    fig_forecast = make_forecast_chart(ts, forecast, theme)
    st.plotly_chart(
        fig_forecast,
        use_container_width=True,
        config={"displayModeBar": False, "responsive": True}
    )

    st.markdown('<div class="small-title table-title-mobile-tight">Tabel Simulasi Kebutuhan Operasional</div>', unsafe_allow_html=True)
    show_table(prepare_display_table(simulation_df))

    bullet_card(
        "Catatan simulasi",
        [
            f"Model prediksi yang digunakan adalah <b>{forecast_model_label}</b>, dipilih berdasarkan data historis terbaru.",
            f"Asumsi biaya penanganan adalah <b>{format_rupiah(biaya_per_ton)} per ton</b>.",
            f"Kapasitas truk compactor yang digunakan adalah <b>{kapasitas_truk_compactor_m3} m³</b> per muatan.",
            f"Prediksi sampah dalam satuan ton dikonversi menjadi volume memakai densitas <b>{format_angka(DENSITAS_SAMPAH_KG_PER_M3)} kg/m³</b>.",
            f"Frekuensi hari angkut diasumsikan <b>{hari_operasional_angkut_per_minggu} hari per minggu</b>.",
            "Kebutuhan muatan truk dan hari angkut dibulatkan ke atas agar estimasi lebih aman untuk kebutuhan lapangan.",
            "Dashboard ini digunakan sebagai simulasi awal untuk mendukung perencanaan operasional pengelolaan sampah."
        ]
    )


# ============================================================
# MENU 2
# ============================================================

elif menu == "Kelola Data Upload":
    render_upload_management_section(ts, source_data_type, periode_data, forecast_max_months)


# ============================================================
# MENU 3
# ============================================================

elif menu == "Ringkasan Data & Model":
    st.markdown('<div class="section-title">Ringkasan Data & Model</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Halaman ini menampilkan kondisi data, status model, EDA interaktif, dan evaluasi model dalam satu tampilan ringkas.</div>',
        unsafe_allow_html=True
    )

    eval_df, comparison_df, test_actual, test_forecast, eval_model_label, eval_model_selection_df = evaluate_sarima(ts)

    missing_value_total = int(df_raw.isnull().sum().sum())
    duplicate_total = int(df_raw.duplicated().sum())

    render_clean_html(
        textwrap.dedent(f"""
        <div class="overview-wrap">
            <div class="overview-mini-grid">
                <div class="overview-mini-card">
                    <div class="overview-mini-top">
                        <div class="overview-mini-icon">{kpi_svg("calendar")}</div>
                        <div class="overview-mini-label">Jumlah Data</div>
                    </div>
                    <div class="overview-mini-value">{len(ts.dropna())} baris</div>
                    <div class="overview-mini-note">{periode_data}</div>
                </div>

                <div class="overview-mini-card">
                    <div class="overview-mini-top">
                        <div class="overview-mini-icon">{kpi_svg("waste")}</div>
                        <div class="overview-mini-label">Variabel Utama</div>
                    </div>
                    <div class="overview-mini-value">jumlah_sampah</div>
                    <div class="overview-mini-note">Data bulanan dalam satuan {satuan}</div>
                </div>

                <div class="overview-mini-card">
                    <div class="overview-mini-top">
                        <div class="overview-mini-icon">{kpi_svg("trend_up")}</div>
                        <div class="overview-mini-label">Model Aktif</div>
                    </div>
                    <div class="overview-mini-value">{eval_model_label}</div>
                    <div class="overview-mini-note">Dipilih otomatis berdasarkan AIC terkecil</div>
                </div>

                <div class="overview-mini-card">
                    <div class="overview-mini-top">
                        <div class="overview-mini-icon">{kpi_svg("route")}</div>
                        <div class="overview-mini-label">Kesiapan Data</div>
                    </div>
                    <div class="overview-mini-value">{missing_value_total} missing</div>
                    <div class="overview-mini-note">{duplicate_total} data duplikat terdeteksi</div>
                </div>
            </div>

            <div class="overview-panel-grid">
                <div class="overview-panel">
                    <div class="overview-panel-title">
                        <div class="overview-mini-icon">{kpi_svg("cube")}</div>
                        Profil Data
                    </div>
                    <div class="overview-profile-table">
                        <div class="overview-profile-row">
                            <div class="overview-profile-key">Sumber data</div>
                            <div class="overview-profile-value">{source_data_name}</div>
                        </div>
                        <div class="overview-profile-row">
                            <div class="overview-profile-key">Wilayah</div>
                            <div class="overview-profile-value">{kota}</div>
                        </div>
                        <div class="overview-profile-row">
                            <div class="overview-profile-key">Provinsi</div>
                            <div class="overview-profile-value">{provinsi}</div>
                        </div>
                        <div class="overview-profile-row">
                            <div class="overview-profile-key">Periode</div>
                            <div class="overview-profile-value">{periode_data}</div>
                        </div>
                        <div class="overview-profile-row">
                            <div class="overview-profile-key">Bentuk data</div>
                            <div class="overview-profile-value">Bulanan</div>
                        </div>
                        <div class="overview-profile-row">
                            <div class="overview-profile-key">Satuan</div>
                            <div class="overview-profile-value">{satuan}</div>
                        </div>
                    </div>
                </div>

                <div class="overview-panel">
                    <div class="overview-panel-title">
                        <div class="overview-mini-icon">{kpi_svg("trend_up")}</div>
                        Status Model
                    </div>
                    <div class="overview-model-highlight">
                        <div class="overview-model-kicker">Model Evaluasi Terbaik</div>
                        <div class="overview-model-name">{eval_model_label}</div>
                        <div class="overview-model-caption">
                            Sistem membandingkan beberapa kandidat SARIMA dan memilih konfigurasi dengan nilai AIC terkecil.
                        </div>
                    </div>

                    <div class="overview-step-list">
                        <div class="overview-step">
                            <div class="overview-step-number">1</div>
                            <div class="overview-step-text"><b>Preprocessing otomatis</b> dilakukan setelah data bawaan atau data upload dibaca.</div>
                        </div>
                        <div class="overview-step">
                            <div class="overview-step-number">2</div>
                            <div class="overview-step-text"><b>Seleksi kandidat SARIMA</b> berjalan untuk mencari parameter yang paling sesuai.</div>
                        </div>
                        <div class="overview-step">
                            <div class="overview-step-number">3</div>
                            <div class="overview-step-text"><b>Prediksi sampai 24 bulan</b> digunakan untuk menghitung sampah, anggaran, volume, dan muatan truk.</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """)
    )

    render_eda_section(ts, theme)

    st.markdown('<div class="small-title">Evaluasi Model</div>', unsafe_allow_html=True)
    show_table(prepare_eval_display(eval_df))

    st.markdown('<div class="small-title">Aktual vs Prediksi Data Uji</div>', unsafe_allow_html=True)
    fig_eval = make_eval_chart(test_actual, test_forecast, theme)
    st.plotly_chart(fig_eval, use_container_width=True, config={"displayModeBar": False, "responsive": True})

    st.markdown('<div class="small-title table-title-mobile-tight">Tabel Aktual vs Prediksi</div>', unsafe_allow_html=True)
    show_table(prepare_comparison_display(comparison_df))

    
