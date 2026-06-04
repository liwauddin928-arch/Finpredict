# =============================================================================
#  FinPredict v11.0 — AI Financial Forecasting Dashboard
#  Desain: dark KPI header, clean chart, icon sidebar — inspired by ref image
#  All LSTM logic preserved from v10 — only design upgraded
#  Capstone Dicoding × DBS Foundation
# =============================================================================

import os
import pickle
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ══════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════
SEQ_LEN         = 30
N_FEATURES      = 39
THRESH_LOW      = 0.40
THRESH_HIGH     = 0.70
SAVINGS_TARGET  = 0.10
DTR_CRITICAL    = 7
DTR_WARNING     = 14
DTR_SENTINEL    = 9_999
DOOM_RATE_ALERT = 0.02
DEFICIT_ALERT   = 0.55
WANTS_ALERT     = 0.60

FEATURE_COLS = [
    "total_pengeluaran_harian","total_pemasukan_harian","total_wants_harian",
    "doom_spending_harian","shock_harian","pengeluaran_makanan","usia",
    "saldo_bersih_harian","saldo_berjalan","rasio_wants",
    "is_doom_spending","is_financial_shock","tipe_user_encoded",
    "hari_dalam_bulan","hari_dalam_minggu","is_weekend","bulan",
    "is_harbolnas","fase_bulan_encoded",
    "sin_doy","cos_doy","sin_dom","cos_dom",
    "pengeluaran_lag_1d","pengeluaran_lag_3d","pengeluaran_lag_7d",
    "saldo_lag_1d","wants_lag_3d","doom_lag_7d",
    "rolling_mean_7d","rolling_mean_14d","rolling_mean_30d",
    "rolling_std_7d","rolling_max_7d","rolling_wants_7d",
    "days_to_ruin","interaksi_hedon_muda","kebiasaan_boros_harian","rasio_makanan",
]
REQUIRED_COLS = [
    "user_id","tanggal_transaksi","saldo_berjalan",
    "label_defisit_besok","total_pengeluaran_harian",
    "total_pemasukan_harian","rasio_wants",
]

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FinPredict AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
#  DESIGN TOKENS  — mirip referensi gambar
#  Palette: dark navy header card, mint green accent, coral accent,
#           clean white chart area, light mint bg
# ══════════════════════════════════════════════════════════════
C = {
    # brand
    "primary":   "#4F6EF7",   # indigo
    "secondary": "#7C3AED",   # violet
    # semantic
    "success":   "#22C55E",   # mint green  (ref: green accent)
    "warning":   "#F59E0B",   # amber
    "danger":    "#EF4444",   # coral-red   (ref: coral accent)
    "info":      "#3B82F6",
    # card / ui
    "card_hdr":  "#1A2035",   # dark navy — KPI card header (ref image)
    "card_bg":   "#FFFFFF",
    "surface":   "#FFFFFF",
    "bg":        "#EBF4F0",   # light mint tint (ref bg)
    "sidebar":   "#1A2035",   # dark navy sidebar
    "border":    "#DDE3EC",
    "text":      "#1A2035",
    "muted":     "#6B7A99",
    # chart palette
    "chart": ["#4F6EF7","#22C55E","#F59E0B","#EF4444","#7C3AED",
              "#3B82F6","#F97316","#06B6D4","#84CC16","#E11D48"],
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}
html, body, [class*="css"] {{
    font-family: 'Inter', system-ui, sans-serif !important;
    background: {C["bg"]} !important;
    color: {C["text"]};
}}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{
    padding: 0 !important;
    max-width: 100% !important;
}}

/* ═══════════════════════════════
   SIDEBAR — dark navy icon sidebar
   ═══════════════════════════════ */
[data-testid="stSidebar"] {{
    background: {C["sidebar"]} !important;
    border-right: none !important;
    min-width: 220px !important;
    max-width: 220px !important;
}}
[data-testid="stSidebar"] > div:first-child {{
    padding: 0 !important;
}}
[data-testid="stSidebar"] * {{
    color: #A8B4CF !important;
}}
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stDateInput > div > div {{
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
    color: #E2E8F0 !important;
    font-size: 13px !important;
}}
[data-testid="stSidebar"] .stRadio > div {{
    gap: 2px !important;
}}
[data-testid="stSidebar"] .stRadio label {{
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    padding: 9px 16px !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: all .15s !important;
    border: none !important;
}}
[data-testid="stSidebar"] .stRadio label:hover {{
    background: rgba(79,110,247,0.15) !important;
    color: #fff !important;
}}
[data-testid="stSidebar"] [aria-checked="true"] + label,
[data-testid="stSidebar"] .stRadio label[data-checked="true"] {{
    background: rgba(79,110,247,0.25) !important;
    color: #818CF8 !important;
}}
[data-testid="stSidebar"] .stSlider [data-testid="stSliderThumb"] {{
    background: {C["primary"]} !important;
}}
[data-testid="stSidebar"] hr {{
    border-color: rgba(255,255,255,0.08) !important;
}}

/* ═══════════════════════════════
   MAIN CONTENT WRAPPER
   ═══════════════════════════════ */
.main-wrap {{
    padding: 1.25rem 1.75rem 2rem;
}}

/* ═══════════════════════════════
   TOP NAV BAR  (breadcrumb style)
   ═══════════════════════════════ */
.topnav {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: .875rem 1.75rem;
    background: {C["surface"]};
    border-bottom: 1px solid {C["border"]};
    margin-bottom: 0;
}}
.topnav-left {{
    display: flex;
    align-items: center;
    gap: 10px;
}}
.topnav-logo {{
    width: 32px; height: 32px;
    background: {C["primary"]};
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    color: #fff; font-weight: 800; font-size: 14px;
}}
.topnav-brand {{ font-size: 15px; font-weight: 700; color: {C["text"]}; }}
.topnav-sep {{ color: {C["border"]}; margin: 0 4px; }}
.topnav-crumb {{ font-size: 12px; color: {C["muted"]}; }}
.topnav-right {{
    display: flex; align-items: center; gap: 12px;
}}
.engine-pill {{
    background: linear-gradient(135deg, {C["primary"]}, {C["secondary"]});
    color: #fff;
    font-size: 10px; font-weight: 600;
    padding: 4px 12px; border-radius: 20px;
    letter-spacing: .04em;
}}

/* ═══════════════════════════════
   KPI CARD  — dark header style (ref image)
   ═══════════════════════════════ */
.kpi-card {{
    background: {C["card_bg"]};
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(26,32,53,0.08);
    transition: transform .18s, box-shadow .18s;
    height: 100%;
    border: 1px solid {C["border"]};
}}
.kpi-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(26,32,53,0.13);
}}
.kpi-hdr {{
    background: {C["card_hdr"]};
    padding: .55rem 1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}}
.kpi-hdr-lbl {{
    font-size: 10px;
    font-weight: 700;
    color: #A8B4CF;
    text-transform: uppercase;
    letter-spacing: .08em;
}}
.kpi-hdr-ico {{
    width: 26px; height: 26px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px;
}}
.kpi-body {{
    padding: .8rem 1rem .9rem;
}}
.kpi-value {{
    font-size: 26px;
    font-weight: 700;
    color: {C["text"]};
    line-height: 1.1;
    font-variant-numeric: tabular-nums;
    letter-spacing: -.02em;
}}
.kpi-sub {{
    font-size: 11px;
    margin-top: 5px;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 3px;
}}

/* ═══════════════════════════════
   PANEL  — white card with clean header
   ═══════════════════════════════ */
.panel {{
    background: {C["surface"]};
    border-radius: 12px;
    border: 1px solid {C["border"]};
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(26,32,53,0.05);
}}
.panel-hdr {{
    padding: .75rem 1.1rem;
    border-bottom: 1px solid {C["border"]};
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: {C["surface"]};
}}
.panel-title {{
    font-size: 13px;
    font-weight: 600;
    color: {C["text"]};
    display: flex;
    align-items: center;
    gap: 6px;
}}
.panel-sub {{
    font-size: 11px;
    color: {C["muted"]};
    margin-top: 1px;
}}
.panel-body {{
    padding: 1rem 1.1rem;
}}

/* ═══════════════════════════════
   SECTION HEADER
   ═══════════════════════════════ */
.sec-title {{
    font-size: 15px;
    font-weight: 700;
    color: {C["text"]};
    margin-bottom: 2px;
}}
.sec-sub {{
    font-size: 12px;
    color: {C["muted"]};
    margin-bottom: 12px;
}}

/* ═══════════════════════════════
   ALERT
   ═══════════════════════════════ */
.alert-d {{ background:#FEF2F2; border-left:3px solid {C["danger"]};  padding:.65rem 1rem; border-radius:0 8px 8px 0; color:#991B1B; font-size:13px; margin:.5rem 0; }}
.alert-w {{ background:#FFFBEB; border-left:3px solid {C["warning"]}; padding:.65rem 1rem; border-radius:0 8px 8px 0; color:#92400E; font-size:13px; margin:.5rem 0; }}
.alert-s {{ background:#ECFDF5; border-left:3px solid {C["success"]}; padding:.65rem 1rem; border-radius:0 8px 8px 0; color:#065F46; font-size:13px; margin:.5rem 0; }}
.alert-i {{ background:#EFF6FF; border-left:3px solid {C["info"]};    padding:.65rem 1rem; border-radius:0 8px 8px 0; color:#1E40AF; font-size:13px; margin:.5rem 0; }}

/* ═══════════════════════════════
   HEALTH BAR
   ═══════════════════════════════ */
.hbar {{ background:{C["border"]}; border-radius:100px; height:6px; overflow:hidden; margin:.3rem 0; }}
.hfill {{ height:6px; border-radius:100px; transition:width .5s ease; }}

/* ═══════════════════════════════
   DIVIDER
   ═══════════════════════════════ */
hr.div {{ border:none; border-top:1px solid {C["border"]}; margin:1.25rem 0; }}

/* ═══════════════════════════════
   LSTM BANNER
   ═══════════════════════════════ */
.lstm-banner {{
    background: linear-gradient(135deg,{C["card_hdr"]} 0%,#243060 55%,#312E81 100%);
    border-radius: 12px;
    padding: 1.2rem 1.6rem;
    color: #fff;
    margin-bottom: 1.25rem;
    border: 1px solid rgba(255,255,255,0.06);
}}

/* ═══════════════════════════════
   TABS
   ═══════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    background: {C["bg"]};
    border-radius: 10px;
    padding: 4px;
    border: 1px solid {C["border"]};
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 7px;
    font-size: 13px;
    font-weight: 500;
    padding: 6px 16px;
}}
.stTabs [aria-selected="true"] {{
    background: {C["surface"]};
    box-shadow: 0 1px 4px rgba(0,0,0,.08);
}}

/* ═══════════════════════════════
   INSIGHT CARD
   ═══════════════════════════════ */
.insight-card {{
    background: {C["surface"]};
    border: 1px solid {C["border"]};
    border-radius: 10px;
    padding: .85rem 1rem;
    margin-bottom: .5rem;
    display: flex;
    gap: 10px;
    align-items: flex-start;
    transition: box-shadow .15s;
}}
.insight-card:hover {{ box-shadow: 0 4px 16px rgba(0,0,0,.07); }}
.insight-title {{ font-size:13px; font-weight:600; color:{C["text"]}; margin-bottom:3px; }}
.insight-body  {{ font-size:12px; color:{C["muted"]}; line-height:1.55; }}

/* ═══════════════════════════════
   STATUS BADGE
   ═══════════════════════════════ */
.badge {{
    display: inline-flex; align-items: center;
    font-size: 11px; font-weight: 600;
    padding: 3px 10px; border-radius: 20px;
    gap: 4px;
}}
.badge-s {{ background: rgba(34,197,94,.15);  color: #15803D; }}
.badge-w {{ background: rgba(245,158,11,.15); color: #92400E; }}
.badge-d {{ background: rgba(239,68,68,.15);  color: #991B1B; }}
.badge-i {{ background: rgba(79,110,247,.12); color: #3730A3; }}

/* ═══════════════════════════════
   STAT ROW  (mini metric inside panel)
   ═══════════════════════════════ */
.stat-row {{
    display: flex; justify-content: space-between;
    align-items: center; padding: .4rem 0;
    border-bottom: 1px solid {C["border"]};
    font-size: 12px;
}}
.stat-row:last-child {{ border-bottom: none; }}
.stat-lbl {{ color: {C["muted"]}; font-weight: 500; }}
.stat-val {{ font-weight: 600; color: {C["text"]}; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  FILE FINDER
# ══════════════════════════════════════════════════════════════
def _find(filename: str) -> str | None:
    for p in [filename, f"data/{filename}", f"models/{filename}"]:
        if os.path.exists(p):
            return p
    return None


# ══════════════════════════════════════════════════════════════
#  DATA LOADERS
# ══════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def load_daily() -> tuple[pd.DataFrame, list]:
    path = _find("daily_features_v8.csv")
    if not path:
        return pd.DataFrame(), ["daily_features_v8.csv tidak ditemukan."]
    df = pd.read_csv(path, parse_dates=["tanggal_transaksi"])
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return pd.DataFrame(), [f"Kolom wajib tidak ada: {missing}"]
    return df, []

@st.cache_data(ttl=3600, show_spinner=False)
def load_transactions() -> pd.DataFrame:
    path = _find("transactions_clean_v8.csv")
    return pd.read_csv(path, parse_dates=["tanggal_transaksi"]) if path else pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def load_optional(fname: str) -> pd.DataFrame:
    path = _find(fname)
    return pd.read_csv(path) if path else pd.DataFrame()

@st.cache_resource(show_spinner=False)
def load_model() -> tuple:
    path = _find("model_lstm.h5")
    if not path:
        return None, False, "model_lstm.h5 tidak ditemukan"
    try:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        m = tf.keras.models.load_model(path, compile=False)
        return m, True, f"TensorFlow {tf.__version__}"
    except Exception as e:
        return None, False, str(e)

@st.cache_resource(show_spinner=False)
def load_scalers() -> tuple:
    path = _find("scalers_v8.pkl")
    if not path:
        return {}, False
    try:
        with open(path, "rb") as f:
            return pickle.load(f), True
    except Exception:
        return {}, False


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════
def fmt_rp(n: float) -> str:
    if abs(n) >= 1_000_000_000: return f"Rp {n/1e9:.1f}M"
    if abs(n) >= 1_000_000:     return f"Rp {n/1e6:.1f}Jt"
    if abs(n) >= 1_000:         return f"Rp {n/1e3:.0f}K"
    return f"Rp {n:.0f}"


def kpi_card(label: str, value: str, sub: str = "", sub_color: str = "",
             icon: str = "💰", icon_bg: str = "") -> str:
    """KPI card with dark header (ref image style)."""
    ibg  = icon_bg or C["primary"]
    sclr = sub_color or C["muted"]
    sub_h = f'<div class="kpi-sub" style="color:{sclr}">{sub}</div>' if sub else ""
    return f"""
    <div class="kpi-card">
      <div class="kpi-hdr">
        <span class="kpi-hdr-lbl">{label}</span>
        <span class="kpi-hdr-ico" style="background:{ibg}20;color:{ibg}">{icon}</span>
      </div>
      <div class="kpi-body">
        <div class="kpi-value">{value}</div>
        {sub_h}
      </div>
    </div>"""


def panel_open(title: str, sub: str = "", right_html: str = "") -> str:
    sub_h = f'<div class="panel-sub">{sub}</div>' if sub else ""
    right = f'<div>{right_html}</div>' if right_html else ""
    return f"""<div class="panel">
  <div class="panel-hdr">
    <div><div class="panel-title">{title}</div>{sub_h}</div>
    {right}
  </div>
  <div class="panel-body">"""

def panel_close() -> str:
    return "</div></div>"


def section(title: str, sub: str = "") -> None:
    st.markdown(f'<div class="sec-title">{title}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="sec-sub">{sub}</div>', unsafe_allow_html=True)


def alert(text: str, kind: str = "w") -> str:
    cls = {"d":"alert-d","w":"alert-w","s":"alert-s","i":"alert-i"}.get(kind,"alert-w")
    return f'<div class="{cls}">{text}</div>'


def badge(text: str, kind: str = "i") -> str:
    cls = {"s":"badge-s","w":"badge-w","d":"badge-d","i":"badge-i"}.get(kind,"badge-i")
    return f'<span class="badge {cls}">{text}</span>'


def fig_style(fig: go.Figure, h: int = 340) -> go.Figure:
    """Apply clean chart style matching the reference image."""
    fig.update_layout(
        height=h,
        margin=dict(l=6, r=6, t=36, b=6),
        paper_bgcolor=C["surface"],
        plot_bgcolor=C["surface"],
        font=dict(family="Inter", size=11, color=C["text"]),
        legend=dict(
            bgcolor=C["surface"], bordercolor=C["border"], borderwidth=1,
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=11),
        ),
        xaxis=dict(
            showgrid=True, gridcolor="#F0F4F8", gridwidth=1,
            zeroline=False, linecolor=C["border"],
            tickfont=dict(size=10, color=C["muted"]),
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#F0F4F8", gridwidth=1,
            zeroline=False, linecolor=C["border"],
            tickfont=dict(size=10, color=C["muted"]),
        ),
    )
    return fig


def health_score(df: pd.DataFrame) -> tuple:
    score = 100.0
    if "label_defisit_besok" in df.columns:
        dr = df["label_defisit_besok"].mean()
        if not np.isnan(dr): score -= min(40, dr * 80)
    if "rasio_wants" in df.columns:
        rw = df["rasio_wants"].mean()
        if not np.isnan(rw): score -= min(20, max(0,(rw-.4)*60))
    if "days_to_ruin" in df.columns:
        dtr = df[df["days_to_ruin"]<DTR_SENTINEL]["days_to_ruin"].median()
        if not np.isnan(dtr):
            score -= 25 if dtr<DTR_CRITICAL else 12 if dtr<DTR_WARNING else 4 if dtr<30 else 0
    s = max(0, min(100, int(score)))
    if s >= 80: return s, "Excellent", C["success"]
    if s >= 60: return s, "Good",      C["info"]
    if s >= 40: return s, "Warning",   C["warning"]
    return s, "Danger", C["danger"]


def filter_df(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    if df.empty: return df
    if f["user"] != "All Users" and "user_id" in df.columns:
        df = df[df["user_id"] == f["user"]]
    if f["tipe"] != "All Types" and "tipe_user" in df.columns:
        df = df[df["tipe_user"] == f["tipe"]]
    if "tanggal_transaksi" in df.columns and len(f["dates"]) == 2:
        d0, d1 = pd.Timestamp(f["dates"][0]), pd.Timestamp(f["dates"][1])
        df = df[(df["tanggal_transaksi"] >= d0) & (df["tanggal_transaksi"] <= d1)]
    return df.copy()


# ══════════════════════════════════════════════════════════════
#  LSTM ENGINE  (identical to v10 — no changes)
# ══════════════════════════════════════════════════════════════
def _scale_features(uid, X_raw, scalers):
    if uid in scalers:
        sc = scalers[uid]
    elif scalers:
        sc = next(iter(scalers.values()))
        st.warning(f"Scaler `{uid}` tidak ditemukan, pakai fallback.")
    else:
        return X_raw, None
    n = sc.n_features_in_
    if X_raw.shape[1] < n:
        X_raw = np.hstack([X_raw, np.zeros((X_raw.shape[0], n-X_raw.shape[1]))])
    return sc.transform(X_raw[:, :n]), sc


def _build_seq(X_scaled):
    n, nf = X_scaled.shape
    if n < SEQ_LEN:
        seq = np.vstack([np.zeros((SEQ_LEN-n, nf)), X_scaled])
    else:
        seq = X_scaled[-SEQ_LEN:]
    seq = seq.astype(np.float32)
    if seq.shape[1] < N_FEATURES:
        seq = np.hstack([seq, np.zeros((SEQ_LEN, N_FEATURES-seq.shape[1]))])
    else:
        seq = seq[:, :N_FEATURES]
    return seq.reshape(1, SEQ_LEN, N_FEATURES)


def predict_user(uid, df_daily, model, scalers):
    try:
        df_u = df_daily[df_daily["user_id"]==uid].sort_values("tanggal_transaksi").reset_index(drop=True)
        if df_u.empty: return {"error": f"Tidak ada data {uid}"}
        missing = [c for c in FEATURE_COLS if c not in df_u.columns]
        if missing: return {"error": f"Kolom hilang: {missing[:3]}"}
        X_raw = df_u[FEATURE_COLS].fillna(0).values.astype(np.float32)
        X_sc, _ = _scale_features(uid, X_raw, scalers)
        prob = float(np.clip(model.predict(_build_seq(X_sc), verbose=0).flatten()[0], 0, 1))
        status = "BAHAYA" if prob>=THRESH_HIGH else "WASPADA" if prob>=THRESH_LOW else "AMAN"
        return {"probability":prob,"prediction":int(prob>=.5),"status":status,"error":None}
    except Exception as e:
        return {"error":str(e),"probability":None}


def forecast_30_days(uid, df_daily, model, scalers):
    try:
        df_u = df_daily[df_daily["user_id"]==uid].sort_values("tanggal_transaksi").reset_index(drop=True)
        if df_u.empty or len(df_u)<7: return {"error":"Data historis tidak cukup"}
        missing = [c for c in FEATURE_COLS if c not in df_u.columns]
        if missing: return {"error":f"Kolom hilang: {missing[:3]}"}
        X_raw = df_u[FEATURE_COLS].fillna(0).values.astype(np.float32)
        X_sc, scaler = _scale_features(uid, X_raw, scalers)
        if X_sc.shape[0] < SEQ_LEN:
            X_sc = np.vstack([np.zeros((SEQ_LEN-X_sc.shape[0], X_sc.shape[1])), X_sc])
        cur = X_sc[-SEQ_LEN:].astype(np.float32)
        saldo   = float(df_u["saldo_berjalan"].iloc[-1])
        last_dt = df_u["tanggal_transaksi"].iloc[-1]
        rec     = df_u.tail(60)
        avg_inc = float(rec["total_pemasukan_harian"].mean())
        avg_w   = float(rec["total_wants_harian"].mean())
        avg_n   = float((rec["total_pengeluaran_harian"]-rec["total_wants_harian"]).clip(lower=0).mean())
        try:
            si = FEATURE_COLS.index("saldo_berjalan")
            si_ok = scaler is not None and si < scaler.n_features_in_
        except ValueError:
            si_ok = False
        probs, balances, dates = [], [], []
        for i in range(30):
            prob = float(np.clip(model.predict(_build_seq(cur), verbose=0).flatten()[0], 0, 1))
            probs.append(prob)
            saldo += avg_inc - (prob*avg_w + (1-prob)*avg_n)
            balances.append(saldo)
            dates.append(last_dt + timedelta(days=i+1))
            nr = cur[-1].copy()
            if si_ok:
                d2 = X_raw[-1:].copy(); d2[0,si] = saldo
                nr[si] = scaler.transform(d2[:,:scaler.n_features_in_])[0,si]
            cur = np.vstack([cur[1:], nr])
        return {"dates":dates,"probs":probs,"balances":balances,
                "n_deficit_days":sum(1 for b in balances if b<0),"error":None}
    except Exception as e:
        return {"error":str(e)}


def heuristic_predict(df_u):
    dtr = float(df_u["days_to_ruin"].iloc[-1]) if "days_to_ruin" in df_u.columns else 30
    rw  = float(df_u["rasio_wants"].iloc[-1])   if "rasio_wants"  in df_u.columns else .5
    bal = float(df_u["saldo_berjalan"].iloc[-1])
    if bal<=0 or dtr<=0: p=.88
    elif dtr<DTR_CRITICAL: p=.72
    elif dtr<DTR_WARNING:  p=.52
    else: p=float(np.clip(.28+rw*.2,0,1))
    status = "BAHAYA" if p>=THRESH_HIGH else "WASPADA" if p>=THRESH_LOW else "AMAN"
    return {"probability":p,"prediction":int(p>=.5),"status":status,"error":None,"heuristic":True}


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
def render_sidebar(df_daily: pd.DataFrame) -> dict:
    with st.sidebar:
        # Logo area
        st.markdown(f"""
        <div style="padding:1.25rem 1rem 1rem;border-bottom:1px solid rgba(255,255,255,.08)">
          <div style="display:flex;align-items:center;gap:10px">
            <div style="width:36px;height:36px;background:{C['primary']};border-radius:10px;
                        display:flex;align-items:center;justify-content:center;
                        font-size:18px;flex-shrink:0">🧠</div>
            <div>
              <div style="font-size:14px;font-weight:700;color:#fff;line-height:1.2">FinPredict</div>
              <div style="font-size:10px;color:#64748B">AI Platform v11</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        # Nav
        st.markdown('<div style="padding:.75rem .5rem .25rem;font-size:10px;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:.07em">Menu</div>', unsafe_allow_html=True)
        page = st.radio("nav", [
            "🏠  Executive Summary",
            "📊  Spending Overview",
            "🧠  LSTM Prediction",
            "⚡  Risk & Early Warning",
            "📈  Behavior Analysis",
            "🔍  Category Deep Dive",
            "⚖️  A/B Testing",
            "📋  Data & Dokumentasi",
        ], label_visibility="collapsed")

        st.divider()

        # Filters
        st.markdown('<div style="padding:0 .25rem .25rem;font-size:10px;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:.07em">Filter Global</div>', unsafe_allow_html=True)

        users = ["All Users"]
        tipes = ["All Types"]
        min_d = max_d = datetime.now().date()
        if not df_daily.empty:
            users += sorted(df_daily["user_id"].unique().tolist())
            if "tipe_user" in df_daily.columns:
                tipes += sorted(df_daily["tipe_user"].dropna().unique().tolist())
            min_d = df_daily["tanggal_transaksi"].min().date()
            max_d = df_daily["tanggal_transaksi"].max().date()

        sel_user = st.selectbox("👤 User", users)
        sel_tipe = st.selectbox("🏷️ Tipe User", tipes)
        dr = st.date_input("📅 Periode",
                           (max_d - timedelta(days=89), max_d),
                           min_value=min_d, max_value=max_d)

        # Interactive sliders
        st.markdown('<div style="margin-top:.75rem;font-size:10px;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:.07em">Threshold</div>', unsafe_allow_html=True)

        if not df_daily.empty and "total_pengeluaran_harian" in df_daily.columns:
            avg_sp = float(df_daily["total_pengeluaran_harian"].mean())
            max_sp = float(df_daily["total_pengeluaran_harian"].quantile(.95))
        else:
            avg_sp, max_sp = 200_000, 500_000

        st.markdown('<div style="font-size:11px;color:#64748B;margin-top:6px;margin-bottom:2px">💸 Limit Harian</div>', unsafe_allow_html=True)
        spend_limit = st.slider("sl", min_value=50_000, max_value=int(max_sp),
                                value=int(avg_sp), step=10_000,
                                label_visibility="collapsed", format="Rp %d")

        st.markdown('<div style="font-size:11px;color:#64748B;margin-top:4px;margin-bottom:2px">🎯 Risk Threshold</div>', unsafe_allow_html=True)
        risk_thresh = st.slider("rt", min_value=10, max_value=90, value=55, step=5,
                                label_visibility="collapsed", format="%d%%")

        st.divider()
        st.markdown(f'<div style="font-size:10px;color:#475569;text-align:center;line-height:1.8">FinPredict v11.0 · Dicoding × DBS<br><span style="color:#818CF8">LSTM Engine · Bug-Free</span></div>', unsafe_allow_html=True)

    return {
        "page":        page,
        "user":        sel_user,
        "tipe":        sel_tipe,
        "dates":       dr,
        "spend_limit": spend_limit,
        "risk_thresh": risk_thresh / 100,
    }


# ══════════════════════════════════════════════════════════════
#  SHARED: TOP NAV BAR
# ══════════════════════════════════════════════════════════════
def topnav(page_name: str, model_ok: bool) -> None:
    eng = f'<span class="engine-pill">{"✅ LSTM ACTIVE" if model_ok else "⚡ HEURISTIC"}</span>'
    st.markdown(f"""
    <div class="topnav">
      <div class="topnav-left">
        <div class="topnav-logo">F</div>
        <span class="topnav-brand">FinPredict AI</span>
        <span class="topnav-sep">›</span>
        <span class="topnav-crumb">{page_name}</span>
      </div>
      <div class="topnav-right">{eng}</div>
    </div>
    <div class="main-wrap">""", unsafe_allow_html=True)


def close_wrap() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 1: EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════
def page_executive(df_daily, df_trx, filters, model, scalers, model_ok):
    topnav("Executive Summary", model_ok)

    df_d = filter_df(df_daily, filters)
    df_t = filter_df(df_trx, filters) if not df_trx.empty else pd.DataFrame()
    df_exp = df_t[df_t["tipe_transaksi"]=="Pengeluaran"] if not df_t.empty and "tipe_transaksi" in df_t.columns else pd.DataFrame()
    df_inc = df_t[df_t["tipe_transaksi"]=="Pemasukan"]  if not df_t.empty and "tipe_transaksi" in df_t.columns else pd.DataFrame()

    tot_exp  = float(df_exp["nominal"].sum()) if not df_exp.empty else 0.0
    tot_inc  = float(df_inc["nominal"].sum()) if not df_inc.empty else 0.0
    net_cf   = tot_inc - tot_exp
    sav_rate = max(0., (tot_inc-tot_exp)/max(tot_inc,1))
    deficit  = float(np.nan_to_num(df_d["label_defisit_besok"].mean())) if not df_d.empty else 0.
    avg_day  = float(np.nan_to_num(df_d["total_pengeluaran_harian"].mean())) if not df_d.empty else 0.
    rw_avg   = float(np.nan_to_num(df_d["rasio_wants"].mean())) if not df_d.empty else 0.
    limit    = filters["spend_limit"]

    half    = len(df_d)//2
    prev_sp = float(df_d.iloc[:half]["total_pengeluaran_harian"].mean()) if half>0 else avg_day
    curr_sp = float(df_d.iloc[half:]["total_pengeluaran_harian"].mean()) if half>0 else avg_day
    d_pct   = (curr_sp-prev_sp)/max(abs(prev_sp),1)
    hs, hl, hc = health_score(df_d) if not df_d.empty else (50,"Warning",C["warning"])

    # ── KPI row 1 ──
    st.markdown(f'<div style="font-size:13px;font-weight:600;color:{C["muted"]};margin-bottom:10px">'
                f'Data: {len(df_d):,} hari · {df_d["user_id"].nunique() if not df_d.empty else 0} users</div>',
                unsafe_allow_html=True)

    k = st.columns(4, gap="small")
    over = avg_day > limit
    with k[0]:
        st.markdown(kpi_card(
            "TOTAL SPENDING", fmt_rp(tot_exp),
            f"{'↑' if d_pct>=0 else '↓'} {abs(d_pct):.1%} vs periode lalu",
            C["danger"] if d_pct>0 else C["success"], "💸",
            icon_bg=C["danger"]), unsafe_allow_html=True)
    with k[1]:
        st.markdown(kpi_card(
            "TOTAL PEMASUKAN", fmt_rp(tot_inc),
            "Pemasukan periode ini", C["muted"], "💰",
            icon_bg=C["success"]), unsafe_allow_html=True)
    with k[2]:
        st.markdown(kpi_card(
            "NET CASHFLOW", fmt_rp(abs(net_cf)),
            "✅ Surplus" if net_cf>=0 else "❌ Defisit",
            C["success"] if net_cf>=0 else C["danger"], "⚡",
            icon_bg=C["primary"]), unsafe_allow_html=True)
    with k[3]:
        st.markdown(kpi_card(
            "HEALTH SCORE", f"{hs}/100", hl, hc, "❤️",
            icon_bg=hc), unsafe_allow_html=True)

    st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)

    k2 = st.columns(4, gap="small")
    with k2[0]:
        st.markdown(kpi_card(
            "AVG DAILY SPEND", fmt_rp(avg_day),
            f"{'🔴 Di atas' if over else '🟢 Dalam'} limit {fmt_rp(limit)}",
            C["danger"] if over else C["success"], "📊",
            icon_bg=C["info"]), unsafe_allow_html=True)
    with k2[1]:
        st.markdown(kpi_card(
            "WANTS RATIO", f"{rw_avg:.1%}",
            "🔴 Tinggi" if rw_avg>WANTS_ALERT else "🟢 Sehat",
            C["danger"] if rw_avg>WANTS_ALERT else C["success"], "🎯",
            icon_bg=C["warning"]), unsafe_allow_html=True)
    with k2[2]:
        high_risk = deficit > filters["risk_thresh"]
        st.markdown(kpi_card(
            "DEFICIT RISK", f"{deficit:.1%}",
            f"{'⚠️ Melebihi' if high_risk else '✅ Di bawah'} threshold {filters['risk_thresh']:.0%}",
            C["danger"] if high_risk else C["success"], "🚨",
            icon_bg=C["danger"]), unsafe_allow_html=True)
    with k2[3]:
        st.markdown(kpi_card(
            "SAVINGS RATE", f"{sav_rate:.1%}",
            "✅ On Track" if sav_rate>SAVINGS_TARGET else "⚠️ Di Bawah Target",
            C["success"] if sav_rate>SAVINGS_TARGET else C["warning"], "🏦",
            icon_bg=C["success"]), unsafe_allow_html=True)

    st.markdown("<hr class='div'>", unsafe_allow_html=True)

    # ── Charts row ──
    cl, cr = st.columns([1.65, 1], gap="medium")
    with cl:
        section("📈 Tren Pengeluaran Harian",
                "Moving average · deficit risk overlay · spending limit")
        if not df_d.empty:
            tren = (df_d.groupby("tanggal_transaksi")
                        .agg(sp=("total_pengeluaran_harian","mean"),
                             dr=("label_defisit_besok","mean"))
                        .reset_index())
            tren["r7"]  = tren["sp"].rolling(7,  min_periods=1).mean()
            tren["r30"] = tren["sp"].rolling(30, min_periods=1).mean()

            fig = make_subplots(specs=[[{"secondary_y":True}]])
            # Area fill
            fig.add_trace(go.Scatter(
                x=tren["tanggal_transaksi"], y=tren["sp"],
                fill="tozeroy", fillcolor="rgba(79,110,247,.05)",
                line=dict(width=0), showlegend=False, name="Harian"),
                secondary_y=False)
            # MA lines
            fig.add_trace(go.Scatter(
                x=tren["tanggal_transaksi"], y=tren["r7"],
                line=dict(color=C["primary"], width=2.5), name="MA 7d"),
                secondary_y=False)
            fig.add_trace(go.Scatter(
                x=tren["tanggal_transaksi"], y=tren["r30"],
                line=dict(color=C["warning"], width=2, dash="dash"), name="MA 30d"),
                secondary_y=False)
            # Spending limit — interactive
            fig.add_hline(y=limit, line_dash="dot", line_color=C["danger"],
                          line_width=1.5,
                          annotation_text=f"Limit {fmt_rp(limit)}",
                          annotation_position="right",
                          secondary_y=False)
            # Risk bars
            fig.add_trace(go.Bar(
                x=tren["tanggal_transaksi"], y=tren["dr"]*100,
                marker_color=C["danger"], opacity=0.14, name="Risk %"),
                secondary_y=True)
            # Risk threshold
            fig.add_hline(y=filters["risk_thresh"]*100,
                          line_dash="dash", line_color=C["warning"], line_width=1,
                          annotation_text=f"Threshold {filters['risk_thresh']:.0%}",
                          secondary_y=True)
            fig.update_yaxes(title_text="Pengeluaran (Rp)", tickformat=",.0f", secondary_y=False)
            fig.update_yaxes(title_text="Risiko %", ticksuffix="%",
                             range=[0,200], showgrid=False, secondary_y=True)
            fig_style(fig, 360)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Tidak ada data untuk filter yang dipilih.")

    with cr:
        section("🏥 Financial Health", f"Skor: {hs}/100 · {hl}")

        # Gauge
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=hs,
            number={"font":{"size":34,"family":"Inter","color":hc},"suffix":"/100"},
            gauge={
                "axis":{"range":[0,100],"tickfont":{"size":9,"color":C["muted"]},
                        "tickcolor":C["border"]},
                "bar": {"color":hc,"thickness":0.26},
                "bgcolor":C["bg"],
                "borderwidth":0,
                "steps":[
                    {"range":[0,40],  "color":"#FEF2F2"},
                    {"range":[40,60], "color":"#FFFBEB"},
                    {"range":[60,80], "color":"#EFF6FF"},
                    {"range":[80,100],"color":"#ECFDF5"},
                ],
                "threshold":{"line":{"color":hc,"width":3},"thickness":.8,"value":hs},
            },
        ))
        fig_g.update_layout(height=200, margin=dict(t=20,b=0,l=20,r=20),
                            paper_bgcolor=C["surface"],
                            font=dict(family="Inter"))
        st.plotly_chart(fig_g, use_container_width=True)

        for lbl, val in [
            ("Cashflow Balance",   max(0,100-deficit*100)),
            ("Spending Control",   max(0,100-rw_avg*100)),
            ("Savings Discipline", min(100,sav_rate*200)),
        ]:
            clr = C["success"] if val>=60 else C["warning"] if val>=40 else C["danger"]
            st.markdown(f"""
            <div style="margin-bottom:.55rem">
              <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:2px">
                <span style="color:{C['muted']};font-weight:500">{lbl}</span>
                <span style="color:{clr};font-weight:600">{val:.0f}</span>
              </div>
              <div class="hbar"><div class="hfill" style="width:{min(val,100):.0f}%;background:{clr}"></div></div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='div'>", unsafe_allow_html=True)

    # ── Bottom row: category mini + total orders style ──
    cb, cr2 = st.columns([1, 1.5], gap="medium")

    with cb:
        section("📋 Quick Stats")
        rows = [
            ("Avg Daily Spend", fmt_rp(avg_day)),
            ("Total Users", f"{df_d['user_id'].nunique() if not df_d.empty else 0}"),
            ("Wants Ratio", f"{rw_avg:.1%}"),
            ("Savings Rate", f"{sav_rate:.1%}"),
            ("Deficit Risk", f"{deficit:.1%}"),
        ]
        html = ""
        for lbl, val in rows:
            html += f'<div class="stat-row"><span class="stat-lbl">{lbl}</span><span class="stat-val">{val}</span></div>'
        st.markdown(f'<div style="background:{C["surface"]};border:1px solid {C["border"]};border-radius:12px;padding:.75rem 1rem">{html}</div>', unsafe_allow_html=True)

        # Monthly bar mini
        if not df_d.empty:
            monthly = (df_d.groupby(df_d["tanggal_transaksi"].dt.to_period("M"))
                           ["total_pengeluaran_harian"].sum().reset_index())
            monthly["tanggal_transaksi"] = monthly["tanggal_transaksi"].astype(str)
            fig_m = go.Figure(go.Bar(
                x=monthly["tanggal_transaksi"],
                y=monthly["total_pengeluaran_harian"] / 1e6,
                marker_color=C["primary"],
                marker_line_width=0,
            ))
            fig_m.update_layout(title="Spending Bulanan (Jt Rp)",
                                yaxis_title="Juta Rp", xaxis_title="")
            fig_style(fig_m, 220)
            st.plotly_chart(fig_m, use_container_width=True)

    with cr2:
        section("📦 Total Orders / Monthly Spend")
        if not df_d.empty:
            # Replicate "total orders" bar chart from ref image with green bg
            monthly2 = (df_d.groupby(df_d["tanggal_transaksi"].dt.to_period("M"))
                            ["total_pengeluaran_harian"].sum().reset_index())
            monthly2["tanggal_transaksi"] = monthly2["tanggal_transaksi"].astype(str)
            monthly2["label"] = monthly2["tanggal_transaksi"].str[-7:-5].map(
                {"01":"Jan","02":"Feb","03":"Mar","04":"Apr","05":"Mei","06":"Jun",
                 "07":"Jul","08":"Agu","09":"Sep","10":"Okt","11":"Nov","12":"Des"}
            ).fillna(monthly2["tanggal_transaksi"])

            fig_to = go.Figure(go.Bar(
                x=monthly2["label"],
                y=monthly2["total_pengeluaran_harian"] / 1e6,
                marker_color=C["card_hdr"],
                marker_line_width=0,
                width=0.45,
            ))
            fig_to.update_layout(
                title="",
                paper_bgcolor="#C8E6C9",   # green bg (ref image style)
                plot_bgcolor="#C8E6C9",
                yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,.08)",
                           tickfont=dict(size=10,color="#2D5016"), zeroline=False),
                xaxis=dict(tickfont=dict(size=10,color="#2D5016"),
                           showgrid=False, zeroline=False),
                margin=dict(l=6,r=6,t=12,b=6),
                height=300,
                font=dict(family="Inter"),
            )
            # Green title bar like ref image
            st.markdown(f"""
            <div style="background:{C['card_hdr']};color:#fff;font-size:12px;
                        font-weight:600;padding:.55rem 1rem;border-radius:10px 10px 0 0;
                        text-transform:uppercase;letter-spacing:.06em">
              📦 TOTAL PENGELUARAN BULANAN
            </div>""", unsafe_allow_html=True)
            st.plotly_chart(fig_to, use_container_width=True)

    st.markdown("<hr class='div'>", unsafe_allow_html=True)

    # ── AI Insights ──
    section("🤖 AI-Generated Insights", "Berdasarkan filter aktif")
    insights = []
    if deficit > filters["risk_thresh"]:
        insights.append(("🚨","Risiko Defisit Tinggi",
                          f"Terdeteksi {deficit:.1%} hari berisiko, melampaui threshold {filters['risk_thresh']:.0%}.",C["danger"]))
    if avg_day > limit:
        insights.append(("💸","Melampaui Spending Limit",
                          f"Avg {fmt_rp(avg_day)}/hari melampaui limit {fmt_rp(limit)}.",C["warning"]))
    if rw_avg > WANTS_ALERT:
        insights.append(("⚠️","Wants Ratio Kritis",
                          f"Rasio {rw_avg:.1%} melampaui batas 50%.",C["warning"]))
    if sav_rate > SAVINGS_TARGET:
        insights.append(("✨","Tabungan Kuat",
                          f"{sav_rate:.1%} melampaui target 10%.",C["success"]))
    if not insights:
        insights.append(("✅","Posisi Stabil",
                          f"Tidak ada flag kritis. Health: {hs}/100.",C["success"]))

    cols = st.columns(min(len(insights), 2), gap="small")
    for i,(ic,ti,tx,clr) in enumerate(insights):
        with cols[i%len(cols)]:
            st.markdown(f"""
            <div class="insight-card" style="border-left:3px solid {clr};border-radius:0 10px 10px 0">
              <span style="font-size:20px">{ic}</span>
              <div>
                <div class="insight-title">{ti}</div>
                <div class="insight-body">{tx}</div>
              </div>
            </div>""", unsafe_allow_html=True)

    close_wrap()


# ══════════════════════════════════════════════════════════════
#  PAGE 2: SPENDING OVERVIEW
# ══════════════════════════════════════════════════════════════
def page_spending(df_daily, df_trx, filters, model_ok):
    topnav("Spending Overview", model_ok)
    section("📊 Spending Overview", "Analisis kategori, tipe user, dan pola temporal")

    df_d = filter_df(df_daily, filters)
    df_t = filter_df(df_trx, filters) if not df_trx.empty else pd.DataFrame()
    df_exp = df_t[df_t["tipe_transaksi"]=="Pengeluaran"] if not df_t.empty and "tipe_transaksi" in df_t.columns else pd.DataFrame()

    tab1, tab2, tab3 = st.tabs(["📂 Per Kategori","👥 Per Tipe User","📅 Per Waktu"])

    with tab1:
        if df_exp.empty:
            st.markdown(alert("📂 Upload <b>transactions_clean_v8.csv</b>.", "i"), unsafe_allow_html=True)
        elif "kategori_srs" not in df_exp.columns:
            st.markdown(alert("Kolom <b>kategori_srs</b> tidak ditemukan.", "w"), unsafe_allow_html=True)
        else:
            c1, c2 = st.columns(2)
            with c1:
                kat = df_exp.groupby("kategori_srs")["nominal"].sum().reset_index()
                kat.columns = ["Kategori","Total"]
                fig = px.pie(kat, values="Total", names="Kategori",
                             color_discrete_sequence=C["chart"], hole=0.44)
                fig.update_traces(textposition="inside", textinfo="percent+label", textfont_size=11)
                fig.update_layout(title="Kategori SRS", showlegend=False,
                                  height=340, margin=dict(t=40,b=0,l=0,r=0),
                                  paper_bgcolor=C["surface"],font=dict(family="Inter"))
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                top = kat.sort_values("Total",ascending=False).head(10)
                fig2 = px.bar(top,x="Total",y="Kategori",orientation="h",
                              color="Total",color_continuous_scale=["#EFF6FF",C["primary"]],
                              title="Top 10 Kategori")
                fig2.update_traces(texttemplate="Rp%{x:,.0f}",textposition="outside")
                fig_style(fig2, 340)
                st.plotly_chart(fig2, use_container_width=True)
            if "needs_wants" in df_exp.columns:
                st.markdown("<hr class='div'>", unsafe_allow_html=True)
                nw = (df_exp.groupby(["kategori_srs","needs_wants"])["nominal"]
                            .sum().unstack(fill_value=0)/1e6)
                fig3 = go.Figure()
                for col_n,clr in [("Needs",C["success"]),("Wants",C["danger"])]:
                    if col_n in nw.columns:
                        fig3.add_trace(go.Bar(name=col_n,x=nw.index,y=nw[col_n],marker_color=clr))
                fig3.update_layout(barmode="stack",title="Needs vs Wants (Jt Rp)",xaxis_tickangle=-30)
                fig_style(fig3, 300)
                st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        if df_d.empty or "tipe_user" not in df_d.columns:
            st.info("Data tipe user tidak tersedia.")
        else:
            agg = (df_d.groupby("tipe_user")
                       .agg(avg_spend=("total_pengeluaran_harian","mean"),
                            deficit_risk=("label_defisit_besok","mean"),
                            avg_wants=("rasio_wants","mean"))
                       .reset_index())
            c1,c2 = st.columns(2)
            with c1:
                fig = px.bar(agg.sort_values("avg_spend"),x="avg_spend",y="tipe_user",
                             orientation="h",color="avg_spend",
                             color_continuous_scale=["#EFF6FF",C["primary"]],
                             title="Avg Daily Spending")
                fig.add_vline(x=filters["spend_limit"],line_dash="dot",
                              line_color=C["danger"],
                              annotation_text=f"Limit {fmt_rp(filters['spend_limit'])}")
                fig.update_traces(texttemplate="Rp%{x:,.0f}",textposition="outside")
                fig_style(fig, 340)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig2 = px.scatter(agg,x="avg_wants",y="deficit_risk",size="avg_spend",
                                  color="tipe_user",color_discrete_sequence=C["chart"],
                                  title="Profil Risiko")
                fig2.add_hline(y=filters["risk_thresh"],line_dash="dash",line_color=C["warning"],
                               annotation_text=f"Threshold {filters['risk_thresh']:.0%}")
                fig_style(fig2, 340)
                st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        if df_d.empty:
            st.info("Tidak ada data.")
        else:
            c1,c2 = st.columns(2)
            with c1:
                fm = {2:"Tgl Muda",1:"Tengah",0:"Tgl Tua"}
                fd = df_d.groupby("fase_bulan_encoded")["total_pengeluaran_harian"].mean().rename(index=fm)
                fig = px.bar(x=fd.index,y=fd.values,color=fd.values,
                             color_continuous_scale=["#6EE7B7","#F59E0B","#EF4444"],
                             title="Payday Effect")
                fig.add_hline(y=filters["spend_limit"],line_dash="dot",line_color=C["danger"],
                              annotation_text=f"Limit {fmt_rp(filters['spend_limit'])}")
                fig.update_coloraxes(showscale=False)
                fig_style(fig, 300)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                dm = {0:"Sen",1:"Sel",2:"Rab",3:"Kam",4:"Jum",5:"Sab",6:"Min"}
                dd = df_d.groupby("hari_dalam_minggu")["total_pengeluaran_harian"].mean().rename(index=dm)
                clrs = [C["danger"] if i>=5 else C["primary"] for i in range(7)]
                fig2 = px.bar(x=dd.index,y=dd.values,color=pd.Series(clrs),
                              color_discrete_map="identity",title="Weekend Effect")
                fig_style(fig2, 300)
                st.plotly_chart(fig2, use_container_width=True)

    close_wrap()


# ══════════════════════════════════════════════════════════════
#  PAGE 3: LSTM PREDICTION
# ══════════════════════════════════════════════════════════════
def page_lstm(df_daily, filters, model, model_ok, model_msg, scalers, scalers_ok):
    topnav("LSTM Prediction", model_ok)
    engine_clr = "#6EE7B7" if model_ok else "#FDE68A"
    st.markdown(f"""
    <div class="lstm-banner">
      <div style="display:inline-block;background:rgba(79,110,247,.3);color:{engine_clr};
                  font-size:10px;font-weight:600;padding:3px 14px;border-radius:20px;margin-bottom:10px">
        {"● TF LSTM Active · "+model_msg if model_ok else "● Heuristic Mode"}
      </div>
      <h2 style="font-size:19px;font-weight:700;margin:0 0 4px">LSTM Forecasting Center</h2>
      <p style="font-size:13px;color:#CBD5E1;margin:0">
        Seq {SEQ_LEN}d · {N_FEATURES} fitur · P(defisit besok) · Threshold: {filters['risk_thresh']:.0%}
      </p>
    </div>""", unsafe_allow_html=True)

    if not model_ok:
        st.markdown(alert(f"⚠️ {model_msg}. Heuristik aktif.", "w"), unsafe_allow_html=True)
    if not scalers_ok:
        st.markdown(alert("⚠️ scalers_v8.pkl tidak ditemukan.", "w"), unsafe_allow_html=True)

    all_users = sorted(df_daily["user_id"].unique().tolist()) if not df_daily.empty else []
    uid = filters["user"]
    if uid == "All Users":
        if not all_users: st.info("Tidak ada data."); close_wrap(); return
        uid = st.selectbox("🔍 Pilih user", all_users)
    if not uid or uid == "All Users":
        st.info("Pilih user spesifik."); close_wrap(); return

    df_u = df_daily[df_daily["user_id"]==uid].sort_values("tanggal_transaksi").reset_index(drop=True)
    if df_u.empty:
        st.warning(f"Tidak ada data untuk `{uid}`."); close_wrap(); return

    with st.spinner(f"🧠 Inferensi {uid}..."):
        if model_ok:
            result    = predict_user(uid, df_daily, model, scalers)
            fc_result = forecast_30_days(uid, df_daily, model, scalers)
        else:
            result    = heuristic_predict(df_u)
            fc_result = {"error":"Model tidak tersedia"}

    if result.get("error"):
        st.error(f"Error: {result['error']}"); close_wrap(); return

    prob, status, pred = result["probability"], result["status"], result["prediction"]
    is_h = result.get("heuristic", False)

    SCFG = {
        "AMAN":    (C["success"],"🟢","alert-s","Posisi keuangan stabil."),
        "WASPADA": (C["warning"],"🟡","alert-w","Risiko moderat. Pantau pengeluaran."),
        "BAHAYA":  (C["danger"], "🔴","alert-d","Risiko tinggi! Tindakan segera."),
    }
    s_color,s_icon,s_alert,s_msg = SCFG[status]
    ring = int(prob*180)

    cg,cd = st.columns([1,2])
    with cg:
        st.markdown(f"""
        <div style="background:{C['surface']};border:1px solid {C['border']};border-radius:12px;
                    padding:1.25rem;text-align:center">
          <div style="font-size:10px;font-weight:600;color:{C['muted']};text-transform:uppercase;
                      letter-spacing:.07em;margin-bottom:10px">
            {"Heuristic" if is_h else "LSTM"} Risk Assessment
          </div>
          <div style="width:140px;height:140px;border-radius:50%;
                      background:conic-gradient({s_color} 0deg,{s_color} {ring}deg,#E2E8F0 {ring}deg);
                      display:flex;align-items:center;justify-content:center;
                      margin:0 auto 10px;box-shadow:0 0 0 8px {C['bg']}">
            <div style="width:100px;height:100px;background:#fff;border-radius:50%;
                        display:flex;align-items:center;justify-content:center;flex-direction:column">
              <div style="font-size:24px;font-weight:700;color:{s_color}">{prob:.0%}</div>
              <div style="font-size:10px;color:{C['muted']};margin-top:1px">P(defisit)</div>
            </div>
          </div>
          <div style="font-size:20px;font-weight:700;color:{s_color};margin-bottom:3px">{s_icon} {status}</div>
          <div style="font-size:11px;color:{C['muted']}">{"DEFISIT BESOK" if pred==1 else "AMAN BESOK"}</div>
          <div style="margin-top:8px;font-size:10px;color:{C['muted']};line-height:1.6">
            AMAN &lt;{THRESH_LOW:.0%} · BAHAYA &gt;{THRESH_HIGH:.0%}
          </div>
        </div>""", unsafe_allow_html=True)

    with cd:
        st.markdown(f'<div class="{s_alert}"><b>{status}:</b> {s_msg}</div>', unsafe_allow_html=True)
        RECS = {
            "AMAN":    ["✅ Disiplin saat ini efektif.","💰 Naikkan target tabungan 15–20%.","📈 Jelajahi investasi risiko rendah."],
            "WASPADA": ["⚠️ Kurangi Wants 25% minggu ini.","📊 Aktifkan notifikasi harian.","🚫 Tunda pembelian non-esensial."],
            "BAHAYA":  ["🚨 Hentikan pengeluaran diskresioner.","💳 JANGAN gunakan Paylater.","📞 Cari dana jangka pendek (bukan pinjol)."],
        }
        st.markdown(f"""
        <div style="background:{C['surface']};border:1px solid {C['border']};
                    border-radius:12px;padding:1.1rem;margin-top:.5rem">
          <div style="font-size:13px;font-weight:600;color:{C['text']};margin-bottom:.7rem">💡 Rekomendasi AI</div>""",
            unsafe_allow_html=True)
        for rec in RECS[status]:
            st.markdown(f'<div style="font-size:12px;color:{C["muted"]};padding:.2rem 0">{rec}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Metric row
    st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)
    saldo_now = float(df_u["saldo_berjalan"].iloc[-1])
    dtr_now   = float(df_u["days_to_ruin"].iloc[-1]) if "days_to_ruin" in df_u.columns else DTR_SENTINEL
    rw_now    = float(np.nan_to_num(df_u["rasio_wants"].tail(7).mean())) if "rasio_wants" in df_u.columns else 0
    r7_now    = float(np.nan_to_num(df_u["rolling_mean_7d"].iloc[-1])) if "rolling_mean_7d" in df_u.columns else 0
    dtr_lbl   = f"{int(dtr_now)}d" if dtr_now<DTR_SENTINEL else "∞"
    dtr_sub   = "🔴 Kritis" if dtr_now<DTR_CRITICAL else "🟡 Waspada" if dtr_now<DTR_WARNING else "🟢 Aman"
    dtr_clr   = C["danger"] if dtr_now<DTR_CRITICAL else C["warning"] if dtr_now<DTR_WARNING else C["success"]

    m = st.columns(4)
    with m[0]: st.markdown(kpi_card("SALDO SEKARANG",fmt_rp(saldo_now),"","",icon="💰",icon_bg=C["success"] if saldo_now>0 else C["danger"]), unsafe_allow_html=True)
    with m[1]: st.markdown(kpi_card("DAYS TO RUIN",dtr_lbl,dtr_sub,dtr_clr,icon="⏰",icon_bg=dtr_clr), unsafe_allow_html=True)
    with m[2]: st.markdown(kpi_card("WANTS RATIO 7D",f"{rw_now:.1%}","🔴 Tinggi" if rw_now>WANTS_ALERT else "🟢 OK",C["danger"] if rw_now>WANTS_ALERT else C["success"],icon="🎯",icon_bg=C["warning"]), unsafe_allow_html=True)
    with m[3]: st.markdown(kpi_card("BURN RATE 7D",fmt_rp(r7_now),"Avg harian","",icon="🔥",icon_bg=C["warning"]), unsafe_allow_html=True)

    st.markdown("<hr class='div'>", unsafe_allow_html=True)
    section("📅 30-Day Rolling Forecast", f"User: {uid} · Threshold: {filters['risk_thresh']:.0%}")

    if fc_result.get("error"):
        st.markdown(alert(f"ℹ️ {fc_result['error']}. Proyeksi linear.", "i"), unsafe_allow_html=True)
        avg_spend   = r7_now if r7_now>0 else 80_000
        avg_inc_d   = float(np.nan_to_num(df_u["total_pemasukan_harian"].mean()))
        daily_net   = avg_inc_d - avg_spend
        dates_fc    = [df_u["tanggal_transaksi"].iloc[-1]+timedelta(days=i+1) for i in range(30)]
        balances_fc = [saldo_now+daily_net*(i+1) for i in range(30)]
        probs_fc    = [float(np.clip(.5-daily_net/max(abs(avg_spend),1)*(i/30),.05,.95)) for i in range(30)]
        fc_result   = {"dates":dates_fc,"probs":probs_fc,"balances":balances_fc,
                       "n_deficit_days":sum(1 for b in balances_fc if b<0),"error":None}

    dates_fc, probs_fc = fc_result["dates"], fc_result["probs"]
    balances_fc, n_def = fc_result["balances"], fc_result["n_deficit_days"]
    thresh = filters["risk_thresh"]

    fk = st.columns(4)
    with fk[0]: st.markdown(kpi_card("SALDO 7D",fmt_rp(balances_fc[6]),"","",icon="📅",icon_bg=C["success"] if balances_fc[6]>0 else C["danger"]), unsafe_allow_html=True)
    with fk[1]: st.markdown(kpi_card("SALDO 30D",fmt_rp(balances_fc[-1]),"","",icon="📆",icon_bg=C["success"] if balances_fc[-1]>0 else C["danger"]), unsafe_allow_html=True)
    with fk[2]: st.markdown(kpi_card("AVG RISK PROB",f"{np.mean(probs_fc):.1%}","Output LSTM","",icon="🧠",icon_bg=C["danger"] if np.mean(probs_fc)>THRESH_HIGH else C["warning"]), unsafe_allow_html=True)
    with fk[3]: st.markdown(kpi_card("HARI DEFISIT",f"{n_def}/30","🔴 Tinggi" if n_def>15 else "🟢 OK","",icon="⚠️",icon_bg=C["danger"] if n_def>15 else C["success"]), unsafe_allow_html=True)

    cl2, cr2 = st.columns([1.6, 1])
    with cl2:
        hist_dt  = df_u["tanggal_transaksi"].tail(60)
        hist_bal = df_u["saldo_berjalan"].tail(60)
        fig = make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.55,.45],vertical_spacing=.06)
        fig.add_trace(go.Scatter(x=hist_dt,y=hist_bal,fill="tozeroy",
                                 fillcolor="rgba(79,110,247,.08)",
                                 line=dict(color=C["primary"],width=2),name="Saldo Historis"),row=1,col=1)
        fig.add_trace(go.Scatter(x=dates_fc,y=balances_fc,
                                 line=dict(color=C["success"],width=2.5,dash="dash"),
                                 fill="tozeroy",fillcolor="rgba(34,197,94,.07)",
                                 name="Proyeksi LSTM"),row=1,col=1)
        fig.add_hline(y=0,line=dict(color=C["danger"],dash="dot",width=1.5),row=1,col=1)
        bar_c = [C["danger"] if p>=THRESH_HIGH else C["warning"] if p>=thresh else C["success"]
                 for p in probs_fc]
        fig.add_trace(go.Bar(x=dates_fc,y=[p*100 for p in probs_fc],
                             marker_color=bar_c,name="P(defisit)%",opacity=.85),row=2,col=1)
        fig.add_hline(y=THRESH_HIGH*100,line_dash="dash",line_color=C["danger"],
                      annotation_text="BAHAYA",row=2,col=1)
        fig.add_hline(y=thresh*100,line_dash="dash",line_color=C["warning"],
                      annotation_text=f"Threshold {thresh:.0%}",row=2,col=1)
        # "Hari ini" — fix: add_shape not add_vline (avoids TypeError on datetime axis)
        today_x = pd.Timestamp(hist_dt.iloc[-1]).isoformat()
        for _r in [1,2]:
            fig.add_shape(type="line",xref="x",yref="paper",
                          x0=today_x,x1=today_x,y0=0,y1=1,
                          line=dict(color="#94A3B8",width=2,dash="dash"),row=_r,col=1)
        fig.add_annotation(x=today_x,y=1,xref="x",yref="paper",
                           text="Hari ini",showarrow=False,
                           font=dict(size=11,color="#94A3B8"),
                           xanchor="left",yanchor="top",
                           bgcolor="rgba(255,255,255,.8)",row=1,col=1)
        fig.update_yaxes(title_text="Saldo (Rp)",tickformat=",.0f",row=1,col=1)
        fig.update_yaxes(title_text="Risiko %",ticksuffix="%",range=[0,110],row=2,col=1)
        fig.update_layout(title=f"Forecast 30 Hari — {uid}",hovermode="x unified",
                          legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
        fig_style(fig, 480)
        st.plotly_chart(fig, use_container_width=True)

    with cr2:
        section("📊 7-Hari Breakdown")
        day_lbl = [pd.Timestamp(d).strftime("%a %d/%m") for d in dates_fc[:7]]
        bc7 = [C["danger"] if p>=THRESH_HIGH else C["warning"] if p>=thresh else C["success"]
               for p in probs_fc[:7]]
        fig7 = go.Figure(go.Bar(x=day_lbl,y=[p*100 for p in probs_fc[:7]],
                                marker_color=bc7,text=[f"{p:.0%}" for p in probs_fc[:7]],
                                textposition="outside"))
        fig7.add_hline(y=thresh*100,line_dash="dot",line_color=C["warning"])
        fig7.update_layout(yaxis=dict(range=[0,130]),xaxis_title="")
        fig_style(fig7, 250)
        st.plotly_chart(fig7, use_container_width=True)

        st.markdown(f"""
        <div style="background:{C['surface']};border:1px solid {C['border']};
                    border-radius:10px;padding:.875rem">
          <div style="font-size:12px;font-weight:600;color:{C['text']};margin-bottom:.7rem">
            Proyeksi Saldo
          </div>""", unsafe_allow_html=True)
        for i in [6,13,20,29]:
            dl  = pd.Timestamp(dates_fc[i]).strftime("%d %b")
            bal = balances_fc[i]
            clr = C["success"] if bal>0 else C["danger"]
            st.markdown(f"""
            <div class="stat-row">
              <span class="stat-lbl">Hari {i+1} · {dl}</span>
              <span class="stat-val" style="color:{clr}">{fmt_rp(bal)}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Engine bar
    st.markdown("<hr class='div'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:{C['card_hdr']};border-radius:10px;padding:1rem 1.5rem;
                display:grid;grid-template-columns:repeat(4,1fr);gap:1rem">
      <div style="text-align:center">
        <div style="font-size:16px;font-weight:600;color:#A5B4FC;font-family:monospace">LSTM</div>
        <div style="font-size:11px;color:#64748B;margin-top:2px">Architecture</div>
      </div>
      <div style="text-align:center">
        <div style="font-size:16px;font-weight:600;color:#6EE7B7;font-family:monospace">{SEQ_LEN}×{N_FEATURES}</div>
        <div style="font-size:11px;color:#64748B;margin-top:2px">Seq × Features</div>
      </div>
      <div style="text-align:center">
        <div style="font-size:16px;font-weight:600;color:#FDE68A;font-family:monospace">54,750</div>
        <div style="font-size:11px;color:#64748B;margin-top:2px">Training Obs</div>
      </div>
      <div style="text-align:center">
        <div style="font-size:16px;font-weight:600;color:{'#6EE7B7' if model_ok else '#FDE68A'};font-family:monospace">
          {'✅ Active' if model_ok else '⚡ Fallback'}
        </div>
        <div style="font-size:11px;color:#64748B;margin-top:2px">Engine</div>
      </div>
    </div>""", unsafe_allow_html=True)
    close_wrap()


# ══════════════════════════════════════════════════════════════
#  PAGE 4–8: identical logic to v10, updated visual wrapping
# ══════════════════════════════════════════════════════════════
def page_risk(df_daily, filters, model, model_ok, scalers):
    topnav("Risk & Early Warning", model_ok)
    section("⚡ Risk & Early Warning",
            f"Threshold: {filters['risk_thresh']:.0%} · Limit: {fmt_rp(filters['spend_limit'])}/hari")
    df_d = filter_df(df_daily, filters)
    if df_d.empty:
        st.warning("Tidak ada data."); close_wrap(); return

    c1,c2 = st.columns(2)
    with c1:
        if "days_to_ruin" in df_d.columns:
            dtr  = df_d[df_d["days_to_ruin"]<DTR_SENTINEL]["days_to_ruin"]
            nc   = int((dtr<=DTR_CRITICAL).sum())
            nw   = int(((dtr>DTR_CRITICAL)&(dtr<=DTR_WARNING)).sum())
            ns   = int((dtr>DTR_WARNING).sum())
            fig  = go.Figure()
            fig.add_trace(go.Histogram(x=dtr,nbinsx=40,marker_color=C["primary"],opacity=.8))
            fig.add_vline(x=DTR_CRITICAL,line_dash="dash",line_color=C["danger"],
                          annotation_text=f"🔴 {DTR_CRITICAL}d")
            fig.add_vline(x=DTR_WARNING, line_dash="dash",line_color=C["warning"],
                          annotation_text=f"🟡 {DTR_WARNING}d")
            fig.update_layout(title="Distribusi Days to Ruin",xaxis_title="Hari")
            fig_style(fig,300); st.plotly_chart(fig, use_container_width=True)
            m3 = st.columns(3)
            with m3[0]: st.markdown(kpi_card("KRITIS",f"{nc:,}",f"≤{DTR_CRITICAL}d","",icon="🔴",icon_bg=C["danger"]),unsafe_allow_html=True)
            with m3[1]: st.markdown(kpi_card("WASPADA",f"{nw:,}","","",icon="🟡",icon_bg=C["warning"]),unsafe_allow_html=True)
            with m3[2]: st.markdown(kpi_card("AMAN",f"{ns:,}","","",icon="🟢",icon_bg=C["success"]),unsafe_allow_html=True)
    with c2:
        if "doom_spending_harian" in df_d.columns:
            tl = (df_d.groupby("tanggal_transaksi")
                      .agg(doom=("doom_spending_harian","sum"),shock=("shock_harian","sum"))
                      .reset_index())
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=tl["tanggal_transaksi"],y=tl["doom"]/1000,marker_color=C["warning"],name="Doom (K)"))
            fig2.add_trace(go.Bar(x=tl["tanggal_transaksi"],y=tl["shock"]/1000,marker_color=C["danger"],name="Shock (K)"))
            fig2.add_hline(y=filters["spend_limit"]/1000,line_dash="dot",line_color=C["primary"],
                           annotation_text=f"Limit {fmt_rp(filters['spend_limit'])}")
            fig2.update_layout(barmode="stack",title="Doom Spending vs Shock")
            fig_style(fig2,300); st.plotly_chart(fig2, use_container_width=True)

    if "tipe_user" in df_d.columns:
        st.markdown("<hr class='div'>", unsafe_allow_html=True)
        section("🗓️ Risk Heatmap: Tipe User × Fase Bulan")
        pivot = (df_d.groupby(["tipe_user","fase_bulan_encoded"])["label_defisit_besok"]
                     .mean().unstack())
        pivot.columns = ["Tgl Tua","Tengah","Tgl Muda"]
        fig3 = px.imshow(pivot,text_auto=".1%",color_continuous_scale="RdYlGn_r",
                         title="Deficit Risk",aspect="auto",zmin=0,zmax=1)
        fig3.update_layout(height=340,margin=dict(t=40,b=0,l=0,r=0),
                           paper_bgcolor=C["surface"],font=dict(family="Inter"))
        st.plotly_chart(fig3, use_container_width=True)

    if filters["user"]=="All Users" and model_ok:
        st.markdown("<hr class='div'>", unsafe_allow_html=True)
        section("🏮 Live LSTM Watchlist","Batch inference top-15")
        with st.spinner("Batch inference..."):
            uids = (df_daily.groupby("user_id")["label_defisit_besok"]
                            .mean().sort_values(ascending=False).head(15).index.tolist())
            seqs, vids = [], []
            for u in uids:
                du = df_daily[df_daily["user_id"]==u].sort_values("tanggal_transaksi")
                if du.empty: continue
                if any(c not in du.columns for c in FEATURE_COLS): continue
                Xr = du[FEATURE_COLS].fillna(0).values.astype(np.float32)
                Xs,_ = _scale_features(u,Xr,scalers)
                seqs.append(_build_seq(Xs)[0]); vids.append(u)
            if seqs:
                pb = model.predict(np.array(seqs),verbose=0).flatten()
                rows=[]
                for u,p in zip(vids,pb):
                    p=float(np.clip(p,0,1))
                    st2="BAHAYA" if p>=THRESH_HIGH else "WASPADA" if p>=filters["risk_thresh"] else "AMAN"
                    du=df_daily[df_daily["user_id"]==u]
                    rows.append({"User":u,"Tipe":du["tipe_user"].iloc[0] if "tipe_user" in du.columns else "N/A",
                                 "P(defisit)":f"{p:.1%}","Status":st2,
                                 "Saldo":fmt_rp(float(du["saldo_berjalan"].iloc[-1])),
                                 "DTR":f"{int(du['days_to_ruin'].iloc[-1])}d" if "days_to_ruin" in du.columns else "N/A"})
                st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    close_wrap()


def page_behavior(df_daily, filters, model_ok):
    topnav("Behavior Analysis", model_ok)
    section("📈 Behavior Analysis","Psikologi pengeluaran · momentum")
    df_d = filter_df(df_daily, filters)
    if df_d.empty: st.warning("Tidak ada data."); close_wrap(); return
    c1,c2 = st.columns(2)
    with c1:
        fig = px.histogram(df_d,x="rasio_wants",nbins=40,color_discrete_sequence=[C["primary"]],title="Distribusi Wants Ratio")
        fig.add_vline(x=.5,line_dash="dash",line_color=C["danger"],annotation_text="50%")
        fig_style(fig,290); st.plotly_chart(fig, use_container_width=True)
    with c2:
        samp = df_d.sample(min(3000,len(df_d)),random_state=42)
        fig2 = px.scatter(samp,x="rasio_wants",y="saldo_berjalan",color="label_defisit_besok",
                          color_discrete_map={0:C["success"],1:C["danger"]},opacity=.4,title="Wants vs Saldo")
        fig_style(fig2,290); st.plotly_chart(fig2, use_container_width=True)
    st.markdown("<hr class='div'>", unsafe_allow_html=True)
    c3,c4 = st.columns(2)
    with c3:
        if "pengeluaran_lag_1d" in df_d.columns:
            s2 = df_d.sample(min(2000,len(df_d)),random_state=42)
            fig3 = px.scatter(s2,x="pengeluaran_lag_1d",y="total_pengeluaran_harian",
                              color="label_defisit_besok",color_discrete_map={0:C["success"],1:C["danger"]},
                              opacity=.4,title="Momentum: Kemarin vs Hari Ini")
            mx = float(df_d["total_pengeluaran_harian"].max())
            fig3.add_shape(type="line",x0=0,y0=0,x1=mx,y1=mx,line=dict(color=C["muted"],dash="dash",width=1))
            fig_style(fig3,290); st.plotly_chart(fig3, use_container_width=True)
    with c4:
        if "rolling_std_7d" in df_d.columns:
            vol = df_d.groupby("tanggal_transaksi")["rolling_std_7d"].mean().reset_index()
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=vol["tanggal_transaksi"],y=vol["rolling_std_7d"],
                                      fill="tozeroy",fillcolor="rgba(239,68,68,.08)",
                                      line=dict(color=C["danger"],width=2)))
            fig4.update_layout(title="Volatilitas 7 Hari")
            fig_style(fig4,290); st.plotly_chart(fig4, use_container_width=True)
    close_wrap()


def page_category(df_trx, filters, model_ok):
    topnav("Category Deep Dive", model_ok)
    section("🔍 Category Deep Dive","Granular per kategori")
    df_t   = filter_df(df_trx, filters) if not df_trx.empty else pd.DataFrame()
    df_exp = df_t[df_t["tipe_transaksi"]=="Pengeluaran"] if not df_t.empty and "tipe_transaksi" in df_t.columns else pd.DataFrame()
    if df_exp.empty:
        st.markdown(alert("Upload <b>transactions_clean_v8.csv</b>.","i"),unsafe_allow_html=True); close_wrap(); return
    if "kategori" not in df_exp.columns:
        st.markdown(alert("Kolom <b>kategori</b> tidak ditemukan.","w"),unsafe_allow_html=True); close_wrap(); return
    agg = df_exp.groupby("kategori")["nominal"].agg(["sum","mean","count"]).reset_index()
    agg.columns = ["Kategori","Total","Rata-rata","Frekuensi"]
    agg = agg.sort_values("Total",ascending=False).head(20)
    fig = px.treemap(agg,path=["Kategori"],values="Total",color="Rata-rata",
                     color_continuous_scale="Blues",title="Treemap Kategori")
    fig.update_layout(height=400,margin=dict(t=50,b=0,l=0,r=0),
                      paper_bgcolor=C["surface"],font=dict(family="Inter"))
    st.plotly_chart(fig, use_container_width=True)
    c1,c2 = st.columns(2)
    with c1:
        fig2 = px.bar(agg.head(15),x="Total",y="Kategori",orientation="h",
                      color="Frekuensi",color_continuous_scale="Purples",title="Top 15 by Total")
        fig_style(fig2,380); st.plotly_chart(fig2, use_container_width=True)
    with c2:
        fig3 = px.scatter(agg,x="Frekuensi",y="Rata-rata",size="Total",color="Kategori",title="Frekuensi vs Avg")
        fig_style(fig3,380); st.plotly_chart(fig3, use_container_width=True)
    close_wrap()


def page_ab(model_ok):
    topnav("A/B Testing", model_ok)
    section("⚖️ A/B Testing","Early Warning System · Signifikansi statistik")
    with st.expander("📖 Desain Eksperimen"):
        st.markdown("| | Grup A Kontrol | Grup B Treatment |\n|---|---|---|\n| Intervensi | Tidak ada | Early Warning System |\n| Trigger | — | P(defisit) > 55% |")
    df_ab = load_optional("ab_testing_results.csv")
    if not df_ab.empty and "group" in df_ab.columns:
        ma = df_ab["group"].str.contains("kontrol",case=False,na=False)
        mb = df_ab["group"].str.contains("treatment",case=False,na=False)
        ra_row = df_ab[ma].iloc[0] if ma.any() else None
        rb_row = df_ab[mb].iloc[0] if mb.any() else None
        if ra_row is not None and rb_row is not None:
            ra,rb = float(ra_row["deficit_rate"]),float(rb_row["deficit_rate"])
            k = st.columns(4)
            with k[0]: st.markdown(kpi_card("CONTROL RATE",f"{ra:.1%}","Grup A","",icon="🔴",icon_bg=C["danger"]),unsafe_allow_html=True)
            with k[1]: st.markdown(kpi_card("TREATMENT RATE",f"{rb:.1%}","Grup B","",icon="🟢",icon_bg=C["success"]),unsafe_allow_html=True)
            with k[2]: st.markdown(kpi_card("ABSOLUTE Δ",f"{ra-rb:.1%}","Pengurangan","",icon="📉",icon_bg=C["primary"]),unsafe_allow_html=True)
            with k[3]: st.markdown(kpi_card("RELATIVE Δ",f"{(ra-rb)/max(abs(ra),1e-8):.1%}","vs Kontrol","",icon="✨",icon_bg=C["secondary"]),unsafe_allow_html=True)
            fig = px.bar(pd.DataFrame({"Grup":["A Kontrol","B Treatment"],"Rate":[ra*100,rb*100]}),
                         x="Grup",y="Rate",color="Grup",
                         color_discrete_sequence=[C["danger"],C["success"]],
                         text_auto=".1f",title="Hasil A/B")
            fig.update_traces(texttemplate="%{text}%",textposition="outside")
            fig_style(fig,360); st.plotly_chart(fig, use_container_width=True)
            try:
                pf = float(ra_row.get("chi2_p",1))
                st.markdown(alert(f"{'✅ p='+str(round(pf,4))+' — SIGNIFIKAN' if pf<.05 else '⚠️ p='+str(round(pf,4))+' — Belum signifikan'}",
                                  "s" if pf<.05 else "w"), unsafe_allow_html=True)
            except Exception: pass
    else:
        st.markdown(alert("Jalankan <b>05_ab_testing.py</b> lalu upload <b>ab_testing_results.csv</b>.","i"),unsafe_allow_html=True)
        demo = pd.DataFrame({"Grup":["A Kontrol","B Treatment"],"Rate":[52.3,36.6]})
        fig  = px.bar(demo,x="Grup",y="Rate",color="Grup",
                      color_discrete_sequence=[C["danger"],C["success"]],title="Demo")
        fig_style(fig,320); st.plotly_chart(fig, use_container_width=True)
    close_wrap()


def page_data(df_daily, df_trx, filters, model_ok):
    topnav("Data & Dokumentasi", model_ok)
    section("📋 Data Explorer & Dokumentasi")
    tab1,tab2,tab3 = st.tabs(["🔎 Preview","📖 Kamus","📥 Download"])
    with tab1:
        df_d = filter_df(df_daily,filters); df_t = filter_df(df_trx,filters) if not df_trx.empty else pd.DataFrame()
        st.markdown(f"**Daily Features** — {len(df_d):,} baris")
        st.dataframe(df_d.tail(100),use_container_width=True,height=300)
        if not df_t.empty:
            st.markdown(f"**Transactions** — {len(df_t):,} baris")
            st.dataframe(df_t.tail(200),use_container_width=True,height=300)
    with tab2:
        dd = load_optional("data_dictionary_v8.csv")
        if not dd.empty:
            q = st.text_input("🔍 Cari","")
            if q: dd = dd[dd.apply(lambda r:q.lower() in str(r).lower(),axis=1)]
            st.dataframe(dd,use_container_width=True,height=460)
        else:
            st.markdown("| Kolom | Tipe | Deskripsi |\n|---|---|---|\n| `total_pengeluaran_harian` | float | Total pengeluaran harian |\n| `saldo_berjalan` | float | Saldo kumulatif |\n| `rasio_wants` | float | Rasio Wants [0,1] |\n| `days_to_ruin` | int | Hari hingga saldo=0 |\n| `label_defisit_besok` | binary | TARGET |")
    with tab3:
        df_d2 = filter_df(df_daily,filters)
        c1,c2 = st.columns(2)
        with c1:
            st.download_button("📥 Daily Features",df_d2.to_csv(index=False).encode(),"daily_filtered.csv","text/csv",use_container_width=True)
        with c2:
            df_t2 = filter_df(df_trx,filters) if not df_trx.empty else pd.DataFrame()
            st.download_button("📥 Transactions",df_t2.to_csv(index=False).encode() if not df_t2.empty else b"","trx_filtered.csv","text/csv",use_container_width=True)
    close_wrap()


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main() -> None:
    with st.spinner("Memuat FinPredict AI..."):
        df_daily, errs             = load_daily()
        df_trx                     = load_transactions()
        model, model_ok, model_msg = load_model()
        scalers, scalers_ok        = load_scalers()

    if errs:
        st.error(f"**Data wajib tidak ditemukan:** {errs[0]}")
        st.info("Letakkan `daily_features_v8.csv` di folder yang sama dengan `07_dashboard.py` atau subfolder `data/`.")
        st.stop()

    filters = render_sidebar(df_daily)
    page    = filters["page"]

    if   "Executive" in page: page_executive(df_daily, df_trx, filters, model, scalers, model_ok)
    elif "Spending"  in page: page_spending(df_daily, df_trx, filters, model_ok)
    elif "LSTM"      in page: page_lstm(df_daily, filters, model, model_ok, model_msg, scalers, scalers_ok)
    elif "Risk"      in page: page_risk(df_daily, filters, model, model_ok, scalers)
    elif "Behavior"  in page: page_behavior(df_daily, filters, model_ok)
    elif "Category"  in page: page_category(df_trx, filters, model_ok)
    elif "A/B"       in page: page_ab(model_ok)
    elif "Data"      in page: page_data(df_daily, df_trx, filters, model_ok)


if __name__ == "__main__":
    main()