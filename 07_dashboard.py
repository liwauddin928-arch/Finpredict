# =============================================================================
#  FinPredict — 07_dashboard.py
#  Dashboard Analitik Interaktif (Streamlit)                       Versi 9.0
#  Capstone Coding Camp 2026 powered by DBS Foundation · Tim CC26-PSU290
# =============================================================================
#
#  TUJUAN
#  ──────
#  Menampilkan insight dan kesimpulan Data Science FinPredict secara interaktif,
#  mengikuti tugas Data Science pada Project Brief:
#    • pertanyaan bisnis terukur (Q1–Q5)        • data wrangling end-to-end
#    • EDA + explanatory analysis               • data siap model + Data Dictionary
#    • feature engineering & A/B testing        • dashboard interaktif Streamlit
#
#  HALAMAN
#  ───────
#   1. Ringkasan              KPI, 5 pertanyaan bisnis & jawabannya
#   2. Data Wrangling         gathering → assessing → cleaning → siap model
#   3. Eksplorasi Pengeluaran EDA pola waktu, kategori, perilaku   (Q1, Q2)
#   4. Risiko Defisit         kapan & siapa yang berisiko          (Q3)
#   5. Early Warning          status per pengguna + proyeksi 30 hari (Q5)
#   6. Evaluasi Model         perbandingan LSTM/GRU/XGBoost/RF     (Q4)
#   7. A/B Testing            dampak Early Warning (simulasi)      (Q5)
#   8. Data Dictionary        dokumentasi kolom & kelompok fitur
#   9. Unduh Data             ekspor CSV sesuai filter
#
#  CARA MENJALANKAN
#  ────────────────
#    pip install -r requirements.txt
#    streamlit run 07_dashboard.py
#
#  FILE DATA
#  ─────────
#  Dicari otomatis relatif terhadap lokasi skrip ini, baik pada struktur repo
#  (02_Clean_Data/, 03_Category_Mapping/, …) maupun semua file di satu folder.
#  Wajib : transactions_clean_v8.csv, daily_features_v8.csv
#  Opsional (halaman terkait menampilkan pemberitahuan bila tidak ada):
#     category_mapping_v8.csv · data_dictionary_v8.csv · split_info_v8.csv
#     model_results.csv · ab_testing_results.csv · model_xgb.json
#     scaled_full_v8.csv · raw_dirty_v8.csv
#
#  CATATAN
#  ───────
#  Dashboard TIDAK membuat data pengganti. Jika file wajib tidak ditemukan,
#  aplikasi berhenti dengan pesan yang jelas (versi lama diam-diam memakai
#  data acak sehingga grafik tampak benar padahal palsu).
# =============================================================================

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="FinPredict Analytics",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
#  1. KONFIGURASI & TOKEN DESAIN
# =============================================================================
APP_VERSION = "9.0"
TEAM_ID = "CC26-PSU290"

PRODUCTION_MODEL = "LSTM"   # model yang diintegrasikan ke aplikasi (Project Brief, bagian O)
ACCURACY_TARGET = 0.85      # Side Quest Project Brief: akurasi minimal 85%
HORIZON_DAYS = 30           # horizon proyeksi Early Warning System
DTR_BAHAYA, DTR_WASPADA = 7, 14   # ambang days_to_ruin (sesuai 02_wrangling_eda.py)
AB_EFFECT_SIZE = 0.30       # parameter simulasi di 05_ab_testing.py (ubah bila eksperimen diulang)
PEMICU_OVERSPENDING = ["Harbolnas / Flash Sale", "Doom Spending", "Cafe & Hiburan", "Gaming & Top-up"]

# Warna — gaya neo-brutalism agar senada dengan antarmuka produk FinPredict
INK, PAPER, CARD = "#0B0B0B", "#F4F3EE", "#FFFFFF"
MUTED, GRID = "#5B5B55", "#E7E5DE"
LIME, PURPLE, ORANGE = "#CCFF00", "#A855F7", "#FF8A00"
GREEN, RED, BLUE, TEAL = "#16A34A", "#EF4444", "#3B82F6", "#0EA5A5"
YELLOW, GRAY, PINK = "#EAB308", "#6B7280", "#FFC2CF"
BG_PURPLE, BG_ORANGE, BG_GREEN = "#EBD9FF", "#FFE2BF", "#C9F7D8"
PALETTE = [PURPLE, ORANGE, GREEN, BLUE, RED, TEAL, YELLOW, GRAY]

SRS_COLORS = {
    "Makanan": ORANGE, "Transportasi": BLUE, "Hiburan": PURPLE, "Tagihan": RED,
    "Lain-lain": GRAY, "Tabungan": GREEN, "Pemasukan": TEAL,
}
STATUS_COLORS = {"BAHAYA": RED, "WASPADA": ORANGE, "AMAN": GREEN}

FASE_LABEL = {2: "Tgl Muda (1–10)", 1: "Tengah Bulan (11–21)", 0: "Tgl Tua (22–31)"}
FASE_ORDER = list(FASE_LABEL.values())
HARI = ["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"]
BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]

PAGE_LABELS = [
    "🏠 Ringkasan", "🧹 Data Wrangling", "🔎 Eksplorasi Pengeluaran",
    "⚠️ Risiko Defisit", "🚨 Early Warning", "🤖 Evaluasi Model",
    "⚖️ A/B Testing", "📚 Data Dictionary", "📥 Unduh Data",
]

PLOT_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
}

# =============================================================================
#  2. CSS
# =============================================================================
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
.fp-hero, .fp-kpi, .fp-note, .fp-q, .fp-section, .fp-brand, .fp-step, .fp-status, .fp-badge,
[data-testid="stMarkdownContainer"] { font-family: 'Space Grotesk', system-ui, -apple-system, 'Segoe UI', sans-serif; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1380px; }
footer { visibility: hidden; }
section[data-testid="stSidebar"] { border-right: 3px solid @INK@; }

.fp-brand { background:@LIME@; border:3px solid @INK@; border-radius:14px; padding:12px 14px;
            box-shadow:4px 4px 0 @INK@; margin-bottom:14px; }
.fp-brand .t { font-weight:700; font-size:1.25rem; font-style:italic; letter-spacing:-.02em; color:@INK@; }
.fp-brand .s { font-size:.78rem; color:@INK@; opacity:.8; }

.fp-hero { background:@LIME@; border:3px solid @INK@; border-radius:18px; padding:20px 26px;
           box-shadow:6px 6px 0 @INK@; margin-bottom:20px; }
.fp-hero h1 { margin:0; padding:0; font-size:1.9rem; font-weight:700; font-style:italic;
              text-transform:uppercase; letter-spacing:-.02em; color:@INK@; line-height:1.15; }
.fp-hero p { margin:6px 0 0 0; color:@INK@; font-size:1rem; }
.fp-badge { display:inline-block; background:@INK@; color:@LIME@; border-radius:999px; padding:3px 12px;
            font-size:.72rem; font-weight:600; letter-spacing:.04em; margin-bottom:8px; }

.fp-section { font-size:1.15rem; font-weight:700; color:@INK@; margin:26px 0 8px 0;
              padding-bottom:4px; border-bottom:3px solid @INK@; display:inline-block; }

.fp-kpi { border:2px solid @INK@; border-radius:14px; padding:13px 16px; box-shadow:4px 4px 0 @INK@;
          height:100%; }
.fp-kpi .lbl { font-size:.7rem; text-transform:uppercase; letter-spacing:.07em; color:@MUTED@; font-weight:600; }
.fp-kpi .val { font-size:1.65rem; font-weight:700; line-height:1.2; margin-top:3px; color:@INK@; }
.fp-kpi .note { font-size:.78rem; color:@MUTED@; margin-top:2px; }

.fp-note { background:@CARD@; border:2px solid @INK@; border-left-width:9px; border-radius:10px;
           padding:10px 14px; margin:10px 0 16px 0; font-size:.93rem; line-height:1.5; color:@INK@; }
.fp-note.info { border-left-color:@PURPLE@; } .fp-note.warn { border-left-color:@ORANGE@; }
.fp-note.good { border-left-color:@GREEN@; }  .fp-note.bad  { border-left-color:@RED@; }
.fp-note .tag { font-weight:700; text-transform:uppercase; font-size:.72rem; letter-spacing:.07em; display:block; margin-bottom:2px; }

.fp-q { background:@CARD@; border:2px solid @INK@; border-radius:14px; padding:14px 16px;
        box-shadow:4px 4px 0 @INK@; height:100%; }
.fp-q .qn { display:inline-block; background:@INK@; color:@LIME@; font-weight:700; border-radius:8px;
            padding:1px 9px; font-size:.82rem; }
.fp-q .qt { font-weight:700; font-size:1rem; margin:8px 0 4px 0; color:@INK@; }
.fp-q .qm { font-size:.76rem; color:@MUTED@; margin-bottom:8px; }
.fp-q .qa { font-size:.92rem; line-height:1.45; color:@INK@; }

.fp-step { background:@CARD@; border:2px solid @INK@; border-radius:14px; padding:12px 14px;
           box-shadow:4px 4px 0 @INK@; height:100%; }
.fp-step .n { font-weight:700; font-size:.8rem; background:@PURPLE@; color:#fff; border-radius:999px;
              padding:1px 9px; display:inline-block; }
.fp-step .h { font-weight:700; margin:6px 0 2px 0; color:@INK@; }
.fp-step .d { font-size:.82rem; color:@MUTED@; line-height:1.4; }

.fp-status { border:3px solid @INK@; border-radius:14px; padding:14px 18px; box-shadow:5px 5px 0 @INK@;
             font-size:1rem; color:@INK@; margin-bottom:8px; }
.fp-status b { font-size:1.35rem; letter-spacing:.02em; }

div[data-testid="stPlotlyChart"] { background:@CARD@; border:2px solid @INK@; border-radius:14px;
    padding:6px; box-shadow:4px 4px 0 @INK@; margin-bottom:10px; }
div[data-testid="stDataFrame"] { border:2px solid @INK@; border-radius:12px; overflow:hidden; }
div[data-testid="stExpander"] { border:2px solid @INK@; border-radius:12px; background:@CARD@; }
.stButton > button, .stDownloadButton > button { border:2px solid @INK@; border-radius:10px;
    box-shadow:3px 3px 0 @INK@; font-weight:600; color:@INK@; background:@CARD@; }
.stButton > button:hover, .stDownloadButton > button:hover { background:@LIME@; border-color:@INK@; color:@INK@; }
button[data-baseweb="tab"] { font-weight:600; }
button[data-baseweb="tab"][aria-selected="true"] { background:@LIME@; border-radius:8px 8px 0 0; color:@INK@; }
</style>
"""
for _k, _v in {"@INK@": INK, "@LIME@": LIME, "@CARD@": CARD, "@MUTED@": MUTED,
               "@PURPLE@": PURPLE, "@ORANGE@": ORANGE, "@GREEN@": GREEN, "@RED@": RED}.items():
    _CSS = _CSS.replace(_k, _v)
st.markdown(_CSS, unsafe_allow_html=True)


# =============================================================================
#  3. HELPER FORMAT
# =============================================================================
def num_id(x: float, d: int = 0) -> str:
    """Format angka gaya Indonesia: 1.234.567,8"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    s = f"{x:,.{d}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def rp(x: float) -> str:
    """Rupiah ringkas: Rp 7,25 M · Rp 1,2 jt · Rp 930 rb."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    sign, a = ("-" if x < 0 else ""), abs(float(x))
    if a >= 1e12:
        return f"{sign}Rp {num_id(a / 1e12, 2)} T"
    if a >= 1e9:
        return f"{sign}Rp {num_id(a / 1e9, 2)} M"
    if a >= 1e6:
        return f"{sign}Rp {num_id(a / 1e6, 1)} jt"
    if a >= 1e3:
        return f"{sign}Rp {num_id(a / 1e3, 0)} rb"
    return f"{sign}Rp {num_id(a, 0)}"


def rp_full(x: float) -> str:
    return f"{'-' if x < 0 else ''}Rp {num_id(abs(x), 0)}"


def pct(x: float, d: int = 1) -> str:
    return "–" if x is None or pd.isna(x) else f"{num_id(x * 100, d)}%"


def pp(x: float, d: int = 1) -> str:
    """Selisih dalam poin persentase."""
    return f"{num_id(abs(x) * 100, d)} poin"


def month_labels(series: pd.Series) -> list[str]:
    return [f"{BULAN[m - 1]} {y}" for m, y in zip(series.dt.month, series.dt.year)]


def tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Ubah kolom kategorikal → string agar Plotly tidak membuat grup kosong."""
    out = df.copy()
    for c in out.columns:
        if isinstance(out[c].dtype, pd.CategoricalDtype):
            out[c] = out[c].astype(str)
    return out


def vcounts(s: pd.Series) -> pd.Series:
    """value_counts tanpa kategori yang tidak muncul."""
    vc = s.value_counts()
    return vc[vc > 0]


# =============================================================================
#  4. KOMPONEN UI
# =============================================================================
def esc(x) -> str:
    return html.escape(str(x))


def b(x) -> str:
    return f"<b>{esc(x)}</b>"


def hero(title: str, subtitle: str, badge: str | None = None) -> None:
    badge_html = f'<span class="fp-badge">{esc(badge)}</span><br>' if badge else ""
    st.markdown(
        f'<div class="fp-hero">{badge_html}<h1>{esc(title)}</h1><p>{esc(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def section(title: str, caption: str | None = None) -> None:
    st.markdown(f'<div class="fp-section">{esc(title)}</div>', unsafe_allow_html=True)
    if caption:
        st.caption(caption)


def kpi_html(label: str, value: str, note: str = "", bg: str = CARD) -> str:
    return (f'<div class="fp-kpi" style="background:{bg}"><div class="lbl">{esc(label)}</div>'
            f'<div class="val">{esc(value)}</div><div class="note">{esc(note)}</div></div>')


def kpi_row(items: list[tuple]) -> None:
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        with col:
            st.markdown(kpi_html(*it), unsafe_allow_html=True)


def callout(body_html: str, kind: str = "info", title: str | None = None) -> None:
    """Kotak insight. body_html harus HTML tepercaya (nilai dinamis di-escape pemanggil)."""
    tags = {"info": "Insight", "warn": "Perhatian", "good": "Temuan positif", "bad": "Risiko"}
    st.markdown(
        f'<div class="fp-note {kind}"><span class="tag">{esc(title or tags.get(kind, "Insight"))}</span>{body_html}</div>',
        unsafe_allow_html=True,
    )


def _supports_new_width() -> bool:
    m = re.match(r"(\d+)\.(\d+)", getattr(st, "__version__", "0.0"))
    return bool(m) and (int(m.group(1)), int(m.group(2))) >= (1, 50)


_NEW_WIDTH_API = _supports_new_width()


def call_stretch(fn, *args, **kwargs):
    """Panggil komponen Streamlit selebar kolom, kompatibel lintas versi
    (width='stretch' pada versi baru, use_container_width=True pada versi lama)."""
    options = [{"width": "stretch"}, {"use_container_width": True}]
    if not _NEW_WIDTH_API:
        options.reverse()
    for extra in options + [{}]:
        try:
            return fn(*args, **kwargs, **extra)
        except TypeError:
            continue
    return fn(*args, **kwargs)


def style_fig(fig: go.Figure, height: int = 380, title: str | None = None) -> go.Figure:
    layout = dict(
        template="plotly_white", height=height, paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(family="Space Grotesk, system-ui, sans-serif", size=13, color=INK),
        colorway=PALETTE, separators=",.", legend_title_text="",
        margin=dict(l=14, r=14, t=56 if title else 26, b=14),
    )
    if title:
        layout["title"] = dict(text=title, x=0.01, xanchor="left", font=dict(size=15, color=INK))
    fig.update_layout(**layout)
    fig.update_xaxes(linecolor=INK, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(linecolor=INK, gridcolor=GRID, zeroline=False)
    return fig


def show(fig: go.Figure, height: int = 380, title: str | None = None) -> None:
    style_fig(fig, height, title)
    st.plotly_chart(fig, theme=None, config=PLOT_CONFIG)


def show_df(df: pd.DataFrame, height: int | None = None, **kw) -> None:
    extra = {"hide_index": True, **kw}
    if height:
        extra["height"] = height
    call_stretch(st.dataframe, df, **extra)


# =============================================================================
#  5. AKSES DATA
# =============================================================================
ROOT = Path(__file__).resolve().parent

# kunci → (nama file, folder pada struktur repo)
FILE_SPECS = {
    "transactions": ("transactions_clean_v8.csv", "02_Clean_Data"),
    "daily":        ("daily_features_v8.csv", "02_Clean_Data"),
    "scaled":       ("scaled_full_v8.csv", "02_Clean_Data"),
    "raw":          ("raw_dirty_v8.csv", "01_Raw_Data"),
    "categories":   ("category_mapping_v8.csv", "03_Category_Mapping"),
    "split":        ("split_info_v8.csv", "04_Split_Info"),
    "model":        ("model_xgb.json", "05_Models"),
    "dictionary":   ("data_dictionary_v8.csv", "Data_Dictionary"),
    "model_results": ("model_results.csv", "07_Evaluation_Results"),
    "ab_results":   ("ab_testing_results.csv", "07_Evaluation_Results"),
}

# Hasil assessing raw_dirty_v8.csv (dipakai bila file mentah tidak ikut di-deploy)
RAW_SNAPSHOT = {"rows": 253525, "dup_id": 3747, "tgl_slash": 12766, "nominal_rp": 12473,
                "kategori_nan": 5026, "desk_upper": 3634, "desk_lower": 3684,
                "source": "snapshot hasil assessing"}


def find_file(key: str) -> Path | None:
    name, folder = FILE_SPECS[key]
    for base in dict.fromkeys([ROOT, Path.cwd()]):
        for sub in ("", folder, "data", "dashboard_data"):
            p = base / sub / name
            if p.is_file():
                return p
    return None


@st.cache_resource(show_spinner="Memuat data transaksi…")
def load_transactions(path: str) -> pd.DataFrame:
    """Read-only (cache_resource tidak menyalin objek) — jangan dimodifikasi."""
    cats = ["user_id", "tipe_user", "tipe_transaksi", "kategori", "kategori_srs",
            "metode_pembayaran", "needs_wants"]
    df = pd.read_csv(path, parse_dates=["tanggal_transaksi"], dtype={c: "category" for c in cats})
    df["deskripsi"] = df["deskripsi"].astype("category")
    df["hari_bulan"] = df["tanggal_transaksi"].dt.day.astype("int8")
    df["hari_minggu"] = df["tanggal_transaksi"].dt.dayofweek.astype("int8")
    df["bulan"] = df["tanggal_transaksi"].dt.to_period("M").dt.to_timestamp()
    return df


@st.cache_resource(show_spinner="Memuat fitur harian…")
def load_daily(path: str) -> pd.DataFrame:
    """Read-only — jangan dimodifikasi."""
    df = pd.read_csv(path, parse_dates=["tanggal_transaksi"],
                     dtype={"user_id": "category", "tipe_user": "category"})
    df["fase"] = pd.Categorical(df["fase_bulan_encoded"].map(FASE_LABEL), categories=FASE_ORDER, ordered=True)
    df["bulan_ts"] = df["tanggal_transaksi"].dt.to_period("M").dt.to_timestamp()
    return df


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner="Menilai kualitas data mentah…")
def assess_raw(path: str) -> dict:
    raw = pd.read_csv(path, dtype=str,
                      usecols=["transaction_id", "tanggal_transaksi", "kategori", "deskripsi", "nominal"])
    desc = raw["deskripsi"].dropna()
    return {
        "rows": int(len(raw)),
        "dup_id": int(raw["transaction_id"].duplicated().sum()),
        "tgl_slash": int(raw["tanggal_transaksi"].str.contains("/", na=False).sum()),
        "nominal_rp": int(raw["nominal"].str.contains("Rp", na=False).sum()),
        "kategori_nan": int(raw["kategori"].isna().sum()),
        "desk_upper": int((desc == desc.str.upper()).sum()),
        "desk_lower": int((desc == desc.str.lower()).sum()),
        "source": "dihitung langsung dari raw_dirty_v8.csv",
    }


@st.cache_data(show_spinner=False)
def xgb_gain_importance(model_path: str, feature_names: tuple) -> pd.DataFrame | None:
    """Total gain per fitur, dibaca langsung dari JSON model XGBoost (tanpa library xgboost)."""
    with open(model_path, encoding="utf-8") as fh:
        model = json.load(fh)
    n_feat = int(model["learner"]["learner_model_param"]["num_feature"])
    if len(feature_names) != n_feat:
        return None
    gain = np.zeros(n_feat)
    for tree in model["learner"]["gradient_booster"]["model"]["trees"]:
        left = np.asarray(tree["left_children"])
        idx = np.asarray(tree["split_indices"])
        loss = np.asarray(tree["loss_changes"], dtype=float)
        internal = left != -1
        np.add.at(gain, idx[internal], loss[internal])
    if gain.sum() <= 0:
        return None
    out = pd.DataFrame({"Fitur": list(feature_names), "Gain": gain / gain.sum()})
    return out.sort_values("Gain", ascending=False).reset_index(drop=True)


def scaled_feature_names(scaled_path: str) -> tuple:
    """Urutan fitur array model = urutan kolom sc_* pada scaled_full_v8.csv."""
    cols = pd.read_csv(scaled_path, nrows=0).columns
    return tuple(c[3:] for c in cols if c.startswith("sc_"))


def feature_group(col: str) -> str:
    """Kelompok feature engineering untuk setiap kolom daily_features_v8.csv."""
    if col in ("user_id", "tanggal_transaksi", "tipe_user"):
        return "Identitas"
    if col == "label_defisit_besok":
        return "Target"
    if re.search(r"_lag_|^rolling_", col):
        return "Lag & rolling window"
    if re.search(r"^(sin_|cos_)|hari_dalam|^bulan$|is_weekend|is_harbolnas|fase_bulan", col):
        return "Kalender & musiman"
    if col in ("days_to_ruin", "interaksi_hedon_muda", "kebiasaan_boros_harian"):
        return "Turunan risiko"
    if col in ("usia", "tipe_user_encoded"):
        return "Profil pengguna"
    if col.startswith(("saldo_", "total_pemasukan", "total_pengeluaran")):
        return "Arus kas harian"
    return "Perilaku belanja"


def status_of(dtr: float) -> str:
    if dtr <= DTR_BAHAYA:
        return "BAHAYA"
    if dtr <= DTR_WASPADA:
        return "WASPADA"
    return "AMAN"


def status_series(dtr: pd.Series, bahaya_max: float, waspada_max: float) -> pd.Categorical:
    s = np.where(dtr <= bahaya_max, "BAHAYA", np.where(dtr <= waspada_max, "WASPADA", "AMAN"))
    return pd.Categorical(s, categories=["BAHAYA", "WASPADA", "AMAN"], ordered=True)


def missing_data_screen(paths: dict) -> None:
    hero("Data belum ditemukan", "Dashboard membutuhkan dua file hasil pipeline", badge="Setup")
    st.error("File wajib tidak ditemukan, sehingga dashboard berhenti. "
             "Tidak ada data pengganti yang dibuat agar hasil analisis tidak menyesatkan.")
    rows = []
    for key in ("transactions", "daily"):
        name, folder = FILE_SPECS[key]
        rows.append({"File wajib": name, "Status": "✅ ada" if paths.get(key) else "❌ tidak ada",
                     "Lokasi yang dicari": f"{folder}/  ·  folder yang sama dengan skrip  ·  data/"})
    show_df(pd.DataFrame(rows))
    st.markdown("Jalankan pipeline `01_data_generator.py` → `02_wrangling_eda.py`, lalu letakkan hasilnya "
                "pada salah satu lokasi di atas.")


# =============================================================================
#  6. HELPER ANALISIS
# =============================================================================
def date_id(ts) -> str:
    ts = pd.Timestamp(ts)
    return f"{ts.day} {BULAN[ts.month - 1]} {ts.year}"


def goto(page: str) -> None:
    st.session_state["page"] = page


def read_optional(C: dict, key: str) -> pd.DataFrame | None:
    path = C["paths"].get(key)
    return load_csv(str(path)) if path else None


def agg_value(df: pd.DataFrame, keys: list, how: str) -> pd.DataFrame:
    g = df.groupby(keys, observed=True)["nominal"]
    s = g.sum() if how == "sum" else g.count() if how == "count" else g.mean()
    return s.rename("nilai").reset_index()


def avg_per_user_day(df: pd.DataFrame, key: str, n_users: int, key_counts: pd.Series) -> pd.Series:
    """Rata-rata belanja per pengguna per hari untuk setiap nilai `key` (mis. tanggal 1..31)."""
    tot = df.groupby(key)["nominal"].sum()
    tot.index = tot.index.astype(int)
    denom = key_counts * max(n_users, 1)
    return (tot / denom).reindex(denom.index).fillna(0.0)


def q_card(num: str, question: str, metric: str, answer_html: str) -> str:
    return (f'<div class="fp-q"><span class="qn">{esc(num)}</span><div class="qt">{esc(question)}</div>'
            f'<div class="qm">Metrik: {esc(metric)}</div><div class="qa">{answer_html}</div></div>')


# =============================================================================
#  HALAMAN 1 — RINGKASAN
# =============================================================================
def page_ringkasan(C: dict) -> None:
    tx, dy = C["tx"], C["dy"]
    ex = tx[tx["tipe_transaksi"] == "Pengeluaran"]
    inc = tx[tx["tipe_transaksi"] == "Pemasukan"]
    n_days = int(C["dy_all"].groupby("user_id", observed=True).size().max())

    hero("FinPredict Analytics",
         "Dasbor analitik prediksi arus kas & peringatan dini defisit keuangan Gen Z",
         badge=f"Dataset simulasi · {num_id(C['n_users_all'])} pengguna × {n_days} hari · "
               f"{num_id(len(C['tx_all']))} transaksi")

    callout("Aplikasi keuangan umumnya hanya <b>mencatat masa lalu</b>, sehingga pengguna muda baru sadar "
            "defisit ketika uang sudah habis. FinPredict memproyeksikan saldo dan memberi <b>peringatan dini</b> "
            "sebelum anggaran terlampaui. Halaman ini merangkum temuan data yang mendasarinya.",
            "info", "Latar belakang")

    # ── KPI ──────────────────────────────────────────────────────────────────
    exp_total = float(ex["nominal"].sum())
    wants_total = float(ex.loc[ex["needs_wants"] == "Wants", "nominal"].sum())
    doom = ex.loc[ex["kategori"] == "Doom Spending", "nominal"]
    shock = ex.loc[ex["kategori"] == "Financial Shock", "nominal"]
    risk = float(dy["label_defisit_besok"].mean())

    kpi_row([
        ("Pengguna dianalisis", num_id(C["n_users"]), f"dari {num_id(C['n_users_all'])} pengguna", BG_PURPLE),
        ("Total pengeluaran", rp(exp_total), f"{num_id(len(ex))} transaksi", CARD),
        ("Rasio Wants", pct(wants_total / exp_total) if exp_total else "–", "porsi belanja keinginan", BG_ORANGE),
        ("Hari berisiko defisit", pct(risk), "label_defisit_besok = 1", PINK),
    ])
    st.write("")
    kpi_row([
        ("Total pemasukan", rp(float(inc["nominal"].sum())), f"{num_id(len(inc))} transaksi", BG_GREEN),
        ("Doom spending", rp(float(doom.sum())), f"{num_id(len(doom))} kejadian", CARD),
        ("Financial shock", rp(float(shock.sum())), f"{num_id(len(shock))} kejadian", CARD),
        ("Transaksi Paylater", pct(float((ex["metode_pembayaran"] == "Paylater").mean())) if len(ex) else "–",
         "dari semua transaksi keluar", CARD),
    ])

    # ── 5 pertanyaan bisnis ──────────────────────────────────────────────────
    section("Pertanyaan bisnis & jawabannya",
            "Lima pertanyaan terukur yang memandu analisis. Jawaban dihitung ulang dari data sesuai filter di sidebar.")

    # Q1 — pola pengeluaran
    g = dy.groupby("fase", observed=True)["total_pengeluaran_harian"].mean()
    if len(g) >= 2 and g.min() > 0:
        a1 = (f"Belanja harian tertinggi pada {b(g.idxmax())}: {b(rp(g.max()))} — "
              f"{num_id(g.max() / g.min(), 1)}× lipat {esc(g.idxmin())}.")
    else:
        a1 = "Data fase bulan tidak cukup pada filter ini."
    # Q2 — kategori penyebab overspending
    w = (ex[ex["needs_wants"] == "Wants"].groupby("kategori", observed=True)["nominal"].sum()
         .sort_values(ascending=False))
    if len(w) >= 3 and w.sum() > 0:
        a2 = (f"3 kategori Wants terbesar: {b(w.index[0])}, {b(w.index[1])}, {b(w.index[2])} — "
              f"{b(pct(w.head(3).sum() / w.sum(), 0))} dari seluruh belanja Wants.")
    else:
        a2 = "Data belanja Wants tidak cukup pada filter ini."
    # Q3 — kapan risiko meningkat
    rt = dy.groupby("tipe_user", observed=True)["label_defisit_besok"].mean()
    rf = dy.groupby("fase", observed=True)["label_defisit_besok"].mean()
    a3 = (f"Risiko tertinggi pada tipe {b(rt.idxmax())} ({pct(rt.max())}) dan saat {b(rf.idxmax())} "
          f"({pct(rf.max())}).") if len(rt) and len(rf) else "Data tidak cukup pada filter ini."
    # Q4 — apakah AI bisa memprediksi
    mr = read_optional(C, "model_results")
    if mr is not None and {"Model", "AUC-ROC"}.issubset(mr.columns):
        best = mr.sort_values("AUC-ROC", ascending=False).iloc[0]
        a4 = f"{b(best['Model'])} terbaik (AUC {num_id(best['AUC-ROC'], 3)}). "
        prod = mr[mr["Model"] == PRODUCTION_MODEL]
        if len(prod):
            a4 += f"{esc(PRODUCTION_MODEL)} (model produksi): AUC {num_id(float(prod['AUC-ROC'].iloc[0]), 3)}."
    else:
        a4 = "model_results.csv belum tersedia."
    # Q5 — manfaat early warning
    ab = read_optional(C, "ab_results")
    if ab is not None and {"group", "deficit_rate"}.issubset(ab.columns) and len(ab) >= 2:
        ra = float(ab.loc[ab["group"].str.contains("kontrol"), "deficit_rate"].iloc[0])
        rb = float(ab.loc[ab["group"].str.contains("treatment"), "deficit_rate"].iloc[0])
        a5 = (f"Pada simulasi, defisit turun dari {b(pct(ra))} ke {b(pct(rb))} "
              f"(−{pp(ra - rb)}). <i>Bukan eksperimen pengguna nyata.</i>")
    else:
        a5 = "ab_testing_results.csv belum tersedia."
    a6 = ("Data bersifat <b>sintetis</b> (dibangkitkan oleh 01_data_generator.py) sehingga angka menggambarkan "
          "skenario, bukan perilaku pengguna sebenarnya.")

    cards = [
        ("Q1", "Bagaimana pola pengeluaran pengguna?", "rata-rata belanja harian per fase bulan", a1, "🔎 Eksplorasi Pengeluaran"),
        ("Q2", "Kategori apa penyebab overspending?", "porsi kategori terhadap belanja Wants", a2, "🔎 Eksplorasi Pengeluaran"),
        ("Q3", "Kapan risiko defisit meningkat?", "% hari berisiko per fase, hari, bulan, tipe", a3, "⚠️ Risiko Defisit"),
        ("Q4", "Apakah AI dapat memprediksi defisit?", "AUC-ROC, F1, Recall (validation set)", a4, "🤖 Evaluasi Model"),
        ("Q5", "Bagaimana early warning membantu?", "defisit rate grup kontrol vs treatment", a5, "⚖️ A/B Testing"),
    ]
    for row in (cards[:3], cards[3:]):
        cols = st.columns(3)
        for col, (num, q, metric, ans, page) in zip(cols, row):
            with col:
                st.markdown(q_card(num, q, metric, ans), unsafe_allow_html=True)
                st.button("Buka analisis →", key=f"go_{num}", on_click=goto, args=(page,))
        if row is cards[3:]:
            with cols[2]:
                st.markdown(q_card("i", "Tentang data", "sumber & cakupan", a6), unsafe_allow_html=True)
        st.write("")

    # ── Arus kas bulanan ─────────────────────────────────────────────────────
    section("Arus kas bulanan", "Total pemasukan vs pengeluaran seluruh pengguna pada filter aktif.")
    cf = tx.groupby(["bulan", "tipe_transaksi"], observed=True)["nominal"].sum().reset_index()
    cf["Bulan"] = month_labels(cf["bulan"])
    cf = tidy(cf).rename(columns={"nominal": "Nominal (Rp)", "tipe_transaksi": "Jenis"})
    fig = px.bar(cf, x="Bulan", y="Nominal (Rp)", color="Jenis", barmode="group",
                 color_discrete_map={"Pengeluaran": ORANGE, "Pemasukan": GREEN})
    fig.update_xaxes(categoryorder="array", categoryarray=list(dict.fromkeys(cf["Bulan"])))
    fig.update_yaxes(tickformat=",.0f")
    show(fig, 360)
    inc_days = float((dy["total_pemasukan_harian"] > 0).mean())
    exp_days = float((dy["total_pengeluaran_harian"] > 0).mean())
    callout(f"Pemasukan hanya muncul pada {b(pct(inc_days))} hari-pengguna (gajian atau uang saku bersifat "
            f"sporadis), sedangkan pengeluaran tercatat pada {b(pct(exp_days))} hari. Selisih ritme inilah yang "
            f"membuat saldo mudah menipis tanpa disadari.")

    with st.expander("✅ Kesesuaian dengan Project Brief — tugas Data Science"):
        show_df(pd.DataFrame([
            ("Menentukan satu solusi utama dari analisis masalah", "Ringkasan — latar belakang & solusi"),
            ("Mendefinisikan pertanyaan bisnis yang terukur", "Ringkasan — Q1–Q5 beserta metriknya"),
            ("Data wrangling end-to-end (gathering, assessing, cleaning)", "Data Wrangling"),
            ("Exploratory Data Analysis", "Eksplorasi Pengeluaran · Risiko Defisit"),
            ("Visualisasi & explanatory analysis untuk menjawab pertanyaan bisnis", "Semua halaman: grafik + kotak insight"),
            ("Dashboard interaktif Streamlit", "Filter global, tab, slider ambang, skenario what-if"),
            ("Data siap dipakai model + Data Dictionary", "Data Wrangling (kesiapan model) · Data Dictionary"),
            ("Feature engineering (Side Quest)", "Data Dictionary — kelompok fitur"),
            ("A/B Testing dengan Python (Side Quest)", "A/B Testing"),
        ], columns=["Tugas pada Project Brief", "Ditampilkan di"]))


# =============================================================================
#  HALAMAN 2 — DATA WRANGLING
# =============================================================================
def page_wrangling(C: dict) -> None:
    tx_all, dy_all = C["tx_all"], C["dy_all"]
    hero("Data Wrangling", "Dari data mentah yang kotor menjadi dataset siap analisis dan siap model",
         badge="Gathering → Assessing → Cleaning → Model-ready")

    steps = [
        ("1", "Gathering", "Data transaksi sintetis 150 pengguna Gen Z × 365 hari dibangkitkan oleh 01_data_generator.py, "
                           "sengaja disisipi 5 jenis “racun” data."),
        ("2", "Assessing", "Memeriksa tipe data, nilai kosong, duplikat, serta format tanggal & nominal pada raw_dirty_v8.csv."),
        ("3", "Cleaning", "Standardisasi, imputasi, dan deduplikasi menghasilkan transactions_clean_v8.csv."),
        ("4", "Feature engineering", "Agregasi harian per pengguna + fitur lag, rolling, kalender, dan risiko (daily_features_v8.csv)."),
        ("5", "Model-ready", "Scaling, split berbasis waktu, dan array untuk LSTM/GRU/XGBoost/Random Forest."),
    ]
    for col, (n, h, d) in zip(st.columns(5), steps):
        with col:
            st.markdown(f'<div class="fp-step"><span class="n">{n}</span><div class="h">{esc(h)}</div>'
                        f'<div class="d">{esc(d)}</div></div>', unsafe_allow_html=True)

    # ── Assessing ────────────────────────────────────────────────────────────
    raw_path = C["paths"].get("raw")
    a = assess_raw(str(raw_path)) if raw_path else RAW_SNAPSHOT
    section("Assessing — masalah kualitas pada data mentah", f"Sumber angka: {a['source']}.")
    clean_rows = len(tx_all)
    kpi_row([
        ("Baris data mentah", num_id(a["rows"]), "raw_dirty_v8.csv", CARD),
        ("Duplikat dibuang", num_id(a["dup_id"]), pct(a["dup_id"] / a["rows"]) + " dari data mentah", BG_ORANGE),
        ("Baris data bersih", num_id(clean_rows), "transactions_clean_v8.csv", BG_GREEN),
        ("Pengguna × hari", num_id(len(dy_all)), "baris fitur harian", BG_PURPLE),
    ])
    if a["rows"] - a["dup_id"] != clean_rows:
        callout(f"Baris mentah − duplikat = {b(num_id(a['rows'] - a['dup_id']))}, sedangkan data bersih berisi "
                f"{b(num_id(clean_rows))}. Pastikan file mentah dan bersih berasal dari run pipeline yang sama.",
                "warn", "Periksa konsistensi")

    issues = pd.DataFrame([
        ("Duplikat transaction_id", a["dup_id"], "Hapus duplikat (drop_duplicates)"),
        ("Format tanggal campur (DD/MM/YYYY)", a["tgl_slash"], "Standardisasi ke datetime"),
        ("Nominal berformat teks “Rp …”", a["nominal_rp"], "Konversi ke integer; outlier di-clip pada persentil-99 per kategori"),
        ("Kategori kosong (NaN)", a["kategori_nan"], "Imputasi → “Tidak Diketahui”"),
        ("Deskripsi ALLCAPS / lowercase", a["desk_upper"] + a["desk_lower"], "Normalisasi Title Case"),
    ], columns=["Masalah", "Jumlah baris", "Penanganan"])
    issues["Persen"] = issues["Jumlah baris"] / a["rows"]

    c1, c2 = st.columns([3, 2])
    with c1:
        fig = px.bar(issues.sort_values("Persen"), x="Persen", y="Masalah", orientation="h",
                     text=issues.sort_values("Persen")["Persen"].map(lambda v: pct(v, 1)),
                     color_discrete_sequence=[PURPLE])
        fig.update_xaxes(tickformat=".0%", title="% dari baris data mentah")
        fig.update_yaxes(title="")
        fig.update_traces(textposition="outside", cliponaxis=False)
        show(fig, 340, "Proporsi masalah pada data mentah")
    with c2:
        show_df(issues.assign(**{"% data": issues["Persen"].map(lambda v: pct(v, 1)),
                                 "Jumlah baris": issues["Jumlah baris"].map(num_id)})
                .drop(columns=["Persen"]), height=300)
    worst = issues.sort_values("Persen", ascending=False).iloc[0]
    callout(f"Masalah terbesar: {b(worst['Masalah'])} ({pct(worst['Persen'])} baris). Seluruh masalah dibersihkan "
            f"secara manual di 02_wrangling_eda.py, bukan memakai dataset siap pakai, sehingga tidak ada nilai "
            f"kosong maupun duplikat yang lolos ke tahap analisis.")

    # ── Validasi data bersih (dihitung langsung) ─────────────────────────────
    section("Validasi data bersih", "Dihitung langsung dari file yang sedang dimuat dashboard.")
    checks: list[tuple] = []
    n_nan = int(tx_all.isna().sum().sum())
    checks.append(("Nilai kosong pada transaksi bersih", num_id(n_nan), n_nan == 0))
    n_dup = int(tx_all["transaction_id"].duplicated().sum())
    checks.append(("Duplikat transaction_id", num_id(n_dup), n_dup == 0))
    n_bad = int((tx_all["nominal"] <= 0).sum())
    checks.append(("Nominal ≤ 0", num_id(n_bad), n_bad == 0))
    exp_tx = float(tx_all.loc[tx_all["tipe_transaksi"] == "Pengeluaran", "nominal"].sum())
    exp_dy = float(dy_all["total_pengeluaran_harian"].sum())
    checks.append(("Total pengeluaran transaksi = total fitur harian",
                   f"selisih {rp_full(exp_tx - exp_dy)}", abs(exp_tx - exp_dy) < 1))
    per_user = dy_all.groupby("user_id", observed=True).size()
    checks.append(("Setiap pengguna punya jumlah hari yang sama",
                   f"{per_user.min()}–{per_user.max()} hari", per_user.nunique() == 1))
    cmap = read_optional(C, "categories")
    if cmap is not None:
        unmapped = sorted(set(map(str, tx_all["kategori"].cat.categories)) - set(cmap["nama_kategori"]))
        checks.append(("Semua kategori ada di category_mapping_v8.csv",
                       "lengkap" if not unmapped else "tidak ada di mapping: " + "; ".join(unmapped),
                       not unmapped))
    show_df(pd.DataFrame([(c, r, "✅ Lolos" if ok else "⚠️ Perlu dicek") for c, r, ok in checks],
                         columns=["Pemeriksaan", "Hasil", "Status"]))

    # ── Kesiapan model ───────────────────────────────────────────────────────
    section("Kesiapan data untuk model",
            "Pembagian data berbasis waktu per pengguna — model tidak boleh “melihat masa depan”.")
    sp = read_optional(C, "split")
    if sp is not None and {"train_hari", "val_hari", "test_hari"}.issubset(sp.columns):
        row = sp.iloc[0]
        segs = [("Train", int(row["train_hari"]), PURPLE, "#FFFFFF"),
                ("Validation", int(row["val_hari"]), ORANGE, INK),
                ("Test", int(row["test_hari"]), GREEN, "#FFFFFF")]
        fig = go.Figure()
        for name, n, color, tcolor in segs:
            fig.add_trace(go.Bar(y=["Per pengguna"], x=[n], name=name, orientation="h", marker_color=color,
                                 text=[f"{name}: {n} hari"], textposition="inside", insidetextanchor="middle",
                                 textfont=dict(color=tcolor)))
        fig.update_layout(barmode="stack", showlegend=False)
        fig.update_yaxes(visible=False)
        fig.update_xaxes(title="Jumlah hari")
        show(fig, 170)
        if {"train_start", "train_end", "val_end", "test_end"}.issubset(sp.columns):
            st.caption(f"Train {date_id(row['train_start'])} – {date_id(row['train_end'])} · "
                       f"Validation s.d. {date_id(row['val_end'])} · Test s.d. {date_id(row['test_end'])} "
                       f"(sama untuk {num_id(len(sp))} pengguna).")
    else:
        st.info("split_info_v8.csv tidak ditemukan, visual pembagian data dilewati.")
    callout("• <b>Target</b> <code>label_defisit_besok</code> dibentuk dari kondisi saldo hari berikutnya dan "
            "dipisahkan dari matriks fitur X.<br>"
            "• <b>Split kronologis</b> (bukan acak): train → validation → test pada setiap pengguna.<br>"
            "• <b>tipe_user</b> (teks) tidak dimasukkan sebagai fitur; dipakai versi numerik "
            "<code>tipe_user_encoded</code>.", "info", "Pencegahan data leakage")


# =============================================================================
#  HALAMAN 3 — EKSPLORASI PENGELUARAN  (Q1, Q2)
# =============================================================================
HOW = {"Total (Rp)": ("sum", "Total (Rp)"),
       "Jumlah transaksi": ("count", "Jumlah transaksi"),
       "Rata-rata per transaksi (Rp)": ("mean", "Rata-rata (Rp)")}


def page_eksplorasi(C: dict) -> None:
    tx = C["tx"]
    hero("Eksplorasi Pengeluaran", "EDA pola waktu, kategori, dan perilaku belanja", badge="Q1 · Q2")
    ex = tx[tx["tipe_transaksi"] == "Pengeluaran"]

    with st.expander("🎛️ Filter analisis", expanded=True):
        f1, f2, f3 = st.columns(3)
        with f1:
            srs_opts = sorted(map(str, ex["kategori_srs"].unique()))
            srs_sel = st.multiselect("Kelompok kategori", srs_opts, default=srs_opts)
        with f2:
            met_opts = sorted(map(str, ex["metode_pembayaran"].unique()))
            met_sel = st.multiselect("Metode pembayaran", met_opts, default=met_opts)
        with f3:
            how_label = st.radio("Ukuran", list(HOW), key="eks_how")
    how, how_title = HOW[how_label]

    ex = ex[ex["kategori_srs"].isin(srs_sel) & ex["metode_pembayaran"].isin(met_sel)]
    if ex.empty:
        st.warning("Tidak ada transaksi yang cocok dengan filter analisis.")
        return

    kpi_row([
        ("Transaksi", num_id(len(ex)), "pada filter ini", CARD),
        ("Total pengeluaran", rp(float(ex["nominal"].sum())), "", BG_ORANGE),
        ("Rata-rata / transaksi", rp(float(ex["nominal"].mean())), "", CARD),
        ("Median / transaksi", rp(float(ex["nominal"].median())), "tengah sebaran nominal", CARD),
    ])
    st.write("")

    t_waktu, t_kat, t_perilaku, t_data = st.tabs(["⏱️ Pola waktu", "🏷️ Kategori", "💳 Perilaku", "📄 Data"])

    # ── Pola waktu ───────────────────────────────────────────────────────────
    with t_waktu:
        m = agg_value(ex, ["bulan"], how)
        m["Bulan"] = month_labels(m["bulan"])
        fig = px.bar(m, x="Bulan", y="nilai", color_discrete_sequence=[PURPLE])
        fig.update_xaxes(categoryorder="array", categoryarray=list(m["Bulan"]), title="")
        fig.update_yaxes(tickformat=",.0f", title=how_title)
        show(fig, 340, f"Tren bulanan — {how_title}")

        dates = pd.Series(pd.date_range(*C["date_range"]))
        n_users = C["n_users"]
        dom = avg_per_user_day(ex, "hari_bulan", n_users, dates.dt.day.value_counts().sort_index())
        dow = avg_per_user_day(ex, "hari_minggu", n_users, dates.dt.dayofweek.value_counts().sort_index())

        c1, c2 = st.columns([3, 2])
        with c1:
            fig = go.Figure(go.Scatter(x=dom.index, y=dom.values, mode="lines+markers",
                                       line=dict(color=PURPLE, width=3), marker=dict(size=6),
                                       hovertemplate="Tanggal %{x}<br>%{y:,.0f}<extra></extra>"))
            for x0, x1, lab, col in [(0.5, 10.5, "Tgl muda", LIME), (10.5, 21.5, "Tengah bulan", "#FFF3B0"),
                                     (21.5, 31.5, "Tgl tua", PINK)]:
                fig.add_vrect(x0=x0, x1=x1, fillcolor=col, opacity=0.35, line_width=0, layer="below",
                              annotation_text=lab, annotation_position="top left")
            fig.update_xaxes(title="Tanggal dalam bulan", dtick=3)
            fig.update_yaxes(tickformat=",.0f", title="Rp per pengguna per hari")
            show(fig, 340, "Payday effect — belanja rata-rata per hari")
        with c2:
            fig = px.bar(x=[HARI[i] for i in dow.index], y=dow.values, color_discrete_sequence=[ORANGE])
            fig.update_traces(marker_color=[RED if i >= 5 else ORANGE for i in dow.index])
            fig.update_xaxes(title="")
            fig.update_yaxes(tickformat=",.0f", title="Rp per pengguna per hari")
            show(fig, 340, "Hari dalam minggu")

        muda, tua = dom.loc[1:10].mean(), dom.loc[22:31].mean()
        wk, wd = dow.loc[5:6].mean(), dow.loc[0:4].mean()
        parts = []
        if tua > 0:
            parts.append(f"Belanja harian {b(rp(muda))} pada tanggal 1–10 vs {b(rp(tua))} pada tanggal 22–31 "
                         f"({num_id(muda / tua, 1)}× lipat) — pola {b('payday effect')}: uang cepat keluar setelah gajian.")
        if wd > 0:
            diff = wk / wd - 1
            parts.append(f"Akhir pekan {b(pct(abs(diff)))} {'lebih tinggi' if diff >= 0 else 'lebih rendah'} "
                         f"daripada hari kerja.")
        if parts:
            callout(" ".join(parts))

    # ── Kategori ─────────────────────────────────────────────────────────────
    with t_kat:
        tree = tidy(ex.groupby(["kategori_srs", "kategori"], observed=True)["nominal"].sum().reset_index())
        fig = px.treemap(tree, path=[px.Constant("Semua kategori"), "kategori_srs", "kategori"], values="nominal",
                         color="kategori_srs", color_discrete_map={**SRS_COLORS, "(?)": CARD})
        fig.update_traces(root_color=CARD, textinfo="label+percent root")
        show(fig, 430, "Komposisi pengeluaran (klik untuk memperbesar)")

        c1, c2 = st.columns(2)
        with c1:
            n_top = st.slider("Tampilkan N kategori teratas", 5, 20, 10, key="eks_topn")
            top = tidy(agg_value(ex, ["kategori", "kategori_srs"], how).nlargest(n_top, "nilai").sort_values("nilai"))
            fig = px.bar(top, x="nilai", y="kategori", orientation="h", color="kategori_srs",
                         color_discrete_map=SRS_COLORS)
            fig.update_xaxes(tickformat=",.0f", title=how_title)
            fig.update_yaxes(title="")
            show(fig, 120 + 30 * n_top, f"{n_top} kategori teratas — {how_title}")
        with c2:
            nw = ex.groupby(["tipe_user", "needs_wants"], observed=True)["nominal"].sum().reset_index()
            nw["share"] = nw["nominal"] / nw.groupby("tipe_user", observed=True)["nominal"].transform("sum")
            nw = tidy(nw)
            order = (nw[nw["needs_wants"] == "Wants"].sort_values("share")["tipe_user"].tolist())
            fig = px.bar(nw, x="share", y="tipe_user", color="needs_wants", orientation="h",
                         color_discrete_map={"Needs": GREEN, "Wants": ORANGE},
                         category_orders={"tipe_user": order, "needs_wants": ["Needs", "Wants"]})
            fig.update_xaxes(tickformat=".0%", title="Porsi belanja")
            fig.update_yaxes(title="")
            show(fig, 120 + 30 * 10, "Needs vs Wants per tipe pengguna")

        total = ex["nominal"].sum()
        by_cat = ex.groupby("kategori", observed=True)["nominal"].sum().sort_values(ascending=False)
        wants_by_cat = (ex[ex["needs_wants"] == "Wants"].groupby("kategori", observed=True)["nominal"].sum()
                        .sort_values(ascending=False))
        msg = (f"{b(by_cat.index[0])} adalah pos terbesar ({pct(by_cat.iloc[0] / total)} dari total nominal).")
        if len(wants_by_cat) >= 3:
            msg += (f" Untuk belanja keinginan (Wants), tiga teratas adalah {b(wants_by_cat.index[0])}, "
                    f"{b(wants_by_cat.index[1])}, dan {b(wants_by_cat.index[2])} "
                    f"({pct(wants_by_cat.head(3).sum() / wants_by_cat.sum(), 0)} dari seluruh Wants).")
        wsh = nw[nw["needs_wants"] == "Wants"].sort_values("share", ascending=False)
        if len(wsh):
            msg += f" Porsi Wants tertinggi ada pada {b(wsh.iloc[0]['tipe_user'])} ({pct(wsh.iloc[0]['share'])})."
        callout(msg)

    # ── Perilaku ─────────────────────────────────────────────────────────────
    with t_perilaku:
        c1, c2 = st.columns(2)
        with c1:
            mm = vcounts(ex["metode_pembayaran"])
            fig = px.pie(values=mm.values, names=mm.index.astype(str), hole=0.55,
                         color_discrete_sequence=PALETTE)
            fig.update_traces(textinfo="percent+label", textposition="outside")
            fig.update_layout(showlegend=False)
            show(fig, 360, "Metode pembayaran (jumlah transaksi)")
        with c2:
            pl = (ex.assign(is_paylater=(ex["metode_pembayaran"] == "Paylater"))
                  .groupby("tipe_user", observed=True)["is_paylater"].mean().sort_values())
            fig = px.bar(x=pl.values, y=pl.index.astype(str), orientation="h", color_discrete_sequence=[RED],
                         text=[pct(v) for v in pl.values])
            fig.update_xaxes(tickformat=".0%", title="Transaksi memakai Paylater")
            fig.update_yaxes(title="")
            fig.update_traces(textposition="outside", cliponaxis=False)
            show(fig, 360, "Penggunaan Paylater per tipe pengguna")

        pem = ex[ex["kategori"].isin(PEMICU_OVERSPENDING)]
        if len(pem):
            n_per_tipe = tx.groupby("tipe_user", observed=True)["user_id"].nunique()
            n_per_tipe.index = n_per_tipe.index.astype(str)
            pt = tidy(pem.groupby(["tipe_user", "kategori"], observed=True)["nominal"].sum().reset_index())
            pt["per_pengguna"] = pt["nominal"] / pt["tipe_user"].map(n_per_tipe.to_dict())
            fig = px.bar(pt, x="tipe_user", y="per_pengguna", color="kategori", barmode="stack",
                         color_discrete_sequence=PALETTE)
            fig.update_xaxes(title="", tickangle=-15)
            fig.update_yaxes(tickformat=",.0f", title="Rp per pengguna")
            show(fig, 380, "Kategori pemicu overspending — belanja per pengguna")
            tot_t = pt.groupby("tipe_user")["per_pengguna"].sum().sort_values(ascending=False)
            top_pem = pt.groupby("kategori")["nominal"].sum().sort_values(ascending=False)
            pl_top = pl.sort_values(ascending=False)
            callout(f"Pemicu overspending terbesar: {b(top_pem.index[0])}. Tipe {b(tot_t.index[0])} menghabiskan "
                    f"{b(rp(tot_t.iloc[0]))} per pengguna pada empat kategori pemicu, dan tipe "
                    f"{b(pl_top.index[0])} paling sering memakai Paylater ({pct(pl_top.iloc[0])} transaksi).")
        else:
            st.info("Tidak ada transaksi kategori pemicu pada filter ini.")

    # ── Data ─────────────────────────────────────────────────────────────────
    with t_data:
        q = st.text_input("Cari deskripsi transaksi", "", key="eks_q")
        view = ex[ex["deskripsi"].astype(str).str.contains(q, case=False, regex=False, na=False)] if q else ex
        st.caption(f"{num_id(len(view))} transaksi cocok · tabel menampilkan 300 terbaru.")
        cols = ["tanggal_transaksi", "user_id", "tipe_user", "kategori", "deskripsi", "nominal",
                "metode_pembayaran", "needs_wants"]
        show_df(view.sort_values("tanggal_transaksi", ascending=False).head(300)[cols], height=360)
        td = vcounts(view["deskripsi"]).head(10).sort_values()
        if len(td):
            fig = px.bar(x=td.values, y=td.index.astype(str), orientation="h", color_discrete_sequence=[TEAL])
            fig.update_xaxes(title="Frekuensi")
            fig.update_yaxes(title="")
            show(fig, 380, "10 deskripsi transaksi terbanyak")


# =============================================================================
#  HALAMAN 4 — RISIKO DEFISIT  (Q3)
# =============================================================================
def page_risiko(C: dict) -> None:
    dy = C["dy"]
    hero("Risiko Defisit", "Kapan dan pada siapa risiko defisit meningkat?", badge="Q3")
    base = float(dy["label_defisit_besok"].mean())

    dim = st.radio("Dimensi analisis", ["Fase bulan", "Hari dalam minggu", "Bulan", "Tipe pengguna"],
                   horizontal=True, key="risk_dim")
    if dim == "Fase bulan":
        g = dy.groupby("fase", observed=True)["label_defisit_besok"].agg(["mean", "size"]).reset_index()
        g["label"] = g["fase"].astype(str)
    elif dim == "Hari dalam minggu":
        g = dy.groupby("hari_dalam_minggu")["label_defisit_besok"].agg(["mean", "size"]).reset_index()
        g["label"] = g["hari_dalam_minggu"].map(lambda i: HARI[int(i)])
    elif dim == "Bulan":
        g = dy.groupby("bulan_ts")["label_defisit_besok"].agg(["mean", "size"]).reset_index()
        g["label"] = month_labels(g["bulan_ts"])
    else:
        g = (dy.groupby("tipe_user", observed=True)["label_defisit_besok"].agg(["mean", "size"])
             .reset_index().sort_values("mean"))
        g["label"] = g["tipe_user"].astype(str)
    g = g[["label", "mean", "size"]].reset_index(drop=True)

    fig = px.bar(g, x="label", y="mean", text=g["mean"].map(lambda v: pct(v, 0)))
    fig.update_traces(marker_color=[RED if v > base else GREEN for v in g["mean"]],
                      textposition="outside", cliponaxis=False)
    fig.add_hline(y=base, line_dash="dash", line_color=INK,
                  annotation_text=f"Rata-rata {pct(base)}", annotation_position="top left")
    fig.update_xaxes(categoryorder="array", categoryarray=list(g["label"]), title="", tickangle=-15 if dim == "Tipe pengguna" else 0)
    fig.update_yaxes(tickformat=".0%", title="Hari berisiko defisit", range=[0, min(1.15, g["mean"].max() * 1.2 + 0.05)])
    show(fig, 380, f"Risiko defisit besok menurut {dim.lower()}")

    hi, lo = g.loc[g["mean"].idxmax()], g.loc[g["mean"].idxmin()]
    gap = float(hi["mean"] - lo["mean"])
    msg = (f"Risiko tertinggi: {b(hi['label'])} ({pct(hi['mean'])}); terendah: {b(lo['label'])} "
           f"({pct(lo['mean'])}) — selisih {b(pp(gap))}.")
    msg += (" Selisihnya kecil, jadi dimensi ini bukan pembeda risiko yang kuat."
            if gap < 0.05 else " Selisih yang lebar menandakan dimensi ini pembeda risiko yang penting.")
    callout(msg, "warn" if gap >= 0.05 else "info")

    # ── Heatmap tipe × fase ──────────────────────────────────────────────────
    section("Tipe pengguna × fase bulan", "Persentase hari berisiko defisit pada setiap kombinasi.")
    hm = dy.groupby(["tipe_user", "fase"], observed=True)["label_defisit_besok"].mean().unstack("fase")
    hm = hm.reindex(columns=[c for c in FASE_ORDER if c in hm.columns])
    hm = hm.loc[hm.mean(axis=1).sort_values().index]
    hm.index = hm.index.astype(str)
    hm.columns = [str(c) for c in hm.columns]
    fig = px.imshow(hm, text_auto=".0%", aspect="auto", zmin=0, zmax=1,
                    color_continuous_scale=[[0, "#FFFFFF"], [1, RED]])
    fig.update_layout(coloraxis_showscale=False)
    show(fig, 120 + 48 * len(hm))

    # ── Ambang Early Warning (interaktif) ────────────────────────────────────
    section("Ambang Early Warning", "Geser ambang days_to_ruin untuk melihat dampaknya pada jumlah peringatan dan ketepatannya.")
    s1, s2 = st.columns(2)
    with s1:
        bahaya = st.slider("Batas BAHAYA — days_to_ruin ≤ (hari)", 0, 30, DTR_BAHAYA, key="risk_bahaya")
    with s2:
        waspada = st.slider("Batas WASPADA — days_to_ruin ≤ (hari)", 0, 60, DTR_WASPADA, key="risk_waspada")
    waspada = max(waspada, bahaya)

    tmp = pd.DataFrame({"status": status_series(dy["days_to_ruin"], bahaya, waspada),
                        "label": dy["label_defisit_besok"].to_numpy()})
    summ = tmp.groupby("status", observed=True)["label"].agg(["size", "mean"]).reset_index()
    summ["share"] = summ["size"] / len(tmp)
    summ = tidy(summ)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(summ, x="status", y="share", color="status", color_discrete_map=STATUS_COLORS,
                     text=summ["share"].map(lambda v: pct(v)))
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False)
        fig.update_xaxes(title="")
        fig.update_yaxes(tickformat=".0%", title="Porsi hari-pengguna", range=[0, 1.1])
        show(fig, 340, "Seberapa sering peringatan muncul")
    with c2:
        fig = px.bar(summ, x="status", y="mean", color="status", color_discrete_map=STATUS_COLORS,
                     text=summ["mean"].map(lambda v: pct(v)))
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False)
        fig.update_xaxes(title="")
        fig.update_yaxes(tickformat=".0%", title="Benar defisit keesokan hari", range=[0, 1.1])
        show(fig, 340, "Seberapa tepat peringatan itu")

    d = summ.set_index("status")
    bits = [f"{b(s)}: muncul di {pct(d.loc[s, 'share'])} hari, {pct(d.loc[s, 'mean'])} di antaranya benar defisit "
            f"keesokan hari" for s in ("BAHAYA", "WASPADA", "AMAN") if s in d.index]
    callout("; ".join(bits) + ". Ambang yang lebih longgar menangkap lebih banyak risiko tetapi menambah jumlah "
            "peringatan — pertimbangan utama untuk menghindari <i>notification fatigue</i>.")

    dtr = dy.loc[(dy["days_to_ruin"] > 0) & (dy["days_to_ruin"] < 9999), "days_to_ruin"].clip(upper=120)
    if len(dtr):
        fig = px.histogram(x=dtr, nbins=60, color_discrete_sequence=[PURPLE])
        fig.add_vline(x=bahaya, line_dash="dash", line_color=RED, annotation_text="Bahaya")
        fig.add_vline(x=waspada, line_dash="dash", line_color=ORANGE, annotation_text="Waspada")
        fig.update_xaxes(title="days_to_ruin (hari, dipotong di 120)")
        fig.update_yaxes(title="Jumlah hari-pengguna")
        show(fig, 320, "Sebaran days_to_ruin (hanya saldo positif)")
        st.caption(f"{pct(float((dy['days_to_ruin'] <= 0).mean()))} hari-pengguna sudah berada di saldo ≤ 0 "
                   f"(days_to_ruin = 0) dan tidak ditampilkan pada histogram.")


# =============================================================================
#  HALAMAN 5 — EARLY WARNING PER PENGGUNA  (Q5)
# =============================================================================
STATUS_INFO = {
    "BAHAYA": (RED, "#FFD6D6", f"Saldo sudah ≤ 0 atau diperkirakan habis dalam ≤ {DTR_BAHAYA} hari.",
               ["Pangkas belanja Wants minimal 50% pekan ini.",
                "Tunda Harbolnas/Flash Sale dan top-up game.",
                "Prioritaskan makan pokok dan transportasi.",
                "Hindari Paylater untuk menutup kekurangan."]),
    "WASPADA": (ORANGE, BG_ORANGE, f"Saldo diperkirakan habis dalam {DTR_BAHAYA + 1}–{DTR_WASPADA} hari.",
                ["Kurangi belanja Wants sekitar 30%.",
                 "Pantau saldo setiap hari.",
                 "Tunda pembelian non-esensial.",
                 "Cari tambahan pemasukan bila memungkinkan."]),
    "AMAN": (GREEN, BG_GREEN, f"Saldo diperkirakan cukup lebih dari {DTR_WASPADA} hari.",
             ["Pertahankan pola belanja saat ini.",
              "Sisihkan minimal 10% pemasukan untuk tabungan.",
              "Bangun dana darurat sebelum berinvestasi."]),
}


def page_early_warning(C: dict) -> None:
    dy, tx = C["dy"], C["tx"]
    hero("Early Warning", "Status keuangan per pengguna dan proyeksi saldo 30 hari ke depan", badge="Q5")

    users = sorted(map(str, dy["user_id"].unique()))
    utype = (dy.drop_duplicates("user_id").assign(user_id=lambda d: d["user_id"].astype(str))
             .set_index("user_id")["tipe_user"].astype(str).to_dict())
    uid = st.selectbox("Pilih pengguna", users, format_func=lambda x: f"{x} · {utype.get(x, '')}")
    u = dy[dy["user_id"] == uid].sort_values("tanggal_transaksi")
    if len(u) < 7:
        st.info("Riwayat pengguna kurang dari 7 hari pada rentang tanggal ini.")
        return

    last = u.iloc[-1]
    last_date = pd.Timestamp(last["tanggal_transaksi"])
    saldo, dtr = float(last["saldo_berjalan"]), float(last["days_to_ruin"])
    status = status_of(dtr)
    color, bg, status_text, tips = STATUS_INFO[status]
    w7 = u.tail(7)
    wants7 = float(w7["total_wants_harian"].sum() / max(w7["total_pengeluaran_harian"].sum(), 1))

    st.markdown(f'<div class="fp-status" style="background:{bg}"><b style="color:{color}">{status}</b> '
                f'· per {esc(date_id(last_date))}<br>{esc(status_text)}</div>', unsafe_allow_html=True)
    kpi_row([
        ("Saldo kumulatif", rp(saldo), "pemasukan − pengeluaran sejak awal periode", BG_GREEN if saldo > 0 else PINK),
        ("Days to ruin", "0 hari (defisit)" if saldo <= 0 else f"{num_id(dtr)} hari", "saldo ÷ rata-rata belanja 7 hari", CARD),
        ("Rasio Wants 7 hari", pct(wants7), "porsi belanja keinginan", BG_ORANGE),
        ("Rata-rata belanja 7 hari", rp(float(last["rolling_mean_7d"])), "per hari", CARD),
    ])

    # ── Skenario what-if ─────────────────────────────────────────────────────
    section("Proyeksi 30 hari & skenario",
            "Proyeksi linier sederhana dari rata-rata historis — bukan keluaran model LSTM. "
            "Gunakan untuk membandingkan skenario penghematan.")
    w30, w90 = u.tail(30), u.tail(90)
    base_exp = float(w30["total_pengeluaran_harian"].mean())
    wants_share = float(w30["total_wants_harian"].sum() / max(w30["total_pengeluaran_harian"].sum(), 1))
    inc_hist = float(w90["total_pemasukan_harian"].mean())
    default_budget = int(round((inc_hist * HORIZON_DAYS if inc_hist > 0 else base_exp * HORIZON_DAYS) / 100_000) * 100_000)

    s1, s2, s3 = st.columns(3)
    with s1:
        cut = st.slider("Kurangi belanja Wants (%)", 0, 60, 30, 5, key="ew_cut") / 100
    with s2:
        inc_mode = st.radio("Asumsi pemasukan", ["Rata-rata 90 hari terakhir", "Tanpa pemasukan baru"], key="ew_inc")
    with s3:
        budget = st.number_input("Anggaran 30 hari (Rp)", min_value=0, value=max(default_budget, 0),
                                 step=100_000, key=f"ew_budget_{uid}")
    inc_daily = inc_hist if inc_mode.startswith("Rata-rata") else 0.0
    exp_cut = base_exp * (1 - wants_share * cut)
    days = np.arange(1, HORIZON_DAYS + 1)
    future = pd.date_range(last_date + pd.Timedelta(days=1), periods=HORIZON_DAYS)
    s_base = saldo + days * (inc_daily - base_exp)
    s_cut = saldo + days * (inc_daily - exp_cut)

    if len(u) > 30:
        hist = st.slider("Riwayat yang ditampilkan (hari)", 30, len(u), min(90, len(u)), key="ew_hist")
    else:
        hist = len(u)
    h = u.tail(hist)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h["tanggal_transaksi"], y=h["saldo_berjalan"], name="Saldo historis",
                             mode="lines", line=dict(color=INK, width=3)))
    fig.add_trace(go.Scatter(x=future, y=s_base, name="Proyeksi (pola saat ini)", mode="lines",
                             line=dict(color=ORANGE, width=3, dash="dash")))
    if cut > 0:
        fig.add_trace(go.Scatter(x=future, y=s_cut, name=f"Proyeksi (Wants −{int(cut * 100)}%)", mode="lines",
                                 line=dict(color=GREEN, width=3, dash="dash")))
    fig.add_hline(y=0, line_dash="dot", line_color=RED, annotation_text="Rp 0", annotation_position="bottom left")
    fig.add_shape(type="line", x0=last_date, x1=last_date, y0=0, y1=1, yref="paper",
                  line=dict(color=GRAY, dash="dot"))
    fig.add_annotation(x=last_date, y=1, yref="paper", text="Hari ini", showarrow=False, yanchor="bottom")
    fig.update_yaxes(tickformat=",.0f", title="Saldo kumulatif (Rp)")
    fig.update_xaxes(title="")
    show(fig, 400, f"Saldo {uid}: riwayat dan proyeksi {HORIZON_DAYS} hari")

    proj_spend = base_exp * HORIZON_DAYS
    over = proj_spend / budget - 1 if budget > 0 else None
    if over is not None and over > 0:
        callout(f"Proyeksi pengeluaran {HORIZON_DAYS} hari: {b(rp_full(proj_spend))} — {b(pct(over))} di atas "
                f"anggaran {b(rp_full(budget))}. Early Warning aktif.", "bad", "Peringatan anggaran")
    elif over is not None:
        callout(f"Proyeksi pengeluaran {HORIZON_DAYS} hari: {b(rp_full(proj_spend))}, masih dalam anggaran "
                f"{b(rp_full(budget))} (sisa {b(pct(-over))}).", "good", "Anggaran aman")

    parts = []
    neg_base = np.where(s_base < 0)[0]
    if saldo <= 0:
        parts.append(f"Saldo kumulatif sudah negatif ({b(rp(saldo))}); pada pola saat ini proyeksinya "
                     f"{'membaik' if s_base[-1] > saldo else 'memburuk'} menjadi {b(rp(float(s_base[-1])))} pada hari ke-{HORIZON_DAYS}.")
    elif len(neg_base):
        parts.append(f"Pada pola saat ini saldo diproyeksikan negatif mulai {b('hari ke-' + str(neg_base[0] + 1))} "
                     f"({esc(date_id(future[neg_base[0]]))}).")
    else:
        parts.append("Pada pola saat ini saldo tetap positif selama 30 hari.")
    if cut > 0:
        parts.append(f"Memangkas Wants {b(str(int(cut * 100)) + '%')} menghemat sekitar "
                     f"{b(rp((base_exp - exp_cut) * HORIZON_DAYS))} dalam 30 hari.")
    callout(" ".join(parts))

    c1, c2 = st.columns([3, 2])
    with c1:
        recent = tx[(tx["user_id"] == uid) & (tx["tipe_transaksi"] == "Pengeluaran")
                    & (tx["tanggal_transaksi"] > last_date - pd.Timedelta(days=30))]
        top = tidy(recent.groupby("kategori", observed=True)["nominal"].sum().nlargest(6)
                   .sort_values().reset_index())
        if len(top):
            fig = px.bar(top, x="nominal", y="kategori", orientation="h", color_discrete_sequence=[PURPLE])
            fig.update_xaxes(tickformat=",.0f", title="Rp")
            fig.update_yaxes(title="")
            show(fig, 320, "Pos pengeluaran terbesar (30 hari terakhir)")
    with c2:
        st.markdown("**Rekomendasi**")
        for t in tips:
            st.markdown(f"- {t}")


# =============================================================================
#  HALAMAN 6 — EVALUASI MODEL  (Q4)
# =============================================================================
METRICS = ["AUC-ROC", "F1-Score", "Recall", "Precision", "Accuracy", "Avg Precision"]


def page_model(C: dict) -> None:
    hero("Evaluasi Model", "Perbandingan LSTM, GRU, XGBoost, dan Random Forest", badge="Q4")
    mr = read_optional(C, "model_results")
    if mr is None or "Model" not in mr.columns:
        st.warning("model_results.csv tidak ditemukan. Jalankan 04_modeling.py terlebih dahulu.")
        return
    metrics = [m for m in METRICS if m in mr.columns]
    st.caption("Sumber: model_results.csv — metrik dihitung pada **validation set** dengan threshold 0,5. "
               "Hasil test-set dicetak oleh 04_modeling.py tetapi tidak tersimpan ke CSV.")

    c1, c2 = st.columns([3, 2])
    with c1:
        models = st.multiselect("Model yang dibandingkan", list(mr["Model"]), default=list(mr["Model"]), key="mdl_sel")
    with c2:
        metric = st.radio("Metrik utama", metrics, horizontal=True, key="mdl_metric")
    df = mr[mr["Model"].isin(models)].copy()
    if df.empty:
        st.info("Pilih minimal satu model.")
        return
    df = df.sort_values(metric, ascending=False).reset_index(drop=True)
    df["Label"] = df["Model"].map(lambda m: f"{m} ★" if m == PRODUCTION_MODEL else m)

    fig = px.bar(df, x="Label", y=metric, text=df[metric].map(lambda v: num_id(v, 3)))
    fig.update_traces(marker_color=[PURPLE if m == PRODUCTION_MODEL else "#9CA3AF" for m in df["Model"]],
                      textposition="outside", cliponaxis=False)
    if metric == "Accuracy":
        fig.add_hline(y=ACCURACY_TARGET, line_dash="dash", line_color=RED,
                      annotation_text=f"Target ≥ {int(ACCURACY_TARGET * 100)}%", annotation_position="top left")
    lo = max(0.0, float(df[metric].min()) - 0.1)
    fig.update_yaxes(range=[lo, 1.05], title=metric)
    fig.update_xaxes(title="")
    show(fig, 360, f"{metric} per model  (★ = model produksi)")

    best = df.iloc[0]
    msg = f"Peringkat 1 pada {b(metric)}: {b(best['Model'])} ({num_id(best[metric], 3)})."
    if PRODUCTION_MODEL in set(df["Model"]) and best["Model"] != PRODUCTION_MODEL:
        prod = df[df["Model"] == PRODUCTION_MODEL].iloc[0]
        msg += (f" Model produksi {b(PRODUCTION_MODEL)} berada {num_id(abs(best[metric] - prod[metric]), 3)} "
                f"di bawahnya.")
    callout(msg)

    g1, g2 = st.columns(2)
    with g1:
        mat = df.set_index("Model")[metrics]
        zmin = max(0.0, np.floor(float(mat.values.min()) * 20) / 20)
        fig = px.imshow(mat, text_auto=".3f", aspect="auto", zmin=zmin, zmax=1,
                        color_continuous_scale=[[0, "#FFFFFF"], [1, PURPLE]])
        fig.update_layout(coloraxis_showscale=False)
        show(fig, 120 + 55 * len(mat), "Scorecard semua metrik")
    with g2:
        if {"Recall", "Precision"}.issubset(df.columns):
            fig = px.scatter(df, x="Recall", y="Precision", text="Model",
                             size=df["F1-Score"] if "F1-Score" in df.columns else None, size_max=30,
                             color_discrete_sequence=[PURPLE])
            fig.update_traces(textposition="top center")
            fig.update_xaxes(range=[0.7, 1.0])
            fig.update_yaxes(range=[0.7, 1.0])
            show(fig, 120 + 55 * len(mat), "Recall vs Precision (ukuran = F1)")

    if "Accuracy" in mr.columns:
        section("Pemeriksaan target Project Brief", "Side Quest: akurasi minimal 85%. MAE ≤ 0,02 tidak tercatat di model_results.csv.")
        chk = mr[["Model", "Accuracy"]].copy()
        chk["Target ≥ 85%"] = chk["Accuracy"].map(lambda v: "✅ Terpenuhi" if v >= ACCURACY_TARGET else "❌ Belum")
        chk["Akurasi"] = chk["Accuracy"].map(lambda v: pct(v))
        chk["Model"] = chk["Model"].map(lambda m: f"{m} ★" if m == PRODUCTION_MODEL else m)
        show_df(chk[["Model", "Akurasi", "Target ≥ 85%"]])
        prod = mr[mr["Model"] == PRODUCTION_MODEL]
        if len(prod) and float(prod["Accuracy"].iloc[0]) < ACCURACY_TARGET:
            callout(f"Akurasi {b(PRODUCTION_MODEL)} pada validation set adalah {b(pct(float(prod['Accuracy'].iloc[0])))}, "
                    f"di bawah target {int(ACCURACY_TARGET * 100)}%.", "warn", "Target belum tercapai")

    # ── Feature importance ───────────────────────────────────────────────────
    section("Fitur paling berpengaruh (XGBoost)", "Kontribusi total gain tiap fitur, dibaca dari model_xgb.json.")
    mp, sp = C["paths"].get("model"), C["paths"].get("scaled")
    if mp and sp:
        imp = xgb_gain_importance(str(mp), scaled_feature_names(str(sp)))
        if imp is None:
            st.info("Jumlah fitur pada model tidak cocok dengan scaled_full_v8.csv.")
        else:
            n = st.slider("Tampilkan N fitur teratas", 5, 20, 12, key="mdl_topn")
            top = imp.head(n).sort_values("Gain")
            fig = px.bar(top, x="Gain", y="Fitur", orientation="h", color_discrete_sequence=[PURPLE],
                         text=top["Gain"].map(lambda v: pct(v)))
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_xaxes(tickformat=".0%", title="Porsi total gain")
            fig.update_yaxes(title="")
            show(fig, 120 + 28 * n)
            share = float(imp["Gain"].iloc[0])
            if share >= 0.5:
                callout(f"{b(imp['Fitur'].iloc[0])} menyumbang {b(pct(share))} dari total gain — satu fitur mendominasi. "
                        f"Fitur <code>days_to_ruin</code> diturunkan dari saldo dan rata-rata belanja 7 hari, komponen yang "
                        f"juga dipakai mendefinisikan label. Karena itu skor tinggi perlu ditafsirkan hati-hati: model "
                        f"dapat sebagian besar menirukan aturan label, bukan mempelajari pola perilaku.", "warn", "Catatan interpretasi")
            else:
                callout(f"Fitur teratas: {b(imp['Fitur'].iloc[0])} ({pct(share)} dari total gain).")
    else:
        st.info("Butuh model_xgb.json dan scaled_full_v8.csv untuk menampilkan feature importance.")


# =============================================================================
#  HALAMAN 7 — A/B TESTING  (Q5)
# =============================================================================
def p_text(p: float) -> str:
    return "< 0,001" if p < 0.001 else num_id(p, 3)


def d_label(d: float) -> str:
    a = abs(d)
    return "dapat diabaikan" if a < 0.2 else "kecil" if a < 0.5 else "sedang" if a < 0.8 else "besar"


def page_ab(C: dict) -> None:
    hero("A/B Testing", "Pengaruh Early Warning System terhadap defisit rate", badge="Q5 · Simulasi")
    ab = read_optional(C, "ab_results")
    need = {"group", "n", "deficit_rate"}
    if ab is None or not need.issubset(ab.columns) or len(ab) < 2:
        st.warning("ab_testing_results.csv tidak ditemukan atau tidak lengkap. Jalankan 05_ab_testing.py.")
        return
    A = ab[ab["group"].str.contains("kontrol")].iloc[0]
    Bv = ab[ab["group"].str.contains("treatment")].iloc[0]
    ra, rb, na, nb = float(A["deficit_rate"]), float(Bv["deficit_rate"]), int(A["n"]), int(Bv["n"])
    delta = ra - rb

    with st.expander("📖 Desain eksperimen", expanded=False):
        st.markdown(
            "- **H0**: tidak ada beda defisit rate antara Grup A dan B · **H1**: Grup B (dengan Early Warning) lebih rendah.\n"
            "- **Grup A (kontrol)**: tanpa peringatan · **Grup B (treatment)**: peringatan aktif saat probabilitas defisit melewati ambang.\n"
            "- **Unit**: hari-pengguna pada periode test (pengguna dibagi dua secara acak).\n"
            "- **Metrik utama**: proporsi hari defisit · uji: Chi-square; pembanding: uji-t pengeluaran & Cohen's d.")

    kpi_row([
        ("Defisit rate A (kontrol)", pct(ra), f"n = {num_id(na)} hari-pengguna", PINK),
        ("Defisit rate B (treatment)", pct(rb), f"n = {num_id(nb)} hari-pengguna", BG_GREEN),
        ("Penurunan absolut", pp(delta), "A − B", CARD),
        ("Penurunan relatif", pct(delta / ra) if ra else "–", "relative risk reduction", BG_PURPLE),
    ])
    st.write("")

    ci = lambda p, n: 1.96 * np.sqrt(p * (1 - p) / n)
    dfp = pd.DataFrame({"Grup": ["A (kontrol)", "B (Early Warning)"], "Rate": [ra, rb], "CI": [ci(ra, na), ci(rb, nb)]})
    fig = px.bar(dfp, x="Grup", y="Rate", error_y="CI", color="Grup",
                 color_discrete_map={"A (kontrol)": RED, "B (Early Warning)": GREEN},
                 text=dfp["Rate"].map(lambda v: pct(v)))
    fig.update_traces(textposition="inside")
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="")
    fig.update_yaxes(tickformat=".0%", title="Defisit rate", range=[0, max(ra, rb) * 1.3])
    show(fig, 360, "Defisit rate per grup (±95% CI)")
    st.caption("CI dihitung dengan asumsi observasi independen. Karena satu pengguna menyumbang banyak hari, "
               "selang kepercayaan sebenarnya lebih lebar.")

    section("Uji statistik")
    rows = []
    if "chi2_p" in ab.columns and pd.notna(A["chi2_p"]):
        p = float(A["chi2_p"])
        rows.append(("Chi-square (proporsi defisit)", f"p {p_text(p)}", "Signifikan" if p < 0.05 else "Tidak signifikan"))
    if "ttest_p" in ab.columns and pd.notna(A["ttest_p"]):
        p = float(A["ttest_p"])
        rows.append(("Uji-t Welch (rata-rata pengeluaran harian)", f"p {p_text(p)}",
                     "Signifikan" if p < 0.05 else "Tidak signifikan"))
    if "cohens_d" in ab.columns and pd.notna(A["cohens_d"]):
        d = float(A["cohens_d"])
        rows.append(("Cohen's d (pengeluaran)", num_id(d, 3), f"Efek {d_label(d)}"))
    if rows:
        show_df(pd.DataFrame(rows, columns=["Uji", "Hasil", "Kesimpulan"]))

    callout(f"Defisit rate Grup A {b(pct(ra))} vs Grup B {b(pct(rb))}: selisih {b(pp(delta))}.", "info", "Hasil")
    p_t = float(A["ttest_p"]) if "ttest_p" in ab.columns and pd.notna(A["ttest_p"]) else None
    t_note = ""
    if p_t is not None:
        t_note = (" Uji-t pada pengeluaran harian tidak signifikan, artinya peringatan tidak mengubah besar belanja pada simulasi ini."
                  if p_t >= 0.05 else " Uji-t menunjukkan perbedaan pengeluaran yang signifikan antar grup.")
    callout(f"Hasil ini berasal dari <b>simulasi</b>. Pada 05_ab_testing.py, untuk Grup B setiap hari yang memicu "
            f"peringatan dan berlabel defisit dibalik menjadi aman dengan peluang {b(pct(AB_EFFECT_SIZE, 0))} "
            f"(<i>effect size</i> yang diasumsikan). Penurunan defisit rate adalah konsekuensi asumsi tersebut, bukan "
            f"bukti bahwa peringatan mengubah perilaku.{t_note} Pembuktian efektivitas membutuhkan uji coba pada "
            f"pengguna nyata.", "warn", "Cara membaca hasil")


# =============================================================================
#  HALAMAN 8 — DATA DICTIONARY
# =============================================================================
def page_dictionary(C: dict) -> None:
    hero("Data Dictionary", "Dokumentasi kolom dan kelompok fitur dataset FinPredict", badge="Dokumentasi")
    dd = read_optional(C, "dictionary")
    if dd is None:
        st.warning("data_dictionary_v8.csv tidak ditemukan. Jalankan 06_data_dictionary.py.")
    else:
        kpi_row([("Total entri", num_id(len(dd)), "kolom terdokumentasi", BG_PURPLE),
                 ("File data", num_id(dd["file"].nunique()) if "file" in dd.columns else "–", "tercakup", CARD),
                 ("Kolom fitur harian", num_id(int((dd["file"] == "daily_features_v8.csv").sum())) if "file" in dd.columns else "–",
                  "daily_features_v8.csv", BG_GREEN)])
        f1, f2, f3 = st.columns(3)
        with f1:
            files = ["Semua file"] + (sorted(dd["file"].unique()) if "file" in dd.columns else [])
            sel_file = st.selectbox("File", files, key="dd_file")
        with f2:
            roles = sorted(dd["role"].dropna().unique()) if "role" in dd.columns else []
            sel_roles = st.multiselect("Peran kolom", roles, default=roles)
        with f3:
            q = st.text_input("Cari kolom / deskripsi", "", key="dd_q")
        view = dd.copy()
        if sel_file != "Semua file":
            view = view[view["file"] == sel_file]
        if "role" in view.columns and roles:
            view = view[view["role"].isin(sel_roles) | view["role"].isna()]
        if q:
            mask = pd.Series(False, index=view.index)
            for c in ("kolom", "deskripsi", "business_meaning"):
                if c in view.columns:
                    mask |= view[c].astype(str).str.contains(q, case=False, regex=False, na=False)
            view = view[mask]
        st.caption(f"Menampilkan {num_id(len(view))} dari {num_id(len(dd))} entri · merah = target · oranye = kolom dengan “racun data”.")
        cols = [c for c in ["file", "kolom", "tipe_data", "role", "deskripsi", "business_meaning", "catatan"] if c in view.columns]
        view = view[cols]

        def hl(row):
            if row.get("kolom") == "label_defisit_besok":
                return ["background-color:#FFD6D6"] * len(row)
            if "RACUN" in str(row.get("tipe_data", "")):
                return ["background-color:#FFE2BF"] * len(row)
            return [""] * len(row)
        try:
            show_df(view.style.apply(hl, axis=1), height=420)
        except Exception:
            show_df(view, height=420)

    # ── Kelompok fitur ───────────────────────────────────────────────────────
    section("Feature engineering — kelompok fitur harian",
            "Setiap kolom daily_features_v8.csv dikelompokkan menurut perannya.")
    cols_all = [c for c in C["dy_all"].columns if c not in ("fase", "bulan_ts")]
    grp = pd.DataFrame({"Kolom": cols_all, "Kelompok": [feature_group(c) for c in cols_all]})
    cnt = grp["Kelompok"].value_counts().rename_axis("Kelompok").reset_index(name="Jumlah").sort_values("Jumlah")
    fig = px.bar(cnt, x="Jumlah", y="Kelompok", orientation="h", color_discrete_sequence=[PURPLE], text="Jumlah")
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_yaxes(title="")
    show(fig, 120 + 38 * len(cnt), "Jumlah kolom per kelompok")
    show_df(grp.groupby("Kelompok")["Kolom"].apply(lambda s: ", ".join(s)).reset_index())
    n_feat = int((~grp["Kelompok"].isin(["Identitas", "Target"])).sum())
    callout(f"{b(n_feat)} kolom fitur dibangun dari data transaksi: lag & rolling window menangkap kebiasaan belanja, "
            f"fitur kalender/musiman menangkap payday effect dan Harbolnas, sedangkan fitur turunan risiko "
            f"(misalnya <code>days_to_ruin</code>) merangkum kondisi saldo.")


# =============================================================================
#  HALAMAN 9 — UNDUH DATA
# =============================================================================
def csv_block(label: str, key: str, builder, filename: str, signature: tuple) -> None:
    """Data besar dibuat hanya saat diminta; di-cache pada session sesuai filter."""
    state = f"csv_{key}"
    if st.button(f"⚙️ Siapkan {label}", key=f"prep_{key}"):
        st.session_state[state] = (signature, builder().to_csv(index=False).encode("utf-8"))
    cached = st.session_state.get(state)
    if cached and cached[0] == signature:
        call_stretch(st.download_button, f"📥 Unduh {label}", data=cached[1], file_name=filename,
                     mime="text/csv", key=f"dl_{key}")
    elif cached:
        st.caption("Filter berubah — siapkan ulang file.")


def page_unduh(C: dict) -> None:
    hero("Unduh Data", "Ekspor data sesuai filter di sidebar", badge="Ekspor")
    sig = (C["tipe_sel"], str(C["date_range"][0]), str(C["date_range"][1]))
    st.caption(f"Filter aktif: {len(C['tipe_sel'])} tipe pengguna · {date_id(C['date_range'][0])} – {date_id(C['date_range'][1])}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Transaksi bersih** · {num_id(len(C['tx']))} baris")
        csv_block("transaksi", "tx", lambda: C["tx"], "finpredict_transaksi.csv", sig)
        st.markdown(f"**Fitur harian** · {num_id(len(C['dy']))} baris")
        csv_block("fitur harian", "dy", lambda: C["dy"].drop(columns=["fase", "bulan_ts"]),
                  "finpredict_fitur_harian.csv", sig)
    with c2:
        st.markdown("**Dokumen & hasil evaluasi**")
        for key, label, fname in [("dictionary", "Data Dictionary", "finpredict_data_dictionary.csv"),
                                  ("categories", "Category Mapping", "finpredict_category_mapping.csv"),
                                  ("model_results", "Hasil Model", "finpredict_model_results.csv"),
                                  ("ab_results", "Hasil A/B Testing", "finpredict_ab_results.csv")]:
            df = read_optional(C, key)
            if df is not None:
                call_stretch(st.download_button, f"📥 {label}", data=df.to_csv(index=False).encode("utf-8"),
                             file_name=fname, mime="text/csv", key=f"dl_{key}")

    section("Inventaris file")
    labels = {"transactions": "Transaksi bersih", "daily": "Fitur harian", "scaled": "Fitur ter-scale",
              "raw": "Data mentah", "categories": "Mapping kategori", "split": "Info split",
              "model": "Model XGBoost", "dictionary": "Data Dictionary", "model_results": "Hasil model",
              "ab_results": "Hasil A/B"}
    inv = []
    for key, (name, _) in FILE_SPECS.items():
        p = C["paths"].get(key)
        inv.append({"File": name, "Isi": labels[key], "Status": "✅ ditemukan" if p else "⚠️ tidak ada",
                    "Lokasi": str(p.relative_to(ROOT)) if p and ROOT in p.parents else (str(p) if p else "–")})
    show_df(pd.DataFrame(inv))


# =============================================================================
#  7. SIDEBAR, FILTER, ROUTING
# =============================================================================
PAGE_FUNCS = {
    "🏠 Ringkasan": (page_ringkasan, True),
    "🧹 Data Wrangling": (page_wrangling, False),
    "🔎 Eksplorasi Pengeluaran": (page_eksplorasi, True),
    "⚠️ Risiko Defisit": (page_risiko, True),
    "🚨 Early Warning": (page_early_warning, True),
    "🤖 Evaluasi Model": (page_model, False),
    "⚖️ A/B Testing": (page_ab, False),
    "📚 Data Dictionary": (page_dictionary, False),
    "📥 Unduh Data": (page_unduh, True),
}


def reset_filters(tipe_all: list, d_min, d_max) -> None:
    st.session_state["f_tipe"] = tipe_all
    st.session_state["f_date"] = (d_min, d_max)


def sidebar(tx_all: pd.DataFrame, n_users_all: int):
    tipe_all = sorted(map(str, tx_all["tipe_user"].unique()))
    d_min, d_max = tx_all["tanggal_transaksi"].min().date(), tx_all["tanggal_transaksi"].max().date()
    st.session_state.setdefault("page", PAGE_LABELS[0])
    st.session_state.setdefault("f_tipe", tipe_all)
    st.session_state.setdefault("f_date", (d_min, d_max))

    with st.sidebar:
        st.markdown(f'<div class="fp-brand"><div class="t">FINPREDICT</div>'
                    f'<div class="s">Analytics · v{APP_VERSION}</div></div>', unsafe_allow_html=True)
        page = st.radio("Navigasi", PAGE_LABELS, key="page", label_visibility="collapsed")
        st.markdown("---")
        st.markdown("**🎛️ Filter global**")
        tipe_sel = st.multiselect("Tipe pengguna", tipe_all, key="f_tipe")
        date_sel = st.date_input("Periode", min_value=d_min, max_value=d_max, key="f_date")
        st.button("↺ Reset filter", on_click=reset_filters, args=(tipe_all, d_min, d_max))
        st.markdown("---")
        st.caption(f"Tim {TEAM_ID} · Coding Camp 2026 powered by DBS Foundation")
        st.caption("Data: simulasi (01_data_generator.py)")

    if isinstance(date_sel, (list, tuple)) and len(date_sel) == 2:
        start, end = date_sel
    else:
        start, end = d_min, d_max
        st.sidebar.info("Pilih tanggal akhir untuk menerapkan rentang; sementara memakai seluruh periode.")
    return page, tuple(tipe_sel), (start, end)


def main() -> None:
    paths = {k: find_file(k) for k in FILE_SPECS}
    if not (paths["transactions"] and paths["daily"]):
        missing_data_screen(paths)
        return

    tx_all = load_transactions(str(paths["transactions"]))
    dy_all = load_daily(str(paths["daily"]))
    n_users_all = int(tx_all["user_id"].nunique())

    page, tipe_sel, (start, end) = sidebar(tx_all, n_users_all)
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    tx = tx_all[tx_all["tipe_user"].isin(tipe_sel) & tx_all["tanggal_transaksi"].between(t0, t1)]
    dy = dy_all[dy_all["tipe_user"].isin(tipe_sel) & dy_all["tanggal_transaksi"].between(t0, t1)]

    C = {"paths": paths, "tx_all": tx_all, "dy_all": dy_all, "tx": tx, "dy": dy,
         "n_users_all": n_users_all, "n_users": int(tx["user_id"].nunique()),
         "tipe_sel": tipe_sel, "date_range": (start, end)}

    func, needs_filtered = PAGE_FUNCS[page]
    if needs_filtered and (tx.empty or dy.empty):
        hero("Tidak ada data", "Filter global tidak menyisakan data", badge="Filter")
        st.warning("Tidak ada data pada kombinasi filter ini. Perluas rentang tanggal atau pilih tipe pengguna lain.")
        return
    func(C)


if __name__ == "__main__":
    main()
