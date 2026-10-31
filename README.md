# FinPredict Analytics

**Sistem Analitik Prediktif untuk Deteksi Risiko Defisit Keuangan Generasi Z**

FinPredict Analytics adalah sistem yang dirancang untuk memahami pola arus kas, menganalisis perilaku pengeluaran, dan memberikan peringatan dini (early warning) terhadap potensi defisit keuangan pada pengguna Generasi Z.

Sistem ini mengintegrasikan proses Data Wrangling, Exploratory Data Analysis (EDA), Feature Engineering, Machine Learning, hingga mekanisme Early Warning dalam satu alur kerja yang utuh.

---

## 📋 Daftar Isi
- [Fitur Utama](#-fitur-utama)
- [Dataset](#-dataset)
- [Arsitektur & Alur Kerja](#-arsitektur--alur-kerja)
- [Hasil Analisis Utama](#-hasil-analisis-utama)
- [Performa Model](#-performa-model)
- [Early Warning System](#-early-warning-system)
- [Struktur Proyek](#-struktur-proyek)
- [Cara Menjalankan](#-cara-menjalankan)
- [Tech Stack](#-tech-stack)
- [Keterbatasan](#-keterbatasan)
- [Pengembangan Selanjutnya](#-pengembangan-selanjutnya)
- [Kontributor](#-kontributor)

---

## 🎯 Fitur Utama

- Pembersihan dan persiapan data transaksi keuangan (Data Wrangling)
- Analisis pola pengeluaran berdasarkan waktu (Payday Effect, hari dalam minggu, fase bulan)
- Segmentasi pengguna berdasarkan tipe perilaku keuangan
- Prediksi risiko defisit menggunakan model Machine Learning
- Dashboard interaktif untuk eksplorasi data dan monitoring
- Sistem Early Warning dengan proyeksi saldo 30 hari ke depan
- Simulasi dampak pengurangan belanja Wants terhadap kondisi keuangan

---

## 📊 Dataset

| Item                    | Nilai                  |
|-------------------------|------------------------|
| Jumlah Pengguna         | 150                    |
| Periode                 | 365 hari               |
| Data Mentah             | 253.525 transaksi      |
| Data Bersih             | 249.778 transaksi      |
| Fitur Harian            | 54.750 baris           |
| Total Pengeluaran       | ± Rp7,25 miliar        |
| Rata-rata Transaksi     | Rp29.000               |
| Median Transaksi        | Rp18.000               |

**Pembagian Data (Kronologis):**
- Training   : 255 hari
- Validation : 54 hari
- Testing    : 56 hari

> Dataset bersifat **sintetis**.

---

## 🏗️ Arsitektur & Alur Kerja

1. **Data Wrangling**  
   Pembersihan data (duplikat, format tanggal, nominal teks, kategori kosong, dll.)

2. **Exploratory Data Analysis (EDA)**  
   Analisis pola waktu, komposisi kategori, Needs vs Wants, metode pembayaran, dan perilaku per tipe pengguna.

3. **Feature Engineering**  
   Pembentukan fitur harian dan fitur rolling untuk pemodelan.

4. **Pemodelan Machine Learning**  
   Perbandingan beberapa model (XGBoost, Random Forest, GRU, LSTM).

5. **Early Warning System**  
   Deteksi risiko defisit dan proyeksi saldo 30 hari ke depan.

---

## 📈 Hasil Analisis Utama

### Pola Pengeluaran
- **Payday Effect**: Pengeluaran tanggal 1–10 sekitar **2,2 kali** lebih tinggi dibanding tanggal 22–31.
- Pengeluaran akhir pekan **9,3%** lebih tinggi dibanding hari kerja.

### Komposisi Kategori
- Makan Pokok: **22,5 – 23%** (pos terbesar)
- Tagihan Tetap: **10%**
- Hiburan: **±11%** (Cafe & Hiburan 5%, Gaming & Top-up 3%, dll.)
- Harbolnas / Flash Sale termasuk kategori Wants terbesar.

### Risiko Defisit berdasarkan Dimensi

| Dimensi              | Selisih      | Kekuatan     | Risiko Tertinggi              | Risiko Terendah           |
|----------------------|--------------|--------------|-------------------------------|---------------------------|
| Tipe Pengguna        | 80,8 poin    | Sangat Kuat  | Mahasiswa Dependen (98%)     | First Jobber Sehat (17%) |
| Bulan                | 52,2 poin    | Sangat Kuat  | Mei 2025 (98%)               | Sep 2025 (45%)           |
| Fase Bulan           | 7,7 poin     | Penting      | Tanggal Tua 22–31 (56%)      | Tanggal Muda 1–10 (48%)  |
| Hari dalam Minggu    | 1,7 poin     | Lemah        | Sabtu (53%)                  | Kamis (51%)              |

### Tipe Pengguna
- **Mahasiswa Dependen** dan **Mahasiswa Hustler** memiliki risiko defisit ekstrem (96–98%).
- **First Jobber Sehat** memiliki risiko terendah (17%) dan penggunaan Paylater paling rendah.

---

## 🤖 Performa Model

| Model                  | AUC-ROC | F1-Score | Recall | Keterangan              |
|------------------------|---------|----------|--------|-------------------------|
| XGBoost                | 0,978   | 0,968    | 0,962  | Terbaik                 |
| Random Forest          | 0,977   | 0,968    | 0,960  | Hampir setara XGBoost   |
| GRU                    | 0,965   | 0,900    | 0,949  | Cukup baik              |
| **LSTM (Produksi)**    | 0,914   | 0,810    | 0,864  | Model yang digunakan    |

---

## 🚨 Early Warning System

Contoh kasus (Pengguna U002 – Mahasiswa Hustler):

- Status: **BAHAYA**
- Saldo Kumulatif: **-Rp 16,7 juta**
- Days to Ruin: **0 hari** (sudah defisit)
- Rasio Wants 7 hari terakhir: 46,6%

**Simulasi Keseluruhan:**  
Mekanisme early warning menurunkan tingkat defisit dari **42,1%** menjadi **31,9%** (turun 10,2 poin persentase) — hasil dari simulasi menggunakan data sintetis.

---

## 📁 Struktur Proyek

```text
FinPredict-Analytics/
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
├── notebooks/
│   ├── 01_data_wrangling.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_early_warning.ipynb
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── utils/
├── dashboard/          # Streamlit / Dash / dll
├── models/             # Model yang sudah di-train
├── reports/
├── requirements.txt
└── README.mdSesuaikan struktur di atas dengan struktur folder aktual di repo kamu.⚙️ Cara MenjalankanBash# 1. Clone repository
git clone https://github.com/username/FinPredict-Analytics.git
cd FinPredict-Analytics

# 2. Buat virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate      # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Jalankan notebook atau dashboard
jupyter notebook
# atau
streamlit run dashboard/app.py🛠️ Tech Stack
Bahasa: Python
Data Processing: Pandas, NumPy
Visualisasi: Matplotlib, Seaborn, Plotly
Machine Learning: Scikit-learn, XGBoost, TensorFlow / PyTorch (LSTM & GRU)
Dashboard: Streamlit / Dash (sesuaikan)
Tools Lain: Jupyter Notebook
⚠️ Keterbatasan
Dataset yang digunakan bersifat sintetis, bukan data transaksi nyata.
Masih terdapat kategori yang belum sepenuhnya terpetakan.
Perlu validasi lebih lanjut terkait potensi data leakage pada scaling, fitur rolling, dan pembentukan target.
Efektivitas early warning masih berdasarkan simulasi.
🚀 Pengembangan Selanjutnya
Menyelesaikan pemetaan kategori yang belum lengkap
Menguji model menggunakan data transaksi nyata
Meningkatkan performa model produksi (LSTM) atau mempertimbangkan ensemble
Menambahkan fitur rekomendasi personalisasi yang lebih actionable
Deploy dashboard ke production
👤 Kontributor
Liwa'Uddin 
