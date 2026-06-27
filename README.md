# ♻️ WasteCase

**WasteCase** adalah platform berbasis web untuk **prediksi dan analisis kondisi persampahan Kota Bandung** menggunakan metode **Time Series Forecasting**. Sistem ini memanfaatkan data historis penanganan sampah untuk menghasilkan prediksi volume sampah serta menyajikannya melalui dashboard interaktif berbasis **Streamlit**.

---

## 📌 Latar Belakang

Pengelolaan sampah merupakan salah satu tantangan utama di kawasan perkotaan. Volume sampah yang terus berubah menyebabkan pemerintah perlu melakukan perencanaan operasional yang lebih akurat, mulai dari penyediaan armada hingga alokasi sumber daya.

WasteCase dikembangkan untuk membantu proses tersebut melalui pendekatan **data-driven decision making** dengan memanfaatkan data historis penanganan sampah Kota Bandung.

---

## 🚀 Fitur

- 📈 Prediksi volume penanganan sampah
- 📊 Interactive Dashboard
- 📉 Exploratory Data Analysis (EDA)
- 🔍 Visualisasi tren dan pola musiman
- 📅 Forecasting periode mendatang
- 📋 Perbandingan model forecasting
- ➕ Manajemen data (CRUD)

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Plotly
- Statsmodels
- Scikit-learn

---

## 📂 Dataset

Dataset diperoleh dari:

**Open Data Kota Bandung**

Periode data:

> Januari 2017 – Desember 2024

Jumlah observasi:

> 96 data bulanan

Variabel utama:

- Tahun
- Bulan
- Jumlah Penanganan Sampah (Ton)

---

## 📊 Metodologi

Tahapan pengembangan sistem meliputi:

1. Data Collection
2. Data Understanding
3. Data Preprocessing
4. Exploratory Data Analysis (EDA)
5. Train-Test Split
6. Stationarity Test (ADF)
7. Baseline Forecasting
8. ARIMA Modeling
9. SARIMA Modeling
10. Model Evaluation
11. Web Application Development
12. System Testing

---

## 📈 Model

Model yang digunakan:

- Naive Forecast
- Seasonal Naive
- ARIMA
- SARIMA

Evaluasi menggunakan:

- MAE
- RMSE
- MAPE

Model terbaik:

> **SARIMA (1,2,2)(0,1,1)₁₂**

---

## 💻 Installation

Clone repository

```bash
git clone https://github.com/annisaazzahra31/capstone2026-kelompok5.git
```

Masuk ke folder project

```bash
cd capstone2026-kelompok5
```

Install dependencies

```bash
pip install -r requirements.txt
```

Jalankan aplikasi

```bash
streamlit run app.py
```

---

## 📷 Dashboard Preview

Tambahkan screenshot aplikasi di sini.

```
assets/dashboard.png
```

---

## 📁 Repository Structure

```
.
├── app.py
├── codingan capstone kelompok 5.ipynb
├── jumlah_capaian_penanganan_sampah.csv
├── requirements.txt
└── README.md
```

---

## 👥 Team

Kelompok 5 — Capstone DS47GAB

- Annisa Azzahra Rahmah
- Amanda Catleya Rahiu
- Ariel Saradilla
- Ariel Furqanul Khaq

---

## 🎓 Academic Project

Capstone Project

Bachelor of Data Science

Faculty of Informatics

Telkom University

2026
