# Klasifikasi Tingkat Kecanduan Smartphone Menggunakan Metode Jaringan Syaraf Tiruan (MLP)

## Deskripsi Proyek

Proyek ini mengimplementasikan metode Jaringan Syaraf Tiruan dengan algoritma Multi-Layer Perceptron (MLP) untuk mengklasifikasikan tingkat kecanduan smartphone pada pengguna ke dalam tiga kategori: **Mild** (Ringan), **Moderate** (Sedang), dan **Severe** (Berat). Eksperimen dilakukan dengan membandingkan empat konfigurasi arsitektur MLP yang berbeda berdasarkan jumlah hidden layer dan jumlah neuron.

---

## Struktur Proyek

```
UAS/
├── Dataset/
│   └── Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv
├── Output/
│   ├── confusion_matrix.png
│   ├── perbandingan_akurasi.png
│   ├── kurva_loss.png
│   └── eda_distribusi.png
├── ProjectMLP.py
├── UjiCoba.py
└── README.md
```

---

## Dataset

| Keterangan            | Detail                                                        |
|-----------------------|---------------------------------------------------------------|
| Nama file             | Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv         |
| Jumlah baris          | 7.500                                                         |
| Jumlah kolom          | 16                                                            |
| Kolom target          | `addiction_level` (Mild / Moderate / Severe)                  |
| Data setelah cleaning | 6.681 baris (819 baris dihapus karena target kosong)          |

### Fitur yang Digunakan

| Kolom                     | Tipe        | Keterangan                              |
|---------------------------|-------------|-----------------------------------------|
| `age`                     | Integer     | Usia pengguna (tahun)                   |
| `gender`                  | Kategorikal | Jenis kelamin (Male / Female / Other)   |
| `daily_screen_time_hours` | Float       | Screen time harian (jam)                |
| `social_media_hours`      | Float       | Jam penggunaan media sosial per hari    |
| `gaming_hours`            | Float       | Jam bermain game per hari               |
| `work_study_hours`        | Float       | Jam kerja/belajar via HP per hari       |
| `sleep_hours`             | Float       | Jam tidur per malam                     |
| `notifications_per_day`   | Integer     | Jumlah notifikasi per hari              |
| `app_opens_per_day`       | Integer     | Jumlah buka aplikasi per hari           |
| `weekend_screen_time`     | Float       | Screen time akhir pekan (jam)           |
| `stress_level`            | Kategorikal | Tingkat stres (Low / Medium / High)     |
| `academic_work_impact`    | Kategorikal | Dampak terhadap kerja/kuliah (Yes / No) |

---

## Preprocessing

1. Menghapus kolom ID yang tidak relevan (`transaction_id`, `user_id`, `addicted_label`)
2. Menghapus baris dengan nilai kosong pada kolom target (819 baris)
3. Encoding fitur kategorikal menggunakan `LabelEncoder`
4. Normalisasi seluruh fitur numerik menggunakan `StandardScaler` (mean=0, std=1)

---

## Split Data

| Set      | Jumlah Sampel | Persentase |
|----------|:-------------:|:----------:|
| Training | 5.344         | 80%        |
| Testing  | 1.337         | 20%        |

Split dilakukan secara **stratified** untuk menjaga proporsi kelas tetap seimbang di kedua set.

### Distribusi Kelas

| Kelas    | Training       | Testing        |
|----------|:--------------:|:--------------:|
| Mild     | 1.098 (20.5%)  | 275 (20.6%)    |
| Moderate | 2.299 (43.0%)  | 575 (43.0%)    |
| Severe   | 1.947 (36.4%)  | 487 (36.4%)    |

---

## Konfigurasi Model MLP

### Hyperparameter Umum

| Parameter           | Nilai                |
|---------------------|----------------------|
| Activation function | ReLU                 |
| Optimizer / Solver  | Adam                 |
| Maksimum iterasi    | 500                  |
| Early stopping      | True (patience = 20) |
| Validation fraction | 10%                  |
| Random state        | 42                   |

### Empat Konfigurasi yang Diuji

| Kode    | Hidden Layer | Neuron per Layer | hidden_layer_sizes |
|---------|:------------:|:----------------:|:------------------:|
| 1HL-16N | 1            | 16               | `(16,)`            |
| 1HL-32N | 1            | 32               | `(32,)`            |
| 2HL-16N | 2            | 16               | `(16, 16)`         |
| 2HL-32N | 2            | 32               | `(32, 32)`         |

---

## Hasil Training dan Evaluasi

| Konfigurasi     | Hidden Layer | Neuron | Train Accuracy | Test Accuracy | Iterasi |
|-----------------|:------------:|:------:|:--------------:|:-------------:|:-------:|
| 1HL-16N         | 1            | 16     | 51.22%         | 50.93%        | 29      |
| 1HL-32N         | 1            | 32     | 57.73%         | 53.78%        | 48      |
| **2HL-16N**     | **2**        | **16** | **58.50%**     | **54.75%**    | **48**  |
| 2HL-32N         | 2            | 32     | 52.43%         | 51.61%        | 23      |

Model terbaik: **2HL-16N** dengan Test Accuracy = **54.75%**

### Classification Report — Model Terbaik (2HL-16N)

| Kelas        | Precision | Recall | F1-Score | Support |
|--------------|:---------:|:------:|:--------:|:-------:|
| Mild         | 0.76      | 0.80   | 0.78     | 275     |
| Moderate     | 0.48      | 0.46   | 0.47     | 575     |
| Severe       | 0.50      | 0.51   | 0.50     | 487     |
| **Accuracy** |           |        | **0.55** | **1337**|

---

## Analisis Perbandingan

### 1 Hidden Layer vs 2 Hidden Layer

| Jumlah Hidden Layer | Rata-rata Test Accuracy |
|:-------------------:|:-----------------------:|
| 1 Hidden Layer      | 52.36%                  |
| 2 Hidden Layer      | 53.18%                  |

Penambahan satu hidden layer memberikan peningkatan akurasi sebesar **+0.82%**. Model dengan dua hidden layer mampu mempelajari representasi fitur yang lebih kompleks sehingga menghasilkan performa yang lebih baik pada dataset ini.

### 16 Neuron vs 32 Neuron

| Jumlah Neuron | Rata-rata Test Accuracy |
|:-------------:|:-----------------------:|
| 16 Neuron     | 52.84%                  |
| 32 Neuron     | 52.69%                  |

Penambahan jumlah neuron dari 16 menjadi 32 tidak memberikan peningkatan yang signifikan. Model dengan 16 neuron justru unggul tipis sebesar **+0.15%**. Hal ini menunjukkan bahwa kapasitas model yang lebih besar tidak selalu menghasilkan performa lebih baik, terutama apabila terjadi konvergensi prematur atau data tidak memiliki pola yang cukup kompleks untuk dimanfaatkan oleh neuron tambahan.

---

## Kesimpulan

**1. Model Terbaik**
Dari keempat konfigurasi yang diuji, model **2HL-16N** (2 Hidden Layer, 16 Neuron per layer) menghasilkan performa terbaik dengan Train Accuracy **58.50%** dan Test Accuracy **54.75%**.

**2. Pengaruh Jumlah Hidden Layer**
Penambahan hidden layer terbukti lebih berpengaruh positif terhadap akurasi dibandingkan penambahan jumlah neuron. Model dengan 2 hidden layer secara konsisten mengungguli model dengan 1 hidden layer dalam eksperimen ini.

**3. Pengaruh Jumlah Neuron**
Menambah jumlah neuron dari 16 menjadi 32 tidak memberikan dampak signifikan dan dalam beberapa konfigurasi justru menurunkan performa. Ini mengindikasikan bahwa arsitektur yang lebih besar belum tentu optimal untuk dataset ini.

**4. Tingkat Akurasi**
Akurasi keseluruhan berkisar antara 50–55%, yang tergolong rendah untuk sebuah model klasifikasi. Faktor-faktor yang kemungkinan memengaruhi hal ini antara lain:
- Pola antar kelas Mild, Moderate, dan Severe saling tumpang tindih dalam ruang fitur yang tersedia
- Terdapat 819 baris data (sekitar 11%) yang tidak dapat digunakan karena nilai target kosong
- Arsitektur MLP yang relatif kecil (16–32 neuron) belum cukup kapasitasnya untuk menangkap kompleksitas data

**5. Performa per Kelas**
Kelas **Mild** memiliki performa terbaik (F1-Score = 0.78), sedangkan kelas **Moderate** paling sulit diklasifikasikan (F1-Score = 0.47). Hal ini kemungkinan karena kelas Moderate berada di posisi tengah sehingga banyak sampelnya yang tumpang tindih dengan kedua kelas lainnya.

**6. Rekomendasi Pengembangan**
Untuk meningkatkan akurasi model, dapat dicoba pendekatan berikut:
- Menambah jumlah hidden layer atau neuron yang lebih besar secara bertahap
- Melakukan tuning hyperparameter yang lebih mendalam, seperti learning rate dan batch size
- Menggunakan metode lain sebagai pembanding, seperti Random Forest atau Support Vector Machine (SVM)
- Menangani data yang hilang dengan imputasi, bukan hanya penghapusan baris

---

## Cara Menjalankan Program

### Kebutuhan Library

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

### Menjalankan Training dan Evaluasi

```bash
python ProjectMLP.py
```

### Menjalankan Uji Coba Input Manual

```bash
python UjiCoba.py
```

---

## Contoh Input untuk Uji Coba

### Mild — Kecanduan Ringan

```
Usia                        : 31
Jenis kelamin               : Other
Screen time harian          : 6.06 jam
Jam sosial media            : 1.36 jam
Jam gaming                  : 3.83 jam
Jam kerja/belajar via HP    : 2.35 jam
Jam tidur                   : 4.92 jam
Notifikasi per hari         : 44
Buka aplikasi per hari      : 106
Screen time akhir pekan     : 8.68 jam
Tingkat stres               : High
Dampak pada kerja/kuliah    : No
```

### Moderate — Kecanduan Sedang

```
Usia                        : 32
Jenis kelamin               : Other
Screen time harian          : 7.83 jam
Jam sosial media            : 5.85 jam
Jam gaming                  : 1.51 jam
Jam kerja/belajar via HP    : 3.54 jam
Jam tidur                   : 8.23 jam
Notifikasi per hari         : 178
Buka aplikasi per hari      : 107
Screen time akhir pekan     : 9.77 jam
Tingkat stres               : High
Dampak pada kerja/kuliah    : Yes
```

### Severe — Kecanduan Berat

```
Usia                        : 25
Jenis kelamin               : Male
Screen time harian          : 9.96 jam
Jam sosial media            : 5.92 jam
Jam gaming                  : 3.42 jam
Jam kerja/belajar via HP    : 5.27 jam
Jam tidur                   : 6.21 jam
Notifikasi per hari         : 136
Buka aplikasi per hari      : 177
Screen time akhir pekan     : 12.55 jam
Tingkat stres               : Low
Dampak pada kerja/kuliah    : No
```

---

## Informasi Proyek

| Keterangan  | Detail                                                |
|-------------|-------------------------------------------------------|
| Mata Kuliah | Jaringan Syaraf Tiruan                                |
| Metode      | Multi-Layer Perceptron (MLP)                          |
| Dataset     | Smartphone Usage and Addiction Analysis (7.500 baris) |
| Library     | scikit-learn, pandas, numpy, matplotlib, seaborn      |