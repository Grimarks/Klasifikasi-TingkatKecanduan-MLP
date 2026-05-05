import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

DATASET_PATH = "Dataset/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv"
RANDOM_STATE = 42
TEST_SIZE    = 0.2

# Model terbaik dari hasil eksperimen = 2HL-16N
BEST_CONFIG  = (16, 16)
BEST_NAME    = "2HL-16N"

def garis(char="─", n=60):
    print(char * n)

def input_float(prompt, vmin, vmax):
    while True:
        try:
            val = float(input(f"  {prompt} [{vmin}–{vmax}] : "))
            if vmin <= val <= vmax:
                return val
            print(f"    ⚠ Nilai harus antara {vmin} dan {vmax}. Coba lagi.")
        except ValueError:
            print("    ⚠ Input tidak valid. Masukkan angka.")

def input_int(prompt, vmin, vmax):
    while True:
        try:
            val = int(input(f"  {prompt} [{vmin}–{vmax}] : "))
            if vmin <= val <= vmax:
                return val
            print(f"    ⚠ Nilai harus antara {vmin} dan {vmax}. Coba lagi.")
        except ValueError:
            print("    ⚠ Input tidak valid. Masukkan bilangan bulat.")

def input_pilihan(prompt, pilihan):
    pilihan_lower = [p.lower() for p in pilihan]
    while True:
        tampil = " / ".join(pilihan)
        val = input(f"  {prompt} ({tampil}) : ").strip()
        if val.lower() in pilihan_lower:
            return pilihan[pilihan_lower.index(val.lower())]
        print(f"    ⚠ Pilihan tidak valid. Harus salah satu dari: {tampil}")


# ══════════════════════════════════════════════════════════════
#  STEP 1 — LATIH ULANG MODEL (dari dataset asli)
# ══════════════════════════════════════════════════════════════
def latih_model():
    print("\n  Memuat dataset dan melatih ulang model terbaik...")

    df = pd.read_csv(DATASET_PATH)
    df = df.drop(columns=['transaction_id', 'user_id', 'addicted_label'])
    df = df.dropna(subset=['addiction_level'])

    # Encode kategorikal
    le_gender = LabelEncoder()
    le_stress  = LabelEncoder()
    le_impact  = LabelEncoder()
    le_target  = LabelEncoder()

    df['gender']               = le_gender.fit_transform(df['gender'].astype(str))
    df['stress_level']         = le_stress.fit_transform(df['stress_level'].astype(str))
    df['academic_work_impact'] = le_impact.fit_transform(df['academic_work_impact'].astype(str))
    df['addiction_level']      = le_target.fit_transform(df['addiction_level'])

    X = df.drop(columns=['addiction_level'])
    y = df['addiction_level']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)

    clf = MLPClassifier(
        hidden_layer_sizes = BEST_CONFIG,
        activation         = 'relu',
        solver             = 'adam',
        max_iter           = 500,
        random_state       = RANDOM_STATE,
        early_stopping     = True,
        validation_fraction= 0.1,
        n_iter_no_change   = 20,
        verbose            = False,
    )
    clf.fit(X_train_sc, y_train)

    encoders = {
        'gender'  : le_gender,
        'stress'  : le_stress,
        'impact'  : le_impact,
        'target'  : le_target,
    }
    print(f"  ✓ Model [{BEST_NAME}] berhasil dilatih ({clf.n_iter_} iterasi).")
    return clf, scaler, encoders, list(X.columns)


# ══════════════════════════════════════════════════════════════
#  STEP 2 — TAMPILKAN MENU CONTOH INPUT
# ══════════════════════════════════════════════════════════════
CONTOH_INPUT = {
    "Mild (Kecanduan Ringan)": {
        "age"                     : 31,
        "gender"                  : "Other",
        "daily_screen_time_hours" : 6.06,
        "social_media_hours"      : 1.36,
        "gaming_hours"            : 3.83,
        "work_study_hours"        : 2.35,
        "sleep_hours"             : 4.92,
        "notifications_per_day"   : 44,
        "app_opens_per_day"       : 106,
        "weekend_screen_time"     : 8.68,
        "stress_level"            : "High",
        "academic_work_impact"    : "No",
    },
    "Moderate (Kecanduan Sedang)": {
        "age"                     : 32,
        "gender"                  : "Other",
        "daily_screen_time_hours" : 7.83,
        "social_media_hours"      : 5.85,
        "gaming_hours"            : 1.51,
        "work_study_hours"        : 3.54,
        "sleep_hours"             : 8.23,
        "notifications_per_day"   : 178,
        "app_opens_per_day"       : 107,
        "weekend_screen_time"     : 9.77,
        "stress_level"            : "High",
        "academic_work_impact"    : "Yes",
    },
    "Severe (Kecanduan Berat)": {
        "age"                     : 25,
        "gender"                  : "Male",
        "daily_screen_time_hours" : 9.96,
        "social_media_hours"      : 5.92,
        "gaming_hours"            : 3.42,
        "work_study_hours"        : 5.27,
        "sleep_hours"             : 6.21,
        "notifications_per_day"   : 136,
        "app_opens_per_day"       : 177,
        "weekend_screen_time"     : 12.55,
        "stress_level"            : "Low",
        "academic_work_impact"    : "No",
    },
}

def tampilkan_contoh():
    print()
    garis("═")
    print("  CONTOH DATA INPUT")
    garis("═")
    for label, data in CONTOH_INPUT.items():
        print(f"\n  📌 Contoh [{label}]:")
        garis()
        for k, v in data.items():
            print(f"    {k:<30} : {v}")
    garis("═")


# ══════════════════════════════════════════════════════════════
#  STEP 3 — INPUT MANUAL DARI USER
# ══════════════════════════════════════════════════════════════
def input_manual():
    print()
    garis("═")
    print("  MASUKKAN DATA PENGGUNA SMARTPHONE")
    garis("═")
    print("  Isi setiap pertanyaan sesuai kondisi pengguna.\n")

    data = {}
    data['age']                     = input_int  ("Usia (tahun)",                         18, 35)
    data['gender']                  = input_pilihan("Jenis kelamin",                       ["Male", "Female", "Other"])
    data['daily_screen_time_hours'] = input_float("Screen time harian (jam)",             3.0, 12.0)
    data['social_media_hours']      = input_float("Jam penggunaan sosial media per hari", 0.5,  6.0)
    data['gaming_hours']            = input_float("Jam bermain game per hari",            0.0,  4.0)
    data['work_study_hours']        = input_float("Jam kerja/belajar via HP per hari",    0.5,  6.0)
    data['sleep_hours']             = input_float("Jam tidur per malam",                  4.5,  9.0)
    data['notifications_per_day']   = input_int  ("Jumlah notifikasi per hari",           20,  250)
    data['app_opens_per_day']       = input_int  ("Jumlah buka aplikasi per hari",        15,  180)
    data['weekend_screen_time']     = input_float("Screen time akhir pekan (jam)",        3.6, 14.9)
    data['stress_level']            = input_pilihan("Tingkat stres",                      ["Low", "Medium", "High"])
    data['academic_work_impact']    = input_pilihan("Apakah HP berdampak pada kerja/kuliah?", ["Yes", "No"])

    return data


# ══════════════════════════════════════════════════════════════
#  STEP 4 — PREDIKSI
# ══════════════════════════════════════════════════════════════
def prediksi(data_dict, clf, scaler, encoders, feature_cols):
    # Encode kategorikal
    data_enc = dict(data_dict)
    data_enc['gender']               = encoders['gender'].transform([data_dict['gender']])[0]
    data_enc['stress_level']         = encoders['stress'].transform([data_dict['stress_level']])[0]
    data_enc['academic_work_impact'] = encoders['impact'].transform([data_dict['academic_work_impact']])[0]

    # Susun DataFrame dengan urutan kolom yang sama saat training
    df_input = pd.DataFrame([data_enc])[feature_cols]

    # Normalisasi
    X_sc = scaler.transform(df_input)

    # Prediksi
    pred_enc    = clf.predict(X_sc)[0]
    pred_label  = encoders['target'].inverse_transform([pred_enc])[0]
    proba       = clf.predict_proba(X_sc)[0]
    kelas_names = encoders['target'].classes_

    return pred_label, proba, kelas_names


def tampilkan_hasil(data_dict, pred_label, proba, kelas_names):
    print()
    garis("═")
    print("  HASIL PREDIKSI MODEL MLP")
    garis("═")

    # Emoji & deskripsi per kelas
    info = {
        "Mild"    : ("🟢", "KECANDUAN RINGAN",
                     "Penggunaan smartphone masih dalam batas wajar.\n"
                     "  Pengguna dapat mengontrol diri dengan baik."),
        "Moderate": ("🟡", "KECANDUAN SEDANG",
                     "Penggunaan smartphone mulai mengganggu aktivitas.\n"
                     "  Disarankan membatasi screen time secara bertahap."),
        "Severe"  : ("🔴", "KECANDUAN BERAT",
                     "Penggunaan smartphone sudah berlebihan dan berdampak\n"
                     "  negatif pada kesehatan dan produktivitas.\n"
                     "  Sangat disarankan untuk melakukan digital detox."),
    }

    emoji, status, saran = info.get(pred_label, ("⚪", pred_label, ""))

    print(f"\n  {emoji}  PREDIKSI  : {pred_label.upper()} — {status}")
    print(f"\n  💡 Interpretasi:")
    print(f"    {saran}")

    print(f"\n  📊 Probabilitas per Kelas:")
    garis()
    for kls, prob in zip(kelas_names, proba):
        bar  = "█" * int(prob * 30)
        mark = " ◀" if kls == pred_label else ""
        print(f"    {kls:<10} : {bar:<30} {prob*100:5.1f}%{mark}")
    garis()

    print(f"\n  📋 Ringkasan Data Input:")
    garis()
    for k, v in data_dict.items():
        print(f"    {k:<30} : {v}")
    garis("═")


# ══════════════════════════════════════════════════════════════
#  MAIN — MENU UTAMA
# ══════════════════════════════════════════════════════════════
def main():
    print()
    print("╔" + "═"*58 + "╗")
    print("║   UJI COBA MODEL MLP — KECANDUAN SMARTPHONE             ║")
    print(f"║   Model Terbaik : [{BEST_NAME}] (2 Hidden Layer, 16 Neuron)  ║")
    print("╚" + "═"*58 + "╝")

    # Latih model sekali di awal
    clf, scaler, encoders, feature_cols = latih_model()

    while True:
        print()
        garis("─")
        print("  MENU UTAMA")
        garis("─")
        print("  [1] Input data manual (uji sendiri)")
        print("  [2] Gunakan contoh data (Mild / Moderate / Severe)")
        print("  [3] Tampilkan semua contoh data")
        print("  [0] Keluar")
        garis("─")

        pilihan = input("  Pilih menu [0/1/2/3] : ").strip()

        # ── Menu 1: Input manual
        if pilihan == "1":
            data = input_manual()
            pred_label, proba, kelas_names = prediksi(data, clf, scaler, encoders, feature_cols)
            tampilkan_hasil(data, pred_label, proba, kelas_names)

            lagi = input("\n  Uji data lain? (y/n) : ").strip().lower()
            if lagi != 'y':
                break

        # ── Menu 2: Pakai contoh
        elif pilihan == "2":
            print()
            garis()
            print("  Pilih contoh data:")
            contoh_list = list(CONTOH_INPUT.items())
            for i, (label, _) in enumerate(contoh_list, 1):
                print(f"    [{i}] {label}")
            garis()

            while True:
                try:
                    idx = int(input("  Pilih nomor [1/2/3] : ")) - 1
                    if 0 <= idx < len(contoh_list):
                        break
                    print("    ⚠ Pilih angka 1, 2, atau 3.")
                except ValueError:
                    print("    ⚠ Input tidak valid.")

            label_contoh, data = contoh_list[idx]
            print(f"\n  ✓ Menggunakan contoh: [{label_contoh}]")
            pred_label, proba, kelas_names = prediksi(data, clf, scaler, encoders, feature_cols)
            tampilkan_hasil(data, pred_label, proba, kelas_names)

            lagi = input("\n  Uji data lain? (y/n) : ").strip().lower()
            if lagi != 'y':
                break

        # ── Menu 3: Tampilkan semua contoh
        elif pilihan == "3":
            tampilkan_contoh()

        # ── Menu 0: Keluar
        elif pilihan == "0":
            print()
            print("  Terima kasih! Program selesai.")
            print()
            break

        else:
            print("  Pilihan tidak valid. Masukkan 0, 1, 2, atau 3.")


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()