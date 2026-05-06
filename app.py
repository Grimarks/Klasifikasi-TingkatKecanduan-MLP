import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
DATASET_PATH = "Dataset/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv"
RANDOM_STATE = 42
TEST_SIZE    = 0.2
BEST_CONFIG  = (16, 16)
BEST_NAME    = "2HL-16N"

CONFIGS = {
    "1HL-16N": (16,),
    "1HL-32N": (32,),
    "2HL-16N": (16, 16),
    "2HL-32N": (32, 32),
}

MLP_PARAMS = dict(
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=RANDOM_STATE,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    verbose=False,
)

CONTOH_INPUT = {
    "Mild (Kecanduan Ringan)": {
        "age": 31, "gender": "Other", "daily_screen_time_hours": 6.06,
        "social_media_hours": 1.36, "gaming_hours": 3.83, "work_study_hours": 2.35,
        "sleep_hours": 4.92, "notifications_per_day": 44, "app_opens_per_day": 106,
        "weekend_screen_time": 8.68, "stress_level": "High", "academic_work_impact": "No",
    },
    "Moderate (Kecanduan Sedang)": {
        "age": 30, "gender": "Male", "daily_screen_time_hours": 8.21,
        "social_media_hours": 1.55, "gaming_hours": 1.88, "work_study_hours": 3.72,
        "sleep_hours": 8.12, "notifications_per_day": 185, "app_opens_per_day": 67,
        "weekend_screen_time": 9.94, "stress_level": "Medium", "academic_work_impact": "Yes",
    },
    "Severe (Kecanduan Berat)": {
        "age": 25, "gender": "Male", "daily_screen_time_hours": 9.96,
        "social_media_hours": 5.92, "gaming_hours": 3.42, "work_study_hours": 5.27,
        "sleep_hours": 6.21, "notifications_per_day": 136, "app_opens_per_day": 177,
        "weekend_screen_time": 12.55, "stress_level": "Low", "academic_work_impact": "No",
    },
}

# ─────────────────────────────────────────────
#  LOAD & TRAIN
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Melatih model MLP, harap tunggu...")
def train_model():
    df = pd.read_csv(DATASET_PATH)
    df = df.drop(columns=['transaction_id', 'user_id', 'addicted_label'])
    df = df.dropna(subset=['addiction_level'])
    df = df[df['addiction_level'] != 'None']

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
    feature_cols = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Train all configs
    models  = {}
    results = {}
    for name, hl in CONFIGS.items():
        clf = MLPClassifier(hidden_layer_sizes=hl, **MLP_PARAMS)
        clf.fit(X_train_sc, y_train)
        models[name] = clf
        results[name] = {
            'train_acc': accuracy_score(y_train, clf.predict(X_train_sc)),
            'test_acc' : accuracy_score(y_test,  clf.predict(X_test_sc)),
            'iters'    : clf.n_iter_,
            'y_pred'   : clf.predict(X_test_sc),
        }

    encoders = {
        'gender': le_gender,
        'stress': le_stress,
        'impact': le_impact,
        'target': le_target,
    }

    return models, results, scaler, encoders, feature_cols, X_test_sc, y_test

@st.cache_data(show_spinner=False)
def load_raw_data():
    df = pd.read_csv(DATASET_PATH)
    df = df[df['addiction_level'] != 'None']
    df = df.dropna(subset=['addiction_level'])
    return df

# ─────────────────────────────────────────────
#  PREDICT
# ─────────────────────────────────────────────
def predict(data_dict, model, scaler, encoders, feature_cols):
    data_enc = dict(data_dict)
    data_enc['gender']               = encoders['gender'].transform([data_dict['gender']])[0]
    data_enc['stress_level']         = encoders['stress'].transform([data_dict['stress_level']])[0]
    data_enc['academic_work_impact'] = encoders['impact'].transform([data_dict['academic_work_impact']])[0]

    df_input = pd.DataFrame([data_enc])[feature_cols]
    X_sc     = scaler.transform(df_input)

    pred_enc   = model.predict(X_sc)[0]
    pred_label = encoders['target'].inverse_transform([pred_enc])[0]
    proba      = model.predict_proba(X_sc)[0]
    kelas      = encoders['target'].classes_
    return pred_label, proba, kelas

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Klasifikasi Kecanduan Smartphone — MLP",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 1.5rem;
        text-align: center; color: white;
    }
    .main-header h1 { margin: 0; font-size: 2rem; }
    .main-header p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 1rem; }
    .result-card {
        border-radius: 12px; padding: 1.5rem; text-align: center;
        margin: 1rem 0; border: 2px solid;
    }
    .result-mild     { background: #e8f5e9; border-color: #4caf50; color: #111 !important; }
    .result-moderate { background: #fff8e1; border-color: #ff9800; color: #111 !important; }
    .result-severe   { background: #ffebee; border-color: #f44336; color: #111 !important; }
    .result-card h2, .result-card h4, .result-card p { color: #111 !important; }
    .metric-card {
        background: #f0f4ff; border-radius: 10px; padding: 1rem;
        text-align: center; border-left: 4px solid #2d6a9f;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0; padding: 8px 18px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📱 Klasifikasi Tingkat Kecanduan Smartphone</h1>
    <p>Simulasi Model <strong>Multilayer Perceptron (MLP)</strong> — Deteksi Dini Kecanduan Smartphone</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LOAD DATA & MODEL
# ─────────────────────────────────────────────
models, results, scaler, encoders, feature_cols, X_test_sc, y_test = train_model()
df_raw = load_raw_data()
best_model = models[BEST_NAME]
class_names = list(encoders['target'].classes_)

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1, tab2= st.tabs([
    "🔮 Prediksi", "📊 Evaluasi Model"
])

# ══════════════════════════════════════════════
#  TAB 1 — PREDIKSI
# ══════════════════════════════════════════════
with tab1:
    st.subheader("🔮 Uji Coba Prediksi Tingkat Kecanduan")

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        # Opsi preset
        preset_choice = st.selectbox(
            "📋 Gunakan data contoh (opsional):",
            ["— Input manual —"] + list(CONTOH_INPUT.keys()),
        )

        preset = {}
        if preset_choice != "— Input manual —":
            preset = CONTOH_INPUT[preset_choice]
            st.info(f"Data contoh **{preset_choice}** dimuat. Anda bisa ubah nilai di bawah.")

        st.markdown("#### 👤 Data Pengguna")
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Usia (tahun)", 18, 35, int(preset.get("age", 22)))
        with c2:
            gender = st.selectbox("Jenis Kelamin",
                ["Male", "Female", "Other"],
                index=["Male","Female","Other"].index(preset.get("gender","Male")))

        st.markdown("#### 📱 Pola Penggunaan Smartphone")
        c3, c4 = st.columns(2)
        with c3:
            screen_time = st.slider("Screen Time Harian (jam)", 3.0, 12.0,
                float(preset.get("daily_screen_time_hours", 6.0)), step=0.1)
            social_media = st.slider("Sosial Media (jam/hari)", 0.5, 6.0,
                float(preset.get("social_media_hours", 2.0)), step=0.1)
            gaming = st.slider("Gaming (jam/hari)", 0.0, 4.0,
                float(preset.get("gaming_hours", 1.0)), step=0.1)
        with c4:
            work_study = st.slider("Kerja/Belajar via HP (jam/hari)", 0.5, 6.0,
                float(preset.get("work_study_hours", 3.0)), step=0.1)
            sleep = st.slider("Jam Tidur per Malam", 4.5, 9.0,
                float(preset.get("sleep_hours", 7.0)), step=0.1)
            weekend_screen = st.slider("Screen Time Akhir Pekan (jam)", 3.6, 14.9,
                float(preset.get("weekend_screen_time", 8.0)), step=0.1)

        st.markdown("#### 🔔 Aktivitas & Kondisi")
        c5, c6 = st.columns(2)
        with c5:
            notif = st.slider("Notifikasi per Hari", 20, 250,
                int(preset.get("notifications_per_day", 100)))
            app_opens = st.slider("Buka Aplikasi per Hari", 15, 180,
                int(preset.get("app_opens_per_day", 80)))
        with c6:
            stress = st.selectbox("Tingkat Stres",
                ["Low", "Medium", "High"],
                index=["Low","Medium","High"].index(preset.get("stress_level","Medium")))
            impact = st.selectbox("HP Berdampak pada Kerja/Kuliah?",
                ["Yes", "No"],
                index=["Yes","No"].index(preset.get("academic_work_impact","No")))

        st.markdown("#### 🤖 Pilih Model")
        model_options = {
            name: f"[{name}]  {'⭐ Terbaik — ' if name == BEST_NAME else ''}{len(CONFIGS[name])} Hidden Layer, {CONFIGS[name][0]} Neuron  |  Test Acc: {results[name]['test_acc']*100:.2f}%"
            for name in CONFIGS
        }
        selected_model_name = st.radio(
            "Konfigurasi MLP:",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=list(CONFIGS.keys()).index(BEST_NAME),
        )

        predict_btn = st.button("🚀 Prediksi Sekarang", type="primary", use_container_width=True)

    with col_result:
        st.markdown("#### 📋 Hasil Prediksi")

        if predict_btn or preset_choice != "— Input manual —":
            input_data = {
                "age": age, "gender": gender,
                "daily_screen_time_hours": screen_time,
                "social_media_hours": social_media,
                "gaming_hours": gaming,
                "work_study_hours": work_study,
                "sleep_hours": sleep,
                "notifications_per_day": notif,
                "app_opens_per_day": app_opens,
                "weekend_screen_time": weekend_screen,
                "stress_level": stress,
                "academic_work_impact": impact,
            }

            # Prediksi dengan model yang dipilih
            selected_model = models[selected_model_name]
            pred_label, proba, kelas_names = predict(
                input_data, selected_model, scaler, encoders, feature_cols
            )

            # Prediksi semua model untuk perbandingan
            all_preds = {}
            for name, mdl in models.items():
                lbl, prb, _ = predict(input_data, mdl, scaler, encoders, feature_cols)
                all_preds[name] = {"label": lbl, "proba": prb}

            # Tentukan model terbaik untuk input ini (confidence tertinggi pada prediksi mayoritas)
            # Hitung label mayoritas
            from collections import Counter
            label_votes = Counter(v["label"] for v in all_preds.values())
            majority_label = label_votes.most_common(1)[0][0]
            # Model terbaik = confidence tertinggi pada label mayoritas
            best_for_input = max(
                all_preds.items(),
                key=lambda x: x[1]["proba"][list(kelas_names).index(majority_label)]
            )[0]

            INFO = {
                "Mild":     ("🟢", "KECANDUAN RINGAN",   "result-mild",
                             "Penggunaan smartphone masih dalam batas wajar. Pengguna dapat mengontrol diri dengan baik."),
                "Moderate": ("🟡", "KECANDUAN SEDANG",   "result-moderate",
                             "Penggunaan smartphone mulai mengganggu aktivitas. Disarankan membatasi screen time secara bertahap."),
                "Severe":   ("🔴", "KECANDUAN BERAT",    "result-severe",
                             "Penggunaan smartphone sudah berlebihan dan berdampak negatif. Sangat disarankan digital detox."),
            }
            emoji, status, css_class, saran = INFO.get(pred_label,
                ("⚪", pred_label, "result-mild", ""))

            # Badge model terpilih
            is_best = selected_model_name == best_for_input
            model_badge = f"⭐ Model Terbaik untuk Kasus Ini" if is_best else f"Model dipilih: [{selected_model_name}]"
            badge_color = "#1b5e20" if is_best else "#555"

            st.markdown(f"""
            <div class="result-card {css_class}">
                <div style="font-size:0.78rem;font-weight:700;color:{badge_color};
                            background:rgba(0,0,0,0.07);display:inline-block;
                            padding:2px 10px;border-radius:20px;margin-bottom:8px">
                    {model_badge} [{selected_model_name}]
                </div>
                <h2 style="margin:0;color:#111">{emoji} {pred_label.upper()}</h2>
                <h4 style="margin:0.3rem 0 0.5rem;color:#111">{status}</h4>
                <p style="margin:0;color:#111">{saran}</p>
            </div>
            """, unsafe_allow_html=True)

            # Rekomendasi jika model yang dipilih bukan terbaik untuk kasus ini
            if not is_best:
                best_label = all_preds[best_for_input]["label"]
                best_conf  = max(all_preds[best_for_input]["proba"]) * 100
                st.info(
                    f"💡 **Rekomendasi:** Model **[{best_for_input}]** memberikan confidence tertinggi "
                    f"(**{best_conf:.1f}%**) dengan prediksi **{best_label}** untuk data ini."
                )

            # Perbandingan semua model
            st.markdown("##### 🔍 Perbandingan Semua Model")
            colors_map  = {"Mild": "#4caf50", "Moderate": "#ff9800", "Severe": "#f44336"}
            fig_all, axes_all = plt.subplots(1, 4, figsize=(12, 2.8), sharey=True)
            fig_all.suptitle("Probabilitas per Model", fontsize=10, fontweight='bold', y=1.02)

            for ax, (name, res) in zip(axes_all, all_preds.items()):
                bar_colors = [colors_map.get(k, "#888") for k in kelas_names]
                bars = ax.barh(kelas_names, res["proba"] * 100,
                               color=bar_colors, edgecolor='white', height=0.5)
                for bar, val in zip(bars, res["proba"] * 100):
                    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                            f"{val:.0f}%", va='center', fontsize=8)
                ax.set_xlim(0, 115)
                ax.spines[['top','right']].set_visible(False)
                ax.grid(axis='x', alpha=0.3, linestyle='--')
                title_color = "#1b5e20" if name == best_for_input else "#333"
                star = " ⭐" if name == best_for_input else ("  ◀" if name == selected_model_name else "")
                ax.set_title(f"[{name}]{star}", fontsize=9, fontweight='bold',
                             color=title_color,
                             bbox=dict(boxstyle='round,pad=0.3',
                                       facecolor='#c8f5c8' if name == best_for_input else
                                                 '#dce8ff' if name == selected_model_name else 'none',
                                       edgecolor='none'))
                # Highlight predicted label bar
                pred_idx = list(kelas_names).index(res["label"])
                bars[pred_idx].set_edgecolor('#333')
                bars[pred_idx].set_linewidth(1.8)

            fig_all.tight_layout()
            st.pyplot(fig_all)
            plt.close(fig_all)

            st.caption(
                f"⭐ = model terbaik untuk kasus ini  |  ◀ = model yang Anda pilih  |  "
                f"Bar bertepi tebal = hasil prediksi model tersebut"
            )

            # Input summary
            with st.expander("📋 Ringkasan Data Input"):
                labels = {
                    "age": "Usia", "gender": "Jenis Kelamin",
                    "daily_screen_time_hours": "Screen Time Harian (jam)",
                    "social_media_hours": "Sosial Media (jam)", "gaming_hours": "Gaming (jam)",
                    "work_study_hours": "Kerja/Belajar (jam)", "sleep_hours": "Tidur (jam)",
                    "notifications_per_day": "Notifikasi/hari", "app_opens_per_day": "Buka App/hari",
                    "weekend_screen_time": "Screen Time Weekend (jam)",
                    "stress_level": "Stres", "academic_work_impact": "Dampak Akademik/Kerja",
                }
                rows = [[labels.get(k, k), v] for k, v in input_data.items()]
                st.table(pd.DataFrame(rows, columns=["Fitur", "Nilai"]))
        else:
            st.info("👈 Isi form di sebelah kiri lalu klik **Prediksi Sekarang**")
            st.markdown("""
            **Cara Penggunaan:**
            1. Pilih data contoh dari dropdown **atau** isi input manual
            2. Pilih konfigurasi model MLP yang ingin digunakan
            3. Klik tombol **Prediksi Sekarang**
            4. Lihat hasil prediksi, perbandingan semua model, dan rekomendasi model terbaik
            """)

# ══════════════════════════════════════════════
#  TAB 2 — EVALUASI MODEL
# ══════════════════════════════════════════════
with tab2:
    st.subheader("📊 Perbandingan 4 Konfigurasi MLP")

    # Metrics cards
    cols_m = st.columns(4)
    for i, (name, r) in enumerate(results.items()):
        with cols_m[i]:
            mark = " ⭐" if name == BEST_NAME else ""
            st.metric(
                label=f"[{name}]{mark}",
                value=f"{r['test_acc']*100:.2f}%",
                delta=f"Train: {r['train_acc']*100:.2f}%",
            )

    st.divider()

    col_bar, col_cm = st.columns([1.2, 1])

    with col_bar:
        st.markdown("#### Train vs Test Accuracy")
        fig_bar, axes_bar = plt.subplots(1, 2, figsize=(10, 4))
        names      = list(results.keys())
        train_accs = [results[n]['train_acc'] * 100 for n in names]
        test_accs  = [results[n]['test_acc']  * 100 for n in names]
        x      = np.arange(len(names))
        width  = 0.35

        ax1 = axes_bar[0]
        b1 = ax1.bar(x - width/2, train_accs, width, label='Training', color='#4C72B0', alpha=0.85)
        b2 = ax1.bar(x + width/2, test_accs,  width, label='Testing',  color='#DD8452', alpha=0.85)
        ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=9)
        ax1.set_ylabel("Akurasi (%)"); ax1.set_title("Train vs Test")
        ax1.set_ylim(min(test_accs)-5, 101); ax1.legend(fontsize=8)
        ax1.grid(axis='y', alpha=0.3)
        for bar in [*b1, *b2]:
            ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                     f"{bar.get_height():.1f}%", ha='center', va='bottom', fontsize=7)

        ax2 = axes_bar[1]
        palette4 = ['#4C72B0','#55A868','#DD8452','#C44E52']
        b3 = ax2.bar(names, test_accs, color=palette4, alpha=0.85, edgecolor='black', lw=0.6)
        ax2.set_ylabel("Test Accuracy (%)"); ax2.set_title("Test Accuracy per Config")
        ax2.set_ylim(min(test_accs)-5, 101); ax2.grid(axis='y', alpha=0.3)
        for bar, it, acc in zip(b3, [results[n]['iters'] for n in names], test_accs):
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                     f"{acc:.1f}%\n(iter={it})", ha='center', va='bottom', fontsize=7)

        fig_bar.tight_layout()
        st.pyplot(fig_bar)
        plt.close(fig_bar)

        # Loss curves
        st.markdown("#### Kurva Loss Training")
        fig_loss, axes_loss = plt.subplots(2, 2, figsize=(10, 6))
        for ax, (name, r) in zip(axes_loss.flatten(), results.items()):
            clf = models[name]
            ax.plot(clf.loss_curve_, color='#4C72B0', lw=1.8, label='Train Loss')
            if hasattr(clf, 'validation_scores_') and clf.validation_scores_ is not None:
                val_loss = [1-s for s in clf.validation_scores_]
                ax.plot(val_loss, color='#DD8452', lw=1.8, ls='--', label='Val Loss')
            ax.set_title(f"[{name}]  (iter={r['iters']})", fontsize=10)
            ax.set_xlabel("Iterasi"); ax.set_ylabel("Loss")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
            if name == BEST_NAME:
                ax.set_title(f"[{name}] ⭐  (iter={r['iters']})", fontsize=10)
        fig_loss.suptitle("Kurva Loss per Konfigurasi", fontsize=12, fontweight='bold')
        fig_loss.tight_layout()
        st.pyplot(fig_loss)
        plt.close(fig_loss)

    with col_cm:
        st.markdown("#### Confusion Matrix")
        selected_config = st.selectbox("Pilih konfigurasi:", list(results.keys()),
                                       index=list(results.keys()).index(BEST_NAME))
        y_pred_sel = results[selected_config]['y_pred']
        cm = confusion_matrix(y_test, y_pred_sel)

        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax_cm, colorbar=False, cmap='Blues')
        ax_cm.set_title(f"[{selected_config}] — Acc: {results[selected_config]['test_acc']*100:.2f}%",
                        fontsize=11)
        ax_cm.set_xlabel("Prediksi"); ax_cm.set_ylabel("Aktual")
        fig_cm.tight_layout()
        st.pyplot(fig_cm)
        plt.close(fig_cm)

        # Classification report
        st.markdown("##### Classification Report")
        report = classification_report(y_test, y_pred_sel, target_names=class_names,
                                       output_dict=True)
        report_df = pd.DataFrame(report).T.round(3)
        report_df = report_df.drop(index=['accuracy'], errors='ignore')
        st.dataframe(report_df.style.format("{:.3f}").background_gradient(
            cmap='YlGn', subset=['precision','recall','f1-score']), use_container_width=True)
