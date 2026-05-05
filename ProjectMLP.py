import os
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

# config path
DATASET_PATH = "Dataset/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv"
OUTPUT_DIR   = "Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.2

# Print Header
def header(nomor, judul):
    print(f"\n{'='*70}")
    print(f"  [{nomor}] {judul}")
    print(f"{'='*70}")

#  Dataset dari csv
header(1, "DATASET")

df_raw = pd.read_csv(DATASET_PATH)

print(f"  Jumlah baris    : {df_raw.shape[0]:,}")
print(f"  Jumlah kolom    : {df_raw.shape[1]}")
print(f"\n  Nama Kolom:")
for col in df_raw.columns:
    print(f"    - {col}  ({df_raw[col].dtype})")

print(f"\n  5 baris pertama:")
print(df_raw.head().to_string(index=False))

print(f"\n  Distribusi Target (addiction_level) sebelum cleaning:")
print(df_raw['addiction_level'].value_counts(dropna=False).to_string())

# Preprocessing
header(2, "PREPROCESSING")

# Hapus kolom ID karena tidak relevan untuk model nya
drop_cols = ['transaction_id', 'user_id', 'addicted_label']
df = df_raw.drop(columns=drop_cols)
print(f"  Kolom dihapus   : {drop_cols}")

# Hapus baris yang kosong
before = len(df)
df = df.dropna(subset=['addiction_level'])
after  = len(df)
print(f"  Baris dihapus (NaN target) : {before - after}")
print(f"  Baris tersisa              : {after:,}")

# Cek missing value fitur
print(f"\n  Missing value per kolom (setelah drop NaN target):")
mv = df.isnull().sum()
print(mv[mv > 0] if mv.sum() > 0 else "    Tidak ada missing value.")

# Encode fitur kategorikal
cat_cols   = df.select_dtypes(include='object').columns.tolist()
cat_cols   = [c for c in cat_cols if c != 'addiction_level']
print(f"\n  Kolom kategorikal yang di-encode : {cat_cols}")

le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le
    print(f"    {col:30s} -> {list(le.classes_)}")

# Encode target
le_target = LabelEncoder()
df['addiction_level'] = le_target.fit_transform(df['addiction_level'])
print(f"\n  Label Target:")
for idx, cls in enumerate(le_target.classes_):
    print(f"    {idx} = {cls}")

# Pisahkan fitur dan target
X = df.drop(columns=['addiction_level'])
y = df['addiction_level']

print(f"\n  Fitur (X) shape : {X.shape}")
print(f"  Target (y) shape: {y.shape}")
print(f"\n  Distribusi Target (setelah encode):")
for val, cnt in y.value_counts().items():
    label = le_target.inverse_transform([val])[0]
    print(f"    {val} ({label:10s}) : {cnt:,}  ({cnt/len(y)*100:.1f}%)")

# Normalisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"\n  Normalisasi    : StandardScaler (mean=0, std=1) diterapkan.")

# Split data training dan testing
header(3, "SPLIT DATA (TRAIN / TEST)")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"  Rasio split    : {int((1-TEST_SIZE)*100)}% Train / {int(TEST_SIZE*100)}% Test")
print(f"  Data training  : {X_train.shape[0]:,} sampel")
print(f"  Data testing   : {X_test.shape[0]:,} sampel")
print(f"\n  Distribusi kelas di TRAIN:")
for val in sorted(y_train.unique()):
    cnt   = (y_train == val).sum()
    label = le_target.inverse_transform([val])[0]
    print(f"    {label:10s}: {cnt:,}  ({cnt/len(y_train)*100:.1f}%)")
print(f"\n  Distribusi kelas di TEST:")
for val in sorted(y_test.unique()):
    cnt   = (y_test == val).sum()
    label = le_target.inverse_transform([val])[0]
    print(f"    {label:10s}: {cnt:,}  ({cnt/len(y_test)*100:.1f}%)")

# Konfig MLP
header(4, "KONFIGURASI")

"""
Perbandingan:
  - Hidden Layer : 1 layer  vs  2 layer
  - Neuron       : 16       vs  32

Kombinasi:
  A) (16,)          1 hidden layer, 16 neuron
  B) (32,)          1 hidden layer, 32 neuron
  C) (16, 16)       2 hidden layer, masing-masing 16 neuron
  D) (32, 32)       2 hidden layer, masing-masing 32 neuron
"""

configs = {
    "1HL-16N"  : (16,),
    "1HL-32N"  : (32,),
    "2HL-16N"  : (16, 16),
    "2HL-32N"  : (32, 32),
}

MLP_PARAMS = dict(
    activation   = 'relu',
    solver       = 'adam',
    max_iter     = 500,
    random_state = RANDOM_STATE,
    early_stopping = True,
    validation_fraction = 0.1,
    n_iter_no_change = 20,
    verbose      = False,
)

print(f"\n  Hyperparameter umum:")
print(f"    Activation     : relu")
print(f"    Solver         : adam")
print(f"    Max iterasi    : 500")
print(f"    Early stopping : True (patience=20, val_frac=10%)")
print(f"\n  Konfigurasi yang diuji:")
for name, hl in configs.items():
    n_layer = len(hl)
    print(f"    [{name}]  hidden_layer_sizes={hl}  "
          f"→ {n_layer} hidden layer, {hl[0]} neuron/layer")

# Training
header(5, "TRAINING")

models   = {}
results  = {}

for name, hl in configs.items():
    print(f"\n  Training [{name}] hidden_layer_sizes={hl} ...")
    clf = MLPClassifier(hidden_layer_sizes=hl, **MLP_PARAMS)
    clf.fit(X_train, y_train)
    models[name] = clf

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc  = accuracy_score(y_test,  clf.predict(X_test))
    iters     = clf.n_iter_

    results[name] = {
        'model'     : clf,
        'train_acc' : train_acc,
        'test_acc'  : test_acc,
        'iters'     : iters,
    }

    print(f"    Iterasi selesai   : {iters}")
    print(f"    Akurasi Training  : {train_acc*100:.2f}%")
    print(f"    Akurasi Testing   : {test_acc*100:.2f}%")

# Evaluasi -> Akurasi dan ConMatrix
header(6, "EVALUASI")

CLASS_NAMES = list(le_target.classes_)
print(f"\n  {'Konfigurasi':<12} {'Train Acc':>12} {'Test Acc':>12} {'Iterasi':>10}")
print(f"  {'-'*50}")
for name, r in results.items():
    print(f"  {name:<12} {r['train_acc']*100:>11.2f}% "
          f"{r['test_acc']*100:>11.2f}% {r['iters']:>10}")

for name, r in results.items():
    print(f"\n  ── Classification Report [{name}] ──")
    y_pred = models[name].predict(X_test)
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Confusion Matrix — Perbandingan 4 Konfigurasi MLP\n"
    "Klasifikasi Tingkat Kecanduan Smartphone",
    fontsize=14, fontweight='bold', y=1.01
)

axes_flat = axes.flatten()
for ax, (name, r) in zip(axes_flat, results.items()):
    y_pred = models[name].predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)
    disp   = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f"[{name}]  Acc = {r['test_acc']*100:.2f}%", fontsize=12)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")

plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Confusion matrix disimpan → {cm_path}")

# Analisis hasil
header(7, "ANALISIS HASIL")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "Perbandingan Akurasi 4 Konfigurasi MLP\n"
    "Klasifikasi Tingkat Kecanduan Smartphone",
    fontsize=14, fontweight='bold'
)

names      = list(results.keys())
train_accs = [results[n]['train_acc'] * 100 for n in names]
test_accs  = [results[n]['test_acc']  * 100 for n in names]
x          = np.arange(len(names))
width      = 0.35
colors_tr  = ['#4C72B0', '#4C72B0', '#DD8452', '#DD8452']
colors_te  = ['#55A868', '#55A868', '#C44E52', '#C44E52']

ax1 = axes[0]
bars1 = ax1.bar(x - width/2, train_accs, width, label='Training', color='#4C72B0', alpha=0.85)
bars2 = ax1.bar(x + width/2, test_accs,  width, label='Testing',  color='#DD8452', alpha=0.85)
ax1.set_xticks(x)
ax1.set_xticklabels(names, fontsize=11)
ax1.set_ylabel("Akurasi (%)")
ax1.set_title("Train vs Test Accuracy")
ax1.set_ylim(min(test_accs) - 5, 100)
ax1.legend()
ax1.grid(axis='y', linestyle='--', alpha=0.5)
for bar in bars1:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{bar.get_height():.1f}%", ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{bar.get_height():.1f}%", ha='center', va='bottom', fontsize=9)

ax2 = axes[1]
color_map = ['#4C72B0', '#55A868', '#DD8452', '#C44E52']
bars3 = ax2.bar(names, test_accs, color=color_map, alpha=0.85, edgecolor='black', linewidth=0.7)
ax2.set_ylabel("Akurasi Testing (%)")
ax2.set_title("Test Accuracy per Konfigurasi")
ax2.set_ylim(min(test_accs) - 5, 100)
ax2.grid(axis='y', linestyle='--', alpha=0.5)
iters_list = [results[n]['iters'] for n in names]
for bar, it, acc in zip(bars3, iters_list, test_accs):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{acc:.2f}%\n(iter={it})", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
bar_path = os.path.join(OUTPUT_DIR, "perbandingan_akurasi.png")
plt.savefig(bar_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Bar chart akurasi disimpan → {bar_path}")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle(
    "Kurva Loss Training per Iterasi\n"
    "Klasifikasi Tingkat Kecanduan Smartphone",
    fontsize=14, fontweight='bold'
)

for ax, (name, r) in zip(axes.flatten(), results.items()):
    clf = models[name]
    ax.plot(clf.loss_curve_, color='#4C72B0', linewidth=1.8, label='Train Loss')
    if hasattr(clf, 'validation_scores_') and clf.validation_scores_ is not None:
        val_loss = [1 - s for s in clf.validation_scores_]
        ax.plot(val_loss, color='#DD8452', linewidth=1.8, linestyle='--', label='Val Loss')
    ax.set_title(f"[{name}]  (iters={r['iters']})")
    ax.set_xlabel("Iterasi")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=9)
    ax.grid(linestyle='--', alpha=0.5)

plt.tight_layout()
loss_path = os.path.join(OUTPUT_DIR, "kurva_loss.png")
plt.savefig(loss_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Kurva loss disimpan        → {loss_path}")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Eksplorasi Dataset — Distribusi Fitur Utama",
             fontsize=13, fontweight='bold')

ax = axes[0]
counts = df_raw['addiction_level'].value_counts().dropna()
colors_pie = ['#4C72B0', '#DD8452', '#55A868']
ax.pie(counts, labels=counts.index, autopct='%1.1f%%',
       colors=colors_pie, startangle=140, textprops={'fontsize': 11})
ax.set_title("Distribusi Addiction Level")

ax = axes[1]
df_plot = df_raw.dropna(subset=['addiction_level'])
order   = ['Mild', 'Moderate', 'Severe']
palette = {'Mild': '#55A868', 'Moderate': '#4C72B0', 'Severe': '#C44E52'}
for lvl in order:
    vals = df_plot[df_plot['addiction_level'] == lvl]['daily_screen_time_hours']
    ax.hist(vals, bins=25, alpha=0.6, label=lvl, color=palette[lvl])
ax.set_xlabel("Daily Screen Time (jam)")
ax.set_ylabel("Frekuensi")
ax.set_title("Screen Time per Addiction Level")
ax.legend()
ax.grid(axis='y', alpha=0.4)

ax = axes[2]
cross = pd.crosstab(df_plot['stress_level'], df_plot['addiction_level'])
cross = cross.reindex(columns=order, fill_value=0)
cross.plot(kind='bar', ax=ax, color=[palette[c] for c in order],
           edgecolor='black', linewidth=0.5)
ax.set_title("Stress Level vs Addiction Level")
ax.set_xlabel("Stress Level")
ax.set_ylabel("Jumlah")
ax.tick_params(axis='x', rotation=0)
ax.legend(title="Addiction", fontsize=9)
ax.grid(axis='y', alpha=0.4)

plt.tight_layout()
eda_path = os.path.join(OUTPUT_DIR, "eda_distribusi.png")
plt.savefig(eda_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Plot EDA disimpan          → {eda_path}")

print(f"\n{'─'*70}")
print("  RINGKASAN ANALISIS PERBANDINGAN MODEL")
print(f"{'─'*70}")

best_name = max(results, key=lambda n: results[n]['test_acc'])
best_acc  = results[best_name]['test_acc']

print(f"\n  Model terbaik  : [{best_name}]  (Test Accuracy = {best_acc*100:.2f}%)")
print()

# Perbandingan 1 vs 2 hidden layer
acc_1hl = np.mean([results['1HL-16N']['test_acc'], results['1HL-32N']['test_acc']]) * 100
acc_2hl = np.mean([results['2HL-16N']['test_acc'], results['2HL-32N']['test_acc']]) * 100
print(f"  ▸ 1 Hidden Layer (rata-rata) : {acc_1hl:.2f}%")
print(f"  ▸ 2 Hidden Layer (rata-rata) : {acc_2hl:.2f}%")
if acc_2hl > acc_1hl:
    selisih = acc_2hl - acc_1hl
    print(f"    → 2 Hidden Layer lebih baik +{selisih:.2f}% dari 1 Hidden Layer")
else:
    selisih = acc_1hl - acc_2hl
    print(f"    → 1 Hidden Layer lebih baik +{selisih:.2f}% dari 2 Hidden Layer")

# Perbandingan 16 vs 32 neuron
acc_16 = np.mean([results['1HL-16N']['test_acc'], results['2HL-16N']['test_acc']]) * 100
acc_32 = np.mean([results['1HL-32N']['test_acc'], results['2HL-32N']['test_acc']]) * 100
print(f"\n  ▸ 16 Neuron (rata-rata) : {acc_16:.2f}%")
print(f"  ▸ 32 Neuron (rata-rata) : {acc_32:.2f}%")
if acc_32 > acc_16:
    selisih = acc_32 - acc_16
    print(f"    → 32 Neuron lebih baik +{selisih:.2f}% dari 16 Neuron")
else:
    selisih = acc_16 - acc_32
    print(f"    → 16 Neuron lebih baik +{selisih:.2f}% dari 32 Neuron")

# Detail tiap model
print(f"\n  Detail semua konfigurasi:")
print(f"  {'Konfigurasi':<12} {'Layer':<8} {'Neuron':<8} {'Train Acc':>10} {'Test Acc':>10} {'Iterasi':>8}")
print(f"  {'-'*60}")
for name, r in results.items():
    n_layer = len(configs[name])
    n_neuron = configs[name][0]
    mark = " ◀ BEST" if name == best_name else ""
    print(f"  {name:<12} {n_layer:<8} {n_neuron:<8} "
          f"{r['train_acc']*100:>9.2f}% {r['test_acc']*100:>9.2f}%"
          f" {r['iters']:>8}{mark}")

