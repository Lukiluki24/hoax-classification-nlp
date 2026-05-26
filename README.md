# Hoax Classification NLP — Deteksi Judul Berita Hoaks Bahasa Indonesia

Sistem klasifikasi biner (Faktual vs Hoax) untuk **judul berita** berbahasa Indonesia. Proyek ini membandingkan empat arsitektur NLP dengan kompleksitas berbeda pada split data yang identik:

1. **Ensemble Baseline** — TF-IDF + Soft-Voting (Random Forest + SVM + Naive Bayes)
2. **BiLSTM** — embedding + bidirectional LSTM 2-layer + mean-pooling
3. **IndoBERT Fine-Tuned** — `indobenchmark/indobert-base-p2` + Bayesian (Optuna TPE) hyperparameter search
4. **mBERT Fine-Tuned** — `bert-base-multilingual-cased` (pembanding multilingual)

Hasil terbaik: **IndoBERT — Accuracy 90.55%, F1 89.49%, Recall 90.78%** pada test set.

> Authors: Lucky Wijaya (2802443975), Kelvin Yohanes Justinus (2802447576), Garren Tanavaro (2802516182)
>
> Metodologi IndoBERT mengikuti Simanjuntak et al. (2024), *"Research and Analysis of IndoBERT Hyperparameter Tuning in Fake News Detection"* (JNTETI Vol. 13 No. 1, DOI: 10.22146/jnteti.v1311.8532). Ringkasan paper: [indobert_paper_summary.md](indobert_paper_summary.md).

---

## Daftar Isi

1. [Struktur Proyek](#1-struktur-proyek)
2. [Dataset](#2-dataset)
3. [Pipeline 0 — Preprocessing Umum & Splitting](#3-pipeline-0--preprocessing-umum--splitting)
4. [Pipeline A — Input untuk Model Klasik & BiLSTM](#4-pipeline-a--input-untuk-model-klasik--bilstm)
5. [Pipeline B — Input untuk Transformer (BERT-family)](#5-pipeline-b--input-untuk-transformer-bert-family)
6. [Model 1 — Ensemble Baseline (TF-IDF + Soft Voting)](#6-model-1--ensemble-baseline-tf-idf--soft-voting)
7. [Model 2 — Bidirectional LSTM](#7-model-2--bidirectional-lstm)
8. [Model 3 — IndoBERT Fine-Tuned](#8-model-3--indobert-fine-tuned-model-utama)
9. [Model 4 — mBERT Fine-Tuned](#9-model-4--mbert-fine-tuned-pembanding-multilingual)
10. [Modal Cloud Training (A10G GPU)](#10-modal-cloud-training-a10g-gpu)
11. [Evaluation Pipeline](#11-evaluation-pipeline)
12. [CLI Inference](#12-cli-inference)
13. [Hasil Akhir & Tabel Perbandingan](#13-hasil-akhir--tabel-perbandingan)
14. [Reproduksi Penuh](#14-reproduksi-penuh)
15. [Dependencies](#15-dependencies)

---

## 1. Struktur Proyek

```
project/
├── data/
│   ├── raw/
│   │   ├── turnbackhoax.csv        # Sumber utama — judul dengan tag label [SALAH]/[BENAR]/...
│   │   └── detik.csv               # Tambahan kelas Faktual (semua di-label 0)
│   └── processed/
│       ├── detik_clean.csv
│       ├── pipeline_a/             # train/val/test.csv — input Ensemble & BiLSTM
│       └── pipeline_b/             # train/val/test.csv — input BERT-family
│
├── src/
│   ├── preprocess.py               # Pipeline 0/A/B + stratified 70/10/20 split
│   ├── train_baseline.py           # Ensemble RF + SVM + NB (soft voting)
│   ├── train_bilstm.py             # BiLSTM 2-layer + global mean-pool
│   ├── train_indobert.py           # IndoBERT/mBERT fine-tune + Optuna Bayesian
│   ├── evaluate.py                 # Metrik + confusion matrix PNG
│   ├── eval_indobert_local.py      # Quick eval lokal untuk model BERT
│   └── predict.py                  # CLI inference interaktif
│
├── modal/
│   ├── train_indobert_modal.py     # Training di Modal cloud (A10G GPU)
│   └── README.md                   # Workflow Modal
│
├── models/                         # Artifact terlatih (di-.gitignore)
│   ├── ensemble_baseline.pkl
│   ├── bilstm_model/bilstm_weights.pth
│   ├── indobert_finetuned/         # HF format + best_params.json + test_metrics.json
│   └── mbert_finetuned/            # HF format + best_params.json + test_metrics.json
│
├── results/
│   ├── cm_ensemble_baseline.png
│   ├── cm_bilstm.png
│   ├── cm_indobert_finetuned.png
│   ├── cm_mbert_finetuned.png
│   ├── data_split_visualization.png
│   └── result.md                   # Ringkasan numerik per model
│
├── indobert_paper_summary.md       # Ringkasan paper rujukan
├── requirements.txt
└── README.md                       # (file ini)
```

---

## 2. Dataset

### 2.1 Sumber

| File | Peran | Konten | Sumber |
|---|---|---|---|
| [data/raw/turnbackhoax.csv](data/raw/turnbackhoax.csv) | Sumber label hoax + sebagian factual | Kolom `Title` berisi judul dengan tag `[TAG]` di awal | Scrape TurnBackHoax.ID |
| [data/raw/detik.csv](data/raw/detik.csv) | Tambahan kelas Faktual | Kolom `Title` (semua diberi label `0`) | Scrape detik.com |

### 2.2 Skema label biner

`0 = Faktual`, `1 = Hoax`.

Untuk `turnbackhoax.csv`, fungsi [`extract_label()`](src/preprocess.py#L19-L33) memparse pola `\[([A-Z\s]+)\]` dari judul, lalu memetakan tag tersebut ke label berdasarkan dua set yang ditulis tangan setelah inspeksi data:

- **HOAX_PREFIXES → label 1:** `SALAH`, `KLARIFIKASI`, `HOAX`, `HOAKS`, `FITNAH`, `PENIPUAN`, `DISINFORMASI`, `MISLEADING`
- **REAL_PREFIXES → label 0:** `BENAR`, `FAKTA`
- **Tag lain → label –1** (di-drop sebelum split)

Tag kemudian dihapus dari teks judul (`re.sub(r'\[.*?\]', '', text)`) agar model tidak belajar dari leakage label embedded.

### 2.3 Komposisi akhir & split

Setelah pembersihan, deduplikasi (case-insensitive pada judul), dan stratified split `70/10/20`:

| Split | Baris | Proporsi |
|---|---:|---:|
| Train | 2.737 | 70% |
| Validation | 391 | 10% |
| Test | 783 | 20% |
| **Total** | **3.911** | 100% |

Distribusi kelas tiap split divisualisasikan ke [results/data_split_visualization.png](results/data_split_visualization.png).

> **Anti-leakage:** Deduplikasi dilakukan **sebelum** split bukan sesudah — mencegah judul yang sama muncul di train dan test sekaligus.

---

## 3. Pipeline 0 — Preprocessing Umum & Splitting

Implementasi: [src/preprocess.py](src/preprocess.py). Jalankan:

```powershell
.\.venv\Scripts\python.exe src\preprocess.py
```

Tahapan urutan:

### 3.1 General cleaning ([`general_clean`](src/preprocess.py#L11-L16))

Diterapkan ke semua judul di kedua sumber:

1. Hapus URL: `re.sub(r'http\S+|www\S+', '', text)`
2. Hapus karakter non-ASCII di luar Latin Extended (`[^\x00-\x7FÀ-ɏ]+`) — membersihkan emoji, mojibake, kontrol karakter.
3. Normalisasi whitespace (`\s+ → ' '`).

### 3.2 Ekstraksi label TurnBackHoax

Untuk setiap baris di `turnbackhoax.csv`:
- Jalankan `extract_label(title)` → tuple `(label, clean_title)`.
- Buang baris dengan `label == -1`.
- Print distribusi label hasil ekstraksi.

### 3.3 Penggabungan dengan Detik

Jika `detik.csv` tersedia: load, general-clean, set semua `label = 0`, concat ke TurnBackHoax. Bila tidak ada, warning dan lanjut dengan TurnBackHoax saja.

### 3.4 Deduplikasi

```python
df['_norm'] = df['title'].str.lower().str.strip()
df = df.drop_duplicates(subset=['_norm']).drop(columns=['_norm'])
```

Sengaja dilakukan **sebelum** split untuk eliminate cross-split leakage.

### 3.5 Stratified split 70/10/20

Dua-tahap dengan `random_state=42`:

```python
# Tahap 1: 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42)

# Tahap 2: temp di-split 1/3 (val) + 2/3 (test)
# → val = 10% total, test = 20% total
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=2/3, stratify=y_temp, random_state=42)
```

### 3.6 Visualisasi distribusi

`seaborn.barplot` (Split × Label) → [results/data_split_visualization.png](results/data_split_visualization.png).

Setelah Pipeline 0, dua pipeline preprocessing berbeda diturunkan secara paralel dari split yang sama (Pipeline A dan B di bawah).

---

## 4. Pipeline A — Input untuk Model Klasik & BiLSTM

Dipakai oleh **Ensemble Baseline** dan **BiLSTM**.

Output: `data/processed/pipeline_a/{train,val,test}.csv` (kolom: `text, label`).

### Transformasi ([src/preprocess.py:87-91](src/preprocess.py#L87-L91))

```python
def pipeline_a(text: str) -> str:
    text = str(text).lower()                            # 1. case folding
    text = re.sub(r'[^a-z\s]', '', text)                # 2. hapus angka & tanda baca
    text = ' '.join(w for w in text.split()
                    if w not in indonesia_s)            # 3. stopword removal NLTK ID
    return re.sub(r'\s+', ' ', text).strip()            # 4. squash whitespace
```

| Tahap | Tujuan |
|---|---|
| **Case folding** | Menyatukan varian casing (`Indonesia`/`indonesia`/`INDONESIA` → satu token) |
| **Remove non-letters** | Membuang angka & tanda baca yang tidak menambah sinyal untuk TF-IDF unigram/bigram |
| **Stopword removal** | Korpus `nltk.corpus.stopwords.words('indonesian')` (otomatis di-`download` saat preprocess) — menghapus kata fungsional ("yang", "di", "ke") yang mendominasi frekuensi tanpa nilai diskriminatif |

> Contoh: `"Gemuruh Suara Asteroid yang Sedang Melintas"` → `"gemuruh suara asteroid melintas"`.

---

## 5. Pipeline B — Input untuk Transformer (BERT-family)

Dipakai oleh **IndoBERT** dan **mBERT**.

Output: `data/processed/pipeline_b/{train,val,test}.csv` — **teks mentah hasil Pipeline 0 saja**, **tanpa** lowercasing, **tanpa** punctuation removal, **tanpa** stopword removal.

**Rasional:** tokenizer WordPiece BERT dipre-training pada teks cased dengan tanda baca utuh. Menormalkan teks input akan:
- Memecah token informatif menjadi lebih banyak subword
- Menghapus tanda baca yang justru jadi cue (mis. tanda tanya untuk klaim, kutip untuk quote)
- Stopword Indonesia yang dihapus di Pipeline A justru bagian dari konteks gramatikal yang dimanfaatkan self-attention

Tokenisasi sesungguhnya dilakukan **saat training**, bukan saat preprocess — lihat §8.

---

## 6. Model 1 — Ensemble Baseline (TF-IDF + Soft Voting)

Implementasi: [src/train_baseline.py](src/train_baseline.py). Input: **Pipeline A**.

### 6.1 Vektorisasi

```python
TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
```

- **TF-IDF** (Term Frequency × Inverse Document Frequency) menimbang token: term yang sering muncul di satu dokumen tetapi jarang di korpus mendapat bobot tinggi.
- **Unigram + Bigram** (`ngram_range=(1,2)`) → menangkap kolokasi seperti `"breaking news"`, `"jangan percaya"`.
- **Max 10.000 fitur** — batas dimensi untuk regularisasi & efisiensi.

Fit pada train, transform pada test (tidak ada hyperparameter tuning di model ini, jadi val set tidak dipakai).

### 6.2 Tiga base learner

| Estimator | Konfigurasi | Karakter |
|---|---|---|
| `RandomForestClassifier` | `n_estimators=200, random_state=42` | Non-linear, robust terhadap fitur korelatif & noise |
| `SVC` | `kernel='rbf', probability=True, random_state=42` | Non-linear margin maximization, kuat di high-dim TF-IDF |
| `MultinomialNB` | default | Generative, baseline cepat & efektif pada count/tfidf fitur teks |

### 6.3 Soft voting ensemble

```python
ensemble = VotingClassifier(
    estimators=[('rf', rf), ('svm', svm), ('nb', nb)],
    voting='soft'
)
```

`voting='soft'` merata-ratakan **probabilitas output** ketiga estimator (bukan vote diskrit). Keuntungan: model yang sangat yakin (probabilitas dekat 1) mendominasi tanpa benar-benar mendiktekan keputusan — confidence dari ketiga learner dimanfaatkan.

### 6.4 Artifact & evaluasi

Dump joblib **single file** `models/ensemble_baseline.pkl` berisi dict `{'model': ensemble, 'vectorizer': tfidf}` — vectorizer wajib disertakan agar inference baru bisa men-transform input identik.

Test predict → `evaluate(...)` → metrik + [results/cm_ensemble_baseline.png](results/cm_ensemble_baseline.png).

```powershell
.\.venv\Scripts\python.exe src\train_baseline.py
```

---

## 7. Model 2 — Bidirectional LSTM

Implementasi: [src/train_bilstm.py](src/train_bilstm.py). Input: **Pipeline A**.

### 7.1 Vocab building (train-only)

[`build_vocab_and_sequences`](src/train_bilstm.py#L25-L41):
- `CountVectorizer(max_features=10000)` di-fit **hanya pada train set** (mencegah leakage statistik korpus dari val/test).
- Mapping `word → index+1`. Index `0` dicadangkan untuk padding token.
- Val & test memakai vocab yang sama; **out-of-vocab di-skip** (bukan UNK token tersendiri).

### 7.2 Sequence encoding

Tiap kalimat:
1. Split whitespace → list kata
2. Map ke list integer (skip OOV)
3. Truncate jika `len > max_len=100`, else pad dengan `0` di kanan ke length 100
4. `torch.tensor(dtype=torch.long)`

### 7.3 Arsitektur ([`BiLSTMClassifier`](src/train_bilstm.py#L10-L23))

```python
nn.Embedding(vocab_size, embed_dim=128, padding_idx=0)
   ↓
nn.LSTM(input=128, hidden=256, num_layers=2,
        batch_first=True, bidirectional=True, dropout=0.3)
   ↓
torch.mean(out, dim=1)                # global average pooling sepanjang sequence
   ↓
nn.Linear(hidden_dim * 2, 2)          # ×2 karena bidirectional → 512 → 2 logit
```

**Pilihan desain:**

| Komponen | Alasan |
|---|---|
| `padding_idx=0` | Gradient untuk token padding di-zero-kan otomatis |
| `bidirectional=True` | LSTM forward + backward; representasi tiap token tahu konteks kiri-kanan |
| `num_layers=2 + dropout=0.3` | Dua layer untuk kapasitas; dropout antar-layer (PyTorch `dropout=` LSTM hanya aktif kalau `num_layers > 1`) |
| `torch.mean(out, dim=1)` | **Global mean pooling** alih-alih ambil last hidden state — lebih robust terhadap posisi padding & semantik kalimat panjang |

### 7.4 Training

| Parameter | Nilai |
|---|---|
| Loss | `CrossEntropyLoss` |
| Optimizer | `Adam(lr=1e-3)` |
| Epochs | **5** (fix, tanpa early stopping) |
| Batch size | 32 |
| Device | CUDA jika tersedia, else CPU |

State dict disimpan ke `models/bilstm_model/bilstm_weights.pth`, eval test → [results/cm_bilstm.png](results/cm_bilstm.png).

```powershell
.\.venv\Scripts\python.exe src\train_bilstm.py
```

---

## 8. Model 3 — IndoBERT Fine-Tuned (Model Utama)

Implementasi: [src/train_indobert.py](src/train_indobert.py). Input: **Pipeline B**.

Base model: `indobenchmark/indobert-base-p2` — BERT base (12 layer, 768 hidden, 12 attention heads) di-pretrain pada korpus Bahasa Indonesia.

### 8.1 Pipeline penuh

1. Load train/val/test dari `data/processed/pipeline_b/`, drop NaN.
2. Tokenize ketiga split dengan `AutoTokenizer.from_pretrained(model_name)`.
3. Bungkus ke `TensorDataset(input_ids, attention_mask, labels)`.
4. **Optuna Bayesian search** dengan TPESampler (lihat §8.3).
5. **Final training** dengan best params + early stopping.
6. Save model + tokenizer + `best_params.json` + `test_metrics.json` ke `models/indobert_finetuned/`.
7. Evaluate test → metrik + confusion matrix PNG.

### 8.2 Tokenisasi

[src/train_indobert.py:44-53](src/train_indobert.py#L44-L53):

```python
tokenizer(
    list(texts),
    add_special_tokens=True,   # tambah [CLS] di awal, [SEP] di akhir
    max_length=128,            # MAX_LEN
    padding='max_length',      # pad ke 128 token (tensor uniform)
    truncation=True,           # potong > 128
    return_attention_mask=True,
    return_tensors='pt',
)
```

**Mengapa `MAX_LEN=128` (bukan 512 seperti paper)?**

Paper rujukan memakai body article lengkap (perlu 512). Proyek ini **headline-only** (p95 ≈ 27 token). Memakai 512:
- Boros memori 4× lipat
- Padding `[PAD]` mendominasi >95% sequence, attention computation banyak yang sia-sia

Pilihan 128 memberi margin ~5× dari p95, aman untuk outlier judul panjang sambil hemat ~75% compute vs 512.

### 8.3 Bayesian hyperparameter search (Optuna TPE)

Sesuai paper Tabel III:

| Hyperparameter | Search space | Tipe |
|---|---|---|
| `lr` | `{2e-5, 3e-5, 5e-5}` | Categorical |
| `batch_size` | `{16, 32}` | Categorical |
| `epochs` | `1..10` | Integer |

```python
study = optuna.create_study(
    direction='minimize',                        # minimize val_loss
    sampler=optuna.samplers.TPESampler(seed=42), # Tree-structured Parzen Estimator
)
study.optimize(objective, n_trials=20)
```

**TPESampler** memodelkan distribusi `p(params | val_loss)` dengan KDE atas dua himpunan (`good` trials = bawah quantile, `bad` trials = atas quantile), lalu memilih sampel berikutnya yang memaksimalkan `p(good) / p(bad)`. Lebih sample-efficient dari grid/random search di dimensi rendah seperti ini.

Setiap trial:
1. Inisialisasi model fresh `AutoModelForSequenceClassification.from_pretrained(..., num_labels=2)`.
2. Train + early stop (lihat §8.4).
3. Return `best_val_loss`.
4. `torch.cuda.empty_cache()` antar trial untuk hindari OOM.

### 8.4 Loop training per trial ([`fit_with_early_stop`](src/train_indobert.py#L103-L126))

| Komponen | Setting | Catatan |
|---|---|---|
| Optimizer | `AdamW(lr, weight_decay=0.01)` | Standard AdamW untuk BERT |
| LR scheduler | `linear` decay | dari `get_scheduler('linear', ...)` |
| Warmup | **10%** dari total steps | LR ramp-up linear dari 0 ke `lr` selama 10% pertama, lalu linear decay ke 0 — stabilitas awal fine-tuning |
| Gradient clipping | `clip_grad_norm_(..., max_norm=1.0)` | Mencegah gradient explosion (well-known issue saat fine-tuning BERT) |
| Early stopping | `patience=3` epoch pada `val_loss` | Reload best state setelah trigger |

### 8.5 Final training

Setelah search selesai:

```python
best_epochs = max(study.best_params['epochs'], max_epochs_final)   # default 10
```

Model baru di-init fresh, dilatih lagi dengan best params tetapi `max_epochs` ditingkatkan supaya benar-benar konvergen — early stopping tetap memproteksi dari overfit.

Save via `model.save_pretrained()` + `tokenizer.save_pretrained()` ke `models/indobert_finetuned/`, ditemani:
- `best_params.json`
- `test_metrics.json` (setelah eval)

### 8.6 Best params yang ditemukan

Dari [models/indobert_finetuned/best_params.json](models/indobert_finetuned/best_params.json):

```json
{
  "lr": 3e-05,
  "batch_size": 32,
  "epochs": 3,
  "best_val_loss": 0.2134
}
```

### 8.7 Hasil test set

`Accuracy 0.9055 | Precision 0.8824 | Recall 0.9078 | F1 0.8949`

### 8.8 Menjalankan

```powershell
.\.venv\Scripts\python.exe src\train_indobert.py `
  --data-dir data\processed\pipeline_b `
  --output-dir models\indobert_finetuned `
  --model-name indobenchmark/indobert-base-p2 `
  --model-tag "IndoBERT Finetuned" `
  --n-trials 20
```

CLI args:

| Flag | Default |
|---|---|
| `--data-dir` | `data/processed/pipeline_b` |
| `--output-dir` | `models/indobert_finetuned` |
| `--n-trials` | `20` |
| `--model-name` | `indobenchmark/indobert-base-p2` |
| `--model-tag` | `IndoBERT Finetuned` |

---

## 9. Model 4 — mBERT Fine-Tuned (Pembanding Multilingual)

Base model: `bert-base-multilingual-cased` — BERT base yang di-pretrain pada Wikipedia 104 bahasa (cased).

**Script training identik** dengan IndoBERT — file yang sama (`train_indobert.py`) dijalankan dengan argumen `--model-name` berbeda. Search space, training loop, dan evaluasi sama persis.

Tujuan: membandingkan apakah model multilingual general-purpose dapat mengejar model monolingual yang dipre-train khusus Bahasa Indonesia.

### Best params

Dari [models/mbert_finetuned/best_params.json](models/mbert_finetuned/best_params.json):

```json
{
  "lr": 5e-05,
  "batch_size": 32,
  "epochs": 10,
  "best_val_loss": 0.2689
}
```

### Hasil test set

`Accuracy 0.8391 | Precision 0.8721 | Recall 0.7464 | F1 0.8043`

> mBERT kalah signifikan dari IndoBERT pada **recall** (74.6% vs 90.8%) — yaitu lebih banyak hoax yang lolos (false negatives). Konsisten dengan ekspektasi: model monolingual berbahasa Indonesia mengungguli multilingual baseline pada teks Indonesia.

---

## 10. Modal Cloud Training (A10G GPU)

Implementasi: [modal/train_indobert_modal.py](modal/train_indobert_modal.py) · Petunjuk: [modal/README.md](modal/README.md).

Karena 20 trial Optuna × ≤10 epoch berat untuk laptop, ada workflow Modal yang mengangkut training ke **A10G 24GB** (~$1.10/jam, total ~$1.50–$2.00 per model).

### 10.1 Image & volume

| Komponen | Konfigurasi |
|---|---|
| Base image | `debian_slim(python_version="3.11")` |
| Pip pin | `torch 2.4.0`, `transformers 4.44.2`, `optuna 3.6.1`, `scikit-learn 1.5.1`, `pandas 2.2.2`, `numpy 1.26.4`, `matplotlib 3.9.2`, `tqdm 4.66.5` |
| `add_local_dir` | `data/processed/pipeline_b/` → `/root/data/pipeline_b` <br> `src/` → `/root/src` |
| Volume | `indobert-hoax-models` (persistent) mount ke `/models` |
| GPU | `A10G` |
| Train timeout | 4 jam |
| Eval timeout | 15 menit |

Training di cloud mengimport `run()` dari `train_indobert.py` lokal yang sudah dibake — jadi **logic training 1:1 sama** dengan lokal, hanya berjalan di container GPU.

### 10.2 Registry model

Didefinisikan di [modal/train_indobert_modal.py:39-52](modal/train_indobert_modal.py#L39-L52):

```python
MODELS = {
    "indobert": {
        "hf_name": "indobenchmark/indobert-base-p2",
        "tag": "IndoBERT Finetuned",
        "subdir": "indobert_finetuned",
        "cm_filename": "cm_indobert_finetuned.png",
    },
    "mbert": {
        "hf_name": "bert-base-multilingual-cased",
        "tag": "mBERT Finetuned",
        "subdir": "mbert_finetuned",
        "cm_filename": "cm_mbert_finetuned.png",
    },
}
```

Untuk menambah model BERT-family baru, cukup append entry di sini.

### 10.3 Modal functions

| Function | Tujuan |
|---|---|
| `train(model, n_trials=20)` | Optuna search + final fine-tune; persist model + CM + metrics.json ke volume |
| `eval_cm(model)` | Re-evaluasi cepat (~30 detik) tanpa retrain — regenerate CM PNG |
| `list_artifacts()` / `list_artifact_paths()` | Inspeksi isi volume |
| `read_artifact(path)` | Stream single file |

### 10.4 Local entrypoints

| Command | Action |
|---|---|
| `modal run modal/train_indobert_modal.py::main --model indobert` | Train IndoBERT (default `n_trials=20`) |
| `modal run modal/train_indobert_modal.py::main --model mbert` | Train mBERT |
| `modal run modal/train_indobert_modal.py::train_both` | Train kedua berurutan |
| `modal run modal/train_indobert_modal.py::download` | Stream seluruh volume → `models/` lokal (via `modal volume get --force`) |
| `modal run modal/train_indobert_modal.py::ls` | List artifacts di volume |
| `modal run modal/train_indobert_modal.py::make_cm --model X` | Re-gen CM PNG saja |

Workflow rekomendasi:

```powershell
# 1. Preprocess lokal (data di-bake ke image)
.\.venv\Scripts\python.exe src\preprocess.py

# 2. Train detached (training lanjut walau laptop dimatikan)
.\.venv\Scripts\python.exe -m modal run --detach `
  modal\train_indobert_modal.py::main --model indobert

# 3. Cek log
.\.venv\Scripts\python.exe -m modal app logs indobert-hoax

# 4. Download artifact saat selesai
.\.venv\Scripts\python.exe -m modal run modal\train_indobert_modal.py::download
```

Saat training selesai, fungsi `train()` juga **men-copy CM PNG dari `results/` di dalam container ke volume** sehingga ikut ke-download.

---

## 11. Evaluation Pipeline

Implementasi: [src/evaluate.py](src/evaluate.py)

```python
def evaluate(y_true, y_pred, model_name='Model'):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)     # default pos_label=1 = Hoax
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    # print + plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=['Faktual', 'Hoax']).plot(cmap='Reds')
    plt.savefig(f'results/cm_{model_name.lower().replace(" ", "_")}.png')
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
```

**Detail:**
- Semua metrik **binary**, `pos_label=1` (Hoax). Precision/Recall/F1 diukur terhadap kelas Hoax.
- Confusion matrix: `display_labels=['Faktual', 'Hoax']`, colormap `Reds`.
- Output PNG: `results/cm_<model_name_snake>.png`.
- Return dict dipakai Modal untuk persist ke `test_metrics.json`.

### Quick eval lokal

[src/eval_indobert_local.py](src/eval_indobert_local.py) — load `models/indobert_finetuned/`, jalankan inference test set (batch=32, MAX_LEN=128), regenerate CM. Berguna saat checkpoint sudah ada tapi PNG-nya hilang/stale.

---

## 12. CLI Inference

Implementasi: [src/predict.py](src/predict.py).

```powershell
.\.venv\Scripts\python.exe src\predict.py
```

- Memuat `models/indobert_finetuned/` (`BertTokenizer` + `BertForSequenceClassification`).
- Device priority: CUDA → Intel XPU → CPU. `intel_extension_for_pytorch` di-import optional (tidak fatal kalau tidak ada).
- Loop interaktif: input judul → tokenisasi → forward → `softmax` → return label + confidence (%).
- `LABELS = {0: 'Faktual', 1: 'Hoax'}`.
- Exit: ketik `exit` atau `Ctrl+C`.

> Catatan: file ini memakai `max_length=512` saat tokenisasi (inference-time), sedangkan model dilatih dengan 128. Aman fungsional karena positional embedding BERT menampung 512 dan attention-mask membatasi yang dihitung, tetapi tidak konsisten dengan training pipeline.

---

## 13. Hasil Akhir & Tabel Perbandingan

Dari [results/result.md](results/result.md) dan `test_metrics.json` per model.

| Model | Accuracy | Precision | Recall | F1-Score |
|---|:---:|:---:|:---:|:---:|
| **IndoBERT Finetuned** | **0.9055** | 0.8824 | **0.9078** | **0.8949** |
| Ensemble Baseline (TF-IDF + RF/SVM/NB) | 0.8780 | 0.8770 | 0.8542 | 0.8654 |
| BiLSTM | 0.8421 | 0.8281 | 0.8281 | 0.8281 |
| mBERT Finetuned | 0.8391 | 0.8721 | 0.7464 | 0.8043 |

Best hyperparameters yang ditemukan Optuna:

| Model | LR | Batch | Epochs (best) | val_loss |
|---|---|---|---|---|
| IndoBERT | `3e-5` | 32 | 3 | 0.2134 |
| mBERT | `5e-5` | 32 | 10 | 0.2689 |

### Observasi

- **IndoBERT** unggul jelas di Accuracy, Recall, dan F1. Pre-training Bahasa Indonesia (corpus berbahasa target) terbukti material.
- **Recall adalah metrik operasional terpenting** — false negative (hoax lolos) lebih merugikan daripada false positive. IndoBERT 90.78% vs mBERT 74.64% adalah selisih sangat signifikan.
- **Ensemble TF-IDF** mengejutkan kuat (F1 0.865) dan **mengungguli BiLSTM serta mBERT**. Untuk teks pendek dengan vocabulary terbatas (~3.9k sample), unigram+bigram TF-IDF + soft-voting masih sangat efektif.
- **BiLSTM** dengan embedding dari nol tidak cukup mengejar TF-IDF — embedding yang baik butuh data jauh lebih besar.
- **mBERT recall jatuh ke 0.7464** — model cenderung memprediksi Faktual (kelas mayoritas), konsisten dengan literatur bahwa multilingual model kurang sensitif terhadap nuansa kelas minoritas pada bahasa spesifik.

Confusion matrices: [cm_indobert_finetuned.png](results/cm_indobert_finetuned.png) · [cm_ensemble_baseline.png](results/cm_ensemble_baseline.png) · [cm_bilstm.png](results/cm_bilstm.png) · [cm_mbert_finetuned.png](results/cm_mbert_finetuned.png).

---

## 14. Reproduksi Penuh

```powershell
# 0. Setup (sekali)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 1. Preprocess → bangun pipeline_a/ + pipeline_b/
.\.venv\Scripts\python.exe src\preprocess.py

# 2. Train Ensemble Baseline (cepat, CPU)
.\.venv\Scripts\python.exe src\train_baseline.py

# 3. Train BiLSTM (CPU/GPU lokal cukup)
.\.venv\Scripts\python.exe src\train_bilstm.py

# 4a. Train IndoBERT — LOKAL (lambat tanpa GPU bagus)
.\.venv\Scripts\python.exe src\train_indobert.py

# 4b. Train IndoBERT + mBERT — Modal cloud (rekomendasi)
.\.venv\Scripts\python.exe -m modal run --detach `
  modal\train_indobert_modal.py::train_both --n-trials 20
.\.venv\Scripts\python.exe -m modal run modal\train_indobert_modal.py::download

# 5. CLI inference
.\.venv\Scripts\python.exe src\predict.py
```

---

## 15. Dependencies

Dari [requirements.txt](requirements.txt):

```
torch>=2.0.0
transformers>=4.40.0
datasets>=2.18.0
optuna>=3.6.0
scikit-learn>=1.4.0
pandas>=2.2.0
numpy>=1.26.0
PySastrawi>=1.2.0
regex>=2024.4.28
matplotlib>=3.8.0
seaborn>=0.13.0
jupyter>=1.0.0
tqdm>=4.66.0
```

Tambahan runtime:
- **`nltk`** — auto-installed via dependency turunan; `preprocess.py` melakukan `nltk.download('stopwords', quiet=True)` saat pertama kali.
- **`modal`** — install terpisah (`pip install modal`) hanya jika pakai cloud training.

> Catatan: `PySastrawi` tercantum di requirements tetapi implementasi aktual `preprocess.py` memakai **NLTK Indonesian stopwords**. PySastrawi dapat di-swap di Pipeline A bila ingin stemming/stopword Sastrawi.

---

## Referensi

- **Paper utama:** Simanjuntak, A. et al. (2024). *Research and Analysis of IndoBERT Hyperparameter Tuning in Fake News Detection.* JNTETI Vol. 13 No. 1. DOI: [10.22146/jnteti.v1311.8532](https://doi.org/10.22146/jnteti.v1311.8532). Ringkasan lokal: [indobert_paper_summary.md](indobert_paper_summary.md).
- **IndoBERT:** [indobenchmark/indobert-base-p2](https://huggingface.co/indobenchmark/indobert-base-p2)
- **mBERT:** [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased)
- **Dataset asal:** TurnBackHoax.ID + detik.com (mirror Kaggle: [vijayandika/hoax-news-indonesia](https://www.kaggle.com/datasets/vijayandika/hoax-news-indonesia/)).
- **Optuna TPE:** Bergstra et al. (2011), *Algorithms for Hyper-Parameter Optimization.*
