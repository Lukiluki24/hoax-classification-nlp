# Modal Cloud Training

Cloud training untuk IndoBERT fine-tuning di Modal (A10G GPU 24GB).

## Setup (one-time)

```powershell
# 1. Install modal (sudah dilakukan di .venv)
.\.venv\Scripts\python.exe -m pip install modal

# 2. Authenticate
.\.venv\Scripts\python.exe -m modal token new
```

Browser akan terbuka untuk login. Token akan disimpan ke `~/.modal/config.toml`.

## Workflow

### 1. Regenerate processed data (lokal)

```powershell
.\.venv\Scripts\python.exe src\preprocess.py
```

Ini menghasilkan `data/processed/pipeline_b/{train,val,test}.csv` yang
akan di-bake ke container image saat Modal build.

### 2. Train di Modal

Dua model didukung via `--model`:

| `--model` | HuggingFace ID | Folder output |
|---|---|---|
| `indobert` (default) | `indobenchmark/indobert-base-p2` | `models/indobert_finetuned/` |
| `mbert` | `bert-base-multilingual-cased` | `models/mbert_finetuned/` |

**Train IndoBERT:**
```powershell
.\.venv\Scripts\python.exe -m modal run --detach modal\train_indobert_modal.py::main --model indobert
```

**Train mBERT:**
```powershell
.\.venv\Scripts\python.exe -m modal run --detach modal\train_indobert_modal.py::main --model mbert
```

**Train kedua-duanya berurutan:**
```powershell
.\.venv\Scripts\python.exe -m modal run --detach modal\train_indobert_modal.py::train_both
```

`--detach` adalah flag bawaan Modal — training tetap jalan di cloud meski
terminal/laptop dimatikan. Hapus untuk lihat output realtime.

**Cek progress / logs:**
```powershell
.\.venv\Scripts\python.exe -m modal app logs indobert-hoax
```

**Tes cepat dengan trial sedikit:**
```powershell
.\.venv\Scripts\python.exe -m modal run modal\train_indobert_modal.py::main --model mbert --n-trials 3
```

**Estimasi:**
- A10G ~$1.10/jam
- 20 trials Optuna (rata-rata ~2-4 epoch per trial) + final training: ~60-90 menit
- **Total cost: ~$1.50 - $2.00**

### 3. Inspect artifact di volume

```powershell
.\.venv\Scripts\python.exe -m modal run modal\train_indobert_modal.py::ls
```

### 4. Download model ke lokal

```powershell
.\.venv\Scripts\python.exe -m modal run modal\train_indobert_modal.py::download
```

File akan disimpan ke `models/indobert_finetuned/` dan/atau `models/mbert_finetuned/`
lokal. Confusion matrix PNG juga disalin ke `results/`.

### 5. Inference lokal (CPU OK)

```powershell
.\.venv\Scripts\python.exe src\predict.py
```

## Hyperparameter (paper-aligned)

Mengikuti Simanjuntak et al. (2024):

| Param | Search Space | Best (paper Bayesian) |
|---|---|---|
| Learning rate | `{2e-5, 3e-5, 5e-5}` | `2e-5` |
| Batch size | `{16, 32}` | `16` |
| Epochs | `1..10` | `8` |
| Direction | minimize `val_loss` | — |

**Perbedaan dari paper:**
- `MAX_LEN=128` (paper 512, karena pakai full body). Kami headline-only, p95 = ~27 token.
- Tambahan: warmup 10%, gradient clipping 1.0, early stopping patience=3 di final training.

## Troubleshooting

- **OOM di A10G:** kurangi `n_trials`, atau ganti GPU ke A100-40GB di
  `@app.function(gpu="A100-40GB", ...)`.
- **"Volume not found":** Modal auto-create. Kalau gagal,
  cek dengan `modal volume list`.
- **Re-upload data:** preprocess.py di lokal lalu re-run modal command;
  `add_local_dir` akan re-bake otomatis (dengan cache layer).
