# Indonesian Hoax News Detection System — AI Agent Instruction

> Fine-Tuned IndoBERT for Indonesian Headline Classification  
> Authors: Lucky Wijaya (2802443975), Kelvin Yohanes Justinus (2802447576), Garren Tanavaro (2802516182)

---

## Overview

This project builds an **automated Indonesian hoax news detection system** that classifies news headlines as **Hoax** or **Faktual (Factual)** using a fine-tuned `indobenchmark/indobert-base-p2` Transformer model.

### Problem Context
- Indonesia faces massive hoax proliferation — over 800,000 websites flagged by the Ministry of Communication and Informatics.
- Readers typically only read headlines before sharing, making headlines the primary vector of misinformation.
- Manual fact-checking is impractical at scale; automated classification is critical.

### Why IndoBERT
- Pre-trained on Indonesian corpora; captures language-specific contextual semantics.
- Self-attention mechanism outperforms CNN-LSTM and traditional ML models on this task.
- With Bayesian hyperparameter optimization, prior work achieved up to **94.32% accuracy**.

### Goal
Fine-tune IndoBERT to classify Indonesian news headlines, benchmark it against three other architectures, and expose the final model via a **CLI inference script** that outputs a binary label and confidence score.

---

## Project Structure

```
hoax-detection/
├── data/
│   ├── raw/
│   │   ├── turnbackhoax.csv      # Training data (labeled headlines)
│   │   └── detik.csv             # Test data (factual news, Oct–Dec 2023)
│   └── processed/
│       ├── pipeline_a/           # TF-IDF preprocessed splits
│       └── pipeline_b/           # IndoBERT tokenized splits
│
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   ├── 02_preprocessing.ipynb    # Preprocessing pipelines A & B
│   ├── 03_baseline.ipynb         # Ensemble baseline training
│   ├── 04_bilstm.ipynb           # Bi-LSTM training
│   └── 05_indobert.ipynb         # IndoBERT fine-tuning + Optuna tuning
│
├── src/
│   ├── preprocess.py             # Shared cleaning + pipeline A/B logic
│   ├── train_baseline.py         # Ensemble (RF + SVM + NB) training
│   ├── train_bilstm.py           # Bi-LSTM training
│   ├── train_indobert.py         # IndoBERT fine-tuning with Optuna
│   ├── evaluate.py               # Metrics: accuracy, precision, recall, F1, confusion matrix
│   └── predict.py                # CLI inference script
│
├── models/
│   ├── ensemble_baseline.pkl     # Saved ensemble model
│   ├── bilstm_model/             # Saved Bi-LSTM weights
│   └── indobert_finetuned/       # Saved fine-tuned IndoBERT + tokenizer
│
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites
- Python **3.10+**
- pip
- Git

### Steps

```bash
# 1. Clone the repository
git clone <repo-url>
cd hoax-detection

# 2. Create and activate virtual environment (see .venv section below)

# 3. Install dependencies
pip install -r requirements.txt
```

### `requirements.txt`

```txt
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

---

## Virtual Environment (.venv)

Always work inside a virtual environment to isolate project dependencies.

```bash
# Create the virtual environment
python -m venv .venv

# Activate — Windows
.venv\Scripts\activate

# Activate — macOS/Linux
source .venv/bin/activate

# Verify activation (should point to .venv)
where python   # Windows
which python   # macOS/Linux

# Deactivate when done
deactivate
```

> Install all packages **after** activating `.venv`. Never install globally.

---

## Dataset

Download from Kaggle: [hoax-news-indonesia by vijayandika](https://www.kaggle.com/datasets/vijayandika/hoax-news-indonesia/)

Place the files as:
```
data/raw/turnbackhoax.csv
data/raw/detik.csv
```

| File | Role | Labels |
|---|---|---|
| `turnbackhoax.csv` | Training | `[SALAH]`, `[KLARIFIKASI]`, `[BENAR]` embedded in headlines |
| `detik.csv` | Testing | Factual news (no embedded label) |

**Label mapping:**
- `[SALAH]` → `1` (Hoax)
- `[KLARIFIKASI]` → `1` (Hoax)
- `[BENAR]` → `0` (Faktual)
- `detik.csv` rows → `0` (Faktual)

---

## Detailed Preprocessing

### Step 0 — General Data Cleaning (All Models)

Applied to both CSV files before any pipeline-specific processing.

```python
import pandas as pd
import re

def general_clean(text: str) -> str:
    text = re.sub(r'http\S+|www\S+', '', text)       # Remove URLs
    text = re.sub(r'[^\x00-\x7FÀ-ɏ]+', ' ', text)   # Remove emojis / non-ASCII symbols
    text = re.sub(r'\s+', ' ', text).strip()           # Normalize whitespace
    return text

df = pd.read_csv('data/raw/turnbackhoax.csv')
df.dropna(subset=['title'], inplace=True)              # Remove missing values
df['title'] = df['title'].apply(general_clean)
```

---

### Step 1 — Label Extraction (`turnbackhoax.csv` only)

```python
def extract_label(text: str) -> tuple[int, str]:
    label_map = {'SALAH': 1, 'KLARIFIKASI': 1, 'BENAR': 0}
    match = re.search(r'\[(SALAH|KLARIFIKASI|BENAR)\]', text, re.IGNORECASE)
    label = label_map.get(match.group(1).upper(), -1) if match else -1
    clean_text = re.sub(r'\[.*?\]', '', text).strip()
    return label, clean_text

df[['label', 'title']] = df['title'].apply(
    lambda t: pd.Series(extract_label(t))
)
df = df[df['label'] != -1]   # Drop rows with unrecognized labels
```

---

### Step 2 — Data Splitting

```python
from sklearn.model_selection import train_test_split

X, y = df['title'], df['label']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.667, random_state=42, stratify=y_temp
)
# Splits: Train 70% | Validation 10% | Test 20%
```

`detik.csv` is reserved as a **held-out real-world test set** (all labels = 0).

---

### Pipeline A — For Ensemble Baseline (TF-IDF)

Applied to `X_train`, `X_val`, `X_test`.

```python
from PySastrawi.StopWordRemoverFactory import StopWordRemoverFactory

factory = StopWordRemoverFactory()
remover = factory.create_stop_word_remover()

def pipeline_a(text: str) -> str:
    text = text.lower()                          # Case folding
    text = re.sub(r'[^a-z\s]', '', text)         # Remove punctuation and numbers
    text = remover.remove(text)                  # Remove Indonesian stop words (Sastrawi)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

X_train_a = X_train.apply(pipeline_a)
X_val_a   = X_val.apply(pipeline_a)
X_test_a  = X_test.apply(pipeline_a)
```

**Rationale:**
- Lowercase normalizes casing variance (`Hoax` == `hoax`)
- Punctuation/numbers removed — low signal for TF-IDF frequency counting
- Stop words removed via Sastrawi corpus to reduce dimensionality and focus on content-bearing terms

---

### Pipeline B — For Fine-Tuned IndoBERT

IndoBERT's self-attention requires the **full original sentence structure**. Do NOT lowercase, remove punctuation, numbers, or stop words.

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p2')
MAX_LEN = 512

def pipeline_b(texts, tokenizer, max_len=MAX_LEN):
    return tokenizer(
        list(texts),
        add_special_tokens=True,     # Adds [CLS] at start, [SEP] at end
        max_length=max_len,
        padding='max_length',        # Pad shorter sequences to MAX_LEN
        truncation=True,             # Truncate longer sequences to MAX_LEN
        return_attention_mask=True,
        return_tensors='pt'
    )

train_enc = pipeline_b(X_train, tokenizer)
val_enc   = pipeline_b(X_val,   tokenizer)
test_enc  = pipeline_b(X_test,  tokenizer)
```

**Rationale:**
- Punctuation, numbers, and stop words retained — Transformer self-attention uses them for contextual meaning
- WordPiece tokenization splits OOV Indonesian words into sub-word units
- `[CLS]` token (position 0) carries the sequence-level classification representation
- `[SEP]` marks sentence boundaries
- All sequences padded or truncated to exactly **512 tokens** (BERT maximum input length)

---

## Model Training

### Model 1 — Ensemble Baseline (TF-IDF + Soft Voting)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import joblib

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train_a)
X_val_tfidf   = tfidf.transform(X_val_a)
X_test_tfidf  = tfidf.transform(X_test_a)

rf  = RandomForestClassifier(n_estimators=200, random_state=42)
svm = SVC(probability=True, kernel='rbf', random_state=42)
nb  = MultinomialNB()

ensemble = VotingClassifier(
    estimators=[('rf', rf), ('svm', svm), ('nb', nb)],
    voting='soft'
)
ensemble.fit(X_train_tfidf, y_train)

joblib.dump({'model': ensemble, 'vectorizer': tfidf}, 'models/ensemble_baseline.pkl')
```

**Expected benchmark accuracy: ~91.18%** (Pekandi et al., 2025)

---

### Model 2 — Bidirectional LSTM

```python
import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                               batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.bilstm(x)
        return self.fc(out[:, -1, :])
```

**Expected benchmark accuracy: ~92%** (Arkaan et al., 2024)

---

### Model 3 — Fine-Tuned IndoBERT with Bayesian Optimization (Main Model)

#### Setup

```python
from transformers import BertForSequenceClassification, AdamW, get_scheduler
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BertForSequenceClassification.from_pretrained(
    'indobenchmark/indobert-base-p2',
    num_labels=2
).to(device)
```

#### Hyperparameter Tuning with Optuna

```python
import optuna
from torch.utils.data import DataLoader, TensorDataset

def build_dataset(enc, labels):
    return TensorDataset(
        enc['input_ids'], enc['attention_mask'],
        torch.tensor(labels.values, dtype=torch.long)
    )

train_dataset = build_dataset(train_enc, y_train)
val_dataset   = build_dataset(val_enc,   y_val)

def objective(trial):
    lr         = trial.suggest_float('lr', 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    epochs     = trial.suggest_int('epochs', 2, 5)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size)

    m = BertForSequenceClassification.from_pretrained(
        'indobenchmark/indobert-base-p2', num_labels=2
    ).to(device)

    optimizer = AdamW(m.parameters(), lr=lr)
    scheduler = get_scheduler('linear', optimizer,
                               num_warmup_steps=0,
                               num_training_steps=epochs * len(train_loader))

    best_val_loss, patience_counter = float('inf'), 0

    for _ in range(epochs):
        m.train()
        for input_ids, attention_mask, labels in train_loader:
            input_ids, attention_mask, labels = (
                input_ids.to(device), attention_mask.to(device), labels.to(device)
            )
            loss = m(input_ids, attention_mask=attention_mask, labels=labels).loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        val_loss = _compute_val_loss(m, val_loader, device)
        if val_loss < best_val_loss:
            best_val_loss, patience_counter = val_loss, 0
        else:
            patience_counter += 1
            if patience_counter >= 3:   # Early stopping patience = 3
                break

    return best_val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)
print("Best params:", study.best_params)
```

**Tuned hyperparameters:**

| Parameter | Search Range |
|---|---|
| Learning rate | `1e-5` to `5e-5` (log scale) |
| Batch size | `8`, `16`, `32` |
| Epochs | `2` to `5` |

**Early stopping:** halts training if validation loss does not improve for 3 consecutive epochs.

#### Save the Final Model

```python
model.save_pretrained('models/indobert_finetuned/')
tokenizer.save_pretrained('models/indobert_finetuned/')
```

**Expected accuracy: ~94.32%** (Simanjuntak et al., 2024)

---

## Evaluation

### Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

def evaluate(y_true, y_pred, model_name='Model'):
    print(f"\n=== {model_name} ===")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score : {f1_score(y_true, y_pred):.4f}")

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Hoax', 'Faktual'])
    disp.plot(cmap='Reds')
    plt.title(f'Confusion Matrix — {model_name}')
    plt.tight_layout()
    plt.savefig(f'results/cm_{model_name.lower().replace(" ", "_")}.png')
    plt.show()
```

### Metric Definitions

| Metric | Formula | Purpose |
|---|---|---|
| **Accuracy** | (TP + TN) / Total | Overall correctness; may mislead on imbalanced data |
| **Precision** | TP / (TP + FP) | Of all predicted Hoax, how many are actually Hoax |
| **Recall** | TP / (TP + FN) | Of all real Hoax, how many were correctly caught |
| **F1-Score** | 2 × (P × R) / (P + R) | **Primary metric** — harmonic mean balancing precision and recall |

> **Recall is critical.** Missing actual hoaxes (false negatives) carries significant real-world consequences. Maximize recall without sacrificing too much precision.

### Model Comparison

| Model | Architecture | Expected Accuracy | Reference |
|---|---|---|---|
| **Fine-Tuned IndoBERT** *(main)* | Transformer + Bayesian Optuna | ~94.32% | Simanjuntak et al., 2024 |
| Ensemble Baseline | RF + SVM + NB, TF-IDF soft voting | ~91.18% | Pekandi et al., 2025 |
| Bidirectional LSTM | Bi-LSTM sequential deep learning | ~92.00% | Arkaan et al., 2024 |
| mBERT / XLM-R + BERTopic | Multilingual Transformer + topic modeling | ~90.51% | Hutama & Suhartono, 2022 |

### Success Criteria

- Fine-tuned IndoBERT achieves **accuracy >= 90%** and **F1-score >= 90%**
- IndoBERT outperforms all comparison models
- CLI script correctly outputs `Hoax` or `Faktual` with a confidence score for any input headline

---

## CLI Inference

After training, run predictions interactively via terminal:

```python
# src/predict.py
import torch
from transformers import BertForSequenceClassification, BertTokenizer

MODEL_PATH = 'models/indobert_finetuned/'
LABELS = {0: 'Faktual', 1: 'Hoax'}

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

def predict(headline: str) -> tuple[str, float]:
    inputs = tokenizer(
        headline, return_tensors='pt',
        max_length=512, truncation=True, padding='max_length'
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs    = torch.softmax(logits, dim=1).squeeze()
    label_id = torch.argmax(probs).item()
    return LABELS[label_id], round(probs[label_id].item() * 100, 2)

if __name__ == '__main__':
    print("=== Indonesian Hoax Headline Detector ===")
    while True:
        headline = input("\nMasukkan judul berita (atau 'exit' untuk keluar): ").strip()
        if headline.lower() == 'exit':
            break
        label, confidence = predict(headline)
        print(f"Prediksi  : {label}")
        print(f"Confidence: {confidence}%")
```

```bash
# Run the CLI
python src/predict.py
```

---

## Agent Rules

- Always activate `.venv` before running any script or notebook.
- Run notebooks strictly in order: `01` → `02` → `03` → `04` → `05`.
- Use **Pipeline A only** for Ensemble and Bi-LSTM inputs. Use **Pipeline B only** for IndoBERT inputs. Never mix.
- Never apply stop word removal, case folding, or punctuation stripping to IndoBERT input — these degrade the self-attention mechanism's contextual understanding.
- Always evaluate on the **test split** (20%), not the validation split used during Optuna trials.
- Report all four metrics (Accuracy, Precision, Recall, F1-Score) for every model in the comparison table.
- Save trained models to `models/` immediately after each training run completes.
- The **primary optimization target is F1-Score**. Tune hyperparameters toward maximizing F1, not raw accuracy.
- Do not modify `data/raw/` files. All transformations must write to `data/processed/`.
