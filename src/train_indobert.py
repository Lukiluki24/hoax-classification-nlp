"""
IndoBERT fine-tuning with Bayesian (Optuna) hyperparameter search.

Aligned with Simanjuntak et al. (2024) "Research and Analysis of IndoBERT
Hyperparameter Tuning in Fake News Detection":
  - LR search:    {2e-5, 3e-5, 5e-5}     (categorical)
  - Batch size:   {16, 32}
  - Epochs:       1..10
  - Direction:    minimize val_loss
  - Early stopping in final training (patience=3 on val_loss)

Headline-only adaptation: MAX_LEN=128 (data p95 ~= 27 tokens). Paper used 512
because they fed full article body; padding headlines to 512 wastes ~95% of
compute on [PAD] tokens.
"""
import os
import argparse
import json

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from tqdm import tqdm
import optuna

from evaluate import evaluate

DEFAULT_MODEL_NAME = 'indobenchmark/indobert-base-p2'
DEFAULT_MODEL_TAG = 'IndoBERT Finetuned'
MAX_LEN = 128  # headline-only; p95 ~= 27 tokens, leaves ample margin


# -------------------------- helpers ------------------------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return torch.device('xpu')
    return torch.device('cpu')


def tokenize(texts, tokenizer, max_len=MAX_LEN):
    return tokenizer(
        list(texts),
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )


def to_dataset(enc, labels):
    return TensorDataset(
        enc['input_ids'],
        enc['attention_mask'],
        torch.tensor(labels.values, dtype=torch.long),
    )


def compute_val_loss(model, loader, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            total += model(input_ids, attention_mask=attention_mask, labels=labels).loss.item()
    return total / len(loader)


def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    for input_ids, attention_mask, labels in tqdm(loader, leave=False, desc='train'):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        loss = model(input_ids, attention_mask=attention_mask, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # BERT stability
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()


def build_optimizer_scheduler(model, lr, num_training_steps, warmup_ratio=0.1):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    num_warmup = int(warmup_ratio * num_training_steps)
    scheduler = get_scheduler(
        'linear',
        optimizer,
        num_warmup_steps=num_warmup,
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler


# -------------------------- core training loop -------------------------------
def fit_with_early_stop(model, train_loader, val_loader, lr, epochs, device, patience=3):
    """Train with early stopping on val_loss. Returns best_val_loss and reloads best weights."""
    optimizer, scheduler = build_optimizer_scheduler(
        model, lr, num_training_steps=epochs * len(train_loader)
    )
    best_val = float('inf')
    best_state = None
    waited = 0
    for epoch in range(epochs):
        train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss = compute_val_loss(model, val_loader, device)
        print(f"  epoch {epoch + 1}/{epochs}  val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            waited = 0
        else:
            waited += 1
            if waited >= patience:
                print(f"  early stop @ epoch {epoch + 1}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val


# -------------------------- entrypoint ---------------------------------------
def run(data_dir='data/processed/pipeline_b',
        output_dir='models/indobert_finetuned',
        n_trials=20,
        max_epochs_final=10,
        model_name=DEFAULT_MODEL_NAME,
        model_tag=DEFAULT_MODEL_TAG):
    print(f"Loading data from {data_dir}...")
    print(f"Model: {model_name}  ->  tag '{model_tag}'")
    train_df = pd.read_csv(f'{data_dir}/train.csv').dropna()
    val_df = pd.read_csv(f'{data_dir}/val.csv').dropna()
    test_df = pd.read_csv(f'{data_dir}/test.csv').dropna()
    print(f"  train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_enc = tokenize(train_df['text'].astype(str), tokenizer)
    val_enc = tokenize(val_df['text'].astype(str), tokenizer)
    test_enc = tokenize(test_df['text'].astype(str), tokenizer)

    train_ds = to_dataset(train_enc, train_df['label'])
    val_ds = to_dataset(val_enc, val_df['label'])
    test_ds = to_dataset(test_enc, test_df['label'])

    device = get_device()
    print(f"Device: {device}")

    # ---------- Optuna objective (paper search space) ----------
    def objective(trial: optuna.Trial):
        lr = trial.suggest_categorical('lr', [2e-5, 3e-5, 5e-5])
        batch_size = trial.suggest_categorical('batch_size', [16, 32])
        epochs = trial.suggest_int('epochs', 1, 10)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
        best_val = fit_with_early_stop(model, train_loader, val_loader, lr, epochs, device, patience=3)
        # free GPU
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return best_val

    print(f"Starting Optuna Bayesian search (n_trials={n_trials})...")
    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    print("Best params:", study.best_params)
    print(f"Best val_loss: {study.best_value:.4f}")

    # ---------- Final training with best params + early stopping ----------
    best_lr = study.best_params['lr']
    best_bs = study.best_params['batch_size']
    best_epochs = max(study.best_params['epochs'], max_epochs_final)

    print(f"Final training: lr={best_lr}, batch={best_bs}, max_epochs={best_epochs}")
    train_loader = DataLoader(train_ds, batch_size=best_bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=best_bs)
    test_loader = DataLoader(test_ds, batch_size=best_bs)

    final_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    fit_with_early_stop(final_model, train_loader, val_loader, best_lr, best_epochs, device, patience=3)

    # ---------- Save ----------
    os.makedirs(output_dir, exist_ok=True)
    final_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, 'best_params.json'), 'w') as f:
        json.dump({**study.best_params, 'best_val_loss': study.best_value}, f, indent=2)
    print(f"Model saved -> {output_dir}")

    # ---------- Evaluate ----------
    print("Evaluating on test set...")
    final_model.eval()
    all_preds, y_true = [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(test_loader, desc='test'):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = final_model(input_ids, attention_mask=attention_mask)
            all_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    return evaluate(y_true, all_preds, model_name=model_tag)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data/processed/pipeline_b')
    p.add_argument('--output-dir', default='models/indobert_finetuned')
    p.add_argument('--n-trials', type=int, default=20)
    p.add_argument('--model-name', default=DEFAULT_MODEL_NAME)
    p.add_argument('--model-tag', default=DEFAULT_MODEL_TAG)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(data_dir=args.data_dir, output_dir=args.output_dir, n_trials=args.n_trials,
        model_name=args.model_name, model_tag=args.model_tag)
