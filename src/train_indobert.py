import torch
try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    pass
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import optuna
import os
from evaluate import evaluate
from tqdm import tqdm

MAX_LEN = 512

def pipeline_b(texts, tokenizer, max_len=MAX_LEN):
    return tokenizer(
        list(texts),
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

def build_dataset(enc, labels):
    return TensorDataset(
        enc['input_ids'], enc['attention_mask'],
        torch.tensor(labels.values, dtype=torch.long)
    )

def _compute_val_loss(model, val_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in val_loader:
            input_ids, attention_mask, labels = (
                input_ids.to(device), attention_mask.to(device), labels.to(device)
            )
            loss = model(input_ids, attention_mask=attention_mask, labels=labels).loss
            total_loss += loss.item()
    return total_loss / len(val_loader)

def main():
    print("Loading Pipeline B data for IndoBERT...")
    train_df = pd.read_csv('data/processed/pipeline_b/train.csv').dropna()
    val_df = pd.read_csv('data/processed/pipeline_b/val.csv').dropna()
    test_df = pd.read_csv('data/processed/pipeline_b/test.csv').dropna()
    
    # Use smaller subset to save time if needed, but per instructions we train on all
    tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p2')
    
    print("Tokenizing data...")
    train_enc = pipeline_b(train_df['text'].astype(str), tokenizer)
    val_enc   = pipeline_b(val_df['text'].astype(str), tokenizer)
    test_enc  = pipeline_b(test_df['text'].astype(str), tokenizer)

    train_dataset = build_dataset(train_enc, train_df['label'])
    val_dataset   = build_dataset(val_enc,   val_df['label'])
    test_dataset  = build_dataset(test_enc,  test_df['label'])

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device('xpu')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # For optuna to run quickly during demo, we could reduce n_trials, 
    # but the instructions say n_trials=20
    # To save time for the agent, we will just run 2 trials unless strict
    N_TRIALS = 20 # Restored to full training
    
    print("Starting Optuna hyperparameter tuning...")
    def objective(trial):
        lr         = trial.suggest_float('lr', 1e-5, 5e-5, log=True)
        batch_size = trial.suggest_categorical('batch_size', [8, 16]) # Avoid 32 to prevent OOM
        epochs     = trial.suggest_int('epochs', 2, 4)

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

        for epoch in range(epochs):
            m.train()
            # print(f"Trial {trial.number}, Epoch {epoch+1}/{epochs}")
            for input_ids, attention_mask, labels in tqdm(train_loader, leave=False):
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
                if patience_counter >= 3:
                    break

        return best_val_loss

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS)
    print("Best params:", study.best_params)

    print("Training final model with best params...")
    best_lr = study.best_params['lr']
    best_bs = study.best_params['batch_size']
    best_epochs = study.best_params['epochs']

    train_loader = DataLoader(train_dataset, batch_size=best_bs, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=best_bs)
    test_loader  = DataLoader(test_dataset,  batch_size=best_bs)

    final_model = BertForSequenceClassification.from_pretrained(
        'indobenchmark/indobert-base-p2', num_labels=2
    ).to(device)

    optimizer = AdamW(final_model.parameters(), lr=best_lr)
    scheduler = get_scheduler('linear', optimizer,
                               num_warmup_steps=0,
                               num_training_steps=best_epochs * len(train_loader))

    for epoch in range(best_epochs):
        final_model.train()
        print(f"Final Model Epoch {epoch+1}/{best_epochs}")
        for input_ids, attention_mask, labels in tqdm(train_loader):
            input_ids, attention_mask, labels = (
                input_ids.to(device), attention_mask.to(device), labels.to(device)
            )
            loss = final_model(input_ids, attention_mask=attention_mask, labels=labels).loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    os.makedirs('models/indobert_finetuned', exist_ok=True)
    final_model.save_pretrained('models/indobert_finetuned/')
    tokenizer.save_pretrained('models/indobert_finetuned/')
    
    print("Evaluating IndoBERT on Test Set...")
    final_model.eval()
    all_preds = []
    y_true_test = []
    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(test_loader):
            input_ids, attention_mask, labels = (
                input_ids.to(device), attention_mask.to(device), labels.to(device)
            )
            outputs = final_model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            y_true_test.extend(labels.cpu().numpy())
            
    evaluate(y_true_test, all_preds, model_name='IndoBERT Finetuned')

if __name__ == "__main__":
    main()
