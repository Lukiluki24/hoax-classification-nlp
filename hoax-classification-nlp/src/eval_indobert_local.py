"""Quick local evaluation script — load fine-tuned model & regenerate confusion matrix."""
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
from evaluate import evaluate

MODEL_DIR = "models/indobert_finetuned"
DATA_DIR = "data/processed/pipeline_b"
MAX_LEN = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

test_df = pd.read_csv(f"{DATA_DIR}/test.csv").dropna()
enc = tokenizer(
    list(test_df["text"].astype(str)),
    add_special_tokens=True, max_length=MAX_LEN,
    padding="max_length", truncation=True,
    return_attention_mask=True, return_tensors="pt",
)
ds = TensorDataset(enc["input_ids"], enc["attention_mask"],
                   torch.tensor(test_df["label"].values, dtype=torch.long))
loader = DataLoader(ds, batch_size=32)

preds, ys = [], []
with torch.no_grad():
    for ids, mask, lbl in tqdm(loader, desc="eval"):
        ids, mask = ids.to(device), mask.to(device)
        out = model(ids, attention_mask=mask)
        preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
        ys.extend(lbl.numpy())

evaluate(ys, preds, model_name="IndoBERT Finetuned")
