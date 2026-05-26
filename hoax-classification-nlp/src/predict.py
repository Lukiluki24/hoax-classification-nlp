import torch
try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    pass
from transformers import BertForSequenceClassification, BertTokenizer
import os

MODEL_PATH = 'models/indobert_finetuned/'
LABELS = {0: 'Faktual', 1: 'Hoax'}

def predict(headline: str, model, tokenizer) -> tuple[str, float]:
    inputs = tokenizer(
        headline, return_tensors='pt',
        max_length=512, truncation=True, padding='max_length'
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs    = torch.softmax(logits, dim=1).squeeze()
    label_id = torch.argmax(probs).item()
    return LABELS[label_id], round(probs[label_id].item() * 100, 2)

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Please run train_indobert.py first.")
        return

    print("Loading model...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device('xpu')
    else:
        device = torch.device('cpu')
    
    model = model.to(device)
    model.eval()

    print("=== Indonesian Hoax Headline Detector ===")
    while True:
        try:
            headline = input("\nMasukkan judul berita (atau 'exit' untuk keluar): ").strip()
            if headline.lower() == 'exit':
                break
            if not headline:
                continue
            
            label, confidence = predict(headline, model, tokenizer)
            print(f"Prediksi  : {label}")
            print(f"Confidence: {confidence}%")
        except KeyboardInterrupt:
            break

if __name__ == '__main__':
    main()
