import os
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import BertForSequenceClassification, BertTokenizer

# 1. Inisialisasi FastAPI
app = FastAPI(title="IndoBERT Hoax Detection API Backend")

# Mengaktifkan CORS agar Ekstensi Chrome (murni Frontend) bisa berkomunikasi dengan server ini
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Pengaturan Path dan Konfigurasi Model (Sesuai dengan project directory kamu)
MODEL_PATH = 'models/indobert_finetuned/'
MAX_LEN = 128  # Disesuaikan menjadi 128 sesuai pipeline latihanmu agar tidak boros compute dan akurat
LABELS = {0: 'Faktual', 1: 'Hoax'}

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model tidak ditemukan di {MODEL_PATH}. Pastikan sudah running train_indobert.py!")

print("Memuat model IndoBERT Finetuned ke memory...")
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

# Menentukan Device (Mengikuti pola fleksibel di predict.py milikmu)
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch, 'xpu') and torch.xpu.is_available():
    device = torch.device('xpu')
else:
    device = torch.device('cpu')

model = model.to(device)
model.eval()
print(f"Model berhasil dimuat menggunakan device: {device}")

# 3. Skema Data Input dari Ekstensi Browser
class HeadlineRequest(BaseModel):
    text: str

# 4. Endpoint Deteksi / Inference
@app.post("/predict")
async def predict_hoax(req: HeadlineRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Teks judul tidak boleh kosong!")
        
    try:
        # Tokenisasi menggunakan Pipeline B (Teks mentah tanpa membuang tanda baca/stopword)
        inputs = tokenizer(
            req.text,
            return_tensors='pt',
            max_length=MAX_LEN,  # Tetap di 128 token sesuai rancangan optimalisasi short text kelompokmu
            truncation=True,
            padding='max_length',
            return_attention_mask=True
        )
        
        # Pindahkan tensor input ke device yang sama dengan model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Jalankan prediksi tanpa menghitung gradient (hemat memori)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
        probs = F.softmax(logits, dim=1).squeeze()
        label_id = torch.argmax(probs).item()
        confidence = probs[label_id].item()
        
        return {
            "headline": req.text,
            "prediction": LABELS[label_id],
            "confidence": round(confidence * 100, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan internal backend: {str(e)}")   