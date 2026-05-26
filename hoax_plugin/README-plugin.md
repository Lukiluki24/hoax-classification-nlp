
# Detektor Hoaks IndoBERT (Ekstensi Chrome)

Bagian dari proyek  **Indonesian Hoax News Detection System** . Ekstensi Google Chrome ini berfungsi sebagai antarmuka ( *frontend* ) yang mendeteksi indikasi hoaks pada judul berita di internet secara *real-time* dan otomatis, ditenagai oleh model NLP IndoBERT p2 kami yang telah di- *fine-tune* .

## Fitur Utama

* **Pemindaian Otomatis Real-Time:** Ekstensi otomatis memindai elemen judul (`<h1>`, `<h2>`, `<h3>`, `<h4>`) begitu halaman web berita selesai dimuat.
* **Filter Heuristik Pintar:** Dilengkapi logika penyaring bawaan untuk membedakan judul berita asli dengan teks antarmuka web (UI) menggunakan:
  * Batas jumlah kata minimum (minimal 5 kata).
  * Pemblokiran struktur DOM (mengabaikan `<nav>`, `<header>`, `<footer>`, dll.).
  * Pengecekan daftar hitam ( *blacklist* ) (mengabaikan kata kunci UI seperti "Tag Terpopuler", "Baca Juga").
  * Penyaringan karakter khusus (mengabaikan tagar `#`).
* **Integrasi Visual yang Mulus:** Hasil prediksi (`LIKELY TO BE HOAX` atau `LIKELY TO BE FACT`) disuntikkan secara elegan ke samping judul asli dalam bentuk *badge tooltip* tanpa merusak tata letak halaman.
* **Arsitektur Bypass CORS:** Menggunakan *Background Script* (Service Worker) Manifest V3 untuk menjembatani komunikasi ke `localhost`, sukses melewati pemblokiran keamanan *Private Network Access* (PNA) bawaan browser yang ketat.

## Struktur Direktori

Proyek ini beroperasi dengan arsitektur  *Client-Server* , yang dipisah menjadi dua direktori sejajar ( *sibling directories* ). Klien berjalan di browser Chrome, sedangkan Server (model IndoBERT) berjalan di backend Python FastAPI.

**Plaintext**

```
hoax-detector-nlp/
│
├── hoax-classification-nlp/           # Direktori Backend & Model
│   ├── models/
│   │   └── indobert_finetuned/        # Artifact model terlatih (wajib ada)
│   └── server.py                      # API Backend (FastAPI)
│
└── hoax_plugin/                       # Direktori Ekstensi Chrome (Frontend)
    ├── manifest.json                  # Konfigurasi & izin ekstensi
    ├── background.js                  # Service worker penanganan request (bypass CORS)
    ├── content.js                     # Skrip pemindai judul dan penyuntik badge
    ├── content.css                    # Desain visual untuk badge tooltip
    ├── popup.html                     # Tampilan UI untuk dropdown ekstensi
    └── popup.js                       # Skrip pengecek status koneksi server
```

## Panduan Instalasi & Penggunaan (Mode Developer)

Untuk menjalankan ekstensi ini di komputermu, ikuti dua tahapan berikut:

### Tahap 1: Menyalakan Server API Backend (FastAPI)

Ekstensi ini membutuhkan server backend untuk melakukan inferensi menggunakan model IndoBERT.

1. Buka terminal di dalam folder proyek backend (`hoax-classification-nlp`).
2. Aktifkan  *virtual environment* :
   **PowerShell**
   **PowerShell**

   ```
   .\.venv\Scripts\Activate.ps1
   ```
3. Pastikan dependensi yang dibutuhkan sudah terinstal:
   **Bash**
   **Bash**

   ```
   pip install fastapi uvicorn
   ```
4. Jalankan server lokal:
   **Bash**
   **Bash**

   ```
   uvicorn server:app --reload --port 8000
   ```

   *Biarkan terminal ini tetap terbuka dan berjalan di latar belakang.*

### Tahap 2: Memasang Ekstensi di Google Chrome

1. Buka Google Chrome.
2. Ketik `chrome://extensions/` di kolom URL lalu tekan Enter.
3. Nyalakan tombol **Developer mode** (Mode Pengembang) di pojok kanan atas.
4. Klik tombol **Load unpacked** di menu kiri atas.
5. Cari *workspace* kamu dan pilih folder `hoax_plugin` yang terpisah tadi.
6. Ekstensi IndoBERT Hoax Auto-Filter sekarang sudah terpasang.
