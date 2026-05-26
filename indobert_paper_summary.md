Berikut adalah ringkasan lengkap dan terstruktur dari artikel penelitian tersebut yang diformat ke dalam Markdown. Anda dapat menyalin teks di bawah ini dan menyimpannya sebagai file `[paper]_summary.md`.

---

# Ringkasan Penelitian: Research and Analysis of IndoBERT Hyperparameter Tuning in Fake News Detection

## 1. Informasi Umum Jurnal

* 
**Judul Paper:** Research and Analysis of IndoBERT Hyperparameter Tuning in Fake News Detection 


* 
**Penulis:** Anugerah Simanjuntak, Rosni Lumbantoruan, Kartika Sianipar, Rut Gultom, Mario Simaremare, Samuel Situmeang, Erwin Panggabean 


* 
**Instansi:** Institut Teknologi Del (Toba) & STMIK Pelita Nusantara (Medan) 


* **Publikasi:** Jurnal Nasional Teknik Elektro dan Teknologi Informasi (JNTETI), Vol. 13, No. 1, Februari 2024 


* 
**DOI:** 10.22146/jnteti.v1311.8532 



---

## 2. Latar Belakang dan Permasalahan

* 
**Prevalensi Hoaks:** Kemajuan teknologi informasi meningkatkan penyebaran berita palsu (*fake news*). Kementerian Komunikasi dan Informatika (Kominfo RI) mendeteksi sekitar 800.000 situs penyebar hoaks di Indonesia pada tahun 2017.


* 
**Dampak pada Masyarakat:** Survei menunjukkan 38%–61,1% masyarakat pedesaan dan 45.3%–79,6% masyarakat perkotaan menjadi korban atau mempercayai berita palsu.


* 
**Solusi Otomatisasi:** Verifikasi manual membutuhkan banyak waktu dan tenaga. Model berbasis *Transformer* seperti BERT terbukti unggul dalam klasifikasi teks otomatis.


* 
**Kesenjangan Penelitian (*Gap*):** Model **IndoBERT** (varian BERT bahasa Indonesia) sering kali digunakan dengan nilai *hyperparameter* yang statis/bawaan (*fine-tuning* dasar), tanpa mengeksplorasi peningkatan performa melalui optimasi nilai *hyperparameter*.



---

## 3. Metodologi Penelitian

Penelitian ini berfokus pada optimasi model **IndoBERT-base-p1** menggunakan tiga metode *hyperparameter tuning*: **Grid Search**, **Random Search**, dan **Bayesian Optimization** dengan bantuan *framework* **Optuna**.

### A. Dataset

* 
**Sumber:** Dataset dari penelitian terdahulu, yaitu `TurnBackHoax.ID`.


* 
**Jumlah Data:** Total 1.116 baris data bahasa Indonesia, terdiri dari:


* 
**Real News (Label 0):** 683 artikel.


* 
**Fake News (Label 1):** 433 artikel.




* 
**Atribut:** *Label*, *Headline* (judul), dan *Body* (konten teks).


* 
**Pembagian Data (*Splitting*):** 60% Data Pelatihan (*Training*), 20% Data Validasi (*Validation*), dan 20% Data Pengujian (*Testing*).



### B. Pra-pemrosesan Teks (*Text Preprocessing*)

1. 
**Remove Missing Value:** Menghapus data kosong.


2. 
**Remove Punctuation:** Menghapus tanda baca (seperti @, %, -, dll.).


3. 
**Case Folding:** Mengubah semua huruf menjadi huruf kecil (*lowercase*).


4. 
**Stopwords Removal:** Menghapus kata-kata tidak penting menggunakan korpus *Sastrawi* (contoh: "yang", "di", "ke"). Tahap ini berhasil mengurangi jumlah total kata sebesar 0,23% (dari 1.676.646 kata menjadi 1.672.798 kata).


5. 
**Tokenization:** Memotong teks menjadi token kata individu.



### C. Eksperimen Ruang Parameter

Proses pencarian mencoba **36 kombinasi berbeda** dengan arah minimasi terhadap nilai fungsi objektif (*validation loss*). Teknik *Early Stopping* diterapkan untuk mencegah *overfitting*.

* **Pilihan Hyperparameter Dasar (Model Fine-Tuning Statis):**
* 
*Learning Rate:* $2 \times 10^{-5}$ 


* 
*Batch Size:* 16 


* 
*Epochs:* 10 




* **Rentang Ruang Pencarian Tuning:**
* 
*Learning Rate:* $2 \times 10^{-5}$, $3 \times 10^{-5}$, $5 \times 10^{-5}$ 


* 
*Batch Size:* 16, 32 


* 
*Epochs:* Maksimal 10 





---

## 4. Hasil dan Pembahasan

### A. Nilai Hyperparameter Optimal yang Ditemukan

Berdasarkan uji coba, masing-masing metode menghasilkan kombinasi nilai parameter terbaik sebagai berikut:

| Metode Tuning | Learning Rate | Epoch | Batch Size |
| --- | --- | --- | --- |
| **Grid Search** | $5 \times 10^{-5}$ | 8 | 32 |
| **Random Search** | $5 \times 10^{-5}$ | 9 | 32 |
| **Bayesian Optimization** | $2 \times 10^{-5}$ | 8 | 16 |

*(Sumber: Table III)* 

### B. Analisis Grafik Loss & Overfitting

* 
**Grid Search & Random Search:** Mengalami masalah *overfitting*. Grafik *validation loss* sempat turun namun kembali naik setelah beberapa *epoch* awal, sementara *training loss* terus menurun. Random Search mencatat performa *loss* terburuk karena ketidakmampuannya menggeneralisasi data baru.


* 
**Bayesian Optimization:** Mengalami *early stop* pada **epoch ke-7** dan terbukti **bebeas dari overfitting**. Kurva *validation loss* selaras menurunkan nilai seiring bertambahnya epoch, membuktikan kemampuan generalisasi yang baik.



### C. Perbandingan Kinerja Deteksi Berita Palsu (Label "Fake")

Metrik di bawah menunjukkan perbandingan antara model dasar (*Fine-Tuning*) dengan hasil setelah dilakukan *Hyperparameter Tuning*:

| Metode | Precision | Recall | F1-Score | Accuracy |
| --- | --- | --- | --- | --- |
| **Model Dasar (Fine-Tuning)** | 0.8632 | 0.9266 | 0.8938 | 0.9313 |
| **Model + Grid Search** | 0.8661 (+0.33%) | 0.8899 (-4.12%) | 0.8899 (-0.44%) | 0.9194 (-1.29%) |
| **Model + Random Search** | 0.8462 (-2.01%) | 0.9083 (+1.95%) | 0.8761 (-2.02%) | 0.9164 (-1.63%) |
| **Model + Bayesian Optimization** | **0.8879 (+2.78%)** | **0.9450** | **0.9156 (+2.38%)** | **0.9432 (+1.26%)** |

*(Sumber: Table VII, VIII, IX, X)* 

> 
> **Catatan Utama:** Hanya metode **Bayesian Optimization** yang berhasil memberikan peningkatan performa positif di seluruh metrik evaluasi dibandingkan model dasar (*fine-tuning*). Grid Search dan Random Search justru mengalami penurunan performa akibat kendala interaksi non-linear parameter yang tidak mampu dicakup secara efektif.
> 
> 

### D. Perbandingan Waktu (*Time-Cost*)

Proses pencarian otomatis ini memerlukan komputasi tambahan, namun waktu tambahan ini **hanya dibutuhkan sekali** pada fase training.

* Model Dasar (Fine-Tuning): **70 menit** 


* Model + Random Search: **77 menit** (+10.00%) 


* Model + Bayesian Optimization: **84 menit** (+20.00%) 


* Model + Grid Search: **88 menit** (+25.71%) 



---

## 5. Kesimpulan dan Saran

### Kesimpulan

1. Metode **Bayesian Optimization** terbukti menjadi teknik *hyperparameter tuning* terbaik untuk model IndoBERT-base-p1 dalam mendeteksi hoaks berbahasa Indonesia menggunakan dataset TurnBackHoax.ID.


2. Model dengan optimasi Bayesian berhasil mencapai performa tertinggi dengan **Precision 88,79%**, **Recall 94,50%**, **F1-Score 91,56%**, dan **Accuracy 94,32%**.


3. Meskipun membutuhkan tambahan waktu pelatihan sebanyak 20% (menjadi 84 menit), konsekuensi biaya waktu ini dinilai sebanding (*tolerable*) dengan peningkatan akurasi dan hilangnya efek *overfitting*.



### Saran Penelitian Selanjutnya

* Disarankan melakukan evaluasi komparatif menggunakan **algoritma evolusioner (*evolutionary algorithms*)** serta optimasi berbasis gradien (*gradient-based optimization*) untuk mengeksplorasi efisiensi waktu dan efektivitas akurasi model secara berkala.