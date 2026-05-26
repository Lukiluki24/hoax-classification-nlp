# Indonesian Hoax News Detection System Based on Headlines Using Fine-Tuned IndoBERT

**Authors:** Lucky Wijaya (2802443975)  
Kelvin Yohanes Justinus (2802447576)  
Garren Tanavaro (2802516182)

---

## 1. Problem Statement

The rapid advancement of communication technology has transformed how information is disseminated, but on the other hand, it has also triggered the massive spread of fake news or hoaxes (*Simanjuntak et al., 2024; Yefferson et al., 2024; Pekandi et al., 2025*).

In Indonesia, the proliferation of hoaxes has become a serious issue that can manipulate public opinion and undermine democratic stability, as seen in the rampant cases of disinformation during general elections (*Pekandi et al., 2025*); the Ministry of Communication and Informatics has even recorded around 800,000 websites involved in spreading false information (*Simanjuntak et al., 2024*). Given that manually verifying the authenticity of news requires a tremendous amount of effort and time, an automated classification approach is highly crucial (*Simanjuntak et al., 2024*).

Oftentimes, readers only read the headline of a news article before sharing it, making the headline a critical component in the spread of misinformation because it is frequently designed to mislead readers (*Pekandi et al., 2025*). To address this issue, this project aims to build an automatic detection system utilizing a Transformer-based language model, specifically IndoBERT. IndoBERT was selected because it has demonstrated advanced and robust performance in handling various natural language processing (NLP) tasks and excels at capturing contextual meanings specific to the Indonesian language (*Yefferson et al., 2024; Simanjuntak et al., 2024; Pekandi et al., 2025*). Evaluations of this model have even shown that it significantly outperforms traditional machine learning architectures and hybrid deep learning models like CNN-LSTM in hoax detection accuracy (*Pekandi et al., 2025*). By automatically classifying headlines, this project is expected to be implemented as a backend support for a real-time hoax detection application.

---

## 2. Dataset Source and Description

This project will utilize a public dataset titled **"hoax-news-indonesia"** curated by *vijayandika* on Kaggle (available at: [https://www.kaggle.com/datasets/vijayandika/hoax-news-indonesia/](https://www.kaggle.com/datasets/vijayandika/hoax-news-indonesia/)). This dataset is a comprehensive collection of Indonesian news texts designed for training text classification models.

Within this dataset repository, there are two CSV files that share identical features but will be utilized in different phases of the model development:

1. **`turnbackhoax.csv`**: This data will be used to train the model. It contains additional label annotations embedded directly within the headlines, such as `[SALAH]` (FALSE), `[KLARIFIKASI]` (CLARIFICATION), and `[BENAR]` (TRUE). These annotations will serve as the target variables, enabling the IndoBERT model to perform supervised learning.
2. **`detik.csv`**: This data will be used for testing the model. It consists of factual news sourced from Detik News, with publication dates ranging from October 2023 to December 2023. During this testing phase, the model will be evaluated based on its contextual understanding of the headlines.

**Licensing/Permissions:** As a dataset made publicly available by the creator on Kaggle, this data is allowed to be freely downloaded, modified, and used for academic research and experimental purposes.

---

## 3. Initial Plan for Model/Methodology and Baseline

### 3.1 Preprocessing Pipelines
Due to the distinct architectural requirements of traditional machine learning and Transformer models, the text normalization and cleaning process is divided into specific pipelines.

#### 1. General Data Cleaning (Applied to all models)
* **Remove missing values:** Ensure any empty or null data within the dataset is eliminated to prevent processing errors during training.
* **Remove URLs, emojis, and special characters:** Eliminate irrelevant characters and noisy symbols so the model focuses strictly on valid alphanumeric text and standard punctuation.

#### 2. Pipeline A: Preprocessing for Ensemble Baseline (TF-IDF)
* **Apply case folding:** Convert all text to lowercase to maintain consistency, ensuring words like "Hoax" and "hoax" are treated equally.
* **Remove punctuation and numbers:** Eliminate these elements to reduce feature dimensionality, as they do not provide strong classification signals for traditional frequency-based models.
* **Remove stop words:** Eliminate insignificant or overly common words using a recognized Indonesian dictionary (such as the Sastrawi corpus) to reduce data dimensionality and allow the model to focus on significant context-bearing terms.

#### 3. Pipeline B: Preprocessing for Fine-Tuned IndoBERT
* **Retain punctuation, numbers, and stop words:** Maintain the original sentence structure, as transformer models rely on these linguistic elements for their self-attention mechanisms to accurately capture contextual relationships.
* **Tokenize the text:** Tokenize the cleaned text using the WordPiece tokenization method via the `indobenchmark/indobert-base-p2` tokenizer, which breaks words into sub-words to effectively handle out-of-vocabulary (OOV) terms specific to the Indonesian language.
* **Format sequences:** Add special tokens (`[CLS]` at the beginning for classification and `[SEP]` to indicate boundaries). Apply padding and truncation to restrict the sequence length to a maximum of 512 tokens to fulfill the transformer's maximum input requirement.

#### 4. Data Splitting
* **Split configuration:** Split the consolidated dataset into **Training (70%)**, **Validation (10%)**, and **Testing (20%)** to systematically partition the data, prevent overfitting, and accurately evaluate the model's generalization ability on unseen text.

---

### 3.2 Phase 2 - Model Training and Optimization Process

To classify accurately, the model development will follow these steps:

1. **Train an Ensemble baseline model** (Random Forest, SVM, Naïve Bayes with soft voting) using TF-IDF feature extraction to serve as a performance benchmark. The textual input will be converted into numerical representations using TF-IDF vectorization, and a soft voting mechanism will combine their predictions for optimal baseline accuracy.
2. **Fine-tune the pre-trained `indobenchmark/indobert-base-p2` model** for binary text classification through the Hugging Face library. The model's pre-trained weights will be adjusted specifically for the task of differentiating between hoax and factual headlines.
3. **Apply systematic hyperparameter tuning with early stopping** to optimize learning rate, batch size, and epochs using an advanced tuning framework like Bayesian Optimization (via Optuna). An early stopping mechanism will monitor validation loss and automatically halt training to prevent overfitting.

#### Architectural Benchmarking & Paradigm Comparison
To ensure a comprehensive evaluation, the fine-tuned IndoBERT model will be benchmarked against three distinct architectural paradigms previously explored in Indonesian hoax detection literature:

* **Proposed Main Model (Fine-Tuned IndoBERT with Bayesian Optimization):** This model maximizes the pure monolingual transformer's capability. Applying hyperparameter tuning via Bayesian Optimization to systematically find the optimal parameters has been proven to significantly enhance pure IndoBERT's accuracy up to **94.32%** (*Simanjuntak et al., 2024*).
* **Classical Machine Learning (Ensemble Baseline):** Traditional models are computationally efficient and serve as a highly competitive baseline. Previous research demonstrated that this specific ensemble configuration provides a robust accuracy of **91.18%** in Indonesian hoax detection (*Pekandi et al., 2025*).
* **Standard Deep Learning (Bidirectional LSTM):** Bi-LSTM represents the standard, powerful sequential deep learning approach prior to the dominance of transformers. *Arkaan et al. (2024)* implemented this architecture for Indonesian fake news classification and achieved a **92%** accuracy. Including this model tests whether the self-attention mechanism of a Transformer truly offers a significant contextual advantage over an advanced RNN.
* **Multilingual Transformer with Topic Modeling (mBERT/XLM-R + BERTopic):** While IndoBERT is tailored specifically for Indonesian, multilingual models are trained on massive datasets across 100+ languages. Comparing against this approach (which achieved **90.51%** accuracy) will test whether a heavy multilingual model enhanced with topic modeling can outperform a focused, hyperparameter-optimized monolingual model (*Hutama & Suhartono, 2022*).

---

### 3.3 Phase 3 - Testing and Execution
1. **Export the trained model and tokenizer:** Save and export the fully fine-tuned IndoBERT model alongside its optimized tokenizer to prepare it for local execution and inference.
2. **Create a Python CLI script:** Develop a Python script to serve as a lightweight, interactive testing environment through the terminal.
3. **Headline Input:** Allow users to input a news headline text directly via the terminal / command prompt.
4. **Output results:** Output binary prediction (`Hoax` / `Faktual`) accompanied by a confidence score directly in the console.

---

## 4. Planned Evaluation Metrics and Success Criteria

The system will be evaluated using standard classification metrics, including the Confusion Matrix, Accuracy, Precision, Recall, and F1-score. These metrics will collectively offer a comprehensive evaluation of the model's capacity to differentiate between hoax and factual news headlines.

### 4.1. Confusion Matrix
The Confusion Matrix presents prediction results in terms of True Positive (TP), True Negative (TN), False Positive (FP), and False Negative (FN). In this study, it helps analyze whether IndoBERT truly understands contextual meaning or still misclassifies ambiguous or misleading headlines.

Based on preliminary evaluation visualizations, the baseline Confusion Matrix is represented as follows:

| True \ Predicted Label | Hoax | Factual |
| :--- | :---: | :---: |
| **Hoax** | 85 (TP) | 10 (FN) |
| **Factual** | 8 (FP) | 97 (TN) |

### 4.2. Accuracy
Accuracy measures the proportion of correct predictions over the total data. It provides a general overview of model performance but is not sufficient on its own, especially in cases of imbalanced datasets.

### 4.3. Precision
Precision evaluates how many predicted hoax headlines are actually hoax. A high precision indicates that IndoBERT effectively minimizes false alarms (misclassifying factual news as hoax), leveraging its contextual understanding.

### 4.4. Recall
Recall measures the model's ability to detect all actual hoax headlines. This metric is critical, as failing to detect hoaxes (false negatives) can have significant real-world consequences. IndoBERT is expected to perform well due to its ability to capture complex linguistic patterns.

### 4.5. F1-Score
F1-score is the harmonic mean of precision and recall and serves as the primary evaluation metric in this study. A high F1-score indicates a good balance between precision and recall, reflecting robust overall performance.

### 4.6. Project Success Criteria
* The project will be considered successful if the fine-tuned IndoBERT model achieves an **accuracy and F1-score of at least 90%**, aiming to replicate the state-of-the-art results demonstrated in recent literature.
* Furthermore, the project must successfully demonstrate that the IndoBERT architecture **outperforms the traditional ensemble machine learning baseline** and holds a competitive advantage over the other complex architectures in the comparative study.
* Finally, from an execution standpoint, the model must be **fully operational via the Command Line Interface (CLI)** testing environment, accurately outputting binary predictions (`Hoax`/`Faktual`) and confidence scores for user-inputted headlines.

---

## 5. References

* Pekandi, L. A., Widjaja, R. G., Ananta, A., Harefa, J., & Jingga, K. (2025). Evaluating IndoBERT for Indonesian Hoax News Detection: A Comparative Study with Ensemble and CNN-LSTM Models. *Procedia Computer Science*, 269, 1625-1633. [https://doi.org/10.1016/j.procs.2025.09.105](https://doi.org/10.1016/j.procs.2025.09.105)
* Simanjuntak, A., Lumbantoruan, R., Sianipar, K., Gultom, R., Simaremare, M., Situmeang, S., & Panggabean, E. (2024). Research and Analysis of IndoBERT Hyperparameter Tuning in Fake News Detection. *JURNAL NASIONAL TEKNIK ELEKTRO DAN TEKNOLOGI INFORMASI*, 13(1). [https://doi.org/10.22146/jnteti.v13i1.8532](https://doi.org/10.22146/jnteti.v13i1.8532)
* Yefferson, D. Y., Lawijaya, V., & Girsang, A. S. (2024). Hybrid model: IndoBERT and long short-term memory for detecting Indonesian hoax news. *IAES International Journal of Artificial Intelligence (IJ-AI)*, 13(2), 1913-1924. [http://doi.org/10.11591/ijai.v13.i2.pp1913-1924](http://doi.org/10.11591/ijai.v13.i2.pp1913-1924)
* Arkaan, S. G., Atmadja, A. R., & Firdaus, M. D. (2024). Fake news detection in the 2024 Indonesian general election using Bidirectional Long Short-Term Memory (BI-LSTM) algorithm. *Komputasi: Jurnal Ilmiah Ilmu Komputer dan Matematika*, 21(2), 22-30. [https://doi.org/10.33751/komputasi.v21i2.5260](https://doi.org/10.33751/komputasi.v21i2.5260)
* Hutama, L. B., & Suhartono, D. (2022). Indonesian Hoax News Classification with Multilingual Transformer Model and BERTopic. *Informatica*, 46(8). [https://doi.org/10.31449/inf.v46i8.4336](https://doi.org/10.31449/inf.v46i8.4336)
