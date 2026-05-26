import pandas as pd
import re
from sklearn.model_selection import train_test_split
import os

# --- Label maps (paper-aligned + expanded variants found in actual data) ---
HOAX_PREFIXES = {"SALAH", "KLARIFIKASI", "HOAX", "HOAKS", "FITNAH", "PENIPUAN", "DISINFORMASI", "MISLEADING"}
REAL_PREFIXES = {"BENAR", "FAKTA"}


def general_clean(text: str) -> str:
    text = str(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^\x00-\x7FÀ-ɏ]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_label(text: str):
    """Return (label, clean_text). label=1 hoax, 0 factual, -1 unknown."""
    m = re.search(r'\[([A-Z\s]+)\]', str(text), re.IGNORECASE)
    if m:
        tag = m.group(1).upper().strip()
        if tag in HOAX_PREFIXES:
            label = 1
        elif tag in REAL_PREFIXES:
            label = 0
        else:
            label = -1
    else:
        label = -1
    clean_text = re.sub(r'\[.*?\]', '', str(text)).strip()
    return label, clean_text


def main():
    print("Loading turnbackhoax...")
    tbh = pd.read_csv('data/raw/turnbackhoax.csv')
    tbh.columns = tbh.columns.str.lower()
    tbh.dropna(subset=['title'], inplace=True)
    tbh['title'] = tbh['title'].apply(general_clean)

    tbh[['label', 'title']] = tbh['title'].apply(lambda t: pd.Series(extract_label(t)))
    before = len(tbh)
    tbh = tbh[tbh['label'] != -1].copy()
    print(f"  dropped {before - len(tbh)} rows with unrecognized prefix")
    print(f"  turnbackhoax labels: {tbh['label'].value_counts().to_dict()}")

    # detik = all factual
    if os.path.exists('data/raw/detik.csv'):
        print("Loading detik...")
        det = pd.read_csv('data/raw/detik.csv')
        det.columns = det.columns.str.lower()
        det.dropna(subset=['title'], inplace=True)
        det['title'] = det['title'].apply(general_clean)
        det['label'] = 0
        df = pd.concat([tbh[['title', 'label']], det[['title', 'label']]], ignore_index=True)
    else:
        print("Warning: detik.csv not found. Using turnbackhoax only.")
        df = tbh[['title', 'label']].copy()

    # --- Dedupe BEFORE split to prevent train/test leak ---
    before = len(df)
    df['_norm'] = df['title'].str.lower().str.strip()
    df = df.drop_duplicates(subset=['_norm']).drop(columns=['_norm']).reset_index(drop=True)
    print(f"Deduped: {before} -> {len(df)} rows ({before - len(df)} duplicates removed)")
    print(f"Final composition: {df['label'].value_counts().to_dict()}")

    # --- Split: 70 / 10 / 20 (per AGENT_INSTRUCTION) ---
    print("Splitting (70/10/20 stratified)...")
    X, y = df['title'], df['label']
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=2 / 3, random_state=42, stratify=y_temp
    )
    print(f"  train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # --- Pipeline A (TF-IDF / BiLSTM): lowercase + remove punct + stopwords ---
    print("Pipeline A (TF-IDF / BiLSTM)...")
    import nltk
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    indonesia_s = set(stopwords.words('indonesian'))

    def pipeline_a(text: str) -> str:
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', '', text)
        text = ' '.join(w for w in text.split() if w not in indonesia_s)
        return re.sub(r'\s+', ' ', text).strip()

    os.makedirs('data/processed/pipeline_a', exist_ok=True)
    pd.DataFrame({'text': X_train.apply(pipeline_a), 'label': y_train}).to_csv('data/processed/pipeline_a/train.csv', index=False)
    pd.DataFrame({'text': X_val.apply(pipeline_a),   'label': y_val}).to_csv('data/processed/pipeline_a/val.csv', index=False)
    pd.DataFrame({'text': X_test.apply(pipeline_a),  'label': y_test}).to_csv('data/processed/pipeline_a/test.csv', index=False)

    # --- Pipeline B (IndoBERT): raw text, no lowercase/punct/stopword removal ---
    print("Pipeline B (IndoBERT raw)...")
    os.makedirs('data/processed/pipeline_b', exist_ok=True)
    pd.DataFrame({'text': X_train, 'label': y_train}).to_csv('data/processed/pipeline_b/train.csv', index=False)
    pd.DataFrame({'text': X_val,   'label': y_val}).to_csv('data/processed/pipeline_b/val.csv', index=False)
    pd.DataFrame({'text': X_test,  'label': y_test}).to_csv('data/processed/pipeline_b/test.csv', index=False)

    # --- Visualization ---
    import matplotlib.pyplot as plt
    import seaborn as sns
    plot_data = []
    for split_name, y_data in zip(['Train', 'Validation', 'Test'], [y_train, y_val, y_test]):
        for label_val, count in y_data.value_counts().items():
            plot_data.append({'Split': split_name, 'Label': 'Hoax' if label_val == 1 else 'Faktual', 'Count': count})
    plot_df = pd.DataFrame(plot_data)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x='Split', y='Count', hue='Label', palette=['#1f77b4', '#ff7f0e'])
    plt.title('Class Distribution across Train, Validation, and Test Sets')
    plt.ylabel('Number of Samples')
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/data_split_visualization.png')
    plt.close()
    print("Done. Visualization saved to results/data_split_visualization.png")


if __name__ == "__main__":
    main()
