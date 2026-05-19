import pandas as pd
import re
from sklearn.model_selection import train_test_split
import os

# --- Step 0: General Clean ---
def general_clean(text: str) -> str:
    text = re.sub(r'http\S+|www\S+', '', text)       # Remove URLs
    text = re.sub(r'[^\x00-\x7FÀ-ɏ]+', ' ', text)   # Remove emojis / non-ASCII symbols
    text = re.sub(r'\s+', ' ', text).strip()           # Normalize whitespace
    return text

# --- Step 1: Label Extraction ---
def extract_label(text: str) -> tuple[int, str]:
    label_map = {'SALAH': 1, 'KLARIFIKASI': 1, 'BENAR': 0}
    match = re.search(r'\[(SALAH|KLARIFIKASI|BENAR)\]', text, re.IGNORECASE)
    label = label_map.get(match.group(1).upper(), -1) if match else -1
    clean_text = re.sub(r'\[.*?\]', '', text).strip()
    return label, clean_text

def main():
    print("Loading data...")
    df = pd.read_csv('data/raw/turnbackhoax.csv')
    df.columns = df.columns.str.lower()
    df.dropna(subset=['title'], inplace=True)
    df['title'] = df['title'].apply(general_clean)
    
    # Extract labels
    df[['label', 'title']] = df['title'].apply(
        lambda t: pd.Series(extract_label(t))
    )
    df = df[df['label'] != -1]
    
    # Load Detik data
    if os.path.exists('data/raw/detik.csv'):
        print("Loading detik data...")
        df_detik = pd.read_csv('data/raw/detik.csv')
        df_detik.columns = df_detik.columns.str.lower()
        df_detik.dropna(subset=['title'], inplace=True)
        df_detik['title'] = df_detik['title'].apply(general_clean)
        # Detik labels = 0 (BENAR)
        df_detik['label'] = 0
        df_detik.to_csv('data/processed/detik_clean.csv', index=False)
        
        # Combine both datasets BEFORE splitting
        print("Combining datasets...")
        df_combined = pd.concat([df[['title', 'label']], df_detik[['title', 'label']]], ignore_index=True)
    else:
        print("Warning: data/raw/detik.csv not found. Using only turnbackhoax data.")
        df_combined = df[['title', 'label']]
        
    # --- Step 2: Data Splitting ---
    print("Splitting data...")
    X, y = df_combined['title'], df_combined['label']
    
    # First split: Train (80%) and Temp (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Second split: Temp into Validation (10% of total) and Test (10% of total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    # --- Pipeline A (TF-IDF / BiLSTM) ---
    print("Running Pipeline A...")
    import nltk
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    indonesia_s = set(stopwords.words('indonesian'))

    def pipeline_a(text: str) -> str:
        text = str(text).lower()                     # Case folding
        text = re.sub(r'[^a-z\s]', '', text)         # Remove punctuation and numbers
        
        # Remove Indonesian stop words (NLTK)
        words = text.split()
        words = [w for w in words if w not in indonesia_s]
        text = ' '.join(words)
        
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    X_train_a = X_train.apply(pipeline_a)
    X_val_a   = X_val.apply(pipeline_a)
    X_test_a  = X_test.apply(pipeline_a)
    
    # Save Pipeline A
    os.makedirs('data/processed/pipeline_a', exist_ok=True)
    pd.DataFrame({'text': X_train_a, 'label': y_train}).to_csv('data/processed/pipeline_a/train.csv', index=False)
    pd.DataFrame({'text': X_val_a, 'label': y_val}).to_csv('data/processed/pipeline_a/val.csv', index=False)
    pd.DataFrame({'text': X_test_a, 'label': y_test}).to_csv('data/processed/pipeline_a/test.csv', index=False)
    
    # --- Pipeline B (IndoBERT) ---
    print("Running Pipeline B...")
    os.makedirs('data/processed/pipeline_b', exist_ok=True)
    pd.DataFrame({'text': X_train, 'label': y_train}).to_csv('data/processed/pipeline_b/train.csv', index=False)
    pd.DataFrame({'text': X_val, 'label': y_val}).to_csv('data/processed/pipeline_b/val.csv', index=False)
    pd.DataFrame({'text': X_test, 'label': y_test}).to_csv('data/processed/pipeline_b/test.csv', index=False)
    
    # --- Visualization ---
    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_data = []
    for split_name, y_data in zip(['Train', 'Validation', 'Test'], [y_train, y_val, y_test]):
        for label_val, count in y_data.value_counts().items():
            plot_data.append({
                'Split': split_name,
                'Label': 'Hoax' if label_val == 1 else 'Faktual',
                'Count': count
            })
    plot_df = pd.DataFrame(plot_data)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x='Split', y='Count', hue='Label', palette=['#1f77b4', '#ff7f0e'])
    plt.title('Class Distribution across Train, Validation, and Test Sets')
    plt.ylabel('Number of Samples')
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/data_split_visualization.png')
    print("Visualization saved to results/data_split_visualization.png")
    
    print("Preprocessing completed.")

if __name__ == "__main__":
    main()