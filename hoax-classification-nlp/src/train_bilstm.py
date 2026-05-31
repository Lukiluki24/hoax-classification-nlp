import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from evaluate import evaluate

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                               batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.bilstm(x)
        # Use global average pooling over the sequence length to handle padding correctly
        out = torch.mean(out, dim=1)
        return self.fc(out)

def build_vocab_and_sequences(texts, max_vocab=10000, max_len=100):
    vectorizer = CountVectorizer(max_features=max_vocab)
    vectorizer.fit(texts)
    vocab = vectorizer.vocabulary_
    
    # Simple tokenization based on vocab
    sequences = []
    for text in texts:
        words = str(text).split()
        seq = [vocab[w] + 1 for w in words if w in vocab] # +1 for padding index 0
        if len(seq) > max_len:
            seq = seq[:max_len]
        else:
            seq = seq + [0] * (max_len - len(seq))
        sequences.append(seq)
    
    return torch.tensor(sequences, dtype=torch.long), vocab

def main():
    print("Loading Pipeline A data for BiLSTM...")
    train_df = pd.read_csv('data/processed/pipeline_a/train.csv').dropna()
    val_df = pd.read_csv('data/processed/pipeline_a/val.csv').dropna()
    test_df = pd.read_csv('data/processed/pipeline_a/test.csv').dropna()
    
    X_train_seq, vocab = build_vocab_and_sequences(train_df['text'])
    
    # Re-use the same vectorizer vocab for val and test
    def text_to_seq(texts, vocab, max_len=100):
        sequences = []
        for text in texts:
            words = str(text).split()
            seq = [vocab[w] + 1 for w in words if w in vocab]
            if len(seq) > max_len:
                seq = seq[:max_len]
            else:
                seq = seq + [0] * (max_len - len(seq))
            sequences.append(seq)
        return torch.tensor(sequences, dtype=torch.long)

    X_val_seq = text_to_seq(val_df['text'], vocab)
    X_test_seq = text_to_seq(test_df['text'], vocab)

    y_train = torch.tensor(train_df['label'].values, dtype=torch.long)
    y_val = torch.tensor(val_df['label'].values, dtype=torch.long)
    y_test = torch.tensor(test_df['label'].values, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_seq, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_seq, y_val), batch_size=32)
    test_loader = DataLoader(TensorDataset(X_test_seq, y_test), batch_size=32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = len(vocab) + 1 # +1 for padding
    model = BiLSTMClassifier(vocab_size=vocab_size).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 5
    print("Training BiLSTM...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    os.makedirs('models/bilstm_model', exist_ok=True)
    torch.save(model.state_dict(), 'models/bilstm_model/bilstm_weights.pth')
    
    print("Evaluating BiLSTM on Test Set...")
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
    
    evaluate(y_test.numpy(), all_preds, model_name='BiLSTM')

if __name__ == "__main__":
    main()
