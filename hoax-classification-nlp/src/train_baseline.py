import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import joblib
import os
from evaluate import evaluate

def main():
    print("Loading Pipeline A data...")
    train_df = pd.read_csv('data/processed/pipeline_a/train.csv').dropna()
    test_df = pd.read_csv('data/processed/pipeline_a/test.csv').dropna()
    
    X_train_a, y_train = train_df['text'], train_df['label']
    X_test_a, y_test = test_df['text'], test_df['label']

    print("Vectorizing with TF-IDF...")
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train_a)
    X_test_tfidf  = tfidf.transform(X_test_a)

    print("Training Ensemble Baseline...")
    rf  = RandomForestClassifier(n_estimators=200, random_state=42)
    svm = SVC(probability=True, kernel='rbf', random_state=42)
    nb  = MultinomialNB()

    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('svm', svm), ('nb', nb)],
        voting='soft'
    )
    ensemble.fit(X_train_tfidf, y_train)

    os.makedirs('models', exist_ok=True)
    joblib.dump({'model': ensemble, 'vectorizer': tfidf}, 'models/ensemble_baseline.pkl')
    
    print("Evaluating Ensemble Baseline on Test Set...")
    y_pred = ensemble.predict(X_test_tfidf)
    evaluate(y_test, y_pred, model_name='Ensemble Baseline')

if __name__ == "__main__":
    main()
