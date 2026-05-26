import os
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

def evaluate(y_true, y_pred, model_name='Model'):
    print(f"\n=== {model_name} ===")
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Faktual', 'Hoax'])
    
    os.makedirs('results', exist_ok=True)
    
    fig, ax = plt.subplots()
    disp.plot(cmap='Reds', ax=ax)
    plt.title(f'Confusion Matrix — {model_name}')
    plt.tight_layout()
    plt.savefig(f'results/cm_{model_name.lower().replace(" ", "_")}.png')
    plt.close()
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
