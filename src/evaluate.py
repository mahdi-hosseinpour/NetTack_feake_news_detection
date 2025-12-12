# src/evaluate.py
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from src.config import Config


def evaluate_model(model, data):

    model.eval()
    data = data.to(Config.DEVICE)

    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)

    mask = data.test_mask
    y_true = data.y[mask].cpu().numpy()
    y_pred = pred[mask].cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    print("\n" + "=" * 50)
    print("Last results for test datas:")
    print("=" * 50)
    print(f"Accuracy:           {acc:.4f} ({acc * 100:.2f}%)")
    print(f"F1-Score (Macro):   {f1_macro:.4f}")
    print(f"F1-Score (Weighted): {f1_weighted:.4f}")
    print("=" * 50)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(set(y_true)),
                yticklabels=sorted(set(y_true)))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(Config.RESULTS_DIR / "confusion_matrix.png")
    plt.show()

    return acc, f1_macro, f1_weighted