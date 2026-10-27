"""Lưu kết quả đánh giá; dùng lại cho Test và external."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from config import CLASSES, REPORTS
from utils import write_json


def save_evaluation(y, predictions, prefix):
    labels = list(range(len(CLASSES)))
    metrics = {'n': len(y), 'accuracy': accuracy_score(y, predictions),
               'macro_f1': f1_score(y, predictions, labels=labels, average='macro', zero_division=0)}
    write_json(REPORTS / f'{prefix}_metrics.json', metrics)
    pd.DataFrame([metrics]).to_csv(REPORTS / f'{prefix}_metrics.csv', index=False)
    report = classification_report(y, predictions, labels=labels, target_names=CLASSES,
                                   output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(REPORTS / f'{prefix}_classification_report.csv')
    (REPORTS / f'{prefix}_classification_report.txt').write_text(
        classification_report(y, predictions, labels=labels, target_names=CLASSES, zero_division=0), encoding='utf-8')
    cm = confusion_matrix(y, predictions, labels=labels)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(REPORTS / f'{prefix}_confusion_matrix.csv')
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap='Blues')
    for i in labels:
        for j in labels:
            ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                    color='white' if cm[i,j] > cm.max()/2 else 'black')
    ax.set(xticks=labels, yticks=labels, xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel='Predicted', ylabel='True', title=prefix)
    fig.tight_layout()
    fig.savefig(REPORTS / f'{prefix}_confusion_matrix.png', dpi=150)
    plt.close(fig)
    return metrics
