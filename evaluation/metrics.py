"""
evaluation/metrics.py — Official competition metric calculation and per-bracket scoring.
"""
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report


def calculate_macro_f1(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """Computes Macro-averaged F1 score across all 16 age brackets."""
    return float(f1_score(y_true, y_pred, average='macro'))


def print_classification_summary(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
    """Prints per-bracket precision, recall, and F1 scores."""
    score = calculate_macro_f1(y_true, y_pred)
    print(f"\n==========================================")
    print(f" Macro F1 Score: {score:.4f}")
    print(f"==========================================")
    
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    print("\nPer-Bracket Performance Breakdown:")
    for cls in range(16):
        s_cls = str(cls)
        if s_cls in report_dict:
            metrics = report_dict[s_cls]
            print(
                f"  Bracket {cls:2d} | F1: {metrics['f1-score']:.3f} | "
                f"Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f} | "
                f"Support: {int(metrics['support']):4d}"
            )
            
    return report_dict

