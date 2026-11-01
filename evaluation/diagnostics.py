"""
evaluation/diagnostics.py — Diagnoses exact vs off-by-one bracket predictions.
"""
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def analyze_bracket_confusion(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Measures the exact hit rate vs off-by-one rate.
    Because Macro F1 gives 0 credit for being off by 1 bracket,
    this metric pinpoints how close the model is to unlocking higher F1 scores.
    """
    diff = np.abs(np.array(y_true) - np.array(y_pred))
    n_total = len(diff)

    exact = (diff == 0).sum() / n_total
    off_by_1 = (diff == 1).sum() / n_total
    off_by_2 = (diff == 2).sum() / n_total
    off_by_more = (diff > 2).sum() / n_total

    print("\n--- Diagnostic: Prediction Proximity Analysis ---")
    print(f"  Exact Bracket Matches (diff == 0) : {exact*100:5.2f}%")
    print(f"  Off by 1 Bracket      (diff == 1) : {off_by_1*100:5.2f}%")
    print(f"  Off by 2 Brackets     (diff == 2) : {off_by_2*100:5.2f}%")
    print(f"  Off by >2 Brackets    (diff  > 2) : {off_by_more*100:5.2f}%")
    print(f"  Total within ±1 Bracket           : {(exact + off_by_1)*100:5.2f}%")
    print("-------------------------------------------------")

    return {
        "exact_rate": exact,
        "off_by_1_rate": off_by_1,
        "off_by_2_rate": off_by_2,
        "off_by_more_rate": off_by_more,
    }

