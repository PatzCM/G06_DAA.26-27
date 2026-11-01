"""
encoders/target.py — Out-of-fold target encoding with smoothing to prevent data leakage.
"""
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from features.base import BaseTransformer


class KFoldTargetEncoder(BaseTransformer):
    """
    Computes smoothed target encodings using out-of-fold cross-validation during fit,
    and maps learned conditional expectations onto test splits.
    
    Formula: smoothed_mean = (count * category_mean + smoothing * global_mean) / (count + smoothing)
    """

    def __init__(self, cols: Optional[List[str]] = None, n_splits: int = 5, smoothing: float = 10.0, random_state: int = 2026):
        self.cols = cols
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.random_state = random_state
        self.learned_mappings: Dict[str, pd.Series] = {}
        self.global_mean: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if y is None:
            raise ValueError("KFoldTargetEncoder requires a target series y to fit.")
            
        self.global_mean = float(y.mean())
        target_cols = self.cols or list(X.select_dtypes(include=['object', 'string']).columns)
        self.cols = target_cols

        # Learn global smoothed mappings for transform()
        for col in self.cols:
            stats = y.groupby(X[col]).agg(['count', 'mean'])
            smoothed = (stats['count'] * stats['mean'] + self.smoothing * self.global_mean) / (stats['count'] + self.smoothing)
            self.learned_mappings[col] = smoothed

        return self

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Generates genuine out-of-fold target encodings for the training set to prevent leakage.
        """
        if y is None:
            return self.transform(X)

        self.fit(X, y)
        X_out = X.copy()
        
        kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        for col in self.cols:
            oof_col = pd.Series(index=X.index, dtype=float)
            for tr_idx, val_idx in kf.split(X, y):
                X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
                X_val = X.iloc[val_idx]
                
                stats = y_tr.groupby(X_tr[col]).agg(['count', 'mean'])
                smoothed = (stats['count'] * stats['mean'] + self.smoothing * self.global_mean) / (stats['count'] + self.smoothing)
                
                oof_col.iloc[val_idx] = X_val[col].map(smoothed).fillna(self.global_mean)
                
            X_out[f"{col}_TE"] = oof_col

        # TODO (Growth Area):
        # 1. Expand from scalar target encoding (mean age) to multi-class probabilistic target encoding:
        #    Compute P(AGE_RANGE = k | Category = c) for each of the 16 brackets!
        # 2. Add additive Gaussian noise to OOF encodings to prevent tree overfitting.

        return X_out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        for col in self.cols:
            if col in self.learned_mappings:
                mapped = X[col].map(self.learned_mappings[col]).fillna(self.global_mean)
                X_out[f"{col}_TE"] = mapped
        return X_out

