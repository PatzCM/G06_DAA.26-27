"""
models/tree_baselines.py — Bagged tree baselines (RandomForest and ExtraTrees) for ensemble diversity.
"""
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from .base import BaseModelWrapper
from config import cfg


class RandomForestModel(BaseModelWrapper):
    """RandomForest baseline providing smooth bagging probabilities."""

    def __init__(self, n_estimators: int = 300, min_samples_leaf: int = 3, **kwargs):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=cfg.RANDOM_SEED,
            n_jobs=-1,
            **kwargs,
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)


class ExtraTreesModel(BaseModelWrapper):
    """ExtraTrees baseline introducing extreme randomization for uncorrelated errors."""

    def __init__(self, n_estimators: int = 300, min_samples_leaf: int = 3, **kwargs):
        self.model = ExtraTreesClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=cfg.RANDOM_SEED,
            n_jobs=-1,
            **kwargs,
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

