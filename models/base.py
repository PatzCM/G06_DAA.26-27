"""
models/base.py — Abstract interface for classifiers supporting probability outputs.
"""
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np
import pandas as pd


class BaseModelWrapper(ABC):
    """
    Standard interface across XGBoost, LightGBM, and scikit-learn tree ensembles.
    """

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
    ):
        """Fit model weights, optionally observing a validation set for early stopping."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict discrete class index (0 to 15)."""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probability distribution of shape (N, 16)."""
        pass

