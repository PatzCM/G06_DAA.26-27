"""
features/base.py — Abstract base class for feature transformations.
"""
from abc import ABC, abstractmethod
import pandas as pd


class BaseTransformer(ABC):
    """
    Interface for sklearn-compatible feature transformers.
    Follows fit() and transform() contract to prevent train-test data leakage.
    """
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit transformer state on training data."""
        return self

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature transformation."""
        pass

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Fit and transform in a single call."""
        return self.fit(X, y).transform(X)

