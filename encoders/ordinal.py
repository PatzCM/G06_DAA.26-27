"""
encoders/ordinal.py — Generic OrdinalEncoder wrapper handling unseen categories.
"""
from typing import List, Optional
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from features.base import BaseTransformer


class CategoricalOrdinalEncoder(BaseTransformer):
    """
    Encodes remaining string/object columns into integer tokens using sklearn's OrdinalEncoder,
    safely mapping unknown categories to -1.
    """

    def __init__(self, fill_na_value: float = -1.0):
        self.fill_na_value = fill_na_value
        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.categorical_cols: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self.categorical_cols = list(X.select_dtypes(include=['object', 'string', 'category']).columns)
        if self.categorical_cols:
            self.encoder.fit(X[self.categorical_cols].astype(str))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.categorical_cols:
            encoded_vals = self.encoder.transform(X[self.categorical_cols].astype(str))
            X[self.categorical_cols] = encoded_vals

        X = X.fillna(self.fill_na_value)
        return X

