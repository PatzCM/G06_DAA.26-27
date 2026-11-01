"""
features/numeric.py — Automatically converts numeric columns masked as strings and calculates BMI.
"""
from typing import List, Optional
import numpy as np
import pandas as pd
from .base import BaseTransformer


class NumericFeatureTransformer(BaseTransformer):
    """
    Detects and converts columns containing numeric string values (e.g. '0200', '15.0')
    into genuine floating point numbers, and computes body mass index (BMI).
    """

    def __init__(self, threshold_numeric_ratio: float = 0.5):
        self.threshold = threshold_numeric_ratio
        self.discovered_numeric_cols: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Discovers which string columns are predominantly numeric."""
        self.discovered_numeric_cols = []
        object_cols = X.select_dtypes(include=['object', 'string']).columns

        for col in object_cols:
            if col in ['BM1', 'BM2']:
                continue
            series = X[col]
            valid_non_null = series.dropna()
            if len(valid_non_null) == 0:
                continue
            
            # Test how many values parse as numeric
            parsed = pd.to_numeric(valid_non_null, errors='coerce')
            ratio = parsed.notna().sum() / len(valid_non_null)
            if ratio >= self.threshold:
                self.discovered_numeric_cols.append(col)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        
        # Convert detected numeric columns
        for col in self.discovered_numeric_cols:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')

        # BMI Computation
        if 'BM1' in X.columns and 'BM2' in X.columns:
            height = pd.to_numeric(X['BM1'], errors='coerce')
            weight = pd.to_numeric(X['BM2'], errors='coerce')
            
            X['HEIGHT'] = height
            X['WEIGHT'] = weight
            # Formula: weight (kg) / [height (m)]^2
            X['BMI'] = weight / ((height / 100.0) ** 2)
            
            # TODO (Growth Area):
            # Add WHO BMI categories (Underweight < 18.5, Normal 18.5-25, Overweight 25-30, Obese > 30)
            # Add non-linear interactions: HEIGHT * WEIGHT, log(BMI)
            
            X = X.drop(columns=['BM1', 'BM2'])

        return X

