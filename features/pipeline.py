"""
features/pipeline.py — Orchestrates domain feature transformers sequentially.
"""
from typing import List
import pandas as pd
from .base import BaseTransformer
from .numeric import NumericFeatureTransformer
from .health import HealthFeatureTransformer
from .demographics import DemographicFeatureTransformer
from data.cleaning import clean_survey_nulls


class FeaturePipeline(BaseTransformer):
    """
    Sequentially fits and applies all modular feature extractors.
    """

    def __init__(self, transformers: List[BaseTransformer] = None):
        self.transformers = transformers or [
            NumericFeatureTransformer(),
            HealthFeatureTransformer(),
            DemographicFeatureTransformer(),
        ]

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        X_curr = clean_survey_nulls(X)
        for t in self.transformers:
            t.fit(X_curr, y)
            X_curr = t.transform(X_curr)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_curr = clean_survey_nulls(X)
        for t in self.transformers:
            X_curr = t.transform(X_curr)
        return X_curr

