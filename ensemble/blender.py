"""
ensemble/blender.py — Weighted convex combination of model predicted probability tensors.
"""
from typing import List, Optional
import numpy as np


class ProbabilityBlender:
    """
    Blends multiple probability arrays with given or uniformly distributed weights.
    """

    def __init__(self, weights: Optional[List[float]] = None):
        self.weights = weights

    def blend(self, probas: List[np.ndarray]) -> np.ndarray:
        if not probas:
            raise ValueError("No probability arrays provided for blending.")

        n_models = len(probas)
        if self.weights is None:
            # Equal weighting by default
            w = np.ones(n_models) / n_models
        else:
            w = np.array(self.weights, dtype=float)
            w = w / w.sum()  # Normalize to unit simplex

        blended = np.zeros_like(probas[0], dtype=float)
        for weight, proba in zip(w, probas):
            blended += weight * proba

        return blended

    def predict(self, probas: List[np.ndarray]) -> np.ndarray:
        blended = self.blend(probas)
        return blended.argmax(axis=1)

