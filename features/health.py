"""
features/health.py — Health perception, chronic illnesses, and medical limitations.
"""
import pandas as pd
from .base import BaseTransformer


class HealthFeatureTransformer(BaseTransformer):
    """
    Extracts high-signal health indicators from survey answers:
    - Chronic disease counts (CD1A - CD1P)
    - Ordinal health perception mapping (HS1)
    - Physical limitations and disability flags
    """

    HS1_MAP = {
        'Muito mau': 0,
        'Mau': 1,
        'Razoável': 2,
        'Bom': 3,
        'Muito bom': 4,
        'Sem resposta': -1,
    }

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # 1. Chronic disease total counter
        chronic_cols = [c for c in X.columns if c.startswith('CD1') and len(c) == 4]
        if chronic_cols:
            X['CHRONIC_DISEASE_COUNT'] = (X[chronic_cols] == 'Sim').sum(axis=1)

        # 2. Ordinal Health Perception (HS1)
        if 'HS1' in X.columns:
            X['HS1_ORDERED'] = X['HS1'].map(self.HS1_MAP).fillna(-1)

        # 3. Longstanding health problem (HS2)
        if 'HS2' in X.columns:
            X['HAS_CHRONIC_CONDITION'] = (X['HS2'] == 'Sim').astype(int)

        # TODO (Growth Area):
        # 1. Cluster chronic diseases into cardio-metabolic vs musculo-skeletal.
        # 2. Check medicine intake columns (MD1, MD2) and doctor consultation frequency (AM1-AM7).
        # 3. Create ratio: CHRONIC_DISEASE_COUNT / (HS1_ORDERED + 1) to measure vulnerability.

        return X

