"""
data/cleaning.py — Standardizes non-response strings and categorical nulls.
"""
from typing import List, Optional
import numpy as np
import pandas as pd
from config import cfg


def clean_survey_nulls(df: pd.DataFrame, custom_nulls: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Standardize text survey non-answers into NaN so transformers can handle them systematically.
    """
    df = df.copy()
    null_values = custom_nulls or cfg.NULL_MARKERS
    
    # Replace non-informative string representations with standard NaN
    df = df.replace({val: np.nan for val in null_values})
    
    # TODO (Growth Area):
    # Some columns like 'Sem resposta' vs 'Não aplicável' carry distinction:
    # 'Não aplicável' often means condition did not occur (structural zero),
    # whereas 'Sem resposta' means missing response (random missingness).
    # You can create indicator columns for missingness patterns here!
    
    return df

