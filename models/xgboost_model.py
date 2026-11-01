"""
models/xgboost_model.py — Wrapped XGBoost multi-class classifier with early stopping.
"""
from typing import Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from .base import BaseModelWrapper
from config import cfg


class XGBoostModel(BaseModelWrapper):
    """
    XGBoost classifier configured for 16-class multinomial logloss.
    """

    def __init__(self, custom_params: Optional[Dict[str, Any]] = None, **kwargs):
        params = {
            'n_estimators': cfg.XGB_N_ESTIMATORS,
            'learning_rate': cfg.XGB_LEARNING_RATE,
            'max_depth': cfg.XGB_MAX_DEPTH,
            'subsample': cfg.XGB_SUBSAMPLE,
            'colsample_bytree': cfg.XGB_COLSAMPLE,
            'min_child_weight': 3,
            'random_state': cfg.RANDOM_SEED,
            'n_jobs': -1,
            'eval_metric': 'mlogloss',
            'early_stopping_rounds': 50,
        }
        if custom_params:
            params.update(custom_params)
        if kwargs:
            params.update(kwargs)
            
        self.params = params
        self.model = XGBClassifier(**params)
        self.best_iteration_: Optional[int] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
    ):
        if eval_set:
            X_val, y_val = eval_set
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            self.best_iteration_ = getattr(self.model, 'best_iteration', None)
        else:
            self.model.fit(X_train, y_train, verbose=False)
            
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

