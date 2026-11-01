"""
models/lightgbm_model.py — Wrapped LightGBM multi-class booster.
"""
from typing import Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
import lightgbm as lgb
from .base import BaseModelWrapper
from config import cfg


class LightGBMModel(BaseModelWrapper):
    """
    LightGBM multi-class classifier using native Dataset engine to avoid sklearn wrapper deprecations.
    """

    def __init__(self, custom_params: Optional[Dict[str, Any]] = None, **kwargs):
        params = {
            'objective': 'multiclass',
            'num_class': 16,
            'metric': 'multi_logloss',
            'learning_rate': cfg.LGB_LEARNING_RATE,
            'num_leaves': cfg.LGB_NUM_LEAVES,
            'max_depth': -1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 20,
            'feature_pre_filter': False,
            'random_state': cfg.RANDOM_SEED,
            'n_jobs': -1,
            'verbose': -1,
        }
        if custom_params:
            params.update(custom_params)
        if kwargs:
            params.update(kwargs)

        self.params = params
        self.num_boost_round = cfg.LGB_N_ESTIMATORS
        self.booster: Optional[lgb.Booster] = None
        self.best_iteration_: Optional[int] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
    ):
        ds_train = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        callbacks = [lgb.log_evaluation(0)]

        valid_sets = []
        if eval_set:
            X_val, y_val = eval_set
            ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_train, free_raw_data=False)
            valid_sets.append(ds_val)
            callbacks.append(lgb.early_stopping(50, verbose=False))

        self.booster = lgb.train(
            self.params,
            ds_train,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets if valid_sets else None,
            callbacks=callbacks,
        )
        self.best_iteration_ = self.booster.best_iteration
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.booster.predict(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return proba.argmax(axis=1)

