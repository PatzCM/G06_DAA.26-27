"""
ensemble/stacker.py — Multi-model Out-Of-Fold (OOF) Stacking with a LogisticRegression meta-learner.
"""
from typing import List, Tuple, Dict, Any, Type
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from models.base import BaseModelWrapper
from config import cfg


class OutOfFoldStacker:
    """
    K-Fold Cross-Validation Stacking:
    1. Splits training data into K folds.
    2. Trains each base model on (K-1) folds and generates out-of-fold probability predictions.
    3. Concurrently averages test fold predictions across all folds.
    4. Fits a regularized meta-learner (LogisticRegression) on stacked OOF probabilities.
    """

    def __init__(
        self,
        base_model_factories: List[Tuple[str, Type[BaseModelWrapper], Dict[str, Any]]],
        n_splits: int = 5,
        meta_c: float = 1.0,
        random_state: int = 2026,
    ):
        self.factories = base_model_factories
        self.n_splits = n_splits
        self.meta_c = meta_c
        self.random_state = random_state
        self.meta_learner = LogisticRegression(
            C=self.meta_c,
            max_iter=1000,
            random_state=self.random_state,
        )
        self.n_classes = 16

    def fit_predict(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Executes full stacking cycle.
        
        Returns:
            oof_meta_preds: Out-of-fold discrete predictions on training set.
            test_meta_preds: Discrete test set predictions.
            test_meta_probas: Probability distribution for test set.
        """
        kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        n_train = len(X_train)
        n_test = len(X_test)
        n_models = len(self.factories)

        # Feature matrix for meta-learner: N x (n_models * 16)
        oof_features = np.zeros((n_train, n_models * self.n_classes), dtype=float)
        test_features = np.zeros((n_test, n_models * self.n_classes), dtype=float)

        print(f"[Stacker] Starting {self.n_splits}-fold CV stacking across {n_models} base models...")

        for m_idx, (name, model_cls, model_kwargs) in enumerate(self.factories):
            print(f"  -> Generating OOF probabilities for base model: {name}")
            start_col = m_idx * self.n_classes
            end_col = (m_idx + 1) * self.n_classes

            for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
                X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
                X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

                # Fresh instance for each fold
                model = model_cls(**(model_kwargs or {}))
                model.fit(X_tr, y_tr, eval_set=(X_val, y_val))

                # Store out-of-fold validation probas
                val_proba = model.predict_proba(X_val)
                oof_features[val_idx, start_col:end_col] = val_proba

                # Accumulate test set predictions (divided by n_splits)
                test_proba = model.predict_proba(X_test)
                test_features[:, start_col:end_col] += test_proba / self.n_splits

        # Fit meta-learner
        print("[Stacker] Fitting LogisticRegression meta-learner on stacked OOF probabilities...")
        self.meta_learner.fit(oof_features, y_train)

        oof_meta_preds = self.meta_learner.predict(oof_features)
        test_meta_preds = self.meta_learner.predict(test_features)
        test_meta_probas = self.meta_learner.predict_proba(test_features)

        # TODO (Growth Area):
        # 1. Try passing original top features alongside probabilities:
        #    augmented_meta_train = np.hstack([oof_features, X_train[top_10_features]])
        # 2. Try an ElasticNet or small neural net as the meta-learner.

        return oof_meta_preds, test_meta_preds, test_meta_probas

