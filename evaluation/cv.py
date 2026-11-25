"""
evaluation/cv.py — Stratified K-fold cross-validation with out-of-fold (OOF) probabilities,
blend-weight tuning, and full-data refit. Shared by cv.py (experiments) and predict.py (submission).

Why: a single 80/20 holdout moves by ±0.01–0.02 Macro F1 just from the split, which is as big
as most feature changes. Average over 5 folds and look at the std before trusting a difference.
"""
import itertools
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from config import cfg
from encoders.ordinal import CategoricalOrdinalEncoder
from evaluation.metrics import calculate_macro_f1
from features.pipeline import FeaturePipeline
from models.lightgbm_model import LightGBMModel
from models.xgboost_model import XGBoostModel

N_CLASSES = 16
MODEL_REGISTRY = {"xgb": XGBoostModel, "lgb": LightGBMModel}


def preprocess(X_fit: pd.DataFrame, y_fit: pd.Series, *others: pd.DataFrame) -> List[pd.DataFrame]:
    """Fit feature pipeline + encoder on X_fit only, then transform X_fit and every frame in others."""
    feat = FeaturePipeline()
    enc = CategoricalOrdinalEncoder()
    out = [enc.fit_transform(feat.fit_transform(X_fit, y_fit))]
    out += [enc.transform(feat.transform(o)) for o in others]
    return out


@dataclass
class CVResult:
    oof: Dict[str, np.ndarray] = field(default_factory=dict)          # name -> (n, 16) probabilities
    fold_scores: Dict[str, List[float]] = field(default_factory=dict)  # name -> Macro F1 per fold
    best_iters: Dict[str, List[int]] = field(default_factory=dict)     # name -> best iteration per fold

    def summary(self, name: str) -> Tuple[float, float]:
        s = np.array(self.fold_scores[name])
        return float(s.mean()), float(s.std())

    def refit_rounds(self, name: str, data_factor: float = 1.1) -> int:
        """Trees for a full-data refit: mean CV best iteration, a bit more since there is ~25% more data."""
        return int(round(np.mean(self.best_iters[name]) * data_factor))


def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    model_params: Dict[str, dict],
    n_splits: int = cfg.N_SPLITS,
    seed: int = cfg.RANDOM_SEED,
) -> CVResult:
    """
    model_params: {"xgb": {...overrides...}, "lgb": {...}}.
    Note: early stopping uses the same fold that is scored, so scores are very slightly optimistic,
    but equally so for every experiment — fine for comparing changes.
    """
    res = CVResult()
    for name in model_params:
        res.oof[name] = np.zeros((len(X), N_CLASSES))
        res.fold_scores[name] = []
        res.best_iters[name] = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        t0 = time.time()
        X_tr, X_va = preprocess(X.iloc[tr], y.iloc[tr], X.iloc[va])
        y_tr, y_va = y.iloc[tr], y.iloc[va]
        msg = []
        for name, params in model_params.items():
            model = MODEL_REGISTRY[name](**params)
            model.fit(X_tr, y_tr, eval_set=(X_va, y_va))
            proba = model.predict_proba(X_va)
            res.oof[name][va] = proba
            score = calculate_macro_f1(y_va, proba.argmax(axis=1))
            res.fold_scores[name].append(score)
            res.best_iters[name].append(int(model.best_iteration_ or 0))
            msg.append(f"{name}={score:.4f} (it {model.best_iteration_})")
        print(f"  fold {fold}/{n_splits}: " + "  ".join(msg) + f"  [{time.time() - t0:.0f}s]", flush=True)
    return res


def blend_predict(probas: List[np.ndarray], weights, prior: np.ndarray, prior_power: float) -> np.ndarray:
    """Weighted average of probabilities, divided by class_prior**power (helps rare brackets in Macro F1)."""
    P = sum(w * p for w, p in zip(weights, probas))
    return (P / prior ** prior_power).argmax(axis=1)


def tune_blend(
    oof: Dict[str, np.ndarray], y: pd.Series, step: float = 0.1,
    prior_powers=(0.0, 0.25, 0.5, 0.75, 1.0),
) -> Tuple[Dict[str, float], float, float]:
    """Grid-search blend weights (summing to 1) and prior power on the OOF predictions."""
    names = list(oof)
    probas = [oof[n] for n in names]
    prior = np.bincount(y, minlength=N_CLASSES) / len(y)
    grid = np.round(np.arange(0, 1 + 1e-9, step), 4)

    best = (-1.0, None, None)
    for w in itertools.product(grid, repeat=len(names) - 1):
        if sum(w) > 1 + 1e-9:
            continue
        weights = list(w) + [1 - sum(w)]
        for pw in prior_powers:
            score = calculate_macro_f1(y, blend_predict(probas, weights, prior, pw))
            if score > best[0]:
                best = (score, weights, pw)

    score, weights, pw = best
    return dict(zip(names, [round(float(x), 4) for x in weights])), pw, score
