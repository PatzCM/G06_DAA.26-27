"""
predict.py — Build the Kaggle submission.

Default: the group's OOF stacking (XGB + LGB + RF -> LogisticRegression meta-learner).

Optional --blend flow:
  1. 5-fold CV of XGBoost + LightGBM -> OOF Macro F1 (honest estimate), best iterations per fold.
  2. Tune blend weights + prior correction on the OOF probabilities.
  3. Refit each model on ALL training data with the CV-chosen number of trees.
  4. Predict test, validate the file format, save, and print a description to paste on Kaggle.

    python predict.py              # stacking (default, unchanged behaviour)
    python predict.py --blend      # CV-tuned XGB + LGB blend, refit on all data
"""
import argparse

import numpy as np
import pandas as pd

from config import cfg
from cv import git_commit
from data.loader import load_train_data, load_test_data
from ensemble.stacker import OutOfFoldStacker
from evaluation.cv import MODEL_REGISTRY, N_CLASSES, blend_predict, cross_validate, preprocess, tune_blend
from evaluation.metrics import calculate_macro_f1
from evaluation.submission import validate_submission
from models.lightgbm_model import LightGBMModel
from models.tree_baselines import RandomForestModel
from models.xgboost_model import XGBoostModel


def save_submission(test_preds: np.ndarray, n_test: int, filename: str, description: str) -> None:
    sub = pd.DataFrame({"RowId": np.arange(n_test), "AGE_RANGE": np.asarray(test_preds).astype(int)})
    validate_submission(sub, n_test)
    out_path = cfg.SUBMISSION_DIR / filename
    sub.to_csv(out_path, index=False)
    print(f"[Done] Saved to: {out_path}")
    print("\nPaste this as the Kaggle submission description:")
    print(f"  {description}")


def generate_blend_submission(models=("xgb", "lgb"), output_filename: str = "submission_blend.csv"):
    X_train, y_train, _ = load_train_data()
    X_test, _ = load_test_data()

    print(f">>> 1. {cfg.N_SPLITS}-fold CV for {list(models)}...")
    res = cross_validate(X_train, y_train, {m: {} for m in models})
    for m in models:
        mean, std = res.summary(m)
        print(f"  {m}: OOF Macro F1 {mean:.4f} ± {std:.4f} | best iterations {res.best_iters[m]}")

    print(">>> 2. Tuning blend weights + prior correction on OOF...")
    weights, prior_power, oof_f1 = tune_blend(res.oof, y_train)
    print(f"  OOF blend Macro F1: {oof_f1:.4f}  weights={weights}  prior_power={prior_power}")

    print(">>> 3. Refitting on ALL training data...")
    X_all, X_te = preprocess(X_train, y_train, X_test)
    test_probas = []
    for m in models:
        n_trees = res.refit_rounds(m)
        print(f"  {m}: {n_trees} trees")
        model = MODEL_REGISTRY[m](n_estimators=n_trees).fit(X_all, y_train)
        test_probas.append(model.predict_proba(X_te))

    print(">>> 4. Predicting test set...")
    prior = np.bincount(y_train, minlength=N_CLASSES) / len(y_train)
    test_preds = blend_predict(test_probas, [weights[m] for m in models], prior, prior_power)

    desc = (f"{git_commit()} | blend {weights} prior^{prior_power} | "
            f"CV OOF {oof_f1:.4f} | " + " ".join(f"{m} {res.summary(m)[0]:.4f}" for m in models))
    save_submission(test_preds, len(X_test), output_filename, desc)


def generate_stacking_submission(output_filename: str = "submission_wip_stacking.csv"):
    X_train, y_train, _ = load_train_data()
    X_test, _ = load_test_data()
    X_train_enc, X_test_enc = preprocess(X_train, y_train, X_test)

    base_factories = [
        ("XGBoost", XGBoostModel, {}),
        ("LightGBM", LightGBMModel, {}),
        ("RandomForest", RandomForestModel, {"n_estimators": 300, "min_samples_leaf": 3}),
    ]
    stacker = OutOfFoldStacker(base_factories, n_splits=cfg.N_SPLITS, random_state=cfg.RANDOM_SEED)
    oof_preds, test_preds, _ = stacker.fit_predict(X_train_enc, y_train, X_test_enc)

    # NOTE: the meta-learner is scored on the same OOF rows it was trained on, so this number
    # is optimistic — do not compare it directly with the blend's CV score.
    print(f"\n[Stacking] in-sample meta Macro F1 (optimistic): {calculate_macro_f1(y_train, oof_preds):.4f}")
    save_submission(test_preds, len(X_test), output_filename, f"{git_commit()} | stacking XGB+LGB+RF")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", action="store_true", help="use the CV-tuned blend instead of stacking")
    args = parser.parse_args()
    if args.blend:
        generate_blend_submission()
    else:
        generate_stacking_submission()
