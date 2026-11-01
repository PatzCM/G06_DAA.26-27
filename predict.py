"""
predict.py — End-to-end full training + Stacking inference to output Kaggle submission.
"""
from pathlib import Path
import pandas as pd
from config import cfg
from data.loader import load_train_data, load_test_data
from features.pipeline import FeaturePipeline
from encoders.ordinal import CategoricalOrdinalEncoder
from models.xgboost_model import XGBoostModel
from models.lightgbm_model import LightGBMModel
from models.tree_baselines import RandomForestModel
from ensemble.stacker import OutOfFoldStacker
from evaluation.metrics import calculate_macro_f1


def generate_stacking_submission(output_filename: str = "submission_wip_stacking.csv"):
    print(">>> 1. Loading Train and Test Data...")
    X_train, y_train, _ = load_train_data()
    X_test, df_test = load_test_data()

    print(">>> 2. Applying Feature Engineering Pipeline...")
    feat_pipe = FeaturePipeline()
    X_train_fe = feat_pipe.fit_transform(X_train, y_train)
    X_test_fe = feat_pipe.transform(X_test)

    print(">>> 3. Applying Ordinal Encoding...")
    encoder = CategoricalOrdinalEncoder()
    X_train_enc = encoder.fit_transform(X_train_fe)
    X_test_enc = encoder.transform(X_test_fe)

    print(">>> 4. Initializing Base Models for Stacking...")
    base_factories = [
        ("XGBoost", XGBoostModel, {}),
        ("LightGBM", LightGBMModel, {}),
        ("RandomForest", RandomForestModel, {"n_estimators": 300, "min_samples_leaf": 3}),
    ]

    stacker = OutOfFoldStacker(base_factories, n_splits=5, random_state=cfg.RANDOM_SEED)

    print(">>> 5. Executing Stacking Fit & Predict...")
    oof_preds, test_preds, _ = stacker.fit_predict(X_train_enc, y_train, X_test_enc)

    oof_f1 = calculate_macro_f1(y_train, oof_preds)
    print(f"\n[Stacking Result] Cross-Validated Training OOF Macro F1: {oof_f1:.4f}")

    print(">>> 6. Formatting Submission...")
    sub = pd.DataFrame({
        "RowId": df_test.index,
        "AGE_RANGE": test_preds
    })

    out_path = cfg.SUBMISSION_DIR / output_filename
    sub.to_csv(out_path, index=False)
    print(f"[Done] Submission file successfully saved to: {out_path}")
    print(f"Row count: {len(sub)}")


if __name__ == "__main__":
    generate_stacking_submission()

