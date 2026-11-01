"""
runner.py — Main orchestrator to train, cross-validate, and evaluate the ML pipeline.
"""
from sklearn.model_selection import train_test_split
from config import cfg
from data.loader import load_train_data
from features.pipeline import FeaturePipeline
from encoders.ordinal import CategoricalOrdinalEncoder
from models.xgboost_model import XGBoostModel
from models.lightgbm_model import LightGBMModel
from ensemble.blender import ProbabilityBlender
from evaluation.metrics import print_classification_summary
from evaluation.diagnostics import analyze_bracket_confusion


def run_training_pipeline():
    print(">>> 1. Loading training dataset...")
    X, y, _ = load_train_data()

    print(">>> 2. Splitting into Train and Holdout Validation sets (80/20 stratified)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_SEED, stratify=y
    )

    print(">>> 3. Fitting FeaturePipeline...")
    feat_pipe = FeaturePipeline()
    X_train_fe = feat_pipe.fit_transform(X_train, y_train)
    X_val_fe = feat_pipe.transform(X_val)

    print(">>> 4. Encoding categorical columns...")
    encoder = CategoricalOrdinalEncoder()
    X_train_enc = encoder.fit_transform(X_train_fe)
    X_val_enc = encoder.transform(X_val_fe)

    print(">>> 5. Training XGBoost Model...")
    xgb = XGBoostModel()
    xgb.fit(X_train_enc, y_train, eval_set=(X_val_enc, y_val))
    p_xgb = xgb.predict_proba(X_val_enc)

    print(">>> 6. Training LightGBM Model...")
    lgb = LightGBMModel()
    lgb.fit(X_train_enc, y_train, eval_set=(X_val_enc, y_val))
    p_lgb = lgb.predict_proba(X_val_enc)

    print(">>> 7. Blending Predictions (60% XGB + 40% LGB)...")
    blender = ProbabilityBlender(weights=[0.6, 0.4])
    y_pred = blender.predict([p_xgb, p_lgb])

    print(">>> 8. Evaluating Performance...")
    print_classification_summary(y_val, y_pred)
    analyze_bracket_confusion(y_val, y_pred)


if __name__ == "__main__":
    run_training_pipeline()

