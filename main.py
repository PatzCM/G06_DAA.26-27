"""
main.py — The boss script. Runs everything in order.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from load_data import load_train, load_test
from feature_engineering import engineer_features
from encoding import encode_text, encode_test
from models import train_random_forest, train_xgboost, plot_feature_importances

# ============================================================
# 1: Load data
# ============================================================
X, y, df = load_train()

# ============================================================
# 1.5: Feature Engineering 
# ============================================================
X = engineer_features(X, df)

# ============================================================
# 2: Turn portuguese text into numbers
# ============================================================
X, encoder, text_cols = encode_text(X)

# TODO: Ensure that we don't cause Overfitting by splitting the data into
# training and testing sets before encoding. This is important because if we
# encode the entire dataset before splitting, we might introduce information
# from the test set into the training set, leading to overfitting.

# ============================================================
# 3: Split the data into training and testing sets
# ============================================================
X_train, X_value, y_train, y_value = train_test_split(
    X, y, test_size=0.2, random_state=2026, stratify=y
)
# NOTE: Use stratified sampling to ensure that the distribution of the target
# variable is similar in both the training and testing sets.

print(f"Study set shape: {X_train.shape[0]} people")
print(f"Validation set shape: {X_value.shape[0]} people")

# ============================================================
# 4: Train models
# ============================================================
rf, rf_score = train_random_forest(X_train, y_train, X_value, y_value)
xgb_model, xgb_score = train_xgboost(X_train, y_train, X_value, y_value)

# Pick the best model
if xgb_score > rf_score:
    best_model = xgb_model
    print(f"\n>>> Best model: XGBoost ({xgb_score:.4f})")
else:
    best_model = rf
    print(f"\n>>> Best model: Random Forest ({rf_score:.4f})")

# ============================================================
# 5: Feature importances
# ============================================================
# NOTE: Commented out because it is no longer needed.
# plot_feature_importances(rf, X.columns)

# ============================================================
# 6: Predict on test set and create Kaggle submission
# ============================================================
# NOTE: Here we actually start with the prediction of the test set and create
# the Kaggle submission file. We will use the best model we have trained so far.

print("\nLoading test dataset...")
X_test, df_test = load_test()
X_test = engineer_features(X_test, df_test)
X_test = encode_test(X_test, encoder, text_cols)

print("\nPredicting test dataset...")
test_predictions = best_model.predict(X_test)

submission = pd.DataFrame({
    'RowId': df_test.index,
    'AGE_RANGE': test_predictions
})

submission_file = './ML Comp/datasets/submission.csv'
submission.to_csv(submission_file, index=False)

print(f"\nSubmission file created successfully at {submission_file}!")
print(f"Row count: {len(submission)} (should match test dataset row count: {len(df_test)})")
print(submission.head())

