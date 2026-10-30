"""
models.py — Train and evaluate ML models.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from xgboost import XGBClassifier


def train_random_forest(X_train, y_train, X_value, y_value):
    """Train a Random Forest model and print its Macro F1 Score."""
    print("Training a Random Forest model...")

    rf = RandomForestClassifier(n_estimators=100, random_state=2026, n_jobs=-1)

    rf.fit(X_train, y_train)
    y_predict = rf.predict(X_value)
    score = f1_score(y_value, y_predict, average='macro')
    print(f"Random Forest F1 Score: {score:.4f}")

    return rf, score


def train_xgboost(X_train, y_train, X_value, y_value):
    """Train an XGBoost model and print its Macro F1 Score."""
    print("Training an XGBoost model...")

    xgb_model = XGBClassifier(
        n_estimators=180,
        learning_rate=0.08,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=2026,
        n_jobs=-1,
        eval_metric='mlogloss'
    )

    xgb_model.fit(X_train, y_train)
    y_predict_xgb = xgb_model.predict(X_value)
    score_xgb = f1_score(y_value, y_predict_xgb, average='macro')
    print(f"XGBoost F1 Score: {score_xgb:.4f}")

    return xgb_model, score_xgb


def plot_feature_importances(model, feature_names, save_path='./ML Comp/datasets/feature_importance.png'):
    """
    Extract the feature importances score to understand which features are most
    important for the model's predictions. This can help us identify which clues
    are most relevant for predicting the target variable.
    """
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top15 = importances.nlargest(15)
    print("Top 15 most important features:")
    print(top15)

    # Bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top15, y=top15.index)
    plt.title('Top 15 Most Important Features')
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    return importances

