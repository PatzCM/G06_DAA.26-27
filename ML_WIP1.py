import sklearn as skl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedShuffleSplit
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, BaggingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

from sklearn.metrics import classification_report

df = pd.read_csv('./ML Comp/datasets/dataset_train.csv')

df.head()

df.info()

sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()

# --- 1: Separate Clues (X) and Answers (y) ---
# y == target answer we want to predict

y = df['AGE_RANGE']

# X == clues we will use to predict the answer and we drop the useless columns and target from the dataset
X = df.drop(['AGE_RANGE',
                        'PID',
                        'HHID',
                        'COUNTRY',
                        'PROXY'],
                        axis=1)


print(f"Clues (X) shape: {X.shape}")
print(f"Answers (y) shape: {y.shape}")

# --- 2: Turn portuguese text into numbers ---
# We will use OrdinalEncoder to convert the categorical text data into numerical data
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
text = X.select_dtypes(include=['object']).columns
X[text] = encoder.fit_transform(X[text])

X = X.fillna(-1) 
 # IN16 has NaN values so we will fill them with -1 because of values in the OrdinalEncoder. This is a common practice to handle missing values in categorical data.


print("All words converted successfully into numbers!")
print(f"Any NaN values left? {X.isnull().sum().sum()}")

# TODO: Ensure that we don't cause Overfitting by splitting the data into training and testing sets before encoding. This is important because if we encode the entire dataset before splitting, we might introduce information from the test set into the training set, leading to overfitting.

# --- 3: Split the data into training and testing sets ---
X_train, X_value, y_train, y_value = train_test_split(X, y, test_size=0.2, random_state=2026, stratify=y)
# NOTE: Use stratified sampling to ensure that the distribution of the target variable is similar in both the training and testing sets. This is important because if the distribution of the target variable is different in the training and testing sets, it can lead to biased results and poor model performance.

print(f"Study set shape: {X_train.shape[0]} people")
print(f"Validation set shape: {X_value.shape[0]} people")

# --- 4: Train a Random Forest Model ---
print("Training a Random Forest model...")

rf = RandomForestClassifier(n_estimators = 100, random_state=2026, n_jobs=-1)

rf.fit(X_train, y_train)
y_predict = rf.predict(X_value)
score = f1_score(y_value, y_predict, average='macro')
print(f"Random Forest F1 Score: {score:.4f}")


# --- 5: XGBoost Model ---
print("Training an XGBoost model...")
xgb_model = XGBClassifier(
    n_estimators=100,
    random_state=2026,
    learning_rate=0.1,
    n_jobs=-1,
    eval_metric='mlogloss'
)

xgb_model.fit(X_train, y_train)
y_predict_xgb = xgb_model.predict(X_value)
score_xgb = f1_score(y_value, y_predict_xgb, average='macro')
print(f"XGBoost F1 Score: {score_xgb * 100:.4f}")




# XXX Extract the feature importances score to understand which features are most important for the model's predictions. This can help us identify which clues are most relevant for predicting the target variable.
importances = pd.Series(rf.feature_importances_, index=X.columns)
top15 = importances.nlargest(15)
print("Top 15 most important features:")
print(top15)

# Bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x=top15, y=top15.index)
plt.title('Top 15 Most Important Features')
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
pallete='Blues_d'
plt.tight_layout()
plt.savefig('./ML Comp/datasets/feature_importance.png')
plt.show()

# --- NOTE: Here we actually start with the prediction of the test set and create the Kaggle submission file. We will use the best model we have trained so far to predict the test set and create a submission file for Kaggle. ---

print("\nLoading test dataset...")
df_test = pd.read_csv('./ML Comp/datasets/dataset_test_X.csv')

X_test = df_test.drop(['PID', 'HHID', 'COUNTRY', 'PROXY'], axis=1)
X_test[text] = encoder.transform(X_test[text])
X_test = X_test.fillna(-1)
print("\nPredicting test dataset...")
test_predictions = rf.predict(X_test)

submission = pd.DataFrame({
    'RowId': df_test.index,
    'AGE_RANGE': test_predictions
})

submission_file = './ML Comp/datasets/submission.csv'
submission.to_csv(submission_file, index=False)

print(f"\nSubmission file created successfully at {submission_file}!")
print(f"Row count: {len(submission)} (should match test dataset row count: {len(df_test)})")
print(submission.head())