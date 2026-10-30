"""
load_data.py — Loads the CSV files and drops useless columns.
"""
import pandas as pd

# Columns that give zero predictive information
USELESS_COLS = ['PID', 'HHID', 'COUNTRY', 'PROXY']


def load_train(path='./ML Comp/datasets/dataset_train.csv'):
    """Load training data and separate into X (clues) and y (answer)."""
    df = pd.read_csv(path)

    # y == target answer we want to predict
    y = df['AGE_RANGE']

    # X == clues we will use to predict the answer and we drop the useless columns and target from the dataset
    X = df.drop(columns=['AGE_RANGE'] + USELESS_COLS)

    print(f"Clues (X) shape: {X.shape}")
    print(f"Answers (y) shape: {y.shape}")
    return X, y, df


def load_test(path='./ML Comp/datasets/dataset_test_X.csv'):
    """Load test data (no AGE_RANGE column — that's what we predict)."""
    df_test = pd.read_csv(path)
    X_test = df_test.drop(columns=USELESS_COLS)

    print(f"Test loaded: {X_test.shape[0]} people, {X_test.shape[1]} columns")
    return X_test, df_test
