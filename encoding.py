"""
encoding.py — Turn portuguese text into numbers.
"""
from sklearn.preprocessing import OrdinalEncoder


def encode_text(X):
    """
    Encode all text columns using OrdinalEncoder.
    Returns: encoded X, the encoder object, and the list of text column names.
    """
    # We will use OrdinalEncoder to convert the categorical text data into numerical data
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    text_cols = X.select_dtypes(include=['object']).columns

    X = X.copy()
    X[text_cols] = encoder.fit_transform(X[text_cols])

    # IN16 has NaN values so we will fill them with -1 because of values in the OrdinalEncoder.
    # This is a common practice to handle missing values in categorical data.
    X = X.fillna(-1)

    print("All words converted successfully into numbers!")
    print(f"Any NaN values left? {X.isnull().sum().sum()}")

    return X, encoder, text_cols


def encode_test(X_test, encoder, text_cols):
    """
    Encode the test set using the SAME encoder from training.
    Uses .transform() (NOT .fit_transform()) to avoid data leakage.
    """
    X_test = X_test.copy()
    X_test[text_cols] = encoder.transform(X_test[text_cols])
    X_test = X_test.fillna(-1)
    return X_test

