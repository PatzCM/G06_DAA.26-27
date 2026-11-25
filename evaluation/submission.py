"""
evaluation/submission.py — Sanity checks before uploading to Kaggle (we already lost submissions
to format errors such as "ID column RowId not found").
"""
import numpy as np
import pandas as pd

EXPECTED_COLUMNS = ["RowId", "AGE_RANGE"]


def validate_submission(sub: pd.DataFrame, n_test_rows: int, n_classes: int = 16) -> None:
    """Raise ValueError with every problem found; return silently if the file is valid."""
    problems = []
    if list(sub.columns) != EXPECTED_COLUMNS:
        problems.append(f"columns are {list(sub.columns)}, expected {EXPECTED_COLUMNS}")
    else:
        if len(sub) != n_test_rows:
            problems.append(f"{len(sub)} rows, expected {n_test_rows}")
        if sub.isna().any().any():
            problems.append("contains NaN values")
        if not np.array_equal(sub["RowId"].to_numpy(), np.arange(len(sub))):
            problems.append("RowId is not the sequence 0..n-1")
        if not pd.api.types.is_integer_dtype(sub["AGE_RANGE"]):
            problems.append(f"AGE_RANGE dtype is {sub['AGE_RANGE'].dtype}, expected integers")
        elif not sub["AGE_RANGE"].between(0, n_classes - 1).all():
            problems.append(f"AGE_RANGE has values outside 0..{n_classes - 1}")

    if problems:
        raise ValueError("Invalid submission:\n  - " + "\n  - ".join(problems))
    print(f"[Submission] Format OK: {len(sub)} rows, columns {EXPECTED_COLUMNS}, "
          f"{sub['AGE_RANGE'].nunique()} distinct brackets predicted.")
