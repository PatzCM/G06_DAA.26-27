"""
data/loader.py — Robust dataset loading with shape verification.
"""
from typing import Tuple, List, Optional
import pandas as pd
from config import cfg


def load_train_data(path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Loads raw training dataset and separates clues (X) and target (y).
    
    Returns:
        X: Feature matrix excluding target and ID columns.
        y: Target series (AGE_RANGE).
        raw_df: Original full dataframe (useful for row indices or raw metadata).
    """
    csv_path = path or cfg.TRAIN_PATH
    raw_df = pd.read_csv(csv_path)
    
    if cfg.TARGET_COL not in raw_df.columns:
        raise KeyError(f"Target column '{cfg.TARGET_COL}' not found in {csv_path}")
        
    y = raw_df[cfg.TARGET_COL].astype(int)
    
    drop_cols = [cfg.TARGET_COL] + [c for c in cfg.ID_COLS if c in raw_df.columns]
    X = raw_df.drop(columns=drop_cols)
    
    print(f"[DataLoader] Train loaded: {X.shape[0]} rows, {X.shape[1]} raw feature columns.")
    return X, y, raw_df


def load_test_data(path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads raw test dataset without target column.
    
    Returns:
        X_test: Feature matrix excluding ID columns.
        raw_test_df: Original full test dataframe (retaining RowId index for submission).
    """
    csv_path = path or cfg.TEST_PATH
    raw_test_df = pd.read_csv(csv_path)
    
    drop_cols = [c for c in cfg.ID_COLS if c in raw_test_df.columns]
    X_test = raw_test_df.drop(columns=drop_cols)
    
    print(f"[DataLoader] Test loaded: {X_test.shape[0]} rows, {X_test.shape[1]} raw feature columns.")
    return X_test, raw_test_df

