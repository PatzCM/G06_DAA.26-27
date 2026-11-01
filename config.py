"""
config.py — Centralized configuration for the ML age prediction competition.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class ProjectConfig:
    # Directory paths
    BASE_DIR: Path = Path(__file__).resolve().parent
    DATASET_DIR: Path = BASE_DIR / "datasets"
    TRAIN_PATH: Path = DATASET_DIR / "dataset_train.csv"
    TEST_PATH: Path = DATASET_DIR / "dataset_test_X.csv"
    SUBMISSION_DIR: Path = DATASET_DIR
    
    # Target and metadata columns
    TARGET_COL: str = "AGE_RANGE"
    ID_COLS: List[str] = field(default_factory=lambda: ["PID", "HHID", "COUNTRY", "PROXY"])
    
    # Validation strategy
    RANDOM_SEED: int = 2026
    N_SPLITS: int = 5
    TEST_SIZE: float = 0.2
    
    # Missing / special category markers in Portuguese survey data
    NULL_MARKERS: List[str] = field(default_factory=lambda: [
        "Não aplicável",
        "Sem resposta",
        "Não sabe",
        "recusa",
        "-1"
    ])
    
    # Model defaults (Mid-tier baseline settings with room for tuning)
    XGB_LEARNING_RATE: float = 0.05
    XGB_MAX_DEPTH: int = 5
    XGB_N_ESTIMATORS: int = 1000
    XGB_SUBSAMPLE: float = 0.8
    XGB_COLSAMPLE: float = 0.7
    
    LGB_LEARNING_RATE: float = 0.05
    LGB_NUM_LEAVES: int = 31
    LGB_N_ESTIMATORS: int = 1000


# Global singleton instance
cfg = ProjectConfig()

