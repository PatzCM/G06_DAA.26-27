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
    
    # Questions the survey only asks from bracket 9 upwards (always "Não aplicável" for brackets 0-8).
    # They encode age by questionnaire design, not by profile: fine for the competition model, but
    # excluded from the anomaly model (otherwise it can never flag cross-boundary anomalies).
    ROUTING_COLS: List[str] = field(default_factory=lambda: [
        "PC1A", "PC1B", "PC1C", "PC1D", "PC1E", "IN15",
        "HA1A", "HA1B", "HA1C", "HA1D", "HA1E", "HA1F", "HA1G",
    ])

    # Missing / special category markers in Portuguese survey data
    NULL_MARKERS: List[str] = field(default_factory=lambda: [
        "Não aplicável",
        "Sem resposta",
        "Não sabe",
        "recusa",
        "-1"
    ])
    
    # Model defaults (Mid-tier baseline settings with room for tuning)
    # XGBoost: shallower, more regularized trees. 5-fold CV (lr=0.1) went from 0.4618
    # (depth 5, colsample 0.7, min_child_weight 3) to 0.4713 with these values.
    XGB_LEARNING_RATE: float = 0.05
    XGB_MAX_DEPTH: int = 3
    XGB_N_ESTIMATORS: int = 3000
    XGB_SUBSAMPLE: float = 0.8
    XGB_COLSAMPLE: float = 0.3
    XGB_MIN_CHILD_WEIGHT: float = 5
    XGB_REG_LAMBDA: float = 5.0
    
    # LightGBM: small, heavily regularized trees. 5-fold CV (lr=0.1) went from 0.4512
    # (num_leaves=31, colsample=0.8, min_child=20) to 0.4698 with these values — see experiments.csv.
    LGB_LEARNING_RATE: float = 0.05
    LGB_NUM_LEAVES: int = 4
    LGB_COLSAMPLE: float = 0.2
    LGB_MIN_CHILD_SAMPLES: int = 80
    LGB_LAMBDA_L2: float = 10.0
    LGB_N_ESTIMATORS: int = 3000


# Global singleton instance
cfg = ProjectConfig()

