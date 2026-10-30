"""
feature_engineering.py — Create new smart columns from raw survey data.
Add your feature engineering functions here.
"""
import pandas as pd


def engineer_features(X, df_original):
    """
    Takes X (the clues DataFrame) and df_original (the full raw DataFrame)
    and returns X with new engineered columns added.
    
    Add your feature engineering code inside this function.
    """
    X = X.copy()

    # TODO: Add your feature engineering here
    # Example ideas discussed:
    # - Height & Weight as real numbers + BMI
    # - Chronic disease counter (CD1A to CD1P)
    # - Household children clue (HAS_ADULT_CHILDREN)
    # - Young kids in household (HAS_YOUNG_KIDS)
    
    # NOTE: Height and Weight are stored in BM1 and BM2, respectively, as strings.
    #   Some people answered "Sem resposta" so we force them to NaN with errors='coerce'.
    X['HEIGHT'] = pd.to_numeric(X['BM1'], errors='coerce')
    X['WEIGHT'] = pd.to_numeric(X['BM2'], errors='coerce')
    X['BMI'] = X['WEIGHT'] / (X['HEIGHT'] / 100) ** 2

    # NOTE: CHronic disease counter (CD1A to CD1P) We sum the number of chronic diseases each person has.

    chronic_disease_columns = [c for c in X.columns if c.startswith('CD1') and len(c) == 4]
    X['CHRONIC_DISEASE_COUNT'] = (X[chronic_disease_columns] == 'Sim').sum(axis=1) 


    #NOTE: Adult children counter
    X['HAS_ADULT_CHILDREN'] = X['HHTYPE'].isin([
        'Casal com todos os filhos >= 25 anos',
        'Pai ou mãe com todos os filhos >= 25 anos'
    ]).astype(int)

    X['HAS_YOUNG_CHILDREN'] = (X['HHNBPERS_0_13_COD'] != '0.0').astype(int)

    # Drop the original columns that we don't need anymore
    X = X.drop(columns=['BM1', 'BM2'])
    return X

