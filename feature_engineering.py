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

    # Ordinal fix for HS1
    hs1_order = {
        'Muito mau': 0,
        'Mau': 1,
        'Razoável': 2,
        'Bom': 3,
        'Muito bom': 4,
        'Sem resposta': -1,
    }
    X['HS1'] = X['HS1'].map(hs1_order)

    # Ordinal fix for HATLEVEL
    education_order = {
        'Nenhum nível completado': 0,
        'Básico 1º e 2º ciclo': 1,
        'Básico 3º ciclo': 2,
        'Secundário': 3,
        'Pós secundário': 4,
        'C.T.Superior Profissional': 5,
        'Bacharelato ou licenciatura 1º ciclo': 6,
        'Licenciatura pré-bolonha ou mestrado 2º ciclo': 7,
        'Doutoramento': 8,
    }
    X['HATLEVEL'] = X['HATLEVEL'].map(education_order)

    # Ordinal fix for HHINCOME
    income_order = {
        '1º quintil': 0,
        '2º quintil': 1,
        '3º quintil': 2,
        '4º quintil': 3,
        '5º quintil': 4,
    }
    X['HHINCOME'] = X['HHINCOME'].map(income_order)

    #NOTE: Adult children counter
    X['HAS_ADULT_CHILDREN'] = X['HHTYPE'].isin([
        'Casal com todos os filhos >= 25 anos',
        'Pai ou mãe com todos os filhos >= 25 anos'
    ]).astype(int)

    X['HAS_YOUNG_CHILDREN'] = (X['HHNBPERS_0_13_COD'] != '0.0').astype(int)

    # Milestone indicators from MAINSTAT
    X['IS_RETIRED'] = (X['MAINSTAT'] == 'Reformado, com reforma antecipada ou cessou a atividade').astype(int)
    X['IS_STUDENT'] = (X['MAINSTAT'] == 'Aluno, estudante, ou em estágio curricular').astype(int)
    X['IS_DISABLED'] = (X['MAINSTAT'] == 'Incapacitado permanente para o trabalho').astype(int)
    # Drop the original columns that we don't need anymore
    X = X.drop(columns=['BM1', 'BM2'])
    return X

