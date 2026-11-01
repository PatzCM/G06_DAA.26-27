"""
features/demographics.py — Demographic milestones, household composition, education, and labor status.
"""
import pandas as pd
from .base import BaseTransformer


class DemographicFeatureTransformer(BaseTransformer):
    """
    Transforms life-stage and socio-economic clues:
    - Education level progression (HATLEVEL)
    - Income quintiles (HHINCOME)
    - Employment/Retirement status (MAINSTAT)
    - Marital and Household life stage (MARSTALEGAL, HHTYPE)
    """

    HATLEVEL_MAP = {
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

    HHINCOME_MAP = {
        '1º quintil': 0,
        '2º quintil': 1,
        '3º quintil': 2,
        '4º quintil': 3,
        '5º quintil': 4,
    }

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # 1. Ordered Education
        if 'HATLEVEL' in X.columns:
            X['HATLEVEL_ORDERED'] = X['HATLEVEL'].map(self.HATLEVEL_MAP).fillna(-1)

        # 2. Ordered Income Quintile
        if 'HHINCOME' in X.columns:
            X['HHINCOME_ORDERED'] = X['HHINCOME'].map(self.HHINCOME_MAP).fillna(-1)

        # 3. Key Milestone Indicators from MAINSTAT
        if 'MAINSTAT' in X.columns:
            X['IS_RETIRED'] = (
                X['MAINSTAT'] == 'Reformado, com reforma antecipada ou cessou a atividade'
            ).astype(int)
            X['IS_STUDENT'] = (
                X['MAINSTAT'] == 'Aluno, estudante, ou em estágio curricular'
            ).astype(int)
            X['IS_DISABLED'] = (
                X['MAINSTAT'] == 'Incapacitado permanente para o trabalho'
            ).astype(int)

        # 4. Marital Signals (Widowed corresponds strongly with older brackets)
        if 'MARSTALEGAL' in X.columns:
            X['IS_WIDOWED'] = (X['MARSTALEGAL'] == 'Viúvo').astype(int)
            X['IS_SINGLE'] = (X['MARSTALEGAL'] == 'Solteiro').astype(int)

        # 5. Household composition
        if 'HHTYPE' in X.columns:
            X['HAS_ADULT_CHILDREN'] = X['HHTYPE'].isin([
                'Casal com todos os filhos >= 25 anos',
                'Pai ou mãe com todos os filhos >= 25 anos'
            ]).astype(int)

        if 'HHNBPERS_0_13_COD' in X.columns:
            X['HAS_YOUNG_CHILDREN'] = (X['HHNBPERS_0_13_COD'] != '0.0').astype(int)

        # TODO (Growth Area):
        # 1. Create composite status: IS_RETIRED * IS_WIDOWED (amplifies probability of brackets 13-15).
        # 2. Add cross-feature: IS_STUDENT * HATLEVEL_ORDERED (helps isolate college students bracket 1-2).
        # 3. Add partner correlation: PARTNERS == 'Sem relação de conjugalidade'.

        return X

