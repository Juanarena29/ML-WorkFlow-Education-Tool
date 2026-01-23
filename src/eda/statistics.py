"""
Funciones de estadistica para el EDA.
"""

from __future__ import annotations

import pandas as pd


def get_correlation_matrix(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """
    Calcula la matriz de correlacion para columnas numericas.
    """
    if df is None or df.empty or len(numeric_cols) < 2:
        return pd.DataFrame()

    return df[numeric_cols].corr()
