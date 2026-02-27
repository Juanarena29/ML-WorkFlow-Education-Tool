"""
Funciones de estadistica para el EDA.
"""

from __future__ import annotations

from typing import List

import pandas as pd


def get_correlation_matrix(
    df: pd.DataFrame,
    numeric_cols: List[str],
) -> pd.DataFrame:
    """
    Calcula la matriz de correlacion para columnas numericas.

    Args:
        df: DataFrame con los datos.
        numeric_cols: Lista de nombres de columnas numericas.

    Returns:
        DataFrame con la matriz de correlacion, o vacio si no hay datos suficientes.
    """
    if df is None or df.empty or len(numeric_cols) < 2:
        return pd.DataFrame()

    valid_cols = [c for c in numeric_cols if c in df.columns]
    if len(valid_cols) < 2:
        return pd.DataFrame()

    return df[valid_cols].corr()
