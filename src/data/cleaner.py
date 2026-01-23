"""
Dataset cleaning utilities
Funciones de imputacion y aplicacion de configuraciones de limpieza.
"""
from __future__ import annotations
from typing import Any, Dict, List
import pandas as pd
from pandas.api.types import is_numeric_dtype


def impute_numeric(series: pd.Series, method: str, value: Any = None) -> pd.Series:
    """
    Imputa una serie numerica segun el metodo indicado.

    Args:
        series: Serie numerica a imputar.
        method: 'median', 'mean', 'constant', 'interpolate', 'drop_rows'.
        value: Valor constante si method == 'constant'.

    Returns:
        Serie con imputacion aplicada.
    """
    if method == "median":
        return series.fillna(series.median())
    if method == "mean":
        return series.fillna(series.mean())
    if method == "constant":
        return series.fillna(value)
    if method == "interpolate":
        return series.interpolate(limit_direction="both")
    if method == "drop_rows":
        return series
    return series


def impute_categorical(series: pd.Series, method: str, value: Any = None) -> pd.Series:
    """
    Imputa una serie categorica segun el metodo indicado.

    Args:
        series: Serie categorica a imputar.
        method: 'mode', 'unknown', 'constant', 'drop_rows'.
        value: Valor constante si method == 'constant'.

    Returns:
        Serie con imputacion aplicada.
    """
    if method == "mode":
        if series.dropna().empty:
            return series.fillna("Desconocido")
        return series.fillna(series.mode(dropna=True)[0])
    if method == "unknown":
        return series.fillna("Desconocido")
    if method == "constant":
        return series.fillna(value)
    if method == "drop_rows":
        return series
    return series


def list_columns_with_nans(df: pd.DataFrame) -> List[str]:
    """
    Devuelve una lista de columnas que contienen al menos un NaN.
    """
    if df is None or df.empty:
        return []
    return [col for col in df.columns if df[col].isna().any()]


def suggest_imputation_method(series: pd.Series, col_type: str) -> str:
    """
    Sugiere un metodo de imputacion basico segun el tipo de columna.
    """
    if col_type == "numeric":
        # sesgo de la distribucion para sugerir mediana o media
        skew = series.dropna().skew()
        if abs(skew) >= 1:
            return "median"
        return "mean"

    return "mode"


def apply_cleaning_config(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Aplica la configuracion de limpieza a un DataFrame.

    Args:
        df: DataFrame original.
        config: Diccionario con configuracion de limpieza.

    Returns:
        DataFrame limpio.
    """
    if df is None or df.empty:
        return df

    cleaned = df.copy()

    imputations = config.get("imputations", {})
    drop_duplicates = config.get("drop_duplicates", False)

    for col, settings in imputations.items():
        method = settings.get("method")
        value = settings.get("value")

        if col not in cleaned.columns:
            continue

        if method == "drop_rows":
            cleaned = cleaned[cleaned[col].notna()]
            continue

        if is_numeric_dtype(cleaned[col]):
            cleaned[col] = impute_numeric(cleaned[col], method, value)
        else:
            cleaned[col] = impute_categorical(cleaned[col], method, value)

    if drop_duplicates:
        cleaned = cleaned.drop_duplicates()

    return cleaned
