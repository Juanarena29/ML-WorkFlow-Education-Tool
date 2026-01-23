"""
Modulo de funciones para detectar automaticamente
tipo de columnas y entender el dataset.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype


def detect_column_types(df: pd.DataFrame, sample_size: int = 5000, datetime_threshold: float = 0.8,
                        id_unique_threshold: float = 0.95) -> Dict[str, str]:
    """
    Detecta el tipo de cada columna de un dataFrame

    Tipos posibles:
    - numeric
    - categorical
    - datetime
    - id
    - text

    Args:
        df: DataFrame de entrada.
        sample_size: Maximo de filas a muestrear por columna.
        datetime_threshold: Proporcion minima para detectar datetime.
        id_unique_threshold: Proporcion minima para detectar IDs.

    Returns:
        Diccionario {nombre columna: tipo_detectado}.
    """

    if df is None or df.empty:
        return {}

    result: Dict[str, str] = {}
    n_rows = len(df)

    for col in df.columns:
        series = df[col]
        sample = _get_sample(series, sample_size)
        non_null = sample.dropna()

        # por defecto, si col vacia -> categorica (despues se recomienda borrarla)
        if non_null.empty:
            result[col] = "categorical"
            continue
        if _looks_like_id(col, series, id_unique_threshold, n_rows):
            result[col] = "id"
            continue
        if is_datetime64_any_dtype(series):
            result[col] = "datetime"
            continue
        if is_numeric_dtype(series):
            result[col] = "numeric"
            continue
        if _looks_like_datetime(non_null, datetime_threshold):
            result[col] = "datetime"
            continue
        if _looks_like_text(non_null):
            result[col] = "text"
        else:
            result[col] = "categorical"

    return result


def detect_problem_type(
    df: pd.DataFrame,
    target_column: str,
    classification_max_classes: int = 50,
    regression_min_unique_ratio: float = 0.05,
) -> str:
    """
    Detecta si el problema es de regresion o clasificacion.

    Reglas:
    - Si el target es numerico y tiene muchos valores unicos, se asume regresion.
    - Si el target es categorico o tiene pocas clases, se asume clasificacion.

    Args:
        df: DataFrame con el target.
        target_column: Nombre de la columna target.
        classification_max_classes: Maximo de clases para tratar como clasificacion.

    Returns:
        'regression' o 'classification'.

    Raises:
        ValueError: Si la columna target no existe o es invalida.
    """
    if df is None or df.empty:
        raise ValueError("Dataset vacio o invalido.")

    if target_column not in df.columns:
        raise ValueError(f"Target '{target_column}' no existe en el dataset.")

    target = df[target_column].dropna()
    if target.empty:
        raise ValueError("La columna target no tiene valores validos.")

    # cantidad de clases distintas en el target
    n_unique = target.nunique()

    if is_numeric_dtype(target):
        # ratio de valores unicos sobre el total del target
        unique_ratio = n_unique / max(len(target), 1)
        if n_unique > classification_max_classes:
            return "regression"
        if unique_ratio >= regression_min_unique_ratio and n_unique >= 10:
            return "regression"
        return "classification"

    if n_unique <= classification_max_classes:
        return "classification"

    return "classification"


def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analiza valores faltantes por columna.

    Devuelve un DataFrame con:
    - columna
    - total_nans
    - porcentaje_nans

    Args:
        df: DataFrame de entrada

    Returns:
        DataFrame con cantidad de nans por columna
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["columna", "total_nans", "porcentaje_nans"])

    total_rows = len(df)
    nan_counts = df.isna().sum()

    # porcentaje de NaNs respecto al total de filas
    nan_percentages = (nan_counts / max(total_rows, 1)) * 100

    summary = pd.DataFrame({
        "columna": nan_counts.index,
        "total_nans": nan_counts.values,
        "porcentaje_nans": nan_percentages.values,
    })

    summary = summary.sort_values(
        "porcentaje_nans", ascending=False).reset_index(drop=True)
    return summary


def _get_sample(series: pd.Series, sample_size: int) -> pd.Series:
    """
    Devuelve una muestra aleatoria de la serie para analisis rapido

    Args:
        series: Serie original.
        sample_size: Cantidad maxima de filas en la muestra.

    Returns:
        Serie muestreada.
    """
    if sample_size <= 0 or len(series) <= sample_size:
        return series
    return series.sample(sample_size, random_state=42)


def _looks_like_id(
    col_name: str,
    series: pd.Series,
    id_unique_threshold: float,
    n_rows: int,
) -> bool:
    """
    Heuristica para detectar columnas ID.

    Considera nombre de columna y proporcion de valores unicos.
    """
    lowered = str(col_name).strip().lower()
    if lowered in {"id"} or lowered.endswith("_id") or lowered.endswith("id"):
        # porcentaje de valores unicos sobre el total
        unique_ratio = series.nunique(dropna=True) / max(n_rows, 1)
        return unique_ratio >= 0.5

    if not is_numeric_dtype(series):
        return False

    # porcentaje de valores unicos sobre el total
    unique_ratio = series.nunique(dropna=True) / max(n_rows, 1)
    if unique_ratio < id_unique_threshold:
        return False

    return True


def _looks_like_datetime(series: pd.Series, threshold: float) -> bool:
    """
    Detecta si una serie de strings puede interpretarse como datetime.
    """
    parsed = pd.to_datetime(series, errors="coerce")
    non_null_ratio = parsed.notna().mean()
    return non_null_ratio >= threshold


def _looks_like_text(series: pd.Series) -> bool:
    """
    Heuristica simple para diferenciar texto libre de categorias cortas.
    """
    str_values = series.astype(str)
    # longitud promedio de las cadenas
    avg_len = str_values.str.len().mean()
    # porcentaje de valores unicos sobre el total
    unique_ratio = series.nunique(dropna=True) / max(len(series), 1)

    if avg_len >= 30:
        return True

    return avg_len >= 15 and unique_ratio >= 0.7
