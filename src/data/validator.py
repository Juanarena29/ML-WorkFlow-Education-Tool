"""
Validaciones del dataset y del target
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import pandas as pd


def validate_dataset(df: Optional[pd.DataFrame], min_rows: int = 10) -> List[str]:
    """
    valida requisitos minimos del dataset.
    Args:
        min_rows: es el minimo de filas por defecto 10
    Returns:
        devuelve lista de errores -> vacia si todo ok
    """
    errors: List[str] = []

    if df is None or df.empty:
        return ["El dataset está vacio o no se pudo cargar."]

    if len(df) < min_rows:
        errors.append(
            f"El dataset es muy pequeño (filas= {len(df)} - minimo= {min_rows})")

    empty_name_cols = [col for col in df.columns if str(col).strip() == ""]

    if empty_name_cols:
        errors.append(
            "Hay columnas con nombre vacio, Renombre antes de continuar.")

    return errors


def validate_target(df: pd.DataFrame, target_column: str, problem_type: Optional[str] = None,
                    max_classes: int = 50, ) -> List[str]:
    """
    Valida la columna target segun el tipo de problema.
    Args:
        df: DataFrame con el target
        target_column: Nombre de la columna target
        problem_type: 'regression' o 'classification' si ya se conoce.
        max_classes: Maximo de clases para advertencia en clasificacion.

    Returns:
        Lista de errores/advertencias.
    """
    errors: List[str] = []

    if df is None or df.empty:
        return ["Dataset invalido."]

    if target_column not in df.columns:
        return [f"Target '{target_column}' no existe en el dataset."]

    target = df[target_column]
    if target.isna().any():
        nan_count = target.isna().sum()
        errors.append(f"El target tiene {nan_count} NaNs. Debe limpiarse.")

    if problem_type == "classification":
        n_classes = target.nunique(dropna=True)
        if n_classes < 2:
            errors.append(
                f"Clasificacion requiere al menos 2 clases. Target tiene {n_classes}."
            )
        elif n_classes > max_classes:
            errors.append(
                f"Advertencia: target tiene {n_classes} clases. Revisar."
            )

    return errors


# ---------------------------------------------------------------------------
# Validacion de columnas para prediccion
# ---------------------------------------------------------------------------


def validate_prediction_columns(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_column: Optional[str] = None,
    has_target: bool = False,
) -> Tuple[List[str], Optional[pd.Series], pd.DataFrame]:
    """Valida un DataFrame de prediccion y separa target si corresponde.

    Args:
        df: DataFrame subido por el usuario.
        feature_cols: columnas features esperadas por el modelo.
        target_column: nombre de la columna target (puede ser None).
        has_target: si el usuario indico que el CSV incluye el target.

    Returns:
        (errores, y_true_o_None, df_solo_features)
    """
    errors: List[str] = []
    y_true: Optional[pd.Series] = None
    df_out = df

    if has_target:
        if not target_column:
            errors.append("No hay columna target definida en el proyecto.")
            return errors, y_true, df_out
        if target_column not in df_out.columns:
            errors.append(
                f"El CSV no contiene la columna target '{target_column}'."
            )
            return errors, y_true, df_out
        y_true = df_out[target_column]
        df_out = df_out.drop(columns=[target_column])  # returns new df

    missing = [col for col in feature_cols if col not in df_out.columns]
    if missing:
        errors.append(f"Faltan columnas requeridas: {', '.join(missing)}")

    return errors, y_true, df_out
