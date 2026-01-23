"""
funciones:
 - cargar csv desde Streamlit uploadedFile u objetos simialres
 - validar requisitos minimos del dataset
 - generar info basica para la ui.
"""

from __future__ import annotations
from io import BytesIO
from typing import Dict, Any,  Optional, Tuple, List
import pandas as pd
import streamlit as st

from src.data.validator import validate_dataset
from src.utils.constants import IS_CLOUD, get_max_rows, get_max_columns


def truncate_dataset_if_needed(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Trunca el dataset si excede los límites del runtime mode.

    En modo cloud, limita filas y columnas para evitar saturar recursos.
    En modo local, no aplica ningún límite.

    Args:
        df: DataFrame a verificar/truncar.

    Returns:
        (df_truncado, mensajes) donde mensajes indica si hubo truncado.
    """
    messages: List[str] = []

    if df is None or df.empty:
        return df, messages

    if not IS_CLOUD:
        return df, messages

    max_rows = get_max_rows()
    max_cols = get_max_columns()

    original_rows = len(df)
    original_cols = len(df.columns)

    # Truncar columnas primero (para evitar procesar datos innecesarios)
    if original_cols > max_cols:
        df = df.iloc[:, :max_cols]
        messages.append(
            f"⚠️ Dataset truncado: {original_cols} → {max_cols} columnas (límite cloud)."
        )

    # Truncar filas
    if original_rows > max_rows:
        df = df.head(max_rows)
        messages.append(
            f"⚠️ Dataset truncado: {original_rows} → {max_rows} filas (límite cloud)."
        )

    return df, messages


@st.cache_data(show_spinner=False)
def _read_csv_cached(data: bytes, **read_csv_kwargs: Any) -> pd.DataFrame:
    """
    Lee un CSV desde bytes usando pandas, con cache de Streamlit.

    Args:
        data: Contenido del archivo en bytes.
        read_csv_kwargs: Parametros adicionales para pandas.read_csv().

    Returns:
        DataFrame cargado.
    """
    buffer = BytesIO(data)
    return pd.read_csv(buffer, **read_csv_kwargs)


def load_dataset(uploaded_file: Any, **read_csv_kwargs: Any) -> pd.DataFrame:
    """
    Carga un dataset CSV desde un objeto subido en Streamlit.

    Args:
        uploaded_file: st.file_uploader() o file-like compatible.
        read_csv_kwargs: Parametros extra para pandas.read_csv().

    Returns:
        DataFrame cargado.

    Raises:
        ValueError: Si el archivo es invalido o no es CSV.
    """
    if uploaded_file is None:
        raise ValueError("No se recibio ningun archivo.")

    file_name = getattr(uploaded_file, "name", "")
    if file_name and not file_name.lower().endswith(".csv"):
        raise ValueError("El archivo debe ser un CSV.")

    if hasattr(uploaded_file, "getvalue"):
        data = uploaded_file.getvalue()
    else:
        data = uploaded_file.read()

    if not data:
        raise ValueError("El archivo esta vacio.")

    try:
        return _read_csv_cached(data, **read_csv_kwargs)
    except Exception as exc:
        raise ValueError(f"No se pudo leer el CSV: {exc}") from exc


def get_basic_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Genera informacion basica del dataset para la UI.

    Incluye: filas, columnas, memoria usada y tipos detectados por pandas.

    Args:
        df: DataFrame de entrada.

    Returns:
        Diccionario con metricas basicas.
    """
    memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "memory_mb": round(memory_mb, 2),
        "dtypes": dtypes
    }


def get_high_nan_columns(df: pd.DataFrame, threshold: float = 0.3) -> Dict[str, float]:
    """
    Devuelve columnas con porcentaje de NaNs por encima del umbral (30%)

    Args:
        df: DataFrame de entrada.
        threshold: Umbral de proporcion (0-1).

    Returns:
        Dict {columna: proporcion_nan}.
    """
    if df is None or df.empty:
        return {}

    nan_ratio = df.isna().mean()
    high_nan = nan_ratio[nan_ratio > threshold]
    return high_nan.to_dict()


def get_all_nan_columns(df: pd.DataFrame) -> List[str]:
    """
    Devuelve columnas que estan completamente vacias (todo NaN).
    """
    if df is None or df.empty:
        return []

    return [col for col in df.columns if df[col].isna().all()]


def load_and_validate(uploaded_file: Any, min_rows: int = 10,
                      **read_csv_kwargs: Any) -> Tuple[Optional[pd.DataFrame], Dict[str, Any], List[str]]:
    """
    Carga un CSV y aplica validaciones basicas.

    Args:
        uploaded_file: st.file_uploader() o file-like compatible.
        min_rows: Minimo de filas requeridas.
        read_csv_kwargs: Parametros extra para pandas.read_csv().

    Returns:
        (df, info, errors) donde info puede estar vacio si hay errores.
    """
    df = load_dataset(uploaded_file, **read_csv_kwargs)
    errors = validate_dataset(df, min_rows=min_rows)
    info: Dict[str, Any] = {}
    if not errors:
        info = get_basic_info(df)
    return df, info, errors
