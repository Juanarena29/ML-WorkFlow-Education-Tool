"""
UI helpers for dataset info and warnings.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd

from src.data.loader import get_high_nan_columns

from src.ui.learn_explanations import (
    render_learn_one_info_explanation,
    render_learn_one_high_nan_warning_explanation,
    render_learn_one_dtypes_explanation
)


def render_basic_info(info: dict) -> None:
    """Renderiza las métricas básicas del dataset."""
    cols = st.columns(3)
    cols[0].metric("Filas", info["rows"])
    cols[1].metric("Columnas", info["columns"])
    cols[2].metric("Memoria (MB)", f"{info['memory_mb']:.2f}")


def render_dtypes_table(dtypes: dict) -> None:
    """Renderiza una tabla con nombre de columnas y tipos detectados."""
    dtype_rows = [{"Columna": col, "Tipo": dtype}
                  for col, dtype in dtypes.items()]
    st.dataframe(dtype_rows, use_container_width=True)


def render_high_nan_warning(df: pd.DataFrame, threshold: float = 0.3) -> bool:
    """
    Muestra un warning si existen columnas con alto % de NaNs.
    Devuelve True si se mostró el warning, False si no.
    """
    high_nan = get_high_nan_columns(df, threshold=threshold)
    if not high_nan:
        return False

    cols = ", ".join([f"{col} ({ratio:.0%})" for col,
                     ratio in high_nan.items()])
    st.warning(f"Columnas con más de {int(threshold * 100)}% NaNs: {cols}.")
    return True


def render_dataset_preview(df: pd.DataFrame, info: dict, learn: bool) -> None:
    """preview del dataset cargado en el momento"""
    render_basic_info(info)

    if learn:
        render_learn_one_info_explanation()

    has_high_nan = render_high_nan_warning(df, threshold=0.3)
    if learn and has_high_nan:
        render_learn_one_high_nan_warning_explanation()

    st.subheader("Tipos de datos detectados (puedes modificarlos luego)")
    render_dtypes_table(info["dtypes"])
    if learn:
        render_learn_one_dtypes_explanation()

    st.subheader("Vista previa del dataset")
    st.write(
        "Una vista previa del dataset ayuda a comprender su estructura y el tipo de datos disponibles."
    )
    st.dataframe(df.head(50), use_container_width=True)
