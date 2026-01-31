"""
UI helpers for data types detection and target selection.
"""
from __future__ import annotations

import streamlit as st

from src.data.analyzer import detect_problem_type
from src.data.validator import validate_target
from src.utils.constants import COLUMN_TYPES


def render_auto_types(auto_types: dict) -> None:
    """
    Renderiza los tipos detectados automáticamente en una tabla.

    Args:
        auto_types: Diccionario con tipos detectados por columna.
    """
    rows = [{"columna": col, "tipo_detectado": col_type}
            for col, col_type in auto_types.items()]
    st.dataframe(rows, use_container_width=True)


def render_type_selectors(df, auto_types: dict) -> dict:
    """
    Renderiza selectores para confirmar o ajustar tipos de columnas.

    Args:
        df: DataFrame con las columnas.
        auto_types: Tipos detectados automáticamente.

    Returns:
        Diccionario con tipos seleccionados por columna.
    """
    selected_types = {}

    for col in df.columns:
        default_type = auto_types.get(col, "categorical")
        default_index = COLUMN_TYPES.index(default_type)
        selected_types[col] = st.selectbox(
            f"{col}",
            options=COLUMN_TYPES,
            index=default_index,
            key=f"type_{col}",
        )

    return selected_types


def render_problem_type_preview(
    df, target_column: str, override: str | None
) -> None:
    """
    Muestra una vista previa del tipo de problema detectado y validaciones del target.

    Args:
        df: DataFrame del dataset.
        target_column: Columna target seleccionada.
    """
    if not target_column:
        return

    try:
        detected_type = detect_problem_type(df, target_column)
    except ValueError as exc:
        st.error(str(exc))
        return

    if override:
        st.info(
            f"Tipo de problema seleccionado manualmente: {override}."
        )
        st.caption(f"Autodetectado: {detected_type}")
    else:
        st.info(f"Tipo de problema detectado: {detected_type}")

    # validaciones basicas del target para mostrar alertas
    validation_messages = validate_target(
        df, target_column, problem_type=override or detected_type
    )
    for msg in validation_messages:
        st.warning(msg)


def render_problem_type_selector() -> None:
    """selectbox de tipo"""
    options = {
        "Auto (detectar)": None,
        "Clasificacion": "classification",
        "Regresion": "regression",
    }
    current = st.session_state.get("problem_type_override")
    labels = list(options.keys())
    values = list(options.values())
    default_index = values.index(current) if current in values else 0
    selected_label = st.selectbox(
        "Opciones",
        options=labels,
        index=default_index,
        help="Si eliges una opcion manual, se usara en la deteccion de tipos.",
    )
    st.session_state.problem_type_override = options[selected_label]
