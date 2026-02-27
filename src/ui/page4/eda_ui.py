"""
Componentes de UI para la pagina 4 - EDA.

Cada funcion renderiza una seccion concreta de la pagina,
recibiendo los datos como parametros (sin acceder al estado global).
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
import streamlit as st

from src.eda.visualizations import (
    correlation_fig,
    distribution_categorical_fig,
    distribution_numeric_fig,
    relations_scatter_fig,
    target_distribution_fig,
    target_relation_fig,
)
from src.ui.learn_explanations import (
    render_learn_four_correlation_explanation,
    render_learn_four_distribution_explanations,
    render_learn_four_invert_explanation,
    render_learn_four_relations_explanation,
    render_learn_four_target_explanation,
)


# ---------------------------------------------------------------------------
# Utilidades internas de UI
# ---------------------------------------------------------------------------

def _ensure_selectbox_value(key: str, options: List[str]) -> None:
    """Limpia del session_state una key de selectbox cuyo valor
    ya no pertenece a las opciones disponibles."""
    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]


# ---------------------------------------------------------------------------
# Tab: Distribuciones
# ---------------------------------------------------------------------------

def render_distributions_tab(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    learn: bool,
) -> None:
    """Renderiza el tab de distribuciones numéricas y categóricas."""
    st.subheader("Distribuciones")

    if learn:
        render_learn_four_distribution_explanations()

    if numeric_cols:
        _ensure_selectbox_value("eda_num_col", numeric_cols)
        num_col = st.selectbox(
            "Selecciona una columna numerica",
            options=numeric_cols,
            key="eda_num_col",
        )
        fig = distribution_numeric_fig(df, num_col)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay columnas numericas para mostrar histogramas.")

    if categorical_cols:
        _ensure_selectbox_value("eda_cat_col", categorical_cols)
        cat_col = st.selectbox(
            "Selecciona una columna categorica",
            options=categorical_cols,
            key="eda_cat_col",
        )
        fig = distribution_categorical_fig(df, cat_col)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay columnas categoricas para mostrar barplots.")


# ---------------------------------------------------------------------------
# Tab: Correlaciones
# ---------------------------------------------------------------------------

def render_correlations_tab(
    df: pd.DataFrame,
    numeric_cols: List[str],
    learn: bool,
) -> None:
    """Renderiza el tab de matriz de correlación."""
    st.subheader("Correlaciones")

    if learn and len(numeric_cols) >= 2:
        render_learn_four_correlation_explanation()

    if len(numeric_cols) < 2:
        st.info("Se necesitan al menos 2 columnas numericas para correlaciones.")
        return

    fig = correlation_fig(df, numeric_cols)
    if fig is None:
        st.info("No hay suficientes datos numericos para correlacion.")
    else:
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab: Relaciones
# ---------------------------------------------------------------------------

def render_relations_tab(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    learn: bool,
) -> None:
    """Renderiza el tab de scatter plots entre variables."""
    st.subheader("Relaciones")

    if learn and len(numeric_cols) >= 2:
        render_learn_four_relations_explanation()

    if len(numeric_cols) < 2:
        st.info("Se necesitan al menos 2 columnas numericas para scatter plots.")
        return

    _ensure_selectbox_value("eda_x_col", numeric_cols)
    _ensure_selectbox_value("eda_y_col", numeric_cols)
    x_col = st.selectbox("Eje X", options=numeric_cols, key="eda_x_col")
    y_col = st.selectbox("Eje Y", options=numeric_cols, key="eda_y_col")

    color_options: List[str] = ["(sin color)"] + categorical_cols
    _ensure_selectbox_value("eda_color", color_options)
    color_choice = st.selectbox(
        "Color", options=color_options, key="eda_color")
    color_col: Optional[str] = None if color_choice == "(sin color)" else color_choice

    fig = relations_scatter_fig(df, x_col, y_col, color_col)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab: Target
# ---------------------------------------------------------------------------

def _render_target_relation(
    df: pd.DataFrame,
    target_column: str,
    feature: str,
    feature_type: str,
    invertir: bool,
) -> None:
    """Renderiza el scatter/box de relación target ↔ feature,
    invirtiendo ejes si se solicita."""
    if not invertir:
        fig = target_relation_fig(df, target_column, feature, feature_type)
    else:
        fig = target_relation_fig(df, feature, target_column, feature_type)
    st.plotly_chart(fig, use_container_width=True)


def render_target_tab(
    df: pd.DataFrame,
    target_column: Optional[str],
    problem_type: str,
    numeric_cols: List[str],
    categorical_cols: List[str],
    learn: bool,
) -> None:
    """Renderiza el tab de análisis del target."""
    st.subheader("Analisis del Target")

    if learn:
        render_learn_four_target_explanation()

    if not target_column:
        st.info("No se ha definido una columna target.")
        return

    fig = target_distribution_fig(df, target_column, problem_type)
    st.plotly_chart(fig, use_container_width=True)

    invertir: bool = st.checkbox(
        "Invertir", value=False, key="eda_invert_target")

    if numeric_cols:
        _ensure_selectbox_value("eda_target_feature", numeric_cols)
        feature = st.selectbox(
            "Relacionar target con feature numerica",
            options=numeric_cols,
            key="eda_target_feature",
        )
        _render_target_relation(
            df, target_column, feature, "numeric", invertir)

    elif categorical_cols:
        _ensure_selectbox_value("eda_target_feature_cat", categorical_cols)
        feature = st.selectbox(
            "Relacionar target con feature categorica",
            options=categorical_cols,
            key="eda_target_feature_cat",
        )
        if learn and target_column:
            render_learn_four_invert_explanation()

        _render_target_relation(
            df, target_column, feature, "categorical", invertir)
