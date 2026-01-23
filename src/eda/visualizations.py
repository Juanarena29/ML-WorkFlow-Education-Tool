"""
Funciones de visualizacion para el EDA.
"""

from __future__ import annotations

import plotly.express as px

from src.eda.statistics import get_correlation_matrix


def distribution_numeric_fig(df, column: str):
    """
    Construye un histograma para una columna numerica.
    """
    return px.histogram(df, x=column, nbins=30, title=f"Distribucion de {column}")


def distribution_categorical_fig(df, column: str):
    """
    Construye un barplot de conteo para una columna categorica.
    """
    counts = df[column].value_counts(dropna=False).reset_index()
    counts.columns = [column, "conteo"]
    return px.bar(counts, x=column, y="conteo", title=f"Distribucion de {column}")


def correlation_fig(df, numeric_cols):
    """
    Construye la figura de matriz de correlacion.
    """
    corr = get_correlation_matrix(df, numeric_cols)
    if corr.empty:
        return None
    return px.imshow(corr, text_auto=True, title="Matriz de correlacion")


def relations_scatter_fig(df, x_col: str, y_col: str, color_col: str | None = None):
    """
    Construye un scatter plot para dos columnas numericas.
    """
    return px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"{x_col} vs {y_col}")


def target_distribution_fig(df, target_column: str, problem_type: str):
    """
    Construye la figura de distribucion del target.
    """
    if problem_type == "classification":
        counts = df[target_column].value_counts(dropna=False).reset_index()
        counts.columns = [target_column, "conteo"]
        return px.bar(
            counts, x=target_column, y="conteo", title=f"Distribucion de {target_column}"
        )

    return px.histogram(
        df, x=target_column, nbins=30, title=f"Distribucion de {target_column}"
    )


def target_relation_fig(df, target_column: str, feature_column: str, feature_type: str):
    """
    Construye la figura de relacion entre target y una feature.
    """
    if feature_type == "numeric":
        return px.scatter(
            df,
            x=feature_column,
            y=target_column,
            color=None,
            title=f"{feature_column} vs {target_column}"
        )

    return px.box(
        df,
        x=feature_column,
        y=target_column,
        title=f"{feature_column} vs {target_column}",
    )
