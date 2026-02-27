"""
Componentes de UI para la pagina 6 - Resultados.

Cada funcion renderiza una seccion concreta, recibiendo datos
como parametros sin acceder al estado global.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from src.ml.evaluator import (
    build_metrics_table,
    confusion_matrix_fig,
    confusion_matrix_normalized_fig,
    decode_predictions,
    extract_feature_importance,
    feature_importance_fig,
    format_metric_value,
    format_metrics_table,
    get_results_score_map,
    get_roc_scores,
    get_train_test_split,
    metrics_comparison_fig,
    residuals_fig,
    roc_curve_fig,
)
from src.ui.learn_explanations import (
    render_learn_six_confusion_explanation,
    render_learn_six_details_explanation,
    render_learn_six_feature_explanation,
    render_learn_six_graphmodels_explanation,
    render_learn_six_metrics_explanation,
    render_learn_six_residuals_explanation,
    render_learn_six_savemodel_explanation,
)
from src.ml.predictor import register_model_hash
from src.utils.file_handler import save_model, save_project_config
from src.utils.session import MLProject


# ---------------------------------------------------------------------------
# Metricas comparativas
# ---------------------------------------------------------------------------


def render_metrics_section(
    metrics: Dict[str, Dict[str, float]],
    problem_type: str,
    learn: bool,
) -> None:
    """Tabla comparativa + bar chart de metricas seleccionada."""
    if not metrics:
        st.info("No hay metricas disponibles.")
        return

    df_metrics = build_metrics_table(metrics)
    df_display = format_metrics_table(df_metrics, problem_type)

    st.subheader("Tabla comparativa")
    st.dataframe(df_display, use_container_width=True)

    if learn:
        render_learn_six_metrics_explanation()

    # Selector de metricas
    score_map = get_results_score_map(problem_type)
    available = {
        label: col for label, col in score_map.items()
        if col in df_metrics.columns
    }
    if not available:
        return

    metric_label = st.selectbox(
        "Metrica para comparar", options=list(available.keys()),
    )
    metric_name = available[metric_label]

    fig = metrics_comparison_fig(
        df_metrics, metric_name, metric_label, problem_type)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Graficos por modelo — Clasificacion
# ---------------------------------------------------------------------------


def _render_classification_charts(
    y_test_plot: pd.Series,
    y_pred: Any,
    y_test_roc: Optional[pd.Series],
    pipeline: Any,
    X_test: pd.DataFrame,
    model_name: str,
    learn: bool,
) -> None:
    """Renderiza matrices de confusion y curva ROC para un modelo."""
    if learn:
        render_learn_six_confusion_explanation()

    st.plotly_chart(
        confusion_matrix_fig(y_test_plot, y_pred), use_container_width=True,
    )
    st.plotly_chart(
        confusion_matrix_normalized_fig(y_test_plot, y_pred),
        use_container_width=True,
    )

    # ROC: requiere target numerico y modelo con probabilidades
    if y_test_roc is None:
        st.info("No se pudo preparar el target para ROC AUC en este modelo.")
        return

    if y_test_roc.nunique() != 2:
        return

    y_score = get_roc_scores(pipeline, X_test)
    if y_score is None:
        return

    try:
        fig = roc_curve_fig(y_test_roc, y_score)
        st.plotly_chart(fig, use_container_width=True)
    except (ValueError, TypeError) as exc:
        st.info(f"No se pudo generar la curva ROC para {model_name}: {exc}")


# ---------------------------------------------------------------------------
# Graficos por modelo — Regresion
# ---------------------------------------------------------------------------


def _render_regression_charts(
    y_test: pd.Series,
    y_pred: Any,
    learn: bool,
) -> None:
    """Renderiza grafico de residuos para un modelo de regresion."""
    if learn:
        render_learn_six_residuals_explanation()

    fig = residuals_fig(y_test, y_pred)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Graficos por modelo (orquestacion)
# ---------------------------------------------------------------------------


def render_model_charts(
    trained_models: Dict[str, Any],
    problem_type: str,
    df_limpio: pd.DataFrame,
    target_column: str,
    split_config: Dict[str, Any],
    target_encoder: Any,
    learn: bool,
) -> None:
    """Renderiza graficos por cada modelo entrenado."""
    if learn:
        render_learn_six_graphmodels_explanation()

    st.subheader("Gráficos por modelo")

    # Reconstruir split
    try:
        _X_train, X_test, _y_train, y_test = get_train_test_split(
            df_limpio, target_column, problem_type, split_config,
        )
    except (KeyError, ValueError, TypeError) as exc:
        st.error(f"No se pudo reconstruir el split: {exc}")
        return

    if not trained_models:
        st.warning("No hay modelos entrenados para graficar.")
        return

    for model_name, pipeline in trained_models.items():
        with st.expander(f"{model_name} - gráficos"):
            try:
                y_pred = pipeline.predict(X_test)
            except (ValueError, TypeError) as exc:
                st.error(f"No se pudo predecir con {model_name}: {exc}")
                continue

            if problem_type == "classification":
                y_test_plot, y_pred_dec, y_test_roc = decode_predictions(
                    y_test, y_pred, target_encoder,
                )
                _render_classification_charts(
                    y_test_plot, y_pred_dec, y_test_roc,
                    pipeline, X_test, model_name, learn,
                )
            else:
                _render_regression_charts(y_test, y_pred, learn)


# ---------------------------------------------------------------------------
# Detalles por modelo
# ---------------------------------------------------------------------------


def render_model_details(
    trained_models: Dict[str, Any],
    metrics: Dict[str, Dict[str, float]],
    numeric_features: List[str],
    categorical_features: List[str],
    learn: bool,
) -> None:
    """Metricas detalladas + feature importance por modelo."""
    st.subheader("Detalles por modelo")

    if learn:
        render_learn_six_details_explanation()

    for model_name, pipeline in trained_models.items():
        with st.expander(model_name):
            st.write("Metricas")
            model_metrics = metrics.get(model_name, {})
            formatted = {
                k: format_metric_value(k, v)
                for k, v in model_metrics.items()
            }
            st.json(formatted)

            importances = extract_feature_importance(
                pipeline, numeric_features, categorical_features,
            )
            if importances is None or importances.empty:
                st.write("Este modelo no expone importancias.")
            else:
                if learn:
                    render_learn_six_feature_explanation()
                fig = feature_importance_fig(importances)
                st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Guardado de modelos y configuracion
# ---------------------------------------------------------------------------


def render_save_section(
    project: MLProject,
    learn: bool,
) -> None:
    """Selector de modelo a guardar + botones de accion."""
    if not project.trained_models:
        return

    model_options = list(project.trained_models.keys())
    selected_model = st.selectbox(
        "Selecciona un modelo para guardar",
        options=model_options,
    )

    if learn:
        render_learn_six_savemodel_explanation()

    cols = st.columns(3)

    with cols[0]:
        if st.button("¡PRUEBA TU MODELO!"):
            st.switch_page("pages/7-Predicciones.py")

    with cols[1]:
        if st.button("Guardar modelo seleccionado"):
            model = project.trained_models[selected_model]
            filename = f"{selected_model}_{project.problem_type}"
            try:
                path = save_model(model, filename)
                register_model_hash(filename)
            except (OSError, ValueError, TypeError) as exc:
                st.error(f"No se pudo guardar el modelo: {exc}")
            else:
                st.success(f"Modelo guardado en {path}")

    with cols[2]:
        if st.button("Guardar configuración del proyecto"):
            try:
                path = save_project_config(project.to_dict())
            except (OSError, ValueError, TypeError) as exc:
                st.error(f"No se pudo guardar la configuración: {exc}")
            else:
                st.success(f"Configuración guardada en {path}")
