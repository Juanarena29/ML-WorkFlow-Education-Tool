"""
Componentes de UI para la pagina 7 - Predicciones.

Cada funcion renderiza una seccion concreta, recibiendo datos
como parametros sin acceder al estado global.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from src.data.loader import load_dataset
from src.data.validator import validate_prediction_columns
from src.ml.evaluator import confusion_matrix_fig, confusion_matrix_normalized_fig
from src.ml.predictor import (
    PredictionResult,
    align_labels_for_comparison,
    load_model_safe,
    prediction_vs_real_fig,
    verify_model_integrity,
)
from src.ui.learn_explanations import (
    render_learn_seven_csv_explanation,
    render_learn_seven_graph_explanation,
    render_learn_seven_prediction_explanation,
    render_learn_seven_whatmodel_explanation,
)
from src.utils.file_handler import list_saved_models
from src.utils.session import MLProject, add_operation_log


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


def render_prediction_header(learn: bool) -> None:
    """Titulo de la pagina y explicacion Learn."""
    st.title("Prediccion")
    if learn:
        render_learn_seven_prediction_explanation()


# ---------------------------------------------------------------------------
# Selector de modelo
# ---------------------------------------------------------------------------


def render_model_selector(
    project: MLProject,
    learn: bool,
) -> Tuple[Optional[Any], Optional[str]]:
    """Renderiza selector de modelo (sesion o guardado).

    Returns:
        (model, model_name) — ambos None si no hay modelo listo.
    """
    if learn:
        render_learn_seven_whatmodel_explanation()

    source = st.radio(
        "Origen del modelo",
        options=["En sesion", "Guardado"],
        horizontal=True,
    )

    if source == "En sesion":
        return _select_session_model(project.trained_models)
    return _select_saved_model()


def _select_session_model(
    trained_models: Dict[str, Any],
) -> Tuple[Optional[Any], Optional[str]]:
    if not trained_models:
        st.info(
            "No hay modelos en sesion. Entrena modelos o carga uno guardado.",
        )
        return None, None

    model_name = st.selectbox(
        "Modelo en sesion",
        options=list(trained_models.keys()),
    )
    return trained_models[model_name], model_name


def _select_saved_model() -> Tuple[Optional[Any], Optional[str]]:
    saved = list_saved_models()
    if not saved:
        st.info("No hay modelos guardados en la carpeta models.")
        return None, None

    selected_file = st.selectbox("Modelo guardado", options=saved)

    # Mostrar estado de integridad
    is_valid, integrity_msg = verify_model_integrity(selected_file)
    if not is_valid:
        st.warning(f"⚠️ {integrity_msg}")

    if st.button("Cargar modelo guardado"):
        try:
            loaded = load_model_safe(selected_file)
            st.session_state.prediction_model = loaded
            st.session_state.prediction_model_name = selected_file.replace(
                ".pkl", "",
            )
            add_operation_log(
                "load_model",
                f"Modelo cargado para prediccion: {selected_file}.",
                status="success",
            )
            st.success("Modelo cargado en la sesion.")
        except (ValueError, FileNotFoundError, OSError) as exc:
            add_operation_log("load_model", str(exc), status="error")
            st.error(str(exc))

    if "prediction_model" in st.session_state:
        model = st.session_state.prediction_model
        model_name = st.session_state.get(
            "prediction_model_name", "modelo_guardado",
        )
        st.success(f"Modelo listo: {model_name}.")
        return model, model_name

    return None, None


# ---------------------------------------------------------------------------
# Carga de dataset de prediccion
# ---------------------------------------------------------------------------


def render_dataset_upload(
    feature_cols: List[str],
    target_column: Optional[str],
    learn: bool,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """Sube o carga un CSV y devuelve (df_features, y_true_or_None).

    Retorna (None, None) si no hay datos disponibles.
    """
    st.subheader("Dataset para prediccion")

    if learn:
        render_learn_seven_csv_explanation()

    uploaded_file = st.file_uploader(
        "Sube un CSV con filas nuevas", type=["csv"],
    )

    df = _load_prediction_data(uploaded_file, learn)
    if df is None:
        return None, None

    st.dataframe(df.head(50), use_container_width=True)

    has_target = st.checkbox(
        "El archivo posee el dato a predecir (target)",
        value=True,
    )

    errors, y_true, df_clean = validate_prediction_columns(
        df, feature_cols, target_column, has_target,
    )
    if errors:
        for err in errors:
            st.error(err)
        return None, None

    return df_clean, y_true


def _load_prediction_data(
    uploaded_file: Any,
    learn: bool,
) -> Optional[pd.DataFrame]:
    """Resuelve la fuente de datos: upload del usuario o sample educativo."""
    if uploaded_file is not None:
        try:
            return load_dataset(uploaded_file)
        except ValueError as exc:
            add_operation_log(
                "load_dataset_prediction", str(exc), status="error",
            )
            st.error(str(exc))
            return None

    if not learn:
        return None

    # Modo learn: cargar dataset de ejemplo
    sample_path = Path(__file__).resolve().parents[3] / "TestEDUCATOR.csv"
    if not sample_path.exists():
        st.warning("No se encontro el dataset de ejemplo TestEDUCATOR.csv.")
        return None

    try:
        df = pd.read_csv(sample_path)
    except (OSError, pd.errors.ParserError) as exc:
        add_operation_log(
            "load_dataset_prediction", str(exc), status="error",
        )
        st.error(f"No se pudo cargar el dataset de ejemplo: {exc}")
        return None

    st.info(
        "Dataset de prediccion de ejemplo cargado automaticamente (modo learn).",
    )
    return df


# ---------------------------------------------------------------------------
# Resultados de prediccion
# ---------------------------------------------------------------------------


def render_prediction_results(
    result: PredictionResult,
    y_true: Optional[pd.Series],
    problem_type: str,
    model_name: Optional[str],
    target_encoder: Any,
    learn: bool,
) -> None:
    """Muestra tabla de predicciones, graficos comparativos y descarga."""
    st.success("Predicciones generadas.")
    st.dataframe(result.output_df.head(50), use_container_width=True)

    if y_true is not None:
        _render_comparison_charts(
            y_true, result.predictions, problem_type, target_encoder, learn,
        )

    # Boton de descarga
    csv_data = result.output_df.to_csv(index=False)
    safe_name = model_name or "modelo"
    st.download_button(
        "Descargar predicciones",
        data=csv_data,
        file_name=f"predicciones_{safe_name}.csv",
        mime="text/csv",
    )


def _render_comparison_charts(
    y_true: pd.Series,
    preds: Any,
    problem_type: str,
    target_encoder: Any,
    learn: bool,
) -> None:
    """Graficos de comparacion: confusion matrix o scatter real vs pred."""
    if learn:
        render_learn_seven_graph_explanation()

    y_true_plot, preds_plot = align_labels_for_comparison(
        y_true, preds, problem_type, target_encoder,
    )

    if problem_type == "classification":
        st.plotly_chart(
            confusion_matrix_fig(y_true_plot, preds_plot),
            use_container_width=True,
        )
        st.plotly_chart(
            confusion_matrix_normalized_fig(y_true_plot, preds_plot),
            use_container_width=True,
        )
    elif pd.api.types.is_numeric_dtype(y_true):
        fig = prediction_vs_real_fig(y_true, preds_plot)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("El target no es numerico. No se puede graficar.")
