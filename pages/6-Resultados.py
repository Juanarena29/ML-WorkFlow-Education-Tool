import streamlit as st

from src.ui.learn_explanations import render_learn_six_results_explanation
from src.ui.page1.dataset_upload_ui import render_next_step_button
from src.ui.page6.results_ui import (
    render_metrics_section,
    render_model_charts,
    render_model_details,
    render_save_section,
)
from src.utils.session import check_step_access, get_project, initialize_session


def main() -> None:
    initialize_session()

    if not check_step_access("training"):
        return

    project = get_project()
    if not project.metrics:
        st.error("No hay resultados para mostrar.")
        return

    learn: bool = project.ui_mode == "learn"
    problem_type: str = project.problem_type or "classification"

    st.title("Resultados y Comparacion")

    if learn:
        render_learn_six_results_explanation()

    # --- Metricas comparativas ---
    render_metrics_section(project.metrics, problem_type, learn)

    st.markdown("---")

    # --- Graficos por modelo ---
    has_data = (
        project.df_limpio is not None
        and not project.df_limpio.empty
        and project.target_column
        and project.train_test_split_config
    )

    if has_data:
        render_model_charts(
            trained_models=project.trained_models,
            problem_type=problem_type,
            df_limpio=project.df_limpio,
            target_column=project.target_column,
            split_config=project.train_test_split_config,
            target_encoder=project.target_label_encoder,
            learn=learn,
        )
    else:
        st.warning(
            "No se pueden generar graficos: falta el dataset limpio, "
            "la columna target o la configuracion de split."
        )

    # --- Detalles por modelo ---
    render_model_details(
        trained_models=project.trained_models,
        metrics=project.metrics,
        numeric_features=project.get_numeric_features(),
        categorical_features=project.get_categorical_features(),
        learn=learn,
    )

    # --- Guardado y navegacion ---
    render_save_section(project, learn)

    render_next_step_button(
        "Siguiente: Predicciones",
        "pages/7-Predicciones.py",
        project.is_step_completed("training"),
    )

    st.divider()
    st.markdown(
        "🔍 **Ver el codigo fuente:** "
        "[Repositorio en GitHub](https://github.com/Juanarena29/ML-WorkFlow-Education-Tool)"
    )


if __name__ == "__main__":
    main()
