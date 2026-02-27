import streamlit as st

from src.ml.predictor import execute_prediction
from src.ui.page7.prediction_ui import (
    render_dataset_upload,
    render_model_selector,
    render_prediction_header,
    render_prediction_results,
)
from src.utils.session import (
    add_operation_log,
    check_step_access,
    get_project,
    initialize_session,
)


def main() -> None:
    initialize_session()

    if not check_step_access("types"):
        return

    project = get_project()
    learn: bool = project.ui_mode == "learn"

    # --- Header ---
    render_prediction_header(learn)

    feature_cols = project.get_feature_columns()
    if not feature_cols:
        st.error(
            "No hay features definidas. Completa la deteccion de tipos primero.",
        )
        return

    # --- Seleccion de modelo ---
    model, model_name = render_model_selector(project, learn)

    # --- Carga de dataset ---
    df, y_true = render_dataset_upload(
        feature_cols, project.target_column, learn,
    )
    if df is None:
        return

    if model is None:
        st.warning("Selecciona y carga un modelo antes de predecir.")
        return

    # --- Prediccion ---
    if st.button("Generar predicciones"):
        try:
            pred_result = execute_prediction(
                model=model,
                df=df,
                feature_cols=feature_cols,
                problem_type=project.problem_type or "classification",
                target_encoder=project.target_label_encoder,
            )
        except (ValueError, TypeError) as exc:
            add_operation_log("predict", str(exc), status="error")
            st.error(str(exc))
            return

        render_prediction_results(
            result=pred_result,
            y_true=y_true,
            problem_type=project.problem_type or "classification",
            model_name=model_name,
            target_encoder=project.target_label_encoder,
            learn=learn,
        )

    st.divider()
    st.markdown(
        "🔍 **Ver el codigo fuente:** "
        "[Repositorio en GitHub](https://github.com/Juanarena29/ML-WorkFlow-Education-Tool)"
    )


if __name__ == "__main__":
    main()
