import streamlit as st

from src.ui.page1.dataset_upload_ui import render_next_step_button
from src.ui.page5.training_ui import (
    handle_training,
    render_gridsearch_config,
    render_model_selection,
    render_scoring_selector,
    render_split_config,
    render_training_header,
)
from src.utils.session import check_step_access, get_project, initialize_session


def main() -> None:
    initialize_session()

    if not check_step_access("cleaning"):
        return

    project = get_project()
    df = project.df_limpio

    validation_errors = project.validate_for_training()
    if validation_errors:
        for err in validation_errors:
            st.error(err)
        return

    learn: bool = project.ui_mode == "learn"
    problem_type: str = project.problem_type or "classification"

    # --- Header ---
    render_training_header(problem_type, learn)

    # --- Seleccion de modelos ---
    selected_models = render_model_selection(problem_type, learn)
    if not selected_models:
        return

    # --- Config Train/Test split ---
    split_config = render_split_config(problem_type, learn)

    # --- Config GridSearchCV ---
    gridsearch_config = render_gridsearch_config(
        problem_type, project.runtime_mode, learn,
    )

    # --- Scoring ---
    target_nunique = (
        df[project.target_column].nunique(dropna=True)
        if project.target_column and project.target_column in df.columns
        else 0
    )
    scoring = render_scoring_selector(problem_type, target_nunique, learn)

    # --- Entrenamiento ---
    handle_training(
        project=project,
        df=df,
        selected_models=selected_models,
        split_config=split_config,
        gridsearch_config=gridsearch_config,
        scoring=scoring,
    )

    # --- Navegacion ---
    render_next_step_button(
        "Siguiente: Resultados",
        "pages/6-Resultados.py",
        project.is_step_completed("training"),
    )

    st.divider()
    st.markdown(
        "🔍 **Ver el código fuente:** "
        "[Repositorio en GitHub](https://github.com/Juanarena29/ML-WorkFlow-Education-Tool)"
    )


if __name__ == "__main__":
    main()
