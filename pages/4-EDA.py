import streamlit as st

from src.ui.learn_explanations import render_learn_four_eda_explanation
from src.ui.page1.dataset_upload_ui import render_next_step_button
from src.ui.page4.eda_ui import (
    render_correlations_tab,
    render_distributions_tab,
    render_relations_tab,
    render_target_tab,
)
from src.utils.session import check_step_access, get_project, initialize_session


def main() -> None:
    initialize_session()

    if not check_step_access("cleaning"):
        return

    project = get_project()
    df = project.df_limpio

    if df is None or df.empty:
        st.error("No hay dataset limpio disponible.")
        return

    learn: bool = project.ui_mode == "learn"

    st.title("EDA - Analisis Exploratorio")

    if learn:
        render_learn_four_eda_explanation()
    else:
        st.write(
            "Explora distribuciones, correlaciones y relaciones entre variables."
        )

    # Listas de columnas segun tipos detectados
    numeric_cols = project.get_numeric_features()
    categorical_cols = project.get_categorical_features()

    tabs = st.tabs(["Distribuciones", "Correlaciones", "Relaciones", "Target"])

    with tabs[0]:
        render_distributions_tab(df, numeric_cols, categorical_cols, learn)

    with tabs[1]:
        render_correlations_tab(df, numeric_cols, learn)

    with tabs[2]:
        render_relations_tab(df, numeric_cols, categorical_cols, learn)

    with tabs[3]:
        render_target_tab(
            df,
            project.target_column,
            project.problem_type or "classification",
            numeric_cols,
            categorical_cols,
            learn,
        )

    render_next_step_button(
        "Siguiente: Entrenamiento",
        "pages/5-Entrenamiento.py",
        project.is_step_completed("cleaning"),
    )

    st.divider()
    st.markdown(
        "🔍 **Ver el código fuente:** "
        "[Repositorio en GitHub](https://github.com/Juanarena29/ML-WorkFlow-Education-Tool)"
    )


if __name__ == "__main__":
    main()
