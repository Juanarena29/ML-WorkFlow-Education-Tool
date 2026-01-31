import streamlit as st

from src.utils.session import (
    get_project,
    initialize_session,
    check_step_access
)
from src.ui.page1.render_dataset_info_ui import (
    render_dataset_preview,
)
from src.ui.page1.dataset_upload_ui import (
    render_dataset_uploader,
    render_next_step_button
)
from src.data.loader import (get_basic_info)
from src.ui.page1.handlers_ui import (
    handle_auto_load_sample_dataset,
    handle_empty_columns_removal,
    handle_file_upload_and_validation,
    confirm_and_save_dataset
)
from src.ui.learn_explanations import (
    render_learn_one_dataset_explanation,
)


def main() -> None:
    initialize_session()

    if not check_step_access("home"):
        return

    project = get_project()

    learn = project.ui_mode == "learn"

    st.title("Carga de Dataset")

    if learn:
        render_learn_one_dataset_explanation()

    uploaded_file = render_dataset_uploader("Selecciona un CSV", types=["csv"])

    if learn and uploaded_file is None and project.df_original is None:
        if handle_auto_load_sample_dataset(project):
            st.rerun()
        return

    # Caso: no subió nada pero ya existe un dataset cargado en sesión/proyecto
    if uploaded_file is None and project.df_original is not None:
        st.info("Ya hay un dataset cargado. Puedes subir otro para reemplazarlo.")
        info = get_basic_info(project.df_original)
        render_dataset_preview(project.df_original, info, learn)
        render_next_step_button(
            "Siguiente: Detección de Tipos",
            "pages/2-Deteccion de tipos.py",
            project.is_step_completed("load"),
        )
        return

    # Caso: todavía no subió archivo
    if uploaded_file is None:
        return

    df = handle_file_upload_and_validation(uploaded_file, project)
    if df is None:
        return

    df_view = handle_empty_columns_removal(df, learn)
    info = get_basic_info(df_view)
    render_dataset_preview(df_view, info, learn)

    # Siempre intentar confirmar y guardar si aún no se ha hecho
    confirm_and_save_dataset(df_view, project, info)

    # Mostrar botón de siguiente si el paso ya está completado
    render_next_step_button(
        "Siguiente: Detección de Tipos",
        "pages/2-Deteccion de tipos.py",
        project.is_step_completed("load"),
    )

    st.divider()
    st.markdown(
        "🔍 **Ver el código fuente:** "
        "[Repositorio en GitHub](https://github.com/Juanarena29/ML-WorkFlow-Education-Tool)"
    )


if __name__ == "__main__":
    main()
