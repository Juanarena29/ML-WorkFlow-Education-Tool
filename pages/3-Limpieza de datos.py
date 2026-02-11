import streamlit as st

from src.data.analyzer import analyze_missing_values
from src.data.cleaner import apply_cleaning_config
from src.ui.page3.cleaning_ui import build_cleaning_config, show_cleaning_suggestions
from src.ui.learn_explanations import (
    render_learn_three_duplicates_explanation,
    render_learn_three_imputation_explanation,
    render_learn_three_nans_explanation,
    render_learn_three_suggestions_explanation,
    render_learn_three_treatment_explanation,
)
from src.savings.project_updates import save_cleaning_result
from src.utils.session import (
    add_operation_log,
    check_step_access,
    get_project,
    initialize_session,
)
from src.ui.page1.dataset_upload_ui import render_next_step_button


def main() -> None:
    initialize_session()

    if not check_step_access("types"):
        return

    project = get_project()
    df = project.df_original
    learn = project.ui_mode == "learn"

    if df is None or df.empty:
        st.error("No hay dataset cargado.")
        return

    if not project.column_types:
        st.error("Debes confirmar los tipos de columnas antes de limpiar datos.")
        return

    st.title("Configuracion de Tratamiento de Datos")
    if learn:
        render_learn_three_treatment_explanation()
    else:
        st.write("Define como tratar valores faltantes y duplicados.")

    missing_summary = analyze_missing_values(df)
    if missing_summary.empty:
        st.info("No se detectaron valores faltantes.")
    else:
        st.subheader("Resumen de NaNs")
        if learn and not missing_summary.empty:
            render_learn_three_nans_explanation()
        st.dataframe(missing_summary, use_container_width=True)

    show_cleaning_suggestions(df, project.column_types)
    if learn:
        render_learn_three_suggestions_explanation()

    duplicate_count = df.duplicated().sum()
    drop_duplicates = st.checkbox(
        f"Eliminar duplicados (detectados: {duplicate_count})",
        value=False,
    )
    if learn and duplicate_count > 0:
        render_learn_three_duplicates_explanation()

    cleaning_config = build_cleaning_config(df, project.column_types)
    cleaning_config["drop_duplicates"] = drop_duplicates

    st.subheader("Imputacion por columna con NaNs")
    if learn:
        render_learn_three_imputation_explanation()

    preview_df = apply_cleaning_config(df, cleaning_config)
    st.subheader("Vista previa")
    st.dataframe(preview_df.head(50), use_container_width=True)

    if st.button("Aplicar tratamiento"):
        save_cleaning_result(project, preview_df, cleaning_config)
        st.session_state.confirmations["cleaning_confirmed"] = True
        add_operation_log(
            "apply_cleaning",
            f"Tratamiento aplicado. Filas: {len(preview_df)}.",
            status="success",
        )
        st.success("Tratamiento aplicado y dataset guardado.")

    render_next_step_button(
        "Siguiente: EDA",
        "pages/4-EDA.py",
        project.is_step_completed("cleaning"),
    )


if __name__ == "__main__":
    main()
