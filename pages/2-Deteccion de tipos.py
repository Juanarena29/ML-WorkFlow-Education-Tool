import streamlit as st

from src.data.analyzer import detect_column_types, detect_problem_type
from src.data.validator import validate_target
from src.ui.learn_explanations import (
    render_learn_two_title_explanation,
    render_learn_two_target_explanation,
    render_learn_two_type_explanation,
)
from src.ui.page2.render_types_ui import (
    render_auto_types,
    render_problem_type_preview,
    render_problem_type_selector,
    render_type_selectors,
)
from src.savings.project_updates import save_types_and_target
from src.utils.session import (
    add_operation_log,
    check_step_access,
    get_project,
    initialize_session,
)
from src.ui.page1.dataset_upload_ui import render_next_step_button


def main() -> None:
    initialize_session()

    if not check_step_access("load"):
        return

    project = get_project()

    df = project.df_original

    learn = project.ui_mode == "learn"

    if df is None or df.empty:
        st.error("No hay dataset cargado. Vuelve a la pagina de carga.")
        return

    st.title("Deteccion y Confirmacion de Tipos")
    if learn:
        render_learn_two_title_explanation()

    auto_types = detect_column_types(df)

    st.subheader("Deteccion automatica")
    render_auto_types(auto_types)

    st.subheader("ELIJA EL TARGET")
    target_default = project.target_column or df.columns[0]
    if target_default not in df.columns:
        target_default = df.columns[0]
    target_column = st.selectbox(
        "Selecciona la columna target",
        options=list(df.columns),
        index=list(df.columns).index(target_default),
        key="target_column",
    )

    if learn:
        render_learn_two_target_explanation()

    st.subheader("Elija el tipo de problema (REGRESIÓN o CLASIFICACIÓN)")
    render_problem_type_selector()
    override = st.session_state.get("problem_type_override")
    render_problem_type_preview(df, target_column, override)

    if learn:
        render_learn_two_type_explanation()

    st.subheader("Ajuste el tipo de dato de cada columna (si es necesario)")

    with st.form("types_form"):
        selected_types = render_type_selectors(df, auto_types)
        st.write("Por favor, confirme los tipos y target")
        submitted = st.form_submit_button("Confirmar tipos y target")

    if submitted:
        if override:
            problem_type = override
        else:
            try:
                problem_type = detect_problem_type(df, target_column)
            except ValueError as exc:
                st.error(str(exc))
                add_operation_log("detect_types", str(exc), status="error")
                return

        validation_messages = validate_target(
            df, target_column, problem_type=problem_type
        )
        if validation_messages:
            for msg in validation_messages:
                st.error(msg)
            add_operation_log(
                "detect_types", "Validacion de target fallida", status="warning"
            )
            return

        save_types_and_target(project, selected_types,
                              target_column, problem_type)

        # paso completado
        st.session_state.confirmations["types_confirmed"] = True

        add_operation_log(
            "detect_types",
            f"Tipos confirmados. Target: {target_column}. Problema: {problem_type}.",
            status="success",
        )
        st.success("Tipos y target confirmados.")

    render_next_step_button(
        "Siguiente: Limpieza de Datos",
        "pages/3-Limpieza de datos.py",
        project.is_step_completed("types"),
    )


if __name__ == "__main__":
    main()
