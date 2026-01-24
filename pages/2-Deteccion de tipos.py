import streamlit as st
from src.data.analyzer import detect_column_types, detect_problem_type
from src.data.validator import validate_target
from src.utils.constants import COLUMN_TYPES
from src.utils.session import (
    add_operation_log, check_step_access, get_project, initialize_session)


def _render_auto_types(auto_types: dict) -> None:
    """
    Renderiza los tipos detectados automáticamente en una tabla.

    Args:
        auto_types: Diccionario con tipos detectados por columna.
    """
    rows = [{"columna": col, "tipo_detectado": col_type}
            for col, col_type in auto_types.items()]
    st.dataframe(rows, use_container_width=True)


def _render_type_selectors(df, auto_types: dict) -> dict:
    """
    Renderiza selectores para confirmar o ajustar tipos de columnas.

    Args:
        df: DataFrame con las columnas.
        auto_types: Tipos detectados automáticamente.

    Returns:
        Diccionario con tipos seleccionados por columna.
    """
    selected_types = {}

    for col in df.columns:
        default_type = auto_types.get(col, "categorical")
        default_index = COLUMN_TYPES.index(default_type)
        selected_types[col] = st.selectbox(
            f"{col}",
            options=COLUMN_TYPES,
            index=default_index,
            key=f"type_{col}",
        )

    return selected_types


def _render_problem_type_preview(
    df, target_column: str, override: str | None
) -> None:
    """
    Muestra una vista previa del tipo de problema detectado y validaciones del target.

    Args:
        df: DataFrame del dataset.
        target_column: Columna target seleccionada.
    """
    if not target_column:
        return

    detected_type = None
    try:
        detected_type = detect_problem_type(df, target_column)
    except ValueError as exc:
        st.error(str(exc))
        return

    if override:
        st.info(
            f"Tipo de problema seleccionado manualmente: {override}."
        )
        st.caption(f"Autodetectado: {detected_type}")
    else:
        st.info(f"Tipo de problema detectado: {detected_type}")

    # validaciones basicas del target para mostrar alertas
    validation_messages = validate_target(
        df, target_column, problem_type=override or detected_type
    )
    for msg in validation_messages:
        st.warning(msg)


def _render_problem_type_selector() -> None:
    options = {
        "Auto (detectar)": None,
        "Clasificacion": "classification",
        "Regresion": "regression",
    }
    current = st.session_state.get("problem_type_override")
    labels = list(options.keys())
    values = list(options.values())
    default_index = values.index(current) if current in values else 0
    selected_label = st.selectbox(
        "Opciones",
        options=labels,
        index=default_index,
        help="Si eliges una opcion manual, se usara en la deteccion de tipos.",
    )
    st.session_state.problem_type_override = options[selected_label]
    if st.session_state.problem_type_override:
        st.info(
            f"Preferencia guardada: {st.session_state.problem_type_override}."
        )


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
        with st.expander("¿PARA QUE HACEMOS ESTO?"):
            st.markdown(
                "Tal como vimos en el paso anterior, la **selección de tipos de datos** es clave para que el modelo (y nosotros) sepamos **cómo usar correctamente la información**.\n\n"
                "**ML WorkFlow** detecta automáticamente los tipos de cada columna y los clasifica como "
                "**numéricos**, **categóricos**, **identificadores**, **fechas** o **textos**.\n\n"
                "Por seguridad y transparencia, el usuario puede **revisar y corregir** esta detección antes de continuar, "
                "evitando errores comunes que afectarían el entrenamiento del modelo."
            )

    auto_types = detect_column_types(df)

    st.subheader("Deteccion automatica")
    _render_auto_types(auto_types)

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
        with st.expander('¿QUÉ ES EL "TARGET"?'):
            st.markdown(
                'El **target** (u **objetivo**) es la columna que indica **qué queremos que el modelo prediga.** \n\n'
                'Es el valor que el modelo intenta aprender a partir del resto de los datos. \n\n'
                'Por ejemplo, en un dataset inmobiliario, lo más lógico es que el target sea el **precio** de la propiedad.'
                'Para nuestro ejemplo, el target será **es_caro**, entrenaremos al modelo para que prediga si una propiedad debería ser cara o barata.'
            )

    st.subheader("Elija el tipo de problema (REGRESIÓN o CLASIFICACIÓN)")
    _render_problem_type_selector()
    override = st.session_state.get("problem_type_override")
    _render_problem_type_preview(df, target_column, override)

    if learn:
        with st.expander('¿REGRESIÓN? ¿CLASIFICACIÓN?'):
            st.markdown("""
                **El tipo de problema depende de lo que querés predecir (target):**

                - **Regresión**: cuando el resultado es un **número**  
                Ej.: precio de una casa, ventas, temperatura.

                - **Clasificación**: cuando el resultado es una **categoría**  
                Ej.: sí/no, aprobado/desaprobado, tipo de cliente.

                **Regla rápida:**  
                número → *Regresión* | categorías → *Clasificación*

                ⚠️ Si el target es numérico pero representa categorías (0, 1, 2), sigue siendo **Clasificación**.
                """)

    st.subheader("Ajuste el tipo de dato de cada columna (si es necesario)")

    with st.form("types_form"):
        selected_types = _render_type_selectors(df, auto_types)
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

        project.column_types = selected_types
        project.target_column = target_column
        project.problem_type = problem_type
        project.update_metadata()

        # paso completado
        st.session_state.confirmations["types_confirmed"] = True

        add_operation_log(
            "detect_types",
            f"Tipos confirmados. Target: {target_column}. Problema: {problem_type}.",
            status="success",
        )
        st.success("Tipos y target confirmados.")

    if project.is_step_completed("types"):
        if st.button("Siguiente: Limpieza de Datos"):
            st.switch_page("pages/3-Limpieza de datos.py")


if __name__ == "__main__":
    main()
