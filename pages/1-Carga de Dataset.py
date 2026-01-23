import streamlit as st

from src.data.loader import (
    get_all_nan_columns,
    get_basic_info,
    get_high_nan_columns,
    load_and_validate,
    truncate_dataset_if_needed,
)
from src.utils.constants import IS_CLOUD, CLOUD_MAX_ROWS, CLOUD_MAX_COLUMNS
from src.utils.session import (
    add_operation_log,
    get_project,
    initialize_session,
    reset_project,
    check_step_access
)


def _render_basic_info(info: dict) -> None:
    """Crea las métricas de info básica del dataframe."""
    cols = st.columns(3)
    cols[0].metric("Filas", info["rows"])
    cols[1].metric("Columnas", info["columns"])
    cols[2].metric("Memoria (MB)", f"{info['memory_mb']:.2f}")


def _render_dtypes_table(dtypes: dict) -> None:
    """Renderiza una tabla con nombre de columnas y tipos detectados."""
    dtype_rows = [{"Columna": col, "Tipo": dtype}
                  for col, dtype in dtypes.items()]
    st.dataframe(dtype_rows, use_container_width=True)


def _render_high_nan_warning(df, threshold: float = 0.3) -> bool:
    """
    Muestra un warning si existen columnas con alto % de NaNs.
    Devuelve True si se mostró el warning, False si no.
    """
    high_nan = get_high_nan_columns(df, threshold=threshold)
    if not high_nan:
        return False

    cols = ", ".join([f"{col} ({ratio:.0%})" for col,
                     ratio in high_nan.items()])
    st.warning(f"Columnas con más de {int(threshold * 100)}% NaNs: {cols}.")
    return True


def main() -> None:
    initialize_session()

    if not check_step_access("home"):
        return

    project = get_project()

    learn = project.ui_mode == "learn"

    st.title("Carga de Dataset")

    if learn:
        with st.expander("📦 ¿Qué es un dataset?"):
            st.markdown(
                "Un **dataset** es una tabla de datos donde cada fila representa un algo (por ejemplo, una casa) y cada columna una característica de ese algo (precio, metros cuadrados, ubicación, etc.).\n\n"
                "En esta app los datasets se cargan en formato **CSV**, un tipo de archivo sencillo que puede abrirse y editarse con Excel, Google Sheets u otras herramientas similares. \n\n"
                "En el modo **EDUCACIÓN** se incluye un **dataset de ejemplo** para que puedas recorrer todo el flujo sin necesidad de subir datos propios."
            )

    uploaded_file = st.file_uploader("Selecciona un CSV", type=["csv"])

    # Caso: no subió nada pero ya existe un dataset cargado en sesión/proyecto
    if uploaded_file is None and project.df_original is not None:
        st.info("Ya hay un dataset cargado. Puedes subir otro para reemplazarlo.")

        df_loaded = project.df_original
        info_loaded = get_basic_info(df_loaded)

        _render_basic_info(info_loaded)

        if learn:
            with st.expander("📊 ¿Qué es esta información?"):
                st.markdown(
                    "Este resumen muestra cuánta información tiene tu dataset:\n\n"
                    "- **Filas**: cantidad de registros disponibles.\n"
                    "- **Columnas**: variables o características de cada registro.\n"
                    "- **Tamaño**: espacio que ocupa el dataset en memoria.\n\n"
                    "Cuantos más datos relevantes haya, mayor suele ser el potencial del modelo para aprender."
                )

        st.subheader("Tipos detectados")
        _render_dtypes_table(info_loaded["dtypes"])

        if learn:
            with st.expander("🧩 ¿Qué son los tipos de datos?"):
                st.markdown(
                    "Los tipos de datos indican cómo está almacenada la información en cada columna y cómo puede ser utilizada por el modelo.\n\n"
                    "En esta tabla podés encontrar principalmente:\n\n"
                    "- **int64**: números enteros. Ejemplos: cantidad de habitaciones, pisos, años.\n"
                    "- **float64**: números con decimales. Ejemplos: precios, promedios, porcentajes, superficies.\n"
                    "- **object**: texto o categorías. Ejemplos: ciudad, tipo de propiedad, nombre de una categoría.\n\n"
                    "Cada tipo de dato se trata de forma distinta durante el entrenamiento del modelo, por eso es importante identificarlos correctamente."
                )

        st.subheader("Vista previa del dataset")
        st.write(
            "Una vista previa del dataset ayuda a comprender su estructura y el tipo de datos disponibles."
        )
        st.dataframe(df_loaded.head(50), use_container_width=True)
        return

    # Caso: todavía no subió archivo
    if uploaded_file is None:
        return

    # Caso: subió archivo
    try:
        df, _, errors = load_and_validate(uploaded_file, min_rows=10)
    except ValueError as exc:
        add_operation_log("load_dataset", str(exc), status="error")
        st.error(str(exc))
        return

    if errors:
        for err in errors:
            st.error(err)
        add_operation_log("load_dataset", "Dataset inválido", status="warning")
        return

    # Aplicar truncado si estamos en modo cloud
    df, truncate_messages = truncate_dataset_if_needed(df)
    for msg in truncate_messages:
        st.warning(msg)

    if IS_CLOUD:
        st.caption(
            f"ℹ️ Modo cloud: límite de {CLOUD_MAX_ROWS:,} filas y {CLOUD_MAX_COLUMNS} columnas."
        )

    # Detectar columnas completamente vacías y permitir eliminarlas
    all_nan_cols = get_all_nan_columns(df)

    # df_view es lo que se muestra (si el usuario decide eliminar columnas vacías, se refleja al instante)
    df_view = df

    if all_nan_cols:
        cols_str = ", ".join(map(str, all_nan_cols))
        st.warning(f"Columnas completamente vacías detectadas: {cols_str}.")

        if learn:
            with st.expander("⚠️ ¿Qué significa este aviso?"):
                st.markdown(
                    "Este aviso indica que se detectaron **columnas completamente vacías**, es decir, columnas que no contienen ningún dato útil.\n\n"
                    "Estas columnas no aportan información al modelo y pueden eliminarse de forma segura para simplificar el dataset y mejorar el procesamiento."
                )

        drop_empty = st.checkbox(
            "Eliminar columnas vacías (recomendado)", value=True)
        if drop_empty:

            df_view = df.drop(columns=all_nan_cols)
            if df_view.empty or df_view.shape[1] == 0:
                st.error(
                    "El dataset quedó sin columnas después de eliminar las vacías.")
                return

    # Info básica (sobre df_view, que es lo que el usuario está viendo/confirmando)
    info = get_basic_info(df_view)
    _render_basic_info(info)

    if learn:
        with st.expander("📊 ¿Qué es esta información?"):
            st.markdown(
                "Este resumen muestra cuánta información tiene tu dataset:\n\n"
                "- **Filas**: cantidad de registros disponibles.\n"
                "- **Columnas**: variables o características de cada registro.\n"
                "- **Tamaño**: espacio que ocupa el dataset en memoria.\n\n"
                "Cuantos más datos relevantes haya, mayor suele ser el potencial del modelo para aprender."
            )

    # Warning de alto porcentaje de NaNs (sobre df_view)
    has_high_nan = _render_high_nan_warning(df_view, threshold=0.3)

    if learn and has_high_nan:
        with st.expander("⚠️ ¿Qué significa este aviso?"):
            st.markdown(
                "Este aviso indica que algunas columnas del dataset tienen un **alto porcentaje de valores faltantes (NaNs)**.\n\n"
                "Cuando una columna tiene muchos NaNs, aporta poca información y puede afectar el entrenamiento del modelo si no se trata correctamente.\n\n"
                "Más adelante, estas columnas podrán eliminarse, completarse o transformarse según el caso."
            )

    st.subheader("Tipos de datos detectados (puedes modificarlos luego)")
    _render_dtypes_table(info["dtypes"])

    if learn:
        with st.expander("🧩 ¿Qué son los tipos de datos?"):
            st.markdown(
                "Los tipos de datos indican cómo está almacenada la información en cada columna y cómo puede ser utilizada por el modelo.\n\n"
                "En esta tabla podés encontrar principalmente:\n\n"
                "- **int64**: números enteros. Ejemplos: cantidad de habitaciones, pisos, años.\n"
                "- **float64**: números con decimales. Ejemplos: precios, promedios, porcentajes, superficies.\n"
                "- **object**: texto o categorías. Ejemplos: ciudad, tipo de propiedad, nombre de una categoría.\n\n"
                "Cada tipo de dato se trata de forma distinta durante el entrenamiento del modelo, por eso es importante identificarlos correctamente."
            )

    st.subheader("Vista previa del dataset")
    st.write("Una vista previa del dataset ayuda a comprender su estructura y el tipo de datos disponibles.")
    st.dataframe(df_view.head(50), use_container_width=True)

    if learn:
        st.write("¡Confirma la carga del dataset y avanza al siguiente paso!")

    if st.button("Confirmar carga"):
        # Si ya había un dataset cargado y sube otro, reseteamos el proyecto pero mantenemos metadata general
        if project.df_original is not None:
            reset_project(keep_metadata=True)
            project = get_project()

        # Guardamos lo que el usuario vio/decidió (df_view ya tiene el drop de vacías si corresponde)
        project.df_original = df_view
        project.update_metadata()

        add_operation_log(
            "load_dataset",
            f"Dataset cargado con {info['rows']} filas y {info['columns']} columnas.",
            status="success",
        )
        st.success("Dataset cargado y guardado en el proyecto.")

    if project.is_step_completed("load"):
        if st.button("Siguiente: Detección de Tipos"):
            st.switch_page("pages/2-Deteccion de tipos.py")


if __name__ == "__main__":
    main()
