import pandas as pd
import streamlit as st

from src.data.analyzer import analyze_missing_values
from src.data.cleaner import (
    apply_cleaning_config,
    list_columns_with_nans,
    suggest_imputation_method,
)
from src.utils.constants import IMPUTATION_METHODS_CATEGORICAL, IMPUTATION_METHODS_NUMERIC
from src.utils.session import (
    add_operation_log,
    check_step_access,
    get_project,
    initialize_session,
)


def _build_cleaning_config(df: pd.DataFrame, column_types: dict) -> dict:
    config = {"imputations": {}, "drop_duplicates": False}

    columns_with_nans = list_columns_with_nans(df)
    for col in columns_with_nans:

        col_type = column_types.get(col)
        if col_type == "numeric":
            method_key = st.selectbox(
                f"{col} (numerica)",
                options=list(IMPUTATION_METHODS_NUMERIC.keys()),
                key=f"impute_{col}",
            )
            method = IMPUTATION_METHODS_NUMERIC[method_key]
        else:
            method_key = st.selectbox(
                f"{col} (categorica)",
                options=list(IMPUTATION_METHODS_CATEGORICAL.keys()),
                key=f"impute_{col}",
            )
            method = IMPUTATION_METHODS_CATEGORICAL[method_key]

        value = None
        if method == "constant":
            value = st.text_input(
                f"Valor constante para {col}",
                key=f"constant_{col}",
            )

        config["imputations"][col] = {"method": method, "value": value}

    return config


def _show_suggestions(df: pd.DataFrame, column_types: dict) -> None:
    missing = analyze_missing_values(df)
    if missing.empty:
        return

    for _, row in missing.iterrows():
        col = row["columna"]
        nan_pct = row["porcentaje_nans"]
        col_type = column_types.get(col)

        suggestion = suggest_imputation_method(df[col], col_type)

        st.caption(
            f"Sugerencia para '{col}': {suggestion}. NaNs: {nan_pct:.1f}%"
        )


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
        with st.expander("📌 ¿Qué estamos haciendo en esta etapa?"):
            st.markdown(
                "En este paso vas a **limpiar y preparar** el dataset antes de entrenar modelos.\n\n"
                "Acá definís qué hacer con **valores faltantes (NaNs)** y **filas duplicadas**.\n\n"
                "Una buena limpieza suele mejorar la calidad del entrenamiento y evita errores más adelante."
            )
    else:
        st.write("Define como tratar valores faltantes y duplicados.")

    missing_summary = analyze_missing_values(df)
    if missing_summary.empty:
        st.info("No se detectaron valores faltantes.")
    else:
        st.subheader("Resumen de NaNs")
        if learn and not missing_summary.empty:
            with st.expander("❓ ¿Qué son los NaNs y por qué importan?"):
                st.markdown(
                    "**NaN** significa *dato faltante* (celda vacía).\n\n"
                    "El modelo no puede aprender correctamente si hay muchos valores faltantes sin tratar.\n\n"
                    "Este resumen te ayuda a ver **qué columnas** tienen NaNs y **cuántos** para decidir el mejor tratamiento."
                )
        st.dataframe(missing_summary, use_container_width=True)

    _show_suggestions(df, project.column_types)
    if learn:
        with st.expander("💡 ¿Qué significan estas sugerencias?"):
            st.markdown(
                "Las sugerencias son recomendaciones automáticas basadas en tus datos y en los tipos de columnas.\n\n"
                "No son obligatorias: podés usarlas como guía y después ajustar la configuración a tu criterio."
            )

    duplicate_count = df.duplicated().sum()
    drop_duplicates = st.checkbox(
        f"Eliminar duplicados (detectados: {duplicate_count})",
        value=False,
    )
    if learn and duplicate_count > 0:
        with st.expander("🧾 ¿Qué es un duplicado? ¿Conviene eliminarlo?"):
            st.markdown(
                "Una fila duplicada es una fila **idéntica a otra** (mismos valores en todas las columnas).\n\n"
                "Eliminar duplicados suele ser recomendable porque evita que el modelo “cuente dos veces” el mismo registro.\n\n"
                "Si tus duplicados son intencionales (por ejemplo, eventos repetidos reales), entonces **no conviene eliminarlos**."
            )

    cleaning_config = _build_cleaning_config(df, project.column_types)
    cleaning_config["drop_duplicates"] = drop_duplicates

    st.subheader("Imputacion por columna con NaNs")
    if learn:
        with st.expander("🧩 ¿Qué significa imputar valores faltantes?"):
            st.markdown(
                "Imputar significa **reemplazar valores faltantes** por un valor razonable.\n\n"
                "Ejemplos comunes:\n\n"
                "- En columnas numéricas: usar **media** o **mediana**.\n"
                "- En columnas categóricas: usar el valor más frecuente (**moda**).\n"
                "- También podés completar con un valor fijo (ej. \"Desconocido\") o eliminar filas/columnas si corresponde.\n\n"
                "La mejor opción depende del significado de la columna."
            )

    preview_df = apply_cleaning_config(df, cleaning_config)
    st.subheader("Vista previa")
    st.dataframe(preview_df.head(50), use_container_width=True)

    if st.button("Aplicar tratamiento"):
        project.df_limpio = preview_df
        project.cleaning_config = cleaning_config
        project.update_metadata()
        st.session_state.confirmations["cleaning_confirmed"] = True

        add_operation_log(
            "apply_cleaning",
            f"Tratamiento aplicado. Filas: {len(preview_df)}.",
            status="success",
        )
        st.success("Tratamiento aplicado y dataset guardado.")

    if project.is_step_completed("cleaning"):
        if st.button("Siguiente: EDA"):
            st.switch_page("pages/4-EDA.py")


if __name__ == "__main__":
    main()
