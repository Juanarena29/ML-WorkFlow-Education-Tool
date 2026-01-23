import streamlit as st

from src.eda.visualizations import (
    correlation_fig,
    distribution_categorical_fig,
    distribution_numeric_fig,
    relations_scatter_fig,
    target_distribution_fig,
    target_relation_fig,
)
from src.utils.session import check_step_access, get_project, initialize_session


def main() -> None:
    initialize_session()

    if not check_step_access("cleaning"):
        return

    project = get_project()
    df = project.df_limpio

    learn = project.ui_mode == "learn"

    if df is None or df.empty:
        st.error("No hay dataset limpio disponible.")
        return

    st.title("EDA - Analisis Exploratorio")

    if learn:
        with st.expander("🔎 ¿Qué es el EDA y para qué sirve?"):
            st.markdown(
                "El **EDA (Análisis Exploratorio de Datos)** es un paso previo al Machine Learning.\n\n"
                "Sirve para entender rápidamente el dataset: cómo se distribuyen los datos, qué variables se relacionan "
                "y si hay patrones visibles.\n\n"
                "En esta app, el EDA es **automático**: vos elegís columnas y la herramienta genera visualizaciones útiles."
            )
    else:
        st.write(
            "Explora distribuciones, correlaciones y relaciones entre variables.")

    # listas de columnas segun tipos detectados
    numeric_cols = project.get_numeric_features()
    categorical_cols = project.get_categorical_features()

    tabs = st.tabs(["Distribuciones", "Correlaciones", "Relaciones", "Target"])

    def _ensure_selectbox_value(key: str, options: list) -> None:
        if key in st.session_state and st.session_state[key] not in options:
            del st.session_state[key]

    with tabs[0]:
        st.subheader("Distribuciones")
        if learn:
            with st.expander("📊 ¿Qué estoy viendo acá?"):
                st.markdown(
                    "Estos gráficos te muestran **cómo se repiten los valores** de una columna.\n\n"
                    "- En números: ves qué valores aparecen más y cuáles son poco comunes.\n"
                    "- En categorías: ves qué opciones son más frecuentes que otras.\n\n"
                    "Sirve para tener una idea rápida de los datos antes de entrenar un modelo."
                )
        if numeric_cols:
            _ensure_selectbox_value("eda_num_col", numeric_cols)
            num_col = st.selectbox(
                "Selecciona una columna numerica",
                options=numeric_cols,
                key="eda_num_col",
            )
            fig = distribution_numeric_fig(df, num_col)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay columnas numericas para mostrar histogramas.")

        if categorical_cols:
            _ensure_selectbox_value("eda_cat_col", categorical_cols)
            cat_col = st.selectbox(
                "Selecciona una columna categorica",
                options=categorical_cols,
                key="eda_cat_col",
            )
            fig = distribution_categorical_fig(df, cat_col)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay columnas categoricas para mostrar barplots.")

    with tabs[1]:
        st.subheader("Correlaciones")
        if learn and len(numeric_cols) >= 2:
            with st.expander("🔗 ¿Qué significa correlación?"):
                st.markdown(
                    "La **correlación** mide qué tan relacionados están dos valores numéricos.\n\n"
                    "- Cerca de **1**: suben juntos.\n"
                    "- Cerca de **-1**: uno sube cuando el otro baja.\n"
                    "- Cerca de **0**: no hay relación lineal clara.\n\n"
                    "Importante: correlación **no** significa que una variable cause a la otra."
                )
        if len(numeric_cols) < 2:
            st.info("Se necesitan al menos 2 columnas numericas para correlaciones.")
        else:
            fig = correlation_fig(df, numeric_cols)
            if fig is None:
                st.info("No hay suficientes datos numericos para correlacion.")
            else:
                st.plotly_chart(fig, use_container_width=True)
    with tabs[2]:
        st.subheader("Relaciones")
        if learn and len(numeric_cols) >= 2:
            with st.expander("📈 ¿Qué muestra este gráfico de relaciones?"):
                st.markdown(
                    "Este gráfico compara dos columnas numéricas:\n\n"
                    "- **Eje X** y **Eje Y** son variables numéricas.\n"
                    "- Cada punto es una fila del dataset.\n\n"
                    "La opción **Color** te permite separar los puntos por una categoría (por ejemplo: barrio, tipo, etc.) "
                    "para ver si se forman grupos o patrones."
                )

        if len(numeric_cols) < 2:
            st.info("Se necesitan al menos 2 columnas numericas para scatter plots.")
        else:
            _ensure_selectbox_value("eda_x_col", numeric_cols)
            _ensure_selectbox_value("eda_y_col", numeric_cols)
            x_col = st.selectbox(
                "Eje X", options=numeric_cols, key="eda_x_col")
            y_col = st.selectbox(
                "Eje Y", options=numeric_cols, key="eda_y_col")

            color_options = ["(sin color)"] + categorical_cols
            _ensure_selectbox_value("eda_color", color_options)
            color_choice = st.selectbox(
                "Color", options=color_options, key="eda_color")
            color_col = None if color_choice == "(sin color)" else color_choice

            fig = relations_scatter_fig(df, x_col, y_col, color_col)
            st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:

        st.subheader("Analisis del Target")
        if learn:
            with st.expander("🎯 ¿Qué es el target?"):
                st.markdown(
                    "El **target** es la variable que querés predecir.\n\n"
                    "Por ejemplo:\n"
                    "- Si querés predecir un **precio**, el target es numérico (regresión).\n"
                    "- Si querés predecir una **clase** (sí/no, categoría), es clasificación.\n\n"
                    "Este apartado te ayuda a ver cómo se comporta el target y cómo se relaciona con otras variables."
                )
        if not project.target_column:
            st.info("No se ha definido una columna target.")
        else:
            fig = target_distribution_fig(
                df,
                project.target_column,
                project.problem_type or "classification",
            )
            st.plotly_chart(fig, use_container_width=True)
            invertir = st.checkbox(
                "Invertir", value=False, key="eda_invert_target")
            if numeric_cols:
                _ensure_selectbox_value("eda_target_feature", numeric_cols)
                feature = st.selectbox(
                    "Relacionar target con feature numerica",
                    options=numeric_cols,
                    key="eda_target_feature",
                )
                if not invertir:
                    fig = target_relation_fig(
                        df, project.target_column, feature, "numeric"
                    )
                else:
                    fig = target_relation_fig(
                        df, feature, project.target_column, "numeric"
                    )
                st.plotly_chart(fig, use_container_width=True)
            elif categorical_cols:
                _ensure_selectbox_value(
                    "eda_target_feature_cat", categorical_cols)
                feature = st.selectbox(
                    "Relacionar target con feature categorica",
                    options=categorical_cols,
                    key="eda_target_feature_cat",
                )
                if learn and project.target_column:
                    with st.expander("🔁 ¿Qué hace 'Invertir'?"):
                        st.markdown(
                            "Cambia qué variable va en cada eje del gráfico.\n\n"
                            "No modifica los datos: solo cambia la forma de visualizar la relación."
                        )

                if not invertir:
                    fig = target_relation_fig(
                        df, project.target_column, feature, "categorical"
                    )
                else:
                    fig = target_relation_fig(
                        df, feature, project.target_column, "categorical"
                    )
                st.plotly_chart(fig, use_container_width=True)
    if project.is_step_completed("cleaning"):
        if st.button("Siguiente: Entrenamiento"):
            st.switch_page("pages/5-Entrenamiento.py")


if __name__ == "__main__":
    main()
