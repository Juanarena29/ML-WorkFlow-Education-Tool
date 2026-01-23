import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data.loader import load_dataset
from src.ml.evaluator import (
    compute_confusion_matrix,
    compute_confusion_matrix_normalized,
)
from src.utils.file_handler import list_saved_models, load_model
from src.utils.session import add_operation_log, check_step_access, get_project, initialize_session


def _select_model(project) -> tuple[object | None, str | None]:
    source = st.radio(
        "Origen del modelo",
        options=["En sesion", "Guardado"],
        horizontal=True,
    )

    model = None
    model_name = None

    if source == "En sesion":
        if not project.trained_models:
            st.info("No hay modelos en sesion. Entrena modelos o carga uno guardado.")
        else:
            model_name = st.selectbox(
                "Modelo en sesion", options=list(project.trained_models.keys())
            )
            model = project.trained_models[model_name]
    else:
        saved_models = list_saved_models()
        if not saved_models:
            st.info("No hay modelos guardados en la carpeta models.")
        else:
            selected_file = st.selectbox(
                "Modelo guardado", options=saved_models
            )
            if st.button("Cargar modelo guardado"):
                try:
                    loaded_model = load_model(selected_file)
                    st.session_state.prediction_model = loaded_model
                    st.session_state.prediction_model_name = selected_file.replace(
                        ".pkl", "")
                    add_operation_log(
                        "load_model",
                        f"Modelo cargado para prediccion: {selected_file}.",
                        status="success",
                    )
                    st.success("Modelo cargado en la sesion.")
                except Exception as exc:
                    add_operation_log("load_model", str(exc), status="error")
                    st.error(str(exc))

            if "prediction_model" in st.session_state:
                model = st.session_state.prediction_model
                model_name = st.session_state.get(
                    "prediction_model_name", "modelo_guardado")
                st.success(f"Modelo listo: {model_name}.")

    return model, model_name


def _build_prediction_output(
    df: pd.DataFrame,
    preds,
    proba,
) -> pd.DataFrame:
    output = df.copy()
    output["prediccion"] = preds

    if proba is None:
        return output

    if hasattr(proba, "ndim") and proba.ndim == 2:
        if proba.shape[1] == 2:
            output["probabilidad"] = proba[:, 1]
        else:
            for idx in range(proba.shape[1]):
                output[f"proba_{idx}"] = proba[:, idx]

    return output


def main() -> None:
    initialize_session()

    if not check_step_access("types"):
        return

    project = get_project()
    
    learn = project.ui_mode == "learn"
    
    st.title("Prediccion")

    if learn:
        with st.expander("📌 ¿Qué hacemos en esta etapa?"):
            st.markdown(
                "En esta página vas a usar un **modelo ya entrenado** para generar predicciones sobre **datos nuevos**.\n\n"
                "Podés subir un CSV con nuevas filas y la app calculará el valor predicho para cada una. "
                "Si además incluís el **target real**, también vas a ver un gráfico comparando **real vs predicho**."
            )

    feature_cols = project.get_feature_columns()
    if not feature_cols:
        st.error("No hay features definidas. Completa la deteccion de tipos primero.")
        return

    if learn:
        with st.expander("🧠 ¿Qué modelo se usa para predecir?"):
            st.markdown(
                "Para hacer predicciones necesitás un **modelo entrenado**.\n\n"
                "- Podés usar un modelo que entrenaste recién en esta app (queda disponible en la sesión).\n"
                "- O podés **cargar un modelo guardado** (por ejemplo un `.pkl/.joblib`) para reutilizarlo.\n\n"
                "La recomendación es usar un modelo entrenado con datos similares a los que vas a predecir."
            )

    model, model_name = _select_model(project)

    st.subheader("Dataset para prediccion")
    uploaded_file = st.file_uploader(
        "Sube un CSV con filas nuevas", type=["csv"])

    if learn:
        with st.expander("📄 ¿Cómo debe ser el CSV de entrada?"):
            st.markdown(
                "El archivo debe tener **las mismas columnas features** que se usaron para entrenar el modelo.\n\n"
                "Tenés dos formas de usar esta página:\n"
                "1. **Sin target**: subís solo las features y la app **solo predice**.\n"
                "2. **Con target**: subís las features **y también** la columna target real. "
                "Además de predecir, la app muestra un gráfico comparando **predicción vs valor real** "
                "(o matriz de confusión en clasificación).\n\n"
                "Si faltan columnas requeridas, la app te avisará antes de predecir."
            )

    if uploaded_file is None:
        return

    try:
        df = load_dataset(uploaded_file)
    except ValueError as exc:
        add_operation_log("load_dataset_prediction", str(exc), status="error")
        st.error(str(exc))
        return

    st.dataframe(df.head(50), use_container_width=True)

    
    has_target = st.checkbox(
        "El archivo posee el dato a predecir (target)",
        value=False,
    )
    y_true = None
    if has_target:
        if not project.target_column:
            st.error("No hay columna target definida en el proyecto.")
            return
        if project.target_column not in df.columns:
            st.error("El CSV no contiene la columna target indicada.")
            return
        y_true = df[project.target_column]
        df = df.drop(columns=[project.target_column])

    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        st.error(f"Faltan columnas requeridas: {', '.join(missing)}")
        return

    if model is None:
        st.warning("Selecciona y carga un modelo antes de predecir.")
        return

    if st.button("Generar predicciones"):
        try:
            X = df[feature_cols]
            preds = model.predict(X)
            y_true_plot = y_true
            if project.problem_type == "classification" and project.target_label_encoder:
                if pd.api.types.is_numeric_dtype(pd.Series(preds)) and (
                    y_true is not None
                    and not pd.api.types.is_numeric_dtype(y_true)
                ):
                    try:
                        preds = project.target_label_encoder.inverse_transform(
                            pd.Series(preds).astype(int)
                        )
                    except Exception:
                        try:
                            y_true_plot = pd.Series(
                                project.target_label_encoder.transform(y_true),
                                index=y_true.index,
                            )
                        except Exception:
                            pass
            proba = None
            if project.problem_type == "classification" and hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
            output = _build_prediction_output(df, preds, proba)
        except Exception as exc:
            add_operation_log("predict", str(exc), status="error")
            st.error(str(exc))
            return

        st.success("Predicciones generadas.")
        st.dataframe(output.head(50), use_container_width=True)

        if y_true is not None:
            if learn and y_true is not None:
                with st.expander("📊 ¿Cómo interpretar el gráfico?"):
                    st.markdown(
                        "**Regresión:** el gráfico muestra *Real (eje X)* vs *Predicción (eje Y)*.\n"
                        "- Cuanto más cerca estén los puntos de la **línea ideal**, mejor predice el modelo.\n"
                        "- Puntos muy alejados indican casos donde el modelo se equivoca más.\n\n"
                        "**Clasificación:** se muestra una **matriz de confusión**.\n"
                        "- La diagonal son aciertos (predijo la clase correcta).\n"
                        "- Fuera de la diagonal son errores (confusiones entre clases)."
                    )

            if project.problem_type == "classification":
                cm, labels = compute_confusion_matrix(y_true_plot, preds)
                fig = px.imshow(
                    cm,
                    x=labels,
                    y=labels,
                    text_auto=True,
                    labels={"x": "Predicho", "y": "Real"},
                    title="Matriz de confusion",
                )
                st.plotly_chart(fig, use_container_width=True)

                cm_norm, labels = compute_confusion_matrix_normalized(
                    y_true_plot, preds
                )
                fig_norm = px.imshow(
                    cm_norm,
                    x=labels,
                    y=labels,
                    text_auto=".2f",
                    labels={"x": "Predicho", "y": "Real"},
                    title="Matriz de confusion (normalizada)",
                )
                st.plotly_chart(fig_norm, use_container_width=True)
            elif pd.api.types.is_numeric_dtype(y_true):
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=y_true,
                        y=preds,
                        mode="markers",
                        name="Prediccion vs real",
                    )
                )
                min_val = min(y_true.min(), pd.Series(preds).min())
                max_val = max(y_true.max(), pd.Series(preds).max())
                fig.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode="lines",
                        name="Linea ideal",
                    )
                )
                fig.update_layout(
                    title="Prediccion vs valor real",
                    xaxis_title="Real",
                    yaxis_title="Prediccion",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("El target no es numerico. No se puede graficar.")

        csv_data = output.to_csv(index=False)
        safe_name = model_name or "modelo"
        st.download_button(
            "Descargar predicciones",
            data=csv_data,
            file_name=f"predicciones_{safe_name}.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
