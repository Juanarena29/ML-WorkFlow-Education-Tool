import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.ml.evaluator import (
    build_metrics_table,
    compute_confusion_matrix,
    compute_confusion_matrix_normalized,
    compute_residuals,
    compute_roc_curve,
    extract_feature_importance,
    get_train_test_split,
)
from src.utils.file_handler import save_model, save_project_config

from src.utils.session import check_step_access, get_project, initialize_session


def _render_metrics(metrics: dict, problem_type: str) -> None:
    if not metrics:
        st.info("No hay metricas disponibles.")
        return

    df_metrics = build_metrics_table(metrics)
    st.subheader("Tabla comparativa")
    st.dataframe(df_metrics, use_container_width=True)

    if problem_type == "regression":
        score_map = {
            "MAE – Error absoluto medio": "mae",
            "RMSE – Error cuadrático medio": "rmse",
            "R² – Capacidad explicativa": "r2",
        }
    else:
        score_map = {
            "Accuracy – Exactitud": "accuracy",
            "Precision – Confiabilidad de positivos": "precision",
            "Recall – Cobertura de positivos": "recall",
            "F1 Score – Balance Precision/Recall": "f1",
            "ROC AUC – Probabilidades": "roc_auc",
        }

    available_score_map = {
        label: col for label, col in score_map.items() if col in df_metrics.columns}

    if not available_score_map:
        return

    metric_label = st.selectbox(
        "Metrica para comparar",   options=list(available_score_map.keys()))
    metric_name = available_score_map[metric_label]
    fig = px.bar(
        df_metrics.sort_values(metric_name, ascending=False),
        x=metric_name,
        y="modelo",
        orientation="h",
        title=f"Comparacion por {metric_label}",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_model_details(models: dict, metrics: dict, project, learn) -> None:
    st.subheader("Detalles por modelo")
    if learn:
        with st.expander("📊 ¿Qué es el detalle por modelo?"):
            st.markdown(
                "Este apartado muestra los **valores exactos de las métricas** para cada modelo.\n\n"
                "A diferencia de los gráficos, acá podés comparar modelos de forma directa y objetiva.\n\n"
                "Usalo para confirmar cuál modelo rinde mejor según la métrica que elegiste."
            )
    for model_name, model in models.items():
        with st.expander(model_name):
            st.write("Metricas")
            model_metrics = metrics.get(model_name, {})
            st.json(model_metrics)

            if project.best_params.get(model_name):
                st.write("Mejores hiperparametros")
                st.json(project.best_params[model_name])

            importances = extract_feature_importance(
                model,
                project.get_numeric_features(),
                project.get_categorical_features(),
            )
            if importances is None or importances.empty:
                st.write("Este modelo no expone importancias.")
            else:
                st.write("Feature importance (top 20)")
                if learn:
                    with st.expander("🔍 ¿Qué significa la importancia de variables?"):
                        st.markdown(
                            "La **importancia de variables** indica qué columnas influyen más en las predicciones del modelo.\n\n"
                            "Variables más importantes tienen mayor impacto en el resultado final.\n\n"
                            "Esto ayuda a entender el modelo y a detectar qué datos son más relevantes."
                        )
                top = importances.head(20)
                fig = px.bar(
                    top.sort_values("importance", ascending=True),
                    x="importance",
                    y="feature",
                    orientation="h",
                )
                st.plotly_chart(fig, use_container_width=True)


def _plot_confusion_matrix(y_true, y_pred) -> None:
    cm, labels = compute_confusion_matrix(y_true, y_pred)
    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        text_auto=True,
        labels={"x": "Predicho", "y": "Real"},
        title="Matriz de confusion",
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_confusion_matrix_normalized(y_true, y_pred) -> None:
    cm, labels = compute_confusion_matrix_normalized(y_true, y_pred)
    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        text_auto=".2f",
        labels={"x": "Predicho", "y": "Real"},
        title="Matriz de confusion (normalizada)",
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_residuals(y_true, y_pred) -> None:
    residuals = compute_residuals(y_true, y_pred)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode="markers"))
    fig.update_layout(
        title="Residuos vs Prediccion",
        xaxis_title="Prediccion",
        yaxis_title="Residuo",
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_roc_curve(y_true, pipeline, X_test) -> None:
    if y_true.nunique() != 2:
        return

    if hasattr(pipeline, "predict_proba"):
        y_score = pipeline.predict_proba(X_test)[:, 1]
    elif hasattr(pipeline, "decision_function"):
        y_score = pipeline.decision_function(X_test)
    else:
        return

    fpr, tpr, roc_auc = compute_roc_curve(y_true, y_score)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                  name=f"AUC={roc_auc:.3f}"))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Base", line=dict(dash="dash")))
    fig.update_layout(
        title="Curva ROC",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    initialize_session()

    if not check_step_access("training"):
        return

    project = get_project()
    if not project.metrics:
        st.error("No hay resultados para mostrar.")
        return

    learn = project.ui_mode == "learn"

    st.title("Resultados y Comparación")
    if learn:
        with st.expander("📌 ¿Qué estoy viendo en esta pantalla?"):
            st.markdown(
                "Acá podés **comparar modelos** y entender cuál funciona mejor para tu dataset.\n\n"
                "Vas a ver:\n"
                "- Una comparación por métricas (números).\n"
                "- Gráficos por cada modelo (para ver aciertos/errores).\n"
                "- La opción de guardar el modelo que elijas."
            )
    _render_metrics(project.metrics, project.problem_type or "classification")
    if learn:
        with st.expander("📏 ¿Cómo interpreto las métricas?"):
            st.markdown(
                "Las métricas son una forma de resumir qué tan bien predice el modelo.\n\n"
                "- En general, **más alto es mejor** (por ejemplo: Accuracy, F1, R²).\n"
                "- En métricas de error, **más bajo es mejor** (por ejemplo: MAE, RMSE).\n\n"
                "Lo importante es comparar modelos usando **la misma métrica**."
            )
    st.markdown("---")
    # Validaciones mínimas (evita excepciones generales)
    if project.df_limpio is None or project.df_limpio.empty:
        st.error("No hay dataset limpio disponible para reconstruir el split.")
        _render_model_details(project.trained_models,
                              project.metrics, project, learn)
        return

    if not project.target_column:
        st.error("No se encontró la columna target del proyecto.")
        _render_model_details(project.trained_models,
                              project.metrics, project, learn)
        return

    if not project.train_test_split_config:
        st.error("No se encontró la configuración de train/test split.")
        _render_model_details(project.trained_models,
                              project.metrics, project, learn)
        return

    if learn:
        with st.expander("📊 ¿Para qué sirven los gráficos por modelo?"):
            st.markdown(
                "Los gráficos ayudan a ver el comportamiento real del modelo, no solo un número.\n\n"
                "- En **clasificación**, muestran qué clases se confunden entre sí.\n"
                "- En **regresión**, muestran qué tan lejos están las predicciones de los valores reales.\n\n"
                "Si dos modelos tienen métricas parecidas, los gráficos suelen ayudarte a decidir mejor."
            )

    st.subheader("Gráficos por modelo")

    if learn and project.problem_type == "classification":
        with st.expander("🧩 ¿Qué es la matriz de confusión y la curva ROC?"):
            st.markdown(
                "**Matriz de confusión**: muestra aciertos y errores por clase.\n"
                "Ayuda a ver en qué casos el modelo se equivoca más.\n\n"
                "**Curva ROC**: evalúa qué tan bien el modelo separa las clases usando probabilidades.\n"
                "Es útil cuando importa distinguir positivos y negativos."
            )

    # Reconstrucción del split con manejo de error más específico
    try:
        X_train, X_test, y_train, y_test = get_train_test_split(
            project.df_limpio,
            project.target_column,
            project.problem_type or "classification",
            project.train_test_split_config,
        )
    except (KeyError, ValueError, TypeError) as exc:
        st.error(f"No se pudo reconstruir el split: {exc}")
        _render_model_details(project.trained_models,
                              project.metrics, project, learn)
        return

    if not project.trained_models:
        st.warning("No hay modelos entrenados para graficar.")
        _render_model_details(project.trained_models,
                              project.metrics, project, learn)
        return

    for model_name, pipeline in project.trained_models.items():
        with st.expander(f"{model_name} - gráficos"):
            # Predicción con fallback controlado
            try:
                y_pred = pipeline.predict(X_test)
            except (ValueError, TypeError) as exc:
                st.error(f"No se pudo predecir con {model_name}: {exc}")
                continue

            y_test_plot = y_test

            # Normalización/decodificación solo para clasificación, evitando excepciones generales
            if project.problem_type == "classification" and project.target_label_encoder is not None:
                y_pred_is_num = pd.api.types.is_numeric_dtype(
                    pd.Series(y_pred))
                y_test_is_num = pd.api.types.is_numeric_dtype(y_test_plot)

                # Si predicción es numérica pero y_test es texto/categoría, intentar invertir predicción
                if y_pred_is_num and not y_test_is_num:
                    try:
                        y_pred = project.target_label_encoder.inverse_transform(
                            pd.Series(y_pred).astype(int)
                        )
                    except (ValueError, TypeError):
                        # Si no se puede invertir, intentamos transformar y_test a numérico para comparar
                        try:
                            y_test_plot = pd.Series(
                                project.target_label_encoder.transform(
                                    y_test_plot),
                                index=y_test_plot.index,
                            )
                        except (ValueError, TypeError):
                            # Si tampoco se puede, seguimos con lo que haya (sin romper)
                            pass

            if project.problem_type == "classification":
                if learn:
                    with st.expander("🧩 ¿Qué es la matriz de confusión y la curva ROC?"):
                        st.markdown(
                            "**Matriz de confusión**: muestra aciertos y errores por clase.\n"
                            "Te ayuda a ver, por ejemplo, si el modelo confunde 'A' con 'B'.\n\n"
                            "**Curva ROC**: es una forma de evaluar modelos que trabajan con probabilidades.\n"
                            "Suele ser útil cuando querés separar bien positivos y negativos."
                        )
                _plot_confusion_matrix(y_test_plot, y_pred)
                _plot_confusion_matrix_normalized(y_test_plot, y_pred)

                # ROC necesita y_test numérico si hay encoder
                y_test_roc = y_test_plot
                if project.target_label_encoder is not None and not pd.api.types.is_numeric_dtype(y_test_roc):
                    try:
                        y_test_roc = pd.Series(
                            project.target_label_encoder.transform(y_test_roc),
                            index=y_test_roc.index,
                        )
                    except (ValueError, TypeError):
                        # Si no se puede transformar, evitamos romper el flujo y omitimos ROC
                        st.info(
                            "No se pudo preparar el target para ROC AUC en este modelo.")
                        continue

                try:
                    _plot_roc_curve(y_test_roc, pipeline, X_test)
                except (ValueError, TypeError) as exc:
                    st.info(
                        f"No se pudo generar la curva ROC para {model_name}: {exc}")
            else:
                _plot_residuals(y_test, y_pred)

    _render_model_details(project.trained_models,
                          project.metrics, project, learn)

    if project.trained_models:
        model_options = list(project.trained_models.keys())
        selected_model = st.selectbox(
            "Selecciona un modelo para guardar",
            options=model_options,
        )
        if learn:
            with st.expander("💾 ¿Qué significa guardar un modelo?"):
                st.markdown(
                    "Guardar un modelo significa conservar el modelo ya entrenado para usarlo después sin volver a entrenar.\n\n"
                    "Por ejemplo, podés cargarlo más adelante para hacer predicciones con nuevos datos."
                )
    cols = st.columns(3)
    if project.trained_models:
        with cols[1]:
            if st.button("Guardar modelo seleccionado"):
                model = project.trained_models[selected_model]
                filename = f"{selected_model}_{project.problem_type}"
                try:
                    path = save_model(model, filename)
                except (OSError, ValueError, TypeError) as exc:
                    st.error(f"No se pudo guardar el modelo: {exc}")
                else:
                    st.success(f"Modelo guardado en {path}")
    with cols[2]:
        if st.button("Guardar configuración del proyecto"):
            try:
                path = save_project_config(project.to_dict())
            except (OSError, ValueError, TypeError) as exc:
                st.error(f"No se pudo guardar la configuración: {exc}")
            else:
                st.success(f"Configuración guardada en {path}")
    with cols[0]:
        if st.button("¡PRUEBA TU MODELO!"):
            st.switch_page("pages/7-Predicciones.py")


if __name__ == "__main__":
    main()
