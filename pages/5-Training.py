import streamlit as st

from src.ml.model_trainer import train_models
from src.ml.models_config import get_models_for_problem_type, get_param_grids
from src.utils.constants import IS_CLOUD, get_max_cv_folds
from src.utils.session import (
    add_operation_log,
    check_step_access,
    get_project,
    initialize_session,
)


def _render_model_selection(models: dict) -> dict:
    selected = {}
    for name, model in models.items():
        # checkbox por modelo para activar/desactivar
        enabled = st.checkbox(name, value=False, key=f"model_{name}")
        if enabled:
            selected[name] = model
    return selected


def main() -> None:
    initialize_session()

    if not check_step_access("cleaning"):
        return

    project = get_project()
    df = project.df_limpio

    learn = project.ui_mode == "learn"

    validation_errors = project.validate_for_training()
    if validation_errors:
        for err in validation_errors:
            st.error(err)
        return

    st.title("Entrenamiento de Modelos")

    if learn:
        with st.expander("🚀 ¿Qué pasa cuando entreno modelos?"):
            st.markdown(
                "Entrenar significa que la app va a **aprender patrones** a partir de tus datos para poder predecir el target.\n\n"
                "Para evaluar si el modelo funciona bien, el dataset se divide en dos partes:\n"
                "- **Train**: donde aprende.\n"
                "- **Test**: donde se prueba con datos que no vio.\n\n"
                "Al final vas a ver métricas y podrás comparar modelos para elegir el mejor."
            )

    st.info(f"Tipo de problema: {project.problem_type}")

    available_models = get_models_for_problem_type(
        project.problem_type or "classification")

    st.subheader("Seleccion de modelos")
    if learn:
        with st.expander("🤖 ¿Qué significa seleccionar modelos?"):
            st.markdown(
                "Un **modelo** es una forma distinta de aprender y hacer predicciones.\n\n"
                "Podés entrenar varios para comparar resultados. Si estás empezando, lo más simple es elegir **1 o 2 modelos** "
                "y después probar más.\n\n"
                "La app se encarga de aplicar el preprocesamiento necesario y evaluar cada modelo con el mismo criterio."
            )

    selected_models = _render_model_selection(available_models)

    if not selected_models:
        st.warning("Selecciona al menos un modelo para entrenar.")
        return

    st.subheader("Configuracion de train/test split")
    if learn:
        with st.expander("🧪 ¿Qué es el train/test split?"):
            st.markdown(
                "Este ajuste define qué parte de tus datos se usa para **probar** el modelo.\n\n"
                "- **Proporción de test**: por ejemplo 0.2 significa 20% test y 80% train.\n"
                "- **Random state**: cambia cómo se mezclan y separan los datos. Sirve para poder repetir resultados.\n\n"
                "Recomendación para empezar: **test = 0.2** y dejar el random state como está."
            )

    test_size = st.slider("Proporcion de test", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input(
        "Random state", min_value=0, max_value=9999, value=22)
    stratify = True
    if project.problem_type == "classification":
        stratify = st.checkbox("Usar stratify", value=True)
        if learn and project.problem_type == "classification":
            with st.expander("⚖️ ¿Qué hace 'stratify'?"):
                st.markdown(
                    "En clasificación, **stratify** intenta que el conjunto de train y test tengan proporciones similares de cada clase.\n\n"
                    "Suele ser recomendable cuando las clases están desbalanceadas (por ejemplo, muchos 'No' y pocos 'Sí')."
                )

    st.subheader("GridSearchCV")
    if learn:
        with st.expander("🔍 ¿Qué es GridSearchCV y cuándo conviene usarlo?"):
            st.markdown(
                "GridSearchCV prueba distintas configuraciones del modelo para encontrar una que funcione mejor.\n\n"
                "- Si está **desactivado**: el modelo entrena más rápido (recomendado para una primera prueba).\n"
                "- Si lo activás: tarda más, pero puede mejorar el resultado.\n\n"
                "Presets:\n"
                "- **Ligero**: rápido (ideal para empezar).\n"
                "- **Medio**: balance.\n"
                "- **Completo**: más lento y exhaustivo."
            )

    use_gridsearch = st.checkbox("Usar GridSearchCV", value=False)

    if not use_gridsearch:
        st.caption(
            "GridSearchCV desactivado: entrenamiento más rápido (ideal para una primera prueba).")

    max_folds = get_max_cv_folds()
    cv_folds = st.number_input(
        "CV folds", min_value=2, max_value=max_folds, value=min(5, max_folds), disabled=not use_gridsearch)

    if IS_CLOUD and use_gridsearch:
        st.caption(
            f"⚠️ Modo cloud: máximo {max_folds} folds para evitar timeouts.")

    grid_preset = "ligero"
    if use_gridsearch:
        grid_preset = st.selectbox(
            "Preset de grid",
            options=["ligero", "medio", "completo"],
            index=0,
        )
    if project.problem_type == "regression":
        scoring_map = {
            "R²": "r2",
            "MAE": "neg_mean_absolute_error",
            "RMSE": "neg_root_mean_squared_error",
        }
    else:
        scoring_map = {
            "Accuracy": "accuracy",
            "F1": "f1_weighted",
            "ROC AUC": "roc_auc",
        }
    scoring_label = st.selectbox(
        "Scoring", options=list(scoring_map.keys()), index=0)
    if learn:
        with st.expander("📏 ¿Qué significa 'scoring'?"):
            st.markdown(
                "El **scoring** es la regla que usa la app para decidir qué resultado es “mejor”.\n\n"
                "Elegí uno según tu objetivo:\n"
                "- En **clasificación**: *accuracy* (simple), *f1_weighted* (mejor si hay desbalance), *roc_auc* (útil para probabilidades).\n"
                "- En **regresión**: *r2* (qué tan bien explica), *MAE/RMSE* (error promedio).\n\n"
                "Si no estás seguro: empezá con la opción que aparece por defecto."
            )

    if st.button("Entrenar modelos"):
        try:
            with st.spinner("Entrenando modelos..."):
                trained, metrics, best_params, target_encoder = train_models(
                    df=df,
                    target_column=project.target_column,
                    numeric_features=project.get_numeric_features(),
                    categorical_features=project.get_categorical_features(),
                    models=selected_models,
                    problem_type=project.problem_type or "classification",
                    test_size=test_size,
                    random_state=int(random_state),
                    stratify=stratify,
                    use_gridsearch=use_gridsearch,
                    param_grids=get_param_grids(
                        project.problem_type or "classification", preset=grid_preset
                    ),
                    cv=int(cv_folds),
                    scoring=scoring_label if use_gridsearch else None,
                )
        except Exception as exc:
            add_operation_log("train_models", str(exc), status="error")
            st.error(str(exc))
            return

        project.trained_models = trained
        project.metrics = metrics
        project.best_params = best_params
        project.target_label_encoder = target_encoder
        project.train_test_split_config = {
            "test_size": test_size,
            "random_state": int(random_state),
            "stratify": stratify,
        }
        project.update_metadata()

        st.session_state.confirmations["training_started"] = True
        add_operation_log(
            "train_models",
            f"Modelos entrenados: {', '.join(trained.keys())}.",
            status="success",
        )
        st.success("Entrenamiento completado.")

    if project.is_step_completed("training"):
        if st.button("Siguiente: Resultados"):
            st.switch_page("pages/6-Results.py")


if __name__ == "__main__":
    main()
