"""
Componentes de UI para la pagina 5 - Entrenamiento.

Cada funcion renderiza una seccion concreta de la pagina,
recibiendo los datos como parametros (sin acceder al estado global).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from src.ml.model_trainer import train_models
from src.ml.models_config import (
    get_models_for_problem_type,
    get_param_grids,
    get_scoring_options,
)
from src.savings.project_updates import save_training_results
from src.ui.learn_explanations import (
    render_learn_five_gridsearch_explanation,
    render_learn_five_scoring_explanation,
    render_learn_five_select_explanation,
    render_learn_five_stratify_explanation,
    render_learn_five_training_explanation,
    render_learn_five_traintest_explanation,
)
from src.utils.constants import get_max_cv_folds, get_max_rows
from src.utils.session import MLProject, add_operation_log


# ---------------------------------------------------------------------------
# Secciones de UI
# ---------------------------------------------------------------------------


def render_training_header(problem_type: str, learn: bool) -> None:
    """Titulo, explicación Learn y badge de tipo de problema."""
    st.title("Entrenamiento de Modelos")

    if learn:
        render_learn_five_training_explanation()

    st.info(f"Tipo de problema: {problem_type}")


# ---------------------------------------------------------------------------
# Seleccion de modelos
# ---------------------------------------------------------------------------


def render_model_selection(
    problem_type: str,
    learn: bool,
) -> Dict[str, Any]:
    """Renderiza checkboxes de selección de modelos y devuelve los elegidos."""
    st.subheader("Seleccion de modelos")

    if learn:
        render_learn_five_select_explanation()

    available_models = get_models_for_problem_type(problem_type)
    selected: Dict[str, Any] = {}
    for name, model in available_models.items():
        if st.checkbox(name, value=False, key=f"model_{name}"):
            selected[name] = model

    if not selected:
        st.warning("Selecciona al menos un modelo para entrenar.")

    return selected


# ---------------------------------------------------------------------------
# Train/Test split
# ---------------------------------------------------------------------------


def render_split_config(
    problem_type: str,
    learn: bool,
) -> Dict[str, Any]:
    """Renderiza controles de train/test split y devuelve la config."""
    st.subheader("Configuracion de train/test split")

    if learn:
        render_learn_five_traintest_explanation()

    test_size: float = st.slider(
        "Proporcion de test", 0.1, 0.4, 0.2, 0.05,
    )
    random_state: int = int(
        st.number_input(
            "Random state", min_value=0, max_value=9999, value=22,
        )
    )

    stratify = True
    if problem_type == "classification":
        stratify = st.checkbox("Usar stratify", value=True)
        if learn:
            render_learn_five_stratify_explanation()

    return {
        "test_size": test_size,
        "random_state": random_state,
        "stratify": stratify,
    }


# ---------------------------------------------------------------------------
# GridSearchCV
# ---------------------------------------------------------------------------


def render_gridsearch_config(
    problem_type: str,
    runtime_mode: str,
    learn: bool,
) -> Dict[str, Any]:
    """Renderiza controles de GridSearchCV y devuelve la config."""
    st.subheader("GridSearchCV")

    if learn:
        render_learn_five_gridsearch_explanation()

    use_gridsearch: bool = st.checkbox("Usar GridSearchCV", value=False)

    if not use_gridsearch:
        st.caption(
            "GridSearchCV desactivado: entrenamiento más rápido "
            "(ideal para una primera prueba)."
        )

    max_folds = get_max_cv_folds()
    cv_folds: int = int(
        st.number_input(
            "CV folds",
            min_value=2,
            max_value=max_folds,
            value=min(5, max_folds),
            disabled=not use_gridsearch,
        )
    )

    if runtime_mode == "demo" and use_gridsearch:
        st.caption(
            f"⚠️ Modo demo: máximo {max_folds} folds para evitar timeouts."
        )

    grid_preset = "ligero"
    if use_gridsearch:
        grid_preset = st.selectbox(
            "Preset de grid",
            options=["ligero", "medio", "completo"],
            index=0,
        )

    return {
        "use_gridsearch": use_gridsearch,
        "cv_folds": cv_folds,
        "grid_preset": grid_preset,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def render_scoring_selector(
    problem_type: str,
    target_nunique: int,
    learn: bool,
) -> Optional[str]:
    """Renderiza selector de scoring y devuelve el valor sklearn elegido."""
    scoring_map = get_scoring_options(problem_type, target_nunique)
    scoring_label = st.selectbox(
        "Scoring", options=list(scoring_map.keys()), index=0,
    )

    if learn:
        render_learn_five_scoring_explanation()

    return scoring_map[scoring_label]


# ---------------------------------------------------------------------------
# Ejecucion del entrenamiento
# ---------------------------------------------------------------------------


def _validate_demo_limits(
    df: pd.DataFrame,
    runtime_mode: str,
) -> bool:
    """Verifica que el dataset no exceda los limites de modo demo.
    Retorna True si es valido, False si excede limites."""
    if runtime_mode != "demo":
        return True

    max_rows = get_max_rows()
    if len(df) > max_rows:
        st.error(
            f"Modo demo: el dataset tiene {len(df):,} filas, "
            f"pero el máximo permitido es {max_rows:,}. "
            "Reduce el dataset antes de entrenar."
        )
        return False
    return True


def handle_training(
    project: MLProject,
    df: pd.DataFrame,
    selected_models: Dict[str, Any],
    split_config: Dict[str, Any],
    gridsearch_config: Dict[str, Any],
    scoring: Optional[str],
) -> None:
    """Renderiza el boton de entrenar y ejecuta el entrenamiento."""
    if not st.button("Entrenar modelos"):
        return

    if not _validate_demo_limits(df, project.runtime_mode):
        return

    problem_type = project.problem_type or "classification"
    param_grids = get_param_grids(
        problem_type, preset=gridsearch_config["grid_preset"],
    )

    try:
        with st.spinner("Entrenando modelos..."):
            trained, metrics, best_params, target_encoder = train_models(
                df=df,
                target_column=project.target_column,
                numeric_features=project.get_numeric_features(),
                categorical_features=project.get_categorical_features(),
                models=selected_models,
                problem_type=problem_type,
                test_size=split_config["test_size"],
                random_state=split_config["random_state"],
                stratify=split_config["stratify"],
                use_gridsearch=gridsearch_config["use_gridsearch"],
                param_grids=param_grids,
                cv=gridsearch_config["cv_folds"],
                scoring=scoring if gridsearch_config["use_gridsearch"] else None,
            )
    except MemoryError as exc:
        add_operation_log("train_models", str(exc), status="error")
        st.error(
            f"⚠️ Memoria insuficiente: {exc}\n\n"
            "Intenta reducir el dataset, usar menos modelos o desactivar GridSearchCV."
        )
        return
    except ValueError as exc:
        add_operation_log("train_models", str(exc), status="error")
        st.error(f"Error en los datos: {exc}")
        return
    except Exception as exc:
        add_operation_log("train_models", str(exc), status="error")
        st.error(f"Error inesperado durante el entrenamiento: {exc}")
        return

    save_training_results(
        project=project,
        trained_models=trained,
        metrics=metrics,
        best_params=best_params,
        target_encoder=target_encoder,
        split_config=split_config,
    )

    st.session_state.confirmations["training_started"] = True
    add_operation_log(
        "train_models",
        f"Modelos entrenados: {', '.join(trained.keys())}.",
        status="success",
    )
    st.success("Entrenamiento completado.")
