"""
Entrenamiento y evaluacion basica de modelos.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
import time
import math

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

from src.ml.pipeline_builder import create_full_pipeline


def train_models(
    df: pd.DataFrame,
    target_column: str,
    numeric_features: List[str],
    categorical_features: List[str],
    models: Dict[str, Any],
    problem_type: str,
    test_size: float = 0.2,
    random_state: int = 22,
    stratify: bool = True,
    use_gridsearch: bool = False,
    param_grids: Optional[Dict[str, Dict[str, List[Any]]]] = None,
    cv: int = 5,
    scoring: Optional[str] = None,
    n_jobs: int = -1,
) -> Tuple[
    Dict[str, Any],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, Any]],
    Optional[LabelEncoder],
]:
    """
    Entrena multiples modelos y devuelve metricas basicas.

    Args:
        df: DataFrame limpio.
        target_column: Columna target.
        numeric_features: Lista de features numericas.
        categorical_features: Lista de features categoricas.
        models: Diccionario nombre -> estimador.
        problem_type: 'regression' o 'classification'.
        test_size: Proporcion del set de test.
        random_state: Semilla para reproducibilidad.
        stratify: Si True, usa stratify para clasificacion.

    Returns:
        (trained_models, metrics, best_params) por modelo.
    """
    if df is None or df.empty:
        raise ValueError("Dataset vacio o invalido.")

    if target_column not in df.columns:
        raise ValueError(f"Target '{target_column}' no existe en el dataset.")

    X = df.drop(columns=[target_column])
    y = df[target_column]
    target_encoder: Optional[LabelEncoder] = None
    if problem_type == "classification" and not is_numeric_dtype(y):
        target_encoder = LabelEncoder()
        y = pd.Series(target_encoder.fit_transform(y), index=y.index)

    # usar stratify solo en clasificacion cuando esta habilitado
    stratify_y = y if problem_type == "classification" and stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_y,
    )

    trained_models: Dict[str, Any] = {}
    metrics: Dict[str, Dict[str, float]] = {}
    best_params: Dict[str, Dict[str, Any]] = {}

    for name, estimator in models.items():
        pipeline = create_full_pipeline(
            estimator, numeric_features, categorical_features)
        start = time.perf_counter()

        if use_gridsearch and param_grids and name in param_grids:
            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grids[name],
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                refit=True,
            )
            grid.fit(X_train, y_train)
            fitted_pipeline = grid.best_estimator_
            best_params[name] = grid.best_params_
        else:
            fitted_pipeline = pipeline.fit(X_train, y_train)
            best_params[name] = {}

        elapsed = time.perf_counter() - start

        y_pred = fitted_pipeline.predict(X_test)

        if problem_type == "regression":
            model_metrics = _evaluate_regression(y_test, y_pred)
        else:
            model_metrics = _evaluate_classification(
                y_test, y_pred, fitted_pipeline, X_test)

        model_metrics["train_time_sec"] = round(elapsed, 4)
        trained_models[name] = fitted_pipeline
        metrics[name] = model_metrics

    return trained_models, metrics, best_params, target_encoder


def _evaluate_regression(y_true, y_pred) -> Dict[str, float]:
    """
    Calcula metricas basicas para regresion.
    """
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def _evaluate_classification(y_true, y_pred, pipeline, X_test) -> Dict[str, float]:
    """
    Calcula metricas basicas para clasificacion.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    # ROC-AUC solo si es binario y el modelo lo permite
    if y_true.nunique() == 2:
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(X_test)
        elif hasattr(pipeline, "decision_function"):
            proba = pipeline.decision_function(X_test)
        else:
            proba = None

        if proba is not None:
            try:
                if proba.ndim == 2:
                    proba = proba[:, 1]
                metrics["roc_auc"] = roc_auc_score(y_true, proba)
            except Exception:
                pass

    return metrics
