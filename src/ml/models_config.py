"""
Configuracion de modelos base para entrenamiento.
"""

from __future__ import annotations

from typing import Any, Dict

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor


def get_regression_models() -> Dict[str, Any]:
    """
    Devuelve modelos de regresion base.
    """
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(random_state=22),
        "Lasso": Lasso(random_state=22),
        "RandomForest": RandomForestRegressor(random_state=22),
        "GradientBoosting": GradientBoostingRegressor(random_state=22),
        "XGBoost": XGBRegressor(
            random_state=22,
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
        ),
    }

    return models


def get_classification_models() -> Dict[str, Any]:
    """
    Devuelve modelos de clasificacion base.
    """
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(random_state=22),
        "GradientBoosting": GradientBoostingClassifier(random_state=22),
        "SVC": SVC(probability=True, random_state=22),
        "XGBoost": XGBClassifier(
            random_state=22,
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            eval_metric="logloss",
        ),
    }

    return models


def get_models_for_problem_type(problem_type: str) -> Dict[str, Any]:
    """
    Devuelve modelos segun el tipo de problema.
    """
    if problem_type == "regression":
        return get_regression_models()
    return get_classification_models()


def get_param_grids(problem_type: str, preset: str = "ligero") -> Dict[str, Dict[str, list]]:
    """
    Devuelve grids de hiperparametros para GridSearchCV.
    light medium y full para que se evaluen con distinta profundidad los mejores parametros.
    """
    if problem_type == "regression":
        grids_light = {
            "Ridge": {"model__alpha": [0.1, 1.0]},
            "Lasso": {"model__alpha": [0.01, 0.1]},
            "RandomForest": {"model__n_estimators": [100], "model__max_depth": [None, 10]},
            "GradientBoosting": {"model__n_estimators": [100], "model__learning_rate": [0.05, 0.1]},
            "XGBoost": {
                "model__n_estimators": [200],
                "model__max_depth": [4],
                "model__learning_rate": [0.05, 0.1],
            },
        }
        grids_medium = {
            "Ridge": {"model__alpha": [0.1, 1.0, 10.0]},
            "Lasso": {"model__alpha": [0.01, 0.1, 1.0]},
            "RandomForest": {"model__n_estimators": [100, 200], "model__max_depth": [None, 10]},
            "GradientBoosting": {"model__n_estimators": [100, 150], "model__learning_rate": [0.05, 0.1]},
            "XGBoost": {
                "model__n_estimators": [200, 300],
                "model__max_depth": [4, 6],
                "model__learning_rate": [0.05, 0.1],
            },
        }
        grids_full = {
            "Ridge": {"model__alpha": [0.01, 0.1, 1.0, 10.0]},
            "Lasso": {"model__alpha": [0.001, 0.01, 0.1, 1.0]},
            "RandomForest": {
                "model__n_estimators": [100, 200, 400],
                "model__max_depth": [None, 10, 20],
            },
            "GradientBoosting": {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.03, 0.05, 0.1],
            },
            "XGBoost": {
                "model__n_estimators": [200, 300, 500],
                "model__max_depth": [4, 6, 8],
                "model__learning_rate": [0.03, 0.05, 0.1],
            },
        }
    else:
        grids_light = {
            "LogisticRegression": {"model__C": [0.1, 1.0]},
            "RandomForest": {"model__n_estimators": [100], "model__max_depth": [None, 10]},
            "GradientBoosting": {"model__n_estimators": [100], "model__learning_rate": [0.05, 0.1]},
            "SVC": {"model__C": [0.5, 1.0], "model__gamma": ["scale"]},
            "XGBoost": {
                "model__n_estimators": [200],
                "model__max_depth": [4],
                "model__learning_rate": [0.05, 0.1],
            },
        }
        grids_medium = {
            "LogisticRegression": {"model__C": [0.1, 1.0, 10.0]},
            "RandomForest": {"model__n_estimators": [100, 200], "model__max_depth": [None, 10]},
            "GradientBoosting": {"model__n_estimators": [100, 150], "model__learning_rate": [0.05, 0.1]},
            "SVC": {"model__C": [0.5, 1.0], "model__gamma": ["scale", "auto"]},
            "XGBoost": {
                "model__n_estimators": [200, 300],
                "model__max_depth": [4, 6],
                "model__learning_rate": [0.05, 0.1],
            },
        }
        grids_full = {
            "LogisticRegression": {"model__C": [0.01, 0.1, 1.0, 10.0]},
            "RandomForest": {
                "model__n_estimators": [100, 200, 400],
                "model__max_depth": [None, 10, 20],
            },
            "GradientBoosting": {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.03, 0.05, 0.1],
            },
            "SVC": {"model__C": [0.5, 1.0, 2.0], "model__gamma": ["scale", "auto"]},
            "XGBoost": {
                "model__n_estimators": [200, 300, 500],
                "model__max_depth": [4, 6, 8],
                "model__learning_rate": [0.03, 0.05, 0.1],
            },
        }

    if preset == "medio":
        return grids_medium
    if preset == "completo":
        return grids_full
    return grids_light
