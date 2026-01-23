"""
Funciones de evaluacion y preparacion de datos para resultados.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split


def build_metrics_table(metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Convierte un dict de metricas en DataFrame.
    """
    rows = []
    for model_name, model_metrics in metrics.items():
        row = {"modelo": model_name}
        row.update(model_metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def get_train_test_split(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str,
    split_config: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Reconstruye el train/test split segun configuracion guardada.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    test_size = split_config.get("test_size", 0.2)
    random_state = split_config.get("random_state", 42)
    stratify = split_config.get("stratify", True)

    stratify_y = y if problem_type == "classification" and stratify else None

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_y,
    )


def compute_confusion_matrix(
    y_true: pd.Series,
    y_pred: pd.Series,
    labels: List = None,
):
    """
    Calcula matriz de confusion y devuelve labels ordenados.
    """
    if labels is None:
        labels = sorted(pd.Series(y_true).dropna().unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return cm, labels


def compute_confusion_matrix_normalized(
    y_true: pd.Series,
    y_pred: pd.Series,
    labels: List = None,
):
    """
    Calcula matriz de confusion normalizada por filas.
    """
    if labels is None:
        labels = sorted(pd.Series(y_true).dropna().unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums
    return cm_norm, labels


def compute_roc_curve(
    y_true: pd.Series,
    y_score,
):
    """
    Calcula la curva ROC y AUC para clasificacion binaria.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def compute_residuals(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    """
    Calcula residuales (y_true - y_pred).
    """
    return y_true - y_pred


def get_feature_names(preprocessor, numeric_features: List[str], categorical_features: List[str]) -> List[str]:
    """
    Obtiene nombres de features despues del preprocesamiento.
    """
    if hasattr(preprocessor, "get_feature_names_out"):
        return list(preprocessor.get_feature_names_out())

    feature_names: List[str] = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "num":
            feature_names.extend(cols)
        elif name == "cat":
            if hasattr(transformer, "named_steps"):
                encoder = transformer.named_steps.get("onehot")
                if encoder is not None and hasattr(encoder, "get_feature_names_out"):
                    feature_names.extend(
                        list(encoder.get_feature_names_out(cols)))
                else:
                    feature_names.extend(cols)
            else:
                feature_names.extend(cols)

    return feature_names


def extract_feature_importance(
    pipeline,
    numeric_features: List[str],
    categorical_features: List[str],
) -> Optional[pd.DataFrame]:
    """
    Extrae feature importance o coeficientes del modelo si esta disponible.
    """
    estimator = pipeline.named_steps.get("model")
    preprocessor = pipeline.named_steps.get("preprocessor")

    if estimator is None or preprocessor is None:
        return None

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        coef = estimator.coef_
        if coef.ndim > 1:
            importances = np.mean(np.abs(coef), axis=0)
        else:
            importances = np.abs(coef)
    else:
        return None

    feature_names = get_feature_names(
        preprocessor, numeric_features, categorical_features)
    if len(feature_names) != len(importances):
        return None

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df
