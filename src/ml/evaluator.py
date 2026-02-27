"""
Funciones de evaluacion y preparacion de datos para resultados.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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


# ---------------------------------------------------------------------------
# Funciones de formateo de metricas
# ---------------------------------------------------------------------------


def format_metric_value(key: str, value: Any) -> str:
    """Formatea un valor de metrica para visualizacion."""
    if key in ("mae", "rmse"):
        return f"{value:,.2f}"
    if key in ("r2", "accuracy", "precision", "recall", "f1", "roc_auc"):
        return f"{value:.4f}"
    if key == "train_time_sec":
        return f"{value:.4f}s"
    return str(value)


def format_metrics_table(
    df_metrics: pd.DataFrame,
    problem_type: str,
) -> pd.DataFrame:
    """Devuelve una copia del DataFrame de metricas con formato legible."""
    df_display = df_metrics.copy()
    if problem_type == "regression":
        for col in ("mae", "rmse"):
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(
                    lambda x: f"{x:,.2f}")
        if "r2" in df_display.columns:
            df_display["r2"] = df_display["r2"].apply(lambda x: f"{x:.4f}")
    return df_display


# ---------------------------------------------------------------------------
# Mapas de scoring para UI
# ---------------------------------------------------------------------------


def get_results_score_map(problem_type: str) -> Dict[str, str]:
    """Devuelve {label_visible: nombre_columna} para el selector de metricas."""
    if problem_type == "regression":
        return {
            "MAE – Error absoluto medio": "mae",
            "RMSE – Error cuadrático medio": "rmse",
            "R² – Capacidad explicativa": "r2",
        }
    return {
        "Accuracy – Exactitud": "accuracy",
        "Precision – Confiabilidad de positivos": "precision",
        "Recall – Cobertura de positivos": "recall",
        "F1 Score – Balance Precision/Recall": "f1",
        "ROC AUC – Probabilidades": "roc_auc",
    }


# ---------------------------------------------------------------------------
# Funciones generadoras de figuras Plotly
# ---------------------------------------------------------------------------


def metrics_comparison_fig(
    df_metrics: pd.DataFrame,
    metric_name: str,
    metric_label: str,
    problem_type: str,
) -> go.Figure:
    """Genera un bar chart horizontal comparando modelos por una metrica."""
    fig = px.bar(
        df_metrics.sort_values(metric_name, ascending=False),
        x=metric_name,
        y="modelo",
        orientation="h",
        title=f"Comparacion por {metric_label}",
    )
    if problem_type == "regression" and metric_name in ("mae", "rmse"):
        fig.update_xaxes(tickformat=",.2f")
    elif metric_name == "r2":
        fig.update_xaxes(tickformat=".4f")
    return fig


def confusion_matrix_fig(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> go.Figure:
    """Genera una heatmap de la matriz de confusion."""
    cm, labels = compute_confusion_matrix(y_true, y_pred)
    return px.imshow(
        cm,
        x=labels,
        y=labels,
        text_auto=True,
        labels={"x": "Predicho", "y": "Real"},
        title="Matriz de confusion",
    )


def confusion_matrix_normalized_fig(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> go.Figure:
    """Genera una heatmap de la matriz de confusion normalizada."""
    cm, labels = compute_confusion_matrix_normalized(y_true, y_pred)
    return px.imshow(
        cm,
        x=labels,
        y=labels,
        text_auto=".2f",
        labels={"x": "Predicho", "y": "Real"},
        title="Matriz de confusion (normalizada)",
    )


def roc_curve_fig(
    y_true: pd.Series,
    y_score: Any,
) -> go.Figure:
    """Genera la curva ROC a partir de scores ya calculados."""
    fpr, tpr, roc_auc = compute_roc_curve(y_true, y_score)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC={roc_auc:.3f}")
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Base", line=dict(dash="dash"),
        )
    )
    fig.update_layout(
        title="Curva ROC",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    return fig


def residuals_fig(
    y_true: pd.Series,
    y_pred: Any,
) -> go.Figure:
    """Genera el grafico de residuos vs prediccion."""
    residuals = compute_residuals(
        y_true, pd.Series(y_pred, index=y_true.index))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode="markers"))
    fig.update_layout(
        title="Residuos vs Prediccion",
        xaxis_title="Prediccion",
        yaxis_title="Residuo",
        xaxis=dict(tickformat=",.2f"),
        yaxis=dict(tickformat=",.2f"),
    )
    return fig


def feature_importance_fig(
    importances_df: pd.DataFrame,
    top_n: int = 20,
) -> go.Figure:
    """Genera un bar chart horizontal con las top N features mas importantes."""
    top = importances_df.head(top_n)
    return px.bar(
        top.sort_values("importance", ascending=True),
        x="importance",
        y="feature",
        orientation="h",
        title=f"Feature importance (top {top_n})",
    )


def get_roc_scores(
    pipeline: Any,
    X_test: pd.DataFrame,
) -> Optional[Any]:
    """Extrae scores para ROC del pipeline (predict_proba o decision_function).
    Retorna None si el modelo no lo soporta."""
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X_test)[:, 1]
    if hasattr(pipeline, "decision_function"):
        return pipeline.decision_function(X_test)
    return None


def decode_predictions(
    y_test: pd.Series,
    y_pred: Any,
    target_encoder: Any,
) -> Tuple[pd.Series, Any, Optional[pd.Series]]:
    """Alinea y_test y y_pred al mismo espacio de labels cuando hay encoder.

    Returns:
        (y_test_plot, y_pred_decoded, y_test_numeric_for_roc)
        y_test_numeric_for_roc es None si no se pudo transformar.
    """
    y_test_plot = y_test
    y_test_roc: Optional[pd.Series] = None

    if target_encoder is None:
        return y_test_plot, y_pred, y_test_roc

    y_pred_is_num = pd.api.types.is_numeric_dtype(pd.Series(y_pred))
    y_test_is_num = pd.api.types.is_numeric_dtype(y_test_plot)

    if y_pred_is_num and not y_test_is_num:
        try:
            y_pred = target_encoder.inverse_transform(
                pd.Series(y_pred).astype(int)
            )
        except (ValueError, TypeError):
            try:
                y_test_plot = pd.Series(
                    target_encoder.transform(y_test_plot),
                    index=y_test_plot.index,
                )
            except (ValueError, TypeError):
                pass

    # Preparar versión numérica para ROC
    if not pd.api.types.is_numeric_dtype(y_test_plot):
        try:
            y_test_roc = pd.Series(
                target_encoder.transform(y_test_plot),
                index=y_test_plot.index,
            )
        except (ValueError, TypeError):
            y_test_roc = None
    else:
        y_test_roc = y_test_plot

    return y_test_plot, y_pred, y_test_roc
