"""
Tests para src/ml/evaluator.py

Cubre:
- Construcción de tabla de métricas
- Cálculo de matriz de confusión
- Cálculo de curva ROC
- Cálculo de residuales
"""

import pytest
import pandas as pd
import numpy as np

from src.ml.evaluator import (
    build_metrics_table,
    compute_confusion_matrix,
    compute_confusion_matrix_normalized,
    compute_residuals,
    compute_roc_curve,
)


class TestBuildMetricsTable:
    """Tests para build_metrics_table."""

    def test_converts_dict_to_dataframe(self):
        """Convierte dict de métricas a DataFrame."""
        metrics = {
            "Model1": {"accuracy": 0.85, "f1": 0.82},
            "Model2": {"accuracy": 0.90, "f1": 0.88},
        }

        result = build_metrics_table(metrics)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "modelo" in result.columns
        assert "accuracy" in result.columns
        assert "f1" in result.columns

    def test_empty_metrics_returns_empty_dataframe(self):
        """Dict vacío devuelve DataFrame vacío."""
        result = build_metrics_table({})
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_preserves_model_names(self):
        """Preserva los nombres de los modelos."""
        metrics = {"MyModel": {"score": 0.9}}
        result = build_metrics_table(metrics)

        assert result.iloc[0]["modelo"] == "MyModel"


class TestComputeConfusionMatrix:
    """Tests para compute_confusion_matrix."""

    def test_binary_classification(self):
        """Calcula matriz para clasificación binaria."""
        y_true = pd.Series([0, 0, 1, 1, 0, 1])
        y_pred = pd.Series([0, 1, 1, 1, 0, 0])

        cm, labels = compute_confusion_matrix(y_true, y_pred)

        assert cm.shape == (2, 2)
        assert len(labels) == 2

    def test_multiclass_classification(self):
        """Calcula matriz para clasificación multiclase."""
        y_true = pd.Series(["a", "b", "c", "a", "b", "c"])
        y_pred = pd.Series(["a", "b", "a", "a", "c", "c"])

        cm, labels = compute_confusion_matrix(y_true, y_pred)

        assert cm.shape == (3, 3)
        assert len(labels) == 3
        assert set(labels) == {"a", "b", "c"}

    def test_uses_provided_labels(self):
        """Usa labels proporcionados si se especifican."""
        y_true = pd.Series([0, 1, 0, 1])
        y_pred = pd.Series([0, 1, 1, 1])

        cm, labels = compute_confusion_matrix(y_true, y_pred, labels=[0, 1])

        assert labels == [0, 1]


class TestComputeConfusionMatrixNormalized:
    """Tests para compute_confusion_matrix_normalized."""

    def test_normalized_values_between_0_and_1(self):
        """Valores normalizados están entre 0 y 1."""
        y_true = pd.Series([0, 0, 1, 1, 0, 1])
        y_pred = pd.Series([0, 1, 1, 1, 0, 0])

        cm, _ = compute_confusion_matrix_normalized(y_true, y_pred)

        assert cm.min() >= 0
        assert cm.max() <= 1

    def test_rows_sum_to_one(self):
        """Cada fila suma aproximadamente 1."""
        y_true = pd.Series([0, 0, 0, 1, 1, 1])
        y_pred = pd.Series([0, 0, 1, 1, 1, 0])

        cm, _ = compute_confusion_matrix_normalized(y_true, y_pred)

        row_sums = cm.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, [1.0, 1.0])


class TestComputeRocCurve:
    """Tests para compute_roc_curve."""

    def test_returns_fpr_tpr_auc(self):
        """Devuelve FPR, TPR y AUC."""
        y_true = pd.Series([0, 0, 1, 1, 0, 1])
        y_score = np.array([0.1, 0.3, 0.8, 0.9, 0.2, 0.7])

        fpr, tpr, roc_auc = compute_roc_curve(y_true, y_score)

        assert len(fpr) > 0
        assert len(tpr) > 0
        assert 0 <= roc_auc <= 1

    def test_perfect_predictions_auc_1(self):
        """Predicciones perfectas dan AUC = 1."""
        y_true = pd.Series([0, 0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        _, _, roc_auc = compute_roc_curve(y_true, y_score)

        assert roc_auc == 1.0

    def test_random_predictions_auc_around_half(self):
        """Predicciones aleatorias dan AUC ~ 0.5."""
        np.random.seed(42)
        y_true = pd.Series([0] * 50 + [1] * 50)
        y_score = np.random.rand(100)

        _, _, roc_auc = compute_roc_curve(y_true, y_score)

        # Con semilla fija, debería estar cerca de 0.5
        assert 0.3 <= roc_auc <= 0.7


class TestComputeResiduals:
    """Tests para compute_residuals."""

    def test_computes_difference(self):
        """Calcula la diferencia y_true - y_pred."""
        y_true = pd.Series([10, 20, 30])
        y_pred = pd.Series([12, 18, 33])

        residuals = compute_residuals(y_true, y_pred)

        expected = pd.Series([-2, 2, -3])
        pd.testing.assert_series_equal(residuals, expected)

    def test_perfect_predictions_zero_residuals(self):
        """Predicciones perfectas dan residuales cero."""
        y_true = pd.Series([1.0, 2.0, 3.0])
        y_pred = pd.Series([1.0, 2.0, 3.0])

        residuals = compute_residuals(y_true, y_pred)

        assert (residuals == 0).all()
