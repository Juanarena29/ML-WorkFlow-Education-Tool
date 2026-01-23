"""
Tests para src/ml/model_trainer.py

Cubre:
- Entrenamiento de modelos de clasificación
- Entrenamiento de modelos de regresión
- Validación de inputs
- Métricas generadas
"""

import pytest
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression, LinearRegression

from src.ml.model_trainer import train_models


class TestTrainModelsClassification:
    """Tests para train_models con clasificación."""

    def test_trains_single_model(self, sample_df_classification):
        """Entrena un modelo de clasificación correctamente."""
        models = {"LogReg": LogisticRegression(max_iter=200)}

        trained, metrics, best_params, encoder = train_models(
            df=sample_df_classification,
            target_column="target",
            numeric_features=["feature_num1", "feature_num2"],
            categorical_features=["feature_cat"],
            models=models,
            problem_type="classification",
            test_size=0.2,
            random_state=42,
        )

        assert "LogReg" in trained
        assert "LogReg" in metrics
        assert "accuracy" in metrics["LogReg"]
        assert "precision" in metrics["LogReg"]
        assert "recall" in metrics["LogReg"]
        assert "f1" in metrics["LogReg"]

    def test_returns_label_encoder_for_string_target(self, sample_df_classification):
        """Devuelve LabelEncoder cuando el target es string."""
        models = {"LogReg": LogisticRegression(max_iter=200)}

        _, _, _, encoder = train_models(
            df=sample_df_classification,
            target_column="target",
            numeric_features=["feature_num1", "feature_num2"],
            categorical_features=["feature_cat"],
            models=models,
            problem_type="classification",
        )

        assert encoder is not None
        assert hasattr(encoder, "classes_")

    def test_trains_multiple_models(self, sample_df_classification):
        """Entrena múltiples modelos."""
        models = {
            "LogReg1": LogisticRegression(max_iter=200, random_state=1),
            "LogReg2": LogisticRegression(max_iter=200, random_state=2),
        }

        trained, metrics, _, _ = train_models(
            df=sample_df_classification,
            target_column="target",
            numeric_features=["feature_num1", "feature_num2"],
            categorical_features=["feature_cat"],
            models=models,
            problem_type="classification",
        )

        assert len(trained) == 2
        assert len(metrics) == 2


class TestTrainModelsRegression:
    """Tests para train_models con regresión."""

    def test_trains_regression_model(self, sample_df_regression):
        """Entrena un modelo de regresión correctamente."""
        models = {"LinReg": LinearRegression()}

        trained, metrics, _, encoder = train_models(
            df=sample_df_regression,
            target_column="target",
            numeric_features=["feature_num1", "feature_num2"],
            categorical_features=["feature_cat"],
            models=models,
            problem_type="regression",
        )

        assert "LinReg" in trained
        assert "mae" in metrics["LinReg"]
        assert "rmse" in metrics["LinReg"]
        assert "r2" in metrics["LinReg"]
        assert encoder is None  # No encoder para regresión

    def test_regression_metrics_are_numeric(self, sample_df_regression):
        """Las métricas de regresión son valores numéricos válidos."""
        models = {"LinReg": LinearRegression()}

        _, metrics, _, _ = train_models(
            df=sample_df_regression,
            target_column="target",
            numeric_features=["feature_num1", "feature_num2"],
            categorical_features=["feature_cat"],
            models=models,
            problem_type="regression",
        )

        assert isinstance(metrics["LinReg"]["mae"], float)
        assert isinstance(metrics["LinReg"]["rmse"], float)
        assert isinstance(metrics["LinReg"]["r2"], float)
        assert metrics["LinReg"]["mae"] >= 0
        assert metrics["LinReg"]["rmse"] >= 0


class TestTrainModelsValidation:
    """Tests de validación de inputs."""

    def test_raises_on_empty_dataframe(self, empty_df):
        """Lanza error con DataFrame vacío."""
        models = {"LogReg": LogisticRegression()}

        with pytest.raises(ValueError, match="vacio"):
            train_models(
                df=empty_df,
                target_column="target",
                numeric_features=[],
                categorical_features=[],
                models=models,
                problem_type="classification",
            )

    def test_raises_on_none_dataframe(self):
        """Lanza error con DataFrame None."""
        models = {"LogReg": LogisticRegression()}

        with pytest.raises(ValueError, match="vacio"):
            train_models(
                df=None,
                target_column="target",
                numeric_features=[],
                categorical_features=[],
                models=models,
                problem_type="classification",
            )

    def test_raises_on_missing_target(self, sample_df_classification):
        """Lanza error si el target no existe."""
        models = {"LogReg": LogisticRegression()}

        with pytest.raises(ValueError, match="no existe"):
            train_models(
                df=sample_df_classification,
                target_column="nonexistent",
                numeric_features=["feature_num1"],
                categorical_features=[],
                models=models,
                problem_type="classification",
            )

    def test_includes_train_time(self, sample_df_classification):
        """Incluye tiempo de entrenamiento en métricas."""
        models = {"LogReg": LogisticRegression(max_iter=200)}

        _, metrics, _, _ = train_models(
            df=sample_df_classification,
            target_column="target",
            numeric_features=["feature_num1", "feature_num2"],
            categorical_features=["feature_cat"],
            models=models,
            problem_type="classification",
        )

        assert "train_time_sec" in metrics["LogReg"]
        assert metrics["LogReg"]["train_time_sec"] >= 0
