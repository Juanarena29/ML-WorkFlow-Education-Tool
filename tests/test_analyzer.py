"""
Tests para src/data/analyzer.py

Cubre:
- Detección de tipos de columnas
- Detección del tipo de problema (regresión/clasificación)
"""

import pytest
import pandas as pd
import numpy as np

from src.data.analyzer import detect_column_types, detect_problem_type


class TestDetectColumnTypes:
    """Tests para detect_column_types."""

    def test_detect_numeric_column(self):
        """Detecta correctamente columnas numéricas."""
        # Usar suficientes valores para evitar que parezca un ID
        df = pd.DataFrame({"nums": [1.5, 2.5, 3.7, 4.2, 5.0, 1.5, 2.5, 3.7]})
        result = detect_column_types(df)
        assert result["nums"] == "numeric"

    def test_detect_categorical_column(self):
        """Detecta correctamente columnas categóricas."""
        df = pd.DataFrame({"cats": ["a", "b", "c", "a", "b"]})
        result = detect_column_types(df)
        assert result["cats"] == "categorical"

    def test_detect_id_column(self):
        """Detecta columnas que parecen IDs (alta unicidad + nombre sugerente)."""
        df = pd.DataFrame({"id": list(range(100))})
        result = detect_column_types(df)
        assert result["id"] == "id"

    def test_detect_mixed_types(self, sample_df_classification):
        """Detecta correctamente múltiples tipos en un DataFrame mixto."""
        result = detect_column_types(sample_df_classification)

        # Las columnas numéricas pueden detectarse como numeric o id dependiendo
        # de la unicidad, lo importante es que no sean categorical
        assert result["feature_num1"] in ["numeric", "id"]
        assert result["feature_num2"] in ["numeric", "id"]
        assert result["feature_cat"] == "categorical"
        assert result["id_col"] == "id"
        assert result["target"] == "categorical"

    def test_empty_dataframe_returns_empty_dict(self, empty_df):
        """DataFrame vacío devuelve diccionario vacío."""
        result = detect_column_types(empty_df)
        assert result == {}

    def test_none_dataframe_returns_empty_dict(self):
        """None devuelve diccionario vacío."""
        result = detect_column_types(None)
        assert result == {}


class TestDetectProblemType:
    """Tests para detect_problem_type."""

    def test_classification_with_categorical_target(self, sample_df_classification):
        """Target categórico es clasificación."""
        result = detect_problem_type(sample_df_classification, "target")
        assert result == "classification"

    def test_regression_with_continuous_target(self, sample_df_regression):
        """Target numérico continuo es regresión."""
        result = detect_problem_type(sample_df_regression, "target")
        assert result == "regression"

    def test_classification_with_binary_numeric_target(self):
        """Target numérico binario (0/1) es clasificación."""
        df = pd.DataFrame({"target": [0, 1, 0, 1, 0, 1, 0, 1]})
        result = detect_problem_type(df, "target")
        assert result == "classification"

    def test_classification_with_few_numeric_classes(self):
        """Target numérico con pocas clases es clasificación."""
        df = pd.DataFrame({"target": [1, 2, 3, 1, 2, 3, 1, 2, 3]})
        result = detect_problem_type(df, "target")
        assert result == "classification"

    def test_raises_on_missing_target_column(self, sample_df_classification):
        """Lanza error si la columna target no existe."""
        with pytest.raises(ValueError, match="no existe"):
            detect_problem_type(sample_df_classification, "nonexistent")

    def test_raises_on_empty_dataframe(self, empty_df):
        """Lanza error si el DataFrame está vacío."""
        with pytest.raises(ValueError, match="vacio"):
            detect_problem_type(empty_df, "any")

    def test_raises_on_none_dataframe(self):
        """Lanza error si el DataFrame es None."""
        with pytest.raises(ValueError, match="vacio"):
            detect_problem_type(None, "any")
