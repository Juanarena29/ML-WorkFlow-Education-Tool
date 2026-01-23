"""
Tests para src/data/validator.py

Cubre:
- Validación del dataset
- Validación del target
"""

import pytest
import pandas as pd
import numpy as np

from src.data.validator import validate_dataset, validate_target


class TestValidateDataset:
    """Tests para validate_dataset."""

    def test_valid_dataset_returns_empty_list(self, sample_df_classification):
        """Dataset válido no genera errores."""
        errors = validate_dataset(sample_df_classification)
        assert errors == []

    def test_none_dataset_returns_error(self):
        """Dataset None genera error."""
        errors = validate_dataset(None)
        assert len(errors) == 1
        assert "vacio" in errors[0].lower()

    def test_empty_dataset_returns_error(self, empty_df):
        """Dataset vacío genera error."""
        errors = validate_dataset(empty_df)
        assert len(errors) == 1
        assert "vacio" in errors[0].lower()

    def test_small_dataset_returns_error(self):
        """Dataset muy pequeño genera error."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        errors = validate_dataset(df, min_rows=10)
        assert len(errors) == 1
        assert "pequeño" in errors[0].lower()

    def test_empty_column_name_returns_error(self):
        """Columna con nombre vacío genera error."""
        df = pd.DataFrame({" ": [1, 2, 3], "valid": [4, 5, 6]})
        # Renombrar para simular columna vacía
        df.columns = ["", "valid"]
        # Crear dataframe con suficientes filas
        df = pd.concat([df] * 5, ignore_index=True)
        errors = validate_dataset(df, min_rows=10)
        assert any("vacio" in e.lower() or "nombre" in e.lower()
                   for e in errors)


class TestValidateTarget:
    """Tests para validate_target."""

    def test_valid_classification_target(self, sample_df_classification):
        """Target válido de clasificación no genera errores."""
        errors = validate_target(
            sample_df_classification,
            "target",
            problem_type="classification"
        )
        assert errors == []

    def test_missing_target_column(self, sample_df_classification):
        """Target inexistente genera error."""
        errors = validate_target(sample_df_classification, "nonexistent")
        assert len(errors) == 1
        assert "no existe" in errors[0].lower()

    def test_target_with_nans(self):
        """Target con NaNs genera advertencia."""
        df = pd.DataFrame({"target": [1, np.nan, 3, np.nan, 5]})
        errors = validate_target(df, "target")
        assert any("nan" in e.lower() for e in errors)

    def test_classification_with_single_class(self):
        """Clasificación con una sola clase genera error."""
        df = pd.DataFrame({"target": ["a", "a", "a", "a", "a"]})
        errors = validate_target(df, "target", problem_type="classification")
        assert any("2 clases" in e.lower() for e in errors)

    def test_classification_with_too_many_classes(self):
        """Clasificación con muchas clases genera advertencia."""
        df = pd.DataFrame({"target": list(range(100))})
        errors = validate_target(
            df, "target",
            problem_type="classification",
            max_classes=50
        )
        assert any("clases" in e.lower() for e in errors)

    def test_none_dataset_returns_error(self):
        """Dataset None genera error."""
        errors = validate_target(None, "target")
        assert len(errors) == 1

    def test_empty_dataset_returns_error(self, empty_df):
        """Dataset vacío genera error."""
        errors = validate_target(empty_df, "target")
        assert len(errors) == 1
