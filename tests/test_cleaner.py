"""
Tests para src/data/cleaner.py

Cubre:
- Imputación de valores numéricos
- Imputación de valores categóricos
- Aplicación de configuración de limpieza
"""

import pytest
import pandas as pd
import numpy as np

from src.data.cleaner import (
    impute_numeric,
    impute_categorical,
    list_columns_with_nans,
    suggest_imputation_method,
    apply_cleaning_config,
)


class TestImputeNumeric:
    """Tests para impute_numeric."""

    def test_impute_median(self):
        """Imputa con mediana correctamente."""
        series = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
        result = impute_numeric(series, "median")

        assert result.isna().sum() == 0
        assert result.iloc[1] == 3.0  # mediana de [1, 3, 5]
        assert result.iloc[3] == 3.0

    def test_impute_mean(self):
        """Imputa con media correctamente."""
        series = pd.Series([1.0, np.nan, 2.0, np.nan, 3.0])
        result = impute_numeric(series, "mean")

        assert result.isna().sum() == 0
        assert result.iloc[1] == 2.0  # media de [1, 2, 3]

    def test_impute_constant(self):
        """Imputa con valor constante."""
        series = pd.Series([1.0, np.nan, 3.0])
        result = impute_numeric(series, "constant", value=99)

        assert result.isna().sum() == 0
        assert result.iloc[1] == 99

    def test_impute_interpolate(self):
        """Imputa con interpolación."""
        series = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
        result = impute_numeric(series, "interpolate")

        assert result.isna().sum() == 0
        assert result.iloc[1] == 2.0
        assert result.iloc[3] == 4.0

    def test_drop_rows_returns_unchanged(self):
        """drop_rows no modifica la serie (se maneja en apply_cleaning_config)."""
        series = pd.Series([1.0, np.nan, 3.0])
        result = impute_numeric(series, "drop_rows")

        assert result.isna().sum() == 1  # No cambia


class TestImputeCategorical:
    """Tests para impute_categorical."""

    def test_impute_mode(self):
        """Imputa con moda correctamente."""
        series = pd.Series(["a", "a", "b", None, "a"])
        result = impute_categorical(series, "mode")

        assert result.isna().sum() == 0
        assert result.iloc[3] == "a"  # moda

    def test_impute_unknown(self):
        """Imputa con 'Desconocido'."""
        series = pd.Series(["a", None, "b"])
        result = impute_categorical(series, "unknown")

        assert result.isna().sum() == 0
        assert result.iloc[1] == "Desconocido"

    def test_impute_constant(self):
        """Imputa con valor constante."""
        series = pd.Series(["a", None, "b"])
        result = impute_categorical(series, "constant", value="MISSING")

        assert result.isna().sum() == 0
        assert result.iloc[1] == "MISSING"

    def test_mode_on_empty_series_uses_desconocido(self):
        """Si la serie está vacía (solo NaNs), usa 'Desconocido'."""
        series = pd.Series([None, None, None])
        result = impute_categorical(series, "mode")

        assert result.iloc[0] == "Desconocido"


class TestListColumnsWithNans:
    """Tests para list_columns_with_nans."""

    def test_identifies_columns_with_nans(self, df_with_nans):
        """Identifica correctamente columnas con NaNs."""
        result = list_columns_with_nans(df_with_nans)

        assert "col_some_nan" in result
        assert "col_all_nan" in result
        assert "col_cat" in result
        assert "col_clean" not in result

    def test_empty_df_returns_empty_list(self, empty_df):
        """DataFrame vacío devuelve lista vacía."""
        result = list_columns_with_nans(empty_df)
        assert result == []

    def test_none_returns_empty_list(self):
        """None devuelve lista vacía."""
        result = list_columns_with_nans(None)
        assert result == []


class TestSuggestImputationMethod:
    """Tests para suggest_imputation_method."""

    def test_suggests_median_for_skewed_numeric(self):
        """Sugiere mediana para distribución sesgada."""
        # Distribución muy sesgada
        series = pd.Series([1, 1, 1, 1, 1, 100])
        result = suggest_imputation_method(series, "numeric")
        assert result == "median"

    def test_suggests_mean_for_symmetric_numeric(self):
        """Sugiere media para distribución simétrica."""
        series = pd.Series([1, 2, 3, 4, 5])
        result = suggest_imputation_method(series, "numeric")
        assert result == "mean"

    def test_suggests_mode_for_categorical(self):
        """Sugiere moda para categóricas."""
        series = pd.Series(["a", "b", "c"])
        result = suggest_imputation_method(series, "categorical")
        assert result == "mode"


class TestApplyCleaningConfig:
    """Tests para apply_cleaning_config."""

    def test_applies_numeric_imputation(self):
        """Aplica imputación numérica según config."""
        df = pd.DataFrame({"num": [1.0, np.nan, 3.0]})
        config = {
            "imputations": {
                "num": {"method": "median", "value": None}
            }
        }

        result = apply_cleaning_config(df, config)
        assert result["num"].isna().sum() == 0

    def test_applies_categorical_imputation(self):
        """Aplica imputación categórica según config."""
        df = pd.DataFrame({"cat": ["a", None, "b"]})
        config = {
            "imputations": {
                "cat": {"method": "unknown", "value": None}
            }
        }

        result = apply_cleaning_config(df, config)
        assert result["cat"].isna().sum() == 0
        assert result["cat"].iloc[1] == "Desconocido"

    def test_drop_rows_removes_nans(self):
        """drop_rows elimina filas con NaN."""
        df = pd.DataFrame({"col": [1, np.nan, 3, np.nan, 5]})
        config = {
            "imputations": {
                "col": {"method": "drop_rows", "value": None}
            }
        }

        result = apply_cleaning_config(df, config)
        assert len(result) == 3
        assert result["col"].isna().sum() == 0

    def test_drop_duplicates(self):
        """Elimina duplicados si está configurado."""
        df = pd.DataFrame({"col": [1, 1, 2, 2, 3]})
        config = {"drop_duplicates": True}

        result = apply_cleaning_config(df, config)
        assert len(result) == 3

    def test_empty_config_returns_copy(self):
        """Config vacía devuelve copia del DataFrame."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        config = {}

        result = apply_cleaning_config(df, config)
        assert result.equals(df)
        assert result is not df  # Es una copia

    def test_none_df_returns_none(self):
        """DataFrame None devuelve None."""
        result = apply_cleaning_config(None, {})
        assert result is None
