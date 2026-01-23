"""
Tests para src/data/loader.py

Cubre:
- Funciones básicas de info del dataset
- Truncado de datasets en modo cloud
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from src.data.loader import (
    get_basic_info,
    get_high_nan_columns,
    get_all_nan_columns,
    truncate_dataset_if_needed,
)


class TestGetBasicInfo:
    """Tests para get_basic_info."""

    def test_returns_correct_row_count(self, sample_df_classification):
        """Devuelve cantidad correcta de filas."""
        info = get_basic_info(sample_df_classification)
        assert info["rows"] == len(sample_df_classification)

    def test_returns_correct_column_count(self, sample_df_classification):
        """Devuelve cantidad correcta de columnas."""
        info = get_basic_info(sample_df_classification)
        assert info["columns"] == len(sample_df_classification.columns)

    def test_returns_memory_in_mb(self, sample_df_classification):
        """Devuelve memoria en MB."""
        info = get_basic_info(sample_df_classification)
        assert "memory_mb" in info
        assert isinstance(info["memory_mb"], float)
        assert info["memory_mb"] >= 0

    def test_returns_dtypes_dict(self, sample_df_classification):
        """Devuelve diccionario de tipos."""
        info = get_basic_info(sample_df_classification)
        assert "dtypes" in info
        assert isinstance(info["dtypes"], dict)
        assert len(info["dtypes"]) == len(sample_df_classification.columns)


class TestGetHighNanColumns:
    """Tests para get_high_nan_columns."""

    def test_identifies_high_nan_columns(self):
        """Identifica columnas con alto % de NaNs."""
        df = pd.DataFrame({
            "low_nan": [1, 2, 3, 4, np.nan],  # 20% NaN
            "high_nan": [1, np.nan, np.nan, np.nan, np.nan],  # 80% NaN
        })

        result = get_high_nan_columns(df, threshold=0.3)

        assert "high_nan" in result
        assert "low_nan" not in result

    def test_empty_df_returns_empty_dict(self, empty_df):
        """DataFrame vacío devuelve dict vacío."""
        result = get_high_nan_columns(empty_df)
        assert result == {}

    def test_none_returns_empty_dict(self):
        """None devuelve dict vacío."""
        result = get_high_nan_columns(None)
        assert result == {}


class TestGetAllNanColumns:
    """Tests para get_all_nan_columns."""

    def test_identifies_all_nan_columns(self, df_with_nans):
        """Identifica columnas completamente vacías."""
        result = get_all_nan_columns(df_with_nans)

        assert "col_all_nan" in result
        assert "col_some_nan" not in result
        assert "col_clean" not in result

    def test_empty_df_returns_empty_list(self, empty_df):
        """DataFrame vacío devuelve lista vacía."""
        result = get_all_nan_columns(empty_df)
        assert result == []


class TestTruncateDatasetIfNeeded:
    """Tests para truncate_dataset_if_needed."""

    def test_no_truncation_in_local_mode(self):
        """No trunca en modo local."""
        df = pd.DataFrame({"col": range(50000)})

        with patch("src.data.loader.IS_CLOUD", False):
            result, messages = truncate_dataset_if_needed(df)

        assert len(result) == 50000
        assert messages == []

    def test_truncates_rows_in_cloud_mode(self):
        """Trunca filas en modo cloud."""
        df = pd.DataFrame({"col": range(30000)})

        with patch("src.data.loader.IS_CLOUD", True):
            with patch("src.data.loader.get_max_rows", return_value=20000):
                with patch("src.data.loader.get_max_columns", return_value=100):
                    result, messages = truncate_dataset_if_needed(df)

        assert len(result) == 20000
        assert len(messages) == 1
        assert "filas" in messages[0].lower()

    def test_truncates_columns_in_cloud_mode(self):
        """Trunca columnas en modo cloud."""
        df = pd.DataFrame({f"col{i}": [1, 2, 3] for i in range(150)})

        with patch("src.data.loader.IS_CLOUD", True):
            with patch("src.data.loader.get_max_rows", return_value=20000):
                with patch("src.data.loader.get_max_columns", return_value=100):
                    result, messages = truncate_dataset_if_needed(df)

        assert len(result.columns) == 100
        assert len(messages) == 1
        assert "columnas" in messages[0].lower()

    def test_truncates_both_rows_and_columns(self):
        """Trunca filas y columnas si ambos exceden."""
        df = pd.DataFrame({f"col{i}": range(30000) for i in range(150)})

        with patch("src.data.loader.IS_CLOUD", True):
            with patch("src.data.loader.get_max_rows", return_value=20000):
                with patch("src.data.loader.get_max_columns", return_value=100):
                    result, messages = truncate_dataset_if_needed(df)

        assert len(result) == 20000
        assert len(result.columns) == 100
        assert len(messages) == 2

    def test_none_df_returns_none(self):
        """DataFrame None devuelve None."""
        result, messages = truncate_dataset_if_needed(None)
        assert result is None
        assert messages == []

    def test_empty_df_returns_empty(self, empty_df):
        """DataFrame vacío devuelve vacío."""
        result, messages = truncate_dataset_if_needed(empty_df)
        assert result.empty
        assert messages == []
