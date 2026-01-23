"""
Fixtures compartidos para tests.
"""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_df_classification():
    """DataFrame de ejemplo para clasificación."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "feature_num1": np.random.randn(n),
        "feature_num2": np.random.rand(n) * 100,
        "feature_cat": np.random.choice(["A", "B", "C"], n),
        "id_col": range(1, n + 1),
        "target": np.random.choice(["yes", "no"], n),
    })


@pytest.fixture
def sample_df_regression():
    """DataFrame de ejemplo para regresión."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "feature_num1": np.random.randn(n),
        "feature_num2": np.random.rand(n) * 100,
        "feature_cat": np.random.choice(["A", "B", "C"], n),
        "target": np.random.rand(n) * 1000,
    })


@pytest.fixture
def df_with_nans():
    """DataFrame con valores faltantes."""
    return pd.DataFrame({
        "col_clean": [1, 2, 3, 4, 5],
        "col_some_nan": [1.0, np.nan, 3.0, np.nan, 5.0],
        "col_all_nan": [np.nan, np.nan, np.nan, np.nan, np.nan],
        "col_cat": ["a", "b", None, "a", "b"],
    })


@pytest.fixture
def empty_df():
    """DataFrame vacío."""
    return pd.DataFrame()
