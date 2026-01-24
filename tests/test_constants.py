"""
Tests para src/utils/constants.py (Runtime Mode)

Cubre:
- Funciones de límites según runtime mode
"""

import pytest
from unittest.mock import patch
import importlib


class TestRuntimeModeFunctions:
    """Tests para funciones de runtime mode."""

    def test_get_max_rows_returns_int_or_inf(self):
        """get_max_rows devuelve un número válido."""
        from src.utils.constants import get_max_rows

        result = get_max_rows()
        assert isinstance(result, (int, float))
        assert result > 0

    def test_get_max_columns_returns_int_or_inf(self):
        """get_max_columns devuelve un número válido."""
        from src.utils.constants import get_max_columns

        result = get_max_columns()
        assert isinstance(result, (int, float))
        assert result > 0

    def test_get_max_cv_folds_returns_int(self):
        """get_max_cv_folds devuelve un entero válido."""
        from src.utils.constants import get_max_cv_folds

        result = get_max_cv_folds()
        assert isinstance(result, int)
        assert result >= 2

    def test_cloud_limits_are_defined(self):
        """Constantes de límites cloud están definidas."""
        from src.utils.constants import (
            CLOUD_MAX_ROWS,
            CLOUD_MAX_COLUMNS,
            CLOUD_MAX_CV_FOLDS,
        )

        assert CLOUD_MAX_ROWS > 0
        assert CLOUD_MAX_COLUMNS > 0
        assert CLOUD_MAX_CV_FOLDS >= 2


class TestRuntimeModeDetection:
    """Tests para detección de runtime mode."""

    def test_runtime_mode_is_string(self):
        """RUNTIME_MODE es un string válido."""
        from src.utils.constants import RUNTIME_MODE

        assert isinstance(RUNTIME_MODE, str)
        assert RUNTIME_MODE in ("demo", "full")

    def test_is_demo_is_boolean(self):
        """IS_DEMO es un booleano."""
        from src.utils.constants import IS_DEMO

        assert isinstance(IS_DEMO, bool)
