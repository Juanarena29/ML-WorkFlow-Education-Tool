"""
Constantes compartidas del proyecto.
"""

import os

# =============================================================================
# RUNTIME MODE
# =============================================================================
# Detecta si estamos en Streamlit Cloud o en local.
# En Streamlit Cloud, la variable de entorno STREAMLIT_RUNTIME existe.
# También puedes forzar el modo con RUNTIME_MODE=cloud o RUNTIME_MODE=local.

_runtime_env = os.getenv("RUNTIME_MODE", "").lower()
if _runtime_env in ("cloud", "demo"):
    RUNTIME_MODE = "cloud"
elif _runtime_env in ("local", "full"):
    RUNTIME_MODE = "local"
else:
    # Auto-detección: Streamlit Cloud define algunas variables específicas
    RUNTIME_MODE = "cloud" if os.getenv("STREAMLIT_SHARING_MODE") else "local"

IS_CLOUD = RUNTIME_MODE == "cloud"

# =============================================================================
# LÍMITES PARA MODO CLOUD
# =============================================================================
CLOUD_MAX_ROWS = 20000
CLOUD_MAX_COLUMNS = 100
CLOUD_MAX_CV_FOLDS = 3

# =============================================================================
# TIPOS DE COLUMNAS
# =============================================================================
COLUMN_TYPES = ["numeric", "categorical", "datetime", "id", "text"]

IMPUTATION_METHODS_NUMERIC = {
    "Mediana": "median",
    "Media": "mean",
    "Eliminar filas": "drop_rows",
    "Valor constante": "constant",
    "Interpolacion": "interpolate",
}

IMPUTATION_METHODS_CATEGORICAL = {
    "Moda": "mode",
    "Categoria 'Desconocido'": "unknown",
    "Eliminar filas": "drop_rows",
    "Valor constante": "constant",
}


# =============================================================================
# FUNCIONES AUXILIARES DE RUNTIME
# =============================================================================
def get_max_rows() -> int:
    """Devuelve el máximo de filas permitido según el runtime mode."""
    return CLOUD_MAX_ROWS if IS_CLOUD else float("inf")


def get_max_columns() -> int:
    """Devuelve el máximo de columnas permitido según el runtime mode."""
    return CLOUD_MAX_COLUMNS if IS_CLOUD else float("inf")


def get_max_cv_folds() -> int:
    """Devuelve el máximo de folds para CV según el runtime mode."""
    return CLOUD_MAX_CV_FOLDS if IS_CLOUD else 10
