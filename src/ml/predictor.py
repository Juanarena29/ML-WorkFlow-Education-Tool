"""
Capa de servicio para carga segura de modelos y ejecucion de predicciones.

Responsabilidades:
- Validar nombres de archivo antes de cargar .pkl
- Registrar y verificar hashes SHA-256 de modelos guardados
- Ejecutar model.predict() y construir el DataFrame de salida
- Generar figuras Plotly de comparacion (prediccion vs real)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go

from src.utils.file_handler import get_models_dir, load_model


# ---------------------------------------------------------------------------
# Dataclass de resultado
# ---------------------------------------------------------------------------


@dataclass
class PredictionResult:
    """Resultado estructurado de una prediccion."""

    predictions: Any
    probabilities: Optional[Any] = None
    output_df: Optional[pd.DataFrame] = None


# ---------------------------------------------------------------------------
# Seguridad — hashes de modelos
# ---------------------------------------------------------------------------

_HASH_FILENAME = ".model_hashes.json"


def _hash_registry_path(base_dir: str = "models") -> Path:
    return get_models_dir(base_dir) / _HASH_FILENAME


def _compute_file_hash(filepath: Path) -> str:
    """Calcula SHA-256 de un archivo."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _load_hash_registry(base_dir: str = "models") -> Dict[str, str]:
    path = _hash_registry_path(base_dir)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_hash_registry(registry: Dict[str, str], base_dir: str = "models") -> None:
    path = _hash_registry_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(registry, fh, indent=2)


def register_model_hash(filename: str, base_dir: str = "models") -> None:
    """Registra el hash SHA-256 de un modelo recien guardado."""
    if not filename.endswith(".pkl"):
        filename = f"{filename}.pkl"
    filepath = Path(base_dir) / filename
    if not filepath.exists():
        return
    registry = _load_hash_registry(base_dir)
    registry[filename] = _compute_file_hash(filepath)
    _save_hash_registry(registry, base_dir)


# ---------------------------------------------------------------------------
# Validacion de nombre de archivo
# ---------------------------------------------------------------------------


def validate_model_filename(filename: str) -> List[str]:
    """Valida que el nombre de archivo sea seguro (sin path traversal)."""
    errors: List[str] = []
    if not filename:
        errors.append("Nombre de archivo vacio.")
        return errors
    if ".." in filename or "/" in filename or "\\" in filename:
        errors.append(
            "Nombre de archivo invalido: caracteres de navegacion de directorios no permitidos.",
        )
    if not filename.endswith(".pkl"):
        errors.append("Solo se permiten archivos .pkl.")
    return errors


# ---------------------------------------------------------------------------
# Verificacion de integridad
# ---------------------------------------------------------------------------


def verify_model_integrity(
    filename: str,
    base_dir: str = "models",
) -> Tuple[bool, str]:
    """Verifica SHA-256 del archivo contra el registro.

    Returns:
        (is_valid, mensaje)
    """
    filepath = Path(base_dir) / filename
    if not filepath.exists():
        return False, f"Archivo no encontrado: {filename}"

    registry = _load_hash_registry(base_dir)
    if filename not in registry:
        return False, (
            f"El modelo '{filename}' no tiene hash registrado. "
            "Podria ser un archivo externo no verificado."
        )

    current_hash = _compute_file_hash(filepath)
    if current_hash != registry[filename]:
        return False, (
            f"Hash de '{filename}' no coincide. "
            "El archivo pudo haber sido modificado externamente."
        )
    return True, "Integridad verificada."


# ---------------------------------------------------------------------------
# Carga segura de modelos
# ---------------------------------------------------------------------------


def load_model_safe(
    filename: str,
    base_dir: str = "models",
    require_hash: bool = False,
) -> Any:
    """Carga un modelo .pkl con validaciones de seguridad.

    Raises:
        ValueError: nombre inseguro o hash invalido (si require_hash=True).
        FileNotFoundError: archivo no existe.
    """
    name_errors = validate_model_filename(filename)
    if name_errors:
        raise ValueError("; ".join(name_errors))

    is_valid, msg = verify_model_integrity(filename, base_dir)
    if not is_valid and require_hash:
        raise ValueError(f"Modelo rechazado: {msg}")

    return load_model(filename, base_dir)


# ---------------------------------------------------------------------------
# Ejecucion de predicciones
# ---------------------------------------------------------------------------


def execute_prediction(
    model: Any,
    df: pd.DataFrame,
    feature_cols: List[str],
    problem_type: str,
    target_encoder: Any = None,
) -> PredictionResult:
    """Ejecuta predict (y predict_proba) y devuelve un PredictionResult.

    Raises:
        ValueError, TypeError: si el modelo falla al predecir.
    """
    X = df[feature_cols]
    preds = model.predict(X)

    # Decodificar etiquetas si es clasificacion con encoder
    if problem_type == "classification" and target_encoder is not None:
        if pd.api.types.is_numeric_dtype(pd.Series(preds)):
            try:
                preds = target_encoder.inverse_transform(
                    pd.Series(preds).astype(int),
                )
            except (ValueError, TypeError):
                pass  # conservar predicciones numericas

    # Probabilidades
    proba = None
    if problem_type == "classification" and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)

    output = build_prediction_output(df, preds, proba)

    return PredictionResult(
        predictions=preds,
        probabilities=proba,
        output_df=output,
    )


def build_prediction_output(
    df: pd.DataFrame,
    preds: Any,
    proba: Optional[Any],
) -> pd.DataFrame:
    """Construye DataFrame de salida con predicciones y probabilidades."""
    output = df.copy()
    output["prediccion"] = preds

    if proba is None:
        return output

    if hasattr(proba, "ndim") and proba.ndim == 2:
        if proba.shape[1] == 2:
            output["probabilidad"] = proba[:, 1]
        else:
            for idx in range(proba.shape[1]):
                output[f"proba_{idx}"] = proba[:, idx]

    return output


# ---------------------------------------------------------------------------
# Alineacion de labels para graficos comparativos
# ---------------------------------------------------------------------------


def align_labels_for_comparison(
    y_true: pd.Series,
    preds: Any,
    problem_type: str,
    target_encoder: Any = None,
) -> Tuple[pd.Series, Any]:
    """Alinea y_true y preds al mismo espacio de labels para graficar.

    Si las predicciones ya fueron decodificadas (texto) pero y_true tambien
    es texto, no hace nada. Si hay mismatch, intenta transformar y_true
    al espacio de preds.

    Returns:
        (y_true_plot, preds) alineados.
    """
    if problem_type != "classification" or target_encoder is None:
        return y_true, preds

    preds_is_num = pd.api.types.is_numeric_dtype(pd.Series(preds))
    y_true_is_num = pd.api.types.is_numeric_dtype(y_true)

    # Si preds es numerico y y_true es texto, transformar y_true a numerico
    if preds_is_num and not y_true_is_num:
        try:
            y_true = pd.Series(
                target_encoder.transform(y_true),
                index=y_true.index,
            )
        except (ValueError, TypeError):
            pass

    return y_true, preds


# ---------------------------------------------------------------------------
# Figuras de comparacion
# ---------------------------------------------------------------------------


def prediction_vs_real_fig(
    y_true: pd.Series,
    y_pred: Any,
) -> go.Figure:
    """Scatter plot de prediccion vs valor real (regresion)."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            name="Prediccion vs real",
        ),
    )
    min_val = min(float(y_true.min()), float(pd.Series(y_pred).min()))
    max_val = max(float(y_true.max()), float(pd.Series(y_pred).max()))
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Linea ideal",
        ),
    )
    fig.update_layout(
        title="Prediccion vs valor real",
        xaxis_title="Real",
        yaxis_title="Prediccion",
        xaxis=dict(tickformat=",.2f"),
        yaxis=dict(tickformat=",.2f"),
    )
    return fig
