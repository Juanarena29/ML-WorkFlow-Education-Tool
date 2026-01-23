"""
Funciones para guardar y cargar modelos y configuraciones.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

import joblib


def get_models_dir(base_dir: str = "models") -> Path:
    """
    Devuelve el Path de la carpeta de modelos y la crea si no existe.
    """
    path = Path(base_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_saved_models(base_dir: str = "models") -> List[str]:
    """
    Lista archivos .pkl en la carpeta de modelos.
    """
    models_dir = get_models_dir(base_dir)
    return sorted([p.name for p in models_dir.glob("*.pkl")])


def save_model(model: Any, filename: str, base_dir: str = "models") -> str:
    """
    Guarda un modelo con joblib.
    """
    models_dir = get_models_dir(base_dir)
    if not filename.endswith(".pkl"):
        filename = f"{filename}.pkl"
    path = models_dir / filename
    joblib.dump(model, path)
    return str(path)


def load_model(filename: str, base_dir: str = "models") -> Any:
    """
    Carga un modelo guardado con joblib.
    """
    path = Path(base_dir) / filename
    return joblib.load(path)


def save_project_config(
    config: Dict[str, Any],
    filename: Optional[str] = None,
    base_dir: str = "projectconfigs",
) -> str:
    """
    Guarda configuracion del proyecto como JSON.
    """
    models_dir = get_models_dir(base_dir)
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"project_config_{timestamp}.json"
    path = models_dir / filename
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
    return str(path)
