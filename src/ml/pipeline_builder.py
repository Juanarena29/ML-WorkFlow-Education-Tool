"""
Construccion de pipelines de preprocesamiento y modelos.
"""

from __future__ import annotations

from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessing_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """
    Crea un preprocesador para features numericas y categoricas.

    Args:
        numeric_features: Lista de columnas numericas.
        categorical_features: Lista de columnas categoricas.

    Returns:
        ColumnTransformer con pasos de imputacion y escalado/encoding.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(
            ("cat", categorical_pipeline, categorical_features))

    if not transformers:
        raise ValueError(
            "No hay features numericas ni categoricas para preprocesar.")

    preprocessor = ColumnTransformer(
        transformers=transformers, remainder="drop")

    return preprocessor


def create_full_pipeline(
    model,
    numeric_features: List[str],
    categorical_features: List[str],
) -> Pipeline:
    """
    Combina el preprocesador con un modelo en un solo pipeline.

    Args:
        model: Estimador de scikit-learn.
        numeric_features: Lista de columnas numericas.
        categorical_features: Lista de columnas categoricas.

    Returns:
        Pipeline completo (preprocesamiento + modelo).
    """
    preprocessor = build_preprocessing_pipeline(
        numeric_features, categorical_features)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
