"""
actualiza estado de MLProject
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from src.utils.session import MLProject


def apply_loaded_dataset(project: MLProject, df: pd.DataFrame) -> None:
    """
    Reemplaza el dataset original y limpia datos derivados.
    """
    if project.df_original is not None:
        project.clear_derived_data()
    project.df_original = df
    project.update_metadata()


def save_types_and_target(
    project: MLProject,
    selected_types: Dict[str, str],
    target_column: str,
    problem_type: str,
) -> None:
    project.column_types = selected_types
    project.target_column = target_column
    project.problem_type = problem_type
    project.update_metadata()


def save_cleaning_result(
    project: MLProject,
    cleaned_df: pd.DataFrame,
    cleaning_config: Dict[str, Any],
) -> None:
    project.df_limpio = cleaned_df
    project.cleaning_config = cleaning_config
    project.update_metadata()


def save_training_results(
    project: MLProject,
    trained_models: Dict[str, Any],
    metrics: Dict[str, Dict[str, float]],
    best_params: Dict[str, Any],
    target_encoder: Any,
    split_config: Dict[str, Any],
) -> None:
    project.trained_models = trained_models
    project.metrics = metrics
    project.best_params = best_params
    project.target_label_encoder = target_encoder
    project.train_test_split_config = split_config
    project.update_metadata()
