from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data.analyzer import analyze_missing_values
from src.data.cleaner import list_columns_with_nans, suggest_imputation_method
from src.utils.constants import IMPUTATION_METHODS_CATEGORICAL, IMPUTATION_METHODS_NUMERIC


def build_cleaning_config(df: pd.DataFrame, column_types: dict) -> dict:
    config = {"imputations": {}, "drop_duplicates": False}

    columns_with_nans = list_columns_with_nans(df)
    for col in columns_with_nans:
        col_type = column_types.get(col)
        if col_type == "numeric":
            method_key = st.selectbox(
                f"{col} (numerica)",
                options=list(IMPUTATION_METHODS_NUMERIC.keys()),
                key=f"impute_{col}",
            )
            method = IMPUTATION_METHODS_NUMERIC[method_key]
        else:
            method_key = st.selectbox(
                f"{col} (categorica)",
                options=list(IMPUTATION_METHODS_CATEGORICAL.keys()),
                key=f"impute_{col}",
            )
            method = IMPUTATION_METHODS_CATEGORICAL[method_key]

        value = None
        if method == "constant":
            value = st.text_input(
                f"Valor constante para {col}",
                key=f"constant_{col}",
            )

        config["imputations"][col] = {"method": method, "value": value}

    return config


def show_cleaning_suggestions(df: pd.DataFrame, column_types: dict) -> None:
    missing = analyze_missing_values(df)
    if missing.empty:
        return

    for _, row in missing.iterrows():
        col = row["columna"]
        nan_pct = row["porcentaje_nans"]
        col_type = column_types.get(col)
        suggestion = suggest_imputation_method(df[col], col_type)
        st.caption(
            f"Sugerencia para '{col}': {suggestion}. NaNs: {nan_pct:.1f}%"
        )
