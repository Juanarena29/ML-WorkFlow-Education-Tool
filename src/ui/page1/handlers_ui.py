from pathlib import Path
from typing import Optional
import pandas as pd
import streamlit as st
from src.utils.constants import CLOUD_MAX_COLUMNS, CLOUD_MAX_ROWS
from src.savings.project_updates import (
    apply_loaded_dataset,
)
from src.utils.session import (
    MLProject,
    add_operation_log,
)
from src.data.loader import (
    get_all_nan_columns,
    get_basic_info,
    load_and_validate,
    truncate_dataset_if_needed,
)
from src.ui.page1.render_dataset_info_ui import (
    render_basic_info,
    render_dtypes_table
)
from src.ui.page1.dataset_upload_ui import (
    render_next_step_button
)
from src.ui.learn_explanations import (
    render_learn_one_empty_columns_warning_explanation
)


def handle_auto_load_sample_dataset(project: MLProject) -> bool:
    sample_path = Path(__file__).resolve().parents[3] / "DatosEDUCATOR.csv"
    if sample_path.exists():
        try:
            df_sample = pd.read_csv(sample_path)
        except Exception as exc:
            st.error(f"No se pudo cargar el dataset de ejemplo: {exc}")
            return False
        apply_loaded_dataset(project, df_sample)
        add_operation_log(
            "load_dataset",
            "Dataset de ejemplo cargado en modo learn.",
            status="success",
        )
        st.info("Dataset de ejemplo cargado automáticamente (modo educación).")
        return True

    st.warning("No se encontró el dataset de ejemplo DatosEDUCATOR.csv.")
    return False


def handle_existing_dataset_display(project: MLProject) -> None:
    st.info("Ya hay un dataset cargado. Puedes subir otro para reemplazarlo.")
    df_loaded = project.df_original
    info_loaded = get_basic_info(df_loaded)

    render_basic_info(info_loaded)

    st.subheader("Tipos detectados")
    render_dtypes_table(info_loaded["dtypes"])

    st.subheader("Vista previa del dataset")
    st.write(
        "Una vista previa del dataset ayuda a comprender su estructura y el tipo de datos disponibles."
    )
    st.dataframe(df_loaded.head(50), use_container_width=True)


def handle_file_upload_and_validation(
    uploaded_file, project: MLProject
) -> Optional[pd.DataFrame]:
    try:
        df, errors = load_and_validate(uploaded_file, min_rows=10)
    except ValueError as exc:
        add_operation_log("load_dataset", str(exc), status="error")
        st.error(str(exc))
        return None

    if errors:
        for err in errors:
            st.error(err)
        add_operation_log("load_dataset", "Dataset inválido", status="warning")
        return None

    df, truncate_messages = truncate_dataset_if_needed(df)
    for msg in truncate_messages:
        st.warning(msg)

    if project.runtime_mode == "demo":
        st.caption(
            f"ℹ️ Modo demo: límite de {CLOUD_MAX_ROWS:,} filas y {CLOUD_MAX_COLUMNS} columnas."
        )

    return df


def handle_empty_columns_removal(df: pd.DataFrame, learn: bool) -> pd.DataFrame:
    all_nan_cols = get_all_nan_columns(df)
    if not all_nan_cols:
        return df

    cols_str = ", ".join(map(str, all_nan_cols))
    st.warning(f"Columnas completamente vacías detectadas: {cols_str}.")

    if learn:
        render_learn_one_empty_columns_warning_explanation()

    drop_empty = st.checkbox(
        "Eliminar columnas vacías (recomendado)", value=True)
    if drop_empty:
        df_view = df.drop(columns=all_nan_cols)
        if df_view.empty or df_view.shape[1] == 0:
            st.error(
                "El dataset quedó sin columnas después de eliminar las vacías.")
            return df
        return df_view

    return df


def confirm_and_save_dataset(df: pd.DataFrame, project: MLProject, info: dict) -> bool:
    # Si ya está cargado, no mostrar el botón de confirmar
    if project.is_step_completed("load"):
        st.success("✅ Dataset ya cargado y guardado en el proyecto.")
        return True

    if not st.button("Confirmar carga"):
        return False

    apply_loaded_dataset(project, df)
    add_operation_log(
        "load_dataset",
        f"Dataset cargado con {info['rows']} filas y {info['columns']} columnas.",
        status="success",
    )
    st.success("✅ Dataset cargado y guardado en el proyecto.")
    return True
