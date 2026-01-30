from __future__ import annotations
import streamlit as st


def render_next_step_button(label: str, target_page: str, enabled: bool) -> None:
    """Renderiza botón para avanzar al siguiente paso si está disponible."""
    if not enabled:
        return
    if st.button(label):
        st.switch_page(target_page)


def render_dataset_uploader(label: str = "Selecciona un CSV", types: list[str] | None = None):
    """Renderiza el file_uploader y devuelve el archivo seleccionado."""
    return st.file_uploader(label, type=types or ["csv"])
