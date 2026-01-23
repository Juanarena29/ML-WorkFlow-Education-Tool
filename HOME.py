import streamlit as st
from src.utils.session import initialize_session, get_project


st.set_page_config(
    page_title="ML WorkFlow (for dummies)",
    layout="wide"
)


def main() -> None:
    # Inicializar session state
    initialize_session()

    project = get_project()

    learn = project.ui_mode == "learn"

    if not project.home_completed:
        cols = st.columns(5)
        with cols[1]:
            if st.button("Modo APRENDER"):
                project.ui_mode = "learn"
                project.home_completed = True
                st.rerun()
        with cols[3]:
            if st.button("Modo HERRAMIENTA"):
                project.ui_mode = "tool"
                project.home_completed = True
                st.rerun()

    if project.home_completed and not learn:
        st.title("ML WorkFlow Tool")
        st.markdown("""
        Bienvenido al entrenador automático de modelos ML.

        **Flujo del proceso:**
        1. 📁 Carga tu dataset
        2. 🔍 Confirma los tipos de datos
        3. ⚙️ Configura el tratamiento de datos
        4. 📊 Explora tus datos (EDA)
        5. 🤖 Entrena los modelos que tú quieras
        6. 📈 Analiza resultados de los modelos
        7. 🎯 Predice nuevos datos con tus modelos
        👈 **Navega usando la barra lateral**
        """)

    if project.home_completed and learn:
        st.title("ML WorkFlow EDUCATOR")
        st.markdown("""
        ## 📘 ¿Qué es Machine Learning? (en pocas palabras)

        Imaginá que querés entender **qué hace que una casa sea cara o barata**.

        Tenés datos como:
        - los metros cuadrados,
        - la cantidad de habitaciones,
        - la ubicación,
        - la antigüedad,
        - si tiene garage o no.

        Si mirás **muchos ejemplos reales** de casas junto con su precio, empezás a notar **patrones**.  
        Por ejemplo: en general, las casas más grandes suelen ser más caras.

        En *Machine Learning* usamos esos ejemplos del pasado para entrenar un **sistema que aprende patrones** y luego puede **estimar el precio de una casa nueva**, incluso si nunca la vio antes.

        A este tipo de sistema lo llamamos **modelo predictivo**:  
        una herramienta que aprende a partir de datos para hacer predicciones lo más cercanas posible a la realidad.

        ---

        ### 🧭 ¿Qué vas a hacer en esta app?

        En este recorrido vas a construir ese proceso **paso a paso**, de forma guiada:

        1. Cargar un conjunto de datos  
        2. Definir qué información es importante y cuál es el objetivo a predecir  
        3. Limpiar y preparar los datos  
        4. Analizar patrones y relaciones  
        5. Entrenar modelos predictivos  
        6. Evaluar qué tan buenas son sus predicciones  

        No necesitas saber Machine Learning de antemano:  
        la app te va a explicar **qué se hace, por qué se hace y qué decisiones estás tomando** en cada paso.

        👉 **Cuando estés listo, podés comenzar el recorrido.**  

        """)

    if project.is_step_completed("home"):
        if st.button("EMPEZAR RECORRIDO"):
            st.switch_page("pages/1-LoadDataset.py")
        st.markdown("""
        Si deseas cambiar de modo, intenta recargando la página.
        """)


if __name__ == "__main__":
    main()
