import streamlit as st

####################### PAGE 1 - CARGA DE DATASET #######################


def render_learn_one_dataset_explanation() -> None:
    with st.expander("📦 ¿Qué es un dataset?"):
        st.markdown(
            "Un **dataset** es una tabla de datos donde cada fila representa un algo (por ejemplo, una casa) y cada columna una característica de ese algo (precio, metros cuadrados, ubicación, etc.).\n\n"
            "En esta app los datasets se cargan en formato **CSV**, un tipo de archivo sencillo que puede abrirse y editarse con Excel, Google Sheets u otras herramientas similares. \n\n"
            "En el modo **EDUCACIÓN** se incluye un **dataset de ejemplo** para que puedas recorrer todo el flujo sin necesidad de subir datos propios."
        )


def render_learn_one_info_explanation() -> None:
    with st.expander("📊 ¿Qué es esta información?"):
        st.markdown(
            "Este resumen muestra cuánta información tiene tu dataset:\n\n"
            "- **Filas**: cantidad de registros disponibles.\n"
            "- **Columnas**: variables o características de cada registro.\n"
            "- **Tamaño**: espacio que ocupa el dataset en memoria.\n\n"
            "Cuantos más datos relevantes haya, mayor suele ser el potencial del modelo para aprender."
        )


def render_learn_one_dtypes_explanation() -> None:
    with st.expander("🧩 ¿Qué son los tipos de datos?"):
        st.markdown(
            "Los tipos de datos indican cómo está almacenada la información en cada columna y cómo puede ser utilizada por el modelo.\n\n"
            "En esta tabla podés encontrar principalmente:\n\n"
            "- **int64**: números enteros. Ejemplos: cantidad de habitaciones, pisos, años.\n"
            "- **float64**: números con decimales. Ejemplos: precios, promedios, porcentajes, superficies.\n"
            "- **object**: texto o categorías. Ejemplos: ciudad, tipo de propiedad, nombre de una categoría.\n\n"
            "Cada tipo de dato se trata de forma distinta durante el entrenamiento del modelo, por eso es importante identificarlos correctamente."
        )


def render_learn_one_empty_columns_warning_explanation() -> None:
    with st.expander("⚠️ ¿Qué significa este aviso?"):
        st.markdown(
            "Este aviso indica que se detectaron **columnas completamente vacías**, es decir, columnas que no contienen ningún dato útil.\n\n"
            "Estas columnas no aportan información al modelo y pueden eliminarse de forma segura para simplificar el dataset y mejorar el procesamiento."
        )


def render_learn_one_high_nan_warning_explanation() -> None:
    with st.expander("⚠️ ¿Qué significa este aviso?"):
        st.markdown(
            "Este aviso indica que algunas columnas del dataset tienen un **alto porcentaje de valores faltantes (NaNs)**.\n\n"
            "Cuando una columna tiene muchos NaNs, aporta poca información y puede afectar el entrenamiento del modelo si no se trata correctamente.\n\n"
            "Más adelante, estas columnas podrán eliminarse, completarse o transformarse según el caso."
        )

####################### PAGE 2 - DETECCTION DE TIPOS #######################


def render_learn_two_title_explanation():
    with st.expander("¿PARA QUE HACEMOS ESTO?"):
        st.markdown(
            "Tal como vimos en el paso anterior, la **selección de tipos de datos** es clave para que el modelo (y nosotros) sepamos **cómo usar correctamente la información**.\n\n"
            "**ML WorkFlow** detecta automáticamente los tipos de cada columna y los clasifica como "
            "**numéricos**, **categóricos**, **identificadores**, **fechas** o **textos**.\n\n"
            "Por seguridad y transparencia, el usuario puede **revisar y corregir** esta detección antes de continuar, "
            "evitando errores comunes que afectarían el entrenamiento del modelo."
        )


def render_learn_two_target_explanation():
    with st.expander('¿QUÉ ES EL "TARGET"?'):
        st.markdown(
            'El **target** (u **objetivo**) es la columna que indica **qué queremos que el modelo prediga.** \n\n'
            'Es el valor que el modelo intenta aprender a partir del resto de los datos. \n\n'
            'Por ejemplo, en un dataset inmobiliario, lo más lógico es que el target sea el **precio** de la propiedad. \n\n'
            'Para nuestro ejemplo, el target será **es_caro**, entrenaremos al modelo para que prediga si una propiedad debería ser cara o barata.'
        )


def render_learn_two_type_explanation():
    with st.expander('¿REGRESIÓN? ¿CLASIFICACIÓN?'):
        st.markdown("""
                **El tipo de problema depende de lo que querés predecir (target):**

                - **Regresión**: cuando el resultado es un **número**  
                Ej.: precio de una casa, ventas, temperatura.

                - **Clasificación**: cuando el resultado es una **categoría**  
                Ej.: sí/no, aprobado/desaprobado, tipo de cliente.

                **Regla rápida:**  
                número → *Regresión* | categorías → *Clasificación*

                ⚠️ Si el target es numérico pero representa categorías (0, 1, 2), sigue siendo **Clasificación**.
                """)

    ####################### PAGE 3 - LIMPIEZA DE DATOS #######################


def render_learn_three_treatment_explanation():
    with st.expander("📌 ¿Qué estamos haciendo en esta etapa?"):
        st.markdown(
            "En esta etapa vas a **analizar y preparar** el dataset antes de entrenar modelos.\n\n"
            "Acá se muestra información clave como la **cantidad de valores faltantes (NaNs)** por columna, "
            "la presencia de **filas duplicadas** y algunas **sugerencias automáticas** para tratarlos.\n\n"
            "Este paso funciona como un **análisis exploratorio inicial**: te ayuda a entender los problemas del dataset "
            "y a decidir cómo limpiarlo antes de realizar un EDA más completo y entrenar modelos."
        )


def render_learn_three_nans_explanation():
    with st.expander("❓ ¿Qué son los NaNs y por qué importan?"):
        st.markdown(
            "**NaN** significa *dato faltante* (celda vacía).\n\n"
            "El modelo no puede aprender correctamente si hay muchos valores faltantes sin tratar.\n\n"
            "Este resumen te ayuda a ver **qué columnas** tienen NaNs y **cuántos** para decidir el mejor tratamiento."
        )


def render_learn_three_suggestions_explanation():
    with st.expander("💡 ¿Qué significan estas sugerencias?"):
        st.markdown(
            "Las sugerencias son recomendaciones automáticas basadas en tus datos y en los tipos de columnas.\n\n"
            "No son obligatorias: podés usarlas como guía y después ajustar la configuración a tu criterio."
        )


def render_learn_three_duplicates_explanation():
    with st.expander("🧾 ¿Qué es un duplicado? ¿Conviene eliminarlo?"):
        st.markdown(
            "Una fila duplicada es una fila **idéntica a otra** (mismos valores en todas las columnas).\n\n"
            "Eliminar duplicados suele ser recomendable porque evita que el modelo “cuente dos veces” el mismo registro.\n\n"
            "Si tus duplicados son intencionales (por ejemplo, eventos repetidos reales), entonces **no conviene eliminarlos**."
        )


def render_learn_three_imputation_explanation():
    with st.expander("🧩 ¿Qué significa imputar valores faltantes?"):
        st.markdown(
            "Imputar significa **reemplazar valores faltantes** por un valor razonable.\n\n"
            "Ejemplos comunes:\n\n"
            "- En columnas numéricas: usar **media** o **mediana**.\n"
            "- En columnas categóricas: usar el valor más frecuente (**moda**).\n"
            "- También podés completar con un valor fijo (ej. \"Desconocido\") o eliminar filas/columnas si corresponde.\n\n"
            "La mejor opción depende del significado de la columna."
        )
    ####################### PAGE 4 - EDA #######################


def render_learn_four_eda_explanation():
    with st.expander("🔎 ¿Qué es el EDA y para qué sirve?"):
        st.markdown(
            "El **EDA (Análisis Exploratorio de Datos)** es un paso previo al Machine Learning.\n\n"
            "Sirve para entender rápidamente el dataset: cómo se distribuyen los datos, qué variables se relacionan "
            "y si hay patrones visibles.\n\n"
            "En esta app, el EDA es **automático**: vos elegís columnas y la herramienta genera visualizaciones útiles."
        )


def render_learn_four_distribution_explanations():
    with st.expander("📊 ¿Qué estoy viendo acá?"):
        st.markdown(
            "Estos gráficos te muestran **cómo se repiten los valores** de una columna.\n\n"
            "- En números: ves qué valores aparecen más y cuáles son poco comunes.\n"
            "- En categorías: ves qué opciones son más frecuentes que otras.\n\n"
            "Sirve para tener una idea rápida de los datos antes de entrenar un modelo."
        )


def render_learn_four_correlation_explanation():
    with st.expander("🔗 ¿Qué significa correlación?"):
        st.markdown(
            "La **correlación** mide qué tan relacionados están dos valores numéricos.\n\n"
            "- Cerca de **1**: suben juntos.\n"
            "- Cerca de **-1**: uno sube cuando el otro baja.\n"
            "- Cerca de **0**: no hay relación lineal clara.\n\n"
            "Importante: correlación **no** significa que una variable cause a la otra."
        )


def render_learn_four_relations_explanation():
    with st.expander("📈 ¿Qué muestra este gráfico de relaciones?"):
        st.markdown(
            "Este gráfico compara dos columnas numéricas:\n\n"
            "- **Eje X** y **Eje Y** son variables numéricas.\n"
            "- Cada punto es una fila del dataset.\n\n"
            "La opción **Color** te permite separar los puntos por una categoría (por ejemplo: barrio, tipo, etc.) "
            "para ver si se forman grupos o patrones."
        )


def render_learn_four_target_explanation():
    with st.expander("🎯 ¿Qué es el target?"):
        st.markdown(
            "El **target** es la variable que querés predecir.\n\n"
            "Por ejemplo:\n"
            "- Si querés predecir un **precio**, el target es numérico (regresión).\n"
            "- Si querés predecir una **clase** (sí/no, categoría), es clasificación.\n\n"
            "Este apartado te ayuda a ver cómo se comporta el target y cómo se relaciona con otras variables."
        )


def render_learn_four_invert_explanation():
    with st.expander("🔁 ¿Qué hace 'Invertir'?"):
        st.markdown(
            "Cambia qué variable va en cada eje del gráfico.\n\n"
            "No modifica los datos: solo cambia la forma de visualizar la relación."
        )

    ####################### PAGE 5 - ENTRENAMIENTO #######################


def render_learn_five_training_explanation():
    with st.expander("🚀 ¿Qué pasa cuando entreno modelos?"):
        st.markdown(
            "Entrenar significa que la app va a **aprender patrones** a partir de tus datos para poder predecir el target.\n\n"
            "Para evaluar si el modelo funciona bien, el dataset se divide en dos partes:\n"
            "- **Train**: donde aprende.\n"
            "- **Test**: donde se prueba con datos que no vio.\n\n"
            "Al final vas a ver métricas y podrás comparar modelos para elegir el mejor."
        )


def render_learn_five_select_explanation():
    with st.expander("🤖 ¿Qué significa seleccionar modelos?"):
        st.markdown(
            "Un **modelo** es una forma distinta de aprender y hacer predicciones.\n\n"
            "Podés entrenar varios para comparar resultados. Si estás empezando, lo más simple es elegir **1 o 2 modelos** "
            "y después probar más.\n\n"
            "La app se encarga de aplicar el preprocesamiento necesario y evaluar cada modelo con el mismo criterio."
        )


def render_learn_five_traintest_explanation():
    with st.expander("🧪 ¿Qué es el train/test split?"):
        st.markdown(
            "Este ajuste define qué parte de tus datos se usa para **probar** el modelo.\n\n"
            "- **Proporción de test**: por ejemplo 0.2 significa 20% test y 80% train.\n"
            "- **Random state**: cambia cómo se mezclan y separan los datos. Sirve para poder repetir resultados.\n\n"
            "Recomendación para empezar: **test = 0.2** y dejar el random state como está."
        )


def render_learn_five_stratify_explanation():
    with st.expander("⚖️ ¿Qué hace 'stratify'?"):
        st.markdown(
            "En clasificación, **stratify** intenta que el conjunto de train y test tengan proporciones similares de cada clase.\n\n"
            "Suele ser recomendable cuando las clases están desbalanceadas (por ejemplo, muchos 'No' y pocos 'Sí')."
        )


def render_learn_five_gridsearch_explanation():
    with st.expander("🔍 ¿Qué es GridSearchCV y cuándo conviene usarlo?"):
        st.markdown(
            "GridSearchCV prueba distintas configuraciones del modelo para encontrar una que funcione mejor.\n\n"
            "- Si está **desactivado**: el modelo entrena más rápido (recomendado para una primera prueba).\n"
            "- Si lo activás: tarda más, pero puede mejorar el resultado.\n\n"
            "Presets:\n"
            "- **Ligero**: rápido (ideal para empezar).\n"
            "- **Medio**: balance.\n"
            "- **Completo**: más lento y exhaustivo."
        )


def render_learn_five_scoring_explanation():
    with st.expander("📏 ¿Qué significa 'scoring'?"):
        st.markdown(
            "El **scoring** es la regla que usa la app para decidir qué resultado es “mejor”.\n\n"
            "Elegí uno según tu objetivo:\n"
            "- En **clasificación**: *accuracy* (simple), *f1_weighted* (mejor si hay desbalance), *roc_auc* (útil para probabilidades).\n"
            "- En **regresión**: *r2* (qué tan bien explica), *MAE/RMSE* (error promedio).\n\n"
            "Si no estás seguro: empezá con la opción que aparece por defecto."
        )
    ####################### PAGE 6 - RESULTADOS #######################


def render_learn_six_details_explanation():
    with st.expander("📊 ¿Qué es el detalle por modelo?"):
        st.markdown(
            "Este apartado muestra los **valores exactos de las métricas** para cada modelo.\n\n"
            "A diferencia de los gráficos, acá podés comparar modelos de forma directa y objetiva.\n\n"
            "Usalo para confirmar cuál modelo rinde mejor según la métrica que elegiste."
        )


def render_learn_six_feature_explanation():
    with st.expander("🔍 ¿Qué significa la importancia de variables?"):
        st.markdown(
            "La **importancia de variables** indica qué columnas influyen más en las predicciones del modelo.\n\n"
            "Variables más importantes tienen mayor impacto en el resultado final.\n\n"
            "Esto ayuda a entender el modelo y a detectar qué datos son más relevantes."
        )


def render_learn_six_residuals_explanation():
    with st.expander("📉 ¿Qué es un gráfico de residuos?"):
        st.markdown(
            "Un **gráfico de residuos** muestra la diferencia entre el valor real y el valor predicho.\n\n"
            "- Si los puntos se agrupan cerca de **0**, el modelo predice bien.\n"
            "- Si hay un patrón claro (curva o tendencia), puede indicar que el modelo no captura bien la relación.\n\n"
            "Sirve para detectar errores sistemáticos y entender dónde el modelo falla."
        )


def render_learn_six_results_explanation():
    with st.expander("📌 ¿Qué estoy viendo en esta pantalla?"):
        st.markdown(
            "Acá podés **comparar modelos** y entender cuál funciona mejor para tu dataset.\n\n"
            "Vas a ver:\n"
            "- Una comparación por métricas (números).\n"
            "- Gráficos por cada modelo (para ver aciertos/errores).\n"
            "- La opción de guardar el modelo que elijas."
        )


def render_learn_six_metrics_explanation():
    with st.expander("📏 ¿Cómo interpreto las métricas?"):
        st.markdown(
            "Las métricas son una forma de resumir qué tan bien predice el modelo.\n\n"
            "- En general, **más alto es mejor** (por ejemplo: Accuracy, F1, R²).\n"
            "- En métricas de error, **más bajo es mejor** (por ejemplo: MAE, RMSE).\n\n"
            "Lo importante es comparar modelos usando **la misma métrica**."
        )


def render_learn_six_graphmodels_explanation():
    with st.expander("📊 ¿Para qué sirven los gráficos por modelo?"):
        st.markdown(
            "Los gráficos ayudan a ver el comportamiento real del modelo, no solo un número.\n\n"
            "- En **clasificación**, muestran qué clases se confunden entre sí.\n"
            "- En **regresión**, muestran qué tan lejos están las predicciones de los valores reales.\n\n"
            "Si dos modelos tienen métricas parecidas, los gráficos suelen ayudarte a decidir mejor."
        )


def render_learn_six_confusion_explanation():
    with st.expander("🧩 ¿Qué es la matriz de confusión y la curva ROC?"):
        st.markdown(
            "**Matriz de confusión**: muestra aciertos y errores por clase.\n"
            "Te ayuda a ver, por ejemplo, si el modelo confunde 'A' con 'B'.\n\n"
            "**Curva ROC**: es una forma de evaluar modelos que trabajan con probabilidades.\n"
            "Suele ser útil cuando querés separar bien positivos y negativos."
        )


def render_learn_six_savemodel_explanation():
    with st.expander("💾 ¿Qué significa guardar un modelo?"):
        st.markdown(
            "Guardar un modelo significa conservar el modelo ya entrenado para usarlo después sin volver a entrenar.\n\n"
            "Por ejemplo, podés cargarlo más adelante para hacer predicciones con nuevos datos."
        )

    ####################### PAGE 7 - PREDICCIONES #######################


def render_learn_seven_prediction_explanation():
    with st.expander("📌 ¿Qué hacemos en esta etapa?"):
        st.markdown(
            "En esta página vas a usar un **modelo ya entrenado** para generar predicciones sobre **datos nuevos**.\n\n"
            "Podés subir un CSV con nuevas filas y la app calculará el valor predicho para cada una. "
            "Si además incluís el **target real**, también vas a ver un gráfico comparando **real vs predicho**. \n\n"
            "No te preocupes, en el modo **EDUCACIÓN** se incluye un dataset de prueba por defecto para que veas los resultados"
        )


def render_learn_seven_whatmodel_explanation():
    with st.expander("🧠 ¿Qué modelo se usa para predecir?"):
        st.markdown(
            "Para hacer predicciones necesitás un **modelo entrenado**.\n\n"
            "- Podés usar un modelo que entrenaste recién en esta app (queda disponible en la sesión).\n"
            "- O podés **cargar un modelo guardado** (por ejemplo un `.pkl/.joblib`) para reutilizarlo.\n\n"
            "La recomendación es usar un modelo entrenado con datos similares a los que vas a predecir."
        )


def render_learn_seven_csv_explanation():
    with st.expander("📄 ¿Cómo debe ser el CSV de entrada?"):
        st.markdown(
            "El archivo debe tener **las mismas columnas features** que se usaron para entrenar el modelo.\n\n"
            "Tenés dos formas de usar esta página:\n"
            "1. **Sin target**: subís solo las features y la app **solo predice**.\n"
            "2. **Con target**: subís las features **y también** la columna target real. "
            "Además de predecir, la app muestra un gráfico comparando **predicción vs valor real** "
            "(o matriz de confusión en clasificación).\n\n"
            "Si faltan columnas requeridas, la app te avisará antes de predecir. \n\n"
            "Los datos de prueba brindados poseen la columna **Target**, con el fin de ver cuanto un modelo puede acertar o fallar su prediccion."
        )


def render_learn_seven_graph_explanation():
    with st.expander("📊 ¿Cómo interpretar el gráfico?"):
        st.markdown(
            "**Regresión:** el gráfico muestra *Real (eje X)* vs *Predicción (eje Y)*.\n"
            "- Cuanto más cerca estén los puntos de la **línea ideal**, mejor predice el modelo.\n"
            "- Puntos muy alejados indican casos donde el modelo se equivoca más.\n\n"
            "**Clasificación:** se muestra una **matriz de confusión**.\n"
            "- La diagonal son aciertos (predijo la clase correcta).\n"
            "- Fuera de la diagonal son errores (confusiones entre clases)."
        )
