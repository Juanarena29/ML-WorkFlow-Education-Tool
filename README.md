# ML WorkFlow Education Tool

Una aplicación web interactiva para entrenar modelos de Machine Learning **sin escribir código**.

Diseñada tanto para **aprender los fundamentos de ML** como para **usar como herramienta práctica** de entrenamiento de modelos.

---

## 🎯 ¿Qué es?

ML WorkFlow es una aplicación construida con Streamlit que guía al usuario paso a paso a través del proceso completo de Machine Learning:

1. **Cargar datos** → Subir un archivo CSV
2. **Detectar tipos** → Identificar variables numéricas, categóricas y el target
3. **Limpiar datos** → Imputar valores faltantes, eliminar duplicados
4. **Explorar datos (EDA)** → Visualizar distribuciones, correlaciones y relaciones
5. **Entrenar modelos** → Seleccionar y entrenar múltiples algoritmos
6. **Analizar resultados** → Comparar métricas, ver matrices de confusión, curvas ROC
7. **Predecir** → Usar los modelos entrenados con nuevos datos

---

## 📘 Modo APRENDER (Learn Mode)

El corazón de este proyecto es el **modo educativo**.

### ¿Para quién es?

- Estudiantes que recién empiezan con Machine Learning
- Personas curiosas que quieren entender qué hay detrás de las predicciones
- Cualquiera que prefiera aprender haciendo, no solo leyendo

### ¿Qué hace diferente?

En cada paso del proceso, el modo APRENDER incluye **explicaciones contextuales** que responden:

- **¿Qué estoy viendo?** → Qué significan los datos, gráficos y métricas
- **¿Por qué importa?** → Para qué sirve cada paso en el flujo de ML
- **¿Qué decisiones estoy tomando?** → Qué implica elegir una opción u otra

Por ejemplo:
- Al cargar un dataset, explica qué es un dataset y qué tipos de datos existen
- Al elegir el target, explica la diferencia entre regresión y clasificación
- Al entrenar, explica qué significa train/test split y por qué se hace
- Al ver resultados, explica cómo interpretar accuracy, precision, recall, etc.

### Filosofía

> “Automatizar sin entender el proceso genera modelos frágiles; entender el proceso genera soluciones confiables.”

No se trata de ejecutar código mágico y ver números. Se trata de **entender el proceso** para poder tomar mejores decisiones cuando trabajes con tus propios datos.

---

## 🔧 Modo HERRAMIENTA (Tool Mode)

Para usuarios que ya conocen el proceso y solo quieren una herramienta rápida para:

- Probar distintos modelos con sus datos
- Comparar algoritmos fácilmente
- Exportar modelos entrenados
- Generar predicciones

Sin explicaciones adicionales, flujo directo al resultado.

---

## 🚀 Instalación

### Requisitos
- Python 3.10+

### Pasos

```bash
# Clonar el repositorio
git clone https://github.com/Juanarena29/ML-WorkFlow-Education-Tool.git
cd ML-WorkFlow-Education-Tool

# Crear entorno virtual
python -m venv venv

# Activar entorno (Windows)
.\venv\Scripts\Activate.ps1

# Activar entorno (Linux/Mac)
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la aplicación
streamlit run HOME.py
```

---

## 📂 Estructura del proyecto

```
ML-WorkFlow-Education-Tool/
├── HOME.py                 # Página principal
├── pages/                  # Páginas del flujo
│   ├── 1-LoadDataset.py
│   ├── 2-TypesDetection.py
│   ├── 3-CleaningConfig.py
│   ├── 4-EDA.py
│   ├── 5-Training.py
│   ├── 6-Results.py
│   └── 7-Prediction.py
├── src/                    # Lógica de negocio
│   ├── data/               # Carga, análisis y limpieza
│   ├── eda/                # Estadísticas y visualizaciones
│   ├── ml/                 # Modelos, pipelines y evaluación
│   └── utils/              # Sesión, constantes, file handling
├── tests/                  # Tests unitarios (pytest)
├── assets/                 # Estilos y datasets de ejemplo
├── models/                 # Modelos exportados (.pkl)
└── projectconfigs/         # Configuraciones guardadas
```

---

## 🤖 Modelos disponibles

### Clasificación
- Logistic Regression
- Random Forest
- Gradient Boosting
- SVC (Support Vector Classifier)
- XGBoost

### Regresión
- Linear Regression
- Ridge
- Lasso
- Random Forest
- Gradient Boosting
- XGBoost

Todos los modelos incluyen:
- Preprocesamiento automático (imputación, escalado, encoding)
- Opción de GridSearchCV para optimización de hiperparámetros
- Métricas completas de evaluación

---

## 📊 Métricas y visualizaciones

### Clasificación
- Accuracy, Precision, Recall, F1-Score
- ROC AUC (para clasificación binaria)
- Matriz de confusión (normal y normalizada)
- Curva ROC

### Regresión
- MAE (Error absoluto medio)
- RMSE (Error cuadrático medio)
- R² (Coeficiente de determinación)
- Gráfico de residuos

---

## ☁️ Deploy en Streamlit Cloud

La aplicación detecta automáticamente si está corriendo en Streamlit Cloud y aplica límites para evitar saturar recursos:

- Máximo 20,000 filas
- Máximo 100 columnas
- Máximo 3 folds en GridSearchCV

En modo local no hay límites.

---

## 🧪 Tests

```bash
# Ejecutar todos los tests
pytest tests/ -v

# Ejecutar con cobertura
pytest tests/ --cov=src
```

---

## 🛠️ Stack tecnológico

- **Frontend**: Streamlit
- **ML**: scikit-learn, XGBoost
- **Visualización**: Plotly
- **Data**: Pandas, NumPy

---

## 📝 Licencia

MIT

---


*Si este proyecto te resulta útil para aprender ML, ¡dale una ⭐ en GitHub!*
