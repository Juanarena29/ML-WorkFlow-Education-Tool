# ML WorkFlow Education Tool

An interactive web application for training Machine Learning models **without writing code**.

Designed both to **learn the fundamentals of Machine Learning** and to **use as a practical model training tool**.

---

## 🎯 What is it?

ML Workflow is a Streamlit-based application that guides the user step by step through the complete Machine Learning workflow:

1. **Load data** → Upload a CSV file  
2. **Detect types** → Identify numerical variables, categorical variables, and the target  
3. **Clean data** → Impute missing values, remove duplicates  
4. **Explore data (EDA)** → Visualize distributions, correlations, and relationships 
5. **Train models** → Select and train multiple algorithms  
6. **Analyze results** → Compare metrics, inspect confusion matrices, ROC curves  
7. **Predict** → Use trained models on new data  

---

## 📘 LEARN Mode

The core of this project is its **educational mode**.

### Who is it for?

- Students who are just starting with Machine Learning  
- Curious learners who want to understand what is behind predictions  
- Anyone who prefers learning by doing, not just reading  

### What makes it different?

At every step of the workflow, LEARN mode provides **contextual explanations** that answer:

- **What am I looking at?** → What the data, charts, and metrics mean  
- **Why does it matter?** → The purpose of each step in the ML workflow  
- **What decisions am I making?** → The implications of choosing one option over another  

For example:
- When loading a dataset, it explains what a dataset is and the different data types  
- When selecting the target, it explains the difference between regression and classification  
- During training, it explains what a train/test split is and why it is used  
- When reviewing results, it explains how to interpret accuracy, precision, recall, etc.  

### Philosophy

> **“The most costly errors in Machine Learning are not in the model, but in the steps before it.”**

This is not about running magical code and looking at numbers.  
It is about **understanding the process** in order to make better decisions when working with your own data.

---

## 🔧 TOOL Mode

For users who already understand the process and only want a fast, practical tool to:

- Test different models on their own data  
- Easily compare algorithms  
- Export trained models  
- Generate predictions  

No additional explanations, a direct path to results.

---

## 🚀 Installation

### Requirements
- Python 3.10+

### Steps

```bash
# Clone the repository
git clone https://github.com/Juanarena29/ML-WorkFlow-Education-Tool.git
cd ML-WorkFlow-Education-Tool

# Create virtual environment
python -m venv venv

# Activate environment (Windows)
.\venv\Scripts\Activate.ps1

# Activate environment (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run HOME.py
```

---

## 📂 Project structure

```
ML-WorkFlow-Education-Tool/
├── HOME.py                      # Main page
├── pages/                       # Workflow pages (thin orchestrators)
│   ├── 1-Carga de Dataset.py
│   ├── 2-Deteccion de tipos.py
│   ├── 3-Limpieza de datos.py
│   ├── 4-EDA.py
│   ├── 5-Entrenamiento.py
│   ├── 6-Resultados.py
│   └── 7-Predicciones.py
├── src/                         # Business logic (layered architecture)
│   ├── data/                    # Loading, analysis, validation, and cleaning
│   │   ├── analyzer.py
│   │   ├── cleaner.py
│   │   ├── loader.py
│   │   └── validator.py         # Dataset & prediction column validation
│   ├── eda/                     # Statistics and visualizations
│   │   ├── statistics.py
│   │   └── visualizations.py
│   ├── ml/                      # Models, pipelines, evaluation, and prediction
│   │   ├── evaluator.py         # Metrics, figures, train/test split
│   │   ├── models_config.py     # Available models & scoring options
│   │   ├── model_trainer.py     # Training orchestration
│   │   ├── pipeline_builder.py  # Preprocessing + model pipelines
│   │   └── predictor.py         # Secure model loading & prediction service
│   ├── savings/                 # Project state persistence
│   │   └── project_updates.py
│   ├── ui/                      # UI components (one sub-package per page)
│   │   ├── learn_explanations.py  # Centralized Learn-mode texts
│   │   ├── footer.py
│   │   ├── page1/               # Dataset upload UI
│   │   ├── page2/               # Type detection UI
│   │   ├── page3/               # Cleaning config UI
│   │   ├── page4/               # EDA UI
│   │   ├── page5/               # Training UI
│   │   ├── page6/               # Results UI
│   │   └── page7/               # Prediction UI
│   └── utils/                   # Session, constants, file handling
│       ├── constants.py
│       ├── file_handler.py
│       └── session.py           # MLProject dataclass & session management
├── tests/                       # Unit tests (pytest)
├── assets/                      # Styling and example datasets
├── models/                      # Exported models (.pkl)
└── projectconfigs/              # Saved configurations
```

---

## 🤖 Available models

### Classification
- Logistic Regression
- Random Forest
- Gradient Boosting
- SVC (Support Vector Classifier)
- XGBoost

### Regression
- Linear Regression
- Ridge
- Lasso
- Random Forest
- Gradient Boosting
- XGBoost

All models include:
- Automatic preprocessing (imputation, scaling, encoding)
- Optional GridSearchCV for hyperparameter optimization
- Complete evaluation metrics

---

## 📊 Metrics and visualizations

Classification
- Accuracy, Precision, Recall, F1-Score
- ROC AUC (binary classification)
- Confusion matrix (raw and normalized)
- ROC curve

### Regression
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)
- Residuals plot

---

## 🧪 Tests

```bash
# Run all tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=src
```

---

## 🏗️ Architecture

The project follows a **layered architecture** with clear separation of concerns:

| Layer | Location | Responsibility |
|-------|----------|----------------|
| **Pages** | `pages/` | Thin orchestrators (~60-80 lines). Import UI components and wire them together. |
| **UI Components** | `src/ui/pageN/` | Render Streamlit widgets. Receive data as parameters, no direct session access. |
| **Learn Texts** | `src/ui/learn_explanations.py` | All educational explanations centralized in one module. |
| **Services / ML** | `src/ml/` | Pure business logic: training, evaluation, prediction. No Streamlit imports. |
| **Data** | `src/data/` | Dataset loading, type analysis, validation, and cleaning. |
| **Utils** | `src/utils/` | Session state (`MLProject` dataclass), constants, file I/O. |

Plotly figure functions (in `evaluator.py` and `predictor.py`) return `go.Figure` objects without calling `st.plotly_chart`, keeping them testable and reusable.

### Layer coupling verification

- `src/ml/` — **zero** `import streamlit` statements. Pure Python + scikit-learn + Plotly.
- `src/eda/` — **zero** `import streamlit` statements. Pure Pandas + Plotly.
- `src/data/` — only `loader.py` imports Streamlit (for `@st.cache_data`). All other modules are Streamlit-free.
- `src/utils/session.py` — imports Streamlit for `st.session_state` management (expected, this is the boundary).

---

## ⚡ Performance

### Caching strategy

| Function | Cache decorator | Location |
|----------|----------------|----------|
| `_read_csv_cached` | `@st.cache_data` | `src/data/loader.py` |

CSV reads are cached by file content (bytes). Training, evaluation, and visualization functions are **not cached** by design — they depend on mutable `session_state` objects that Streamlit cannot hash reliably. The `MLProject` dataclass holds all derived state so re-computation only happens on explicit user actions (button clicks).

### Memory management

- **One copy**: `apply_cleaning_config` in `cleaner.py` creates a single defensive copy (`df.copy()`) to preserve `df_original`.
- **Zero-copy validation**: `validate_prediction_columns` in `validator.py` avoids unnecessary copies — `drop()` already returns a new DataFrame.
- **Lazy rendering**: Results page charts render inside `st.expander()` — Plotly figures are only computed when the user expands a model section.
- **Cloud limits**: `truncate_dataset_if_needed` enforces 20k rows / 100 columns on Streamlit Cloud, preventing memory exhaustion.

### Streamlit Cloud deployment limits

| Resource | Limit |
|----------|-------|
| Rows | 20,000 |
| Columns | 100 |
| GridSearchCV folds | 3 |

No limits in local mode.

---

## 🔒 Model security

Loading `.pkl` files involves `pickle` deserialization, which can execute arbitrary code. The application mitigates this with:

- **Filename validation** — blocks path traversal characters (`..`, `/`, `\`) and non-`.pkl` extensions (`validate_model_filename` in `predictor.py`).
- **SHA-256 hash registry** — every model saved through the app is registered in `models/.model_hashes.json`. When loading, the hash is verified to detect external tampering (`register_model_hash` / `verify_model_integrity`).
- **Integrity warnings** — if a model was not saved by the app or its hash does not match, the user sees a warning before loading.
- **Safe loading pipeline** — `load_model_safe()` chains filename validation + integrity check before calling `joblib.load()`.
- **Input validation** — `validate_prediction_columns` in `validator.py` blocks prediction if the CSV columns don't match the trained schema.

These measures protect against accidental loading of untrusted files. For production environments, consider restricting the `models/` directory permissions.

---

## 🛠️ Tech stack

- **Frontend**: Streamlit
- **Machine Learning**: scikit-learn, XGBoost
- **Visualization**: Plotly
- **Data**: Pandas, NumPy

---

## 📝 License

MIT

---


*If this project helps you learn Machine Learning, consider giving it a ⭐ on GitHub!*
