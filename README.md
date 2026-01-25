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
├── HOME.py                 # Main page
├── pages/                  # Workflow pages
│   ├── 1-LoadDataset.py
│   ├── 2-TypesDetection.py
│   ├── 3-CleaningConfig.py
│   ├── 4-EDA.py
│   ├── 5-Training.py
│   ├── 6-Results.py
│   └── 7-Prediction.py
├── src/                    # Business logic
│   ├── data/               # Loading, analysis, and cleaning
│   ├── eda/                # Statistics and visualizations
│   ├── ml/                 # Models, pipelines, and evaluation
│   └── utils/              # Session, constants, file handling
├── tests/                  # Unit tests (pytest)
├── assets/                 # Styling and example datasets
├── models/                 # Exported models (.pkl)
└── projectconfigs/         # Saved configurations
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

## ☁️ Streamlit Cloud deployment

The application automatically detects when it is running on Streamlit Cloud and applies limits to avoid resource saturation:

- Maximum 20,000 rows
- Maximum 100 columns
- Maximum 3 folds in GridSearchCV

No limits are applied in local mode.

---

## 🧪 Tests

```bash
# Run all tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=src
```

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
