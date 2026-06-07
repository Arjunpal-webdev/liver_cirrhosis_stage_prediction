# 🩺Liver Cirrhosis Stage Prediction & Analytics Dashboard

> **AI-powered clinical decision support system** for predicting liver cirrhosis stages using a tuned XGBoost model with SHAP explainability.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-orange)](https://xgboost.readthedocs.io)
[![Accuracy](https://img.shields.io/badge/Accuracy-91%25-green)](https://xgboost.readthedocs.io)
[![ROC AUC](https://img.shields.io/badge/ROC%20AUC-0.9798-purple)](https://xgboost.readthedocs.io)

---

## 🌟 Features

| Feature | Description |
|---------|-------------|
| **Single Prediction** | Manual patient data entry with real-time stage prediction |
| **SHAP Explainability** | Waterfall plots, force plots, and feature contribution tables |
| **Batch Prediction** | CSV upload for multiple patients with downloadable results |
| **Model Analytics** | 7-tab analytics dashboard with comprehensive performance metrics |
| **Interactive Charts** | Plotly visualizations for ROC curves, confusion matrix, feature importance |
| **Dark Theme** | Professional healthcare dark UI with glassmorphism effects |

---

## 📁 Project Structure

```
liver_cirrhosis_stage_pred/
│
├── app/
│   ├── pages/
│   │   ├── single_prediction.py      # Single patient prediction UI
│   │   ├── batch_prediction.py       # Batch CSV prediction UI
│   │   └── model_analytics.py        # 7-tab model analytics dashboard
│   │
│   ├── components/
│   │   └── ui_components.py          # Reusable UI components
│   │
│   ├── assets/                        # Static assets
│   └── styles/
│       └── main.css                   # Global dark theme CSS
│
├── data/
│   └── liver_cirrhosis.csv            # Dataset
│
├── models/
│   ├── best_xgb_model.pkl             # Tuned XGBoost classifier
│   └── label_encoder.pkl              # Label encoder (Edema column)
│
├── notebooks/
│   ├── liver_cirrhosis_model_training.ipynb
│   └── liver_cirrhosis_stage_eda.ipynb
│
├── services/
│   ├── prediction_service.py          # Model loading & inference
│   ├── preprocessing_service.py       # Feature encoding & transformation
│   └── analytics_service.py          # SHAP & visualization builders
│
├── utils/
│   └── helpers.py                     # Utility functions
│
├── .streamlit/
│   └── config.toml                    # Streamlit dark theme config
│
├── main.py                            # 🚀 Application entry point
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## 🚀 Quick Start

### 1. Clone / Navigate to Project
```bash
cd liver_cirrhosis_stage_pred
```

### 2. Create Virtual Environment (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Dashboard
```bash
streamlit run main.py
```

The dashboard opens at **http://localhost:8501** 🎉

---

## 🤖 Model Details

| Attribute | Value |
|-----------|-------|
| **Algorithm** | XGBoost Classifier (Tuned) |
| **Tuning Method** | RandomizedSearchCV (50 iterations, 3-fold CV) |
| **n_estimators** | 200 |
| **max_depth** | 7 |
| **learning_rate** | 0.1 |
| **subsample** | 0.8 |
| **colsample_bytree** | 0.6 |
| **gamma** | 0.2 |
| **Accuracy** | 91.0% |
| **Macro ROC AUC** | 0.9798 |
| **Classes** | Stage 1, Stage 2, Stage 3 |

---

## 📊 Input Features

### Numerical Features (11)
| Feature | Type | Description |
|---------|------|-------------|
| N_Days | int | Days from registration to event |
| Age | int | Patient age (**collected in years**, converted to days) |
| Bilirubin | float | Serum bilirubin (mg/dL) |
| Cholesterol | float | Serum cholesterol (mg/dL) |
| Albumin | float | Serum albumin (g/dL) |
| Copper | float | Urine copper (μg/day) |
| Alk_Phos | float | Alkaline phosphotase (U/liter) |
| SGOT | float | SGOT/AST (U/mL) |
| Tryglicerides | float | Triglycerides (mg/dL) |
| Platelets | float | Platelets (ml/1000) |
| Prothrombin | float | Prothrombin time (seconds) |

### Categorical Features (7)
| Feature | Options |
|---------|---------|
| Status | C (Censored), D (Dead), CL (Censored/Transplant) |
| Drug | Placebo, D-penicillamine |
| Sex | F, M |
| Ascites | N, Y |
| Hepatomegaly | N, Y |
| Spiders | N, Y |
| Edema | N, S, Y |

---

## 🔄 Data Pipeline

```
User Input (Age in Years)
    ↓
Age → Days (× 365.25)
    ↓
Create DataFrame (exact column order)
    ↓
Label Encode Categoricals (fixed mapping)
    ↓
XGBoost Model Prediction
    ↓
Map Output: 0→Stage 1, 1→Stage 2, 2→Stage 3
    ↓
Display Results + SHAP Explanations
```

---

## 📱 Dashboard Sections

### 🔬 Section 1 — Single Prediction
- Form-based patient data entry (18 features)
- Age collected in years, auto-converted to days
- Real-time stage prediction with confidence
- Probability gauge chart + bar chart
- SHAP waterfall plot
- Top positive / negative feature drivers
- Full SHAP contribution table

### 📋 Section 2 — Batch Prediction
- CSV file upload with column validation
- Preview uploaded data
- Progress tracking during prediction
- Stage distribution pie chart
- Full results table with color-coded stages
- Download predictions as CSV

### 📊 Section 3 — Model Analytics (7 Tabs)
1. **Overview** — Model specs, dataset info, hyperparameters
2. **Comparison** — Bar charts, radar charts, model rankings
3. **ROC AUC** — Multi-class ROC curves, XGBoost vs CatBoost comparison
4. **Confusion Matrix** — Interactive heatmap with counts & percentages
5. **Classification Report** — Per-class precision, recall, F1, support
6. **Feature Importance** — Top 10/15/all features, treemap visualization

---

## ⚠️ Important Notes

1. **Age Handling**: The model was trained with Age in **days**. The UI collects age in **years** and automatically converts it using `age_days = int(age_years * 365.25)`.

2. **Categorical Encoding**: Fixed label encoding maps are used (no re-fitting). The saved `label_encoder.pkl` is used only for reference.

3. **No Scaling**: The model requires no feature scaling — raw values are used directly.

4. **Medical Disclaimer**: This tool is for **research use only** and is not a substitute for clinical diagnosis by qualified medical professionals.

---

## 🛠️ Technology Stack

- **Streamlit** — Web UI framework
- **XGBoost** — Gradient boosting classifier
- **SHAP** — Model explainability
- **Plotly** — Interactive visualizations
- **Pandas / NumPy** — Data processing
- **scikit-learn** — PCA, metrics

---

## 📜 License

This project is for educational and research purposes only.

---

*Built with ❤️ for AI-powered healthcare decision support*
