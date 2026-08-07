"""
config.py
=========
Central configuration for the Customer Churn Prediction & Analytics System.

Every path, filename, and shared constant used across the project (training
scripts, Streamlit dashboard, FastAPI service) is defined here so that there
is a single source of truth and no duplicated "magic strings" anywhere else
in the codebase.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Base directories
# --------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parent.parent

DATASET_DIR: Path = ROOT_DIR / "dataset"
MODELS_DIR: Path = ROOT_DIR / "models"
OUTPUTS_DIR: Path = ROOT_DIR / "outputs"
ASSETS_DIR: Path = ROOT_DIR / "assets"
NOTEBOOKS_DIR: Path = ROOT_DIR / "notebooks"

for _dir in (DATASET_DIR, MODELS_DIR, OUTPUTS_DIR, ASSETS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
DATASET_PATH: Path = DATASET_DIR / "Churn_Modelling.csv"

RAW_ID_COLUMNS = ["RowNumber", "CustomerId", "Surname"]
TARGET_COLUMN = "Exited"

# Columns a raw/manual customer record must contain (before feature
# engineering) in order to run a prediction.
RAW_INPUT_COLUMNS = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]

# --------------------------------------------------------------------------
# Model artifact filenames
# --------------------------------------------------------------------------
SCALER_PATH = MODELS_DIR / "scaler.pkl"
GENDER_ENCODER_PATH = MODELS_DIR / "gender_encoder.pkl"
FEATURE_ORDER_PATH = MODELS_DIR / "feature_order.json"
GEOGRAPHY_CATEGORIES_PATH = MODELS_DIR / "geography_categories.json"
METRICS_PATH = MODELS_DIR / "model_metrics.json"
TRAINING_HISTORY_PATH = MODELS_DIR / "ann_training_history.json"
SHAP_BACKGROUND_PATH = MODELS_DIR / "shap_background.pkl"

MODEL_FILENAMES = {
    "Logistic Regression": "logistic_regression.pkl",
    "KNN Classifier": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgb.pkl",
    "ANN": "ann_model.keras",
}

MODEL_PATHS = {name: MODELS_DIR / fname for name, fname in MODEL_FILENAMES.items()}

# --------------------------------------------------------------------------
# Risk categorization thresholds (used for prediction + business insights)
# --------------------------------------------------------------------------
RISK_THRESHOLDS = {
    "Low": (0.0, 0.30),
    "Medium": (0.30, 0.60),
    "High": (0.60, 1.01),
}

BUSINESS_RECOMMENDATIONS = {
    "High": [
        "Contact the customer within 48 hours via a personal outreach call.",
        "Offer a loyalty / retention discount or fee waiver.",
        "Assign a dedicated relationship manager for the next quarter.",
        "Proactively review product fit (cross-sell a better-suited product).",
    ],
    "Medium": [
        "Send a targeted retention email campaign with a tailored offer.",
        "Encourage adoption of a second product to increase stickiness.",
        "Monitor activity for the next 30-60 days for early warning signs.",
    ],
    "Low": [
        "No immediate action required — continue standard engagement.",
        "Consider upsell / cross-sell opportunities to deepen the relationship.",
    ],
}

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20

# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------
APP_TITLE = "AI-Powered Customer Churn Prediction & Analytics System"
APP_ICON = "🏦"

# Silence noisy libraries at import-time (safe if package absent)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
