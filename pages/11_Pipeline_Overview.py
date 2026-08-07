"""AI/ML Pipeline page: a visual workflow diagram of the end-to-end system."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import visualization as viz
from utils import ui
from utils.data_loader import get_available_models, models_are_ready

st.set_page_config(page_title="AI/ML Pipeline", page_icon="🛠️", layout="wide")
ui.inject_global_css()
ui.page_header("AI/ML Pipeline", "How raw customer data becomes an explainable, actionable retention decision.", "🛠️")

st.plotly_chart(viz.pipeline_diagram_figure(), use_container_width=True)

st.divider()

stage_cols = st.columns(3)

with stage_cols[0]:
    st.markdown("#### 1️⃣ Data Layer")
    st.markdown(
        "- **Source:** `Churn_Modelling.csv` (10,000 bank customers)\n"
        "- **Cleaning:** median/mode imputation if nulls are found\n"
        "- **Encoding:** Gender label-encoded, Geography one-hot encoded\n"
        "- **Feature engineering:** `balance_to_salary`, `tenure_by_age`\n"
        "- Governed by the **Data Integration Center** page for new uploads"
    )
    st.markdown("#### 2️⃣ Modeling Layer")
    st.markdown(
        "- 80/20 stratified train/test split (`random_state=42`)\n"
        "- Features standardized with `StandardScaler`\n"
        "- Six models trained: Logistic Regression, KNN, Naive Bayes, "
        "Random Forest, XGBoost, ANN (Keras)\n"
        "- Artifacts (models, scaler, encoders) saved to `models/`"
    )

with stage_cols[1]:
    st.markdown("#### 3️⃣ Evaluation Layer")
    st.markdown(
        "- Accuracy, Precision, Recall, F1, ROC AUC per model\n"
        "- ROC / Precision-Recall / Lift & Gain curves\n"
        "- Confusion matrix + full classification report\n"
        "- Radar chart profile per model on the **Model Performance** page"
    )
    st.markdown("#### 4️⃣ Explainability Layer")
    st.markdown(
        "- SHAP `TreeExplainer` (RF, XGBoost) or `KernelExplainer` (others)\n"
        "- Global feature importance across a background sample\n"
        "- Local waterfall & force plots for a single customer\n"
        "- Auto-generated plain-English driver sentences"
    )

with stage_cols[2]:
    st.markdown("#### 5️⃣ Serving Layer")
    st.markdown(
        "- **Streamlit dashboard** — single & batch predictions, analytics\n"
        "- **FastAPI service** — programmatic REST access\n"
        "- Same `prepare_features()` pipeline used everywhere → no "
        "train/serve skew"
    )
    st.markdown("#### 6️⃣ Business Action Layer")
    st.markdown(
        "- Risk categorization (Low / Medium / High) from calibrated "
        "probability thresholds\n"
        "- KMeans persona segmentation (**Customer Segmentation** page)\n"
        "- Retention playbooks per persona/risk tier\n"
        "- Exportable CSV / Excel / PDF reports (**Reports Center**)"
    )

st.divider()
st.subheader("Current Deployment Status")
if models_are_ready():
    st.success(f"✅ {len(get_available_models())} of 6 planned models are trained and serving: {', '.join(get_available_models())}")
else:
    st.warning("⚠️ No models trained yet. Run `python -m src.training` to populate the modeling and evaluation layers.")
