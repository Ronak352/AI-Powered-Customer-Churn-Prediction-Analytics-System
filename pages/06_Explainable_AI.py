"""Explainable AI page: SHAP-powered global and local explanations."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import visualization as viz
from src.explainability import (
    ShapNotAvailableError,
    business_friendly_explanation,
    global_shap_values,
    local_shap_values,
    top_features,
)
from src.prediction import predict_single
from utils import ui
from utils.data_loader import get_available_models, models_are_ready

st.set_page_config(page_title="Explainable AI", page_icon="🧠", layout="wide")
ui.inject_global_css()
ui.page_header("Explainable AI", "SHAP-powered global and local explanations for every prediction.", "🧠")

if not models_are_ready():
    ui.models_missing_warning()
    st.stop()

# Tree-based models give the fastest, most reliable SHAP explanations.
preferred_order = ["Random Forest", "XGBoost", "Logistic Regression", "ANN", "KNN Classifier", "Naive Bayes"]
available = get_available_models()
ordered_available = [m for m in preferred_order if m in available] + [m for m in available if m not in preferred_order]

with st.sidebar:
    st.subheader("Model Selection")
    model_choice = st.selectbox("Model to explain", ordered_available)
    st.caption("Tree-based models (Random Forest, XGBoost) render fastest since they use SHAP's TreeExplainer.")

try:
    import shap  # noqa: F401

    shap_installed = True
except ImportError:
    shap_installed = False

if not shap_installed:
    st.warning(
        "⚠️ SHAP is not installed in this environment. Install it to enable "
        "Explainable AI features:",
        icon="⚠️",
    )
    st.code("pip install shap", language="bash")
    st.info(
        "Once installed, this page will show:\n"
        "- **Global explanations**: SHAP summary plot & feature importance ranking\n"
        "- **Local explanations**: SHAP waterfall plot for any single customer\n"
        "- **Business-friendly explanations**: plain-English drivers behind each prediction"
    )
    st.stop()

tab1, tab2 = st.tabs(["🌍 Global Explanation", "🔍 Local Explanation (Single Customer)"])

with tab1:
    st.markdown("#### Global Feature Importance")
    st.caption(f"Computed across a background sample of training data using **{model_choice}**.")

    sample_size = st.slider("Background sample size", 20, 200, 100, step=20)

    if st.button("Compute Global SHAP Values", type="primary"):
        with st.spinner("Computing SHAP values — this can take a moment for non-tree models..."):
            try:
                shap_values, feature_names, X_sample = global_shap_values(model_choice, sample_size=sample_size)
            except ShapNotAvailableError as exc:
                st.error(str(exc))
                st.stop()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not compute SHAP values: {exc}")
                st.stop()

        mean_abs_importance = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({"Feature": feature_names, "Mean |SHAP value|": mean_abs_importance}).sort_values(
            "Mean |SHAP value|", ascending=True
        )

        import plotly.express as px

        fig = px.bar(
            importance_df, x="Mean |SHAP value|", y="Feature", orientation="h",
            color="Mean |SHAP value|", color_continuous_scale="Blues", template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

        positive, negative = top_features(shap_values, feature_names)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 🔺 Top Churn-Increasing Features")
            for f in positive:
                st.markdown(f"- {f}")
        with c2:
            st.markdown("##### 🔻 Top Churn-Reducing Features")
            for f in negative:
                st.markdown(f"- {f}")
    else:
        ui.empty_state("Click **Compute Global SHAP Values** to generate the summary plot and feature ranking.")

with tab2:
    st.markdown("#### Explain a Single Customer's Prediction")

    with st.form("shap_customer_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            credit_score = st.slider("Credit Score", 300, 900, 650, key="shap_cs")
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"], key="shap_geo")
            gender = st.selectbox("Gender", ["Male", "Female"], key="shap_gender")
            age = st.slider("Age", 18, 100, 40, key="shap_age")
        with c2:
            tenure = st.slider("Tenure", 0, 10, 5, key="shap_tenure")
            balance = st.number_input("Balance ($)", min_value=0.0, value=50000.0, step=1000.0, key="shap_bal")
            num_products = st.selectbox("Number of Products", [1, 2, 3, 4], key="shap_prod")
            estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=100000.0, step=1000.0, key="shap_sal")
        with c3:
            has_cr_card = st.radio("Has Credit Card?", ["Yes", "No"], horizontal=True, key="shap_card")
            is_active = st.radio("Is Active Member?", ["Yes", "No"], horizontal=True, key="shap_active")
            explain_submit = st.form_submit_button("🔍 Explain Prediction", type="primary", use_container_width=True)

    if explain_submit:
        customer = {
            "CreditScore": credit_score, "Geography": geography, "Gender": gender, "Age": age,
            "Tenure": tenure, "Balance": balance, "NumOfProducts": num_products,
            "HasCrCard": 1 if has_cr_card == "Yes" else 0,
            "IsActiveMember": 1 if is_active == "Yes" else 0, "EstimatedSalary": estimated_salary,
        }

        with st.spinner("Computing local SHAP explanation..."):
            result = predict_single(customer, model_name=model_choice)
            try:
                shap_row, feature_names, X_row, base_value = local_shap_values(model_choice, customer)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not compute local SHAP values: {exc}")
                st.stop()

        st.markdown(f"**Prediction:** {'🔴 Will Churn' if result.will_churn else '🟢 Will Stay'} "
                    f"({result.probability*100:.1f}% probability) — {ui.risk_badge(result.risk_level)}", unsafe_allow_html=True)

        waterfall_df = pd.DataFrame({"Feature": feature_names, "SHAP Value": shap_row}).sort_values("SHAP Value")

        import plotly.graph_objects as go

        fig = go.Figure(
            go.Waterfall(
                orientation="h",
                y=waterfall_df["Feature"],
                x=waterfall_df["SHAP Value"],
                connector={"line": {"color": "rgba(128,128,128,0.4)"}},
            )
        )
        fig.update_layout(template="plotly_white", title=f"SHAP Waterfall — base value {base_value:.3f}")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### ➡️ SHAP Force Plot")
        st.caption("Each feature pushes the prediction up (red, increases churn risk) or down (green, decreases it) from the base value.")
        prediction_value = base_value + float(np.sum(shap_row))
        force_fig = viz.force_plot_figure(
            feature_names, shap_row, base_value, prediction_value,
            title=f"{model_choice} — Force Plot for This Customer",
        )
        st.plotly_chart(force_fig, use_container_width=True)

        st.markdown("##### 💬 Business-Friendly Explanation")
        for sentence in business_friendly_explanation(shap_row, feature_names):
            st.markdown(f"- {sentence}")
    else:
        ui.empty_state("Fill in the customer details and click **Explain Prediction**.")
