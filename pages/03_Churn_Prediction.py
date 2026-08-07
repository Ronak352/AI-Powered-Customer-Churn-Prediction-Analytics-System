"""Customer Churn Prediction page: manual input, probability, gauge, recommendation."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import visualization as viz
from src.ai_insights import generate_narrative
from src.prediction import predict_single
from src.report_generator import build_prediction_pdf
from utils import ui
from utils.data_loader import get_available_models, models_are_ready

st.set_page_config(page_title="Churn Prediction", page_icon="🎯", layout="wide")
ui.inject_global_css()
ui.page_header("Customer Churn Prediction", "Predict churn risk for a single customer in real time.", "🎯")

if not models_are_ready():
    ui.models_missing_warning()
    st.stop()

with st.sidebar:
    st.subheader("🤖 Model Selection")

    available_models = get_available_models()

    model_choice = st.selectbox(
        "Prediction model",
        available_models
    )

    if "(Not Available)" in model_choice:
        st.warning(
            f"{model_choice} is not available. "
            "Please train the model first."
        )
        st.stop()

    # Remove suffix before sending to prediction engine
    model_choice = model_choice.replace(" (Not Available)", "")

    st.caption(
        "Different models may give slightly different probabilities — compare them on the Model Performance page."
    )

    st.markdown("---")

    st.markdown("### Available Models")

    for model in available_models:
        if "(Not Available)" in model:
            st.markdown(f"❌ {model}")
        else:
            st.markdown(f"✅ {model}")


st.subheader("Customer Details")

with st.form("prediction_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        credit_score = st.slider("Credit Score", 300, 900, 650)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 100, 40)
    with c2:
        tenure = st.slider("Tenure (years with bank)", 0, 10, 5)
        balance = st.number_input("Account Balance ($)", min_value=0.0, value=50000.0, step=1000.0)
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
        estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=100000.0, step=1000.0)
    with c3:
        has_cr_card = st.radio("Has Credit Card?", ["Yes", "No"], horizontal=True)
        is_active = st.radio("Is Active Member?", ["Yes", "No"], horizontal=True)
        st.write("")
        st.write("")
        submitted = st.form_submit_button("🎯 Predict Churn", use_container_width=True, type="primary")

if submitted:
    customer = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": 1 if has_cr_card == "Yes" else 0,
        "IsActiveMember": 1 if is_active == "Yes" else 0,
        "EstimatedSalary": estimated_salary,
    }

    with st.spinner("Running prediction..."):
        result = predict_single(customer, model_name=model_choice)

    st.divider()
    res_col, gauge_col = st.columns([1, 1])

    # Try to pull the top SHAP drivers for this customer. Explainability is
    # optional (needs the shap package and a compatible model), so any
    # failure here just means the report/narrative skip this section.
    drivers = None
    try:
        from src.explainability import business_friendly_explanation, local_shap_values

        shap_values, feature_names, _, _ = local_shap_values(model_choice, customer)
        drivers = business_friendly_explanation(shap_values, feature_names)
    except Exception:
        drivers = None

    with res_col:
        verdict = "🔴 Will Churn" if result.will_churn else "🟢 Will Stay"
        st.markdown(f"## {verdict}")
        st.markdown(ui.risk_badge(result.risk_level), unsafe_allow_html=True)
        st.write("")
        ui.render_kpi_row(
            [
                ("Churn Probability", f"{result.probability*100:.1f}%", ""),
                ("Confidence Score", f"{result.confidence*100:.1f}%", ""),
                ("Model Used", result.model_used, ""),
            ]
        )
        st.write("")
        st.markdown("#### 💡 Business Recommendations")
        for rec in result.recommendations:
            st.markdown(f"- {rec}")

    with gauge_col:
        st.plotly_chart(viz.gauge_chart(result.probability, "Churn Risk"), height=350, use_container_width=True)

    st.divider()
    st.markdown("#### 🧠 AI Summary")
    narrative = generate_narrative(customer, result.as_dict(), drivers)
    st.info(narrative)

    st.markdown("#### 📄 Download Report")
    pdf_bytes = build_prediction_pdf(customer, result.as_dict(), explanation=drivers)
    st.download_button(
        "⬇️ Download Prediction Report (PDF)",
        data=pdf_bytes,
        file_name=f"churn_report_{model_choice.replace(' ', '_').lower()}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

    with st.expander("View raw prediction payload"):
        st.json(result.as_dict())
else:
    ui.empty_state("Fill in the customer details above and click **Predict Churn** to see the result.")
