"""
app.py
======
Executive Dashboard — home page and entry point for the AI-Powered Customer
Churn Prediction & Analytics System. Run with:

    streamlit run app.py

Additional pages live in the ``pages/`` directory and are auto-discovered
by Streamlit's native multi-page app support (shown in the sidebar).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))

from src import config, visualization as viz
from src.segmentation import radar_values_for_persona
from utils import ui
from utils.data_loader import (
    dataset_is_ready,
    get_available_models,
    get_model_comparison,
    get_raw_dataset,
    get_segmented_dataset,
    models_are_ready,
)

st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

ui.inject_global_css()

ui.page_header(
    config.APP_TITLE,
    "Deep learning + classical ML powered churn prediction with explainable, "
    "actionable retention analytics.",
    icon=config.APP_ICON,
)

with st.sidebar:
    st.markdown(f"### {config.APP_ICON} Navigation")
    st.caption("Use the pages above to explore analytics, run predictions, and review model performance.")
    st.divider()
    if models_are_ready():
        st.success(f"✅ {len(get_available_models())} model(s) trained and ready.")
    else:
        st.warning("⚠️ No models trained yet.")
        st.code("python -m src.training", language="bash")
    st.divider()
    st.caption("Built with Python, scikit-learn, XGBoost, TensorFlow/Keras, SHAP, and Streamlit.")

if not dataset_is_ready():
    st.error(
        f"Dataset not found at `{config.DATASET_PATH}`. Please place "
        f"`Churn_Modelling.csv` inside the `dataset/` folder."
    )
    st.stop()

df = get_raw_dataset()
comparison_df = get_model_comparison()

# --------------------------------------------------------------------------
# Executive Dashboard — tabbed layout
# --------------------------------------------------------------------------
tab_overview, tab_risk, tab_business, tab_reco = st.tabs(
    ["📊 Overview", "⚠️ Risk Analysis", "💡 Business Insights", "🎯 Recommendations"]
)

# ==========================================================================
# TAB 1 — OVERVIEW
# ==========================================================================
with tab_overview:
    total_customers = len(df)
    churned = int(df["Exited"].sum())
    churn_rate = churned / total_customers * 100
    active_members = int(df["IsActiveMember"].sum())
    avg_balance = df["Balance"].mean()

    ui.render_kpi_row(
        [
            ("Total Customers", f"{total_customers:,}", "Full dataset"),
            ("Churned Customers", f"{churned:,}", f"{churn_rate:.1f}% churn rate"),
            ("Active Members", f"{active_members:,}", f"{active_members/total_customers*100:.1f}% of base"),
            ("Avg. Balance", f"${avg_balance:,.0f}", "Across all customers"),
        ]
    )

    if not comparison_df.empty:
        best_row = comparison_df.sort_values("F1 Score", ascending=False).iloc[0]
        st.write("")
        ui.render_kpi_row(
            [
                ("Best Model", best_row["Model"], "Ranked by F1 Score"),
                ("Best Model Accuracy", f"{best_row['Accuracy']*100:.2f}%", ""),
                ("Best Model ROC AUC", f"{best_row.get('ROC AUC', 0)*100:.2f}%" if best_row.get("ROC AUC") else "N/A", ""),
                ("Models Trained", str(len(comparison_df)), "of 6 planned models"),
            ]
        )

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        sunburst_df = df.assign(Status=df["Exited"].map({0: "Retained", 1: "Churned"}))
        st.plotly_chart(
            viz.sunburst_chart(sunburst_df, ["Geography", "Gender", "Status"], None, "Customer Base — Geography → Gender → Status"),
            use_container_width=True,
        )
    with c2:
        try:
            segmented_df, profile_df, persona_df = get_segmented_dataset()
            st.plotly_chart(
                viz.treemap_chart(segmented_df, ["Persona", "Geography"], None, "Customer Base by Persona Segment"),
                use_container_width=True,
            )
        except Exception as exc:  # noqa: BLE001
            st.info(f"Segmentation unavailable: {exc}")

    with st.expander("📄 Dataset Summary", expanded=False):
        c1, c2 = st.columns([2, 1])
        with c1:
            st.dataframe(df.head(10), use_container_width=True)
        with c2:
            st.write("**Shape:**", df.shape)
            st.write("**Columns:**", len(df.columns))
            st.write("**Missing values:**", int(df.isnull().sum().sum()))
            st.write("**Target:** `Exited` (1 = churned, 0 = retained)")

    st.write("")
    st.subheader("Explore the System")

    nav_items = [
        ("📊", "Customer Analytics", "Demographic & behavioral breakdowns.", "pages/01_Customer_Analytics.py"),
        ("🔍", "Exploratory Data Analysis", "Correlations, distributions, pairplots.", "pages/02_Exploratory_Data_Analysis.py"),
        ("🎯", "Churn Prediction", "Predict churn risk for a single customer.", "pages/03_Churn_Prediction.py"),
        ("📁", "Batch CSV Prediction", "Score an entire customer list at once.", "pages/04_Batch_CSV_Prediction.py"),
        ("📈", "Model Performance", "Compare all 6 models, lift/gain, radar.", "pages/05_Model_Performance.py"),
        ("🧠", "Explainable AI", "SHAP global, local & force-plot explanations.", "pages/06_Explainable_AI.py"),
        ("💡", "Business Insights", "Segment-level risk & retention insights.", "pages/07_Business_Insights.py"),
        ("🧬", "Customer Segmentation", "KMeans personas: who to retain first.", "pages/08_Customer_Segmentation.py"),
        ("📑", "Reports Center", "Download CSV / Excel / PDF reports.", "pages/09_Reports_Center.py"),
        ("🧮", "Data Integration Center", "Upload, validate & audit new data.", "pages/10_Data_Integration_Center.py"),
        ("🛠️", "AI/ML Pipeline", "How raw data becomes a retention action.", "pages/11_Pipeline_Overview.py"),
    ]

    for row_start in range(0, len(nav_items), 4):
        cols = st.columns(4)
        for col, (icon, title, desc, target) in zip(cols, nav_items[row_start : row_start + 4]):
            with col:
                st.markdown(ui.nav_card(icon, title, desc), unsafe_allow_html=True)
                st.page_link(target, label=f"Open {title} →", use_container_width=True)

# ==========================================================================
# TAB 2 — RISK ANALYSIS
# ==========================================================================
with tab_risk:
    churned_df = df[df["Exited"] == 1]
    revenue_at_risk = churned_df["Balance"].sum()
    high_balance_threshold = df["Balance"].quantile(0.75)
    high_value_churned = churned_df[churned_df["Balance"] >= high_balance_threshold]

    ui.render_kpi_row(
        [
            ("Overall Churn Rate", f"{df['Exited'].mean()*100:.1f}%", ""),
            ("Churned Customers", f"{len(churned_df):,}", ""),
            ("Revenue at Risk (Balance)", f"${revenue_at_risk:,.0f}", "Sum of balances of churned customers"),
            ("High-Value Customers Lost", f"{len(high_value_churned):,}", "Top quartile balance"),
        ]
    )

    st.write("")
    age_bins_df = df.copy()
    age_bins_df["Age Group"] = pd.cut(
        age_bins_df["Age"], bins=[17, 30, 40, 50, 60, 100], labels=["18-30", "31-40", "41-50", "51-60", "60+"]
    )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(viz.churn_rate_by_category(age_bins_df, "Age Group", "Exited", "Churn Rate by Age Group"), use_container_width=True)
    with c2:
        st.plotly_chart(viz.churn_rate_by_category(df, "Geography", "Exited", "Churn Rate by Geography"), use_container_width=True)

    try:
        segmented_df, profile_df, persona_df = get_segmented_dataset()
        st.subheader("Persona Risk Radar")
        persona_choice = st.selectbox("Compare a persona's profile", profile_df["Persona"].unique(), key="risk_radar_persona")
        cats, vals = radar_values_for_persona(profile_df, persona_choice)
        if cats:
            st.plotly_chart(viz.radar_chart(cats, vals, f"{persona_choice} — Normalized Profile (0-100)"), use_container_width=True)
    except Exception as exc:  # noqa: BLE001
        st.info(f"Segmentation unavailable: {exc}")

    st.divider()
    st.subheader("High-Risk Customer Segment (Top Quartile Balance, Churned)")
    ui.interactive_table(
        high_value_churned[["CustomerId", "Surname", "Geography", "Age", "Balance", "NumOfProducts", "IsActiveMember"]].sort_values(
            "Balance", ascending=False
        ),
        key="overview_high_risk_table",
    )

# ==========================================================================
# TAB 3 — BUSINESS INSIGHTS
# ==========================================================================
with tab_business:

    def _generate_insights(df: pd.DataFrame) -> list[str]:
        insights = []
        age_bins_df = df.copy()
        age_bins_df["Age Group"] = pd.cut(
            age_bins_df["Age"], bins=[17, 30, 40, 50, 60, 100], labels=["18-30", "31-40", "41-50", "51-60", "60+"]
        )
        age_rates = age_bins_df.groupby("Age Group", observed=True)["Exited"].mean()
        worst_age = age_rates.idxmax()
        insights.append(f"Customers aged **{worst_age}** have the highest churn rate at **{age_rates.max()*100:.1f}%**.")

        geo_rates = df.groupby("Geography")["Exited"].mean()
        worst_geo = geo_rates.idxmax()
        insights.append(f"**{worst_geo}** has the highest churn rate among all geographies at **{geo_rates.max()*100:.1f}%**.")

        active_rate = df[df["IsActiveMember"] == 0]["Exited"].mean()
        inactive_vs_active = df[df["IsActiveMember"] == 1]["Exited"].mean()
        insights.append(
            f"Inactive members churn at **{active_rate*100:.1f}%** vs **{inactive_vs_active*100:.1f}%** for active members."
        )

        product_rates = df.groupby("NumOfProducts")["Exited"].mean()
        worst_product = product_rates.idxmax()
        insights.append(f"Customers with **{worst_product} product(s)** churn at the highest rate (**{product_rates.max()*100:.1f}%**).")
        return insights

    st.subheader("📌 Auto-Generated Strategic Insights")
    for insight in _generate_insights(df):
        st.markdown(f"- {insight}")

    st.divider()
    try:
        segmented_df, profile_df, persona_df = get_segmented_dataset()
        st.subheader("Revenue at Risk by Persona")
        st.plotly_chart(
            viz.treemap_chart(persona_df, ["Persona"], "Revenue at Risk", "Revenue at Risk by Persona"),
            use_container_width=True,
        )
        ui.interactive_table(persona_df, key="business_persona_summary")
    except Exception as exc:  # noqa: BLE001
        st.info(f"Segmentation unavailable: {exc}")

    st.caption("For the full drill-down (per-segment churn drivers, credit score effects, etc.) see the **Business Insights** page.")

# ==========================================================================
# TAB 4 — RECOMMENDATIONS
# ==========================================================================
with tab_reco:
    st.subheader("🎯 Retention Strategy Recommendations")

    try:
        segmented_df, profile_df, persona_df = get_segmented_dataset()
        from src.segmentation import PERSONA_DESCRIPTIONS

        for _, row in persona_df.iterrows():
            persona = row["Persona"]
            with st.container(border=True):
                st.markdown(f"#### {persona}")
                st.caption(PERSONA_DESCRIPTIONS.get(persona, ""))
                c1, c2, c3 = st.columns(3)
                c1.metric("Customers", f"{int(row['Customers']):,}")
                c2.metric("Churn Rate", f"{row['Churn_Rate']*100:.1f}%")
                c3.metric("Revenue at Risk", f"${row['Revenue at Risk']:,.0f}")
    except Exception as exc:  # noqa: BLE001
        st.info(f"Segmentation unavailable: {exc}")

    st.markdown(
        """
- **Targeted outreach for high-balance, inactive customers** — this segment carries the greatest revenue risk.
- **Product bundling campaigns** for single-product customers to increase switching costs.
- **Geography-specific retention offers** in the highest-churn market identified above.
- **Proactive engagement programs** (app nudges, loyalty perks) to reactivate inactive members.
- **Credit-building support** or tailored offers for lower credit-score segments showing elevated churn.
"""
    )
    st.caption("Full per-customer prioritization is available on the **Reports Center** and **Customer Segmentation** pages.")

st.divider()
st.caption(
    "AI-Powered Customer Churn Prediction & Analytics System · "
    "Built with Python, TensorFlow/Keras, scikit-learn, XGBoost, SHAP & Streamlit."
)
