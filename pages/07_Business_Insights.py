"""Business Insights page: segmentation, risk analysis, retention strategies."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import config, visualization as viz
from src.segmentation import PERSONA_DESCRIPTIONS, radar_values_for_persona
from utils import ui
from utils.data_loader import dataset_is_ready, get_raw_dataset, get_segmented_dataset

st.set_page_config(page_title="Business Insights", page_icon="💡", layout="wide")
ui.inject_global_css()
ui.page_header("Business Insights", "Segment-level churn risk analysis and retention recommendations.", "💡")

if not dataset_is_ready():
    st.error("Dataset not found. Please add `Churn_Modelling.csv` to the `dataset/` folder.")
    st.stop()

df = get_raw_dataset()

churned_df = df[df["Exited"] == 1]
revenue_at_risk = churned_df["Balance"].sum()
high_balance_threshold = df["Balance"].quantile(0.75)
high_value_churned = churned_df[churned_df["Balance"] >= high_balance_threshold]

age_bins_df = df.copy()
age_bins_df["Age Group"] = pd.cut(
    age_bins_df["Age"], bins=[17, 30, 40, 50, 60, 100], labels=["18-30", "31-40", "41-50", "51-60", "60+"]
)


def _generate_insights(df: pd.DataFrame) -> list[str]:
    """Derive a handful of headline insights directly from the data so this
    section always reflects the actual dataset rather than hardcoded text.
    """
    insights = []

    age_rates = age_bins_df.groupby("Age Group", observed=True)["Exited"].mean()
    worst_age = age_rates.idxmax()
    insights.append(f"Customers aged **{worst_age}** have the highest churn rate at **{age_rates.max()*100:.1f}%**.")

    geo_rates = df.groupby("Geography")["Exited"].mean()
    worst_geo = geo_rates.idxmax()
    insights.append(f"**{worst_geo}** has the highest churn rate among all geographies at **{geo_rates.max()*100:.1f}%**.")

    active_rate = df[df["IsActiveMember"] == 0]["Exited"].mean()
    inactive_vs_active = df[df["IsActiveMember"] == 1]["Exited"].mean()
    insights.append(
        f"Inactive members churn at **{active_rate*100:.1f}%** vs **{inactive_vs_active*100:.1f}%** for active members — "
        f"activity is a strong retention signal."
    )

    product_rates = df.groupby("NumOfProducts")["Exited"].mean()
    worst_product = product_rates.idxmax()
    insights.append(f"Customers with **{worst_product} product(s)** churn at the highest rate (**{product_rates.max()*100:.1f}%**).")

    low_credit = df[df["CreditScore"] < df["CreditScore"].median()]["Exited"].mean()
    high_credit = df[df["CreditScore"] >= df["CreditScore"].median()]["Exited"].mean()
    insights.append(
        f"Below-median credit score customers churn at **{low_credit*100:.1f}%** vs **{high_credit*100:.1f}%** "
        f"for above-median — lower credit score correlates with higher churn."
    )

    high_bal_rate = df[df["Balance"] >= high_balance_threshold]["Exited"].mean()
    low_bal_rate = df[df["Balance"] < high_balance_threshold]["Exited"].mean()
    insights.append(
        f"Top-quartile balance customers churn at **{high_bal_rate*100:.1f}%** vs **{low_bal_rate*100:.1f}%** "
        f"for the rest — high-balance customers are disproportionately likely to leave."
    )

    return insights


RETENTION_RECOMMENDATIONS = [
    "**Targeted outreach for high-balance, inactive customers** — this segment carries the greatest revenue risk.",
    "**Product bundling campaigns** for single-product customers to increase switching costs.",
    "**Geography-specific retention offers** in the highest-churn market identified above.",
    "**Proactive engagement programs** (app nudges, loyalty perks) to reactivate inactive members.",
    "**Credit-building support** or tailored offers for lower credit-score segments showing elevated churn.",
]

tab_revenue, tab_segments, tab_retention, tab_exec = st.tabs(
    ["💰 Revenue Risk", "🧬 Segments", "🎯 Retention", "📑 Executive Report"]
)

# ==========================================================================
# REVENUE RISK
# ==========================================================================
with tab_revenue:
    ui.render_kpi_row(
        [
            ("Overall Churn Rate", f"{df['Exited'].mean()*100:.1f}%", ""),
            ("Churned Customers", f"{len(churned_df):,}", ""),
            ("Revenue at Risk (Balance)", f"${revenue_at_risk:,.0f}", "Sum of balances of churned customers"),
            ("High-Value Customers Lost", f"{len(high_value_churned):,}", "Top quartile balance"),
        ]
    )

    st.write("")
    st.subheader("Churn Rate by Segment")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(viz.churn_rate_by_category(age_bins_df, "Age Group", "Exited", "Churn Rate by Age Group"), use_container_width=True)
    with c2:
        st.plotly_chart(viz.churn_rate_by_category(df, "Geography", "Exited", "Churn Rate by Geography"), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(viz.churn_rate_by_category(df, "Gender", "Exited", "Churn Rate by Gender"), use_container_width=True)
    with c2:
        st.plotly_chart(viz.churn_rate_by_category(df, "NumOfProducts", "Exited", "Churn Rate by Number of Products"), use_container_width=True)

    st.plotly_chart(viz.churn_rate_by_category(df, "IsActiveMember", "Exited", "Churn Rate: Active vs Inactive Members"), use_container_width=True)

    st.divider()
    st.subheader("High-Risk Customer Segment (Top Quartile Balance, Churned)")
    ui.interactive_table(
        high_value_churned[["CustomerId", "Surname", "Geography", "Age", "Balance", "NumOfProducts", "IsActiveMember"]].sort_values(
            "Balance", ascending=False
        ),
        key="business_insights_high_risk_table",
    )

# ==========================================================================
# SEGMENTS
# ==========================================================================
with tab_segments:
    try:
        segmented_df, profile_df, persona_df = get_segmented_dataset()
    except Exception as exc:  # noqa: BLE001
        st.info(f"Segmentation unavailable: {exc}")
        segmented_df = profile_df = persona_df = None

    if persona_df is not None:
        st.subheader("Persona Summary")
        ui.interactive_table(persona_df, key="business_segments_persona_table")

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(viz.treemap_chart(segmented_df, ["Persona"], None, "Customers by Persona"), use_container_width=True)
        with c2:
            st.plotly_chart(viz.treemap_chart(persona_df, ["Persona"], "Revenue at Risk", "Revenue at Risk by Persona"), use_container_width=True)

        st.subheader("Persona Profile Radar")
        persona_choice = st.selectbox("Persona", profile_df["Persona"].unique(), key="segments_radar_persona")
        cats, vals = radar_values_for_persona(profile_df, persona_choice)
        if cats:
            st.plotly_chart(viz.radar_chart(cats, vals, f"{persona_choice} — Normalized Profile (0-100)"), use_container_width=True)
        st.caption(PERSONA_DESCRIPTIONS.get(persona_choice, ""))

# ==========================================================================
# RETENTION
# ==========================================================================
with tab_retention:
    st.subheader("📌 Auto-Generated Strategic Insights")
    for insight in _generate_insights(df):
        st.markdown(f"- {insight}")

    st.subheader("🎯 Retention Strategy Recommendations")
    for rec in RETENTION_RECOMMENDATIONS:
        st.markdown(f"- {rec}")

# ==========================================================================
# EXECUTIVE REPORT
# ==========================================================================
with tab_exec:
    st.subheader("Executive Summary")
    kpis = {
        "Overall Churn Rate": f"{df['Exited'].mean()*100:.1f}%",
        "Churned Customers": f"{len(churned_df):,}",
        "Revenue at Risk": f"${revenue_at_risk:,.0f}",
        "High-Value Customers Lost": f"{len(high_value_churned):,}",
    }
    ui.render_kpi_row(list(zip(kpis.keys(), kpis.values(), [""] * len(kpis))))

    st.write("")
    st.markdown("##### Key Insights")
    for insight in _generate_insights(df):
        st.markdown(f"- {insight}")

    st.markdown("##### Recommended Actions")
    for rec in RETENTION_RECOMMENDATIONS:
        st.markdown(f"- {rec}")

    st.info("💡 For a downloadable PDF/Excel version of this report, visit the **Reports Center** page.")
    st.page_link("pages/09_Reports_Center.py", label="Open Reports Center →", use_container_width=False)
