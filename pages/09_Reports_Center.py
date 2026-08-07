"""Reports Center: download business & model-level reports as CSV, Excel, or PDF."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.report_generator import build_business_report_pdf, build_excel_report
from utils import ui
from utils.data_loader import (
    dataset_is_ready,
    get_model_comparison,
    get_raw_dataset,
    get_segmented_dataset,
    models_are_ready,
)

st.set_page_config(page_title="Reports Center", page_icon="📑", layout="wide")
ui.inject_global_css()
ui.page_header("Reports Center", "Download business and model-level reports as CSV, Excel, or PDF.", "📑")

if not dataset_is_ready():
    st.error("Dataset not found. Please add `Churn_Modelling.csv` to the `dataset/` folder.")
    st.stop()

df = get_raw_dataset()
comparison_df = get_model_comparison()

tab_business, tab_model, tab_custom = st.tabs(["💼 Business Report", "📈 Model Report", "🗂️ Custom Export"])

# ==========================================================================
# BUSINESS REPORT
# ==========================================================================
with tab_business:
    st.subheader("Business Insights Report")
    st.caption("Headline KPIs, auto-generated insights, top segments, and retention recommendations — ready for leadership.")

    churned_df = df[df["Exited"] == 1]
    revenue_at_risk = churned_df["Balance"].sum()
    high_balance_threshold = df["Balance"].quantile(0.75)
    high_value_churned = churned_df[churned_df["Balance"] >= high_balance_threshold]

    kpis = {
        "Overall Churn Rate": f"{df['Exited'].mean()*100:.1f}%",
        "Churned Customers": f"{len(churned_df):,}",
        "Revenue at Risk": f"${revenue_at_risk:,.0f}",
        "High-Value Customers Lost": f"{len(high_value_churned):,}",
    }
    ui.render_kpi_row(list(zip(kpis.keys(), kpis.values(), [""] * len(kpis))))

    age_bins_df = df.copy()
    age_bins_df["Age Group"] = pd.cut(age_bins_df["Age"], bins=[17, 30, 40, 50, 60, 100], labels=["18-30", "31-40", "41-50", "51-60", "60+"])
    age_rates = age_bins_df.groupby("Age Group", observed=True)["Exited"].mean()
    geo_rates = df.groupby("Geography")["Exited"].mean()

    insights = [
        f"Customers aged **{age_rates.idxmax()}** have the highest churn rate at **{age_rates.max()*100:.1f}%**.",
        f"**{geo_rates.idxmax()}** has the highest churn rate among all geographies at **{geo_rates.max()*100:.1f}%**.",
        f"Inactive members churn at **{df[df['IsActiveMember']==0]['Exited'].mean()*100:.1f}%** vs "
        f"**{df[df['IsActiveMember']==1]['Exited'].mean()*100:.1f}%** for active members.",
    ]
    recommendations = [
        "Targeted outreach for high-balance, inactive customers.",
        "Product bundling campaigns for single-product customers.",
        f"Geography-specific retention offers in {geo_rates.idxmax()}.",
        "Proactive engagement programs to reactivate inactive members.",
    ]

    segment_table = None
    try:
        _, _, persona_df = get_segmented_dataset()
        segment_table = persona_df[["Persona", "Customers", "Churn_Rate", "Revenue at Risk"]].round(2)
        st.markdown("##### Top Customer Segments")
        ui.interactive_table(segment_table, key="reports_business_segments")
    except Exception:  # noqa: BLE001
        pass

    st.markdown("##### Insights & Recommendations")
    for i in insights:
        st.markdown(f"- {i}")
    for r in recommendations:
        st.markdown(f"- {r}")

    pdf_bytes = build_business_report_pdf(kpis, insights, recommendations, segment_table)
    st.download_button(
        "⬇️ Download Business Report (PDF)",
        data=pdf_bytes,
        file_name="business_insights_report.pdf",
        mime="application/pdf",
        type="primary",
    )

# ==========================================================================
# MODEL REPORT
# ==========================================================================
with tab_model:
    st.subheader("Model Performance Report")
    if not models_are_ready() or comparison_df.empty:
        st.info("Run `python -m src.training` to generate model metrics before exporting a model report.")
    else:
        ui.interactive_table(comparison_df, key="reports_model_comparison")

        c1, c2 = st.columns(2)
        with c1:
            csv_bytes = comparison_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download model_comparison.csv", data=csv_bytes, file_name="model_comparison.csv", mime="text/csv"
            )
        with c2:
            excel_bytes = build_excel_report({"Model Comparison": comparison_df})
            st.download_button(
                "⬇️ Download model_comparison.xlsx",
                data=excel_bytes,
                file_name="model_comparison.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

# ==========================================================================
# CUSTOM EXPORT
# ==========================================================================
with tab_custom:
    st.subheader("Build a Custom Multi-Sheet Excel Export")
    st.caption("Pick which datasets to bundle into a single downloadable workbook.")

    options = {"Raw Dataset": df, "Model Comparison": comparison_df if not comparison_df.empty else None}
    try:
        _, profile_df, persona_df = get_segmented_dataset()
        options["Cluster Profile"] = profile_df
        options["Persona Summary"] = persona_df
    except Exception:  # noqa: BLE001
        pass

    available_sheets = {k: v for k, v in options.items() if v is not None}
    chosen = st.multiselect("Sheets to include", list(available_sheets.keys()), default=list(available_sheets.keys()))

    if chosen:
        sheets = {name: available_sheets[name] for name in chosen}
        excel_bytes = build_excel_report(sheets)
        st.download_button(
            "⬇️ Download custom_report.xlsx",
            data=excel_bytes,
            file_name="custom_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )
    else:
        st.info("Select at least one sheet to enable the download.")
