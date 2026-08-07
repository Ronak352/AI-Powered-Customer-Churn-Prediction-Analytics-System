"""Customer Analytics page: demographic & behavioral breakdowns."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import config, visualization as viz
from utils import ui
from utils.data_loader import dataset_is_ready, get_raw_dataset

st.set_page_config(page_title="Customer Analytics", page_icon="📊", layout="wide")
ui.inject_global_css()
ui.page_header("Customer Analytics", "Demographic and behavioral breakdown of the customer base.", "📊")

if not dataset_is_ready():
    st.error("Dataset not found. Please add `Churn_Modelling.csv` to the `dataset/` folder.")
    st.stop()

df = get_raw_dataset()

# --------------------------------------------------------------------------
# KPIs
# --------------------------------------------------------------------------
ui.render_kpi_row(
    [
        ("Total Customers", f"{len(df):,}", ""),
        ("Exited Customers", f"{int(df['Exited'].sum()):,}", f"{df['Exited'].mean()*100:.1f}%"),
        ("Active Members", f"{int(df['IsActiveMember'].sum()):,}", f"{df['IsActiveMember'].mean()*100:.1f}%"),
        ("Avg. Estimated Salary", f"${df['EstimatedSalary'].mean():,.0f}", ""),
    ]
)
st.write("")

with st.sidebar:
    st.subheader("Filters")
    geo_filter = st.multiselect("Geography", sorted(df["Geography"].unique()), default=list(df["Geography"].unique()))
    gender_filter = st.multiselect("Gender", sorted(df["Gender"].unique()), default=list(df["Gender"].unique()))
    age_range = st.slider("Age Range", int(df["Age"].min()), int(df["Age"].max()), (int(df["Age"].min()), int(df["Age"].max())))

filtered = df[
    df["Geography"].isin(geo_filter)
    & df["Gender"].isin(gender_filter)
    & df["Age"].between(*age_range)
]
st.caption(f"Showing {len(filtered):,} of {len(df):,} customers based on current filters.")

tab1, tab2, tab3 = st.tabs(["Demographics", "Financials", "Engagement"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(viz.histogram(filtered, "Age", "Age Distribution"), use_container_width=True)
        st.plotly_chart(viz.donut_chart(filtered, "Gender", "Gender Split"), use_container_width=True)
    with c2:
        st.plotly_chart(viz.count_bar(filtered, "Geography", "Customers by Geography"), use_container_width=True)
        st.plotly_chart(viz.pie_chart(filtered, "Exited", "Exited vs Retained"), use_container_width=True)

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(viz.histogram(filtered, "Balance", "Balance Distribution"), use_container_width=True)
        st.plotly_chart(viz.histogram(filtered, "CreditScore", "Credit Score Distribution"), use_container_width=True)
    with c2:
        st.plotly_chart(viz.histogram(filtered, "EstimatedSalary", "Estimated Salary Distribution"), use_container_width=True)
        st.plotly_chart(viz.count_bar(filtered, "NumOfProducts", "Products Distribution"), use_container_width=True)

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(viz.donut_chart(filtered, "IsActiveMember", "Active vs Inactive Members"), use_container_width=True)
    with c2:
        st.plotly_chart(viz.donut_chart(filtered, "HasCrCard", "Has Credit Card"), use_container_width=True)

    st.plotly_chart(
        viz.box_plot(filtered.assign(Exited=filtered["Exited"].map({0: "Retained", 1: "Churned"})), "Exited", "Tenure", "Tenure by Churn Status"),
        use_container_width=True,
    )
