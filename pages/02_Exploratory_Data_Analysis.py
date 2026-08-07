"""Exploratory Data Analysis page: correlations, distributions, statistics."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import visualization as viz
from utils import ui
from utils.data_loader import dataset_is_ready, get_enriched_dataset, get_raw_dataset

st.set_page_config(page_title="Exploratory Data Analysis", page_icon="🔍", layout="wide")
ui.inject_global_css()
ui.page_header("Exploratory Data Analysis", "Correlations, distributions, and statistical deep-dives.", "🔍")

if not dataset_is_ready():
    st.error("Dataset not found. Please add `Churn_Modelling.csv` to the `dataset/` folder.")
    st.stop()

raw_df = get_raw_dataset()
enriched_df = get_enriched_dataset()
exited_labeled = raw_df.assign(Exited=raw_df["Exited"].map({0: "Retained", 1: "Churned"}))

tab_uni, tab_bi, tab_corr, tab_adv = st.tabs(["📊 Univariate", "🔁 Bivariate", "🔗 Correlation", "🧪 Advanced"])

# ==========================================================================
# UNIVARIATE
# ==========================================================================
with tab_uni:
    with st.expander("📈 Statistical Summary (`df.describe()`)", expanded=False):
        st.dataframe(raw_df.describe().T, use_container_width=True)

    with st.expander("🧩 Missing Value Check", expanded=False):
        missing = raw_df.isnull().sum()
        st.dataframe(missing[missing >= 0].rename("Missing Count"), use_container_width=True)
        st.success("No missing values found in the dataset." if missing.sum() == 0 else f"{missing.sum()} missing values found.")

    st.subheader("Numeric Distribution")
    num_col = st.selectbox("Select a numeric column", ["Age", "Balance", "CreditScore", "EstimatedSalary", "Tenure"])
    st.plotly_chart(viz.histogram(raw_df, num_col, f"Distribution of {num_col}"), use_container_width=True)

    st.subheader("Categorical Distribution")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(viz.count_bar(raw_df, "Geography", "Geography Distribution"), use_container_width=True)
    with c2:
        st.plotly_chart(viz.count_bar(raw_df, "Gender", "Gender Distribution"), use_container_width=True)

    st.subheader("Engineered Feature Distributions")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(viz.histogram(enriched_df, "balance_to_salary", "Balance-to-Salary Ratio"), use_container_width=True)
    with c2:
        st.plotly_chart(viz.histogram(enriched_df, "tenure_by_age", "Tenure-by-Age Ratio"), use_container_width=True)

# ==========================================================================
# BIVARIATE
# ==========================================================================
with tab_bi:
    st.subheader("Feature vs. Churn")
    bivar_col = st.selectbox("Select a feature to compare against churn", ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"])
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(viz.box_plot(exited_labeled, "Exited", bivar_col, f"{bivar_col} by Churn Status (Box Plot)"), use_container_width=True)
    with c2:
        st.plotly_chart(viz.violin_plot(exited_labeled, "Exited", bivar_col, f"{bivar_col} by Churn Status (Violin Plot)"), use_container_width=True)

    st.subheader("Scatter Analysis")
    c1, c2 = st.columns(2)
    with c1:
        x_axis = st.selectbox("X Axis", ["Age", "CreditScore", "Balance"], index=0, key="scatter_x")
    with c2:
        y_axis = st.selectbox("Y Axis", ["Balance", "EstimatedSalary", "CreditScore"], index=0, key="scatter_y")
    st.plotly_chart(
        viz.scatter_plot(exited_labeled, x_axis, y_axis, f"{x_axis} vs {y_axis} (colored by churn status)", color="Exited"),
        use_container_width=True,
    )

# ==========================================================================
# CORRELATION
# ==========================================================================
with tab_corr:
    st.subheader("Correlation Heatmap")
    numeric_cols = enriched_df.select_dtypes(include="number").drop(columns=["Gender_Label"], errors="ignore")
    st.plotly_chart(viz.correlation_heatmap(numeric_cols), use_container_width=True)
    st.caption("Includes the engineered `balance_to_salary` and `tenure_by_age` features alongside raw fields.")

# ==========================================================================
# ADVANCED
# ==========================================================================
with tab_adv:
    st.subheader("Pairplot (Scatter Matrix)")
    st.caption("Every pairwise relationship between selected numeric features at once, colored by churn status.")
    default_dims = ["Age", "Balance", "CreditScore", "EstimatedSalary"]
    dims = st.multiselect(
        "Numeric features to include",
        ["Age", "Balance", "CreditScore", "EstimatedSalary", "Tenure", "NumOfProducts"],
        default=default_dims,
        max_selections=5,
    )
    if len(dims) >= 2:
        st.plotly_chart(
            viz.pairplot_matrix(exited_labeled, dims, color="Exited", title="Pairplot — colored by Churn Status"),
            use_container_width=True,
        )
    else:
        st.info("Select at least 2 features to build the pairplot.")

    st.divider()
    st.subheader("Outlier Check (IQR method)")
    outlier_col = st.selectbox("Column", ["CreditScore", "Age", "Balance", "EstimatedSalary", "Tenure"], key="adv_outlier_col")
    q1, q3 = raw_df[outlier_col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = raw_df[(raw_df[outlier_col] < lo) | (raw_df[outlier_col] > hi)]
    c1, c2, c3 = st.columns(3)
    c1.metric("Outlier Count", f"{len(outliers):,}")
    c2.metric("% of Dataset", f"{len(outliers)/len(raw_df)*100:.2f}%")
    c3.metric("Valid Range (IQR)", f"[{lo:,.1f}, {hi:,.1f}]")
    st.plotly_chart(viz.box_plot(raw_df.assign(_all="All Customers"), "_all", outlier_col, f"{outlier_col} — Outlier Boundaries"), use_container_width=True)
