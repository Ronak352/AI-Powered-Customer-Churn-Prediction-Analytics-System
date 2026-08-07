"""Customer Segmentation page: KMeans clustering + business persona labeling."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import visualization as viz
from src.segmentation import DEFAULT_N_CLUSTERS, PERSONA_DESCRIPTIONS, radar_values_for_persona
from utils import ui
from utils.data_loader import dataset_is_ready, get_segmented_dataset

st.set_page_config(page_title="Customer Segmentation", page_icon="🧬", layout="wide")
ui.inject_global_css()
ui.page_header("Customer Segmentation", "KMeans clustering translated into business personas — who to retain first.", "🧬")

if not dataset_is_ready():
    st.error("Dataset not found. Please add `Churn_Modelling.csv` to the `dataset/` folder.")
    st.stop()

with st.sidebar:
    st.subheader("Clustering")
    n_clusters = st.slider("Number of clusters (k)", 2, 8, DEFAULT_N_CLUSTERS)
    st.caption(
        "Clusters are built with KMeans on CreditScore, Age, Tenure, Balance, "
        "NumOfProducts, IsActiveMember, and EstimatedSalary (standardized), "
        "then automatically labeled as a business persona based on value "
        "(balance + salary) and risk (churn rate + inactivity) relative to "
        "the other clusters."
    )

try:
    segmented_df, profile_df, persona_df = get_segmented_dataset(n_clusters=n_clusters)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not run segmentation: {exc}")
    st.stop()

ui.render_kpi_row(
    [
        ("Total Customers", f"{len(segmented_df):,}", ""),
        ("Personas Identified", str(persona_df["Persona"].nunique()), f"from {n_clusters} clusters"),
        ("Highest Revenue at Risk", persona_df.iloc[0]["Persona"], f"${persona_df.iloc[0]['Revenue at Risk']:,.0f}"),
        ("Overall Churn Rate", f"{segmented_df['Exited'].mean()*100:.1f}%", ""),
    ]
)

st.write("")
tab_overview, tab_personas, tab_explore = st.tabs(["📊 Overview", "🧬 Persona Deep-Dive", "🔎 Explore Customers"])

# ==========================================================================
# OVERVIEW
# ==========================================================================
with tab_overview:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(viz.treemap_chart(segmented_df, ["Persona", "Geography"], None, "Customers by Persona → Geography"), use_container_width=True)
    with c2:
        st.plotly_chart(viz.treemap_chart(persona_df, ["Persona"], "Revenue at Risk", "Revenue at Risk by Persona"), use_container_width=True)

    st.subheader("Persona Summary")
    ui.interactive_table(persona_df, key="segmentation_persona_summary")

    st.subheader("Cluster Profile (Raw)")
    with st.expander("Show raw KMeans cluster centroids"):
        ui.interactive_table(profile_df, key="segmentation_cluster_profile")

# ==========================================================================
# PERSONA DEEP-DIVE
# ==========================================================================
with tab_personas:
    persona_choice = st.selectbox("Select a persona", profile_df["Persona"].unique(), key="segmentation_persona_choice")
    persona_rows = segmented_df[segmented_df["Persona"] == persona_choice]
    persona_summary_row = persona_df[persona_df["Persona"] == persona_choice].iloc[0]

    st.caption(PERSONA_DESCRIPTIONS.get(persona_choice, ""))

    ui.render_kpi_row(
        [
            ("Customers", f"{len(persona_rows):,}", f"{len(persona_rows)/len(segmented_df)*100:.1f}% of base"),
            ("Churn Rate", f"{persona_summary_row['Churn_Rate']*100:.1f}%", ""),
            ("Avg. Balance", f"${persona_summary_row['Avg_Balance']:,.0f}", ""),
            ("Revenue at Risk", f"${persona_summary_row['Revenue at Risk']:,.0f}", ""),
        ]
    )

    c1, c2 = st.columns(2)
    with c1:
        cats, vals = radar_values_for_persona(profile_df, persona_choice)
        if cats:
            st.plotly_chart(viz.radar_chart(cats, vals, f"{persona_choice} — Normalized Profile (0-100)"), use_container_width=True)
    with c2:
        st.plotly_chart(viz.count_bar(persona_rows, "Geography", f"{persona_choice} — Geography Split"), use_container_width=True)

    st.subheader(f"Recommended Action for {persona_choice}")
    action_map = {
        "High Value At Risk": "🚨 Immediate priority — personal outreach, dedicated relationship manager, retention offer.",
        "High Value Loyal": "🛡️ Protect & grow — loyalty perks, cross-sell higher-tier products, quarterly check-ins.",
        "Standard At Risk": "📧 Automated retention campaign — targeted email/app nudges, low-cost incentives.",
        "Standard Loyal": "📈 Upsell candidate — gradual product bundling, standard engagement cadence.",
    }
    st.info(action_map.get(persona_choice, "Review manually."))

# ==========================================================================
# EXPLORE
# ==========================================================================
with tab_explore:
    st.subheader("Browse Customers by Persona")
    persona_filter = st.multiselect("Personas", segmented_df["Persona"].unique(), default=list(segmented_df["Persona"].unique()))
    filtered = segmented_df[segmented_df["Persona"].isin(persona_filter)]
    st.caption(f"Showing {len(filtered):,} of {len(segmented_df):,} customers.")
    cols_to_show = ["CustomerId", "Surname", "Geography", "Age", "Balance", "EstimatedSalary", "NumOfProducts", "IsActiveMember", "Exited", "Cluster", "Persona"]
    ui.interactive_table(filtered[cols_to_show], key="segmentation_explore_table", height=500)

    st.download_button(
        "⬇️ Download segmented_customers.csv",
        data=filtered[cols_to_show].to_csv(index=False),
        file_name="segmented_customers.csv",
        mime="text/csv",
    )
