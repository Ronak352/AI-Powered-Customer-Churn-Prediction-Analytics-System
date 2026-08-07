"""
segmentation.py
================
Unsupervised customer segmentation used by the Executive Dashboard and the
Customer Segmentation page.

Pipeline:
    1. Select a small set of business-meaningful numeric features
       (balance, salary, credit score, age, tenure, products, activity,
       churn probability if available).
    2. Standard-scale them and run KMeans (default k=4).
    3. Translate each numeric cluster into a human-readable "persona" name
       (e.g. "High Value Loyal", "High Value At Risk") based on where the
       cluster centroid sits relative to the population median on value
       (balance/salary) and risk (churn rate/inactivity) axes.

The persona-naming step is what makes this useful for business users: raw
cluster IDs ("Cluster 2") mean nothing to a retention team, but "High Value
At Risk" tells them exactly who to act on.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

SEGMENTATION_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "IsActiveMember",
    "EstimatedSalary",
]

DEFAULT_N_CLUSTERS = 4

PERSONA_DESCRIPTIONS = {
    "High Value Loyal": (
        "Above-median balance/salary, highly active, low churn rate. "
        "Your best customers — protect the relationship and look for "
        "cross-sell opportunities."
    ),
    "High Value At Risk": (
        "Above-median balance/salary but showing disengagement (inactive "
        "and/or high churn rate). Highest revenue-at-risk segment — "
        "prioritize proactive retention outreach."
    ),
    "Standard Loyal": (
        "Below-median balance/salary, stable and active. Reliable, "
        "lower-risk base — good candidates for gradual product upsell."
    ),
    "Standard At Risk": (
        "Below-median balance/salary with elevated churn signals. Lower "
        "individual revenue impact, but high volume — good fit for "
        "low-cost, automated retention campaigns."
    ),
}


def _value_score(row: pd.Series) -> float:
    """Composite 'value' proxy: balance + salary, higher = more valuable."""
    return row["Balance"] + row["EstimatedSalary"]


def run_kmeans_segmentation(
    df: pd.DataFrame, n_clusters: int = DEFAULT_N_CLUSTERS, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster customers with KMeans and attach persona labels.

    Args:
        df: Dataframe containing at least ``SEGMENTATION_FEATURES`` and
            ``Exited``. Extra columns are preserved and passed through.
        n_clusters: Number of KMeans clusters.
        random_state: Reproducibility seed.

    Returns:
        (segmented_df, cluster_profile_df)

        ``segmented_df`` is the input dataframe with two new columns:
        ``Cluster`` (int cluster id) and ``Persona`` (human-readable label).

        ``cluster_profile_df`` is one row per cluster with the mean of each
        segmentation feature, the churn rate, customer count, and the
        assigned persona — used for the segmentation summary table / radar
        chart.
    """
    working = df.copy()
    missing = [c for c in SEGMENTATION_FEATURES if c not in working.columns]
    if missing:
        raise ValueError(f"Dataset is missing required segmentation columns: {missing}")

    X = working[SEGMENTATION_FEATURES].astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    working["Cluster"] = km.fit_predict(X_scaled)

    # Build a per-cluster profile so we can rank clusters on "value" and "risk"
    profile_rows = []
    for cluster_id, group in working.groupby("Cluster"):
        churn_rate = group["Exited"].mean() if "Exited" in group.columns else np.nan
        inactivity_rate = 1 - group["IsActiveMember"].mean()
        value = group["Balance"].mean() + group["EstimatedSalary"].mean()
        profile_rows.append(
            {
                "Cluster": cluster_id,
                "Customers": len(group),
                "Avg CreditScore": group["CreditScore"].mean(),
                "Avg Age": group["Age"].mean(),
                "Avg Tenure": group["Tenure"].mean(),
                "Avg Balance": group["Balance"].mean(),
                "Avg Salary": group["EstimatedSalary"].mean(),
                "Avg NumOfProducts": group["NumOfProducts"].mean(),
                "Active Rate": group["IsActiveMember"].mean(),
                "Churn Rate": churn_rate,
                "_value_score": value,
                "_risk_score": (churn_rate if not np.isnan(churn_rate) else 0) + inactivity_rate,
            }
        )
    profile_df = pd.DataFrame(profile_rows)

    value_median = profile_df["_value_score"].median()
    risk_median = profile_df["_risk_score"].median()

    def _assign_persona(row) -> str:
        is_high_value = row["_value_score"] >= value_median
        is_high_risk = row["_risk_score"] >= risk_median
        if is_high_value and is_high_risk:
            return "High Value At Risk"
        if is_high_value and not is_high_risk:
            return "High Value Loyal"
        if not is_high_value and is_high_risk:
            return "Standard At Risk"
        return "Standard Loyal"

    profile_df["Persona"] = profile_df.apply(_assign_persona, axis=1)

    cluster_to_persona: Dict[int, str] = dict(zip(profile_df["Cluster"], profile_df["Persona"]))
    working["Persona"] = working["Cluster"].map(cluster_to_persona)

    profile_df = profile_df.drop(columns=["_value_score", "_risk_score"]).sort_values("Cluster").reset_index(drop=True)

    return working, profile_df


def persona_summary(segmented_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the segmented dataframe to one row per persona (across all
    clusters that share that persona) — used for the KPI cards and treemap.
    """
    agg = (
        segmented_df.groupby("Persona")
        .agg(
            Customers=("Persona", "count"),
            Churn_Rate=("Exited", "mean"),
            Avg_Balance=("Balance", "mean"),
            Avg_Salary=("EstimatedSalary", "mean"),
            Active_Rate=("IsActiveMember", "mean"),
        )
        .reset_index()
    )
    agg["Revenue at Risk"] = agg["Customers"] * agg["Churn_Rate"] * agg["Avg_Balance"]
    return agg.sort_values("Revenue at Risk", ascending=False).reset_index(drop=True)


def radar_values_for_persona(profile_df: pd.DataFrame, persona: str) -> Tuple[List[str], List[float]]:
    """Return (categories, normalized 0-100 values) for a radar chart
    comparing one persona's average profile against the feature ranges
    present in ``profile_df``.
    """
    row = profile_df[profile_df["Persona"] == persona]
    if row.empty:
        return [], []
    row = row.iloc[0]
    categories = ["CreditScore", "Age", "Tenure", "Balance", "Salary", "Products", "Active Rate"]
    raw = [
        row["Avg CreditScore"], row["Avg Age"], row["Avg Tenure"], row["Avg Balance"],
        row["Avg Salary"], row["Avg NumOfProducts"], row["Active Rate"] * 100,
    ]
    # Normalize each metric to 0-100 using the min/max across all clusters so
    # the radar shape is comparable across personas.
    normalized = []
    for i, cat in enumerate(categories):
        col_map = {
            "CreditScore": "Avg CreditScore", "Age": "Avg Age", "Tenure": "Avg Tenure",
            "Balance": "Avg Balance", "Salary": "Avg Salary", "Products": "Avg NumOfProducts",
            "Active Rate": "Active Rate",
        }
        col = col_map[cat]
        series = profile_df[col] * (100 if col == "Active Rate" else 1)
        lo, hi = series.min(), series.max()
        val = raw[i]
        norm = 50.0 if hi == lo else (val - lo) / (hi - lo) * 100
        normalized.append(round(float(norm), 1))
    return categories, normalized
