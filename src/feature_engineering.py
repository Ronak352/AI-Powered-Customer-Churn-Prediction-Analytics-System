"""
feature_engineering.py
=======================
Engineered features used by every model in the project. These reproduce the
two engineered features created in the original notebook, exposed as a
reusable, well-documented function so the exact same logic runs during
training, single-customer prediction, and batch CSV prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_balance_to_salary_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'balance_to_salary' = Balance / EstimatedSalary.

    A high ratio can indicate a customer holding a large balance relative to
    income — a segment shown in EDA to correlate with churn risk.
    """
    df = df.copy()
    df["balance_to_salary"] = df["Balance"] / df["EstimatedSalary"].replace(0, np.nan)
    df["balance_to_salary"] = df["balance_to_salary"].fillna(0)
    return df


def add_tenure_by_age_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'tenure_by_age' = Tenure / Age.

    Captures how long a customer has stayed relative to their age — a proxy
    for "relationship depth".
    """
    df = df.copy()
    df["tenure_by_age"] = df["Tenure"] / df["Age"].replace(0, np.nan)
    df["tenure_by_age"] = df["tenure_by_age"].fillna(0)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full engineered-feature pipeline in the same order as training."""
    df = add_balance_to_salary_ratio(df)
    df = add_tenure_by_age_ratio(df)
    return df
