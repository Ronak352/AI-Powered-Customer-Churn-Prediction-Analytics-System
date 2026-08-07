"""
evaluation.py
=============
Reusable evaluation utilities shared by the training pipeline and the
"Model Performance" dashboard page. Keeps every metric computation in one
place so training-time and dashboard-time numbers can never drift apart.
"""

from __future__ import annotations

import json
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from src import config


def get_confusion_matrix(y_true, y_pred) -> np.ndarray:
    """Return the raw confusion matrix array."""
    return confusion_matrix(y_true, y_pred)


def get_classification_report(y_true, y_pred) -> Dict:
    """Return sklearn's classification report as a nested dict."""
    return classification_report(y_true, y_pred, output_dict=True, zero_division=0)


def get_roc_curve_data(y_true, y_proba):
    """Return (false_positive_rate, true_positive_rate, thresholds) for plotting."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    return fpr, tpr, thresholds


def get_precision_recall_curve_data(y_true, y_proba):
    """Return (precision, recall, thresholds) for plotting."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    return precision, recall, thresholds


def load_model_comparison() -> pd.DataFrame:
    """Load the saved model comparison table produced by training.py.

    Returns an empty dataframe with a helpful message column if training has
    not been run yet, instead of raising, so the dashboard can render a
    friendly empty state.
    """
    if not config.METRICS_PATH.exists():
        return pd.DataFrame()
    with open(config.METRICS_PATH) as f:
        records = json.load(f)
    return pd.DataFrame(records)


def load_training_history() -> Optional[Dict]:
    """Load the saved ANN training history (loss/accuracy per epoch), if any."""
    if not config.TRAINING_HISTORY_PATH.exists():
        return None
    with open(config.TRAINING_HISTORY_PATH) as f:
        return json.load(f)


def best_model_name(comparison_df: pd.DataFrame, metric: str = "F1 Score") -> Optional[str]:
    """Return the name of the best-performing trained model by a given metric."""
    if comparison_df.empty or metric not in comparison_df.columns:
        return None
    return comparison_df.sort_values(metric, ascending=False).iloc[0]["Model"]


def radar_metrics_for_model(comparison_df: pd.DataFrame, model_name: str):
    """Return (categories, values 0-100) of Accuracy/Precision/Recall/F1/ROC AUC
    for one model, scaled to 0-100 for the Model Center radar chart.
    """
    row = comparison_df[comparison_df["Model"] == model_name]
    if row.empty:
        return [], []
    row = row.iloc[0]
    categories = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
    values = [round(float(row.get(c, 0) or 0) * 100, 2) for c in categories]
    return categories, values
