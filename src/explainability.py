"""
explainability.py
==================
Explainable AI (XAI) utilities built on SHAP (SHapley Additive exPlanations).

Provides:
    * Global explanations — SHAP summary plot data / feature importance
    * Local explanations — SHAP waterfall / force plot data for one customer
    * Business-friendly text explanations for a single prediction

SHAP is an optional dependency. Every public function here fails softly
(returns None / raises a clear, catchable ImportError with install
instructions) so the rest of the dashboard keeps working if the library is
not installed in the current environment. Install it with:

    pip install shap

then re-run the Explainable AI page.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

import json
from pathlib import Path

from src import config
from src.prediction import _load_model, prepare_features

logger = logging.getLogger(__name__)


class ShapNotAvailableError(ImportError):
    """Raised when SHAP is requested but not installed."""


def _require_shap():
    try:
        import shap  # noqa: F401

        return shap
    except ImportError as exc:  # pragma: no cover
        raise ShapNotAvailableError(
            "SHAP is not installed. Run `pip install shap` to enable "
            "Explainable AI features."
        ) from exc


def load_background_sample() -> pd.DataFrame:
    """
    Load SHAP background sample and align it with the
    currently deployed model feature order.
    """

    if not config.SHAP_BACKGROUND_PATH.exists():
        raise FileNotFoundError(
            "SHAP background sample not found — run training.py first."
        )

    background_df = joblib.load(config.SHAP_BACKGROUND_PATH)

    if Path(config.FEATURE_ORDER_PATH).exists():
        with open(config.FEATURE_ORDER_PATH, "r") as f:
            feature_order = json.load(f)

        background_df = background_df.reindex(
            columns=feature_order,
            fill_value=0
        )

    return background_df


def build_explainer(model_name: str):
    """Build the appropriate SHAP explainer for a given trained model.

    Tree-based models (Random Forest, XGBoost) use the fast TreeExplainer.
    Everything else (Logistic Regression, KNN, Naive Bayes, ANN) falls back
    to KernelExplainer on a small background sample for tractability.
    """
    shap = _require_shap()
    model = _load_model(model_name)
    background_raw = load_background_sample()

    try:
        if "Geography" in background_raw.columns:
          background_scaled = prepare_features(background_raw)
        else:
          background_scaled = background_raw.copy()
    except Exception:
         background_scaled = background_raw.copy()

    tree_models = {"Random Forest", "XGBoost"}
    if model_name in tree_models:
        explainer = shap.TreeExplainer(model)
    elif model_name == "ANN":
        predict_fn = lambda x: model.predict(x, verbose=0).ravel()  # noqa: E731
        explainer = shap.KernelExplainer(predict_fn, background_scaled.values[:50])
    else:
        predict_fn = lambda x: model.predict_proba(x)[:, 1]  # noqa: E731
        explainer = shap.KernelExplainer(predict_fn, background_scaled.values[:50])

    return explainer, background_scaled


def global_shap_values(model_name: str, sample_size: int = 200):
    """Compute SHAP values across a sample of the training background for a
    global summary plot / feature-importance ranking.

    Returns:
        (shap_values: np.ndarray, feature_names: list[str], X_sample: DataFrame)
    """
    explainer, background_scaled = build_explainer(model_name)
    X_sample = background_scaled.iloc[: min(sample_size, len(background_scaled))]

    shap_values = explainer.shap_values(X_sample.values)

    if isinstance(shap_values, list):
         shap_values = shap_values[-1]

    if len(np.array(shap_values).shape) == 3:
          shap_values = shap_values[:, :, 1]

    return shap_values, list(X_sample.columns), X_sample


def local_shap_values(model_name: str, customer: Dict):
    """Compute SHAP values for a single customer record (local explanation).

    Returns:
        (shap_values: np.ndarray of shape (n_features,), feature_names, X_row DataFrame, base_value)
    """
    explainer, _ = build_explainer(model_name)
    raw_df = pd.DataFrame([customer])

    X_row = prepare_features(raw_df)

    with open(config.FEATURE_ORDER_PATH, "r") as f:
              feature_order = json.load(f)

    X_row = X_row.reindex(
                      columns=feature_order,
                       fill_value=0
                    )
   

    shap_values = explainer.shap_values(X_row.values)

    if isinstance(shap_values, list):
       shap_values = shap_values[-1]

    if len(np.array(shap_values).shape) == 3:
     shap_values = shap_values[:, :, 1]

    base_value = explainer.expected_value

    if isinstance(base_value, (list, np.ndarray)):
      base_value = float(np.atleast_1d(base_value)[-1])
    
    return np.array(shap_values).ravel(), list(X_row.columns), X_row, float(base_value)


def top_features(shap_values: np.ndarray, feature_names: List[str], k: int = 5) -> Tuple[List[str], List[str]]:
    """Split feature impacts into top-k positive (churn-increasing) and
    top-k negative (churn-reducing) drivers, ranked by absolute SHAP value.

    Returns:
        (top_positive: list[str], top_negative: list[str]) each formatted as
        "FeatureName (+0.123)" style strings for direct display.
    """
    mean_values = np.mean(shap_values, axis=0) if shap_values.ndim == 2 else shap_values
    pairs = sorted(zip(feature_names, mean_values), key=lambda p: p[1], reverse=True)

    positive = [f"{name} ({val:+.3f})" for name, val in pairs if val > 0][:k]
    negative = [f"{name} ({val:+.3f})" for name, val in pairs if val < 0][:k]
    return positive, negative


def business_friendly_explanation(shap_values_row: np.ndarray, feature_names: List[str], k: int = 3) -> List[str]:
    """Translate a single customer's top SHAP drivers into plain-English
    sentences for the "Prediction Explanation" panel.
    """
    pairs = sorted(zip(feature_names, shap_values_row), key=lambda p: abs(p[1]), reverse=True)[:k]
    sentences = []
    for name, val in pairs:
        direction = "increases" if val > 0 else "decreases"
        sentences.append(f"**{name}** {direction} this customer's churn risk (impact: {val:+.3f}).")
    return sentences
