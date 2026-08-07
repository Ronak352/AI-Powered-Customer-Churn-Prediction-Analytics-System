"""
prediction.py
=============
The core prediction engine used by both the Streamlit dashboard and the
optional FastAPI service. Wraps the full inference pipeline:

    raw customer record
        -> feature engineering
        -> encoding (gender label-encode, geography one-hot)
        -> column alignment to training feature order
        -> scaling
        -> model.predict_proba / model.predict
        -> risk categorization + confidence + business recommendation

Supports every model saved by ``training.py`` (classical sklearn/XGBoost
models via joblib, and the Keras ANN via ``keras.models.load_model``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src import config
from src.feature_engineering import engineer_features
from src.preprocessing import load_preprocessing_artifacts

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Structured result returned for a single customer prediction."""

    will_churn: bool
    probability: float  # probability of churn (0-1)
    risk_level: str  # "Low" | "Medium" | "High"
    confidence: float  # how far the probability is from the decision boundary (0-1)
    recommendations: List[str] = field(default_factory=list)
    model_used: str = ""

    def as_dict(self) -> Dict:
        return {
            "Prediction": "Churn" if self.will_churn else "No Churn",
            "Probability (%)": round(self.probability * 100, 2),
            "Risk Level": self.risk_level,
            "Confidence (%)": round(self.confidence * 100, 2),
            "Model Used": self.model_used,
            "Recommendation": " | ".join(self.recommendations),
        }


def available_models() -> List[str]:
    """Return the list of model names that have a saved artifact on disk."""
    names = []
    for name, path in config.MODEL_PATHS.items():
        if path.exists():
            names.append(name)
    return names


def _load_model(model_name: str):
    """Load a single model artifact by name, dispatching on file type."""
    path = config.MODEL_PATHS.get(model_name)
    if path is None or not path.exists():
        raise FileNotFoundError(
            f"No saved artifact found for model '{model_name}'. "
            f"Run `python -m src.training` first."
        )
    if path.suffix == ".keras":
        from tensorflow import keras

        return keras.models.load_model(path)
    return joblib.load(path)


def risk_level_for_probability(probability: float) -> str:
    """Map a churn probability to a business-friendly risk bucket."""
    for level, (low, high) in config.RISK_THRESHOLDS.items():
        if low <= probability < high:
            return level
    return "High"


def confidence_from_probability(probability: float) -> float:
    """Distance from the 0.5 decision boundary, scaled to 0-1.

    A probability of 0.5 (maximum uncertainty) yields confidence 0.0; a
    probability of 0.0 or 1.0 (maximum certainty) yields confidence 1.0.
    """
    return float(abs(probability - 0.5) * 2)


def prepare_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature-engineering + encoding + column-alignment
    pipeline on a raw (unscaled) dataframe of customer records, using the
    encoders/feature order saved during training.

    Args:
        raw_df: DataFrame containing at least ``config.RAW_INPUT_COLUMNS``.

    Returns:
        A scaled numpy-compatible DataFrame ready to feed into any model,
        with columns aligned to the exact order used at training time.
    """
    missing = set(config.RAW_INPUT_COLUMNS) - set(raw_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    gender_encoder, scaler, feature_order = load_preprocessing_artifacts()

    df = raw_df.copy()
    df["Gender"] = gender_encoder.transform(df["Gender"])
    df = engineer_features(df)

    dummies = pd.get_dummies(df["Geography"], prefix="Geography")
    df = df.drop(columns=["Geography"])
    df = pd.concat([df, dummies], axis=1)

    # Align to the exact training-time feature order, filling any unseen
    # geography dummy columns with 0.
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_order]

    scaled = scaler.transform(df)
    return pd.DataFrame(scaled, columns=feature_order, index=df.index)


def _predict_proba_any_model(model, X_scaled: pd.DataFrame) -> np.ndarray:
    """Return churn probabilities for either an sklearn/XGBoost model or a
    Keras ANN, presenting a single unified interface to callers.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_scaled.values)[:, 1]
    # Keras model: outputs a single sigmoid probability per row.
    preds = model.predict(X_scaled.values, verbose=0)
    return preds.ravel()


def predict_single(customer: Dict, model_name: Optional[str] = None) -> PredictionResult:
    """Predict churn for a single customer supplied as a raw dict.

    Args:
        customer: dict with keys matching ``config.RAW_INPUT_COLUMNS``.
        model_name: Which saved model to use. Defaults to the best available
            model (Random Forest > others by default ordering) if not given.

    Returns:
        A populated PredictionResult.
    """
    raw_df = pd.DataFrame([customer])
    X_scaled = prepare_features(raw_df)

    model_name = model_name or _default_model_name()
    model = _load_model(model_name)

    probability = float(_predict_proba_any_model(model, X_scaled)[0])
    will_churn = probability >= 0.5
    risk_level = risk_level_for_probability(probability)
    confidence = confidence_from_probability(probability)
    recommendations = config.BUSINESS_RECOMMENDATIONS.get(risk_level, [])

    return PredictionResult(
        will_churn=will_churn,
        probability=probability,
        risk_level=risk_level,
        confidence=confidence,
        recommendations=recommendations,
        model_used=model_name,
    )


def predict_batch(df: pd.DataFrame, model_name: Optional[str] = None) -> pd.DataFrame:
    """Run predictions for every row of a batch dataframe (CSV upload flow).

    Returns the original dataframe with additional columns:
    Prediction, Probability (%), Risk Level, Confidence (%), Recommendation.
    """
    X_scaled = prepare_features(df)

    model_name = model_name or _default_model_name()
    model = _load_model(model_name)

    probabilities = _predict_proba_any_model(model, X_scaled)

    result_df = df.copy()
    result_df["Prediction"] = np.where(probabilities >= 0.5, "Churn", "No Churn")
    result_df["Probability (%)"] = np.round(probabilities * 100, 2)
    result_df["Risk Level"] = [risk_level_for_probability(p) for p in probabilities]
    result_df["Confidence (%)"] = np.round(
        [confidence_from_probability(p) * 100 for p in probabilities], 2
    )
    result_df["Recommendation"] = [
        " | ".join(config.BUSINESS_RECOMMENDATIONS.get(rl, [])) for rl in result_df["Risk Level"]
    ]
    result_df["Model Used"] = model_name
    return result_df


def _default_model_name() -> str:
    """Pick a sensible default model: the best-performing available model
    according to the saved comparison table, falling back to the first
    available model artifact.
    """
    from src.evaluation import best_model_name, load_model_comparison

    comparison_df = load_model_comparison()
    if not comparison_df.empty:
        best = best_model_name(comparison_df)
        model_key_map = {
            "Logistic Regression": "Logistic Regression",
            "KNN Classifier": "KNN Classifier",
            "Naive Bayes": "Naive Bayes",
            "Random Forest": "Random Forest",
            "XGBoost": "XGBoost",
            "ANN": "ANN",
        }
        candidate = model_key_map.get(best)
        if candidate and config.MODEL_PATHS[candidate].exists():
            return candidate

    fallback = available_models()
    if not fallback:
        raise FileNotFoundError("No trained models found. Run `python -m src.training` first.")
    return fallback[0]
