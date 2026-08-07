"""
training.py
============
Trains, evaluates, and persists every model used by the dashboard:

    1. Logistic Regression
    2. K-Nearest Neighbors
    3. Naive Bayes
    4. Random Forest
    5. XGBoost
    6. Artificial Neural Network (TensorFlow/Keras)

Run directly to (re)train everything from scratch:

    python -m src.training

Design notes
------------
* XGBoost and TensorFlow are optional/heavy dependencies. If they are not
  installed in the current environment, training for those two models is
  skipped with a clear warning instead of crashing the whole script — this
  lets the rest of the pipeline (scaler, encoders, 4 classical models) still
  run in constrained environments. Install ``xgboost`` and ``tensorflow``
  (see requirements.txt) and re-run this script to produce the remaining
  two artifacts.
* Every model is timed for both training and single-batch prediction, and
  process memory delta is captured, to populate the "Model Comparison"
  dashboard page (Phase 9 of the spec: Accuracy, Precision, Recall, F1,
  ROC AUC, Training Time, Prediction Time, Memory Usage).
"""

from __future__ import annotations

import json
import logging
import os
import time
import tracemalloc
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from src import config
from src.feature_engineering import engineer_features
from src.preprocessing import (
    clean_dataset,
    encode_gender,
    encode_geography,
    fit_scaler,
    load_raw_dataset,
    save_preprocessing_artifacts,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _time_and_memory(callable_fn, *args, **kwargs):
    """Run ``callable_fn`` and return (result, elapsed_seconds, peak_memory_kb)."""
    tracemalloc.start()
    start = time.perf_counter()
    result = callable_fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak / 1024.0  # KB


def build_dataset():
    """Full preprocessing + feature engineering pipeline (training mode)."""
    df = load_raw_dataset()
    df = clean_dataset(df)
    df, gender_encoder = encode_gender(df)
    df = engineer_features(df)
    df = encode_geography(df, training=True)

    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN]

    feature_order = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )

    scaler = fit_scaler(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    save_preprocessing_artifacts(gender_encoder, scaler, feature_order)

    # Save an unscaled sample of X_train for SHAP background distribution.
    joblib.dump(X_train.sample(min(200, len(X_train)), random_state=config.RANDOM_STATE),
                config.SHAP_BACKGROUND_PATH)

    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, feature_order


def evaluate_predictions(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    """Compute the standard classification metric set used across the app."""
    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1 Score": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        try:
            metrics["ROC AUC"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["ROC AUC"] = None
    return metrics


def train_classical_models(X_train, X_test, y_train, y_test) -> Dict[str, Dict[str, Any]]:
    """Train Logistic Regression, KNN, Naive Bayes, Random Forest, XGBoost."""
    results: Dict[str, Dict[str, Any]] = {}

    model_registry = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
        "KNN Classifier": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=config.RANDOM_STATE),
    }

    # XGBoost is optional at runtime.
    try:
        from xgboost import XGBClassifier

        model_registry["XGBoost"] = XGBClassifier(
            eval_metric="logloss", random_state=config.RANDOM_STATE
        )
    except ImportError:
        logger.warning(
            "xgboost is not installed — skipping XGBoost training. "
            "Install with `pip install xgboost` and re-run to include it."
        )

    filename_map = {
        "Logistic Regression": config.MODEL_PATHS["Logistic Regression"],
        "KNN Classifier": config.MODEL_PATHS["KNN Classifier"],
        "Naive Bayes": config.MODEL_PATHS["Naive Bayes"],
        "Random Forest": config.MODEL_PATHS["Random Forest"],
        "XGBoost": config.MODEL_PATHS["XGBoost"],
    }

    for name, model in model_registry.items():
        logger.info("Training %s ...", name)
        _, train_time, train_mem = _time_and_memory(model.fit, X_train, y_train)

        (y_pred,), pred_time, pred_mem = _time_and_memory(lambda: (model.predict(X_test),))
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = evaluate_predictions(y_test, y_pred, y_proba)
        metrics["Training Time (s)"] = round(train_time, 4)
        metrics["Prediction Time (s)"] = round(pred_time, 4)
        metrics["Memory Usage (KB)"] = round(pred_mem, 2)

        joblib.dump(model, filename_map[name])
        results[name] = {"metrics": metrics}
        logger.info("%s -> %s", name, metrics)

    return results


def train_ann(X_train, X_test, y_train, y_test) -> Dict[str, Any]:
    """Train the improved ANN (Phase 10): Dropout, BatchNorm, EarlyStopping,
    ReduceLROnPlateau, and ModelCheckpoint. Returns metrics + history, or an
    empty dict with a warning if TensorFlow is not installed.
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        logger.warning(
            "tensorflow is not installed — skipping ANN training. "
            "Install with `pip install tensorflow` and re-run to include it."
        )
        return {}

    tf.random.set_seed(config.RANDOM_STATE)

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(X_train.shape[1],)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(6, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(
            str(config.MODEL_PATHS["ANN"]), monitor="val_loss", save_best_only=True
        ),
    ]

    start = time.perf_counter()
    history = model.fit(
        X_train,
        y_train,
        epochs=150,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=0,
    )
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    y_proba = model.predict(X_test, verbose=0).ravel()
    pred_time = time.perf_counter() - start
    y_pred = (y_proba > 0.5).astype(int)

    metrics = evaluate_predictions(y_test, y_pred, y_proba)
    metrics["Training Time (s)"] = round(train_time, 4)
    metrics["Prediction Time (s)"] = round(pred_time, 4)
    metrics["Memory Usage (KB)"] = None  # not meaningfully comparable to sklearn models

    model.save(config.MODEL_PATHS["ANN"])

    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(config.TRAINING_HISTORY_PATH, "w") as f:
        json.dump(history_dict, f, indent=2)

    return {"metrics": metrics}


def run_full_training_pipeline() -> pd.DataFrame:
    """Entry point: builds the dataset, trains every model, saves all
    artifacts + a model comparison table (models/model_metrics.json).
    """
    X_train, X_test, y_train, y_test, feature_order = build_dataset()

    all_results = train_classical_models(X_train, X_test, y_train, y_test)

    ann_result = train_ann(X_train, X_test, y_train, y_test)
    if ann_result:
        all_results["ANN"] = ann_result

    comparison_rows = []
    for name, payload in all_results.items():
        row = {"Model": name}
        row.update(payload["metrics"])
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)
    if not comparison_df.empty and "F1 Score" in comparison_df.columns:
        comparison_df = comparison_df.sort_values("F1 Score", ascending=False).reset_index(drop=True)

    with open(config.METRICS_PATH, "w") as f:
        json.dump(comparison_df.to_dict(orient="records"), f, indent=2)

    logger.info("Training complete. Metrics saved to %s", config.METRICS_PATH)
    logger.info("\n%s", comparison_df.to_string(index=False))
    return comparison_df


if __name__ == "__main__":
    run_full_training_pipeline()
