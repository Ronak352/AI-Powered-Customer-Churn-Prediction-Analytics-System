"""Model Performance page: metrics, confusion matrix, ROC/PR/Lift curves, ANN history."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import config, visualization as viz
from src.evaluation import (
    get_classification_report,
    get_confusion_matrix,
    get_precision_recall_curve_data,
    get_roc_curve_data,
    radar_metrics_for_model,
)
from src.preprocessing import clean_dataset, encode_gender, encode_geography, load_preprocessing_artifacts, load_raw_dataset
from src.feature_engineering import engineer_features
from src.prediction import _load_model, _predict_proba_any_model
from utils import ui
from utils.data_loader import get_available_models, get_model_comparison, get_training_history, models_are_ready

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")
ui.inject_global_css()
ui.page_header("Model Performance", "Compare every trained model across accuracy, precision, recall, F1, ROC AUC, and lift.", "📈")

if not models_are_ready():
    ui.models_missing_warning()
    st.stop()

comparison_df = get_model_comparison()

if comparison_df.empty:
    st.info("Run `python -m src.training` to generate the model comparison table.")
    st.stop()

best_model = comparison_df.sort_values("F1 Score", ascending=False).iloc[0]


@st.cache_data(show_spinner=False)
def _build_test_split():
    """Recreate the exact same train/test split used during training, purely
    for computing diagnostics on the held-out test set (no leakage — same
    random_state and split ratio as training.py).
    """
    from sklearn.model_selection import train_test_split

    df = load_raw_dataset()
    df = clean_dataset(df)
    df, _ = encode_gender(df)
    df = engineer_features(df)
    df = encode_geography(df, training=False) if config.GEOGRAPHY_CATEGORIES_PATH.exists() else pd.get_dummies(df, columns=["Geography"], drop_first=True)
    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN]
    _, X_test, _, y_test = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y)
    _, scaler, feature_order = load_preprocessing_artifacts()
    X_test = X_test[feature_order]
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_order, index=X_test.index)
    return X_test_scaled, y_test.values


tab_compare, tab_diag, tab_lift, tab_ann = st.tabs(
    ["📋 Comparison", "🔬 Diagnostics", "📈 Lift / Gain / Radar", "🧠 ANN Training"]
)

# ==========================================================================
# COMPARISON
# ==========================================================================
with tab_compare:
    st.subheader("Model Comparison Table")
    ui.interactive_table(comparison_df, key="model_comparison_table")
    st.success(f"🏆 Best overall model (by F1 Score): **{best_model['Model']}**")

    metric_choice = st.selectbox("Comparison metric", ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"])
    st.plotly_chart(viz.model_comparison_bar(comparison_df, metric_choice), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(viz.model_comparison_bar(comparison_df, "Training Time (s)", "Training Time by Model"), use_container_width=True)
    with c2:
        st.plotly_chart(viz.model_comparison_bar(comparison_df, "Prediction Time (s)", "Prediction Time by Model"), use_container_width=True)

# ==========================================================================
# DIAGNOSTICS
# ==========================================================================
with tab_diag:
    selected_model = st.selectbox("Select a model for detailed diagnostics", get_available_models(), key="diag_model")

    try:
        X_test_scaled, y_test = _build_test_split()
        model = _load_model(selected_model)
        y_proba = _predict_proba_any_model(model, X_test_scaled)
        y_pred = (y_proba >= 0.5).astype(int)

        c1, c2 = st.columns(2)
        with c1:
            cm = get_confusion_matrix(y_test, y_pred)
            st.plotly_chart(viz.confusion_matrix_figure(cm, title=f"{selected_model} — Confusion Matrix"), use_container_width=True)
        with c2:
            fpr, tpr, _ = get_roc_curve_data(y_test, y_proba)
            from sklearn.metrics import auc as sk_auc

            auc_value = sk_auc(fpr, tpr)
            st.plotly_chart(viz.roc_curve_figure(fpr, tpr, auc_value, title=f"{selected_model} — ROC Curve"), use_container_width=True)

        precision, recall, _ = get_precision_recall_curve_data(y_test, y_proba)
        st.plotly_chart(viz.precision_recall_figure(precision, recall, title=f"{selected_model} — Precision-Recall Curve"), use_container_width=True)

        with st.expander("Full Classification Report"):
            report = get_classification_report(y_test, y_pred)
            st.dataframe(pd.DataFrame(report).T, use_container_width=True)

    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not compute diagnostics for {selected_model}: {exc}")

# ==========================================================================
# LIFT / GAIN / RADAR
# ==========================================================================
with tab_lift:
    lift_model = st.selectbox("Select a model", get_available_models(), key="lift_model")

    try:
        X_test_scaled, y_test = _build_test_split()
        model = _load_model(lift_model)
        y_proba = _predict_proba_any_model(model, X_test_scaled)

        n_bins = st.slider("Number of deciles", 5, 20, 10, key="lift_bins")
        curve_df = viz.lift_gain_curves(y_test, y_proba, n_bins=n_bins)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(viz.gain_curve_figure(curve_df, title=f"{lift_model} — Cumulative Gain Curve"), use_container_width=True)
        with c2:
            st.plotly_chart(viz.lift_curve_figure(curve_df, title=f"{lift_model} — Lift Curve"), use_container_width=True)

        with st.expander("Lift / Gain Table"):
            st.dataframe(curve_df, use_container_width=True)

    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not compute lift/gain curves for {lift_model}: {exc}")

    st.divider()
    st.subheader("Model Center — Metric Radar")
    radar_model = st.selectbox("Model to profile", get_available_models(), key="radar_model")
    cats, vals = radar_metrics_for_model(comparison_df, radar_model)
    if cats:
        st.plotly_chart(viz.radar_chart(cats, vals, f"{radar_model} — Metric Radar (0-100)"), use_container_width=True)
    else:
        st.info("No metrics available for this model.")

# ==========================================================================
# ANN TRAINING
# ==========================================================================
with tab_ann:
    if "ANN" in get_available_models():
        history = get_training_history()
        if history:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(viz.training_history_figure(history, "accuracy"), use_container_width=True)
            with c2:
                st.plotly_chart(viz.training_history_figure(history, "loss"), use_container_width=True)
        else:
            st.info("ANN training history not found.")
    else:
        st.info("ANN model not trained — run `python -m src.training` with TensorFlow installed to see this tab populated.")
