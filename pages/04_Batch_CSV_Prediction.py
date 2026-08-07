"""Batch CSV Prediction page: upload, validate, predict, download.

Enhanced with a second tab that lets a user upload their own labeled CSV
(any dataset containing the same raw churn feature columns plus an 'Exited'
target column), train all six models on it end-to-end in the browser, see
a full leaderboard, and optionally promote the best model to production.
"""

from __future__ import annotations

import sys
import time
import tracemalloc
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import config, visualization as viz
from src.feature_engineering import engineer_features
from src.prediction import predict_batch
from src.preprocessing import (
    clean_dataset,
    encode_gender,
    encode_geography,
    fit_scaler,
)
from src.training import evaluate_predictions
from utils import ui
from utils.data_loader import get_available_models, models_are_ready

st.set_page_config(page_title="Batch CSV Prediction", page_icon="📁", layout="wide")
ui.inject_global_css()
ui.page_header(
    "Batch CSV Prediction",
    "Score customers in bulk, or train every model fresh on your own labeled data.",
    "📁",
)

tab_predict, tab_train = st.tabs(["🚀 Batch Prediction", "🧠 Train on My Data"])

# ==========================================================================
# TAB 1 — existing batch scoring workflow 
# ==========================================================================
with tab_predict:
    if not models_are_ready():
        ui.models_missing_warning()
    else:
        with st.sidebar:
            st.subheader("Model Selection")

            available_models = get_available_models()

            model_choice = st.selectbox(
                                   "Prediction model",
                                    available_models
                            )
    

        st.markdown(
            f"Upload a CSV containing at least these columns: "
            f"`{'`, `'.join(config.RAW_INPUT_COLUMNS)}`"
        )

        sample_df = pd.DataFrame([{c: "" for c in config.RAW_INPUT_COLUMNS}])
        st.download_button(
            "⬇️ Download CSV Template",
            data=sample_df.to_csv(index=False),
            file_name="churn_prediction_template.csv",
            mime="text/csv",
        )

        uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"], key="predict_upload")

        if uploaded_file is not None:
            try:
                input_df = pd.read_csv(uploaded_file)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not read the uploaded file: {exc}")
                st.stop()

            missing_cols = set(config.RAW_INPUT_COLUMNS) - set(input_df.columns)
            if missing_cols:
                st.error(f"Uploaded CSV is missing required columns: {sorted(missing_cols)}")
                st.stop()

            st.success(f"✅ File validated — {len(input_df):,} customer records found.")
            st.dataframe(input_df.head(), use_container_width=True)

            if st.button("🚀 Run Batch Prediction", type="primary"):
                with st.spinner(f"Scoring {len(input_df):,} customers with {model_choice}..."):
                    try:
                        results_df = predict_batch(input_df, model_name=model_choice)
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Prediction failed: {exc}")
                        st.stop()

                st.session_state["batch_results"] = results_df

        if "batch_results" in st.session_state:
            results_df = st.session_state["batch_results"]
            st.divider()
            st.subheader("Prediction Summary")

            total = len(results_df)
            high = int((results_df["Risk Level"] == "High").sum())
            medium = int((results_df["Risk Level"] == "Medium").sum())
            low = int((results_df["Risk Level"] == "Low").sum())

            ui.render_kpi_row(
                [
                    ("Total Customers", f"{total:,}", ""),
                    ("High Risk", f"{high:,}", f"{high/total*100:.1f}%"),
                    ("Medium Risk", f"{medium:,}", f"{medium/total*100:.1f}%"),
                    ("Low Risk", f"{low:,}", f"{low/total*100:.1f}%"),
                ]
            )

            st.write("")
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(viz.donut_chart(results_df, "Risk Level", "Risk Level Distribution"), use_container_width=True)
            with c2:
                st.plotly_chart(viz.count_bar(results_df, "Prediction", "Churn vs No Churn Predictions"), use_container_width=True)

            st.subheader("Full Results")
            risk_filter = st.multiselect("Filter by risk level", ["Low", "Medium", "High"], default=["Low", "Medium", "High"], key="risk_filter_predict")
            display_df = results_df[results_df["Risk Level"].isin(risk_filter)]
            ui.interactive_table(display_df, key="batch_prediction_results_table")

            st.download_button(
                "⬇️ Download prediction.csv",
                data=results_df.to_csv(index=False),
                file_name="prediction.csv",
                mime="text/csv",
                type="primary",
            )
        else:
            ui.empty_state("Upload a CSV file above to begin batch scoring.")

# ==========================================================================
# TAB 2 — train every model on an uploaded, labeled CSV
# ==========================================================================
with tab_train:
    st.markdown(
        "Upload a **labeled** dataset (same raw columns as the training data, "
        f"plus a target column named **`{config.TARGET_COLUMN}`** with 0/1 values) "
        "to retrain **all six models** end-to-end and automatically select the best "
        "performer — useful for retraining on a fresh export, a different bank/region, "
        "or a rebalanced sample."
    )

    required_cols = config.RAW_INPUT_COLUMNS + [config.TARGET_COLUMN]
    st.caption(f"Required columns: `{'`, `'.join(required_cols)}`")

    train_file = st.file_uploader("Upload labeled training CSV", type=["csv"], key="train_upload")

    include_ann = st.checkbox("Include ANN (TensorFlow) — slower but usually most accurate", value=True)

    if train_file is not None:
        try:
            raw_df = pd.read_csv(train_file)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read the uploaded file: {exc}")
            st.stop()

        missing = set(required_cols) - set(raw_df.columns)
        if missing:
            st.error(f"Uploaded CSV is missing required columns: {sorted(missing)}")
            st.stop()

        st.success(f"✅ File validated — {len(raw_df):,} labeled records found.")
        st.dataframe(raw_df.head(), use_container_width=True)

        if st.button("🧠 Train All Models on This Data", type="primary"):
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split
            from sklearn.naive_bayes import GaussianNB
            from sklearn.neighbors import KNeighborsClassifier

            progress = st.progress(0.0, text="Preparing data...")

            df = clean_dataset(raw_df.copy())
            df, gender_encoder = encode_gender(df)
            df = engineer_features(df)
            df = encode_geography(df, training=True)

            X = df.drop(columns=[config.TARGET_COLUMN])
            y = df[config.TARGET_COLUMN]
            feature_order = list(X.columns)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE,
                stratify=y if y.nunique() > 1 else None,
            )

            scaler = fit_scaler(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            registry = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
                "KNN Classifier": KNeighborsClassifier(),
                "Naive Bayes": GaussianNB(),
                "Random Forest": RandomForestClassifier(n_estimators=300, random_state=config.RANDOM_STATE),
            }
            try:
                from xgboost import XGBClassifier
                registry["XGBoost"] = XGBClassifier(eval_metric="logloss", random_state=config.RANDOM_STATE)
            except ImportError:
                st.warning("xgboost not installed — skipping XGBoost.")

            trained_models: dict = {}
            results_rows = []
            n_steps = len(registry) + (1 if include_ann else 0)
            step = 0

            for name, model in registry.items():
                step += 1
                progress.progress(step / n_steps, text=f"Training {name}...")
                tracemalloc.start()
                t0 = time.perf_counter()
                model.fit(X_train_scaled, y_train)
                train_time = time.perf_counter() - t0
                t0 = time.perf_counter()
                y_pred = model.predict(X_test_scaled)
                pred_time = time.perf_counter() - t0
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None

                metrics = evaluate_predictions(y_test.values, y_pred, y_proba)
                metrics.update({
                    "Model": name,
                    "Training Time (s)": round(train_time, 4),
                    "Prediction Time (s)": round(pred_time, 4),
                    "Memory Usage (KB)": round(peak / 1024.0, 2),
                })
                results_rows.append(metrics)
                trained_models[name] = model

            ann_model = None
            if include_ann:
                step += 1
                progress.progress(step / n_steps, text="Training ANN (TensorFlow)...")
                try:
                    import tensorflow as tf
                    from tensorflow import keras

                    tf.random.set_seed(config.RANDOM_STATE)
                    ann_model = keras.Sequential([
                        keras.layers.Input(shape=(X_train_scaled.shape[1],)),
                        keras.layers.Dense(32, activation="relu"),
                        keras.layers.BatchNormalization(),
                        keras.layers.Dropout(0.3),
                        keras.layers.Dense(16, activation="relu"),
                        keras.layers.BatchNormalization(),
                        keras.layers.Dropout(0.2),
                        keras.layers.Dense(6, activation="relu"),
                        keras.layers.Dense(1, activation="sigmoid"),
                    ])
                    ann_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
                    callbacks = [
                        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
                        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
                    ]
                    t0 = time.perf_counter()
                    ann_model.fit(
                        X_train_scaled, y_train, epochs=80, batch_size=32,
                        validation_split=0.2, callbacks=callbacks, verbose=0,
                    )
                    train_time = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    y_proba = ann_model.predict(X_test_scaled, verbose=0).ravel()
                    pred_time = time.perf_counter() - t0
                    y_pred = (y_proba > 0.5).astype(int)
                    metrics = evaluate_predictions(y_test.values, y_pred, y_proba)
                    metrics.update({
                        "Model": "ANN", "Training Time (s)": round(train_time, 4),
                        "Prediction Time (s)": round(pred_time, 4), "Memory Usage (KB)": None,
                    })
                    results_rows.append(metrics)
                except ImportError:
                    st.warning("tensorflow not installed — skipping ANN.")

            progress.progress(1.0, text="Done!")
            leaderboard = pd.DataFrame(results_rows).sort_values("F1 Score", ascending=False).reset_index(drop=True)
            st.session_state["custom_leaderboard"] = leaderboard
            st.session_state["custom_trained_models"] = trained_models
            st.session_state["custom_ann_model"] = ann_model
            st.session_state["custom_scaler"] = scaler
            st.session_state["custom_gender_encoder"] = gender_encoder
            st.session_state["custom_feature_order"] = feature_order

           # Save SHAP background sample
            st.session_state["custom_shap_background"] = X_train.head(200)
        
    if "custom_leaderboard" in st.session_state:
        st.divider()
        leaderboard = st.session_state["custom_leaderboard"]
        best_row = leaderboard.iloc[0]

        st.subheader("🏆 Leaderboard — Best Model First")
        ui.render_kpi_row([
            ("Best Model", best_row["Model"], "by F1 Score"),
            ("Accuracy", f"{best_row['Accuracy']*100:.1f}%", ""),
            ("ROC-AUC", f"{best_row.get('ROC AUC', 0):.3f}" if pd.notna(best_row.get("ROC AUC")) else "N/A", ""),
            ("F1 Score", f"{best_row['F1 Score']:.3f}", ""),
        ])

        st.dataframe(
            leaderboard.style.highlight_max(subset=["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"], color="#3B82F6"),
            use_container_width=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(viz.model_comparison_bar(leaderboard, "F1 Score", "F1 Score by Model"), use_container_width=True)
        with c2:
            st.plotly_chart(viz.model_comparison_bar(leaderboard, "Accuracy", "Accuracy by Model"), use_container_width=True)

        st.download_button(
            "⬇️ Download leaderboard.csv",
            data=leaderboard.to_csv(index=False),
            file_name="custom_training_leaderboard.csv",
            mime="text/csv",
        )

        st.divider()
        st.markdown(
            f"**Recommended for deployment: `{best_row['Model']}`** — highest F1 Score, "
            "the best balance of precision and recall for churn detection on this data."
        )
        st.warning(
            "⚠️ Promoting a model overwrites the production artifacts used by every other "
            "page in this dashboard (scaler, encoders, and the selected model). Only do "
            "this if you intend to replace the deployed model.",
            icon="⚠️",
        )
        if st.button(f"🚀 Promote '{best_row['Model']}' to Production", type="secondary"):
            model_name = best_row["Model"]
            joblib.dump(st.session_state["custom_scaler"], config.SCALER_PATH)
            joblib.dump(st.session_state["custom_gender_encoder"], config.GENDER_ENCODER_PATH)
                # Save SHAP background data
            joblib.dump(st.session_state["custom_shap_background"], config.SHAP_BACKGROUND_PATH )

            import json
            with open(config.FEATURE_ORDER_PATH, "w") as f:
                json.dump(st.session_state["custom_feature_order"], f, indent=2)

            if model_name == "ANN" and st.session_state["custom_ann_model"] is not None:
                st.session_state["custom_ann_model"].save(config.MODEL_PATHS["ANN"])
            elif model_name in st.session_state["custom_trained_models"]:
                joblib.dump(st.session_state["custom_trained_models"][model_name], config.MODEL_PATHS[model_name])

            st.success(f"✅ '{model_name}' promoted to production. Reload the dashboard to see it reflected everywhere.")
            st.cache_data.clear()
    elif train_file is None:
        ui.empty_state("Upload a labeled CSV above to train all models on your own data.")
